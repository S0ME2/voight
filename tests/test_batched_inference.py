import unittest
from dataclasses import replace
from types import ModuleType
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.artifacts import ArtifactWriter
from app.config import (
    ArtifactSettings,
    BatchSettings,
    DrivingLicenseSettings,
    MrzSettings,
    OcrSettings,
    RuntimeSettings,
    Settings,
    TextModelSettings,
)
from app.inference.batch import (
    BatchedOcr,
    OcrSample,
    ProfileBatchItem,
    ProfileBatchRunner,
    ResourceExhaustedError,
)
from app.inference.contracts import (
    DetectedTextRegion,
    DetectedTextRegions,
    LocalizationResult,
    MrzRecognitionResult,
    RecognitionResult,
)
from app.documents.mrz import PASSPORT
from app.pipeline import RegionProfile
from app.models import Models


WRITER = ArtifactWriter(Path("/unused"), "", "", False)


class DetectionStub:
    def __init__(self, line_counts=None, failing_marker=None):
        self.line_counts = line_counts or {}
        self.failing_marker = failing_marker
        self.batch_sizes = []

    def detect_batch(self, images):
        self.batch_sizes.append(len(images))
        markers = [int(image[0, 0, 0]) for image in images]
        if self.failing_marker in markers:
            raise ValueError(f"bad marker {self.failing_marker}")
        results = []
        for marker, image in zip(markers, images):
            height, width = image.shape[:2]
            count = self.line_counts.get(marker, 1)
            polygons = []
            for index in range(count):
                top = 2 + index * 8
                polygons.append(
                    [[2, top], [width - 3, top], [width - 3, top + 5], [2, top + 5]]
                )
            results.append(DetectedTextRegions(tuple(
                DetectedTextRegion(np.asarray(polygon, dtype=np.float32))
                for polygon in polygons
            )))
        return results


class RecognitionStub:
    def __init__(self, failing_marker=None):
        self.failing_marker = failing_marker
        self.batch_sizes = []

    def recognize_batch(self, images):
        self.batch_sizes.append(len(images))
        markers = [int(round(float(image.mean()))) for image in images]
        if self.failing_marker in markers:
            raise ValueError(f"bad line marker {self.failing_marker}")
        return [RecognitionResult(f"value-{marker}", marker / 100) for marker in markers]


class ProcessRecognitionStub:
    def __init__(self):
        self.chunks = []

    def recognize_batch(self, images):
        self.chunks.append(images)
        return [RecognitionResult(f"value-{int(round(float(image.mean())))}", 0.5) for image in images]


def image(marker):
    return np.full((40, 60, 3), marker, dtype=np.uint8)


def full_document(value):
    height, width = value.shape[:2]
    return [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]


def parse(assignments):
    raw = {
        field: " ".join(token["text"] for token in tokens)
        for field, tokens in assignments.items()
    }
    return raw, raw


class BatchedOcrTests(unittest.TestCase):
    def test_true_batches_restore_variable_line_counts_and_match_single_items(self):
        detector = DetectionStub({10: 1, 20: 3, 30: 2})
        recognizer = RecognitionStub()
        engine = BatchedOcr(
            detector,
            recognizer,
            detection_batch_size=2,
            recognition_batch_size=4,
        )
        samples = [
            OcrSample("passport", image(10)),
            OcrSample("id_card", image(20)),
            OcrSample("driving_license", image(30)),
        ]
        result = engine.run(samples)

        self.assertEqual([], list(result.errors))
        self.assertEqual(
            {"passport": 1, "id_card": 3, "driving_license": 2},
            {key: len(tokens) for key, tokens in result.tokens.items()},
        )
        self.assertEqual([2, 1], detector.batch_sizes)
        self.assertEqual([4, 2], recognizer.batch_sizes)
        self.assertIn(
            2,
            result.diagnostics["text_detection"]["tensor_batch_sizes"],
        )
        self.assertIn(
            4,
            result.diagnostics["text_recognition"]["tensor_batch_sizes"],
        )
        self.assertGreaterEqual(result.diagnostics["line_crop_seconds"], 0.0)
        self.assertGreaterEqual(result.diagnostics["result_unpack_seconds"], 0.0)

        for sample in samples:
            single = BatchedOcr(
                DetectionStub(detector.line_counts),
                RecognitionStub(),
                detection_batch_size=2,
                recognition_batch_size=4,
            ).run([sample])
            self.assertEqual(single.tokens[sample.item_id], result.tokens[sample.item_id])

    def test_detection_failure_isolated_by_recursive_real_batches(self):
        detector = DetectionStub(failing_marker=99)
        engine = BatchedOcr(
            detector,
            RecognitionStub(),
            detection_batch_size=3,
            recognition_batch_size=8,
        )
        result = engine.run(
            [
                OcrSample("first", image(10)),
                OcrSample("bad", image(99)),
                OcrSample("last", image(30)),
            ]
        )

        self.assertEqual(["first", "last"], list(result.tokens))
        self.assertEqual(["bad"], list(result.errors))
        self.assertEqual([3, 1, 2, 1, 1], detector.batch_sizes)
        self.assertEqual(2, result.diagnostics["text_detection"]["retry_split_count"])
        self.assertEqual(1, result.diagnostics["text_detection"]["isolated_failure_count"])

    def test_recognition_failure_isolated_by_recursive_real_batches(self):
        recognizer = RecognitionStub(failing_marker=99)
        engine = BatchedOcr(
            DetectionStub(),
            recognizer,
            detection_batch_size=3,
            recognition_batch_size=3,
        )
        result = engine.run(
            [
                OcrSample("first", image(10)),
                OcrSample("bad", image(99)),
                OcrSample("last", image(30)),
            ]
        )

        self.assertEqual(["first", "last"], list(result.tokens))
        self.assertEqual(["bad"], list(result.errors))
        self.assertEqual([3, 1, 2, 1, 1], recognizer.batch_sizes)
        self.assertEqual(2, result.diagnostics["text_recognition"]["retry_split_count"])
        self.assertEqual(1, result.diagnostics["text_recognition"]["isolated_failure_count"])

    def test_resource_failure_is_not_split(self):
        class ResourceDetector(DetectionStub):
            def detect_batch(self, images):
                self.batch_sizes.append(len(images))
                raise MemoryError("out of memory")

        detector = ResourceDetector()
        with self.assertRaises(ResourceExhaustedError):
            BatchedOcr(detector, RecognitionStub(), detection_batch_size=4, recognition_batch_size=4).run(
                [OcrSample("one", image(10)), OcrSample("two", image(20))]
            )
        self.assertEqual([2], detector.batch_sizes)

    def test_visible_roi_filters_lines_before_recognition_but_mrz_does_not(self):
        class Lines:
            def detect_batch(self, images):
                polygons = np.asarray(
                    [
                        [[2, 2], [20, 2], [20, 6], [2, 6]],
                        [[2, 20], [20, 20], [20, 24], [2, 24]],
                    ],
                    dtype=np.float32,
                )
                return [DetectedTextRegions(tuple(DetectedTextRegion(polygon) for polygon in polygons)) for _ in images]

        recognizer = RecognitionStub()
        result = BatchedOcr(Lines(), recognizer, detection_batch_size=2, recognition_batch_size=4).run(
            [
                OcrSample("visible", image(10), {"top": {"x1": 0, "y1": 0, "x2": 1, "y2": 0.25}}),
                OcrSample("mrz", image(20)),
            ]
        )
        self.assertEqual([3], recognizer.batch_sizes)
        self.assertEqual(4, result.diagnostics["line_filter"]["detected_line_count"])
        self.assertEqual(3, result.diagnostics["line_filter"]["recognition_candidate_count"])
        self.assertEqual(1, result.diagnostics["line_filter"]["filtered_before_recognition_count"])
        self.assertEqual(0, result.diagnostics["line_filter"]["samples"]["mrz"]["filtered_before_recognition_count"])
        self.assertEqual(["value-10"], [token["text"] for token in result.tokens["visible"]])
        self.assertEqual(2, len(result.tokens["mrz"]))

    def test_retained_lines_across_field_rois_keep_reading_order(self):
        class Lines:
            def detect_batch(self, images):
                polygons = np.asarray(
                    [
                        [[2, 20], [20, 20], [20, 24], [2, 24]],
                        [[2, 2], [20, 2], [20, 6], [2, 6]],
                    ],
                    dtype=np.float32,
                )
                return [DetectedTextRegions(tuple(DetectedTextRegion(polygon) for polygon in polygons)) for _ in images]

        class OrderedRecognition:
            def recognize_batch(self, images):
                return [RecognitionResult(f"line-{index}", 0.9) for index, _ in enumerate(images)]

        result = BatchedOcr(Lines(), OrderedRecognition(), detection_batch_size=1, recognition_batch_size=4).run(
            [OcrSample("visible", image(10), {
                "top": {"x1": 0, "y1": 0, "x2": 1, "y2": 0.25},
                "bottom": {"x1": 0, "y1": 0.4, "x2": 1, "y2": 0.8},
            })]
        )
        self.assertEqual(["line-0", "line-1"], [token["text"] for token in result.tokens["visible"]])
        self.assertEqual(2, result.diagnostics["line_filter"]["recognition_candidate_count"])

    def test_ten_inputs_use_four_four_two_prediction_batches(self):
        detector = DetectionStub()
        recognizer = RecognitionStub()
        result = BatchedOcr(
            detector,
            recognizer,
            detection_batch_size=4,
            recognition_batch_size=4,
        ).run([OcrSample(str(index), image(index + 10)) for index in range(10)])

        self.assertFalse(result.errors)
        self.assertEqual([4, 4, 2], detector.batch_sizes)
        self.assertEqual([4, 4, 2], recognizer.batch_sizes)
        self.assertEqual([4, 4, 2], result.diagnostics["text_detection"]["tensor_batch_sizes"])
        self.assertEqual([4, 4, 2], result.diagnostics["text_recognition"]["tensor_batch_sizes"])

    def test_recognition_microbatches_use_process_worker_interface(self):
        recognizer = ProcessRecognitionStub()
        result = BatchedOcr(
            DetectionStub(), recognizer, detection_batch_size=10, recognition_batch_size=4
        ).run([OcrSample(str(index), image(index + 10)) for index in range(10)])

        self.assertFalse(result.errors)
        self.assertEqual([4, 4, 2], [len(chunk) for chunk in recognizer.chunks])
        self.assertEqual([4, 4, 2], result.diagnostics["text_recognition"]["tensor_batch_sizes"])

    def test_bad_input_is_isolated_without_reordering_successes(self):
        result = BatchedOcr(
            DetectionStub(), RecognitionStub(), detection_batch_size=4, recognition_batch_size=4
        ).run(
            [
                OcrSample("first", image(10)),
                OcrSample("bad", np.array([], dtype=np.uint8)),
                OcrSample("last", image(30)),
            ]
        )
        self.assertEqual(["first", "last"], list(result.tokens))
        self.assertEqual(["bad"], list(result.errors))


class ProfileBatchRunnerTests(unittest.TestCase):
    class Localizer:
        def __init__(self, kind):
            self.kind = kind
            self.batch_sizes = []

        def localize_batch(self, images):
            self.batch_sizes.append(len(images))
            self.last_tensor_batch_size = len(images)
            values = []
            for value in images:
                height, width = value.shape[:2]
                if self.kind == "mrz":
                    values.append(LocalizationResult(np.asarray([[0, height - 8], [width - 1, height - 8], [width - 1, height - 2], [0, height - 2]], dtype=np.float32)))
                else:
                    values.append(LocalizationResult(np.asarray(full_document(value), dtype=np.float32)))
            return values

    def _item(self, item_id, marker):
        profile = RegionProfile(
            {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
            {"field": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
        )
        return ProfileBatchItem(
            item_id=item_id,
            image=image(marker),
            profile=profile,
            localization_kind="docaligner",
            parse_fields=parse,
            validate_fields=lambda _values: [],
            artifacts=WRITER,
            canonical_width=60,
            canonical_height=40,
        )

    def test_all_document_regions_share_batch_path_and_keep_order(self):
        localizer = self.Localizer("docaligner")
        runner = ProfileBatchRunner(
            BatchedOcr(
                DetectionStub(),
                RecognitionStub(),
                detection_batch_size=8,
                recognition_batch_size=8,
            ),
            {"docaligner": localizer, "mrz": self.Localizer("mrz")},
            MrzSettings("stub", 100, 1.0, 0.0),
            localization_batch_size=4,
            max_items=6,
        )
        items = [
            self._item("passport:data_page", 10),
            self._item("id_card:front", 20),
            self._item("id_card:back", 30),
            self._item("driving_license:data", 40),
        ]
        outcomes, diagnostics = runner.run(items)

        self.assertEqual(
            [item.item_id for item in items],
            [item.item_id for item in outcomes],
        )
        self.assertTrue(all(outcome.error is None for outcome in outcomes))
        self.assertEqual(
            ["value-10", "value-20", "value-30", "value-40"],
            [outcome.result[0]["field"] for outcome in outcomes],
        )
        self.assertEqual(
            [4],
            diagnostics["localization"]["docaligner"]["tensor_batch_sizes"],
        )
        self.assertEqual([4], localizer.batch_sizes)
        self.assertEqual(
            [4], diagnostics["text_detection"]["tensor_batch_sizes"]
        )
        self.assertEqual(
            [4], diagnostics["text_recognition"]["tensor_batch_sizes"]
        )

    def test_size_limit_is_enforced(self):
        runner = ProfileBatchRunner(
            BatchedOcr(
                DetectionStub(),
                RecognitionStub(),
                detection_batch_size=1,
                recognition_batch_size=1,
            ),
            {"docaligner": self.Localizer("docaligner"), "mrz": self.Localizer("mrz")},
            MrzSettings("stub", 100, 1.0, 0.0),
            localization_batch_size=1,
            max_items=1,
        )
        with self.assertRaisesRegex(ValueError, "maximum is 1"):
            runner.run([self._item("a", 10), self._item("b", 20)])

    def test_ten_localization_jobs_use_four_four_two_model_batches(self):
        localizer = self.Localizer("docaligner")
        runner = ProfileBatchRunner(
            BatchedOcr(DetectionStub(), RecognitionStub(), detection_batch_size=20, recognition_batch_size=20),
            {"docaligner": localizer, "mrz": self.Localizer("mrz")},
            MrzSettings("stub", 100, 1.0, 0.0),
            localization_batch_size=4,
            max_items=10,
        )
        outcomes, diagnostics = runner.run([self._item(str(index), index + 10) for index in range(10)])
        self.assertTrue(all(outcome.error is None for outcome in outcomes))
        self.assertEqual([4, 4, 2], localizer.batch_sizes)
        self.assertEqual(
            [4, 4, 2],
            diagnostics["localization"]["docaligner"]["tensor_batch_sizes"],
        )

    def test_specialized_mrz_failure_isolated_without_losing_siblings(self):
        class MrzRecognizer:
            def __init__(self):
                self.batch_sizes = []

            def recognize_batch(self, images):
                self.batch_sizes.append(len(images))
                markers = [int(round(float(value.mean()))) for value in images]
                if 99 in markers:
                    raise ValueError("bad MRZ")
                self.last_tensor_batch_size = len(images)
                return [MrzRecognitionResult((f"MRZ{marker}",), "recognized") for marker in markers]

        recognizer = MrzRecognizer()
        runner = ProfileBatchRunner(
            BatchedOcr(DetectionStub(), RecognitionStub(), detection_batch_size=4, recognition_batch_size=4),
            {"docaligner": self.Localizer("docaligner"), "mrz": self.Localizer("mrz")},
            MrzSettings("stub", 100, 1.0, 0.0),
            localization_batch_size=4,
            max_items=3,
            mrz_recognizer=recognizer,
            mrz_recognition_batch_size=4,
        )
        items = [
            replace(self._item(name, marker), probe_mrz=True, mrz_profile=PASSPORT)
            for name, marker in (("first", 10), ("bad", 99), ("last", 30))
        ]
        outcomes, diagnostics = runner.run(items)
        self.assertIsNone(outcomes[0].error)
        self.assertIsNotNone(outcomes[1].error)
        self.assertIsNone(outcomes[2].error)
        self.assertEqual([3, 1, 2, 1, 1], recognizer.batch_sizes)
        self.assertEqual(1, diagnostics["mrz_recognition"]["isolated_failure_count"])

    def test_id_card_mrz_scans_front_only_when_back_has_no_polygon(self):
        class MrzProbe:
            def __init__(self, missing=()):
                self.missing = set(missing)
                self.markers = []

            def localize_batch(self, images):
                markers = [int(image[0, 0, 0]) for image in images]
                self.markers.append(markers)
                self.last_tensor_batch_size = len(images)
                return [
                    LocalizationResult(
                        np.empty((0, 2), np.float32)
                        if marker in self.missing
                        else np.asarray([[0, 32], [59, 32], [59, 38], [0, 38]], np.float32)
                    )
                    for marker in markers
                ]

        def run(missing=()):
            probe = MrzProbe(missing)
            runner = ProfileBatchRunner(
                BatchedOcr(DetectionStub(), RecognitionStub(), detection_batch_size=4, recognition_batch_size=4),
                {"docaligner": self.Localizer("docaligner"), "mrz": probe},
                MrzSettings("stub", 100, 1.0, 0.0),
                localization_batch_size=4,
                max_items=2,
            )
            front = replace(self._item("card:front", 10), mrz_fallback_for="card:back")
            back = replace(self._item("card:back", 20), probe_mrz=True, mrz_profile=PASSPORT)
            outcomes, diagnostics = runner.run([front, back])
            return probe, outcomes, diagnostics

        probe, outcomes, diagnostics = run()
        self.assertEqual([[20]], probe.markers)
        self.assertTrue(all(outcome.error is None for outcome in outcomes))
        self.assertFalse(outcomes[0].mrz_detected)
        self.assertTrue(outcomes[1].mrz_detected)
        self.assertEqual(0, diagnostics["id_card_mrz_probe"]["front_fallback_scanned"])
        self.assertEqual("back", diagnostics["id_card_mrz_probe"]["samples"]["card:back"]["side_selected"])

        probe, outcomes, diagnostics = run({20})
        self.assertEqual([[20], [10]], probe.markers)
        self.assertTrue(all(outcome.error is None for outcome in outcomes))
        self.assertTrue(outcomes[0].mrz_detected)
        self.assertFalse(outcomes[1].mrz_detected)
        self.assertEqual(1, diagnostics["id_card_mrz_probe"]["front_fallback_scanned"])
        self.assertEqual("front", diagnostics["id_card_mrz_probe"]["samples"]["card:back"]["side_selected"])


class ModelOwnerTests(unittest.TestCase):
    def test_recognizer_backend_swaps_through_factory_without_pipeline_changes(self):
        class FakeRecognizer:
            def __init__(self, prefix):
                self.prefix = prefix

            def recognize_batch(self, images):
                return [RecognitionResult(f"{self.prefix}-{int(round(float(image.mean())))}", 1.0) for image in images]

        base = Settings.from_env()
        factories = {
            "fake-a": lambda _selection: FakeRecognizer("a"),
            "fake-b": lambda _selection: FakeRecognizer("b"),
        }
        values = []
        for backend in factories:
            settings = replace(
                base,
                models=replace(
                    base.models,
                    text_recognizer=TextModelSettings(backend, "fake"),
                ),
            )
            recognizer = Models(
                settings, text_recognizer_factories=factories
            ).text_recognizer()
            result = BatchedOcr(
                DetectionStub(), recognizer,
                detection_batch_size=1, recognition_batch_size=1,
            ).run([OcrSample("sample", image(10))])
            values.append(result.tokens["sample"][0]["text"])
        self.assertEqual(["a-10", "b-10"], values)

    def test_models_owns_one_cpu_batch_runner_with_configured_sizes(self):
        created = []

        class Wrapper:
            def __init__(self, **kwargs):
                created.append(kwargs)

        paddleocr = ModuleType("paddleocr")
        paddleocr.TextDetection = Wrapper
        paddleocr.TextRecognition = Wrapper
        root = Path("/unused")
        settings = Settings(
            False,
            ArtifactSettings(False, root),
            OcrSettings("cpu"),
            MrzSettings("stub", 100, 1.0, 0.0),
            DrivingLicenseSettings(root, root, 10, 10, 0, 0.3, "stub"),
            batch=BatchSettings(max_files=3),
            runtime=RuntimeSettings(
                target="cpu",
                queue_limit=2,
                localization_batch_size=3,
                text_detection_batch_size=4,
                text_recognition_batch_size=5,
            ),
        )
        models = Models(settings)
        models.document_localizer = lambda: ProfileBatchRunnerTests.Localizer("docaligner")
        models.mrz_localizer = lambda: ProfileBatchRunnerTests.Localizer("mrz")
        models.mrz_recognizer = lambda: None
        with patch.dict("sys.modules", {"paddleocr": paddleocr}):
            runner = models.profile_batch_runner()
            self.assertIs(runner, models.profile_batch_runner())

        self.assertEqual(["cpu", "cpu"], [item["device"] for item in created])
        self.assertEqual(6, runner.max_items)

    def test_models_passes_recognition_acceleration_only_when_requested(self):
        created = []

        class Wrapper:
            def __init__(self, **kwargs):
                created.append(kwargs)

        base = Settings.from_env()
        settings = Settings(
            base.preload,
            base.artifacts,
            OcrSettings("gpu"),
            base.mrz,
            base.driving_license,
            batch=base.batch,
            runtime=RuntimeSettings(
                target="gpu",
                text_recognition_enable_hpi=True,
                text_recognition_use_tensorrt=True,
                text_recognition_precision="fp16",
            ),
            models=base.models,
            profiles=base.profiles,
        )
        paddleocr = ModuleType("paddleocr")
        paddleocr.TextRecognition = Wrapper
        with patch.dict("sys.modules", {"paddleocr": paddleocr}):
            Models(settings).text_recognizer()
        self.assertEqual(True, created[0]["enable_hpi"])
        self.assertEqual(True, created[0]["use_tensorrt"])
        self.assertEqual("fp16", created[0]["precision"])


if __name__ == "__main__":
    unittest.main()
