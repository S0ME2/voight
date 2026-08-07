from pathlib import Path

import numpy as np

from app.artifacts import create_artifact_run
from app.config import (
    ArtifactSettings,
    DrivingLicenseSettings,
    MrzSettings,
    OcrSettings,
    Settings,
)
from app.documents.driving_license_fields import parse_fields
from app.documents.mrz import reconstruct, select
from app.file_type import detect_file_extension
from app.models import Models
from app.roi import assign_tokens_to_rois, crop_normalized_roi


def test_roi_assignment_and_crop():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert crop_normalized_roi(
        image, {"x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.6}
    ).shape == (40, 80, 3)
    assigned, unassigned = assign_tokens_to_rois(
        [{"text": "A", "x1": 20, "y1": 20, "x2": 40, "y2": 40}],
        {"field": {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.5}},
        200,
        100,
        0.3,
    )
    assert [token["text"] for token in assigned["field"]] == ["A"]
    assert not unassigned


def test_document_parser_returns_raw_driving_license_text():
    extracted, _ = parse_fields(
        {
            "birth_place_and_date": [
                {"text": "3. TOSHLOQ 19.10.2005", "x1": 0, "center_y": 0}
            ],
            "personal_id": [{"text": "4d. 12345678901234", "x1": 0, "center_y": 0}],
            "license_number": [{"text": "5. AG2742395", "x1": 0, "center_y": 0}],
        }
    )
    assert extracted["birth_place"] == "3. TOSHLOQ 19.10.2005"
    assert extracted["birth_date"] is None
    assert extracted["personal_id"] == "4d. 12345678901234"
    assert extracted["license_number"] == "5. AG2742395"
    lines = reconstruct(
        [
            {
                "text": "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
                "score": 0.9,
                "x1": 0,
                "center_y": 10,
                "height": 10,
            }
        ]
    )
    assert [line.text for line in select(lines, (1,))] == [lines[0].text]


def test_runtime_configuration_with_stubs(
    tmp_path: Path = Path("/tmp/voightkampff-test"),
):
    settings = Settings(
        True,
        ArtifactSettings(False, tmp_path),
        OcrSettings("cpu"),
        MrzSettings("test", 100, 1.0, 0.0),
        DrivingLicenseSettings(
            tmp_path / "crop.json", tmp_path / "rois.json", 10, 10, 1, 0.3, "test"
        ),
    )
    models = Models(settings)
    loaded = []
    models.ocr = lambda: loaded.append("ocr")
    models.mrz_scanner = lambda: loaded.append("mrz")
    models.document_aligner = lambda: loaded.append("aligner")
    models.preload()
    assert loaded == ["ocr", "mrz", "aligner"]
    writer = create_artifact_run(settings.artifacts, "test", "input.png")
    assert not writer.directory.exists()
    assert detect_file_extension(b"\x89PNG\r\n\x1a\nanything") == "png"
