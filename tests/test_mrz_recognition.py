import unittest

import numpy as np

from app.inference.mrzscanner import MrzScannerRecognizer


class MrzScannerRecognizerTests(unittest.TestCase):
    def test_adapter_builds_one_batch_and_restores_raw_lines(self):
        class Engine:
            providers = ["CPUExecutionProvider"]
            input_infos = {"input": {"shape": ["N", 3, 2, 2]}}

            def __init__(self):
                self.batch_sizes = []

            def __call__(self, **inputs):
                batch = inputs["input"]
                self.batch_sizes.append(batch.shape[0])
                return {"output": batch[:, :1, :1, :1]}

        engine = Engine()

        class Inference:
            model = engine
            input_key = "input"
            output_key = "output"
            delimeter = "<SEP>"

            def preprocess(self, image, normalize=True):
                return {"input": np.full((1, 3, 2, 2), image[0, 0, 0], np.float32)}

            def postprocess(self, output):
                marker = int(output["output"][0, 0, 0, 0])
                return f"LINE{marker}<SEP>SECOND{marker}"

        scanner = type("Scanner", (), {"recognizer": Inference()})()
        adapter = MrzScannerRecognizer(scanner)
        results = adapter.recognize_batch([
            np.full((4, 4, 3), marker, np.uint8) for marker in (1, 2, 3)
        ])
        self.assertEqual([3], engine.batch_sizes)
        self.assertEqual(3, adapter.last_tensor_batch_size)
        self.assertEqual(
            [("LINE1", "SECOND1"), ("LINE2", "SECOND2"), ("LINE3", "SECOND3")],
            [result.lines for result in results],
        )


if __name__ == "__main__":
    unittest.main()
