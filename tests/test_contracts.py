import subprocess
import sys
import unittest

from pydantic import ValidationError

from app.api.schemas import OcrBatchResponse
from app.contracts import (
    BatchItemResult,
    BoundingBox,
    Confidence,
    ConfidenceSource,
    DocumentInput,
    DocumentResult,
    DocumentType,
    ErrorCode,
    ErrorResult,
    FieldResult,
    TimingResult,
)


class ContractTests(unittest.TestCase):
    def test_paired_id_input_and_single_image_inputs(self):
        paired = DocumentInput(
            document_type=DocumentType.ID_CARD,
            front="front.jpg",
            back="back.jpg",
        )
        self.assertEqual(paired.front, "front.jpg")
        DocumentInput(document_type=DocumentType.PASSPORT, image="passport.jpg")

        with self.assertRaisesRegex(ValidationError, "requires front and back"):
            DocumentInput(document_type=DocumentType.ID_CARD, front="front.jpg")
        with self.assertRaisesRegex(ValidationError, "requires image only"):
            DocumentInput(
                document_type=DocumentType.DRIVING_LICENSE,
                image="licence.jpg",
                back="extra.jpg",
            )

    def test_confidence_bounds_and_source_are_explicit(self):
        confidence = Confidence(score=0.8, source=ConfidenceSource.OCR_TOKEN_MEAN)
        self.assertFalse(confidence.calibrated_probability)
        with self.assertRaises(ValidationError):
            Confidence(score=1.1, source=ConfidenceSource.OCR_TOKEN)
        with self.assertRaises(ValidationError):
            BoundingBox(x1=0.8, y1=0.1, x2=0.2, y2=0.9)

    def test_batch_items_have_exactly_one_outcome_and_keep_order(self):
        source = DocumentInput(document_type=DocumentType.PASSPORT, image="a.jpg")
        result = DocumentResult(
            document_type=DocumentType.PASSPORT,
            layout="uzbekistan_passport",
            fields={
                "surname": FieldResult(
                    value="TEST",
                    raw_text=["TEST"],
                    region="data_page",
                )
            },
            timings=TimingResult(total_seconds=0.1),
        )
        failed_source = DocumentInput(
            document_type=DocumentType.PASSPORT, image="b.jpg"
        )
        response = OcrBatchResponse(
            total=2,
            succeeded=1,
            failed=1,
            total_seconds=0.2,
            items=[
                BatchItemResult(index=0, input=source, success=True, result=result),
                BatchItemResult(
                    index=1,
                    input=failed_source,
                    success=False,
                    error=ErrorResult(
                        code=ErrorCode.INVALID_DOCUMENT,
                        detail="document not found",
                    ),
                ),
            ],
        )
        self.assertEqual([item.index for item in response.items], [0, 1])
        with self.assertRaisesRegex(ValidationError, "require error"):
            BatchItemResult(index=0, input=source, success=False)
        with self.assertRaisesRegex(ValidationError, "counts must match"):
            OcrBatchResponse(
                total=1,
                succeeded=1,
                failed=0,
                total_seconds=0.2,
                items=response.items,
            )

    def test_lightweight_contract_imports_do_not_load_model_runtimes(self):
        code = (
            "import sys; import app.config, app.contracts, app.api.schemas, "
            "app.pipeline, app.inference; "
            "assert 'paddle' not in sys.modules; "
            "assert 'onnxruntime' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
