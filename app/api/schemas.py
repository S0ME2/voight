from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.contracts import BatchItemResult, DocumentResult, ErrorResult


class OcrResponse(BaseModel):
    result: DocumentResult


class OcrBatchResponse(BaseModel):
    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    total_seconds: float = Field(ge=0)
    items: list[BatchItemResult]
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def counts_match_items(self):
        if self.total != len(self.items) or self.total != self.succeeded + self.failed:
            raise ValueError("batch counts must match items")
        if self.succeeded != sum(item.success for item in self.items):
            raise ValueError("batch success and failure counts must match item outcomes")
        return self


class OcrErrorResponse(BaseModel):
    error: ErrorResult


class VerificationOcrLine(BaseModel):
    line_id: str | None = None
    text: str
    confidence: float = Field(ge=0, le=1)
    confidence_source: Literal["ocr_token"] = "ocr_token"
    bbox: tuple[float, float, float, float] | None = None
    polygon: list[tuple[float, float]] | None = None
    reading_order: int | None = None
    side: Literal["front", "back", "image"] | None = None


class VerificationOcrResponse(BaseModel):
    lines: list[VerificationOcrLine]


class VerificationIdCardOcrResponse(BaseModel):
    front: list[VerificationOcrLine]
    back: list[VerificationOcrLine]


VerificationOcrResult = VerificationOcrResponse | VerificationIdCardOcrResponse


class VerificationCheckRequest(BaseModel):
    ocr: VerificationOcrResponse
    fields: dict[str, str | int | float | bool]

    @model_validator(mode="after")
    def validate_fields(self):
        if not self.fields or any(not name.strip() for name in self.fields):
            raise ValueError("fields must contain at least one named value")
        return self


class VerificationIdCardCheckRequest(BaseModel):
    ocr: VerificationIdCardOcrResponse
    fields: dict[str, str | int | float | bool]

    @model_validator(mode="after")
    def validate_fields(self):
        if not self.fields or any(not name.strip() for name in self.fields):
            raise ValueError("fields must contain at least one named value")
        return self


class VerificationFieldResult(BaseModel):
    expected: str
    detected: str | None
    score: float = Field(ge=0, le=1)
    score_source: Literal["normalized_sequence_similarity"] = "normalized_sequence_similarity"
    status: Literal["match", "likely_match", "mismatch", "not_found"]
    source: str | None = None
    evidence: list["VerificationEvidence"] = []


class VerificationEvidence(BaseModel):
    line_id: str
    span: tuple[int, int]
    source: Literal["visible_ocr", "mrz"]
    side: Literal["front", "back", "image"] | None = None
    bbox: tuple[float, float, float, float] | None = None


class VerificationSummary(BaseModel):
    match: int = Field(ge=0)
    likely_match: int = Field(ge=0)
    mismatch: int = Field(ge=0)
    not_found: int = Field(ge=0)


class VerificationResponse(BaseModel):
    fields: dict[str, VerificationFieldResult]
    summary: VerificationSummary


class VerificationOcrBatchItem(BaseModel):
    index: int = Field(ge=0)
    success: bool
    result: VerificationOcrResult | None = None
    error: ErrorResult | None = None

    @model_validator(mode="after")
    def exactly_one_outcome(self):
        if self.success != (self.result is not None and self.error is None):
            raise ValueError("batch items require exactly one successful result or error")
        return self


class VerificationOcrBatchResponse(BaseModel):
    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    items: list[VerificationOcrBatchItem]

    @model_validator(mode="after")
    def counts_match_items(self):
        if self.total != len(self.items) or self.total != self.succeeded + self.failed:
            raise ValueError("batch counts must match items")
        if self.succeeded != sum(item.success for item in self.items):
            raise ValueError("batch success and failure counts must match item outcomes")
        return self
