from enum import Enum

from pydantic import BaseModel, Field, model_validator


class DocumentType(str, Enum):
    PASSPORT = "passport"
    ID_CARD = "id_card"
    DRIVING_LICENSE = "driving_license"


class DocumentInput(BaseModel):
    """Logical upload names; the HTTP transport supplies the corresponding files."""

    document_type: DocumentType
    image: str | None = None
    front: str | None = None
    back: str | None = None

    @model_validator(mode="after")
    def require_document_parts(self):
        if self.document_type == DocumentType.ID_CARD:
            if not self.front or not self.back or self.image:
                raise ValueError("id_card requires front and back, and does not accept image")
        elif not self.image or self.front or self.back:
            raise ValueError(f"{self.document_type.value} requires image only")
        return self


class BoundingBox(BaseModel):
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered_corners(self):
        if self.x1 >= self.x2:
            raise ValueError("x1 must be less than x2")
        if self.y1 >= self.y2:
            raise ValueError("y1 must be less than y2")
        return self


class ConfidenceSource(str, Enum):
    DOCUMENT_DETECTION = "document_detection"
    OCR_TOKEN = "ocr_token"
    OCR_TOKEN_MEAN = "ocr_token_mean"
    OCR_TOKEN_MINIMUM = "ocr_token_minimum"


class Confidence(BaseModel):
    score: float = Field(ge=0, le=1)
    source: ConfidenceSource
    calibrated_probability: bool = False


class FieldResult(BaseModel):
    value: str | None
    raw_text: list[str] = Field(default_factory=list)
    confidence: Confidence | None = None
    bounding_box: BoundingBox | None = None
    region: str


class ValidationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_RUN = "not_run"


class ValidationResult(BaseModel):
    code: str
    status: ValidationStatus
    detail: str | None = None
    fields: list[str] = Field(default_factory=list)


class MrzResult(BaseModel):
    raw_lines: list[str] = Field(default_factory=list)
    fields: dict[str, str | None] = Field(default_factory=dict)
    validations: list[ValidationResult] = Field(default_factory=list)


class TimingResult(BaseModel):
    total_seconds: float = Field(ge=0)
    stages: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def non_negative_stages(self):
        if any(value < 0 for value in self.stages.values()):
            raise ValueError("timing stages cannot be negative")
        return self


class DocumentResult(BaseModel):
    document_type: DocumentType
    layout: str
    fields: dict[str, FieldResult]
    document_confidence: Confidence | None = None
    mrz: MrzResult | None = None
    validations: list[ValidationResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timings: TimingResult


class ErrorCode(str, Enum):
    INVALID_UPLOAD = "invalid_upload"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    UPLOAD_TOO_LARGE = "upload_too_large"
    TOO_MANY_ITEMS = "too_many_items"
    INVALID_ARCHIVE = "invalid_archive"
    MISSING_DOCUMENT_SIDE = "missing_document_side"
    INVALID_DOCUMENT = "invalid_document"
    PROFILE_UNAVAILABLE = "profile_unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    QUEUE_FULL = "queue_full"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    PROCESSING_FAILED = "processing_failed"


class ErrorResult(BaseModel):
    code: ErrorCode
    detail: str


class BatchItemResult(BaseModel):
    index: int = Field(ge=0)
    input: DocumentInput
    success: bool
    result: DocumentResult | None = None
    error: ErrorResult | None = None

    @model_validator(mode="after")
    def exactly_one_outcome(self):
        if self.success and (self.result is None or self.error is not None):
            raise ValueError("successful items require result and no error")
        if not self.success and (self.error is None or self.result is not None):
            raise ValueError("failed items require error and no result")
        return self
