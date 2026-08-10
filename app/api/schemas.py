from typing import Any

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


class BatchItemError(BaseModel):
    """Legacy migration schema; new v1 routes use ``ErrorResult``."""

    status_code: int
    code: str
    detail: str


class BatchItemResponse(BaseModel):
    """Legacy migration schema kept until the final cutover task."""

    index: int
    filename: str | None = None
    source_filename: str | None = None
    archive_path: str | None = None
    success: bool
    result: str | dict[str, Any] | None = None
    error: BatchItemError | None = None
    total_seconds: float | None = None


class BatchResponse(BaseModel):
    """Legacy migration schema kept until the final cutover task."""

    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    total_seconds: float = Field(ge=0)
    batch_run_id: str | None = None
    items: list[BatchItemResponse]
