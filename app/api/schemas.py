from typing import Any

from pydantic import BaseModel, Field


class BatchItemError(BaseModel):
    """A serializable error for one file in a batch request."""

    status_code: int
    code: str
    detail: str


class BatchItemResponse(BaseModel):
    """Processing outcome for one uploaded file or ZIP entry."""

    index: int
    filename: str | None = None
    source_filename: str | None = None
    archive_path: str | None = None
    success: bool
    result: str | dict[str, Any] | None = None
    error: BatchItemError | None = None
    total_seconds: float | None = None


class BatchResponse(BaseModel):
    """Order-preserving response returned by every batch endpoint."""

    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    total_seconds: float = Field(ge=0)
    batch_run_id: str | None = None
    items: list[BatchItemResponse]
