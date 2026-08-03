from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from app.file_type import detect_file_extension


@dataclass(frozen=True)
class UploadedDocument:
    data: bytes
    filename: str | None
    content_type: str | None
    extension: str | None
    source_filename: str | None = None
    archive_path: str | None = None


@dataclass(frozen=True)
class UploadedImage(UploadedDocument):
    image: np.ndarray | None = None


def document_from_bytes(
    data: bytes,
    filename: str | None,
    content_type: str | None = None,
    *,
    source_filename: str | None = None,
    archive_path: str | None = None,
) -> UploadedDocument:
    return UploadedDocument(
        data=data,
        filename=filename,
        content_type=content_type,
        extension=detect_file_extension(data),
        source_filename=source_filename,
        archive_path=archive_path,
    )


def image_from_document(upload: UploadedDocument) -> UploadedImage:
    image = cv2.imdecode(
        np.frombuffer(upload.data, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise HTTPException(status_code=422, detail="Upload must be a decodable image")
    return UploadedImage(
        data=upload.data,
        filename=upload.filename,
        content_type=upload.content_type,
        extension=upload.extension,
        source_filename=upload.source_filename,
        archive_path=upload.archive_path,
        image=image,
    )


async def read_upload(file: UploadFile) -> UploadedDocument:
    data = await file.read()
    return document_from_bytes(data, file.filename, file.content_type)


async def read_uploaded_image(file: UploadFile) -> UploadedImage:
    return image_from_document(await read_upload(file))
