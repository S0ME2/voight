from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.api.batch import process_image_batch
from app.api.schemas import BatchResponse
from app.api.upload_types import BatchUploads, SingleUpload
from app.config import Settings
from app.documents.mrz import ID_CARD, PASSPORT
from app.models import Models
from app.uploads import read_upload, read_uploaded_image
from app.workflows import (
    extract_document_mrz,
    extract_driving_license_fields,
    inspect_file_type,
    run_document_mrz,
    run_driving_license_fields,
)


def create_router(settings: Settings, models: Models) -> APIRouter:
    router = APIRouter()

    @router.post("/ocr/file-type")
    async def file_type(file: SingleUpload) -> dict[str, str | None]:
        return inspect_file_type(await read_upload(file), settings)

    @router.post("/ocr/id-card/mrz", response_class=PlainTextResponse)
    async def id_card_mrz(file: SingleUpload) -> PlainTextResponse:
        return PlainTextResponse(
            extract_document_mrz(
                await read_uploaded_image(file),
                "id_card_mrz",
                ID_CARD,
                models,
                settings,
            )
        )

    @router.post(
        "/ocr/id-card/mrz/batch",
        response_model=BatchResponse,
        summary="Process ID-card images or ZIP archives",
    )
    async def id_card_mrz_batch(files: BatchUploads) -> BatchResponse:
        return await process_image_batch(
            files,
            lambda upload, artifacts: run_document_mrz(
                upload,
                "id_card_mrz",
                ID_CARD,
                models,
                settings,
                artifacts=artifacts,
                save_input_artifacts=False,
            ),
            operation="id_card_mrz",
            settings=settings,
        )

    @router.post("/ocr/passport/mrz", response_class=PlainTextResponse)
    async def passport_mrz(file: SingleUpload) -> PlainTextResponse:
        return PlainTextResponse(
            extract_document_mrz(
                await read_uploaded_image(file),
                "passport_mrz",
                PASSPORT,
                models,
                settings,
            )
        )

    @router.post(
        "/ocr/passport/mrz/batch",
        response_model=BatchResponse,
        summary="Process passport images or ZIP archives",
    )
    async def passport_mrz_batch(files: BatchUploads) -> BatchResponse:
        return await process_image_batch(
            files,
            lambda upload, artifacts: run_document_mrz(
                upload,
                "passport_mrz",
                PASSPORT,
                models,
                settings,
                artifacts=artifacts,
                save_input_artifacts=False,
            ),
            operation="passport_mrz",
            settings=settings,
        )

    @router.post("/ocr/driving-license/extract")
    async def driving_license_extract(file: SingleUpload) -> dict:
        return extract_driving_license_fields(
            await read_uploaded_image(file),
            models,
            settings,
        )

    @router.post(
        "/ocr/driving-license/extract/batch",
        response_model=BatchResponse,
        summary="Process driving-licence images or ZIP archives",
    )
    async def driving_license_extract_batch(files: BatchUploads) -> BatchResponse:
        return await process_image_batch(
            files,
            lambda upload, artifacts: run_driving_license_fields(
                upload,
                models,
                settings,
                artifacts=artifacts,
                save_input_artifacts=False,
            ),
            operation="driving_license",
            settings=settings,
        )

    return router
