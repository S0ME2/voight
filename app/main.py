from contextlib import asynccontextmanager
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("OMP_NUM_THREADS", "1")

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.api.v1 import create_v1_router
from app.api.verification import create_verification_router
from app.contracts import ErrorCode, ErrorResult
from app.config import Settings
from app.models import Models

logger = logging.getLogger("uvicorn.error")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate_startup()
    models = Models(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        models.preload()
        if configuration := getattr(models, "configuration", None):
            logger.info("effective Voight configuration: %s", json.dumps(configuration(), sort_keys=True, default=str))
        try:
            yield
        finally:
            if close := getattr(models, "close", None):
                close()

    app = FastAPI(
        title="VoightKampff OCR API",
        version="0.3.0",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "Health", "description": "Service liveness and readiness checks."},
            {"name": "v1 OCR", "description": "Profile-driven OCR for supported document layouts."},
            {"name": "Verification OCR", "description": "Whole-image OCR returning raw text lines."},
            {"name": "Verification checks", "description": "Compare expected fields with OCR lines."},
        ],
    )
    app.include_router(create_v1_router(settings, models))
    app.include_router(create_verification_router(settings, models))

    @app.exception_handler(HTTPException)
    async def contract_errors(_, exc):
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            return JSONResponse(status_code=exc.status_code, content={"error": detail})
        code = ErrorCode.UPLOAD_TOO_LARGE if exc.status_code == 413 else ErrorCode.INVALID_UPLOAD
        return JSONResponse(status_code=exc.status_code, content={"error": ErrorResult(code=code, detail=str(detail)).model_dump(mode="json")})
    return app

app = create_app()
