from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.api.routes import create_router
from app.api.v1 import create_v1_router
from app.contracts import ErrorCode, ErrorResult
from app.config import Settings
from app.models import Models


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate_startup()
    models = Models(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        models.preload()
        try:
            yield
        finally:
            if close := getattr(models, "close", None):
                close()

    app = FastAPI(title="VoightKampff OCR API", version="0.3.0", lifespan=lifespan)
    app.include_router(create_router(settings, models))
    app.include_router(create_v1_router(settings, models))

    @app.exception_handler(HTTPException)
    async def contract_errors(_, exc):
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            return JSONResponse(status_code=exc.status_code, content={"error": detail})
        code = ErrorCode.UPLOAD_TOO_LARGE if exc.status_code == 413 else ErrorCode.INVALID_UPLOAD
        return JSONResponse(status_code=exc.status_code, content={"error": ErrorResult(code=code, detail=str(detail)).model_dump(mode="json")})
    return app


load_dotenv()
app = create_app()
