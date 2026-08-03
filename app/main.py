from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.routes import create_router
from app.config import Settings
from app.models import Models


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    models = Models(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        models.preload()
        yield

    app = FastAPI(title="VoightKampff OCR API", version="0.3.0", lifespan=lifespan)
    app.include_router(create_router(settings, models))
    return app


load_dotenv()
app = create_app()
