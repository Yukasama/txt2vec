"""Router Initializer."""

from fastapi import APIRouter, FastAPI

from vectorize.ai_model.router import router as models_router
from vectorize.config.config import settings
from vectorize.datasets.router import router as dataset_router
from vectorize.inference.router import router as embeddings_router
from vectorize.upload.router import router as upload_router


def register_routers(app: FastAPI) -> None:
    """Register all API routers with the FastAPI application.

    Args:
        app: FastAPI application instance.
    """
    base_router = APIRouter(prefix=settings.prefix)
    base_router.include_router(dataset_router, prefix="/datasets")
    base_router.include_router(upload_router, prefix="/uploads")
    base_router.include_router(embeddings_router, prefix="/embeddings")
    base_router.include_router(models_router, prefix="/models")

    app.include_router(base_router)
