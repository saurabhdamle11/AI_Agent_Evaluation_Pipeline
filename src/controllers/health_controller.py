from fastapi import APIRouter

from src.config.settings import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health", summary="Liveness check")
async def health() -> dict:
    return {"status": "ok", "env": settings.app_env}


@router.get("/readiness", summary="Readiness check")
async def readiness() -> dict:
    # TODO: add Mongo ping + Kafka metadata fetch to confirm connectivity
    return {"status": "ready"}
