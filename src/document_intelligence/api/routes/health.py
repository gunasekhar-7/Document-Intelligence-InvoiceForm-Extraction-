"""Health check routes."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from document_intelligence.api.main import minio_client as global_minio_client
from document_intelligence.config import settings
from document_intelligence.db.session import get_db_session
from document_intelligence.services.minio_client import MinioClient

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "service": "document-intelligence",
        "version": "0.1.0",
    }


@router.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(get_db_session)):
    """Readiness check - verifies all dependencies."""
    checks = {}

    # Database
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {e!s}"

    # MinIO
    try:
        client = global_minio_client or MinioClient()
        client.client.list_buckets()
        checks["minio"] = "healthy"
    except Exception as e:
        checks["minio"] = f"unhealthy: {e!s}"

    # Redis (via ARQ)
    try:
        import redis

        r = redis.from_url(settings.redis_url)
        r.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {e!s}"

    all_healthy = all(v == "healthy" for v in checks.values())

    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks,
    }


@router.get("/health/live")
async def liveness_check():
    """Liveness check - minimal check for container orchestration."""
    return {"status": "alive"}
