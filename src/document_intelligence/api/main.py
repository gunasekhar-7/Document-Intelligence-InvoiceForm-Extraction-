"""FastAPI application factory."""

from contextlib import asynccontextmanager

import redis.asyncio as redis
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from document_intelligence.config import settings
from document_intelligence.db.session import close_db, init_db
from document_intelligence.logging import configure_logging, get_logger
from document_intelligence.middleware.rate_limiter import create_rate_limiter
from document_intelligence.telemetry import setup_tracing

logger = get_logger(__name__)

# Global services
extraction_service = None
minio_client = None
redis_pool = None
arq_pool = None
redis_client = None

# Security
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global extraction_service, minio_client, redis_pool, arq_pool, redis_client

    if settings.app_env == "testing":
        yield
        return

    # Startup
    configure_logging(settings.log_level, json_logs=settings.is_production)
    logger.info("Starting application", env=settings.app_env)

    # Setup OpenTelemetry tracing
    setup_tracing(app)

    if settings.app_env != "testing":
        await init_db()

    # Initialize MinIO
    from document_intelligence.services.minio_client import get_minio_client

    minio_client = get_minio_client()

    # Initialize extraction service (singleton for warm model)
    from document_intelligence.services.extraction_service import ExtractionService

    extraction_service = ExtractionService(minio_client=minio_client)

    # Initialize Redis client for rate limiting and ARQ if not in testing mode
    if settings.app_env != "testing":
        try:
            redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
            arq_pool = redis_pool
        except Exception as redis_err:
            logger.warning("Could not connect to Redis during startup", error=str(redis_err))

    logger.info("Application started")

    yield

    # Shutdown
    logger.info("Shutting down application")
    if redis_pool:
        await redis_pool.close()
    if redis_client:
        await redis_client.close()
    await close_db()
    logger.info("Application stopped")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Document Intelligence API",
        description="Automated invoice and form extraction API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting middleware (Redis-backed in production)
    if settings.is_production:
        app.add_middleware(
            create_rate_limiter(
                settings.redis_url,
                requests_per_window=settings.rate_limit_requests,
                window_seconds=settings.rate_limit_window_seconds,
            )
        )
    else:
        # Simple no-op in development
        @app.middleware("http")
        async def dev_rate_limit_middleware(request: Request, call_next):
            return await call_next(request)

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error("Unhandled exception", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # Health check
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": "0.1.0"}

    # Include routes
    from document_intelligence.api.routes import documents, health, models

    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")

    return app


# Dependency functions
async def get_extraction_service() -> "ExtractionService":
    global extraction_service
    if extraction_service is None:
        raise HTTPException(503, "Extraction service not initialized")
    return extraction_service


async def get_minio_client() -> "MinioClient":
    global minio_client
    if minio_client is None:
        raise HTTPException(503, "MinIO client not initialized")
    return minio_client


async def get_arq_pool():
    global arq_pool
    if arq_pool is None:
        raise HTTPException(503, "ARQ pool not initialized")
    return arq_pool


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token and return user info."""
    if not credentials:
        raise HTTPException(401, "Authorization header required")

    api_key = credentials.credentials
    if not validate_api_key(api_key):
        raise HTTPException(401, "Invalid API key")

    return {"api_key": api_key, "scopes": ["read", "write"]}


def validate_api_key(api_key: str) -> bool:
    """Validate API key against configured keys."""
    # In production, check against database or secret store
    valid_keys = settings.api_keys.split(",") if settings.api_keys else []
    return api_key in valid_keys


app = create_app()
