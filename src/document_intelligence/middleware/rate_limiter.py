"""Rate limiting middleware using Redis."""

import time

import redis.asyncio as redis
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from document_intelligence.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-backed rate limiting middleware."""

    def __init__(
        self,
        app,
        redis_client: redis.Redis,
        requests_per_window: int = 100,
        window_seconds: int = 60,
        excluded_paths: list | None = None,
    ):
        super().__init__(app)
        self.redis = redis_client
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.excluded_paths = excluded_paths or [
            "/health",
            "/health/ready",
            "/health/live",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        # Get client identifier (IP + API key if present)
        client_ip = request.client.host if request.client else "unknown"
        auth_header = request.headers.get("Authorization", "")
        api_key = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
        client_id = f"{client_ip}:{api_key}" if api_key else client_ip

        # Redis key for rate limiting
        key = f"ratelimit:{client_id}"
        current_window = int(time.time() // self.window_seconds)
        window_key = f"{key}:{current_window}"

        # Increment counter
        pipe = self.redis.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, self.window_seconds * 2)
        results = await pipe.execute()

        current_count = results[0]

        # Check limit
        if current_count > self.requests_per_window:
            # Get TTL for retry-after header
            ttl = await self.redis.ttl(window_key)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(max(1, ttl))},
            )

        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_window - current_count)
        )
        response.headers["X-RateLimit-Reset"] = str((current_window + 1) * self.window_seconds)

        return response


def create_rate_limiter(
    redis_url: str,
    requests_per_window: int = None,
    window_seconds: int = None,
):
    """Create rate limiter middleware factory."""
    redis_client = redis.from_url(redis_url, decode_responses=True)

    def middleware(app):
        return RateLimitMiddleware(
            app,
            redis_client,
            requests_per_window=requests_per_window or settings.rate_limit_requests,
            window_seconds=window_seconds or settings.rate_limit_window_seconds,
        )

    return middleware
