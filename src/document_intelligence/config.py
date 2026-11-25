from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application
    app_name: str = "document-intelligence"
    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = Field(
        default="document-intelligence-change-this-min-32-chars-production",
        min_length=32,
    )
    queue_name: str = "document_intelligence:queue"

    # Database
    database_url: str = Field(default="postgresql+asyncpg://localhost:5432/document_intelligence")

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_bucket: str = "documents"
    minio_secure: bool = False

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"

    # Model
    model_path: str = "models/best_model"
    ocr_engine: str = "paddleocr"
    ocr_lang: str = "en"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8501"]
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # Worker
    worker_concurrency: int = 2
    worker_timeout: int = 300
    model_warmup: bool = True

    # Monitoring
    prometheus_port: int = 9090
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "document-intelligence"

    # GPU
    cuda_visible_devices: str = "0"
    torch_cuda_arch_list: str = "7.0;7.5;8.0;8.6"

    # Demo
    streamlit_port: int = 8501

    # Auth
    api_keys: str = ""  # Comma-separated API keys - required in production
    jwt_secret: str = ""  # Required in production - set via environment
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Convenience access
settings = get_settings()
