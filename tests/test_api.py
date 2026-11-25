"""High-coverage integration tests for API routes with clean dependency injection."""

import io
import uuid
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from document_intelligence.config import settings
settings.app_env = "testing"

from document_intelligence.api.main import app
from document_intelligence.api.routes.documents import (
    get_arq_pool,
    get_db_session,
    get_extraction_service,
    get_minio_client,
)
from document_intelligence.db.models import Document, DocumentStatus, ModelVersion, ProcessingJob
from document_intelligence.services.minio_client import MinioClient


@pytest.fixture
def mock_minio():
    """Mock MinIO client for file operations."""
    mock = MagicMock(spec=MinioClient)
    mock.bucket = "documents"
    mock.upload_bytes.return_value = "documents/ab/mock_key.png"
    mock.download_file.return_value = b"%PDF-1.4 mock content"
    mock.delete_file.return_value = None
    mock.file_exists.return_value = True
    return mock


@pytest.fixture
def mock_arq():
    """Mock ARQ pool for background queuing."""
    mock = AsyncMock()
    mock.enqueue_job.return_value = MagicMock(job_id=str(uuid.uuid4()))
    return mock


@pytest.fixture
def mock_extractor():
    """Mock extraction service."""
    mock = MagicMock()
    mock.extract.return_value = MagicMock(
        document_id="test-doc-id",
        fields={
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-2024-001",
            "invoice_date": date(2024, 1, 15),
            "due_date": date(2024, 2, 15),
            "currency": "USD",
            "subtotal": 100.0,
            "tax_amount": 10.0,
            "total_amount": 110.0,
        },
        confidence_scores={
            "vendor_name": 0.95,
            "invoice_number": 0.98,
            "total_amount": 0.99,
        },
        raw_ocr_text="INVOICE\nAcme Corp\nINV-2024-001",
        raw_entities=[],
        processing_time_ms=120,
        model_version="hybrid-v1",
        ocr_engine="paddleocr",
        minio_key="documents/ab/mock_key.png",
        status="completed",
        error_message=None,
    )
    return mock


@pytest.fixture
def client(test_engine, mock_minio, mock_arq, mock_extractor):
    """FastAPI TestClient with all external services and database overridden."""
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db
    app.dependency_overrides[get_minio_client] = lambda: mock_minio
    app.dependency_overrides[get_arq_pool] = lambda: mock_arq
    app.dependency_overrides[get_extraction_service] = lambda: mock_extractor

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def valid_png_bytes():
    """Generate valid in-memory PNG image bytes with standard magic header."""
    img = Image.new("RGB", (200, 200), color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestHealthEndpoints:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_live(self, client):
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestDocumentUploadValidation:
    def test_upload_requires_file(self, client):
        response = client.post("/api/v1/documents")
        assert response.status_code == 422

    def test_upload_rejects_empty_file(self, client):
        response = client.post(
            "/api/v1/documents",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]

    def test_upload_rejects_large_file(self, client):
        large_file = b"x" * (11 * 1024 * 1024)
        response = client.post(
            "/api/v1/documents",
            files={"file": ("large.png", large_file, "image/png")},
        )
        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

    def test_upload_rejects_invalid_mime(self, client):
        response = client.post(
            "/api/v1/documents",
            files={"file": ("test.txt", b"plain text with no header", "text/plain")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]


class TestDocumentLifecycle:
    def test_upload_and_retrieve_document(self, client, valid_png_bytes):
        # 1. Upload valid document
        response = client.post(
            "/api/v1/documents",
            files={"file": ("invoice_sample.png", valid_png_bytes, "image/png")},
        )
        assert response.status_code == 202
        body = response.json()
        doc_id = body["document_id"]
        job_id = body["job_id"]
        assert doc_id is not None
        assert job_id is not None
        assert body["status"] == "queued"

        # 2. Check document metadata
        get_res = client.get(f"/api/v1/documents/{doc_id}")
        assert get_res.status_code == 200
        assert get_res.json()["id"] == doc_id
        assert get_res.json()["filename"] == "invoice_sample.png"

        # 3. Check list documents endpoint
        list_res = client.get("/api/v1/documents")
        assert list_res.status_code == 200
        list_data = list_res.json()
        assert list_data["total"] >= 1
        assert any(d["id"] == doc_id for d in list_data["documents"])

        # 4. Check job status endpoint using exact job_id returned
        job_res = client.get(f"/api/v1/jobs/{job_id}")
        assert job_res.status_code == 200
        assert job_res.json()["job_id"] == job_id
        assert job_res.json()["status"] == "queued"

        # 5. Delete document
        del_res = client.delete(f"/api/v1/documents/{doc_id}")
        assert del_res.status_code == 200
        assert "deleted successfully" in del_res.json()["message"]

        # 6. Verify 404 after deletion
        not_found_res = client.get(f"/api/v1/documents/{doc_id}")
        assert not_found_res.status_code == 404


class TestModelRegistryEndpoints:
    def test_register_and_activate_model(self, client):
        version = f"v-test-{uuid.uuid4().hex[:6]}"
        register_res = client.post(
            "/api/v1/models",
            data={
                "version": version,
                "model_type": "layoutlmv3",
                "model_path": f"models/{version}",
                "metrics": '{"f1": 0.92, "precision": 0.91, "recall": 0.93}',
                "training_config": '{"epochs": 5}',
                "is_active": "false",
            },
        )
        assert register_res.status_code == 201
        model_id = register_res.json()["id"]

        # Activate model
        activate_res = client.post(f"/api/v1/models/{model_id}/activate")
        assert activate_res.status_code == 200
        assert activate_res.json()["is_active"] is True

        # Query active model
        active_res = client.get("/api/v1/models/active/layoutlmv3")
        assert active_res.status_code == 200
        assert active_res.json()["version"] == version
