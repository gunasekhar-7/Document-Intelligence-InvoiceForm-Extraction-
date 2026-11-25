"""API routes for document management."""

import io
import uuid
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from document_intelligence.config import settings
from document_intelligence.db.models import Document, DocumentStatus, ProcessingJob
from document_intelligence.db.session import get_db_session
from document_intelligence.logging import get_logger
from document_intelligence.services.extraction_service import ExtractionService
from document_intelligence.services.minio_client import (
    MinioClient,
    compute_content_hash,
    generate_object_key,
)

logger = get_logger(__name__)

router = APIRouter()


# Request/Response models
class DocumentUploadResponse(BaseModel):
    document_id: str
    job_id: str
    status: str
    message: str


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Any
    filename: str
    original_filename: str | None = None
    mime_type: str | None = None
    file_size_bytes: int | None = None
    page_count: int = 1
    status: Any
    vendor_name: str | None = None
    invoice_number: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    currency: str = "USD"
    subtotal: float | None = None
    tax_amount: float | None = None
    total_amount: float | None = None
    confidence_scores: dict = Field(default_factory=dict)
    processing_time_ms: int | None = None
    model_version: str | None = None
    ocr_engine: str | None = None
    error_message: str | None = None
    created_at: Any = None
    updated_at: Any = None
    completed_at: Any = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    page: int
    page_size: int


class ExtractionResultResponse(BaseModel):
    document_id: str
    fields: dict
    confidence_scores: dict
    raw_ocr_text: str
    raw_entities: list
    processing_time_ms: int
    model_version: str
    ocr_engine: str


class JobStatusResponse(BaseModel):
    job_id: str
    document_id: str
    status: str
    current_step: str | None
    progress_percent: int
    error_message: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None


# Dependency to get extraction service
async def get_extraction_service() -> ExtractionService:
    from document_intelligence.api.main import extraction_service

    if extraction_service is None:
        raise HTTPException(503, "Extraction service not initialized")
    return extraction_service


async def get_minio_client() -> MinioClient:
    from document_intelligence.api.main import minio_client

    if minio_client is None:
        raise HTTPException(503, "MinIO client not initialized")
    return minio_client


async def get_arq_pool():
    from document_intelligence.api.main import get_arq_pool

    return await get_arq_pool()


# File type validation
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/tif",
}

ALLOWED_MAGIC_BYTES = {
    b"%PDF": "application/pdf",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"II*\x00": "image/tiff",
    b"MM\x00*": "image/tiff",
}


def validate_file_type(file_bytes: bytes, content_type: str) -> str:
    """Validate file type using magic bytes."""
    # Check magic bytes
    detected_type = None
    for magic_bytes, mime in ALLOWED_MAGIC_BYTES.items():
        if file_bytes.startswith(magic_bytes):
            detected_type = mime
            break

    if not detected_type:
        raise HTTPException(400, "Unsupported file type. Allowed: PDF, PNG, JPEG, TIFF")

    # Optionally verify against provided content-type
    if content_type and content_type != detected_type:
        logger.warning("Content-Type mismatch", provided=content_type, detected=detected_type)

    return detected_type


@router.post("/documents", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    metadata: str | None = Form(None),
    priority: str = Form("normal"),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    extraction_service: ExtractionService = Depends(get_extraction_service),
    minio_client: MinioClient = Depends(get_minio_client),
    arq_pool=Depends(get_arq_pool),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Upload a document for extraction.

    - **file**: Document file (PDF, PNG, JPG, TIFF)
    - **metadata**: Optional JSON metadata
    - **priority**: Processing priority (low, normal, high, urgent)
    - **Idempotency-Key**: Optional header for safe retries
    """
    # Validate file
    if not file.filename:
        raise HTTPException(400, "Filename is required")

    # Read file
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "Empty file")

    # Check file size (10MB limit)
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10MB)")

    # Validate file type using magic bytes
    mime_type = validate_file_type(file_bytes, file.content_type)

    # Compute content hash
    content_hash = compute_content_hash(file_bytes)

    # Check for duplicate
    existing = await db.execute(select(Document).where(Document.content_hash == content_hash))
    existing_doc = existing.scalar_one_or_none()
    if existing_doc:
        # If idempotency key provided, return existing document
        if idempotency_key:
            # Check if this idempotency key was used for this document
            # For now, just return existing
            return DocumentUploadResponse(
                document_id=str(existing_doc.id),
                job_id="",
                status=existing_doc.status.value,
                message="Document already exists (idempotent)",
            )
        raise HTTPException(409, "Document already exists")

    # Create document record
    document = Document(
        filename=file.filename,
        original_filename=file.filename,
        content_hash=content_hash,
        mime_type=mime_type,
        file_size_bytes=len(file_bytes),
        status=DocumentStatus.QUEUED,
        priority=priority,
    )
    db.add(document)
    await db.flush()

    # Create processing job with consistent job_id
    job_uuid = uuid.uuid4()
    job_id = str(job_uuid)
    job = ProcessingJob(
        id=job_uuid,
        document_id=document.id,
        status=DocumentStatus.QUEUED,
    )
    db.add(job)
    await db.commit()

    # Upload to MinIO
    minio_key = generate_object_key(file.filename, content_hash)
    minio_client.upload_bytes(minio_key, file_bytes)

    # Update document with MinIO key
    document.minio_bucket = minio_client.bucket
    document.minio_object_key = minio_key
    await db.commit()

    # Enqueue job for async processing to designated queue
    try:
        await arq_pool.enqueue_job(
            "process_document",
            str(document.id),
            _job_id=job_id,
            _queue_name=settings.queue_name,
        )
        logger.info(
            "Document queued for processing",
            document_id=str(document.id),
            job_id=job_id,
            queue=settings.queue_name,
        )
    except Exception as e:
        logger.exception("Failed to enqueue job", document_id=str(document.id))
        document.status = DocumentStatus.FAILED
        document.error_message = f"Failed to queue: {e!s}"
        job.status = DocumentStatus.FAILED
        job.error_message = str(e)
        await db.commit()
        raise HTTPException(500, "Failed to queue document for processing")

    return DocumentUploadResponse(
        document_id=str(document.id),
        job_id=job_id,
        status="queued",
        message="Document queued for processing",
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    status: DocumentStatus | None = Query(None),
    vendor_name: str | None = Query(None),
    invoice_number: str | None = Query(None),
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
):
    """List documents with filtering and pagination."""
    query = select(Document)

    if status:
        query = query.where(Document.status == status)
    if vendor_name:
        query = query.where(Document.vendor_name.ilike(f"%{vendor_name}%"))
    if invoice_number:
        query = query.where(Document.invoice_number.ilike(f"%{invoice_number}%"))
    if date_from:
        query = query.where(Document.invoice_date >= date_from)
    if date_to:
        query = query.where(Document.invoice_date <= date_to)

    # Total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)

    # Pagination
    query = query.order_by(desc(Document.created_at))
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    documents = result.scalars().all()

    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(doc) for doc in documents],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Get document by ID."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(404, "Document not found")

    return DocumentResponse.model_validate(document)


@router.get("/documents/{document_id}/extraction", response_model=ExtractionResultResponse)
async def get_extraction_result(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Get detailed extraction results."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(404, "Document not found")

    if document.status != DocumentStatus.COMPLETED:
        raise HTTPException(400, f"Document not processed yet (status: {document.status.value})")

    return ExtractionResultResponse(
        document_id=str(document.id),
        fields={
            "vendor_name": document.vendor_name,
            "invoice_number": document.invoice_number,
            "invoice_date": document.invoice_date.isoformat() if document.invoice_date else None,
            "due_date": document.due_date.isoformat() if document.due_date else None,
            "currency": document.currency,
            "subtotal": float(document.subtotal) if document.subtotal else None,
            "tax_amount": float(document.tax_amount) if document.tax_amount else None,
            "total_amount": float(document.total_amount) if document.total_amount else None,
        },
        confidence_scores=document.confidence_scores or {},
        raw_ocr_text=document.raw_ocr_text or "",
        raw_entities=document.raw_entities or [],
        processing_time_ms=document.processing_time_ms or 0,
        model_version=document.model_version or "unknown",
        ocr_engine=document.ocr_engine or "unknown",
    )


@router.get("/documents/{document_id}/download")
async def download_document(
    document_id: uuid.UUID,
    minio_client: MinioClient = Depends(get_minio_client),
    db: AsyncSession = Depends(get_db_session),
):
    """Download original document file."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(404, "Document not found")

    if not document.minio_object_key:
        raise HTTPException(404, "Document file not found in storage")

    try:
        file_data = minio_client.download_file(document.minio_object_key)
        return StreamingResponse(
            io.BytesIO(file_data),
            media_type=document.mime_type or "application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{document.original_filename or document.filename}"'
            },
        )
    except Exception as e:
        logger.exception("Failed to download document", document_id=str(document_id))
        raise HTTPException(404, "File not found in storage")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: uuid.UUID,
    minio_client: MinioClient = Depends(get_minio_client),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete document and associated files."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(404, "Document not found")

    # Delete from MinIO
    if document.minio_object_key:
        try:
            minio_client.delete_file(document.minio_object_key)
        except Exception as e:
            logger.warning(
                "Failed to delete from MinIO", key=document.minio_object_key, error=str(e)
            )

    # Delete from DB (cascades to processing_jobs)
    await db.delete(document)
    await db.commit()

    return {"message": "Document deleted successfully"}


# Job status endpoints
@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Get processing job status."""
    result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(404, "Job not found")

    return JobStatusResponse(
        job_id=str(job.id),
        document_id=str(job.document_id),
        status=job.status.value,
        current_step=job.current_step,
        progress_percent=job.progress_percent,
        error_message=job.error_message,
        created_at=job.created_at.isoformat() if job.created_at else "",
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@router.get("/documents/{document_id}/jobs", response_model=list[JobStatusResponse])
async def get_document_jobs(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    """Get all processing jobs for a document."""
    result = await db.execute(
        select(ProcessingJob)
        .where(ProcessingJob.document_id == document_id)
        .order_by(ProcessingJob.created_at.desc())
    )
    jobs = result.scalars().all()

    return [
        JobStatusResponse(
            job_id=str(job.id),
            document_id=str(job.document_id),
            status=job.status.value,
            current_step=job.current_step,
            progress_percent=job.progress_percent,
            error_message=job.error_message,
            created_at=job.created_at.isoformat() if job.created_at else "",
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
        )
        for job in jobs
    ]
