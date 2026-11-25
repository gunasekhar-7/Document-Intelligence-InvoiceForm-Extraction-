"""Async worker for document processing using ARQ."""

import asyncio
import uuid

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy import desc, func, select

from document_intelligence.config import settings
from document_intelligence.db.models import Document, DocumentStatus, ProcessingJob
from document_intelligence.db.session import db_manager
from document_intelligence.logging import configure_logging, get_logger
from document_intelligence.services.extraction_service import ExtractionInput, ExtractionService
from document_intelligence.services.minio_client import MinioClient, get_minio_client

logger = get_logger(__name__)

# Global services
extraction_service: ExtractionService | None = None
minio_client: MinioClient | None = None
redis_pool = None
_shutdown_event = asyncio.Event()
_current_job_id: str | None = None


async def startup(ctx):
    """Worker startup - initialize services."""
    global extraction_service, minio_client, redis_pool

    configure_logging(settings.log_level, json_logs=settings.is_production)
    logger.info("Worker starting up")

    # Initialize DB engine
    _ = db_manager.engine

    # Initialize MinIO client
    minio_client = get_minio_client()

    # Initialize extraction service
    extraction_service = ExtractionService(minio_client=minio_client)

    # Create Redis pool
    redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))

    logger.info("Worker started successfully")


async def shutdown(ctx):
    """Worker shutdown - cleanup."""
    global redis_pool

    logger.info("Worker shutting down")
    _shutdown_event.set()

    if _current_job_id:
        logger.info("Waiting for current job to complete", job_id=_current_job_id)
        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=30.0)
        except TimeoutError:
            logger.warning("Shutdown timeout, forcing exit")

    if redis_pool:
        await redis_pool.close()
    await db_manager.close()
    logger.info("Worker stopped")


async def process_document(ctx, document_id: str) -> dict:
    """Process a single document."""
    global _current_job_id
    _current_job_id = document_id

    try:
        doc_uuid = uuid.UUID(str(document_id))
    except (ValueError, TypeError):
        logger.error("Invalid document UUID format", document_id=document_id)
        _current_job_id = None
        return {"status": "error", "message": "Invalid UUID"}

    logger.info("Processing document", document_id=str(doc_uuid))

    async with db_manager.session() as db:
        result = await db.execute(select(Document).where(Document.id == doc_uuid))
        document = result.scalar_one_or_none()

        if not document:
            logger.error("Document not found", document_id=str(doc_uuid))
            _current_job_id = None
            return {"status": "error", "message": "Document not found"}

        if document.status != DocumentStatus.QUEUED:
            logger.warning(
                "Document not in queued status",
                document_id=str(doc_uuid),
                status=document.status.value,
            )
            _current_job_id = None
            return {"status": "skipped", "message": f"Document status: {document.status.value}"}

        # Get latest processing job for this document
        job_result = await db.execute(
            select(ProcessingJob)
            .where(ProcessingJob.document_id == doc_uuid)
            .order_by(desc(ProcessingJob.created_at))
        )
        job = job_result.scalars().first()

        try:
            document.status = DocumentStatus.PROCESSING
            document.processing_started_at = func.now()
            if job:
                job.status = DocumentStatus.PROCESSING
                job.started_at = func.now()
                job.current_step = "downloading"
            await db.commit()

            if _shutdown_event.is_set():
                logger.info("Shutdown requested, requeuing job", document_id=str(doc_uuid))
                document.status = DocumentStatus.QUEUED
                if job:
                    job.status = DocumentStatus.QUEUED
                await db.commit()
                if redis_pool:
                    await redis_pool.enqueue_job(
                        "process_document", str(doc_uuid), _queue_name=settings.queue_name
                    )
                _current_job_id = None
                return {"status": "requeued", "message": "Shutdown requested"}

            # Download from MinIO
            file_data = minio_client.download_file(document.minio_object_key)

            if job:
                job.current_step = "preprocessing"
                job.progress_percent = 20
                await db.commit()

            # Run extraction pipeline
            input_data = ExtractionInput(
                file_bytes=file_data,
                filename=document.filename,
                mime_type=document.mime_type or "application/octet-stream",
                document_id=str(doc_uuid),
                priority=document.priority.value
                if hasattr(document.priority, "value")
                else str(document.priority),
            )

            if job:
                job.current_step = "extracting"
                job.progress_percent = 50
                await db.commit()

            result = extraction_service.extract(input_data)

            if job:
                job.current_step = "saving"
                job.progress_percent = 85
                await db.commit()

            # Update document with extracted data
            document.status = (
                DocumentStatus.COMPLETED if result.status == "completed" else DocumentStatus.FAILED
            )
            document.vendor_name = result.fields.get("vendor_name")
            document.invoice_number = result.fields.get("invoice_number")
            document.invoice_date = result.fields.get("invoice_date")
            document.due_date = result.fields.get("due_date")
            document.currency = result.fields.get("currency", "USD")
            document.subtotal = result.fields.get("subtotal")
            document.tax_amount = result.fields.get("tax_amount")
            document.total_amount = result.fields.get("total_amount")
            document.confidence_scores = result.confidence_scores
            document.raw_ocr_text = result.raw_ocr_text
            document.raw_entities = result.raw_entities
            document.processing_time_ms = result.processing_time_ms
            document.model_version = result.model_version
            document.ocr_engine = result.ocr_engine
            document.completed_at = func.now()
            document.error_message = result.error_message

            if job:
                job.status = document.status
                job.completed_at = func.now()
                job.progress_percent = 100
                job.current_step = "completed"

            await db.commit()
            logger.info(
                "Document processed successfully",
                document_id=str(doc_uuid),
                status=document.status.value,
            )
            _current_job_id = None
            return {"status": "success", "document_id": str(doc_uuid)}

        except Exception as e:
            logger.error("Processing failed", document_id=str(doc_uuid), error=str(e))
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)
            document.completed_at = func.now()

            if job:
                job.status = DocumentStatus.FAILED
                job.error_message = str(e)
                job.completed_at = func.now()

                if job.retry_count < job.max_retries:
                    job.retry_count += 1
                    job.status = DocumentStatus.QUEUED
                    defer_by = min(60 * (2**job.retry_count), 3600)
                    if redis_pool:
                        await redis_pool.enqueue_job(
                            "process_document",
                            str(doc_uuid),
                            _defer_by=defer_by,
                            _queue_name=settings.queue_name,
                        )
                    logger.info(
                        "Re-queued for retry",
                        document_id=str(doc_uuid),
                        retry=job.retry_count,
                        defer_by=defer_by,
                    )

            await db.commit()
            _current_job_id = None
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    from arq import run_worker

    from document_intelligence.workers.settings import WorkerSettings

    asyncio.run(run_worker(WorkerSettings))
