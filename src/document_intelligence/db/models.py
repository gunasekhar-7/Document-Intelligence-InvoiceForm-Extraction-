"""Database models and session management."""

import enum
import uuid

import sqlalchemy.dialects.postgresql as pg
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class GUID(TypeDecorator):
    """Platform-independent GUID/UUID type.
    Uses PostgreSQL's native UUID, falls back to String(36) on SQLite.
    """

    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(pg.UUID(as_uuid=True))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            try:
                return uuid.UUID(str(value))
            except (ValueError, AttributeError):
                return value
        return value


class JSONB_TYPE(TypeDecorator):
    """Platform-independent JSONB type.
    Uses PostgreSQL's native JSONB, falls back to JSON on SQLite.
    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(pg.JSONB())
        return dialect.type_descriptor(JSON())


class INET_TYPE(TypeDecorator):
    """Platform-independent INET type.
    Uses PostgreSQL's native INET, falls back to String(45) on SQLite.
    """

    impl = String(45)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(pg.INET())
        return dialect.type_descriptor(String(45))


class DocumentStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingPriority(str, enum.Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Document(Base):
    __tablename__ = "documents"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    content_hash = Column(String(64), nullable=False, index=True)
    mime_type = Column(String(100))
    file_size_bytes = Column(Integer)
    page_count = Column(Integer, default=1)
    status = Column(
        SQLEnum(DocumentStatus), nullable=False, default=DocumentStatus.QUEUED, index=True
    )
    priority = Column(
        SQLEnum(ProcessingPriority), nullable=False, default=ProcessingPriority.NORMAL
    )

    # Extracted fields
    vendor_name = Column(String(255), index=True)
    vendor_address = Column(Text)
    vendor_tax_id = Column(String(100))
    invoice_number = Column(String(100), index=True)
    invoice_date = Column(Date, index=True)
    due_date = Column(Date)
    currency = Column(String(3), default="USD")
    subtotal = Column(Numeric(14, 2))
    tax_amount = Column(Numeric(14, 2))
    total_amount = Column(Numeric(14, 2))

    # Confidence scores
    confidence_scores = Column(JSONB_TYPE, default=dict)

    # Raw extraction data
    raw_ocr_text = Column(Text)
    raw_entities = Column(JSONB_TYPE, default=list)

    # Processing metadata
    processing_time_ms = Column(Integer)
    model_version = Column(String(50))
    ocr_engine = Column(String(50))
    error_message = Column(Text)

    # Storage
    minio_bucket = Column(String(100))
    minio_object_key = Column(String(500))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), index=True)
    queued_at = Column(DateTime(timezone=True), server_default=func.now())
    processing_started_at = Column(DateTime(timezone=True))

    processing_jobs = relationship(
        "ProcessingJob",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("idx_documents_vendor_date", "vendor_name", "invoice_date"),
        Index("idx_documents_created_desc", "created_at"),
    )


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    document_id = Column(
        GUID, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    worker_id = Column(String(100))
    status = Column(
        SQLEnum(DocumentStatus), nullable=False, default=DocumentStatus.QUEUED, index=True
    )
    current_step = Column(String(100))
    progress_percent = Column(Integer, default=0)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="processing_jobs")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    version = Column(String(50), nullable=False, unique=True)
    model_type = Column(String(50), nullable=False)
    model_path = Column(String(500), nullable=False)
    metrics = Column(JSONB_TYPE, nullable=False)
    training_config = Column(JSONB_TYPE)
    is_active = Column(Boolean, default=False, index=True)
    deployed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    document_id = Column(GUID, ForeignKey("documents.id", ondelete="SET NULL"), index=True)
    user_id = Column(String(255))
    action = Column(String(100), nullable=False)
    details = Column(JSONB_TYPE)
    ip_address = Column(INET_TYPE)
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
