-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create mlflow database for experiment tracking
CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO docintel;

-- Connect to document_intelligence database
\c document_intelligence;

-- Create enums
CREATE TYPE document_status AS ENUM ('queued', 'processing', 'completed', 'failed');
CREATE TYPE processing_priority AS ENUM ('low', 'normal', 'high', 'urgent');

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255),
    content_hash CHAR(64) NOT NULL,
    mime_type VARCHAR(100),
    file_size_bytes BIGINT,
    page_count INTEGER DEFAULT 1,
    status document_status NOT NULL DEFAULT 'queued',
    priority processing_priority NOT NULL DEFAULT 'normal',

    -- Extracted fields
    vendor_name VARCHAR(255),
    vendor_address TEXT,
    vendor_tax_id VARCHAR(100),
    invoice_number VARCHAR(100),
    invoice_date DATE,
    due_date DATE,
    currency CHAR(3) DEFAULT 'USD',
    subtotal DECIMAL(14, 2),
    tax_amount DECIMAL(14, 2),
    total_amount DECIMAL(14, 2),

    -- Confidence scores (per field)
    confidence_scores JSONB DEFAULT '{}',

    -- Raw extraction data
    raw_ocr_text TEXT,
    raw_entities JSONB DEFAULT '[]',

    -- Processing metadata
    processing_time_ms INTEGER,
    model_version VARCHAR(50),
    ocr_engine VARCHAR(50),
    error_message TEXT,

    -- Storage
    minio_bucket VARCHAR(100),
    minio_object_key VARCHAR(500),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    queued_at TIMESTAMPTZ DEFAULT NOW(),
    processing_started_at TIMESTAMPTZ
);

-- Indexes for common queries
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_vendor_date ON documents(vendor_name, invoice_date);
CREATE INDEX idx_documents_hash ON documents(content_hash);
CREATE INDEX idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX idx_documents_completed_at ON documents(completed_at DESC) WHERE completed_at IS NOT NULL;
CREATE INDEX idx_documents_vendor_trgm ON documents USING GIN (vendor_name gin_trgm_ops);

-- Processing jobs table (for tracking async work)
CREATE TABLE processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    worker_id VARCHAR(100),
    status document_status NOT NULL DEFAULT 'queued',
    current_step VARCHAR(100),
    progress_percent INTEGER DEFAULT 0,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_processing_jobs_document ON processing_jobs(document_id);
CREATE INDEX idx_processing_jobs_status ON processing_jobs(status);

-- Model versions table
CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(50) NOT NULL UNIQUE,
    model_type VARCHAR(50) NOT NULL, -- 'layoutlmv3', 'paddleocr', etc.
    model_path VARCHAR(500) NOT NULL,
    metrics JSONB NOT NULL, -- F1 scores, etc.
    training_config JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_model_versions_active ON model_versions(is_active) WHERE is_active = TRUE;

-- Audit log for compliance
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_log_document ON audit_log(document_id);
CREATE INDEX idx_audit_log_created_at ON audit_log(created_at DESC);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- View for document summary
CREATE VIEW document_summary AS
SELECT
    id,
    filename,
    vendor_name,
    invoice_number,
    invoice_date,
    total_amount,
    currency,
    status,
    confidence_scores,
    processing_time_ms,
    created_at,
    completed_at
FROM documents;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO docintel;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO docintel;