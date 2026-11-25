"""OpenTelemetry tracing configuration."""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from document_intelligence.config import settings


def setup_tracing(app=None) -> trace.TracerProvider:
    """Set up OpenTelemetry tracing."""
    resource = Resource.create({SERVICE_NAME: settings.otel_service_name})
    provider = TracerProvider(resource=resource)

    # Only export to network collector if in production
    if settings.is_production:
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=settings.otel_exporter_otlp_endpoint,
                insecure=True,
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        except Exception:
            pass

    trace.set_tracer_provider(provider)

    # Auto-instrument FastAPI if app provided and not already instrumented
    if app and not getattr(app.state, "otel_instrumented", False):
        try:
            FastAPIInstrumentor.instrument_app(app)
            app.state.otel_instrumented = True
        except Exception:
            pass

    # Auto-instrument other libraries
    SQLAlchemyInstrumentor().instrument()
    RedisInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()

    return provider


def get_tracer(name: str = None) -> trace.Tracer:
    """Get a tracer instance."""
    return trace.get_tracer(name or __name__)


# Context propagation helpers
def inject_trace_context(headers: dict) -> dict:
    """Inject current trace context into headers for downstream calls."""
    from opentelemetry.propagate import inject

    inject(headers)
    return headers


def extract_trace_context(headers: dict):
    """Extract trace context from incoming headers."""
    from opentelemetry.propagate import extract

    return extract(headers)


# Custom span attributes for document processing
class DocumentProcessingAttributes:
    DOCUMENT_ID = "document.id"
    DOCUMENT_FILENAME = "document.filename"
    DOCUMENT_MIME_TYPE = "document.mime_type"
    DOCUMENT_SIZE = "document.size_bytes"
    PROCESSING_STEP = "processing.step"
    PROCESSING_DURATION_MS = "processing.duration_ms"
    OCR_ENGINE = "ocr.engine"
    MODEL_VERSION = "model.version"
    EXTRACTION_STATUS = "extraction.status"
    FIELD_COUNT = "extraction.field_count"
    GPU_MEMORY_USED = "gpu.memory_used_bytes"
