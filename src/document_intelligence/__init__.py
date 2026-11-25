"""Document Intelligence - Invoice & Form Extraction System."""

from .baseline_extractor import (
    BaselineExtractor,
    ExtractedField,
    ExtractionResult,
    FieldType,
    create_extractor,
)
from .config import Settings, get_settings, settings
from .logging import configure_logging, get_logger
from .telemetry import DocumentProcessingAttributes, get_tracer, setup_tracing

__version__ = "0.1.0"

__all__ = [
    "BaselineExtractor",
    "DocumentProcessingAttributes",
    "ExtractedField",
    "ExtractionResult",
    "FieldType",
    "Settings",
    "configure_logging",
    "create_extractor",
    "get_logger",
    "get_settings",
    "get_tracer",
    "settings",
    "setup_tracing",
]
