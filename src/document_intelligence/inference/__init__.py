"""Inference package - model serving utilities."""

from document_intelligence.inference.model_server import (
    DocumentIntelligenceModelServer,
    get_model_server,
)

__all__ = ["DocumentIntelligenceModelServer", "get_model_server"]
