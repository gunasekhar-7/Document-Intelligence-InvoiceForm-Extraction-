"""Evaluation package."""

from document_intelligence.evaluation.metrics import (
    EvaluationResult,
    FieldMetrics,
    bootstrap_confidence_interval,
    compare_models,
    compute_field_match,
    evaluate_extraction,
    evaluate_token_classification,
    generate_evaluation_report,
    normalize_field_value,
)

__all__ = [
    "EvaluationResult",
    "FieldMetrics",
    "bootstrap_confidence_interval",
    "compare_models",
    "compute_field_match",
    "evaluate_extraction",
    "evaluate_token_classification",
    "generate_evaluation_report",
    "normalize_field_value",
]
