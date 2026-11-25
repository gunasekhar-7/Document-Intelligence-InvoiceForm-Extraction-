"""Evaluation framework for document extraction models."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


@dataclass
class FieldMetrics:
    """Metrics for a single field."""

    field_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    exact_matches: int = 0
    total_ground_truth: int = 0
    total_predictions: int = 0

    @property
    def precision(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.true_positives / self.total_predictions

    @property
    def recall(self) -> float:
        if self.total_ground_truth == 0:
            return 0.0
        return self.true_positives / self.total_ground_truth

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def exact_match_accuracy(self) -> float:
        if self.total_ground_truth == 0:
            return 0.0
        return self.exact_matches / self.total_ground_truth

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "exact_matches": self.exact_matches,
            "total_ground_truth": self.total_ground_truth,
            "total_predictions": self.total_predictions,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "exact_match_accuracy": self.exact_match_accuracy,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation results."""

    field_metrics: dict[str, FieldMetrics] = field(default_factory=dict)
    overall_metrics: FieldMetrics = field(default_factory=lambda: FieldMetrics("overall"))
    per_document_results: list[dict] = field(default_factory=list)
    error_analysis: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def to_dict(self) -> dict:
        return {
            "field_metrics": {k: v.to_dict() for k, v in self.field_metrics.items()},
            "overall_metrics": self.overall_metrics.to_dict(),
            "per_document_results": self.per_document_results,
            "error_analysis": dict(self.error_analysis),
        }


def normalize_field_value(value: str, field_type: str) -> str:
    """Normalize field value for comparison."""
    if value is None:
        return ""

    normalized = value.strip().lower()

    # Remove common separators
    normalized = normalized.replace(",", "").replace(" ", "")

    # Currency normalization
    if field_type in ["subtotal", "tax_amount", "total_amount"]:
        # Extract just the number
        import re

        match = re.search(r"[\d.]+", normalized)
        if match:
            normalized = match.group(0)

    # Date normalization
    if field_type in ["invoice_date", "due_date"]:
        # Try to parse and reformat
        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y"]:
            try:
                from datetime import datetime

                dt = datetime.strptime(normalized, fmt)
                normalized = dt.strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

    return normalized


def compute_field_match(pred: str, gt: str, field_type: str) -> bool:
    """Check if predicted field matches ground truth."""
    pred_norm = normalize_field_value(pred, field_type)
    gt_norm = normalize_field_value(gt, field_type)
    return pred_norm == gt_norm


def evaluate_extraction(
    predictions: list[dict],
    ground_truth: list[dict],
    field_types: list[str] = None,
) -> EvaluationResult:
    """
    Evaluate extraction results against ground truth.

    Args:
        predictions: List of prediction dicts with 'fields' key
        ground_truth: List of ground truth dicts with same structure
        field_types: List of field types to evaluate

    Returns:
        EvaluationResult with detailed metrics
    """
    if field_types is None:
        field_types = [
            "invoice_number",
            "invoice_date",
            "due_date",
            "vendor_name",
            "vendor_address",
            "vendor_tax_id",
            "subtotal",
            "tax_amount",
            "total_amount",
            "currency",
        ]

    result = EvaluationResult()

    # Initialize field metrics
    for ft in field_types:
        result.field_metrics[ft] = FieldMetrics(ft)

    for pred_doc, gt_doc in zip(predictions, ground_truth):
        pred_fields = pred_doc.get("fields", {})
        gt_fields = gt_doc.get("fields", {})

        doc_result = {
            "document_id": gt_doc.get("document_id", "unknown"),
            "fields": {},
        }

        for field_type in field_types:
            pred_value = pred_fields.get(field_type, "")
            gt_value = gt_fields.get(field_type, "")

            metrics = result.field_metrics[field_type]
            metrics.total_ground_truth += 1 if gt_value else 0
            metrics.total_predictions += 1 if pred_value else 0

            is_match = compute_field_match(pred_value, gt_value, field_type)

            field_result = {
                "predicted": pred_value,
                "ground_truth": gt_value,
                "match": is_match,
            }

            if gt_value and pred_value:
                if is_match:
                    metrics.true_positives += 1
                    metrics.exact_matches += 1
                else:
                    metrics.false_positives += 1
                    metrics.false_negatives += 1
                    # Categorize error
                    if not pred_value:
                        result.error_analysis["missing_prediction"] += 1
                    elif not gt_value:
                        result.error_analysis["hallucination"] += 1
                    else:
                        result.error_analysis["value_mismatch"] += 1
            elif gt_value and not pred_value:
                metrics.false_negatives += 1
                result.error_analysis["missing_prediction"] += 1
            elif pred_value and not gt_value:
                metrics.false_positives += 1
                result.error_analysis["hallucination"] += 1

            doc_result["fields"][field_type] = field_result

        # Update overall metrics
        for ft in field_types:
            m = result.field_metrics[ft]
            result.overall_metrics.true_positives += m.true_positives
            result.overall_metrics.false_positives += m.false_positives
            result.overall_metrics.false_negatives += m.false_negatives
            result.overall_metrics.exact_matches += m.exact_matches
            result.overall_metrics.total_ground_truth += m.total_ground_truth
            result.overall_metrics.total_predictions += m.total_predictions

        result.per_document_results.append(doc_result)

    return result


def evaluate_token_classification(
    predictions: list[list[str]],
    labels: list[list[str]],
) -> dict[str, float]:
    """Evaluate token-level classification (for LayoutLMv3)."""
    # Filter out padding tokens (-100)
    true_predictions = [
        [p for p, l in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for p, l in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)
    ]

    return {
        "f1": f1_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "report": classification_report(true_labels, true_predictions, output_dict=True),
    }


def bootstrap_confidence_interval(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(values) == 0:
        return (0.0, 0.0)

    bootstrapped = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrapped.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrapped, alpha * 100)
    upper = np.percentile(bootstrapped, (1 - alpha) * 100)

    return (float(lower), float(upper))


def compare_models(
    baseline_results: EvaluationResult,
    model_results: EvaluationResult,
) -> dict[str, Any]:
    """Compare two models statistically."""

    comparison = {}

    for field in baseline_results.field_metrics:
        base = baseline_results.field_metrics[field]
        model = model_results.field_metrics[field]

        # McNemar's test for paired comparison
        # Need contingency table: both correct, baseline only, model only, both wrong
        # This requires per-document paired data

        comparison[field] = {
            "baseline_f1": base.f1,
            "model_f1": model.f1,
            "improvement": model.f1 - base.f1,
            "relative_improvement": (model.f1 - base.f1) / base.f1 if base.f1 > 0 else 0,
        }

    return comparison


def generate_evaluation_report(
    result: EvaluationResult,
    output_path: str,
    model_name: str = "model",
    baseline_result: EvaluationResult | None = None,
) -> None:
    """Generate HTML evaluation report."""
    import jinja2

    template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ model_name }} Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric-good { color: green; font-weight: bold; }
        .metric-bad { color: red; }
        .improved { color: green; }
        .regressed { color: red; }
        h1, h2 { color: #333; }
    </style>
</head>
<body>
    <h1>{{ model_name }} Evaluation Report</h1>

    <h2>Overall Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Precision</td><td>{{ "%.4f"|format(overall.precision) }}</td></tr>
        <tr><td>Recall</td><td>{{ "%.4f"|format(overall.recall) }}</td></tr>
        <tr><td>F1 Score</td><td>{{ "%.4f"|format(overall.f1) }}</td></tr>
        <tr><td>Exact Match Accuracy</td><td>{{ "%.4f"|format(overall.exact_match_accuracy) }}</td></tr>
    </table>

    <h2>Per-Field Metrics</h2>
    <table>
        <tr>
            <th>Field</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Exact Match</th>
            <th>Support</th>
        </tr>
        {% for field_name, metrics in field_metrics.items() %}
        <tr>
            <td>{{ field_name }}</td>
            <td>{{ "%.4f"|format(metrics.precision) }}</td>
            <td>{{ "%.4f"|format(metrics.recall) }}</td>
            <td>{{ "%.4f"|format(metrics.f1) }}</td>
            <td>{{ "%.4f"|format(metrics.exact_match_accuracy) }}</td>
            <td>{{ metrics.total_ground_truth }}</td>
        </tr>
        {% endfor %}
    </table>

    {% if baseline_result %}
    <h2>Comparison with Baseline</h2>
    <table>
        <tr>
            <th>Field</th>
            <th>Baseline F1</th>
            <th>Model F1</th>
            <th>Improvement</th>
            <th>Relative</th>
        </tr>
        {% for field, comp in comparison.items() %}
        <tr>
            <td>{{ field }}</td>
            <td>{{ "%.4f"|format(comp.baseline_f1) }}</td>
            <td>{{ "%.4f"|format(comp.model_f1) }}</td>
            <td class="{{ 'improved' if comp.improvement > 0 else 'regressed' }}">
                {{ "%.4f"|format(comp.improvement) }}
            </td>
            <td class="{{ 'improved' if comp.relative_improvement > 0 else 'regressed' }}">
                {{ "%.2f"|format(comp.relative_improvement * 100) }}%
            </td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}

    <h2>Error Analysis</h2>
    <table>
        <tr><th>Error Type</th><th>Count</th></tr>
        {% for error_type, count in error_analysis.items() %}
        <tr><td>{{ error_type }}</td><td>{{ count }}</td></tr>
        {% endfor %}
    </table>
</body>
</html>
    """

    template = jinja2.Template(template_str)

    comparison = None
    if baseline_result:
        comparison = compare_models(baseline_result, result)

    html = template.render(
        model_name=model_name,
        overall=result.overall_metrics,
        field_metrics=result.field_metrics,
        error_analysis=result.error_analysis,
        baseline_result=baseline_result,
        comparison=comparison,
    )

    Path(output_path).write_text(html)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate extraction results")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth JSON")
    parser.add_argument("--output", type=str, default="evaluation_report.html")
    parser.add_argument("--json-output", type=str, default=None, help="Path to save JSON metrics")
    parser.add_argument("--model-name", type=str, default="model")
    parser.add_argument("--baseline", type=str, help="Path to baseline results JSON")
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)
    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    result = evaluate_extraction(predictions, ground_truth)

    baseline_result = None
    if args.baseline:
        with open(args.baseline) as f:
            baseline_data = json.load(f)
        baseline_result = EvaluationResult(**baseline_data)

    generate_evaluation_report(result, args.output, args.model_name, baseline_result)

    # Save JSON result
    json_path = args.json_output
    if not json_path:
        out_p = Path(args.output)
        json_path = str(out_p.with_name(f"{args.model_name}_results.json"))

    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Metrics JSON saved to {json_path}")

    # Print summary
    print(f"\nOverall F1: {result.overall_metrics.f1:.4f}")
    print(f"Exact Match: {result.overall_metrics.exact_match_accuracy:.4f}")
    print("\nPer-field F1:")
    for field, metrics in result.field_metrics.items():
        print(f"  {field}: {metrics.f1:.4f}")
