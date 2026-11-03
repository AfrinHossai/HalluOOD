"""Paired reliability metrics and plots for controlled image shifts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from halluood.evaluation import binary_auprc, binary_auroc
from halluood.shifts import SHIFT_PARAMETERS


class DistributionShiftEvaluationError(ValueError):
    """Raised when clean and shifted evaluation rows cannot be compared."""


def _identity(row: Mapping[str, Any], index: int) -> tuple[str, str, str]:
    try:
        category = str(row["category"])
        question_id = str(row["question_id"])
        image_id = str(row["image_id"])
    except KeyError as exc:
        raise DistributionShiftEvaluationError(
            f"row {index} is missing identity field {exc.args[0]!r}"
        ) from exc
    if not category or not question_id or not image_id:
        raise DistributionShiftEvaluationError(f"row {index} has an empty identity field")
    return category, question_id, image_id


def _entropy(row: Mapping[str, Any], index: int) -> float | None:
    value = row.get("normalized_entropy")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DistributionShiftEvaluationError(f"row {index} normalized_entropy must be numeric")
    score = float(value)
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise DistributionShiftEvaluationError(
            f"row {index} normalized_entropy must be finite and in [0, 1]"
        )
    return score


def _error_label(row: Mapping[str, Any], index: int) -> bool:
    error = row.get("is_error")
    correct = row.get("is_correct")
    if error is not None and not isinstance(error, bool):
        raise DistributionShiftEvaluationError(f"row {index} is_error must be boolean")
    if correct is not None and not isinstance(correct, bool):
        raise DistributionShiftEvaluationError(f"row {index} is_correct must be boolean")
    if error is None and correct is None:
        raise DistributionShiftEvaluationError(f"row {index} requires is_error or is_correct")
    if error is not None and correct is not None and error == correct:
        raise DistributionShiftEvaluationError(f"row {index} has inconsistent error labels")
    return error if error is not None else not correct


def _condition_metrics(
    condition_id: str,
    perturbation_type: str,
    severity: int,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    errors = [_error_label(row, index) for index, row in enumerate(rows)]
    scored: list[tuple[bool, float]] = []
    for index, (error, row) in enumerate(zip(errors, rows, strict=True)):
        entropy = _entropy(row, index)
        if entropy is not None:
            scored.append((error, entropy))
    error_labels = [error for error, _ in scored]
    entropy_scores = [entropy for _, entropy in scored]
    positive_count = sum(error_labels)
    negative_count = len(error_labels) - positive_count
    auroc = (
        binary_auroc([int(value) for value in error_labels], entropy_scores)
        if positive_count and negative_count
        else None
    )
    auprc = (
        binary_auprc([int(value) for value in error_labels], entropy_scores)
        if positive_count
        else None
    )
    total = len(rows)
    return {
        "condition_id": condition_id,
        "perturbation_type": perturbation_type,
        "severity": severity,
        "question_count": total,
        "prediction_count": total,
        "correct_count": total - int(sum(errors)),
        "error_count": int(sum(errors)),
        "error_prevalence": float(np.mean(errors)) if errors else None,
        "accuracy": 1.0 - float(np.mean(errors)) if errors else None,
        "entropy_count": len(entropy_scores),
        "mean_normalized_entropy": float(np.mean(entropy_scores)) if entropy_scores else None,
        "auroc": auroc,
        "auprc": auprc,
        "auroc_error": (
            None
            if positive_count and negative_count
            else "requires both correct and incorrect scored examples"
        ),
        "auprc_error": None if positive_count else "requires at least one incorrect scored example",
        "mean_generation_time_seconds": _mean_optional(rows, "inference_time_seconds"),
        "mean_entropy_scoring_time_seconds": _mean_optional(rows, "uncertainty_time_seconds"),
    }


def _mean_optional(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise DistributionShiftEvaluationError(f"{key} values must be finite numbers")
            values.append(float(value))
    return float(np.mean(values)) if values else None


def summarize_distribution_shift(
    clean_rows: Sequence[Mapping[str, Any]],
    shifted_rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = 30,
) -> dict[str, Any]:
    """Compare clean and transformed results on identical image-question pairs."""

    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise DistributionShiftEvaluationError("expected_count must be a positive integer")
    if len(clean_rows) != expected_count:
        raise DistributionShiftEvaluationError(
            f"expected {expected_count} clean rows, received {len(clean_rows)}"
        )
    clean_by_id: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(clean_rows):
        key = _identity(row, index)
        if key in clean_by_id:
            raise DistributionShiftEvaluationError(f"duplicate clean row identity: {key}")
        clean_by_id[key] = row
        _error_label(row, index)
        _entropy(row, index)

    clean_keys = set(clean_by_id)
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    group_keys: dict[tuple[str, int], set[tuple[str, str, str]]] = {}
    seen: set[tuple[str, str, str, str, int]] = set()
    for index, row in enumerate(shifted_rows):
        key = _identity(row, index)
        perturbation_type = row.get("perturbation_type")
        severity = row.get("severity")
        if perturbation_type not in SHIFT_PARAMETERS:
            raise DistributionShiftEvaluationError(
                f"row {index} has an unsupported perturbation_type"
            )
        if isinstance(severity, bool) or not isinstance(severity, int) or severity not in (1, 2, 3):
            raise DistributionShiftEvaluationError(f"row {index} has invalid severity")
        condition = (perturbation_type, severity)
        row_key = (*key, perturbation_type, severity)
        if row_key in seen:
            raise DistributionShiftEvaluationError(f"duplicate shifted row identity: {row_key}")
        seen.add(row_key)
        if key not in clean_by_id:
            raise DistributionShiftEvaluationError(
                f"shifted row {index} is not in the clean evaluation sample: {key}"
            )
        clean = clean_by_id[key]
        for field in ("question", "normalized_ground_truth_label", "ground_truth_label"):
            if row.get(field) != clean.get(field):
                raise DistributionShiftEvaluationError(
                    f"shifted row {index} changed paired clean field {field!r}"
                )
        groups.setdefault(condition, []).append(row)
        group_keys.setdefault(condition, set()).add(key)
        _error_label(row, index)
        _entropy(row, index)

    condition_metrics = [_condition_metrics("clean", "none", 0, clean_rows)]
    clean_accuracy = condition_metrics[0]["accuracy"]
    clean_entropy = condition_metrics[0]["mean_normalized_entropy"]
    for perturbation_type in SHIFT_PARAMETERS:
        for severity in (1, 2, 3):
            condition = (perturbation_type, severity)
            rows = groups.get(condition)
            if rows is None:
                raise DistributionShiftEvaluationError(
                    f"missing shifted condition {perturbation_type} severity {severity}"
                )
            if group_keys[condition] != clean_keys or len(rows) != expected_count:
                missing = sorted(clean_keys - group_keys[condition])[:3]
                unexpected = sorted(group_keys[condition] - clean_keys)[:3]
                raise DistributionShiftEvaluationError(
                    f"condition {perturbation_type} severity {severity} is not paired; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            metrics = _condition_metrics(
                f"{perturbation_type}_severity_{severity}",
                perturbation_type,
                severity,
                rows,
            )
            metrics["delta_accuracy_vs_clean"] = metrics["accuracy"] - clean_accuracy
            entropy_mean = metrics["mean_normalized_entropy"]
            metrics["delta_mean_entropy_vs_clean"] = (
                entropy_mean - clean_entropy
                if entropy_mean is not None and clean_entropy is not None
                else None
            )
            condition_metrics.append(metrics)

    expected_row_count = expected_count * len(SHIFT_PARAMETERS) * 3
    if len(shifted_rows) != expected_row_count:
        raise DistributionShiftEvaluationError(
            f"expected {expected_row_count} shifted rows across all conditions, "
            f"received {len(shifted_rows)}"
        )
    return {
        "clean_question_count": len(clean_rows),
        "shifted_question_count": len(shifted_rows),
        "paired_image_question_count": len(clean_keys),
        "condition_count": len(condition_metrics) - 1,
        "conditions": condition_metrics,
        "clean_condition": "original unmodified images paired by question, image, and category",
        "error_target": "incorrect, ambiguous, or failed yes/no answer",
        "score_definition": (
            "Normalized entropy of conditional Yes/No candidate-sequence likelihoods; "
            "higher scores are treated as higher risk of answer error."
        ),
    }


def render_severity_plot(
    condition_metrics: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    title: str,
    y_label: str,
) -> str:
    """Render a dependency-free SVG line plot against predefined severity."""

    width, height = 800, 520
    left, top, plot_width, plot_height = 82, 64, 590, 350
    right, bottom = left + plot_width, top + plot_height
    colors = {
        "gaussian_blur": "#1769aa",
        "brightness_reduction": "#b3261e",
        "gaussian_noise": "#188038",
        "contrast_reduction": "#9334e6",
    }
    clean = next((row for row in condition_metrics if row.get("condition_id") == "clean"), None)
    if clean is None:
        raise DistributionShiftEvaluationError("condition metrics must include a clean baseline")

    def value_y(value: float) -> float:
        return top + (1.0 - min(1.0, max(0.0, value))) * plot_height

    def value_x(severity: int) -> float:
        return left + (severity / 3.0) * plot_width

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#202124}.grid{stroke:#dadce0;"
        "stroke-width:1}.axis{stroke:#5f6368;stroke-width:1.5}.series{fill:none;stroke-width:3}"
        ".point{stroke:white;stroke-width:1.5}</style>",
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="18" '
        f'font-weight="bold">{escape(title)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x = left + fraction * plot_width
        y = value_y(fraction)
        elements.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>',
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>',
                f'<text x="{x:.1f}" y="{bottom + 23}" text-anchor="middle" font-size="12">'
                f"{fraction:.1f}</text>",
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12">'
                f"{fraction:.1f}</text>",
            ]
        )
    elements.extend(
        [
            f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
        ]
    )

    legend_y = top + 24
    for kind_index, perturbation_type in enumerate(SHIFT_PARAMETERS):
        rows = [
            row for row in condition_metrics if row.get("perturbation_type") == perturbation_type
        ]
        rows_by_severity = {int(row["severity"]): row for row in rows}
        points = [(0, clean.get(metric))] + [
            (severity, rows_by_severity.get(severity, {}).get(metric)) for severity in (1, 2, 3)
        ]
        valid_points = [(severity, value) for severity, value in points if value is not None]
        if valid_points:
            coordinates = " ".join(
                f"{value_x(severity):.1f},{value_y(float(value)):.1f}"
                for severity, value in valid_points
            )
            elements.append(
                f'<polyline class="series" stroke="{colors[perturbation_type]}" '
                f'points="{coordinates}"/>'
            )
            for severity, value in valid_points:
                elements.append(
                    f'<circle class="point" cx="{value_x(severity):.1f}" '
                    f'cy="{value_y(float(value)):.1f}" r="5" '
                    f'fill="{colors[perturbation_type]}"/>'
                )
        legend_x = left + kind_index * 150
        elements.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" '
                f'y2="{legend_y}" stroke="{colors[perturbation_type]}" stroke-width="3"/>',
                f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-size="11">'
                f"{escape(perturbation_type.replace('_', ' '))}</text>",
            ]
        )
    elements.extend(
        [
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 40}" '
            'text-anchor="middle" font-size="14">'
            "Severity (0 = clean; 1–3 = increasing shift strength)</text>",
            f'<text x="24" y="{top + plot_height / 2:.1f}" text-anchor="middle" '
            f'font-size="14" transform="rotate(-90 24 {top + plot_height / 2:.1f})">'
            f"{escape(y_label)}</text>",
            "</svg>",
        ]
    )
    return "\n".join(elements)
