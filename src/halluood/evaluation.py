"""Small-sample metrics for uncertainty-based error detection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from xml.sax.saxutils import escape

import numpy as np


class EvaluationError(ValueError):
    """Raised when evaluation rows or metric inputs are invalid."""


def _validated_labels_and_scores(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    label_array = np.asarray(labels)
    try:
        score_array = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationError("risk scores must be numeric") from exc
    if label_array.ndim != 1 or score_array.ndim != 1:
        raise EvaluationError("labels and scores must be one-dimensional")
    if label_array.size == 0 or label_array.size != score_array.size:
        raise EvaluationError("labels and scores must have the same non-zero length")
    if not np.isin(label_array, (0, 1)).all():
        raise EvaluationError("labels must contain only 0 (correct) or 1 (incorrect)")
    if not np.isfinite(score_array).all():
        raise EvaluationError("risk scores must be finite")
    return label_array.astype(np.int8), score_array


def binary_auroc(labels: Sequence[int], risk_scores: Sequence[float]) -> float:
    """Compute AUROC, treating larger scores as stronger evidence of error."""

    label_array, score_array = _validated_labels_and_scores(labels, risk_scores)
    positive = score_array[label_array == 1]
    negative = score_array[label_array == 0]
    if positive.size == 0 or negative.size == 0:
        raise EvaluationError("AUROC requires at least one correct and one incorrect example")
    wins = sum(
        float(positive_score > negative_score)
        + 0.5 * float(positive_score == negative_score)
        for positive_score in positive
        for negative_score in negative
    )
    return float(wins / (positive.size * negative.size))


def binary_auprc(labels: Sequence[int], risk_scores: Sequence[float]) -> float:
    """Compute non-interpolated average precision for the incorrect class."""

    label_array, score_array = _validated_labels_and_scores(labels, risk_scores)
    positive_count = int(np.sum(label_array == 1))
    if positive_count == 0:
        raise EvaluationError("AUPRC requires at least one incorrect example")

    order = np.argsort(-score_array, kind="mergesort")
    sorted_scores = score_array[order]
    sorted_labels = label_array[order]
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    index = 0

    while index < sorted_scores.size:
        stop = index + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[index]:
            stop += 1
        group_labels = sorted_labels[index:stop]
        true_positives += int(np.sum(group_labels == 1))
        false_positives += int(np.sum(group_labels == 0))
        recall = true_positives / positive_count
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        index = stop

    return float(average_precision)


def risk_coverage_curve(
    error_labels: Sequence[int], risk_scores: Sequence[float]
) -> list[dict[str, float | int | None]]:
    """Return selective error risk as high-score groups are excluded.

    Thresholds are applied to complete tied-score groups: rows with a score
    greater than the threshold are excluded, so the curve never depends on an
    arbitrary ordering within ties. Risk is undefined when coverage is zero.
    """

    label_array, score_array = _validated_labels_and_scores(error_labels, risk_scores)
    total = int(label_array.size)
    points: list[dict[str, float | int | None]] = [
        {
            "coverage": 0.0,
            "risk": None,
            "retained_count": 0,
            "excluded_count": total,
            "max_retained_entropy": None,
        }
    ]
    for threshold in np.unique(score_array):
        retained = score_array <= threshold
        retained_count = int(np.sum(retained))
        points.append(
            {
                "coverage": retained_count / total,
                "risk": float(np.mean(label_array[retained])),
                "retained_count": retained_count,
                "excluded_count": total - retained_count,
                "max_retained_entropy": float(threshold),
            }
        )
    return points


def _score_statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def evaluate_entropy_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach correctness and summarize binary entropy as an answer-error risk.

    Ambiguous and failed predictions count as incorrect, matching the existing
    baseline's all-question accuracy convention. Inputs must already include a
    `uncertainty_score.normalized_entropy` value from the binary Yes/No scorer.
    """

    if not rows:
        raise EvaluationError("at least one evaluated row is required")

    evaluated: list[dict[str, Any]] = []
    errors: list[int] = []
    entropies: list[float] = []
    categories: dict[str, list[int]] = {}
    category_scores: dict[str, list[float]] = {}

    for row in rows:
        category = row.get("category")
        if not isinstance(category, str) or not category.strip():
            raise EvaluationError("every row must have a non-empty question category")
        uncertainty = row.get("uncertainty_score")
        if not isinstance(uncertainty, Mapping):
            raise EvaluationError("every row must include an uncertainty_score object")
        try:
            entropy = float(uncertainty["normalized_entropy"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise EvaluationError("every uncertainty score needs normalized_entropy") from exc
        if not np.isfinite(entropy) or not 0.0 <= entropy <= 1.0:
            raise EvaluationError("normalized entropy must be finite and between 0 and 1")

        gold = row.get("normalized_ground_truth_label")
        prediction = row.get("normalized_prediction")
        status = row.get("prediction_status")
        if gold not in {"yes", "no"}:
            raise EvaluationError("every row must have a normalized yes/no ground-truth label")
        is_correct = status == "parsed" and prediction == gold
        is_error = not is_correct
        enriched = dict(row)
        enriched["normalized_entropy"] = entropy
        enriched["is_correct"] = is_correct
        enriched["is_error"] = is_error
        evaluated.append(enriched)
        entropies.append(entropy)
        errors.append(int(is_error))
        categories.setdefault(category, []).append(int(is_error))
        category_scores.setdefault(category, []).append(entropy)

    correct_scores = [score for score, error in zip(entropies, errors, strict=True) if not error]
    incorrect_scores = [score for score, error in zip(entropies, errors, strict=True) if error]
    count = len(evaluated)
    error_count = int(sum(errors))
    by_category: dict[str, Any] = {}
    for category in sorted(categories):
        category_errors = categories[category]
        category_entropy = category_scores[category]
        correct_entropy = [
            score
            for score, error in zip(category_entropy, category_errors, strict=True)
            if not error
        ]
        incorrect_entropy = [
            score
            for score, error in zip(category_entropy, category_errors, strict=True)
            if error
        ]
        by_category[category] = {
            "question_count": len(category_errors),
            "incorrect_count": int(sum(category_errors)),
            "error_rate": float(np.mean(category_errors)),
            "entropy_correct": _score_statistics(correct_entropy),
            "entropy_incorrect": _score_statistics(incorrect_entropy),
        }

    summary = {
        "question_count": count,
        "correct_count": count - error_count,
        "incorrect_count": error_count,
        "error_rate": error_count / count,
        "incorrect_prevalence": error_count / count,
        "auroc": binary_auroc(errors, entropies),
        "auprc_average_precision": binary_auprc(errors, entropies),
        "entropy_correct": _score_statistics(correct_scores),
        "entropy_incorrect": _score_statistics(incorrect_scores),
        "entropy_all": _score_statistics(entropies),
        "by_category": by_category,
        "risk_coverage_curve": risk_coverage_curve(errors, entropies),
        "risk_score_direction": "higher_normalized_entropy_means_higher_error_risk",
    }
    return evaluated, summary


def render_risk_coverage_svg(
    curve: Sequence[Mapping[str, float | int | None]],
    *,
    overall_error_rate: float,
    title: str = "POPE answer error risk vs. coverage",
) -> str:
    """Render a small dependency-free SVG risk–coverage chart."""

    width, height = 760, 500
    left, top, plot_width, plot_height = 92, 58, 610, 330
    right, bottom = left + plot_width, top + plot_height

    def x_position(coverage: float) -> float:
        return left + max(0.0, min(1.0, coverage)) * plot_width

    def y_position(risk: float) -> float:
        return top + (1.0 - max(0.0, min(1.0, risk))) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.grid{stroke:#dfe1e5;'
        'stroke-width:1}.axis{stroke:#5f6368;stroke-width:1.5}.curve{fill:none;stroke:#1769aa;'
        'stroke-width:3}.point{fill:#1769aa}.baseline{stroke:#b3261e;stroke-width:2;'
        'stroke-dasharray:7 5}</style>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="18" '
        f'font-weight="bold">{escape(title)}</text>',
    ]
    for tick in range(0, 6):
        fraction = tick / 5
        x = x_position(fraction)
        y = y_position(fraction)
        elements.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>',
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>',
                f'<text x="{x:.1f}" y="{bottom + 22}" text-anchor="middle" font-size="12">'
                f'{fraction:.1f}</text>',
                f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12">'
                f'{fraction:.1f}</text>',
            ]
        )
    elements.extend(
        [
            f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
            f'<line class="baseline" x1="{left}" y1="{y_position(overall_error_rate):.1f}" '
            f'x2="{right}" y2="{y_position(overall_error_rate):.1f}"/>',
            f'<text x="{left + 10}" y="{y_position(overall_error_rate) - 6:.1f}" '
            'font-size="11" fill="#b3261e">all-example error rate</text>',
        ]
    )
    points = [
        (float(point["coverage"]), float(point["risk"]))
        for point in curve
        if point.get("coverage") is not None and point.get("risk") is not None
    ]
    if points:
        path = " ".join(
            ("M" if index == 0 else "L") + f" {x_position(x):.1f} {y_position(y):.1f}"
            for index, (x, y) in enumerate(points)
        )
        elements.append(f'<path class="curve" d="{path}"/>')
        elements.extend(
            f'<circle class="point" cx="{x_position(x):.1f}" cy="{y_position(y):.1f}" r="4"/>'
            for x, y in points
        )
    elements.extend(
        [
            f'<text x="{(left + right) / 2:.1f}" y="{height - 48}" text-anchor="middle" '
            'font-size="14">Coverage retained (high-entropy cases excluded first)</text>',
            f'<text x="22" y="{(top + bottom) / 2:.1f}" text-anchor="middle" '
            'font-size="14" transform="rotate(-90 22 '
            f'{(top + bottom) / 2:.1f})">Error rate among retained answers</text>',
            '</svg>',
        ]
    )
    return "\n".join(elements) + "\n"
