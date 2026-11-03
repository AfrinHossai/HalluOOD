"""Expanded POPE summaries, image-grouped bootstrap intervals, and splits."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

import numpy as np

from halluood.evaluation import render_risk_coverage_svg as _render_risk_coverage_svg
from halluood.evaluation import risk_coverage_curve

CATEGORIES = ("random", "popular", "adversarial")


class RobustnessEvaluationError(ValueError):
    """Raised when the expanded evaluation data is inconsistent."""


def _identity(row: Mapping[str, Any], index: int) -> tuple[str, str, str]:
    try:
        return (str(row["category"]), str(row["question_id"]), str(row["image_id"]))
    except KeyError as exc:
        raise RobustnessEvaluationError(
            f"row {index} is missing identity field {exc.args[0]!r}"
        ) from exc


def _labels_and_score(row: Mapping[str, Any], index: int) -> tuple[bool, bool, float | None]:
    correct = row.get("is_correct")
    ambiguous = row.get("prediction_status") == "ambiguous"
    if not isinstance(correct, bool):
        raise RobustnessEvaluationError(f"row {index} is_correct must be boolean")
    if row.get("is_error") is not None and row["is_error"] is not (not correct):
        raise RobustnessEvaluationError(f"row {index} has inconsistent error labels")
    score = row.get("normalized_entropy")
    if score is None:
        uncertainty = row.get("uncertainty_score")
        if isinstance(uncertainty, Mapping):
            score = uncertainty.get("normalized_entropy")
    if score is None:
        return correct, ambiguous, None
    if isinstance(score, bool):
        raise RobustnessEvaluationError(f"row {index} normalized_entropy must be numeric")
    try:
        score = float(score)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RobustnessEvaluationError(f"row {index} normalized_entropy must be numeric") from exc
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise RobustnessEvaluationError(f"row {index} normalized_entropy must be in [0, 1]")
    return correct, ambiguous, score


def _metrics(
    rows: Sequence[Mapping[str, Any]], weights: np.ndarray | None = None
) -> dict[str, Any]:
    if not rows:
        raise RobustnessEvaluationError("metrics require at least one row")
    if weights is None:
        weights = np.ones(len(rows), dtype=np.float64)
    if weights.shape != (len(rows),) or np.any(weights < 0):
        raise RobustnessEvaluationError("metric weights must be a non-negative row vector")
    outcome_data = [_labels_and_score(row, index) for index, row in enumerate(rows)]
    correct = np.asarray([item[0] for item in outcome_data], dtype=np.float64)
    ambiguous = np.asarray([item[1] for item in outcome_data], dtype=np.float64)
    scored_mask = np.asarray([item[2] is not None for item in outcome_data], dtype=bool)
    scores = np.asarray(
        [item[2] if item[2] is not None else 0.0 for item in outcome_data], dtype=np.float64
    )
    denominator = float(weights.sum())
    auroc: float | None = None
    auprc: float | None = None
    # Weighted tie groups are required for image-cluster bootstrap draws.
    if scored_mask.any():
        labels = (1.0 - correct[scored_mask]).astype(np.int8)
        scores_scored = scores[scored_mask]
        sample_weights = weights[scored_mask]
        unique_scores, inverse = np.unique(scores_scored, return_inverse=True)
        positive = np.bincount(
            inverse, weights=sample_weights * labels, minlength=unique_scores.size
        )
        negative = np.bincount(
            inverse, weights=sample_weights * (1 - labels), minlength=unique_scores.size
        )
        nonzero_groups = (positive + negative) > 0
        positive = positive[nonzero_groups]
        negative = negative[nonzero_groups]
        positive_total = float(positive.sum())
        negative_total = float(negative.sum())
        if positive_total and negative_total:
            negatives_before = np.cumsum(negative) - negative
            auroc = float(
                np.sum(positive * (negatives_before + 0.5 * negative))
                / (positive_total * negative_total)
            )
        else:
            auroc = None
        if positive_total:
            positive_desc = positive[::-1]
            negative_desc = negative[::-1]
            true_positive = np.cumsum(positive_desc)
            false_positive = np.cumsum(negative_desc)
            precision = true_positive / (true_positive + false_positive)
            auprc = float(np.sum((positive_desc / positive_total) * precision))
        else:
            auprc = None
    return {
        "question_count": int(denominator),
        "correct_count": int(np.sum(correct * weights)),
        "accuracy": float(np.sum(correct * weights) / denominator) if denominator else None,
        "ambiguous_count": int(np.sum(ambiguous * weights)),
        "ambiguous_response_rate": float(np.sum(ambiguous * weights) / denominator)
        if denominator
        else None,
        "failed_count": int(
            np.sum(np.asarray([row.get("prediction_status") == "error" for row in rows]) * weights)
        ),
        "entropy_scored_count": int(np.sum(weights[scored_mask])),
        "mean_normalized_entropy": (
            float(np.average(scores[scored_mask], weights=weights[scored_mask]))
            if np.sum(weights[scored_mask])
            else None
        ),
        "auroc": auroc,
        "auprc_average_precision": auprc,
        "error_prevalence_among_scored": (
            float(np.average(1.0 - correct[scored_mask], weights=weights[scored_mask]))
            if np.sum(weights[scored_mask])
            else None
        ),
    }


def summarize_expanded_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize all answered rows overall and by the three POPE strategies."""

    if not rows:
        raise RobustnessEvaluationError("at least one row is required")
    seen: set[tuple[str, str, str]] = set()
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    images_by_category: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        identity = _identity(row, index)
        if identity in seen:
            raise RobustnessEvaluationError(f"duplicate question identity: {identity}")
        seen.add(identity)
        if identity[0] not in CATEGORIES:
            raise RobustnessEvaluationError(f"unsupported POPE category: {identity[0]!r}")
        by_category[identity[0]].append(row)
        images_by_category[identity[0]].add(identity[2])
        _labels_and_score(row, index)
    return {
        "overall": _metrics(rows),
        "by_category": {category: _metrics(by_category[category]) for category in CATEGORIES},
        "question_count": len(rows),
        "image_count": len({identity[2] for identity in seen}),
        "category_image_counts": {
            category: len(images_by_category[category]) for category in CATEGORIES
        },
        "error_definition": "incorrect, ambiguous, or failed yes/no answer",
        "ambiguous_rate_denominator": "all questions, including inference failures",
        "auprc_definition": "non-interpolated average precision for the incorrect class",
    }


def image_grouped_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 1000,
    seed: int = 2027,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Percentile intervals from resampling images with all their rows together."""

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise RobustnessEvaluationError("replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RobustnessEvaluationError("seed must be a non-negative integer")
    if not 0.0 < confidence_level < 1.0:
        raise RobustnessEvaluationError("confidence_level must be in (0, 1)")
    if not rows:
        raise RobustnessEvaluationError("at least one row is required")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[_identity(row, index)[2]].append(row)
    image_ids = sorted(grouped)
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    scopes = {"overall": list(rows), **{category: by_category[category] for category in CATEGORIES}}
    rng = np.random.Generator(np.random.PCG64(seed))
    metric_names = (
        "accuracy",
        "ambiguous_response_rate",
        "mean_normalized_entropy",
        "auroc",
        "auprc_average_precision",
    )
    samples: dict[str, dict[str, list[float]]] = {
        scope: {metric: [] for metric in metric_names} for scope in scopes
    }
    valid_rank_draws: dict[str, dict[str, int]] = {
        scope: {"auroc": 0, "auprc_average_precision": 0} for scope in scopes
    }
    for _ in range(replicates):
        draws = rng.choice(image_ids, size=len(image_ids), replace=True)
        multiplicity = Counter(draws.tolist())
        for scope, scope_rows in scopes.items():
            weights = np.asarray(
                [multiplicity.get(str(row["image_id"]), 0) for row in scope_rows],
                dtype=np.float64,
            )
            if not np.any(weights):
                continue
            result = _metrics(scope_rows, weights)
            for metric in metric_names:
                value = result[metric]
                if value is not None and isfinite(value):
                    samples[scope][metric].append(float(value))
                    if metric in valid_rank_draws[scope]:
                        valid_rank_draws[scope][metric] += 1
    alpha = (1.0 - confidence_level) / 2.0
    intervals: dict[str, Any] = {}
    for scope in scopes:
        intervals[scope] = {}
        for metric in metric_names:
            values = np.asarray(samples[scope][metric], dtype=np.float64)
            intervals[scope][metric] = {
                "lower": float(np.quantile(values, alpha)) if values.size else None,
                "upper": float(np.quantile(values, 1.0 - alpha)) if values.size else None,
                "valid_replicates": int(values.size),
                "requested_replicates": replicates,
            }
    return {
        "method": "image_grouped_percentile_bootstrap",
        "unit_of_resampling": "image_id; every question from each sampled image travels together",
        "seed": seed,
        "replicates": replicates,
        "confidence_level": confidence_level,
        "image_count": len(image_ids),
        "intervals": intervals,
    }


def deterministic_image_split(
    image_ids: Sequence[str], *, calibration_fraction: float = 0.2, seed: int = 2027
) -> tuple[list[str], list[str]]:
    """Return reproducible calibration/evaluation image groups."""

    unique_ids = sorted(set(image_ids))
    if len(unique_ids) < 2:
        raise RobustnessEvaluationError("at least two distinct image IDs are required")
    if not 0.0 < calibration_fraction < 1.0:
        raise RobustnessEvaluationError("calibration_fraction must be in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RobustnessEvaluationError("seed must be a non-negative integer")
    count = min(len(unique_ids) - 1, max(1, round(len(unique_ids) * calibration_fraction)))
    rng = np.random.Generator(np.random.PCG64(seed))
    shuffled = [str(value) for value in rng.permutation(np.asarray(unique_ids, dtype=object))]
    return sorted(shuffled[:count]), sorted(shuffled[count:])


def image_grouped_selective_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    *,
    replicates: int = 1000,
    seed: int = 2027,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap held-out selective metrics by image with a frozen threshold."""

    if not rows:
        raise RobustnessEvaluationError("at least one evaluation row is required")
    if not isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise RobustnessEvaluationError("threshold must be finite and in [0, 1]")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise RobustnessEvaluationError("replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RobustnessEvaluationError("seed must be a non-negative integer")
    if not 0.0 < confidence_level < 1.0:
        raise RobustnessEvaluationError("confidence_level must be in (0, 1)")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        image_id = _identity(row, index)[2]
        _, _, entropy = _labels_and_score(row, index)
        if entropy is None:
            raise RobustnessEvaluationError(
                f"row {index} requires an entropy score for selective bootstrap"
            )
        grouped[image_id].append(row)
    image_ids = sorted(grouped)
    outcomes = [_labels_and_score(row, index) for index, row in enumerate(rows)]
    answered = np.asarray(
        [float(score) <= float(threshold) for _, _, score in outcomes], dtype=bool
    )
    errors = np.asarray([not correct for correct, _, _ in outcomes], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    samples: dict[str, list[float]] = {
        "coverage": [],
        "selective_risk": [],
        "abstention_rate": [],
    }
    for _ in range(replicates):
        draws = rng.choice(image_ids, size=len(image_ids), replace=True)
        multiplicity = Counter(draws.tolist())
        weights = np.asarray(
            [multiplicity.get(str(row["image_id"]), 0) for row in rows], dtype=np.float64
        )
        total_weight = float(weights.sum())
        answered_weight = float(weights[answered].sum())
        if total_weight <= 0:
            continue
        coverage = answered_weight / total_weight
        samples["coverage"].append(coverage)
        samples["abstention_rate"].append(1.0 - coverage)
        if answered_weight > 0:
            samples["selective_risk"].append(
                float(np.sum(errors[answered] * weights[answered]) / answered_weight)
            )

    alpha = (1.0 - confidence_level) / 2.0
    intervals = {}
    for metric, values in samples.items():
        array = np.asarray(values, dtype=np.float64)
        intervals[metric] = {
            "lower": float(np.quantile(array, alpha)) if array.size else None,
            "upper": float(np.quantile(array, 1.0 - alpha)) if array.size else None,
            "valid_replicates": int(array.size),
            "requested_replicates": replicates,
        }
    return {
        "method": "image_grouped_percentile_bootstrap_with_frozen_threshold",
        "unit_of_resampling": (
            "image_id; all evaluation rows for each sampled image travel together"
        ),
        "threshold": float(threshold),
        "seed": seed,
        "replicates": replicates,
        "confidence_level": confidence_level,
        "image_count": len(image_ids),
        "intervals": intervals,
    }


def render_metric_bar_svg(
    summaries: Mapping[str, Mapping[str, Any]], *, metric: str, title: str, y_label: str
) -> str:
    """Render a compact clean-metric comparison with 95% percentile intervals."""

    from xml.sax.saxutils import escape

    width, height = 760, 500
    left, top, plot_width, plot_height = 90, 58, 610, 340
    labels = list(summaries)
    rows = [summaries[label] for label in labels]
    values = [float(row[metric]) if row.get(metric) is not None else 0.0 for row in rows]
    max_value = max([1.0, *values])
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#202124}.axis{stroke:#5f6368;stroke-width:1}</style>",
        (
            f'<text x="{width / 2}" y="30" text-anchor="middle" '
            f'font-size="18" font-weight="bold">{escape(title)}</text>'
        ),
    ]
    bar_width = min(48, plot_width / max(1, len(labels)) * 0.55)
    for tick in range(6):
        value = max_value * tick / 5
        y = top + plot_height * (1 - value / max_value)
        elements.extend(
            [
                (
                    f'<line x1="{left}" y1="{y:.1f}" '
                    f'x2="{left + plot_width}" y2="{y:.1f}" stroke="#dadce0"/>'
                ),
                (
                    f'<text x="{left - 10}" y="{y + 4:.1f}" '
                    f'text-anchor="end" font-size="11">{value:.2f}</text>'
                ),
            ]
        )
    for index, (label, row, value) in enumerate(zip(labels, rows, values, strict=True)):
        center = left + (index + 0.5) * plot_width / len(labels)
        bar_height = value / max_value * plot_height
        y = top + plot_height - bar_height
        elements.append(
            (
                f'<rect x="{center - bar_width / 2:.1f}" y="{y:.1f}" '
                f'width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#1769aa"/>'
            )
        )
        ci = row.get("confidence_interval", {}).get(metric, {})
        low, high = ci.get("lower"), ci.get("upper")
        if low is not None and high is not None:
            low_y = top + plot_height * (1 - float(low) / max_value)
            high_y = top + plot_height * (1 - float(high) / max_value)
            elements.append(
                (
                    f'<line x1="{center:.1f}" y1="{high_y:.1f}" '
                    f'x2="{center:.1f}" y2="{low_y:.1f}" '
                    'stroke="#202124" stroke-width="2"/>'
                )
            )
            elements.append(
                (
                    f'<line x1="{center - 6:.1f}" y1="{high_y:.1f}" '
                    f'x2="{center + 6:.1f}" y2="{high_y:.1f}" stroke="#202124"/>'
                )
            )
            elements.append(
                (
                    f'<line x1="{center - 6:.1f}" y1="{low_y:.1f}" '
                    f'x2="{center + 6:.1f}" y2="{low_y:.1f}" stroke="#202124"/>'
                )
            )
        elements.append(
            (
                f'<text x="{center:.1f}" y="{top + plot_height + 22}" '
                f'text-anchor="middle" font-size="11">{escape(label)}</text>'
            )
        )
    elements.extend(
        [
            (
                f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
                f'x2="{left + plot_width}" y2="{top + plot_height}"/>'
            ),
            (
                f'<text x="24" y="{top + plot_height / 2}" '
                'text-anchor="middle" font-size="14" '
                f'transform="rotate(-90 24 {top + plot_height / 2})">'
                f"{escape(y_label)}</text>"
            ),
            "</svg>",
        ]
    )
    return "\n".join(elements)


def render_risk_coverage_svg(curve: Sequence[Mapping[str, Any]], *, title: str) -> str:
    """Render a dependency-free risk/coverage curve."""
    baseline = next(
        (
            float(row["risk"])
            for row in curve
            if float(row.get("coverage", -1)) == 1.0 and row.get("risk") is not None
        ),
        0.0,
    )
    return _render_risk_coverage_svg(curve, overall_error_rate=baseline, title=title)


def entropy_risk_coverage(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Make the existing tie-aware risk-coverage curve from successful scores."""

    scored = [row for row in rows if _labels_and_score(row, 0)[2] is not None]
    if not scored:
        raise RobustnessEvaluationError("risk-coverage requires entropy-scored rows")
    labels = [int(not bool(row["is_correct"])) for row in scored]
    scores = [float(_labels_and_score(row, index)[2]) for index, row in enumerate(scored)]
    return risk_coverage_curve(labels, scores)
