"""Metrics and plotting for paired POPE augmentation-consistency evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from halluood.evaluation import binary_auprc, binary_auroc, risk_coverage_curve

SCORE_FIELDS = (
    "predictive_entropy",
    "cosine_distance",
    "nearest_neighbor_distance",
    "augmentation_consistency",
)
SCORE_LABELS = {
    "predictive_entropy": "Predictive entropy",
    "cosine_distance": "Cosine distance",
    "nearest_neighbor_distance": "Nearest-neighbor distance",
    "augmentation_consistency": "Augmentation consistency",
}


class AugmentationEvaluationError(ValueError):
    """Raised when augmented scores cannot be compared safely."""


def compare_augmentation_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare entropy, image distances, and view disagreement on paired labels."""

    if len(rows) < 2:
        raise AugmentationEvaluationError("at least two paired rows are required")
    try:
        labels = np.asarray([int(bool(row["is_error"])) for row in rows], dtype=np.int8)
        score_vectors = {
            "predictive_entropy": np.asarray(
                [float(row["normalized_entropy"]) for row in rows], dtype=np.float64
            ),
            "cosine_distance": np.asarray(
                [float(row["cosine_distance"]) for row in rows], dtype=np.float64
            ),
            "nearest_neighbor_distance": np.asarray(
                [float(row["nearest_neighbor_distance"]) for row in rows], dtype=np.float64
            ),
            "augmentation_consistency": np.asarray(
                [float(row["augmentation_consistency_score"]) for row in rows],
                dtype=np.float64,
            ),
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AugmentationEvaluationError("paired rows lack valid labels or scores") from exc
    if not np.isin(labels, (0, 1)).all():
        raise AugmentationEvaluationError("error labels must be boolean or binary")
    if len(set(labels.tolist())) != 2:
        raise AugmentationEvaluationError(
            "comparison requires both correct and incorrect/ambiguous predictions"
        )
    if not np.isfinite(np.concatenate(list(score_vectors.values()))).all():
        raise AugmentationEvaluationError("all comparison scores must be finite")
    if np.any(score_vectors["predictive_entropy"] < 0) or np.any(
        score_vectors["predictive_entropy"] > 1
    ):
        raise AugmentationEvaluationError("normalized predictive entropy must be in [0, 1]")
    if np.any(score_vectors["augmentation_consistency"] < 0) or np.any(
        score_vectors["augmentation_consistency"] > 1
    ):
        raise AugmentationEvaluationError("augmentation-consistency scores must be in [0, 1]")

    incorrect = labels == 1
    table = []
    curves = {}
    for name in SCORE_FIELDS:
        scores = score_vectors[name]
        table.append(
            {
                "score": name,
                "score_label": SCORE_LABELS[name],
                "question_count": int(labels.size),
                "incorrect_count": int(np.sum(incorrect)),
                "incorrect_prevalence": float(np.mean(incorrect)),
                "auroc": binary_auroc(labels.tolist(), scores.tolist()),
                "auprc_average_precision": binary_auprc(labels.tolist(), scores.tolist()),
                "score_mean_correct": float(np.mean(scores[~incorrect])),
                "score_median_correct": float(np.median(scores[~incorrect])),
                "score_mean_incorrect": float(np.mean(scores[incorrect])),
                "score_median_incorrect": float(np.median(scores[incorrect])),
            }
        )
        curves[name] = risk_coverage_curve(labels.tolist(), scores.tolist())

    return {
        "question_count": int(labels.size),
        "correct_count": int(np.sum(~incorrect)),
        "incorrect_or_ambiguous_count": int(np.sum(incorrect)),
        "incorrect_prevalence": float(np.mean(incorrect)),
        "score_comparison_table": table,
        "risk_coverage_curves": curves,
        "positive_class": "incorrect or ambiguous original POPE prediction",
        "auprc_definition": "non-interpolated average precision",
        "risk_coverage_definition": (
            "error rate among retained rows after excluding highest-score tie groups first"
        ),
    }


def render_augmentation_risk_coverage_svg(
    curves: Mapping[str, Sequence[Mapping[str, float | int | None]]],
    *,
    overall_error_rate: float,
    title: str = "POPE error risk–coverage by uncertainty score (n=30)",
) -> str:
    """Render the four paired risk–coverage curves as a standalone SVG."""

    width, height = 1020, 555
    left, top, plot_width, plot_height = 92, 58, 830, 340
    right, bottom = left + plot_width, top + plot_height
    colors = {
        "predictive_entropy": "#1769aa",
        "cosine_distance": "#d05a00",
        "nearest_neighbor_distance": "#26834a",
        "augmentation_consistency": "#8e44ad",
    }

    def x_pos(value: float) -> float:
        return left + max(0.0, min(1.0, value)) * plot_width

    def y_pos(value: float) -> float:
        return top + (1.0 - max(0.0, min(1.0, value))) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.grid{stroke:#dfe1e5;'
        'stroke-width:1}.axis{stroke:#5f6368;stroke-width:1.5}.baseline{stroke:#777;'
        'stroke-width:2;stroke-dasharray:7 5}</style>',
        f'<text x="{width / 2:.1f}" y="27" text-anchor="middle" font-size="18" '
        f'font-weight="bold">{escape(title)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x, y = x_pos(fraction), y_pos(fraction)
        elements.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>',
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>',
                f'<text x="{x:.1f}" y="{bottom + 22}" text-anchor="middle" font-size="12">'
                f"{fraction:.1f}</text>",
                f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12">'
                f"{fraction:.1f}</text>",
            ]
        )
    elements.extend(
        [
            f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
            f'<line class="baseline" x1="{left}" y1="{y_pos(overall_error_rate):.1f}" '
            f'x2="{right}" y2="{y_pos(overall_error_rate):.1f}"/>',
            f'<text x="{left + 8}" y="{y_pos(overall_error_rate) - 6:.1f}" '
            'font-size="11">all-example error rate</text>',
        ]
    )
    for name, curve in curves.items():
        points = [
            (float(point["coverage"]), float(point["risk"]))
            for point in curve
            if point.get("coverage") is not None and point.get("risk") is not None
        ]
        if not points:
            continue
        path = " ".join(
            ("M" if index == 0 else "L") + f" {x_pos(x):.1f} {y_pos(y):.1f}"
            for index, (x, y) in enumerate(points)
        )
        color = colors.get(name, "#202124")
        elements.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        elements.extend(
            f'<circle cx="{x_pos(x):.1f}" cy="{y_pos(y):.1f}" r="3.2" fill="{color}"/>'
            for x, y in points
        )

    for index, name in enumerate(curves):
        column = index % 2
        row = index // 2
        x = left + column * 430
        y = bottom + 48 + row * 25
        color = colors.get(name, "#202124")
        elements.extend(
            [
                f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text x="{x + 36}" y="{y + 4}" font-size="12">'
                f"{escape(SCORE_LABELS.get(name, name.replace('_', ' ')))}</text>",
            ]
        )
    elements.extend(
        [
            f'<text x="{(left + right) / 2:.1f}" y="{height - 20}" text-anchor="middle" '
            'font-size="14">Coverage retained (highest-risk cases excluded first)</text>',
            f'<text x="22" y="{(top + bottom) / 2:.1f}" text-anchor="middle" '
            'font-size="14" transform="rotate(-90 22 '
            f'{(top + bottom) / 2:.1f})">Error rate among retained answers</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"
