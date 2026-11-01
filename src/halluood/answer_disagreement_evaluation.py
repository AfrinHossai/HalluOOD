"""Paired error-detection metrics for sampled answer disagreement."""

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
    "answer_disagreement",
)
SCORE_LABELS = {
    "predictive_entropy": "Predictive entropy",
    "cosine_distance": "Cosine distance",
    "nearest_neighbor_distance": "Nearest-neighbor distance",
    "augmentation_consistency": "Augmentation consistency",
    "answer_disagreement": "Sampled answer disagreement",
}


class AnswerDisagreementEvaluationError(ValueError):
    """Raised when score rows cannot be compared with the fixed labels."""


def compare_error_detection_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare five uncertainty scores on the same baseline error labels."""

    if len(rows) < 2:
        raise AnswerDisagreementEvaluationError("at least two paired rows are required")
    try:
        raw_labels = [row["is_error"] for row in rows]
        if any(not isinstance(label, (bool, np.bool_)) for label in raw_labels):
            raise AnswerDisagreementEvaluationError("is_error labels must be boolean")
        labels = np.asarray(raw_labels, dtype=np.int8)
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
            "answer_disagreement": np.asarray(
                [float(row["answer_disagreement_score"]) for row in rows],
                dtype=np.float64,
            ),
        }
    except AnswerDisagreementEvaluationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AnswerDisagreementEvaluationError(
            "paired rows lack valid labels or numeric scores"
        ) from exc
    if len(set(labels.tolist())) != 2:
        raise AnswerDisagreementEvaluationError(
            "comparison requires both correct and incorrect/ambiguous examples"
        )
    if not np.isfinite(np.concatenate(list(score_vectors.values()))).all():
        raise AnswerDisagreementEvaluationError("all uncertainty scores must be finite")
    for field in (
        "predictive_entropy",
        "augmentation_consistency",
        "answer_disagreement",
    ):
        if np.any(score_vectors[field] < 0) or np.any(score_vectors[field] > 1):
            raise AnswerDisagreementEvaluationError(f"{field} must be in [0, 1]")

    incorrect = labels == 1
    table = []
    curves = {}
    for field in SCORE_FIELDS:
        scores = score_vectors[field]
        table.append(
            {
                "score": field,
                "score_label": SCORE_LABELS[field],
                "question_count": int(labels.size),
                "incorrect_or_ambiguous_count": int(np.sum(incorrect)),
                "error_prevalence": float(np.mean(incorrect)),
                "auroc": binary_auroc(labels.tolist(), scores.tolist()),
                "auprc_average_precision": binary_auprc(labels.tolist(), scores.tolist()),
                "mean_score_correct": float(np.mean(scores[~incorrect])),
                "median_score_correct": float(np.median(scores[~incorrect])),
                "mean_score_incorrect_or_ambiguous": float(np.mean(scores[incorrect])),
                "median_score_incorrect_or_ambiguous": float(np.median(scores[incorrect])),
            }
        )
        curves[field] = risk_coverage_curve(labels.tolist(), scores.tolist())
    return {
        "question_count": int(labels.size),
        "correct_count": int(np.sum(~incorrect)),
        "incorrect_or_ambiguous_count": int(np.sum(incorrect)),
        "error_prevalence": float(np.mean(incorrect)),
        "positive_class": "incorrect or ambiguous original deterministic baseline answer",
        "auprc_definition": "non-interpolated average precision",
        "risk_coverage_definition": (
            "error rate among retained questions after excluding highest-score tie groups"
        ),
        "comparison_table": table,
        "risk_coverage_curves": curves,
    }


def render_answer_disagreement_risk_coverage_svg(
    curves: Mapping[str, Sequence[Mapping[str, float | int | None]]],
    *,
    overall_error_rate: float,
    title: str = "POPE baseline error risk–coverage by uncertainty score (n=30)",
) -> str:
    """Render the five paired risk–coverage curves as a standalone SVG."""

    width, height = 1080, 590
    left, top, plot_width, plot_height = 92, 78, 720, 330
    right, bottom = left + plot_width, top + plot_height
    colors = {
        "predictive_entropy": "#1769aa",
        "cosine_distance": "#d05a00",
        "nearest_neighbor_distance": "#26834a",
        "augmentation_consistency": "#8e44ad",
        "answer_disagreement": "#c2185b",
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
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="18" '
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
            'font-size="11">all-answer error rate</text>',
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
            f'<circle cx="{x_pos(x):.1f}" cy="{y_pos(y):.1f}" r="3.1" fill="{color}"/>'
            for x, y in points
        )

    for index, name in enumerate(curves):
        column = index % 2
        row = index // 2
        x = left + column * 370
        y = bottom + 48 + row * 26
        color = colors.get(name, "#202124")
        label = SCORE_LABELS.get(name, name.replace("_", " "))
        elements.extend(
            [
                f'<line x1="{x}" y1="{y}" x2="{x + 26}" y2="{y}" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text x="{x + 34}" y="{y + 4}" font-size="12">{escape(label)}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{(left + right) / 2:.1f}" y="{height - 20}" text-anchor="middle" '
            'font-size="14">Coverage retained (highest-risk tie groups excluded first)</text>',
            f'<text x="22" y="{(top + bottom) / 2:.1f}" text-anchor="middle" '
            'font-size="14" transform="rotate(-90 22 '
            f'{(top + bottom) / 2:.1f})">Error rate among retained answers</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"
