"""Paired evaluation of embedding distances and predictive entropy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from halluood.embeddings import EmbeddingError, normalize_coco_image_id
from halluood.evaluation import binary_auprc, binary_auroc, risk_coverage_curve


class EmbeddingEvaluationError(ValueError):
    """Raised when paired score or prediction records do not align."""


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    try:
        question_id = str(row["question_id"])
        image_id = normalize_coco_image_id(str(row["image_id"]))
    except (KeyError, EmbeddingError) as exc:
        raise EmbeddingEvaluationError(f"result row has invalid identity fields: {exc}") from exc
    return question_id, image_id


def join_paired_rows(
    entropy_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    embedding_rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = 30,
) -> list[dict[str, Any]]:
    """Join all three sources on question ID and image ID and validate labels."""

    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise EmbeddingEvaluationError("expected_count must be a positive integer")
    if any(len(rows) != expected_count for rows in (entropy_rows, baseline_rows, embedding_rows)):
        raise EmbeddingEvaluationError(
            f"expected {expected_count} rows in entropy, baseline, and embedding inputs"
        )

    indexed: list[dict[tuple[str, str], Mapping[str, Any]]] = []
    for source_name, rows in (
        ("entropy", entropy_rows),
        ("baseline", baseline_rows),
        ("embedding", embedding_rows),
    ):
        mapping: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in rows:
            key = _row_key(row)
            if key in mapping:
                raise EmbeddingEvaluationError(
                    f"{source_name} input contains duplicate question/image key {key!r}"
                )
            mapping[key] = row
        indexed.append(mapping)
    entropy_by_key, baseline_by_key, embedding_by_key = indexed
    expected_keys = set(entropy_by_key)
    if set(baseline_by_key) != expected_keys or set(embedding_by_key) != expected_keys:
        raise EmbeddingEvaluationError(
            "entropy, baseline, and embedding rows do not contain identical "
            "question ID/image ID pairs"
        )

    joined: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        entropy = entropy_by_key[key]
        baseline = baseline_by_key[key]
        embedding = embedding_by_key[key]
        if entropy.get("reference_prediction_match") is not True:
            raise EmbeddingEvaluationError(
                f"entropy prediction differs from the baseline for pair {key!r}"
            )
        for field in (
            "category",
            "question",
            "normalized_ground_truth_label",
            "normalized_prediction",
            "prediction_status",
            "raw_prediction",
            "model_identifier",
            "model_revision",
            "device",
            "precision",
            "torch_version",
            "transformers_version",
            "seed",
            "generation_settings",
        ):
            if entropy.get(field) != baseline.get(field):
                raise EmbeddingEvaluationError(
                    f"entropy and baseline {field} differ for pair {key!r}"
                )
        if embedding.get("category") != entropy.get("category"):
            raise EmbeddingEvaluationError(f"embedding category differs for pair {key!r}")
        is_correct = (
            baseline.get("prediction_status") == "parsed"
            and baseline.get("normalized_prediction")
            == baseline.get("normalized_ground_truth_label")
        )
        if entropy.get("is_correct") is not is_correct or entropy.get("is_error") is not (
            not is_correct
        ):
            raise EmbeddingEvaluationError(
                f"entropy correctness label differs from baseline prediction for pair {key!r}"
            )
        try:
            normalized_entropy = float(entropy["normalized_entropy"])
            cosine_distance = float(embedding["cosine_distance"])
            nearest_distance = float(embedding["nearest_neighbor_distance"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise EmbeddingEvaluationError(f"pair {key!r} lacks a numeric score") from exc
        if not np.isfinite([normalized_entropy, cosine_distance, nearest_distance]).all():
            raise EmbeddingEvaluationError(f"pair {key!r} has a non-finite score")
        if not 0 <= normalized_entropy <= 1:
            raise EmbeddingEvaluationError(f"pair {key!r} has invalid normalized entropy")

        joined.append(
            {
                **dict(entropy),
                "cosine_distance": cosine_distance,
                "nearest_neighbor_distance": nearest_distance,
                "cosine_nearest_reference_image_id": embedding.get(
                    "cosine_nearest_reference_image_id"
                ),
                "nearest_neighbor_reference_image_id": embedding.get(
                    "nearest_neighbor_reference_image_id"
                ),
                "reference_set_id": embedding.get("reference_set_id"),
                "is_correct": bool(is_correct),
                "is_error": not is_correct,
            }
        )
    return joined


def _average_ranks(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise EmbeddingEvaluationError("rank-fusion inputs must be non-empty finite vectors")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    if values.size == 1:
        return np.zeros(1, dtype=np.float64)
    return (ranks - 1.0) / (values.size - 1.0)


def _spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if np.std(left_ranks) == 0 or np.std(right_ranks) == 0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _paired_stratified_bootstrap_delta(
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    metric: str,
    replicates: int,
    random_seed: int,
) -> dict[str, Any]:
    if replicates < 1:
        raise EmbeddingEvaluationError("bootstrap_replicates must be positive")
    negative_indices = np.flatnonzero(labels == 0)
    positive_indices = np.flatnonzero(labels == 1)
    if not negative_indices.size or not positive_indices.size:
        raise EmbeddingEvaluationError(
            "incremental comparison requires both correct and incorrect predictions"
        )
    metric_function = binary_auroc if metric == "auroc" else binary_auprc
    rng = np.random.Generator(np.random.PCG64(random_seed))
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled_indices = np.concatenate(
            [
                rng.choice(negative_indices, size=negative_indices.size, replace=True),
                rng.choice(positive_indices, size=positive_indices.size, replace=True),
            ]
        )
        sampled_labels = labels[sampled_indices]
        deltas[index] = metric_function(
            sampled_labels.tolist(), candidate_scores[sampled_indices].tolist()
        ) - metric_function(sampled_labels.tolist(), baseline_scores[sampled_indices].tolist())
    return {
        "point_difference": float(
            metric_function(labels.tolist(), candidate_scores.tolist())
            - metric_function(labels.tolist(), baseline_scores.tolist())
        ),
        "percentile_95_interval": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": random_seed,
        "bootstrap_design": "paired, stratified image-level resampling by correctness label",
    }


def compare_embedding_and_entropy(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = 2000,
    random_seed: int = 2027,
) -> dict[str, Any]:
    """Compare three raw scores and exploratory unsupervised rank fusion."""

    if len(rows) < 2:
        raise EmbeddingEvaluationError("at least two paired rows are required")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise EmbeddingEvaluationError("random_seed must be a non-negative integer")
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
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise EmbeddingEvaluationError("paired rows are missing valid labels or scores") from exc
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(
        np.concatenate(list(score_vectors.values()))
    ).all():
        raise EmbeddingEvaluationError("labels and scores must be finite binary observations")
    if len(set(labels.tolist())) != 2:
        raise EmbeddingEvaluationError("comparison requires correct and incorrect examples")

    error_mask = labels == 1
    rows_by_score: dict[str, Any] = {}
    table: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}
    for score_name, scores in score_vectors.items():
        metrics = {
            "auroc": binary_auroc(labels.tolist(), scores.tolist()),
            "auprc_average_precision": binary_auprc(labels.tolist(), scores.tolist()),
            "score_mean_correct": float(np.mean(scores[~error_mask])),
            "score_median_correct": float(np.median(scores[~error_mask])),
            "score_mean_incorrect": float(np.mean(scores[error_mask])),
            "score_median_incorrect": float(np.median(scores[error_mask])),
        }
        rows_by_score[score_name] = metrics
        table.append(
            {
                "score": score_name,
                "question_count": int(labels.size),
                "incorrect_count": int(np.sum(error_mask)),
                "incorrect_prevalence": float(np.mean(error_mask)),
                **metrics,
            }
        )
        curves[score_name] = risk_coverage_curve(labels.tolist(), scores.tolist())

    entropy_scores = score_vectors["predictive_entropy"]
    fusion_results: dict[str, Any] = {}
    for signal_name in ("cosine_distance", "nearest_neighbor_distance"):
        fused_scores = (
            _average_ranks(entropy_scores) + _average_ranks(score_vectors[signal_name])
        ) / 2.0
        fusion_results[f"entropy_plus_{signal_name}_rank_fusion"] = {
            "auroc": binary_auroc(labels.tolist(), fused_scores.tolist()),
            "auprc_average_precision": binary_auprc(labels.tolist(), fused_scores.tolist()),
            "delta_auroc_vs_entropy": _paired_stratified_bootstrap_delta(
                labels,
                entropy_scores,
                fused_scores,
                metric="auroc",
                replicates=bootstrap_replicates,
                random_seed=random_seed,
            ),
            "delta_auprc_vs_entropy": _paired_stratified_bootstrap_delta(
                labels,
                entropy_scores,
                fused_scores,
                metric="auprc",
                replicates=bootstrap_replicates,
                random_seed=random_seed,
            ),
        }

    return {
        "question_count": int(labels.size),
        "incorrect_count": int(np.sum(error_mask)),
        "incorrect_prevalence": float(np.mean(error_mask)),
        "score_comparison_table": table,
        "scores": rows_by_score,
        "risk_coverage_curves": curves,
        "embedding_entropy_spearman": {
            signal_name: _spearman_correlation(entropy_scores, score_vectors[signal_name])
            for signal_name in ("cosine_distance", "nearest_neighbor_distance")
        },
        "incremental_rank_fusion_check": {
            "method": (
                "Unsupervised, equal-weight mean of per-sample normalized average ranks "
                "for entropy and one embedding score; no labels used to set weights."
            ),
            "interpretation": (
                "Exploratory complementarity diagnostic only. It does not establish "
                "generalization or calibrated error probabilities."
            ),
            "results": fusion_results,
        },
        "risk_score_direction": "higher_scores_mean_higher_risk_of_an_incorrect_answer",
    }


def render_comparison_risk_coverage_svg(
    curves: Mapping[str, Sequence[Mapping[str, float | int | None]]],
    *,
    overall_error_rate: float,
    title: str,
) -> str:
    """Render three paired risk–coverage curves as an SVG without extra packages."""

    width, height = 820, 530
    left, top, plot_width, plot_height = 95, 62, 640, 340
    right, bottom = left + plot_width, top + plot_height
    colors = {
        "predictive_entropy": "#1769aa",
        "cosine_distance": "#d05a00",
        "nearest_neighbor_distance": "#26834a",
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
        elements.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        elements.extend(
            f'<circle cx="{x_pos(x):.1f}" cy="{y_pos(y):.1f}" r="3.5" fill="{color}"/>'
            for x, y in points
        )

    legend_y = bottom + 60
    for index, name in enumerate(curves):
        x = left + index * 210
        color = colors.get(name, "#202124")
        label = name.replace("_", " ")
        elements.extend(
            [
                f'<line x1="{x}" y1="{legend_y}" x2="{x + 28}" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text x="{x + 36}" y="{legend_y + 4}" font-size="12">'
                f"{escape(label)}</text>",
            ]
        )
    elements.extend(
        [
            f'<text x="{(left + right) / 2:.1f}" y="{height - 38}" text-anchor="middle" '
            'font-size="14">Coverage retained (highest-risk cases excluded first)</text>',
            f'<text x="22" y="{(top + bottom) / 2:.1f}" text-anchor="middle" '
            'font-size="14" transform="rotate(-90 22 '
            f'{(top + bottom) / 2:.1f})">Error rate among retained answers</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"
