"""Synthetic checks for expanded metrics and image-grouped bootstrap behavior."""

from __future__ import annotations

import pytest

from halluood.robustness import (
    RobustnessEvaluationError,
    deterministic_image_split,
    entropy_risk_coverage,
    image_grouped_bootstrap,
    image_grouped_selective_bootstrap,
    summarize_expanded_rows,
)


def _rows() -> list[dict]:
    values = (
        ("random", "yes", True, "parsed", 0.1),
        ("popular", "no", False, "parsed", 0.8),
        ("adversarial", "yes", True, "parsed", 0.2),
        ("random", "no", False, "parsed", 0.7),
        ("popular", "yes", True, "parsed", 0.3),
        ("adversarial", None, False, "ambiguous", 0.9),
    )
    rows = []
    for index, (category, prediction, correct, status, entropy) in enumerate(values):
        rows.append(
            {
                "question_id": index,
                "image_id": f"{100 + index % 2:012d}",
                "category": category,
                "normalized_prediction": prediction,
                "prediction_status": status,
                "is_correct": correct,
                "is_error": not correct,
                "normalized_entropy": entropy,
            }
        )
    return rows


def test_summary_reports_overall_and_category_metrics() -> None:
    summary = summarize_expanded_rows(_rows())

    assert summary["question_count"] == 6
    assert summary["image_count"] == 2
    assert summary["overall"]["accuracy"] == 0.5
    assert summary["overall"]["ambiguous_response_rate"] == pytest.approx(1 / 6)
    assert summary["overall"]["auroc"] == pytest.approx(1.0)
    assert summary["overall"]["auprc_average_precision"] == pytest.approx(1.0)
    assert summary["by_category"]["random"]["question_count"] == 2


def test_image_grouped_bootstrap_is_deterministic_and_reports_category_intervals() -> None:
    first = image_grouped_bootstrap(_rows(), replicates=30, seed=42)
    second = image_grouped_bootstrap(_rows(), replicates=30, seed=42)

    assert first == second
    assert first["unit_of_resampling"].startswith("image_id")
    assert first["image_count"] == 2
    assert first["intervals"]["overall"]["accuracy"]["valid_replicates"] == 30
    assert first["intervals"]["random"]["auroc"]["valid_replicates"] <= 30


def test_group_split_is_reproducible_and_disjoint() -> None:
    ids = [f"{index:012d}" for index in range(10)]
    calibration, evaluation = deterministic_image_split(ids, calibration_fraction=0.2, seed=7)

    assert calibration == deterministic_image_split(ids, seed=7)[0]
    assert len(calibration) == 2
    assert len(set(calibration) & set(evaluation)) == 0
    assert set(calibration) | set(evaluation) == set(ids)


def test_selective_bootstrap_resamples_complete_images_with_frozen_threshold() -> None:
    rows = _rows()
    first = image_grouped_selective_bootstrap(rows, 0.5, replicates=50, seed=12)
    second = image_grouped_selective_bootstrap(rows, 0.5, replicates=50, seed=12)

    assert first == second
    assert first["image_count"] == 2
    assert first["threshold"] == 0.5
    assert first["intervals"]["coverage"]["valid_replicates"] == 50
    assert first["intervals"]["abstention_rate"]["lower"] == pytest.approx(
        1.0 - first["intervals"]["coverage"]["upper"]
    )


def test_risk_coverage_uses_high_entropy_as_risk() -> None:
    curve = entropy_risk_coverage(_rows())

    assert curve[0]["coverage"] == 0.0
    assert curve[-1]["coverage"] == 1.0
    assert curve[-1]["risk"] == 0.5


def test_summary_rejects_duplicate_question_identity_and_bad_entropy() -> None:
    rows = _rows()
    with pytest.raises(RobustnessEvaluationError, match="duplicate"):
        summarize_expanded_rows([*rows, rows[0]])
    rows[0]["normalized_entropy"] = float("nan")
    with pytest.raises(RobustnessEvaluationError, match=r"in \[0, 1\]"):
        summarize_expanded_rows(rows)
