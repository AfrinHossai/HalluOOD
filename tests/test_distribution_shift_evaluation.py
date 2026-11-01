"""Tests for paired clean-versus-shifted reliability metrics."""

from __future__ import annotations

import pytest

from halluood.distribution_shift_evaluation import (
    DistributionShiftEvaluationError,
    render_severity_plot,
    summarize_distribution_shift,
)


def _row(question_id: int, *, error: bool, entropy: float) -> dict:
    return {
        "category": "random",
        "question_id": question_id,
        "image_id": f"{question_id:012d}",
        "question": f"Question {question_id}?",
        "ground_truth_label": "yes",
        "normalized_ground_truth_label": "yes",
        "normalized_entropy": entropy,
        "is_error": error,
        "is_correct": not error,
        "inference_time_seconds": 0.2,
        "uncertainty_time_seconds": 0.4,
    }


def test_paired_summary_computes_accuracy_entropy_and_error_metrics() -> None:
    clean = [_row(1, error=False, entropy=0.1), _row(2, error=True, entropy=0.8)]
    shifted = [
        {
            **_row(1, error=False, entropy=0.2),
            "perturbation_type": "gaussian_blur",
            "severity": 1,
        },
        {
            **_row(2, error=True, entropy=0.7),
            "perturbation_type": "gaussian_blur",
            "severity": 1,
        },
    ]
    all_conditions = []
    for perturbation_type in (
        "gaussian_blur",
        "brightness_reduction",
        "gaussian_noise",
        "contrast_reduction",
    ):
        for severity in (1, 2, 3):
            all_conditions.extend(
                {
                    **_row(question_id, error=error, entropy=entropy + severity * 0.01),
                    "perturbation_type": perturbation_type,
                    "severity": severity,
                }
                for question_id, error, entropy in ((1, False, 0.1), (2, True, 0.8))
            )

    # Replace one condition with the values used to check the metric calculations.
    all_conditions = [
        row
        for row in all_conditions
        if (row["perturbation_type"], row["severity"]) != ("gaussian_blur", 1)
    ] + shifted
    summary = summarize_distribution_shift(clean, all_conditions, expected_count=2)

    clean_metrics = summary["conditions"][0]
    blur_metrics = next(
        row
        for row in summary["conditions"]
        if row["condition_id"] == "gaussian_blur_severity_1"
    )
    assert clean_metrics["accuracy"] == 0.5
    assert clean_metrics["auroc"] == 1.0
    assert clean_metrics["auprc"] == 1.0
    assert blur_metrics["accuracy"] == 0.5
    assert blur_metrics["mean_normalized_entropy"] == pytest.approx(0.45)
    assert blur_metrics["auroc"] == 1.0
    assert blur_metrics["delta_mean_entropy_vs_clean"] == pytest.approx(0.0)
    assert summary["condition_count"] == 12


def test_summary_rejects_changed_question_or_label() -> None:
    clean = [_row(1, error=False, entropy=0.1), _row(2, error=True, entropy=0.8)]
    shifted = []
    for perturbation_type in (
        "gaussian_blur",
        "brightness_reduction",
        "gaussian_noise",
        "contrast_reduction",
    ):
        for severity in (1, 2, 3):
            shifted.extend(
                {
                    **_row(question_id, error=error, entropy=entropy),
                    "perturbation_type": perturbation_type,
                    "severity": severity,
                }
                for question_id, error, entropy in ((1, False, 0.2), (2, True, 0.7))
            )
    shifted[0]["question"] = "Different question?"

    with pytest.raises(DistributionShiftEvaluationError, match="changed paired clean field"):
        summarize_distribution_shift(clean, shifted, expected_count=2)


def test_summary_rejects_missing_or_duplicate_pairs() -> None:
    clean = [_row(1, error=False, entropy=0.1), _row(2, error=True, entropy=0.8)]
    shifted = []
    for perturbation_type in (
        "gaussian_blur",
        "brightness_reduction",
        "gaussian_noise",
        "contrast_reduction",
    ):
        for severity in (1, 2, 3):
            shifted.extend(
                {
                    **_row(question_id, error=error, entropy=entropy),
                    "perturbation_type": perturbation_type,
                    "severity": severity,
                }
                for question_id, error, entropy in ((1, False, 0.2), (2, True, 0.7))
            )
    with pytest.raises(DistributionShiftEvaluationError, match="paired"):
        summarize_distribution_shift(clean, shifted[:-1], expected_count=2)
    with pytest.raises(DistributionShiftEvaluationError, match="duplicate"):
        summarize_distribution_shift(clean, [*shifted, shifted[0]], expected_count=2)


def test_severity_plot_contains_clean_point_and_shift_labels() -> None:
    clean = {
        "condition_id": "clean",
        "perturbation_type": "none",
        "severity": 0,
        "accuracy": 0.8,
    }
    rows = [clean]
    for perturbation_type in (
        "gaussian_blur",
        "brightness_reduction",
        "gaussian_noise",
        "contrast_reduction",
    ):
        rows.extend(
            {
                "condition_id": f"{perturbation_type}_severity_{severity}",
                "perturbation_type": perturbation_type,
                "severity": severity,
                "accuracy": 0.8 - severity * 0.1,
            }
            for severity in (1, 2, 3)
        )

    svg = render_severity_plot(
        rows,
        metric="accuracy",
        title="Accuracy by severity",
        y_label="Accuracy",
    )

    assert "Accuracy by severity" in svg
    assert "gaussian blur" in svg
    assert "Severity (0 = clean" in svg
