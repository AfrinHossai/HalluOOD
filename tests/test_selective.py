"""Tests for image-disjoint selective prediction using synthetic rows."""

from __future__ import annotations

import pytest

from halluood.selective import (
    SelectivePredictionError,
    calibrate_abstention_threshold,
    evaluate_selective_predictions,
    should_abstain,
)


def _calibrate(rows, target=0.5):
    return calibrate_abstention_threshold(
        rows,
        evaluation_image_ids=["COCO_val2014_000000000101.jpg"],
        target_coverage=target,
        calibration_dataset_id="calibration-v1",
        evaluation_dataset_id="evaluation-v1",
        calibration_seed=17,
    )


def test_threshold_keeps_ties_together_and_uses_nearest_coverage() -> None:
    rows = [
        {"image_id": f"cal-{index}", "normalized_entropy": score}
        for index, score in enumerate((0.1, 0.2, 0.2, 0.3, 0.4))
    ]

    calibrated = _calibrate(rows, target=0.6)

    assert calibrated.threshold == 0.2
    assert calibrated.calibration_coverage == 0.6
    assert should_abstain(0.2, calibrated.threshold) is False
    assert should_abstain(0.20001, calibrated.threshold) is True
    assert calibrated.to_dict()["selection_uses_correctness_labels"] is False


def test_equally_close_coverage_prefers_under_target() -> None:
    rows = [
        {"image_id": f"cal-{index}", "normalized_entropy": score}
        for index, score in enumerate((0.1, 0.1, 0.2, 0.2))
    ]

    calibrated = _calibrate(rows, target=0.75)

    assert calibrated.threshold == 0.1
    assert calibrated.calibration_coverage == 0.5


def test_calibration_rejects_image_leak_after_coco_id_normalization() -> None:
    rows = [{"image_id": "000000000101", "normalized_entropy": 0.4}]

    with pytest.raises(SelectivePredictionError, match="overlap"):
        _calibrate(rows)


def test_calibration_rejects_invalid_scores_targets_and_empty_rows() -> None:
    with pytest.raises(SelectivePredictionError, match="must not be empty"):
        _calibrate([])
    with pytest.raises(SelectivePredictionError, match="finite number"):
        _calibrate([{"image_id": "cal-1", "normalized_entropy": float("nan")}])
    with pytest.raises(SelectivePredictionError, match="target_coverage"):
        _calibrate([{"image_id": "cal-1", "normalized_entropy": 0.4}], target=0)


def test_evaluation_reports_coverage_selective_risk_and_abstention() -> None:
    evaluation_rows = [
        {
            "question_id": index,
            "image_id": f"COCO_val2014_{index + 201:012d}.jpg",
            "normalized_entropy": entropy,
            "is_error": is_error,
        }
        for index, (entropy, is_error) in enumerate(
            ((0.05, False), (0.2, True), (0.75, True), (0.95, False))
        )
    ]
    calibration = calibrate_abstention_threshold(
        [
            {"image_id": f"cal-{index}", "normalized_entropy": score}
            for index, score in enumerate((0.1, 0.2, 0.3, 0.4))
        ],
        evaluation_image_ids=[row["image_id"] for row in evaluation_rows],
        target_coverage=0.5,
        calibration_dataset_id="calibration-v1",
        evaluation_dataset_id="evaluation-v1",
        calibration_seed=17,
    )

    metrics = evaluate_selective_predictions(
        evaluation_rows, calibration, evaluation_dataset_id="evaluation-v1"
    )

    assert metrics["coverage"] == 0.5
    assert metrics["selective_risk"] == 0.5
    assert metrics["abstention_rate"] == 0.5
    assert metrics["answered_count"] == 2
    assert metrics["abstained_count"] == 2


def test_evaluation_rejects_wrong_split_and_inconsistent_labels() -> None:
    rows = [{"image_id": "eval-1", "normalized_entropy": 0.1, "is_error": False}]
    calibration = calibrate_abstention_threshold(
        [{"image_id": "cal-1", "normalized_entropy": 0.1}],
        evaluation_image_ids=["eval-1"],
        target_coverage=1.0,
        calibration_dataset_id="calibration-v1",
        evaluation_dataset_id="evaluation-v1",
        calibration_seed=17,
    )
    with pytest.raises(SelectivePredictionError, match="differs"):
        evaluate_selective_predictions(rows, calibration, evaluation_dataset_id="other")

    rows[0]["is_correct"] = False
    with pytest.raises(SelectivePredictionError, match="inconsistent"):
        evaluate_selective_predictions(rows, calibration, evaluation_dataset_id="evaluation-v1")


@pytest.mark.parametrize("score", [-0.01, 1.01, float("inf"), float("nan")])
def test_decision_rejects_invalid_entropy(score: float) -> None:
    with pytest.raises(SelectivePredictionError, match="finite number"):
        should_abstain(score, 0.5)
