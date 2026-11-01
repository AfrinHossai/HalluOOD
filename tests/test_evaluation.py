from __future__ import annotations

import json

import pytest

from halluood.evaluation import (
    EvaluationError,
    binary_auprc,
    binary_auroc,
    evaluate_entropy_rows,
    risk_coverage_curve,
)
from scripts.evaluate_entropy import evaluate_files


def test_auroc_and_average_precision_rank_incorrect_answers_by_risk():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]

    assert binary_auroc(labels, scores) == pytest.approx(0.75)
    assert binary_auprc(labels, scores) == pytest.approx(5 / 6)


def test_rank_metrics_handle_tied_scores_and_auroc_requires_both_classes():
    assert binary_auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert binary_auprc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    with pytest.raises(EvaluationError, match="at least one correct and one incorrect"):
        binary_auroc([1, 1], [0.2, 0.9])


def test_risk_coverage_excludes_high_score_ties_as_a_group():
    curve = risk_coverage_curve([0, 1, 1, 0], [0.1, 0.2, 0.8, 0.8])

    assert [point["coverage"] for point in curve] == [0.0, 0.25, 0.5, 1.0]
    assert [point["risk"] for point in curve] == [None, 0.0, 0.5, 0.5]
    assert curve[-1]["excluded_count"] == 0


def test_entropy_evaluation_marks_ambiguous_predictions_as_errors():
    rows = [
        {
            "category": "random",
            "normalized_ground_truth_label": "yes",
            "normalized_prediction": "yes",
            "prediction_status": "parsed",
            "uncertainty_score": {"normalized_entropy": 0.2},
        },
        {
            "category": "random",
            "normalized_ground_truth_label": "no",
            "normalized_prediction": None,
            "prediction_status": "ambiguous",
            "uncertainty_score": {"normalized_entropy": 0.8},
        },
    ]

    evaluated, summary = evaluate_entropy_rows(rows)

    assert [row["is_correct"] for row in evaluated] == [True, False]
    assert [row["is_error"] for row in evaluated] == [False, True]
    assert summary["entropy_correct"]["mean"] == pytest.approx(0.2)
    assert summary["entropy_incorrect"]["mean"] == pytest.approx(0.8)
    assert summary["by_category"]["random"]["incorrect_count"] == 1


def test_entropy_evaluation_rejects_invalid_entropy_and_metrics_inputs():
    invalid_rows = [
        {
            "category": "random",
            "normalized_ground_truth_label": "yes",
            "normalized_prediction": "yes",
            "prediction_status": "parsed",
            "uncertainty_score": {"normalized_entropy": 1.2},
        }
    ]
    with pytest.raises(EvaluationError, match="between 0 and 1"):
        evaluate_entropy_rows(invalid_rows)
    with pytest.raises(EvaluationError, match="finite"):
        binary_auroc([0, 1], [0.0, float("nan")])


def test_evaluate_files_verifies_reference_and_writes_report_artifacts(tmp_path):
    settings = {"do_sample": False, "max_new_tokens": 32, "num_beams": 1, "seed": 17}
    common = {
        "model_identifier": "test/model",
        "model_revision": "rev-1",
        "device": "cpu",
        "precision": "float32",
        "torch_version": "test-torch",
        "transformers_version": "test-transformers",
        "seed": 17,
        "generation_settings": settings,
    }
    entropy_rows = []
    reference_rows = []
    for category, question_id, prediction, gold, score in (
        ("random", 1, "yes", "yes", 0.1),
        ("popular", 2, "no", "yes", 0.9),
    ):
        row = {
            **common,
            "category": category,
            "question_id": question_id,
            "image_id": f"image-{question_id}",
            "raw_prediction": prediction,
            "normalized_prediction": prediction,
            "normalized_ground_truth_label": gold,
            "prediction_status": "parsed",
            "uncertainty_score": {"normalized_entropy": score},
        }
        entropy_rows.append(row)
        reference_rows.append(
            {key: value for key, value in row.items() if key != "uncertainty_score"}
        )

    input_file = tmp_path / "entropy.jsonl"
    reference_file = tmp_path / "baseline.jsonl"
    output_file = tmp_path / "evaluated.jsonl"
    summary_file = tmp_path / "metrics.json"
    curve_file = tmp_path / "risk_curve.svg"
    input_file.write_text("".join(json.dumps(row) + "\n" for row in entropy_rows))
    reference_file.write_text("".join(json.dumps(row) + "\n" for row in reference_rows))

    summary = evaluate_files(
        input_file=input_file,
        reference_file=reference_file,
        output_file=output_file,
        summary_file=summary_file,
        curve_file=curve_file,
        expected_count=2,
    )

    evaluated_rows = [json.loads(line) for line in output_file.read_text().splitlines()]
    assert summary["auroc"] == pytest.approx(1.0)
    assert summary["auprc_average_precision"] == pytest.approx(1.0)
    assert summary["reference_comparison"]["same_question_set"] is True
    assert all(row["reference_prediction_match"] for row in evaluated_rows)
    assert all("is_correct" in row and "normalized_entropy" in row for row in evaluated_rows)
    assert "<svg" in curve_file.read_text()
    assert json.loads(summary_file.read_text())["question_count"] == 2
