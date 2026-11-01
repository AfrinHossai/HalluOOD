from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from halluood.answer_disagreement_evaluation import (
    AnswerDisagreementEvaluationError,
    compare_error_detection_scores,
    render_answer_disagreement_risk_coverage_svg,
)
from halluood.model import MockVLMAdapter, ModelMetadata, SamplingSettings
from scripts import evaluate_answer_disagreement as evaluation_script
from scripts.evaluate_answer_disagreement import (
    _join_augmentation_rows,
    evaluate_sampled_row,
    run_answer_disagreement_evaluation,
)


def _paired_sources():
    baseline_rows = []
    entropy_rows = []
    embedding_rows = []
    augmentation_rows = []
    generation_settings = {
        "do_sample": False,
        "max_new_tokens": 32,
        "num_beams": 1,
        "seed": 2027,
    }
    for index in range(1, 31):
        category = ("random", "popular", "adversarial")[(index - 1) // 10]
        question_id = 30000 + index
        image_id = f"{index:012d}"
        is_error = index > 24
        gold = "no" if is_error else "yes"
        shared = {
            "question_id": question_id,
            "image_id": image_id,
            "image_filename": f"COCO_val2014_{image_id}.jpg",
            "category": category,
            "question": f"Is object {index} present?",
            "ground_truth_label": gold,
            "normalized_ground_truth_label": gold,
            "raw_prediction": "Yes, present.",
            "normalized_prediction": "yes",
            "prediction_status": "parsed",
            "model_identifier": "halluood/mock-vlm",
            "model_revision": "test-only",
            "device": "cpu",
            "precision": None,
            "torch_version": None,
            "transformers_version": None,
            "seed": 2027,
            "generation_settings": generation_settings,
        }
        baseline_rows.append(shared)
        entropy_rows.append(
            {
                **shared,
                "reference_prediction_match": True,
                "is_correct": not is_error,
                "is_error": is_error,
                "normalized_entropy": 0.9 if is_error else 0.1,
            }
        )
        embedding_rows.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "category": category,
                "seed": 2027,
                "cosine_distance": index / 1000,
                "nearest_neighbor_distance": float(index),
            }
        )
        augmentation_rows.append(
            {
                **shared,
                "is_correct": not is_error,
                "is_error": is_error,
                "augmentation_consistency_score": 0.0,
                "original_prediction_for_comparison": {
                    "raw_prediction": shared["raw_prediction"],
                    "normalized_prediction": shared["normalized_prediction"],
                    "is_correct": not is_error,
                    "is_error": is_error,
                },
            }
        )
    return baseline_rows, entropy_rows, embedding_rows, augmentation_rows


def test_comparison_calculates_all_five_scores_on_shared_error_labels():
    rows = [
        {
            "is_error": False,
            "normalized_entropy": 0.1,
            "cosine_distance": 0.1,
            "nearest_neighbor_distance": 1.0,
            "augmentation_consistency_score": 0.0,
            "answer_disagreement_score": 0.0,
        },
        {
            "is_error": False,
            "normalized_entropy": 0.2,
            "cosine_distance": 0.2,
            "nearest_neighbor_distance": 2.0,
            "augmentation_consistency_score": 0.1,
            "answer_disagreement_score": 0.2,
        },
        {
            "is_error": True,
            "normalized_entropy": 0.8,
            "cosine_distance": 0.8,
            "nearest_neighbor_distance": 8.0,
            "augmentation_consistency_score": 0.7,
            "answer_disagreement_score": 0.8,
        },
        {
            "is_error": True,
            "normalized_entropy": 0.9,
            "cosine_distance": 0.9,
            "nearest_neighbor_distance": 9.0,
            "augmentation_consistency_score": 0.8,
            "answer_disagreement_score": 0.9,
        },
    ]

    result = compare_error_detection_scores(rows)
    table = {row["score"]: row for row in result["comparison_table"]}

    assert result["question_count"] == 4
    assert result["incorrect_or_ambiguous_count"] == 2
    assert table["answer_disagreement"]["auroc"] == pytest.approx(1.0)
    assert table["answer_disagreement"]["auprc_average_precision"] == pytest.approx(1.0)
    assert table["predictive_entropy"]["auroc"] == pytest.approx(1.0)
    assert set(result["risk_coverage_curves"]) == {
        "predictive_entropy",
        "cosine_distance",
        "nearest_neighbor_distance",
        "augmentation_consistency",
        "answer_disagreement",
    }
    svg = render_answer_disagreement_risk_coverage_svg(
        result["risk_coverage_curves"],
        overall_error_rate=result["error_prevalence"],
    )
    assert '<svg xmlns="http://www.w3.org/2000/svg"' in svg
    assert "Sampled answer disagreement" in svg
    assert "Predictive entropy" in svg


def test_comparison_rejects_invalid_labels_and_out_of_range_disagreement():
    rows = [
        {
            "is_error": "false",
            "normalized_entropy": 0.1,
            "cosine_distance": 0.1,
            "nearest_neighbor_distance": 1.0,
            "augmentation_consistency_score": 0.0,
            "answer_disagreement_score": 0.0,
        },
        {
            "is_error": True,
            "normalized_entropy": 0.9,
            "cosine_distance": 0.9,
            "nearest_neighbor_distance": 9.0,
            "augmentation_consistency_score": 0.0,
            "answer_disagreement_score": 1.2,
        },
    ]
    with pytest.raises(AnswerDisagreementEvaluationError, match="boolean"):
        compare_error_detection_scores(rows)
    rows[0]["is_error"] = False
    with pytest.raises(AnswerDisagreementEvaluationError, match=r"in \[0, 1\]"):
        compare_error_detection_scores(rows)


def test_augmentation_pair_join_rejects_mismatched_baseline_labels():
    baseline, entropy, embeddings, augmentation = _paired_sources()
    from scripts.evaluate_augmentation_pilot import _verify_fixed_sample

    paired = _verify_fixed_sample(baseline, entropy, embeddings)
    augmentation[0]["is_error"] = True

    with pytest.raises(ValueError, match="is_error differs"):
        _join_augmentation_rows(paired, augmentation)


def test_evaluate_sampled_row_records_outputs_and_preserves_original_label(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    filename = "COCO_val2014_000000000007.jpg"
    Image.new("RGB", (40, 30), color=(80, 100, 120)).save(image_dir / filename)
    row = {
        "question_id": 30007,
        "image_id": "000000000007",
        "image_filename": filename,
        "category": "random",
        "question": "Is there a bicycle?",
        "normalized_prediction": "yes",
        "raw_prediction": "Yes, there is a bicycle.",
        "prediction_status": "parsed",
        "normalized_ground_truth_label": "no",
        "is_correct": False,
        "is_error": True,
        "normalized_entropy": 0.7,
        "cosine_distance": 0.02,
        "nearest_neighbor_distance": 12.0,
        "augmentation_consistency_score": 0.0,
    }
    original = dict(row)
    adapter = MockVLMAdapter(sampled_answers=["Yes.", "No.", "not sure", "No.", "Yes."])
    settings = SamplingSettings(num_generations=5, seed=77)

    evaluated = evaluate_sampled_row(
        row,
        image_dir=image_dir,
        adapter=adapter,
        settings=settings,
    )

    assert [answer["seed"] for answer in evaluated["sampled_answers"]] == [
        77,
        78,
        79,
        80,
        81,
    ]
    assert [answer["normalized_answer"] for answer in evaluated["sampled_answers"]] == [
        "yes",
        "no",
        "ambiguous",
        "no",
        "yes",
    ]
    assert evaluated["answer_disagreement_details"]["counts"] == {
        "yes": 2,
        "no": 2,
        "ambiguous": 1,
    }
    assert evaluated["is_error"] is True
    assert row == original


def test_full_pilot_workflow_writes_150_samples_without_touching_baseline(
    tmp_path, monkeypatch
):
    baseline, entropy, embeddings, augmentation = _paired_sources()
    files = {
        tmp_path / "baseline.jsonl": baseline,
        tmp_path / "entropy.jsonl": entropy,
        tmp_path / "embeddings.jsonl": embeddings,
        tmp_path / "augmentation.jsonl": augmentation,
    }
    monkeypatch.setattr(
        evaluation_script,
        "_read_jsonl",
        lambda path: files[Path(path)],
    )
    monkeypatch.setattr(evaluation_script, "_load_pope_annotations", lambda *_args: {})

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for row in baseline:
        Image.new("RGB", (40, 30), color=(80, 100, 120)).save(
            image_dir / row["image_filename"]
        )
    adapter = MockVLMAdapter(sampled_answers=["Yes."] * 150)
    adapter._metadata = ModelMetadata(
        model_id="halluood/mock-vlm",
        requested_revision="test-only",
        resolved_revision="test-only",
        backend="mock",
        device="cpu",
        precision=None,
        transformers_version=None,
        torch_version=None,
    )
    output_dir = tmp_path / "results"

    summary = run_answer_disagreement_evaluation(
        baseline_results_file=tmp_path / "baseline.jsonl",
        entropy_results_file=tmp_path / "entropy.jsonl",
        embedding_results_file=tmp_path / "embeddings.jsonl",
        augmentation_results_file=tmp_path / "augmentation.jsonl",
        annotations_dir=tmp_path / "annotations",
        image_dir=image_dir,
        output_dir=output_dir,
        adapter=adapter,
    )

    scores_path = output_dir / "pope_answer_disagreement_30_seed_2027_scores.jsonl"
    saved_rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
    assert len(saved_rows) == 30
    assert sum(len(row["sampled_answers"]) for row in saved_rows) == 150
    assert len(adapter.sampled_calls) == 30
    assert adapter.calls == []
    assert summary["sample"]["original_baseline_predictions_and_error_labels_reused_unchanged"]
    assert summary["sampling"]["model_calls_on_original_baseline_predictions"] == 0
    assert len(summary["comparison_table"]) == 5
    assert (output_dir / "pope_answer_disagreement_30_seed_2027_comparison.csv").is_file()
    assert (output_dir / "pope_answer_disagreement_30_seed_2027_comparison.json").is_file()
    assert (output_dir / "pope_answer_disagreement_30_seed_2027_risk_coverage.svg").is_file()
