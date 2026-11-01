from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from halluood.augmentation_evaluation import (
    AugmentationEvaluationError,
    compare_augmentation_scores,
    render_augmentation_risk_coverage_svg,
)
from halluood.model import GenerationSettings, MockVLMAdapter, ModelMetadata
from scripts import evaluate_augmentation_pilot as pilot_script
from scripts.evaluate_augmentation_pilot import (
    _verify_fixed_sample,
    evaluate_augmented_row,
    run_augmentation_evaluation,
)


def _fixed_sample_sources():
    baseline_rows = []
    entropy_rows = []
    embedding_rows = []
    generation_settings = {
        "do_sample": False,
        "max_new_tokens": 32,
        "num_beams": 1,
        "seed": 2027,
    }
    for index in range(1, 31):
        category = ("random", "popular", "adversarial")[(index - 1) // 10]
        question_id = 2800 + index
        image_id = f"{index:012d}"
        is_error = index > 24
        gold = "no" if is_error else "yes"
        prediction = "yes"
        shared = {
            "question_id": question_id,
            "image_id": image_id,
            "image_filename": f"COCO_val2014_{image_id}.jpg",
            "category": category,
            "question": f"Is object {index} present?",
            "ground_truth_label": gold,
            "normalized_ground_truth_label": gold,
            "raw_prediction": "Yes, present.",
            "normalized_prediction": prediction,
            "prediction_status": "parsed",
            "model_identifier": "test/model",
            "model_revision": "test-revision",
            "device": "cpu",
            "precision": "float32",
            "torch_version": "test-torch",
            "transformers_version": "test-transformers",
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
                "cosine_distance": float(index) / 1000,
                "nearest_neighbor_distance": float(index),
            }
        )
    return baseline_rows, entropy_rows, embedding_rows


def test_fixed_sample_verification_retains_the_same_30_pairs_and_labels():
    baseline, entropy, embeddings = _fixed_sample_sources()

    joined = _verify_fixed_sample(baseline, entropy, embeddings)

    assert len(joined) == 30
    assert len({row["question_id"] for row in joined}) == 30
    assert len({row["image_id"] for row in joined}) == 30
    assert Counter(row["category"] for row in joined) == {
        "random": 10,
        "popular": 10,
        "adversarial": 10,
    }
    assert [row["is_error"] for row in joined] == [row["question_id"] > 2824 for row in joined]


def test_fixed_sample_verification_rejects_wrong_question_image_pair_or_label():
    baseline, entropy, embeddings = _fixed_sample_sources()
    embeddings[0] = {**embeddings[0], "image_id": "999999999999"}
    with pytest.raises(AugmentationEvaluationError, match="identical question ID/image ID"):
        _verify_fixed_sample(baseline, entropy, embeddings)

    baseline, entropy, embeddings = _fixed_sample_sources()
    entropy[0] = {**entropy[0], "is_error": True}
    with pytest.raises(ValueError, match="correctness label"):
        _verify_fixed_sample(baseline, entropy, embeddings)


def test_augmented_row_uses_original_label_and_only_runs_transformed_views(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "COCO_val2014_000000000007.jpg"
    Image.new("RGB", (200, 160), color=(75, 105, 135)).save(image_path)
    row = {
        "question_id": 2807,
        "image_id": "000000000007",
        "image_filename": image_path.name,
        "category": "random",
        "question": "Is there a bicycle?",
        "normalized_ground_truth_label": "no",
        "ground_truth_label": "no",
        "raw_prediction": "Yes, there is a bicycle.",
        "normalized_prediction": "yes",
        "prediction_status": "parsed",
        "is_correct": False,
        "is_error": True,
        "normalized_entropy": 0.7,
        "cosine_distance": 0.02,
        "nearest_neighbor_distance": 12.0,
    }
    original_row = dict(row)

    class PredictionMock(MockVLMAdapter):
        def __init__(self):
            super().__init__(answer="unused")
            self.answers = iter(
                ["Yes.", "No.", "Yes.", "I cannot tell.", "Yes.", "Yes."]
            )

        def generate_answer(self, image, question):
            super().generate_answer(image, question)
            return next(self.answers)

    adapter = PredictionMock()

    evaluated = evaluate_augmented_row(
        row,
        image_dir=image_dir,
        adapter=adapter,
        augmentation_seed=2027,
    )

    assert len(adapter.calls) == 6  # original prediction was reused, not regenerated
    assert evaluated["raw_prediction"] == row["raw_prediction"]
    assert evaluated["is_correct"] is False
    assert evaluated["is_error"] is True
    assert evaluated["augmentation_consistency_score"] == pytest.approx(2 / 6)
    assert evaluated["augmentation_seed"] == 2027
    assert len(evaluated["transformed_views"]) == 6
    assert all(view["image_width"] == 200 for view in evaluated["transformed_views"])
    assert all(view["image_height"] == 160 for view in evaluated["transformed_views"])
    assert row == original_row


def test_comparison_includes_four_scores_metrics_risk_curves_and_svg():
    rows = [
        {
            "is_error": False,
            "normalized_entropy": 0.1,
            "cosine_distance": 0.9,
            "nearest_neighbor_distance": 0.1,
            "augmentation_consistency_score": 0.0,
        },
        {
            "is_error": False,
            "normalized_entropy": 0.2,
            "cosine_distance": 0.8,
            "nearest_neighbor_distance": 0.4,
            "augmentation_consistency_score": 0.25,
        },
        {
            "is_error": True,
            "normalized_entropy": 0.8,
            "cosine_distance": 0.2,
            "nearest_neighbor_distance": 0.6,
            "augmentation_consistency_score": 0.75,
        },
        {
            "is_error": True,
            "normalized_entropy": 0.9,
            "cosine_distance": 0.1,
            "nearest_neighbor_distance": 0.9,
            "augmentation_consistency_score": 1.0,
        },
    ]

    result = compare_augmentation_scores(rows)
    table = {row["score"]: row for row in result["score_comparison_table"]}

    assert result["incorrect_or_ambiguous_count"] == 2
    assert table["predictive_entropy"]["auroc"] == pytest.approx(1.0)
    assert table["augmentation_consistency"]["auroc"] == pytest.approx(1.0)
    assert table["cosine_distance"]["auroc"] == pytest.approx(0.0)
    assert table["augmentation_consistency"]["auprc_average_precision"] == pytest.approx(1.0)
    assert set(result["risk_coverage_curves"]) == {
        "predictive_entropy",
        "cosine_distance",
        "nearest_neighbor_distance",
        "augmentation_consistency",
    }
    svg = render_augmentation_risk_coverage_svg(
        result["risk_coverage_curves"],
        overall_error_rate=result["incorrect_prevalence"],
    )
    assert "<svg xmlns=" in svg
    assert "Augmentation consistency" in svg
    assert "Nearest-neighbor distance" in svg


def test_comparison_rejects_invalid_augmentation_score():
    rows = [
        {
            "is_error": False,
            "normalized_entropy": 0.1,
            "cosine_distance": 0.1,
            "nearest_neighbor_distance": 1.0,
            "augmentation_consistency_score": -0.1,
        },
        {
            "is_error": True,
            "normalized_entropy": 0.8,
            "cosine_distance": 0.8,
            "nearest_neighbor_distance": 3.0,
            "augmentation_consistency_score": 0.7,
        },
    ]
    with pytest.raises(AugmentationEvaluationError, match="must be in"):
        compare_augmentation_scores(rows)


def test_evaluation_workflow_writes_paired_results_without_mutating_inputs(
    tmp_path, monkeypatch
):
    baseline, entropy, embeddings = _fixed_sample_sources()
    source_path_rows = {
        tmp_path / "baseline.jsonl": baseline,
        tmp_path / "entropy.jsonl": entropy,
        tmp_path / "embedding.jsonl": embeddings,
    }
    monkeypatch.setattr(
        pilot_script,
        "_read_jsonl",
        lambda path: source_path_rows[Path(path)],
    )
    monkeypatch.setattr(pilot_script, "_load_pope_annotations", lambda *_args: {})

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for row in baseline:
        Image.new("RGB", (200, 160), color=(80, 100, 120)).save(
            image_dir / row["image_filename"]
        )

    adapter = MockVLMAdapter(answer="Yes.")
    adapter._metadata = ModelMetadata(
        model_id="test/model",
        requested_revision="test-revision",
        resolved_revision="test-revision",
        backend="mock",
        device="cpu",
        precision="float32",
        transformers_version="test-transformers",
        torch_version="test-torch",
        generation=GenerationSettings(),
    )
    output_dir = tmp_path / "results"

    summary = run_augmentation_evaluation(
        baseline_results_file=tmp_path / "baseline.jsonl",
        entropy_results_file=tmp_path / "entropy.jsonl",
        embedding_results_file=tmp_path / "embedding.jsonl",
        annotations_dir=tmp_path / "annotations",
        image_dir=image_dir,
        output_dir=output_dir,
        adapter=adapter,
    )

    stem = "pope_augmentation_consistency_30_seed_2027"
    scores_path = output_dir / f"{stem}_scores.jsonl"
    saved_rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
    assert len(adapter.calls) == 180
    assert summary["augmentation"]["total_transformed_model_calls"] == 180
    assert summary["augmentation"]["model_calls_on_original_images"] == 0
    assert len(saved_rows) == 30
    assert all(len(row["transformed_views"]) == 6 for row in saved_rows)
    assert all(row["transformed_model_call_count"] == 6 for row in saved_rows)
    assert all(
        row["original_prediction_for_comparison"]["raw_prediction"]
        == row["raw_prediction"]
        for row in saved_rows
    )
    assert [row["is_error"] for row in saved_rows] == [row["is_error"] for row in entropy]
    assert (output_dir / f"{stem}_comparison.csv").is_file()
    assert (output_dir / f"{stem}_comparison.json").is_file()
    assert (output_dir / f"{stem}_risk_coverage.svg").is_file()
