from __future__ import annotations

import hashlib

import numpy as np
import pytest

from halluood.embedding_evaluation import (
    EmbeddingEvaluationError,
    compare_embedding_and_entropy,
    join_paired_rows,
    render_comparison_risk_coverage_svg,
)
from halluood.embeddings import reference_set_identifier, select_disjoint_reference_images
from scripts.evaluate_embedding_ood import _verify_reference_bank


def _paired_sources():
    entropy_rows = []
    baseline_rows = []
    embedding_rows = []
    outcomes = (
        ("yes", "yes", "parsed", 0.1, 0.8, 0.1),
        ("no", "no", "parsed", 0.2, 0.7, 0.3),
        ("yes", "yes", "parsed", 0.3, 0.6, 0.2),
        ("no", "no", "parsed", 0.4, 0.5, 0.6),
        ("yes", "no", "parsed", 0.9, 0.4, 0.7),
        ("no", None, "ambiguous", 0.8, 0.3, 0.8),
    )
    for index, (gold, prediction, status, entropy, cosine, nearest) in enumerate(
        outcomes, start=1
    ):
        question_id = index + 100
        image_id = f"{index:012d}"
        category = ("random", "popular", "adversarial")[(index - 1) % 3]
        raw_prediction = prediction.title() if prediction else "I cannot tell."
        correct = status == "parsed" and prediction == gold
        shared = {
            "question_id": question_id,
            "image_id": image_id,
            "category": category,
            "question": f"Is item {index} present?",
            "normalized_ground_truth_label": gold,
            "normalized_prediction": prediction,
            "prediction_status": status,
            "raw_prediction": raw_prediction,
            "model_identifier": "test/model",
            "model_revision": "test-revision",
            "device": "cpu",
            "precision": "float32",
            "torch_version": "test-torch",
            "transformers_version": "test-transformers",
            "seed": 2027,
            "generation_settings": {
                "do_sample": False,
                "max_new_tokens": 32,
                "num_beams": 1,
                "seed": 2027,
            },
        }
        entropy_rows.append(
            {
                **shared,
                "normalized_entropy": entropy,
                "is_correct": correct,
                "is_error": not correct,
                "reference_prediction_match": True,
            }
        )
        baseline_rows.append(shared)
        embedding_rows.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "category": category,
                "cosine_distance": cosine,
                "nearest_neighbor_distance": nearest,
                "reference_set_id": "test-reference-id",
            }
        )
    return entropy_rows, baseline_rows, embedding_rows


def test_join_paired_rows_joins_by_question_and_image_ids_and_counts_ambiguity():
    entropy, baseline, embeddings = _paired_sources()

    joined = join_paired_rows(
        entropy[::-1], baseline[::2] + baseline[1::2], embeddings[::-1], expected_count=6
    )

    assert [(row["question_id"], row["image_id"]) for row in joined] == [
        (101, "000000000001"),
        (102, "000000000002"),
        (103, "000000000003"),
        (104, "000000000004"),
        (105, "000000000005"),
        (106, "000000000006"),
    ]
    assert sum(row["is_error"] for row in joined) == 2
    assert joined[-1]["prediction_status"] == "ambiguous"


def test_join_rejects_mismatched_identity_prediction_and_correctness_labels():
    entropy, baseline, embeddings = _paired_sources()
    embeddings[0] = {**embeddings[0], "image_id": "000000000099"}
    with pytest.raises(EmbeddingEvaluationError, match="identical question ID/image ID"):
        join_paired_rows(entropy, baseline, embeddings, expected_count=6)

    entropy, baseline, embeddings = _paired_sources()
    entropy[0] = {**entropy[0], "reference_prediction_match": False}
    with pytest.raises(EmbeddingEvaluationError, match="differs from the baseline"):
        join_paired_rows(entropy, baseline, embeddings, expected_count=6)

    entropy, baseline, embeddings = _paired_sources()
    baseline[0] = {**baseline[0], "model_revision": "different-revision"}
    with pytest.raises(EmbeddingEvaluationError, match="model_revision differ"):
        join_paired_rows(entropy, baseline, embeddings, expected_count=6)

    entropy, baseline, embeddings = _paired_sources()
    entropy[0] = {**entropy[0], "is_error": True}
    with pytest.raises(EmbeddingEvaluationError, match="correctness label"):
        join_paired_rows(entropy, baseline, embeddings, expected_count=6)


def test_comparison_reports_each_score_curves_and_paired_incremental_diagnostic():
    entropy, baseline, embeddings = _paired_sources()
    joined = join_paired_rows(entropy, baseline, embeddings, expected_count=6)

    result = compare_embedding_and_entropy(
        joined,
        bootstrap_replicates=100,
        random_seed=19,
    )

    assert result["question_count"] == 6
    assert result["incorrect_count"] == 2
    scores = {row["score"]: row for row in result["score_comparison_table"]}
    assert set(scores) == {
        "predictive_entropy",
        "cosine_distance",
        "nearest_neighbor_distance",
    }
    assert scores["predictive_entropy"]["auroc"] == pytest.approx(1.0)
    assert scores["cosine_distance"]["auroc"] == pytest.approx(0.0)
    assert scores["nearest_neighbor_distance"]["auroc"] == pytest.approx(1.0)
    assert all(
        result["risk_coverage_curves"][name][0]["risk"] is None
        for name in scores
    )
    fusion = result["incremental_rank_fusion_check"]["results"]
    assert set(fusion) == {
        "entropy_plus_cosine_distance_rank_fusion",
        "entropy_plus_nearest_neighbor_distance_rank_fusion",
    }
    assert fusion["entropy_plus_nearest_neighbor_distance_rank_fusion"][
        "delta_auroc_vs_entropy"
    ]["point_difference"] == pytest.approx(0.0)
    assert fusion["entropy_plus_cosine_distance_rank_fusion"][
        "delta_auroc_vs_entropy"
    ]["bootstrap_seed"] == 19

    svg = render_comparison_risk_coverage_svg(
        result["risk_coverage_curves"],
        overall_error_rate=result["incorrect_prevalence"],
        title="Test comparison",
    )
    assert '<svg xmlns="http://www.w3.org/2000/svg"' in svg
    assert "predictive entropy" in svg
    assert "nearest neighbor distance" in svg


def test_reference_verifier_checks_ids_seed_disjointness_hashes_and_embeddings(tmp_path):
    evaluation_ids = {f"{index:012d}" for index in range(1, 31)}
    for image_id in evaluation_ids:
        (tmp_path / f"COCO_val2014_{image_id}.jpg").write_bytes(image_id.encode("ascii"))
    for index in range(1001, 1071):
        image_id = f"{index:012d}"
        (tmp_path / f"COCO_val2014_{image_id}.jpg").write_bytes(image_id.encode("ascii"))
    expected_references = select_disjoint_reference_images(
        list(tmp_path.glob("COCO_val2014_*.jpg")),
        evaluation_ids,
        sample_size=50,
        random_seed=2027,
    )
    reference_ids = [image_id for image_id, _ in expected_references]
    rng = np.random.Generator(np.random.PCG64(7))
    embeddings = rng.normal(size=(50, 960)).astype(np.float32)
    image_hashes = {}
    filenames = []
    for image_id in reference_ids:
        filename = f"COCO_val2014_{image_id}.jpg"
        file_path = tmp_path / filename
        image_hashes[image_id] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        filenames.append(filename)

    bank_path = tmp_path / "bank.npz"
    np.savez_compressed(
        bank_path,
        image_ids=np.asarray(reference_ids),
        embeddings=embeddings,
    )
    encoder = {
        "model_identifier": "test/encoder",
        "model_revision": "revision",
    }
    manifest = {
        "reference_count": 50,
        "reference_image_ids": reference_ids,
        "reference_filenames": filenames,
        "reference_image_sha256": image_hashes,
        "reference_selection": {
            "random_seed": 2027,
            "candidate_count_after_exclusion": 70,
        },
        "evaluation_image_ids_excluded": sorted(evaluation_ids),
        "reference_evaluation_overlap": [],
        "feature_storage": {
            "sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest()
        },
        "encoder": encoder,
        "reference_set_id": reference_set_identifier(
            reference_ids,
            embeddings,
            encoder_identifier=encoder["model_identifier"],
            encoder_revision=encoder["model_revision"],
            random_seed=2027,
        ),
    }

    verified_ids, verified_embeddings = _verify_reference_bank(
        bank_path,
        manifest,
        evaluation_ids,
        tmp_path,
    )

    assert verified_ids == reference_ids
    np.testing.assert_array_equal(verified_embeddings, embeddings)
    manifest["reference_evaluation_overlap"] = [reference_ids[0]]
    with pytest.raises(EmbeddingEvaluationError, match="overlaps evaluation images"):
        _verify_reference_bank(bank_path, manifest, evaluation_ids, tmp_path)
