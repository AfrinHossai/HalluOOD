from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from halluood.model import (
    GenerationSettings,
    MockVLMAdapter,
    ModelAdapterError,
    SamplingSettings,
    TransformersVLMAdapter,
)
from halluood.uncertainty import UncertaintyScoreError, answer_disagreement_score
from scripts.run_answer_disagreement import run_answer_disagreement


def test_answer_disagreement_uses_three_empirical_answer_proportions():
    score = answer_disagreement_score(["yes", "no", "ambiguous"])

    assert score.normalized_score == pytest.approx(1.0)
    assert score.entropy_nats == pytest.approx(math.log(3))
    assert score.generation_count == 3
    assert score.to_dict()["counts"] == {"yes": 1, "no": 1, "ambiguous": 1}
    assert score.to_dict()["proportions"] == {
        "yes": pytest.approx(1 / 3),
        "no": pytest.approx(1 / 3),
        "ambiguous": pytest.approx(1 / 3),
    }


def test_answer_disagreement_is_zero_for_unanimous_answers_and_counts_ambiguous():
    assert answer_disagreement_score(["yes", "yes", "yes"]).normalized_score == 0.0
    ambiguous = answer_disagreement_score(["ambiguous", "ambiguous"])
    assert ambiguous.normalized_score == 0.0
    assert ambiguous.to_dict()["counts"]["ambiguous"] == 2


@pytest.mark.parametrize(
    "answers",
    [[], ["yes", "maybe"], [None], [["yes"]]],
)
def test_answer_disagreement_rejects_empty_or_non_normalized_inputs(answers):
    with pytest.raises(UncertaintyScoreError):
        answer_disagreement_score(answers)


@pytest.mark.parametrize(
    "settings",
    [
        {"num_generations": 0},
        {"num_generations": True},
        {"temperature": 0.0},
        {"temperature": math.nan},
        {"top_p": 0.0},
        {"top_p": 1.01},
        {"seed": -1},
        {"max_new_tokens": 0},
    ],
)
def test_sampling_settings_reject_invalid_values(settings):
    with pytest.raises(ValueError):
        SamplingSettings(**settings)


def test_sampling_settings_records_distinct_seeds_for_each_response():
    settings = SamplingSettings(num_generations=3, seed=61, temperature=0.8, top_p=0.92)

    assert settings.generation_seeds == (61, 62, 63)
    assert settings.to_dict()["generation_seeds"] == [61, 62, 63]
    assert settings.to_dict()["temperature"] == 0.8
    assert settings.to_dict()["top_p"] == 0.92


def test_transformers_adapter_runs_sampled_decoding_and_records_seed_per_call():
    outputs = iter(["Yes.", "No.", "Maybe."])
    observed = []
    seeds = []

    def fake_pipeline(**kwargs):
        observed.append(kwargs)
        return [{"generated_text": next(outputs)}]

    torch = SimpleNamespace(manual_seed=seeds.append)
    adapter = TransformersVLMAdapter(
        fake_pipeline,
        model_id="local/test-vlm",
        requested_revision="rev-sampled",
        generation=GenerationSettings(seed=9),
        torch_module=torch,
    )
    settings = SamplingSettings(
        num_generations=3,
        temperature=0.65,
        top_p=0.88,
        seed=90,
        max_new_tokens=12,
    )

    answers = adapter.generate_sampled_answers(
        Image.new("RGBA", (3, 2)), "Is there a dog?", settings
    )

    assert answers == ["Yes.", "No.", "Maybe."]
    assert seeds == [90, 91, 92]
    assert len(observed) == 3
    assert all(call["max_new_tokens"] == 12 for call in observed)
    assert all(
        call["generate_kwargs"]
        == {
            "do_sample": True,
            "num_beams": 1,
            "temperature": 0.65,
            "top_p": 0.88,
        }
        for call in observed
    )


def test_transformers_adapter_aborts_on_sample_generation_failure():
    calls = 0

    def failing_pipeline(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic inference failure")
        return [{"generated_text": "Yes."}]

    adapter = TransformersVLMAdapter(
        failing_pipeline,
        model_id="local/test-vlm",
        requested_revision="rev-sampled",
    )

    with pytest.raises(
        ModelAdapterError,
        match="sample generation 2/3.*synthetic inference failure",
    ):
        adapter.generate_sampled_answers(
            Image.new("RGB", (2, 2)), "Is there a dog?", SamplingSettings(num_generations=3)
        )
    assert calls == 2


def test_mock_adapter_can_return_scripted_sampled_answers():
    adapter = MockVLMAdapter(sampled_answers=["Yes.", "No."])
    settings = SamplingSettings(num_generations=2, seed=4)

    answers = adapter.generate_sampled_answers("mock.jpg", "Any dog?", settings)

    assert answers == ["Yes.", "No."]
    assert adapter.sampled_calls[0]["sampling_settings"]["generation_seeds"] == [4, 5]


def test_single_question_runner_rejects_non_sampling_settings():
    with pytest.raises(ValueError, match="SamplingSettings"):
        run_answer_disagreement(settings=0)


def _write_smoke_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    annotations_dir = tmp_path / "processed"
    image_dir = tmp_path / "images"
    annotations_dir.mkdir()
    image_dir.mkdir()
    filename = "COCO_val2014_000000396068.jpg"
    question = "Is there a remote in the image?"
    annotation = {
        "question_id": 2813,
        "image": filename,
        "text": question,
        "label": "yes",
        "category": "random",
    }
    (annotations_dir / "coco_pope_random_pilot.jsonl").write_text(
        json.dumps(annotation) + "\n", encoding="utf-8"
    )
    Image.new("RGB", (12, 10), color=(80, 100, 120)).save(image_dir / filename)
    row = {
        "question_id": 2813,
        "image_id": "000000396068",
        "image_filename": filename,
        "category": "random",
        "question": question,
        "ground_truth_label": "yes",
        "normalized_ground_truth_label": "yes",
        "raw_prediction": "Yes, there is a remote in the image.",
        "normalized_prediction": "yes",
        "prediction_status": "parsed",
        "model_identifier": "halluood/mock-vlm",
        "model_revision": "test-only",
        "device": "cpu",
        "precision": None,
        "torch_version": None,
        "transformers_version": None,
        "seed": 2027,
        "generation_settings": {
            "max_new_tokens": 32,
            "seed": 2027,
            "num_beams": 1,
            "do_sample": False,
        },
    }
    baseline_file = tmp_path / "baseline.jsonl"
    baseline_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return baseline_file, annotations_dir, image_dir


def test_single_question_workflow_saves_samples_and_preserves_baseline(tmp_path):
    baseline_file, annotations_dir, image_dir = _write_smoke_inputs(tmp_path)
    baseline_bytes = baseline_file.read_bytes()
    output_file = tmp_path / "results" / "sampled.json"
    adapter = MockVLMAdapter(sampled_answers=["Yes.", "No.", "not sure", "No."])

    result = run_answer_disagreement(
        baseline_results_file=baseline_file,
        annotations_dir=annotations_dir,
        image_dir=image_dir,
        output_file=output_file,
        question_id="2813",
        settings=SamplingSettings(num_generations=4, temperature=0.7, top_p=0.9, seed=13),
        adapter=adapter,
    )

    saved = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(saved["sampled_answers"]) == 4
    assert [row["seed"] for row in saved["sampled_answers"]] == [13, 14, 15, 16]
    assert saved["answer_disagreement"]["counts"] == {
        "yes": 1,
        "no": 2,
        "ambiguous": 1,
    }
    assert result["answer_disagreement"]["uncertainty_score"] == pytest.approx(
        -(0.25 * math.log(0.25) * 2 + 0.5 * math.log(0.5)) / math.log(3)
    )
    assert saved["original_baseline_prediction"]["raw_answer"] == (
        "Yes, there is a remote in the image."
    )
    assert baseline_file.read_bytes() == baseline_bytes
    assert adapter.calls == []


def test_single_question_workflow_rejects_empty_or_failed_sample_outputs(tmp_path):
    baseline_file, annotations_dir, image_dir = _write_smoke_inputs(tmp_path)
    settings = SamplingSettings(num_generations=2)

    class EmptyAdapter(MockVLMAdapter):
        def generate_sampled_answers(self, image, question, settings):
            return []

    with pytest.raises(ValueError, match="expected 2"):
        run_answer_disagreement(
            baseline_results_file=baseline_file,
            annotations_dir=annotations_dir,
            image_dir=image_dir,
            output_file=tmp_path / "empty.json",
            settings=settings,
            adapter=EmptyAdapter(),
        )

    class FailedAdapter(MockVLMAdapter):
        def generate_sampled_answers(self, image, question, settings):
            raise RuntimeError("synthetic inference failure")

    failed_output = tmp_path / "failed.json"
    with pytest.raises(ValueError, match="synthetic inference failure"):
        run_answer_disagreement(
            baseline_results_file=baseline_file,
            annotations_dir=annotations_dir,
            image_dir=image_dir,
            output_file=failed_output,
            settings=settings,
            adapter=FailedAdapter(),
        )
    assert not failed_output.exists()
