from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from halluood.model import ModelAdapterError, TransformersVLMAdapter
from halluood.uncertainty import (
    CandidateSequenceLogProbability,
    UncertaintyScoreError,
    binary_entropy_from_log_scores,
    entropy_from_logits,
    entropy_from_probabilities,
    score_binary_answer_entropy,
)


def test_balanced_distribution_has_maximum_binary_entropy():
    balanced = entropy_from_probabilities([0.5, 0.5])
    concentrated = entropy_from_probabilities([0.999, 0.001])

    assert balanced == pytest.approx(math.log(2))
    assert balanced > concentrated


def test_concentrated_distribution_has_low_entropy():
    assert entropy_from_probabilities([1.0, 0.0]) == pytest.approx(0.0)
    assert binary_entropy_from_log_scores(0.0, -1000.0)["normalized_entropy"] == pytest.approx(
        0.0
    )


def test_entropy_from_logits_is_stable_for_large_finite_values():
    entropy = entropy_from_logits(np.array([1e308, 1e308]))

    assert entropy == pytest.approx(math.log(2))


@pytest.mark.parametrize(
    "probabilities",
    ([], [0.0, 0.0], [-0.1, 1.1], [float("nan"), 1.0], [float("inf"), 0.0], [[0.5, 0.5]]),
)
def test_invalid_probabilities_are_rejected(probabilities):
    with pytest.raises(UncertaintyScoreError):
        entropy_from_probabilities(probabilities)


def test_binary_score_normalizes_candidate_sequence_log_probabilities():
    distribution = binary_entropy_from_log_scores(-10.0, -10.0)

    assert distribution["yes_probability"] == pytest.approx(0.5)
    assert distribution["no_probability"] == pytest.approx(0.5)
    assert distribution["entropy_nats"] == pytest.approx(math.log(2))
    assert distribution["normalized_entropy"] == pytest.approx(1.0)


def test_binary_score_uses_adapter_candidate_log_probabilities():
    class FakeAdapter:
        def score_candidate_answers(self, image, question, candidate_answers):
            assert (image, question, candidate_answers) == (
                "image.jpg",
                "Is there a dog?",
                ("Yes", "No"),
            )
            return {
                "Yes": CandidateSequenceLogProbability("Yes", (11, 12), -2.0),
                "No": CandidateSequenceLogProbability("No", (13,), -2.0),
            }

    score = score_binary_answer_entropy(FakeAdapter(), "image.jpg", "Is there a dog?")
    as_dict = score.to_dict()

    assert score.normalized_entropy == pytest.approx(1.0)
    assert as_dict["entropy_type"] == "binary_yes_no_answer_entropy"
    assert as_dict["candidate_token_counts"] == {"yes": 2, "no": 1}


def test_transformers_adapter_scores_multitoken_candidates_from_logits():
    torch = pytest.importorskip("torch")

    class FakeBatch(dict):
        def to(self, *args, **kwargs):
            return self

    class FakeTokenizer:
        all_special_ids = [99]

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def apply_chat_template(
            self,
            messages,
            *,
            add_generation_prompt=False,
            tokenize,
            return_dict,
            return_tensors,
        ):
            assert tokenize and return_dict and return_tensors == "pt"
            prefix = [1, 2, 3]
            if add_generation_prompt:
                ids = prefix
            else:
                answer = messages[-1]["content"][0]["text"]
                answer_ids = {"Yes": [4, 5], "No": [6]}[answer]
                ids = prefix + answer_ids + [99, 98]
            return FakeBatch(input_ids=torch.tensor([ids], dtype=torch.long))

    class FakeModel:
        device = "cpu"
        dtype = torch.float32

        def __call__(self, *, input_ids):
            ids = input_ids
            logits = torch.zeros((1, ids.shape[1], 100), dtype=torch.float32)
            for position in range(2, ids.shape[1] - 1):
                target_id = int(ids[0, position + 1])
                logits[0, position, target_id] = 4.0
            return SimpleNamespace(logits=logits)

    class FakePipeline:
        processor = FakeProcessor()
        model = FakeModel()

        def __call__(self, **kwargs):
            return "unused"

    adapter = TransformersVLMAdapter(
        FakePipeline(),
        model_id="test/model",
        requested_revision="test",
        torch_module=torch,
    )

    scores = adapter.score_candidate_answers(
        Image.new("RGB", (2, 2)), "Is there a dog?", ("Yes", "No")
    )

    assert scores["Yes"].token_ids == (4, 5)
    assert scores["No"].token_ids == (6,)
    assert scores["Yes"].log_probability < 0
    assert scores["No"].log_probability < 0


def test_transformers_adapter_rejects_candidate_prefix_mismatch():
    torch = pytest.importorskip("torch")

    class FakeBatch(dict):
        def to(self, *args, **kwargs):
            return self

    class FakeProcessor:
        tokenizer = SimpleNamespace(all_special_ids=[])

        def apply_chat_template(self, messages, *, add_generation_prompt=False, **kwargs):
            ids = [1, 2] if add_generation_prompt else [1, 8, 9]
            return FakeBatch(input_ids=torch.tensor([ids], dtype=torch.long))

    fake_pipeline = SimpleNamespace(
        processor=FakeProcessor(),
        model=SimpleNamespace(device="cpu", dtype=torch.float32),
    )
    adapter = TransformersVLMAdapter(
        lambda **kwargs: None,
        model_id="test/model",
        requested_revision="test",
        torch_module=torch,
    )
    adapter._pipeline = fake_pipeline

    with pytest.raises(ModelAdapterError, match="prefix"):
        adapter.score_candidate_answers(
            Image.new("RGB", (2, 2)), "Is there a dog?", ("Yes",)
        )
