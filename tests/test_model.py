import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from halluood.model import (
    GenerationSettings,
    MockVLMAdapter,
    ModelAdapterError,
    TransformersVLMAdapter,
)


def test_transformers_adapter_sends_local_image_and_uses_greedy_decoding():
    observed = {}

    def fake_pipeline(**kwargs):
        observed.update(kwargs)
        supplied_image = kwargs["text"][0]["content"][0]["image"]
        assert supplied_image.mode == "RGB"
        assert supplied_image.size == (4, 3)
        return [
            {
                "generated_text": [
                    {"role": "assistant", "content": [{"type": "text", "text": "MOCK_ANSWER"}]}
                ]
            }
        ]

    adapter = TransformersVLMAdapter(
        fake_pipeline,
        model_id="local/test-vlm",
        requested_revision="0123456789abcdef",
        resolved_revision="0123456789abcdef",
        device="mps",
        precision="torch.float16",
        transformers_version="test-transformers",
        torch_version="test-torch",
        generation=GenerationSettings(max_new_tokens=16, seed=42),
    )
    image = Image.new("RGBA", (4, 3), color=(255, 0, 0, 255))

    answer = adapter.generate_answer(image, "Is there a bicycle?")

    assert answer == "MOCK_ANSWER"
    content = observed["text"][0]["content"]
    assert content[0]["type"] == "image"
    assert isinstance(content[0]["image"], Image.Image)
    assert content[1] == {"type": "text", "text": "Is there a bicycle?"}
    assert "images" not in observed
    assert observed["max_new_tokens"] == 16
    assert observed["generate_kwargs"] == {"do_sample": False, "num_beams": 1}
    assert observed["return_full_text"] is False
    assert image.mode == "RGBA"  # the adapter did not mutate the caller's image


def test_transformers_adapter_accepts_image_path_and_plain_text_output(tmp_path):
    image_path = tmp_path / "pilot.jpg"
    Image.new("RGB", (2, 2), color=(20, 30, 40)).save(image_path)

    observed = {}

    def fake_pipeline(**kwargs):
        image = kwargs["text"][0]["content"][0]["image"]
        observed["image_mode"] = image.mode
        observed["image_size"] = image.size
        return [{"generated_text": "MOCK_PATH_ANSWER"}]

    fake_pipeline = Mock(side_effect=fake_pipeline)
    adapter = TransformersVLMAdapter(
        fake_pipeline,
        model_id="local/test-vlm",
        requested_revision="test",
    )

    answer = adapter.generate_answer(image_path, "What is in the image?")

    assert answer == "MOCK_PATH_ANSWER"
    assert observed == {"image_mode": "RGB", "image_size": (2, 2)}


def test_transformers_adapter_validates_question_and_image(tmp_path):
    adapter = TransformersVLMAdapter(
        Mock(), model_id="local/test-vlm", requested_revision="test"
    )

    with pytest.raises(ValueError, match="non-empty"):
        adapter.generate_answer(Image.new("RGB", (1, 1)), "  ")
    with pytest.raises(ValueError, match="could not read image"):
        adapter.generate_answer(tmp_path / "missing.jpg", "What is visible?")


def test_model_metadata_is_serializable_and_records_versions():
    adapter = TransformersVLMAdapter(
        Mock(),
        model_id="local/test-vlm",
        requested_revision="requested-revision",
        resolved_revision="resolved-revision",
        device="mps",
        precision="torch.float16",
        transformers_version="4.test",
        torch_version="2.test",
        generation=GenerationSettings(max_new_tokens=24, seed=73),
    )

    metadata = adapter.metadata.to_dict()
    encoded = json.dumps(metadata)

    assert "local/test-vlm" in encoded
    assert metadata["resolved_revision"] == "resolved-revision"
    assert metadata["device"] == "mps"
    assert metadata["generation"] == {
        "max_new_tokens": 24,
        "seed": 73,
        "num_beams": 1,
        "do_sample": False,
    }


def test_mock_adapter_returns_configured_test_answer_and_records_call():
    adapter = MockVLMAdapter(answer="MOCK_ONLY")
    image_path = Path("test-only.jpg")

    answer = adapter.generate_answer(image_path, "Is there a dog?")

    assert answer == "MOCK_ONLY"
    assert adapter.calls == [{"image": "test-only.jpg", "question": "Is there a dog?"}]
    assert adapter.metadata.backend == "mock"
    assert adapter.metadata.to_dict()["resolved_revision"] == "test-only"


def test_generation_settings_reject_sampling_and_invalid_values():
    with pytest.raises(ValueError, match="greedy"):
        GenerationSettings(do_sample=True)
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationSettings(max_new_tokens=0)
    with pytest.raises(ValueError, match="seed"):
        GenerationSettings(seed=-1)
    with pytest.raises(ValueError, match="num_beams"):
        GenerationSettings(num_beams=0)


def test_extracting_unrecognized_pipeline_output_raises_adapter_error():
    adapter = TransformersVLMAdapter(
        Mock(return_value=[{"unexpected": "shape"}]),
        model_id="local/test-vlm",
        requested_revision="test",
    )

    with pytest.raises(ModelAdapterError, match="could not extract"):
        adapter.generate_answer(Image.new("RGB", (1, 1)), "What is visible?")
