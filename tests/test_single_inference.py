import json
import sys
import types
from unittest.mock import Mock

import pytest
from PIL import Image

from halluood.model import TransformersVLMAdapter
from scripts.run_single_inference import run_single_inference


def _write_pilot(tmp_path, *, question_id=49):
    input_file = tmp_path / "pilot.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "question_id": question_id,
                "image": "COCO_val2014_000000000001.jpg",
                "text": "Is there a backpack in the image?",
                "label": "yes",
                "category": "random",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (5, 4), color=(30, 90, 140)).save(
        image_dir / "COCO_val2014_000000000001.jpg"
    )
    return input_file, image_dir


def test_single_inference_runs_exactly_one_selected_pair_with_mock(
    tmp_path, capsys
):
    input_file, image_dir = _write_pilot(tmp_path)
    adapter = Mock(answer="Yes, a backpack is visible.")
    adapter.metadata = TransformersVLMAdapter(
        Mock(),
        model_id="test/adapter",
        requested_revision="test-revision",
        device="cpu",
        transformers_version="test-transformers",
        torch_version="test-torch",
    ).metadata
    adapter.generate_answer.return_value = "Yes, a backpack is visible."

    result = run_single_inference(
        input_file=input_file,
        image_dir=image_dir,
        question_id="49",
        cache_dir=tmp_path / "model-cache",
        adapter=adapter,
    )

    adapter.generate_answer.assert_called_once_with(
        image_dir / "COCO_val2014_000000000001.jpg",
        "Is there a backpack in the image?",
    )
    assert result["ground_truth_answer"] == "yes"
    assert result["generated_response"] == "Yes, a backpack is visible."
    assert result["model_identifier"] == "test/adapter"
    assert result["device"] == "cpu"
    assert result["inference_time_seconds"] >= 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["question_id"] == 49
    assert printed["question"] == "Is there a backpack in the image?"


def test_single_inference_rejects_missing_or_duplicate_question(tmp_path):
    input_file, image_dir = _write_pilot(tmp_path)
    with pytest.raises(ValueError, match="was not found"):
        run_single_inference(
            input_file=input_file,
            image_dir=image_dir,
            question_id="no-such-id",
            adapter=Mock(),
        )

    duplicate = input_file.read_text(encoding="utf-8")
    input_file.write_text(duplicate + duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="expected exactly one row"):
        run_single_inference(
            input_file=input_file,
            image_dir=image_dir,
            question_id="49",
            adapter=Mock(),
        )


def test_transformers_loader_uses_requested_cache_and_mps_dtype(monkeypatch, tmp_path):
    calls = {}
    seeded = []

    class FakePipeline:
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(_commit_hash="resolved-commit"),
            dtype="torch.float16",
        )

        def __call__(self, **kwargs):
            calls["inference"] = kwargs
            return [{"generated_text": "no"}]

    def fake_pipeline(task, **kwargs):
        calls["load"] = {"task": task, **kwargs}
        return FakePipeline()

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.test"
    fake_torch.float16 = "float16"
    fake_torch.manual_seed = seeded.append
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: True)
    )
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    cache_dir = tmp_path / "hf-cache"
    adapter = TransformersVLMAdapter.from_pretrained(cache_dir=cache_dir)
    answer = adapter.generate_answer(Image.new("RGB", (2, 2)), "Any object?")

    assert answer == "no"
    assert calls["load"] == {
        "task": "image-text-to-text",
        "model": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "revision": "a7da5b986cb59b408707209984f360a5f4ad7e47",
        "device": "mps",
        "model_kwargs": {"cache_dir": str(cache_dir)},
        "dtype": "float16",
    }
    assert adapter.metadata.resolved_revision == "resolved-commit"
    assert adapter.metadata.device == "mps"
    assert seeded == [2027]
    assert sys.modules["transformers"] is fake_transformers
