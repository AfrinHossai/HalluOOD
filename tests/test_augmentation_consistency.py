from __future__ import annotations

import json

from PIL import Image, ImageDraw

from halluood.model import MockVLMAdapter
from scripts.evaluate_augmentation_consistency import run_augmentation_consistency


class ScriptedMockAdapter(MockVLMAdapter):
    def __init__(self, answers):
        super().__init__(answer="unused")
        self._answers = iter(answers)
        self.input_sizes = []

    def generate_answer(self, image, question):
        super().generate_answer(image, question)
        if isinstance(image, Image.Image):
            self.input_sizes.append(image.size)
        else:
            with Image.open(image) as opened:
                self.input_sizes.append(opened.size)
        return next(self._answers)


def test_one_question_run_uses_mock_predictions_and_preserves_baseline_file(tmp_path):
    image_dir = tmp_path / "images"
    annotations_dir = tmp_path / "processed"
    image_dir.mkdir()
    annotations_dir.mkdir()
    image_filename = "COCO_val2014_000000396068.jpg"
    image_path = image_dir / image_filename
    image = Image.new("RGB", (200, 160), color=(70, 100, 130))
    ImageDraw.Draw(image).ellipse((85, 65, 115, 95), fill=(240, 30, 20))
    image.save(image_path)
    image.close()

    question = "Is there a remote in the image?"
    annotation = {
        "question_id": 2813,
        "image": image_filename,
        "text": question,
        "label": "yes",
        "category": "random",
    }
    (annotations_dir / "coco_pope_random_pilot.jsonl").write_text(
        json.dumps(annotation) + "\n",
        encoding="utf-8",
    )
    baseline_row = {
        "question_id": 2813,
        "image_id": "000000396068",
        "image_filename": image_filename,
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
    baseline_path = tmp_path / "baseline.jsonl"
    original_baseline_bytes = (json.dumps(baseline_row) + "\n").encode()
    baseline_path.write_bytes(original_baseline_bytes)
    output_path = tmp_path / "results" / "augmentation.json"
    adapter = ScriptedMockAdapter(
        [
            "Yes, there is a remote.",
            "Yes.",
            "No.",
            "Yes.",
            "No.",
            "Yes.",
            "Yes.",
        ]
    )

    result = run_augmentation_consistency(
        baseline_results_file=baseline_path,
        annotations_dir=annotations_dir,
        image_dir=image_dir,
        output_file=output_path,
        question_id="2813",
        augmentation_seed=31,
        adapter=adapter,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["original_prediction_matches_baseline"] is True
    assert result["uncertainty_score"]["uncertainty_score"] == 2 / 6
    assert result["uncertainty_score"]["disagreement_count"] == 2
    assert result["augmentation"]["random_seed"] == 31
    assert result["augmentation"]["model_call_count"] == 7
    assert len(result["augmentation"]["views"]) == 6
    assert len(adapter.calls) == len(adapter.input_sizes) == 7
    assert all(size == (200, 160) for size in adapter.input_sizes)
    assert (
        saved["baseline_prediction"]["source_file_prediction"]
        == baseline_row["raw_prediction"]
    )
    assert baseline_path.read_bytes() == original_baseline_bytes
