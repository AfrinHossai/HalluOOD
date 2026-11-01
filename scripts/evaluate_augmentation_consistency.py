"""Run one POPE image through deterministic mild augmentations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))

from halluood.augmentations import AugmentationError, create_mild_augmentations
from halluood.data import PopeDataError, PopeRecord, load_pope_records
from halluood.embeddings import normalize_coco_image_id
from halluood.model import (
    GenerationSettings,
    ModelAdapterError,
    TransformersVLMAdapter,
    VLMAdapter,
)
from halluood.uncertainty import augmentation_consistency_score
from scripts.run_baseline import CATEGORIES, normalize_prediction

DEFAULT_BASELINE_RESULTS = Path("data/results/pope_baseline_smollm_500m_30_seed_2027.jsonl")
DEFAULT_ANNOTATIONS_DIR = Path("data/processed")
DEFAULT_IMAGE_DIR = Path("data/images/val2014")
DEFAULT_OUTPUT = Path("data/results/augmentation_consistency_q2813_seed_2027.json")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "halluood"
DEFAULT_QUESTION_ID = "2813"
DEFAULT_SEED = 2027


class AugmentationEvaluationError(ValueError):
    """Raised when a baseline row cannot be evaluated consistently."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AugmentationEvaluationError(f"could not read baseline results {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AugmentationEvaluationError(
                f"invalid JSON in {path} on line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise AugmentationEvaluationError(
                f"expected an object in {path} on line {line_number}"
            )
        rows.append(row)
    if not rows:
        raise AugmentationEvaluationError(f"baseline results contain no rows: {path}")
    return rows


def _select_baseline_row(rows: list[dict[str, Any]], question_id: str) -> dict[str, Any]:
    matches = [row for row in rows if str(row.get("question_id")) == question_id]
    if not matches:
        raise AugmentationEvaluationError(
            f"question_id {question_id!r} was not found in the baseline results"
        )
    if len(matches) != 1:
        raise AugmentationEvaluationError(
            f"question_id {question_id!r} occurs {len(matches)} times in baseline results"
        )
    return matches[0]


def _select_annotation(
    annotations_dir: Path,
    baseline_row: dict[str, Any],
) -> PopeRecord:
    category = baseline_row.get("category")
    if category not in CATEGORIES:
        raise AugmentationEvaluationError(f"unsupported POPE category: {category!r}")
    try:
        records = load_pope_records(annotations_dir / f"coco_pope_{category}_pilot.jsonl")
    except PopeDataError as exc:
        raise AugmentationEvaluationError(f"could not load POPE annotations: {exc}") from exc
    matches = [
        record
        for record in records
        if str(record.question_id) == str(baseline_row["question_id"])
    ]
    if len(matches) != 1:
        raise AugmentationEvaluationError(
            f"question_id {baseline_row['question_id']!r} occurs {len(matches)} times "
            f"in the {category} annotation file"
        )
    record = matches[0]
    if (
        record.image != baseline_row.get("image_filename")
        or record.text != baseline_row.get("question")
        or record.normalized_label != baseline_row.get("normalized_ground_truth_label")
    ):
        raise AugmentationEvaluationError(
            "baseline question, image, or ground-truth label differs from the POPE annotation"
        )
    try:
        image_id = normalize_coco_image_id(record.image)
        baseline_image_id = normalize_coco_image_id(str(baseline_row["image_id"]))
    except (KeyError, ValueError) as exc:
        raise AugmentationEvaluationError(f"invalid POPE image identifier: {exc}") from exc
    if image_id != baseline_image_id:
        raise AugmentationEvaluationError("baseline image ID does not match its POPE filename")
    return record


def _generation_settings(row: dict[str, Any]) -> GenerationSettings:
    value = row.get("generation_settings")
    if not isinstance(value, dict):
        raise AugmentationEvaluationError("baseline row has no generation_settings object")
    try:
        return GenerationSettings(
            max_new_tokens=value["max_new_tokens"],
            seed=value["seed"],
            num_beams=value["num_beams"],
            do_sample=value["do_sample"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AugmentationEvaluationError(f"invalid baseline generation settings: {exc}") from exc


def _verify_model_settings(
    baseline_row: dict[str, Any],
    adapter: VLMAdapter,
) -> None:
    metadata = adapter.metadata.to_dict()
    expected_values = {
        "model_id": baseline_row.get("model_identifier"),
        "resolved_revision": baseline_row.get("model_revision"),
        "device": baseline_row.get("device"),
        "precision": baseline_row.get("precision"),
        "torch_version": baseline_row.get("torch_version"),
        "transformers_version": baseline_row.get("transformers_version"),
        "generation": baseline_row.get("generation_settings"),
    }
    actual_values = {
        "model_id": metadata.get("model_id"),
        "resolved_revision": metadata.get("resolved_revision")
        or metadata.get("requested_revision"),
        "device": metadata.get("device"),
        "precision": metadata.get("precision"),
        "torch_version": metadata.get("torch_version"),
        "transformers_version": metadata.get("transformers_version"),
        "generation": metadata.get("generation"),
    }
    for field, expected in expected_values.items():
        if expected is not None and actual_values[field] != expected:
            raise AugmentationEvaluationError(
                f"loaded adapter {field} differs from the baseline metadata"
            )


def _synchronize_mps(device: str) -> None:
    if device == "mps":
        import torch

        torch.mps.synchronize()


def _run_prediction(
    adapter: VLMAdapter,
    image: Image.Image | Path,
    question: str,
) -> dict[str, Any]:
    device = adapter.metadata.device
    _synchronize_mps(device)
    start = time.perf_counter()
    raw_prediction = adapter.generate_answer(image, question)
    _synchronize_mps(device)
    elapsed = time.perf_counter() - start
    normalized_prediction, status = normalize_prediction(raw_prediction)
    return {
        "raw_prediction": raw_prediction,
        "normalized_prediction": normalized_prediction,
        "prediction_status": status,
        "inference_time_seconds": elapsed,
    }


def evaluate_augmentation_pair(
    *,
    baseline_row: dict[str, Any],
    annotation: PopeRecord,
    image_path: str | Path,
    adapter: VLMAdapter,
    augmentation_seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Re-run one original image and score its answer stability over mild views."""

    if str(baseline_row.get("question_id")) != str(annotation.question_id):
        raise AugmentationEvaluationError("baseline and annotation question IDs do not match")
    if baseline_row.get("image_filename") != annotation.image:
        raise AugmentationEvaluationError("baseline and annotation image filenames do not match")
    _verify_model_settings(baseline_row, adapter)

    image_path = Path(image_path)
    try:
        with Image.open(image_path) as opened:
            opened.load()
            original_size = list(opened.size)
            views = create_mild_augmentations(opened, random_seed=augmentation_seed)
    except (OSError, UnidentifiedImageError, AugmentationError) as exc:
        raise AugmentationEvaluationError(f"could not prepare image augmentations: {exc}") from exc

    try:
        original_prediction = _run_prediction(adapter, image_path, annotation.text)
        view_predictions = []
        for view in views:
            prediction = _run_prediction(adapter, view.image, annotation.text)
            view_predictions.append(
                {
                    "view": view.name,
                    "transform_parameters": view.parameters,
                    "image_width": view.image.width,
                    "image_height": view.image.height,
                    **prediction,
                }
            )
    except Exception as exc:
        raise AugmentationEvaluationError(
            f"inference failed for question {annotation.question_id!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        for view in views:
            view.image.close()

    score = augmentation_consistency_score(
        original_prediction["normalized_prediction"],
        [view["normalized_prediction"] for view in view_predictions],
    )
    baseline_normalized = baseline_row.get("normalized_prediction")
    baseline_status = baseline_row.get("prediction_status")
    baseline_prediction_match = (
        baseline_status in {"parsed", "ambiguous"}
        and baseline_normalized == original_prediction["normalized_prediction"]
        and baseline_status == original_prediction["prediction_status"]
    )
    return {
        "question_id": annotation.question_id,
        "image_id": normalize_coco_image_id(annotation.image),
        "image_filename": annotation.image,
        "image_width": original_size[0],
        "image_height": original_size[1],
        "category": annotation.category or baseline_row["category"],
        "question": annotation.text,
        "ground_truth_label": annotation.label,
        "normalized_ground_truth_label": annotation.normalized_label,
        "baseline_prediction": {
            "source_file_prediction": baseline_row.get("raw_prediction"),
            "normalized_prediction": baseline_normalized,
            "prediction_status": baseline_status,
            "source_results_file": baseline_row.get("_source_results_file"),
        },
        "original_prediction": original_prediction,
        "original_prediction_matches_baseline": baseline_prediction_match,
        "augmentation": {
            "random_seed": augmentation_seed,
            "random_generator": "NumPy PCG64",
            "shift_fraction": 0.01,
            "views": view_predictions,
            "model_call_count": 1 + len(view_predictions),
        },
        "uncertainty_score": score.to_dict(),
        "model_metadata": adapter.metadata.to_dict(),
    }


def run_augmentation_consistency(
    *,
    baseline_results_file: str | Path = DEFAULT_BASELINE_RESULTS,
    annotations_dir: str | Path = DEFAULT_ANNOTATIONS_DIR,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    output_file: str | Path = DEFAULT_OUTPUT,
    question_id: str = DEFAULT_QUESTION_ID,
    augmentation_seed: int = DEFAULT_SEED,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    device: str | None = None,
    adapter: VLMAdapter | None = None,
) -> dict[str, Any]:
    """Evaluate exactly one selected baseline question; never modify baseline files."""

    baseline_path = Path(baseline_results_file)
    baseline_row = _select_baseline_row(_read_jsonl(baseline_path), str(question_id))
    baseline_row["_source_results_file"] = str(baseline_path)
    annotation = _select_annotation(Path(annotations_dir), baseline_row)
    image_path = Path(image_dir) / annotation.image
    if not image_path.is_file():
        raise FileNotFoundError(f"COCO image is missing: {image_path}")

    generation = _generation_settings(baseline_row)
    if adapter is None:
        resolved_device = None if device in {None, "auto"} else device
        adapter = TransformersVLMAdapter.from_pretrained(
            model_id=str(baseline_row["model_identifier"]),
            revision=str(baseline_row["model_revision"]),
            device=resolved_device,
            generation=generation,
            cache_dir=cache_dir,
        )
        _verify_model_settings(baseline_row, adapter)

    result = evaluate_augmentation_pair(
        baseline_row=baseline_row,
        annotation=annotation,
        image_path=image_path,
        adapter=adapter,
        augmentation_seed=augmentation_seed,
    )
    result["baseline_results_file"] = str(baseline_path)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure answer disagreement between one saved POPE baseline case and "
            "mild deterministic image augmentations."
        )
    )
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--question-id", default=DEFAULT_QUESTION_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_augmentation_consistency(
            baseline_results_file=args.baseline_results,
            annotations_dir=args.annotations_dir,
            image_dir=args.image_dir,
            output_file=args.output,
            question_id=args.question_id,
            augmentation_seed=args.seed,
            cache_dir=args.cache_dir,
            device=args.device,
        )
    except (AugmentationEvaluationError, ModelAdapterError, OSError, ValueError) as exc:
        print(f"augmentation consistency failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
