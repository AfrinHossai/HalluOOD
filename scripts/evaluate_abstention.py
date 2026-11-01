"""Calibrate entropy abstention on image-disjoint POPE pilot subsets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from math import isfinite
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    sys.path.insert(1, str(project_root / "src"))

from halluood.data import PopeRecord, load_pope_records
from halluood.embeddings import normalize_coco_image_id
from halluood.model import (
    RECOMMENDED_MODEL_ID,
    RECOMMENDED_MODEL_REVISION,
    GenerationSettings,
    TransformersVLMAdapter,
)
from halluood.selective import (
    SelectivePredictionError,
    calibrate_abstention_threshold,
    evaluate_selective_predictions,
)
from halluood.uncertainty import score_binary_answer_entropy
from scripts.evaluate_entropy import DEFAULT_OUTPUT as DEFAULT_EVALUATION_RESULTS
from scripts.run_baseline import CATEGORIES, select_balanced_questions

DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_IMAGE_DIR = Path("data/images/val2014")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "halluood"
DEFAULT_OUTPUT_DIR = Path("data/results/abstention")
DEFAULT_TARGET_COVERAGES = (0.5, 0.7, 0.8, 0.9)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SelectivePredictionError(f"could not read evaluation rows {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SelectivePredictionError(
                f"invalid JSON on {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise SelectivePredictionError(f"expected an object on {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise SelectivePredictionError(f"evaluation file {path} is empty")
    return rows


def _evaluation_image_ids(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    ids = []
    for index, row in enumerate(rows):
        try:
            image_id = normalize_coco_image_id(row["image_id"])
            question_id = row["question_id"]
        except (KeyError, ValueError) as exc:
            raise SelectivePredictionError(
                f"evaluation row {index} lacks a valid image/question identity"
            ) from exc
        if not isinstance(question_id, (int, str)) or isinstance(question_id, bool):
            raise SelectivePredictionError(f"evaluation row {index} has invalid question_id")
        if row.get("reference_prediction_match") is not True:
            raise SelectivePredictionError(
                f"evaluation row {index} does not confirm the original baseline prediction"
            )
        ids.append(image_id)
    if len(ids) != len(set(ids)):
        raise SelectivePredictionError("evaluation rows must have one question per unique image")
    return tuple(sorted(ids))


def _load_disjoint_calibration_records(
    annotations_dir: Path, evaluation_ids: set[str]
) -> dict[str, list[PopeRecord]]:
    records_by_category: dict[str, list[PopeRecord]] = {}
    for category in CATEGORIES:
        records = load_pope_records(annotations_dir / f"coco_pope_{category}_pilot.jsonl")
        eligible = []
        for record in records:
            try:
                image_id = normalize_coco_image_id(record.image)
            except ValueError as exc:
                raise SelectivePredictionError(
                    f"invalid image identifier {record.image!r} in {category} annotations"
                ) from exc
            if image_id not in evaluation_ids:
                eligible.append(record)
        records_by_category[category] = eligible
    return records_by_category


def _validate_image(image_path: Path) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"calibration image does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"calibration image is unreadable: {image_path}: {exc}") from exc


def _sync(device: str) -> None:
    if device == "mps":
        import torch

        torch.mps.synchronize()


def _validate_model_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "model_identifier",
        "model_revision",
        "device",
        "precision",
        "torch_version",
        "transformers_version",
        "seed",
        "generation_settings",
    )
    metadata = {field: rows[0].get(field) for field in fields}
    if any(row.get(field) != metadata[field] for row in rows for field in fields):
        raise SelectivePredictionError("model or inference metadata varies in evaluation rows")
    if metadata["model_identifier"] != RECOMMENDED_MODEL_ID:
        raise SelectivePredictionError("evaluation model differs from the pinned SmolVLM model")
    if metadata["model_revision"] != RECOMMENDED_MODEL_REVISION:
        raise SelectivePredictionError("evaluation model revision differs from the pinned revision")
    return metadata


def run_abstention_evaluation(
    *,
    evaluation_file: str | Path = DEFAULT_EVALUATION_RESULTS,
    annotations_dir: str | Path = DEFAULT_DATA_DIR,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    calibration_size: int = 20,
    calibration_seed: int = 2027,
    target_coverages: tuple[float, ...] = DEFAULT_TARGET_COVERAGES,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    device: str | None = None,
) -> dict[str, Any]:
    """Score a seeded calibration subset and evaluate targets on held-out rows."""

    if isinstance(calibration_size, bool) or not isinstance(calibration_size, int):
        raise SelectivePredictionError("calibration_size must be a positive integer")
    if calibration_size < 1:
        raise SelectivePredictionError("calibration_size must be a positive integer")
    if isinstance(calibration_seed, bool) or not isinstance(calibration_seed, int):
        raise SelectivePredictionError("calibration_seed must be a non-negative integer")
    if calibration_seed < 0:
        raise SelectivePredictionError("calibration_seed must be a non-negative integer")
    if (
        isinstance(target_coverages, (str, bytes))
        or not isinstance(target_coverages, Sequence)
        or not target_coverages
    ):
        raise SelectivePredictionError("at least one target coverage is required")
    for target in target_coverages:
        if (
            isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not isfinite(target)
            or not 0.0 < target <= 1.0
        ):
            raise SelectivePredictionError("target coverages must be finite values in (0, 1]")
    evaluation_rows = _read_jsonl(Path(evaluation_file))
    evaluation_ids = _evaluation_image_ids(evaluation_rows)
    model_metadata = _validate_model_metadata(evaluation_rows)
    evaluation_dataset_id = (
        f"pope_entropy_{len(evaluation_rows)}_seed_{model_metadata['seed']}"
    )
    calibration_dataset_id = (
        f"pope_pilot_disjoint_{calibration_size}_seed_{calibration_seed}"
    )

    records_by_category = _load_disjoint_calibration_records(
        Path(annotations_dir), set(evaluation_ids)
    )
    selected = select_balanced_questions(
        records_by_category,
        sample_size=calibration_size,
        random_seed=calibration_seed,
    )
    selected_ids = [normalize_coco_image_id(record.image) for _, record in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise SelectivePredictionError("calibration sampler returned repeated image IDs")
    overlap = sorted(set(selected_ids) & set(evaluation_ids))
    if overlap:
        raise SelectivePredictionError(f"calibration and evaluation images overlap: {overlap}")

    image_root = Path(image_dir)
    for _, record in selected:
        _validate_image(image_root / record.image)

    settings = GenerationSettings(
        max_new_tokens=int(model_metadata["generation_settings"]["max_new_tokens"]),
        seed=calibration_seed,
        num_beams=int(model_metadata["generation_settings"]["num_beams"]),
    )
    resolved_device = device or str(model_metadata["device"])
    adapter = TransformersVLMAdapter.from_pretrained(
        model_id=RECOMMENDED_MODEL_ID,
        revision=RECOMMENDED_MODEL_REVISION,
        device=resolved_device,
        generation=settings,
        cache_dir=cache_dir,
    )

    calibration_rows: list[dict[str, Any]] = []
    for category, record in selected:
        image_path = image_root / record.image
        _sync(adapter.metadata.device)
        started = time.perf_counter()
        score = score_binary_answer_entropy(adapter, image_path, record.text)
        _sync(adapter.metadata.device)
        calibration_rows.append(
            {
                "question_id": record.question_id,
                "image_id": normalize_coco_image_id(record.image),
                "image_filename": record.image,
                "category": category,
                "question": record.text,
                "normalized_entropy": score.normalized_entropy,
                "entropy_details": score.to_dict(),
                "scoring_time_seconds": time.perf_counter() - started,
                "calibration_seed": calibration_seed,
                "model_identifier": adapter.metadata.model_id,
                "model_revision": (
                    adapter.metadata.resolved_revision or adapter.metadata.requested_revision
                ),
                "device": adapter.metadata.device,
                "precision": adapter.metadata.precision,
                "scoring_generation_settings": adapter.metadata.to_dict()["generation"],
            }
        )

    # Thresholds are frozen before evaluation labels are passed to the evaluator.
    thresholds = [
        calibrate_abstention_threshold(
            calibration_rows,
            evaluation_image_ids=evaluation_ids,
            target_coverage=target,
            calibration_dataset_id=calibration_dataset_id,
            evaluation_dataset_id=evaluation_dataset_id,
            calibration_seed=calibration_seed,
        )
        for target in target_coverages
    ]
    target_results = [
        {
            "calibration": threshold.to_dict(),
            "evaluation": evaluate_selective_predictions(
                evaluation_rows,
                threshold,
                evaluation_dataset_id=evaluation_dataset_id,
            ),
        }
        for threshold in thresholds
    ]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    calibration_path = output_root / (
        f"pope_abstention_calibration_{calibration_size}_seed_{calibration_seed}.jsonl"
    )
    summary_path = output_root / (
        f"pope_abstention_{calibration_size}cal_{len(evaluation_rows)}eval_"
        f"seed_{calibration_seed}.json"
    )
    calibration_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in calibration_rows),
        encoding="utf-8",
    )
    summary = {
        "method": "binary_predictive_entropy_abstention",
        "score_definition": (
            "Normalized entropy of conditional Yes/No candidate-sequence likelihoods; "
            "answer at or below threshold, abstain above threshold."
        ),
        "calibration_dataset_id": calibration_dataset_id,
        "evaluation_dataset_id": evaluation_dataset_id,
        "calibration_seed": calibration_seed,
        "calibration_sampling": (
            "PCG64 category-balanced sampler; one question per image; evaluation images "
            "excluded before sampling."
        ),
        "calibration_question_count": len(calibration_rows),
        "calibration_image_ids": sorted({row["image_id"] for row in calibration_rows}),
        "evaluation_image_ids": list(evaluation_ids),
        "image_id_overlap_count": len(set(selected_ids) & set(evaluation_ids)),
        "calibration_results_file": str(calibration_path),
        "evaluation_results_file": str(evaluation_file),
        "model_and_evaluation_settings": model_metadata,
        "calibration_model_metadata": adapter.metadata.to_dict(),
        "threshold_selection": (
            "Nearest attainable empirical calibration coverage among observed entropy "
            "cutoffs; tied scores stay together; under-target wins an equal-distance tie. "
            "No correctness labels are used to select thresholds."
        ),
        "targets": target_results,
        "limitations": [
            "Coverage matching does not provide a formal selective-risk guarantee.",
            "The calibration subset and held-out evaluation are small pilot subsets.",
            "Threshold quality can change under different data or shift conditions.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def _parse_coverages(value: str) -> tuple[float, ...]:
    try:
        coverages = tuple(float(piece.strip()) for piece in value.split(",") if piece.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("coverages must be comma-separated numbers") from exc
    if not coverages or any(not 0.0 < coverage <= 1.0 for coverage in coverages):
        raise argparse.ArgumentTypeError("coverages must be in (0, 1]")
    if len(coverages) != len(set(coverages)):
        raise argparse.ArgumentTypeError("coverages must not repeat")
    return coverages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate entropy abstention on disjoint POPE pilot images."
    )
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION_RESULTS)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--target-coverages",
        type=_parse_coverages,
        default=DEFAULT_TARGET_COVERAGES,
        help="comma-separated targets, for example 0.5,0.7,0.8,0.9",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", choices=("mps", "cpu"), default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_abstention_evaluation(
            evaluation_file=args.evaluation,
            annotations_dir=args.annotations_dir,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            calibration_size=args.calibration_size,
            calibration_seed=args.seed,
            target_coverages=args.target_coverages,
            cache_dir=args.cache_dir,
            device=args.device,
        )
    except Exception as exc:
        print(f"abstention evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
