"""Run deterministic SmolVLM predictions and binary entropy on the full POPE pilot."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluood.data import load_pope_records
from halluood.model import (
    RECOMMENDED_MODEL_ID,
    RECOMMENDED_MODEL_REVISION,
    GenerationSettings,
    TransformersVLMAdapter,
    VLMAdapter,
)
from halluood.predictions import normalize_prediction
from halluood.uncertainty import score_binary_answer_entropy

CATEGORIES = ("random", "popular", "adversarial")
IMAGE_ID_RE = re.compile(r"^COCO_(?:train|val)2014_(\d{12})\.jpg$", re.IGNORECASE)
DEFAULT_ANNOTATIONS = Path("data/processed")
DEFAULT_IMAGES = Path("data/images/val2014")
DEFAULT_OUTPUT = Path("data/results/robustness/full_pope_predictions_seed_2027.jsonl")
DEFAULT_CACHE = Path.home() / ".cache" / "huggingface" / "halluood"


class FullPOPEError(ValueError):
    """Raised when the local POPE corpus or result file is invalid."""


def _image_id(filename: str) -> str:
    match = IMAGE_ID_RE.fullmatch(Path(filename).name)
    if match is None:
        raise FullPOPEError(f"unexpected COCO filename: {filename!r}")
    return match.group(1)


def _synchronize_mps(device: str) -> None:
    if device == "mps":
        import torch

        torch.mps.synchronize()


def _validate_corpus(
    annotation_dir: Path, image_dir: Path
) -> tuple[list[tuple[str, Any]], list[str]]:
    records_by_category = {
        category: load_pope_records(annotation_dir / f"coco_pope_{category}_pilot.jsonl")
        for category in CATEGORIES
    }
    expected_ids: set[str] | None = None
    all_rows: list[tuple[str, Any]] = []
    for category in CATEGORIES:
        records = records_by_category[category]
        image_ids = {_image_id(record.image) for record in records}
        if len(records) != 600 or len(image_ids) != 100:
            raise FullPOPEError(
                f"{category} annotations must contain 600 questions on 100 images; "
                f"found {len(records)} questions on {len(image_ids)} images"
            )
        if expected_ids is None:
            expected_ids = image_ids
        elif image_ids != expected_ids:
            raise FullPOPEError(f"the {category} image set differs from the other categories")
        image_counts: dict[str, int] = {}
        category_question_ids: set[str] = set()
        for record in records:
            identity = str(record.question_id)
            if identity in category_question_ids:
                raise FullPOPEError(f"duplicate question ID in {category}: {identity}")
            category_question_ids.add(identity)
            image_counts[record.image] = image_counts.get(record.image, 0) + 1
            if record.category is not None and record.category.casefold() != category:
                raise FullPOPEError(
                    f"question {identity} has category {record.category!r}, expected {category!r}"
                )
        if set(image_counts.values()) != {6}:
            raise FullPOPEError(f"{category} must have six questions per image")
        all_rows.extend((category, record) for record in records)

    assert expected_ids is not None
    for image in sorted({record.image for _, record in all_rows}):
        image_path = image_dir / image
        if not image_path.is_file():
            raise FileNotFoundError(f"required local COCO image is missing: {image_path}")
        try:
            with Image.open(image_path) as opened:
                opened.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise FullPOPEError(f"required COCO image is unreadable: {image_path}: {exc}") from exc
    all_rows.sort(
        key=lambda pair: (
            CATEGORIES.index(pair[0]),
            _image_id(pair[1].image),
            str(pair[1].question_id),
        )
    )
    return all_rows, sorted(expected_ids)


def _read_resume(
    path: Path, expected: dict[tuple[str, str, str], Any]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[tuple[str, str, str], dict[str, Any]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_number == len(lines):
                # A process interruption may leave only the final JSONL record partial.
                break
            raise FullPOPEError(f"invalid resume JSONL line {line_number}: {exc}") from exc
        key = (str(row.get("category")), str(row.get("question_id")), str(row.get("image_id")))
        if key not in expected:
            raise FullPOPEError(f"resume file has an unexpected question identity: {key}")
        if key in completed:
            raise FullPOPEError(f"resume file contains duplicate question identity: {key}")
        source = expected[key]
        if (
            row.get("question") != source.text
            or row.get("normalized_ground_truth_label") != source.normalized_label
        ):
            raise FullPOPEError(f"resume file changed question or gold label for {key}")
        completed[key] = row
    return completed


def run_full_pope(
    *,
    annotation_dir: str | Path = DEFAULT_ANNOTATIONS,
    image_dir: str | Path = DEFAULT_IMAGES,
    output_file: str | Path = DEFAULT_OUTPUT,
    cache_dir: str | Path = DEFAULT_CACHE,
    device: str | None = None,
    seed: int = 2027,
    max_new_tokens: int = 32,
    adapter: VLMAdapter | None = None,
) -> dict[str, Any]:
    """Run or resume deterministic generation and binary entropy for all 1,800 rows."""

    annotations = Path(annotation_dir)
    images = Path(image_dir)
    output = Path(output_file)
    rows, image_ids = _validate_corpus(annotations, images)
    generation = GenerationSettings(max_new_tokens=max_new_tokens, seed=seed)
    if adapter is None:
        adapter = TransformersVLMAdapter.from_pretrained(
            model_id=RECOMMENDED_MODEL_ID,
            revision=RECOMMENDED_MODEL_REVISION,
            device=device,
            generation=generation,
            cache_dir=cache_dir,
        )
    metadata = adapter.metadata.to_dict()
    if metadata["model_id"] != RECOMMENDED_MODEL_ID:
        raise FullPOPEError(f"unexpected model identifier: {metadata['model_id']}")
    resolved_revision = metadata.get("resolved_revision") or metadata.get("requested_revision")
    if resolved_revision != RECOMMENDED_MODEL_REVISION:
        raise FullPOPEError(f"unexpected model revision: {resolved_revision}")
    if metadata.get("generation", {}).get("seed") != seed:
        raise FullPOPEError("adapter generation seed does not match the requested experiment seed")
    expected = {
        (category, str(record.question_id), _image_id(record.image)): record
        for category, record in rows
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_resume(output, expected)
    completed = {
        key: row
        for key, row in completed.items()
        if row.get("prediction_status") != "error"
        and row.get("normalized_entropy") is not None
        and row.get("model_revision") == resolved_revision
        and row.get("generation_settings") == metadata["generation"]
    }
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in completed.values()
        ),
        encoding="utf-8",
    )
    failures = 0
    started_total = time.perf_counter()
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for index, (category, record) in enumerate(rows, start=1):
            key = (category, str(record.question_id), _image_id(record.image))
            existing = completed.get(key)
            if (
                existing is not None
                and existing.get("prediction_status") != "error"
                and existing.get("normalized_entropy") is not None
                and existing.get("model_revision") == resolved_revision
                and existing.get("generation_settings") == metadata["generation"]
            ):
                continue
            image_path = images / record.image
            _synchronize_mps(metadata["device"])
            prediction_start = time.perf_counter()
            prediction: str | None = None
            normalized: str | None = None
            status = "error"
            prediction_error: str | None = None
            try:
                prediction = adapter.generate_answer(image_path, record.text)
                normalized, status = normalize_prediction(prediction)
            except Exception as exc:
                prediction_error = f"{type(exc).__name__}: {exc}"
            finally:
                _synchronize_mps(metadata["device"])
            prediction_seconds = time.perf_counter() - prediction_start
            entropy_result: dict[str, Any] | None = None
            entropy_error: str | None = None
            entropy_seconds: float | None = None
            if status != "error":
                _synchronize_mps(metadata["device"])
                entropy_start = time.perf_counter()
                try:
                    entropy_result = score_binary_answer_entropy(
                        adapter, image_path, record.text
                    ).to_dict()
                except Exception as exc:
                    entropy_error = f"{type(exc).__name__}: {exc}"
                finally:
                    _synchronize_mps(metadata["device"])
                entropy_seconds = time.perf_counter() - entropy_start
            is_correct = status == "parsed" and normalized == record.normalized_label
            row: dict[str, Any] = {
                "question_id": record.question_id,
                "image_id": key[2],
                "image_filename": record.image,
                "category": category,
                "question": record.text,
                "ground_truth_label": record.label,
                "normalized_ground_truth_label": record.normalized_label,
                "raw_prediction": prediction,
                "normalized_prediction": normalized,
                "prediction_status": status,
                "is_correct": is_correct,
                "is_error": not is_correct,
                "normalized_entropy": (
                    entropy_result["normalized_entropy"] if entropy_result is not None else None
                ),
                "uncertainty_score": entropy_result,
                "model_identifier": metadata["model_id"],
                "model_revision": resolved_revision,
                "device": metadata["device"],
                "precision": metadata["precision"],
                "torch_version": metadata["torch_version"],
                "transformers_version": metadata["transformers_version"],
                "seed": seed,
                "generation_settings": metadata["generation"],
                "inference_time_seconds": prediction_seconds,
                "uncertainty_time_seconds": entropy_seconds,
            }
            if prediction_error is not None:
                row["prediction_error"] = prediction_error
            if entropy_error is not None:
                row["uncertainty_error"] = entropy_error
            if status == "error" or entropy_result is None:
                failures += 1
            completed[key] = row
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            # Keep logs sparse but useful for a long local run.
            if index == 1 or index % 25 == 0 or index == len(rows):
                print(
                    f"[{index}/{len(rows)}] completed={len(completed)} failures={failures} "
                    f"elapsed={time.perf_counter() - started_total:.1f}s",
                    flush=True,
                )

    rows_out = [completed[key] for key in expected if key in completed]
    summary = {
        "dataset": "local prepared POPE pilot; Random, Popular, Adversarial",
        "expected_question_count": len(rows),
        "completed_question_count": len(rows_out),
        "image_count": len(image_ids),
        "images_per_category": 100,
        "questions_per_image_per_category": 6,
        "category_question_counts": {
            category: sum(row["category"] == category for row in rows_out)
            for category in CATEGORIES
        },
        "prediction_or_entropy_failure_count": sum(
            row["prediction_status"] == "error" or row.get("normalized_entropy") is None
            for row in rows_out
        ),
        "model_metadata": metadata,
        "seed": seed,
        "generation_settings": metadata["generation"],
        "elapsed_seconds_this_invocation": time.perf_counter() - started_total,
        "results_file": str(output),
        "metric_note": "This runner stores raw outputs; analyze only rows with recorded scores.",
    }
    summary_path = output.with_name(f"{output.stem}_run_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic SmolVLM predictions and binary entropy on all 1,800 POPE rows."
        )
    )
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_full_pope(
            annotation_dir=args.annotations_dir,
            image_dir=args.image_dir,
            output_file=args.output,
            cache_dir=args.cache_dir,
            device=args.device,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as exc:
        print(f"full POPE inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 1 if summary["prediction_or_entropy_failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
