"""Run a deterministic, category-balanced POPE baseline on a small sample."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluood.data import PopeDataError, PopeRecord, load_pope_records
from halluood.model import (
    RECOMMENDED_MODEL_ID,
    RECOMMENDED_MODEL_REVISION,
    GenerationSettings,
    TransformersVLMAdapter,
    VLMAdapter,
)
from halluood.uncertainty import score_binary_answer_entropy

CATEGORIES = ("random", "popular", "adversarial")
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_IMAGE_DIR = Path("data/images/val2014")
DEFAULT_OUTPUT = Path("data/results/pope_baseline_smollm_500m_30_seed_2027.jsonl")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "halluood"
_ANSWER_PREFIX = re.compile(r"^\s*(yes|no)(?=$|[\s,.;:!?])", flags=re.IGNORECASE)
_ANSWER_WORD = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)
_UNCERTAINTY = re.compile(
    r"\b(?:maybe|perhaps|possibly|no mention|cannot determine|cannot tell|"
    r"cannot be determined|can't determine|can't tell|not sure|unsure|uncertain|"
    r"unclear|unknown|not enough information|provided facts|don't know)\b",
    flags=re.IGNORECASE,
)
_DIRECT_NEGATIVE = re.compile(
    r"^\s*there\s+(?:is|are|was|were)\s+no\s+.+?\b"
    r"(?:present|visible|in (?:the )?image|in (?:the )?picture)\b",
    flags=re.IGNORECASE,
)
_DIRECT_POSITIVE = re.compile(
    r"^\s*there\s+(?:is|are|was|were)\s+(?:a|an|one|some|at least one)\s+.+?\b"
    r"(?:present|visible|in (?:the )?image|in (?:the )?picture)\b",
    flags=re.IGNORECASE,
)
_COCO_IMAGE_ID = re.compile(r"^COCO_(?:train|val)2014_(\d{12})\.jpg$", flags=re.IGNORECASE)


def normalize_prediction(raw_prediction: str) -> tuple[str | None, str]:
    """Extract an unambiguous leading yes/no answer, preserving raw text elsewhere."""

    if not isinstance(raw_prediction, str) or not raw_prediction.strip():
        return None, "ambiguous"

    if _UNCERTAINTY.search(raw_prediction):
        return None, "ambiguous"

    match = _ANSWER_PREFIX.match(raw_prediction)
    answer_words = {word.casefold() for word in _ANSWER_WORD.findall(raw_prediction)}
    if match is not None:
        first_answer = match.group(1).casefold()
        if len(answer_words) != 1 or first_answer == "no" and re.match(
            r"^\s*no\s+doubt\b", raw_prediction, flags=re.IGNORECASE
        ):
            return None, "ambiguous"
        return first_answer, "parsed"

    if _DIRECT_NEGATIVE.search(raw_prediction) and "yes" not in answer_words:
        return "no", "parsed"
    if _DIRECT_POSITIVE.search(raw_prediction) and "no" not in answer_words:
        return "yes", "parsed"
    if len(answer_words) > 1:
        return None, "ambiguous"
    return None, "ambiguous"


def select_balanced_questions(
    records_by_category: dict[str, list[PopeRecord]],
    sample_size: int = 30,
    random_seed: int = 2027,
) -> list[tuple[str, PopeRecord]]:
    """Select one question per image/category, balancing strata and image IDs.

    The selection first allocates an even category quota, redistributing quotas
    if a category has fewer image groups. A seeded PCG64 generator shuffles the
    eligible image groups within each category. It prefers distinct images
    across categories and permits cross-category reuse only when necessary.
    One row is then sampled from each selected image group.
    """

    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    if set(records_by_category) != set(CATEGORIES):
        raise ValueError(f"records must be supplied for exactly these categories: {CATEGORIES}")

    grouped: dict[str, dict[str, list[PopeRecord]]] = {}
    for category in CATEGORIES:
        category_groups: dict[str, list[PopeRecord]] = {}
        for record in records_by_category[category]:
            if record.category is not None and record.category.casefold() != category:
                raise PopeDataError(
                    f"record {record.question_id!r} belongs to category "
                    f"{record.category!r}, expected {category!r}"
                )
            category_groups.setdefault(record.image, []).append(record)
        grouped[category] = category_groups

    capacities = {category: len(grouped[category]) for category in CATEGORIES}
    if sum(capacities.values()) < sample_size:
        raise ValueError(
            f"sample_size ({sample_size}) exceeds the {sum(capacities.values())} "
            "available category/image groups"
        )

    base, remainder = divmod(sample_size, len(CATEGORIES))
    quotas = {
        category: min(base + (index < remainder), capacities[category])
        for index, category in enumerate(CATEGORIES)
    }
    while sum(quotas.values()) < sample_size:
        available = [
            category for category in CATEGORIES if quotas[category] < capacities[category]
        ]
        if not available:
            raise ValueError("could not allocate the requested sample across POPE categories")
        category = min(available, key=lambda item: (quotas[item], CATEGORIES.index(item)))
        quotas[category] += 1

    rng = np.random.Generator(np.random.PCG64(random_seed))
    shuffled_images = {
        category: [
            str(image)
            for image in rng.permutation(np.asarray(sorted(grouped[category]), dtype=object))
        ]
        for category in CATEGORIES
    }
    allocation_order = sorted(
        CATEGORIES, key=lambda category: (capacities[category], CATEGORIES.index(category))
    )
    selected_images = {category: [] for category in CATEGORIES}
    used_images: set[str] = set()

    for category in allocation_order:
        distinct_candidates = [
            image for image in shuffled_images[category] if image not in used_images
        ]
        chosen = distinct_candidates[: quotas[category]]
        selected_images[category].extend(chosen)
        used_images.update(chosen)

    for category in allocation_order:
        missing = quotas[category] - len(selected_images[category])
        if missing:
            already_selected = set(selected_images[category])
            additional = [
                image for image in shuffled_images[category] if image not in already_selected
            ][:missing]
            selected_images[category].extend(additional)

    selected: list[tuple[str, PopeRecord]] = []
    for category in CATEGORIES:
        for image in selected_images[category]:
            records = grouped[category][image]
            record_index = int(rng.integers(len(records)))
            selected.append((category, records[record_index]))

    if len(selected) != sample_size:
        raise RuntimeError(f"sampler returned {len(selected)} rows; expected {sample_size}")
    return selected


def _image_id(filename: str) -> str:
    match = _COCO_IMAGE_ID.fullmatch(Path(filename).name)
    return match.group(1) if match else Path(filename).stem


def _validate_image(image_path: Path) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"pilot image does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"pilot image is not readable: {image_path}: {exc}") from exc


def _synchronize_mps(device: str) -> None:
    if device == "mps":
        import torch

        torch.mps.synchronize()


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct_count = sum(
        row["normalized_prediction"] == row["normalized_ground_truth_label"] for row in rows
    )
    parsed_count = sum(row["prediction_status"] == "parsed" for row in rows)
    ambiguous_count = sum(row["prediction_status"] == "ambiguous" for row in rows)
    failed_count = sum(row["prediction_status"] == "error" for row in rows)
    return {
        "question_count": len(rows),
        "correct_count": correct_count,
        "accuracy": correct_count / len(rows) if rows else None,
        "parsed_count": parsed_count,
        "accuracy_on_parsed": correct_count / parsed_count if parsed_count else None,
        "ambiguous_count": ambiguous_count,
        "failed_count": failed_count,
    }


def run_baseline(
    *,
    annotations_dir: str | Path = DEFAULT_DATA_DIR,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    output_file: str | Path = DEFAULT_OUTPUT,
    sample_size: int = 30,
    random_seed: int = 2027,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    device: str | None = None,
    max_new_tokens: int = 32,
    adapter: VLMAdapter | None = None,
    record_uncertainty: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the selected questions, writing row-level JSONL and summary JSON."""

    data_dir = Path(annotations_dir)
    records_by_category = {
        category: load_pope_records(data_dir / f"coco_pope_{category}_pilot.jsonl")
        for category in CATEGORIES
    }
    selected = select_balanced_questions(records_by_category, sample_size, random_seed)
    images_dir = Path(image_dir)
    for _, record in selected:
        _validate_image(images_dir / record.image)

    settings = GenerationSettings(max_new_tokens=max_new_tokens, seed=random_seed)
    if adapter is None:
        adapter = TransformersVLMAdapter.from_pretrained(
            model_id=RECOMMENDED_MODEL_ID,
            revision=RECOMMENDED_MODEL_REVISION,
            device=device,
            generation=settings,
            cache_dir=cache_dir,
        )

    metadata = adapter.metadata.to_dict()
    output_path = Path(output_file)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8", newline="\n") as output_stream:
        for category, record in selected:
            image_path = images_dir / record.image
            raw_prediction: str | None = None
            normalized_prediction: str | None = None
            prediction_status = "error"
            error_message: str | None = None
            _synchronize_mps(adapter.metadata.device)
            start_time = time.perf_counter()
            try:
                raw_prediction = adapter.generate_answer(image_path, record.text)
                normalized_prediction, prediction_status = normalize_prediction(raw_prediction)
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    _synchronize_mps(adapter.metadata.device)
                except Exception as exc:
                    if error_message is None:
                        error_message = f"{type(exc).__name__}: {exc}"
                    prediction_status = "error"
                    normalized_prediction = None
            inference_time = time.perf_counter() - start_time

            uncertainty_score: dict[str, Any] | None = None
            uncertainty_error: str | None = None
            uncertainty_time: float | None = None
            uncertainty_start: float | None = None
            if record_uncertainty and prediction_status != "error":
                try:
                    _synchronize_mps(adapter.metadata.device)
                    uncertainty_start = time.perf_counter()
                    uncertainty_score = score_binary_answer_entropy(
                        adapter,
                        image_path,
                        record.text,
                    ).to_dict()
                except Exception as exc:
                    uncertainty_error = f"{type(exc).__name__}: {exc}"
                finally:
                    try:
                        _synchronize_mps(adapter.metadata.device)
                    except Exception as exc:
                        if uncertainty_error is None:
                            uncertainty_error = f"{type(exc).__name__}: {exc}"
                    if uncertainty_start is not None:
                        uncertainty_time = time.perf_counter() - uncertainty_start

            row = {
                "question_id": record.question_id,
                "image_id": _image_id(record.image),
                "image_filename": record.image,
                "category": category,
                "question": record.text,
                "ground_truth_label": record.label,
                "normalized_ground_truth_label": record.normalized_label,
                "raw_prediction": raw_prediction,
                "normalized_prediction": normalized_prediction,
                "prediction_status": prediction_status,
                "model_identifier": metadata["model_id"],
                "model_revision": metadata["resolved_revision"]
                or metadata["requested_revision"],
                "device": metadata["device"],
                "precision": metadata["precision"],
                "torch_version": metadata["torch_version"],
                "transformers_version": metadata["transformers_version"],
                "seed": random_seed,
                "generation_settings": metadata["generation"],
                "inference_time_seconds": inference_time,
            }
            if error_message is not None:
                row["error"] = error_message
            if record_uncertainty:
                row["uncertainty_score"] = uncertainty_score
                row["uncertainty_time_seconds"] = uncertainty_time
                if uncertainty_error is not None:
                    row["uncertainty_error"] = uncertainty_error
            result_rows.append(row)
            output_stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output_stream.flush()

    by_category = {
        category: _metrics([row for row in result_rows if row["category"] == category])
        for category in CATEGORIES
    }
    summary = {
        "sample_size": sample_size,
        "seed": random_seed,
        "selection_category_counts": {
            category: sum(item[0] == category for item in selected) for category in CATEGORIES
        },
        "distinct_image_count": len({row["image_id"] for row in result_rows}),
        "overall": _metrics(result_rows),
        "by_category": by_category,
        "model_metadata": metadata,
        "generation_settings": metadata["generation"],
        "results_file": str(output_path),
        "summary_file": str(summary_path),
        "accuracy_note": (
            "Ambiguous and failed predictions count as incorrect in all-question accuracy; "
            "accuracy_on_parsed is reported separately."
        ),
    }
    if record_uncertainty:
        uncertainty_rows = [
            row for row in result_rows if row.get("uncertainty_score") is not None
        ]
        summary["uncertainty_method"] = {
            "name": "conditional_binary_answer_sequence_entropy",
            "candidate_answers": ["Yes", "No"],
            "score_direction": "higher_means_more_uncertain",
            "scored_count": len(uncertainty_rows),
            "error_count": sum("uncertainty_error" in row for row in result_rows),
        }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary, result_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic category-balanced POPE baseline. The default is "
            "30 questions (10 per POPE strategy) with seed 2027."
        )
    )
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--record-uncertainty",
        action="store_true",
        help=(
            "also score conditional Yes/No candidate-sequence entropy using model logits "
            "(requires a Transformers adapter; adds two model forward passes per question)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, _ = run_baseline(
            annotations_dir=args.annotations_dir,
            image_dir=args.image_dir,
            output_file=args.output,
            sample_size=args.sample_size,
            random_seed=args.seed,
            cache_dir=args.cache_dir,
            device=None if args.device == "auto" else args.device,
            max_new_tokens=args.max_new_tokens,
            record_uncertainty=args.record_uncertainty,
        )
    except Exception as exc:
        print(f"baseline evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 1 if summary["overall"]["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
