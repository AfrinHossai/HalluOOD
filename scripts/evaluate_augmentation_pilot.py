"""Evaluate augmentation consistency on the fixed 30-question POPE sample."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, UnidentifiedImageError

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))

from halluood.augmentation_evaluation import (
    AugmentationEvaluationError,
    compare_augmentation_scores,
    render_augmentation_risk_coverage_svg,
)
from halluood.augmentations import AugmentationError, create_mild_augmentations
from halluood.data import PopeDataError, PopeRecord, load_pope_records
from halluood.embedding_evaluation import join_paired_rows
from halluood.embeddings import normalize_coco_image_id
from halluood.model import (
    ModelAdapterError,
    TransformersVLMAdapter,
    VLMAdapter,
)
from halluood.uncertainty import augmentation_consistency_score
from scripts.evaluate_augmentation_consistency import (
    DEFAULT_ANNOTATIONS_DIR,
    DEFAULT_BASELINE_RESULTS,
    DEFAULT_CACHE_DIR,
    DEFAULT_IMAGE_DIR,
    _generation_settings,
    _run_prediction,
    _verify_model_settings,
)

DEFAULT_ENTROPY_RESULTS = Path(
    "data/results/pope_entropy_30_seed_2027_evaluated.jsonl"
)
DEFAULT_EMBEDDING_RESULTS = Path(
    "data/results/embedding_distance/pope_embedding_entropy_30_seed_2027_scores.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("data/results/augmentation_evaluation")
EXPECTED_COUNT = 30
EXPECTED_SEED = 2027
_TABLE_FIELDS = (
    "score",
    "score_label",
    "question_count",
    "incorrect_count",
    "incorrect_prevalence",
    "auroc",
    "auprc_average_precision",
    "score_mean_correct",
    "score_median_correct",
    "score_mean_incorrect",
    "score_median_incorrect",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AugmentationEvaluationError(f"could not read JSONL file {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AugmentationEvaluationError(
                f"invalid JSON in {path} on line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise AugmentationEvaluationError(
                f"expected an object in {path} on line {line_number}"
            )
        rows.append(value)
    if not rows:
        raise AugmentationEvaluationError(f"JSONL file has no rows: {path}")
    return rows


def _verify_fixed_sample(
    baseline_rows: list[dict[str, Any]],
    entropy_rows: list[dict[str, Any]],
    embedding_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if any(len(rows) != EXPECTED_COUNT for rows in (baseline_rows, entropy_rows, embedding_rows)):
        raise AugmentationEvaluationError("all source files must contain exactly 30 rows")
    for name, rows in (
        ("baseline", baseline_rows),
        ("entropy", entropy_rows),
        ("embedding", embedding_rows),
    ):
        if any(row.get("seed") != EXPECTED_SEED for row in rows):
            raise AugmentationEvaluationError(
                f"{name} rows do not use the fixed sample/generation seed {EXPECTED_SEED}"
            )
    generation = baseline_rows[0].get("generation_settings", {})
    if (
        generation.get("do_sample") is not False
        or generation.get("num_beams") != 1
        or generation.get("max_new_tokens") != 32
        or generation.get("seed") != EXPECTED_SEED
    ):
        raise AugmentationEvaluationError("baseline generation settings differ from the prior run")
    if any(row.get("generation_settings") != generation for row in baseline_rows):
        raise AugmentationEvaluationError("baseline rows contain different generation settings")
    if any(row.get("reference_prediction_match") is not True for row in entropy_rows):
        raise AugmentationEvaluationError("entropy predictions are not verified against baseline")

    try:
        joined = join_paired_rows(
            entropy_rows,
            baseline_rows,
            embedding_rows,
            expected_count=EXPECTED_COUNT,
        )
    except ValueError as exc:
        raise AugmentationEvaluationError(
            f"could not verify paired sample inputs: {exc}"
        ) from exc
    question_ids = [str(row["question_id"]) for row in joined]
    image_ids = [normalize_coco_image_id(str(row["image_id"])) for row in joined]
    if len(set(question_ids)) != EXPECTED_COUNT or len(set(image_ids)) != EXPECTED_COUNT:
        raise AugmentationEvaluationError(
            "the fixed pilot must contain 30 unique question IDs and 30 unique image IDs"
        )
    category_counts = Counter(row.get("category") for row in joined)
    if category_counts != Counter({"random": 10, "popular": 10, "adversarial": 10}):
        raise AugmentationEvaluationError(
            f"the fixed sample category counts differ from 10 per group: {dict(category_counts)}"
        )
    return joined


def _load_pope_annotations(
    annotations_dir: Path,
    joined_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], PopeRecord]:
    categories = sorted({str(row["category"]) for row in joined_rows})
    records_by_key: dict[tuple[str, str], PopeRecord] = {}
    for category in categories:
        annotation_path = annotations_dir / f"coco_pope_{category}_pilot.jsonl"
        try:
            records = load_pope_records(annotation_path)
        except PopeDataError as exc:
            raise AugmentationEvaluationError(
                f"could not load POPE annotations {annotation_path}: {exc}"
            ) from exc
        for record in records:
            key = category, str(record.question_id)
            if key in records_by_key:
                raise AugmentationEvaluationError(
                    f"duplicate POPE question ID for category/image pair: {key!r}"
                )
            records_by_key[key] = record

    for row in joined_rows:
        key = str(row["category"]), str(row["question_id"])
        record = records_by_key.get(key)
        if record is None:
            raise AugmentationEvaluationError(f"POPE annotation is missing pair {key!r}")
        if (
            record.image != row.get("image_filename")
            or record.text != row.get("question")
            or record.normalized_label != row.get("normalized_ground_truth_label")
        ):
            raise AugmentationEvaluationError(
                f"POPE annotation does not match saved experiment row {key!r}"
            )
        if normalize_coco_image_id(record.image) != normalize_coco_image_id(
            str(row["image_id"])
        ):
            raise AugmentationEvaluationError(f"POPE image ID differs for question {key!r}")
    return records_by_key


def evaluate_augmented_row(
    row: Mapping[str, Any],
    *,
    image_dir: str | Path,
    adapter: VLMAdapter,
    augmentation_seed: int = EXPECTED_SEED,
) -> dict[str, Any]:
    """Run only the transformed views and compare them with the saved original answer."""

    image_filename = Path(str(row["image_filename"])).name
    image_id = normalize_coco_image_id(image_filename)
    if image_id != normalize_coco_image_id(str(row["image_id"])):
        raise AugmentationEvaluationError(
            f"image filename and image ID disagree for question {row['question_id']}"
        )
    image_path = Path(image_dir) / image_filename
    try:
        with Image.open(image_path) as original:
            original.load()
            original_size = list(original.size)
            views = create_mild_augmentations(original, random_seed=augmentation_seed)
    except (OSError, UnidentifiedImageError, AugmentationError) as exc:
        raise AugmentationEvaluationError(
            f"could not prepare image for question {row['question_id']}: {exc}"
        ) from exc

    transformed_answers: list[str | None] = []
    view_rows: list[dict[str, Any]] = []
    try:
        for view in views:
            try:
                prediction = _run_prediction(adapter, view.image, str(row["question"]))
            except Exception as exc:
                raise AugmentationEvaluationError(
                    f"inference failed for question {row['question_id']} view {view.name}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            transformed_answers.append(prediction["normalized_prediction"])
            view_rows.append(
                {
                    "view": view.name,
                    "transform_parameters": view.parameters,
                    "image_width": view.image.width,
                    "image_height": view.image.height,
                    **prediction,
                }
            )
    finally:
        for view in views:
            view.image.close()

    score = augmentation_consistency_score(
        row.get("normalized_prediction"),
        transformed_answers,
    )
    result = dict(row)
    result["original_prediction_for_comparison"] = {
        "raw_prediction": row.get("raw_prediction"),
        "normalized_prediction": row.get("normalized_prediction"),
        "prediction_status": row.get("prediction_status"),
        "is_correct": row["is_correct"],
        "is_error": row["is_error"],
        "source": "unchanged saved baseline/entropy prediction",
    }
    result["augmentation_seed"] = augmentation_seed
    result["original_image_width"] = original_size[0]
    result["original_image_height"] = original_size[1]
    result["transformed_views"] = view_rows
    result["augmentation_consistency_score"] = score.uncertainty_score
    result["augmentation_score_details"] = score.to_dict()
    result["transformed_model_call_count"] = len(view_rows)
    return result


def run_augmentation_evaluation(
    *,
    baseline_results_file: str | Path = DEFAULT_BASELINE_RESULTS,
    entropy_results_file: str | Path = DEFAULT_ENTROPY_RESULTS,
    embedding_results_file: str | Path = DEFAULT_EMBEDDING_RESULTS,
    annotations_dir: str | Path = DEFAULT_ANNOTATIONS_DIR,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    device: str | None = None,
    augmentation_seed: int = EXPECTED_SEED,
    adapter: VLMAdapter | None = None,
) -> dict[str, Any]:
    """Evaluate the exact 30 saved POPE pairs without changing their predictions."""

    if augmentation_seed != EXPECTED_SEED:
        raise AugmentationEvaluationError(
            f"augmentation seed must match the fixed pilot seed {EXPECTED_SEED}"
        )
    baseline_path = Path(baseline_results_file)
    entropy_path = Path(entropy_results_file)
    embedding_path = Path(embedding_results_file)
    baseline_rows = _read_jsonl(baseline_path)
    entropy_rows = _read_jsonl(entropy_path)
    embedding_rows = _read_jsonl(embedding_path)
    joined_rows = _verify_fixed_sample(baseline_rows, entropy_rows, embedding_rows)
    _load_pope_annotations(Path(annotations_dir), joined_rows)

    first_baseline = baseline_rows[0]
    generation = _generation_settings(first_baseline)
    if adapter is None:
        resolved_device = None if device in {None, "auto"} else device
        adapter = TransformersVLMAdapter.from_pretrained(
            model_id=str(first_baseline["model_identifier"]),
            revision=str(first_baseline["model_revision"]),
            device=resolved_device,
            generation=generation,
            cache_dir=cache_dir,
        )
    _verify_model_settings(first_baseline, adapter)

    evaluated_rows = []
    for index, row in enumerate(joined_rows, start=1):
        evaluated_rows.append(
            evaluate_augmented_row(
                row,
                image_dir=image_dir,
                adapter=adapter,
                augmentation_seed=augmentation_seed,
            )
        )
        print(f"Evaluated augmented views for {index}/{len(joined_rows)} questions.", flush=True)

    comparison = compare_augmentation_scores(evaluated_rows)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = "pope_augmentation_consistency_30_seed_2027"
    scores_path = output_path / f"{stem}_scores.jsonl"
    summary_path = output_path / f"{stem}_comparison.json"
    table_path = output_path / f"{stem}_comparison.csv"
    curve_path = output_path / f"{stem}_risk_coverage.svg"
    scores_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in evaluated_rows
        ),
        encoding="utf-8",
    )
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(comparison["score_comparison_table"])

    category_counts = dict(sorted(Counter(row["category"] for row in evaluated_rows).items()))
    summary = {
        **comparison,
        "sample": {
            "question_count": len(evaluated_rows),
            "distinct_image_count": len({row["image_id"] for row in evaluated_rows}),
            "category_counts": category_counts,
            "sampling_seed": EXPECTED_SEED,
            "question_image_pairs_match_baseline_entropy_embedding": True,
            "original_predictions_and_error_labels_reused_unchanged": True,
            "baseline_results_file": str(baseline_path),
            "entropy_results_file": str(entropy_path),
            "embedding_results_file": str(embedding_path),
        },
        "augmentation": {
            "seed": augmentation_seed,
            "transforms_per_question": (
                evaluated_rows[0]["transformed_model_call_count"]
                if evaluated_rows
                else 0
            ),
            "total_transformed_model_calls": sum(
                row["transformed_model_call_count"] for row in evaluated_rows
            ),
            "model_calls_on_original_images": 0,
            "model_metadata": adapter.metadata.to_dict(),
        },
        "output_files": {
            "scores_jsonl": str(scores_path),
            "comparison_json": str(summary_path),
            "comparison_csv": str(table_path),
            "risk_coverage_svg": str(curve_path),
        },
        "interpretation_note": (
            "Augmentation disagreement measures answer instability under these views; "
            "it does not guarantee correctness or estimate hallucination probability."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    curve_path.write_text(
        render_augmentation_risk_coverage_svg(
            comparison["risk_coverage_curves"],
            overall_error_rate=comparison["incorrect_prevalence"],
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "sample": summary["sample"],
                "augmentation": summary["augmentation"],
                "score_comparison_table": comparison["score_comparison_table"],
                "output_files": summary["output_files"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing deterministic image augmentations on the same 30 POPE "
            "question/image pairs used by prior evaluations."
        )
    )
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--entropy-results", type=Path, default=DEFAULT_ENTROPY_RESULTS)
    parser.add_argument("--embedding-results", type=Path, default=DEFAULT_EMBEDDING_RESULTS)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_augmentation_evaluation(
            baseline_results_file=args.baseline_results,
            entropy_results_file=args.entropy_results,
            embedding_results_file=args.embedding_results,
            annotations_dir=args.annotations_dir,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            device=args.device,
            augmentation_seed=args.seed,
        )
    except (AugmentationEvaluationError, ModelAdapterError, OSError, ValueError) as exc:
        print(f"augmentation pilot evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
