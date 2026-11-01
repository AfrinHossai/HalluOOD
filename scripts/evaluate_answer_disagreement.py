"""Evaluate sampled answer disagreement on the fixed 30-question POPE pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, UnidentifiedImageError

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))

from halluood.answer_disagreement_evaluation import (
    AnswerDisagreementEvaluationError,
    compare_error_detection_scores,
    render_answer_disagreement_risk_coverage_svg,
)
from halluood.data import PopeDataError
from halluood.embeddings import normalize_coco_image_id
from halluood.model import (
    ModelAdapterError,
    SamplingSettings,
    TransformersVLMAdapter,
    VLMAdapter,
)
from halluood.uncertainty import answer_disagreement_score
from scripts.evaluate_augmentation_consistency import (
    DEFAULT_ANNOTATIONS_DIR,
    DEFAULT_BASELINE_RESULTS,
    DEFAULT_CACHE_DIR,
    DEFAULT_IMAGE_DIR,
    _generation_settings,
    _verify_model_settings,
)
from scripts.evaluate_augmentation_pilot import (
    DEFAULT_EMBEDDING_RESULTS,
    DEFAULT_ENTROPY_RESULTS,
    _load_pope_annotations,
    _verify_fixed_sample,
)
from scripts.run_baseline import normalize_prediction

DEFAULT_AUGMENTATION_RESULTS = Path(
    "data/results/augmentation_evaluation/"
    "pope_augmentation_consistency_30_seed_2027_scores.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("data/results/answer_disagreement_evaluation")
EXPECTED_QUESTION_COUNT = 30
GENERATIONS_PER_QUESTION = 5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_SEED = 2027
_TABLE_FIELDS = (
    "score",
    "score_label",
    "question_count",
    "incorrect_or_ambiguous_count",
    "error_prevalence",
    "auroc",
    "auprc_average_precision",
    "mean_score_correct",
    "median_score_correct",
    "mean_score_incorrect_or_ambiguous",
    "median_score_incorrect_or_ambiguous",
)


class AnswerDisagreementRunError(ValueError):
    """Raised when the fixed sampled-answer evaluation cannot be completed safely."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AnswerDisagreementRunError(f"could not read JSONL file {path}: {exc}") from exc
    rows = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnswerDisagreementRunError(
                f"invalid JSON in {path} on line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise AnswerDisagreementRunError(
                f"expected an object in {path} on line {line_number}"
            )
        rows.append(row)
    if not rows:
        raise AnswerDisagreementRunError(f"JSONL input contains no rows: {path}")
    return rows


def _join_augmentation_rows(
    paired_rows: list[dict[str, Any]],
    augmentation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(augmentation_rows) != EXPECTED_QUESTION_COUNT:
        raise AnswerDisagreementRunError(
            "augmentation evaluation must contain exactly 30 paired rows"
        )
    indexed = {}
    for row in augmentation_rows:
        try:
            key = (
                str(row["question_id"]),
                normalize_coco_image_id(str(row["image_id"])),
            )
        except (KeyError, ValueError) as exc:
            raise AnswerDisagreementRunError(
                f"augmentation row has invalid question/image identity: {exc}"
            ) from exc
        if key in indexed:
            raise AnswerDisagreementRunError(
                f"augmentation result has duplicate question/image pair {key!r}"
            )
        indexed[key] = row

    if len(paired_rows) != EXPECTED_QUESTION_COUNT:
        raise AnswerDisagreementRunError("paired source files must contain exactly 30 rows")
    joined = []
    for source in paired_rows:
        key = str(source["question_id"]), normalize_coco_image_id(str(source["image_id"]))
        augmented = indexed.pop(key, None)
        if augmented is None:
            raise AnswerDisagreementRunError(
                f"augmentation result is missing question/image pair {key!r}"
            )
        for field in (
            "category",
            "question",
            "normalized_ground_truth_label",
            "normalized_prediction",
            "raw_prediction",
            "prediction_status",
            "is_correct",
            "is_error",
        ):
            if augmented.get(field) != source.get(field):
                raise AnswerDisagreementRunError(
                    f"augmentation {field} differs from the paired baseline for {key!r}"
                )
        provenance = augmented.get("original_prediction_for_comparison")
        if not isinstance(provenance, dict) or (
            provenance.get("raw_prediction") != source.get("raw_prediction")
            or provenance.get("normalized_prediction") != source.get("normalized_prediction")
            or provenance.get("is_correct") is not source.get("is_correct")
            or provenance.get("is_error") is not source.get("is_error")
        ):
            raise AnswerDisagreementRunError(
                f"augmentation baseline provenance differs for question/image pair {key!r}"
            )
        try:
            score = float(augmented["augmentation_consistency_score"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise AnswerDisagreementRunError(
                f"augmentation result has no numeric score for {key!r}"
            ) from exc
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise AnswerDisagreementRunError(
                f"augmentation score for {key!r} must be finite and in [0, 1]"
            )
        joined.append({**source, "augmentation_consistency_score": score})
    if indexed:
        raise AnswerDisagreementRunError(
            f"augmentation results contain unpaired entries: {sorted(indexed)[:3]}"
        )
    return joined


def evaluate_sampled_row(
    row: Mapping[str, Any],
    *,
    image_dir: str | Path,
    adapter: VLMAdapter,
    settings: SamplingSettings,
) -> dict[str, Any]:
    """Sample five answers for one fixed baseline question and attach its score."""

    image_filename = Path(str(row["image_filename"])).name
    image_id = normalize_coco_image_id(image_filename)
    if image_id != normalize_coco_image_id(str(row["image_id"])):
        raise AnswerDisagreementRunError(
            f"image filename and image ID disagree for question {row['question_id']}"
        )
    image_path = Path(image_dir) / image_filename
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise AnswerDisagreementRunError(
            f"could not read image for question {row['question_id']}: {exc}"
        ) from exc

    start = time.perf_counter()
    try:
        raw_answers = adapter.generate_sampled_answers(
            image_path,
            str(row["question"]),
            settings,
        )
    except Exception as exc:
        raise AnswerDisagreementRunError(
            f"sampled inference failed for question {row['question_id']}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    elapsed = time.perf_counter() - start
    if not isinstance(raw_answers, (list, tuple)):
        raise AnswerDisagreementRunError("sampled inference must return answer strings")
    if len(raw_answers) != GENERATIONS_PER_QUESTION:
        raise AnswerDisagreementRunError(
            f"question {row['question_id']} returned {len(raw_answers)} answers; "
            f"expected exactly {GENERATIONS_PER_QUESTION}"
        )
    if any(not isinstance(answer, str) for answer in raw_answers):
        raise AnswerDisagreementRunError(
            f"question {row['question_id']} returned a non-string answer"
        )

    answer_rows = []
    normalized_answers = []
    for index, raw_answer in enumerate(raw_answers):
        normalized, status = normalize_prediction(raw_answer)
        label = normalized if normalized in {"yes", "no"} else "ambiguous"
        normalized_answers.append(label)
        answer_rows.append(
            {
                "generation_index": index,
                "seed": settings.generation_seeds[index],
                "raw_answer": raw_answer,
                "normalized_answer": label,
                "parser_status": status,
            }
        )
    score = answer_disagreement_score(normalized_answers)
    result = dict(row)
    result["sampling_settings"] = settings.to_dict()
    result["sampled_answers"] = answer_rows
    result["answer_disagreement_score"] = score.normalized_score
    result["answer_disagreement_details"] = score.to_dict()
    result["answer_disagreement_inference_time_seconds"] = elapsed
    return result


def run_answer_disagreement_evaluation(
    *,
    baseline_results_file: str | Path = DEFAULT_BASELINE_RESULTS,
    entropy_results_file: str | Path = DEFAULT_ENTROPY_RESULTS,
    embedding_results_file: str | Path = DEFAULT_EMBEDDING_RESULTS,
    augmentation_results_file: str | Path = DEFAULT_AUGMENTATION_RESULTS,
    annotations_dir: str | Path = DEFAULT_ANNOTATIONS_DIR,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    max_new_tokens: int = 32,
    device: str | None = None,
    adapter: VLMAdapter | None = None,
) -> dict[str, Any]:
    """Run five sampled generations for each of the same 30 POPE pilot pairs."""

    sampling_template = SamplingSettings(
        num_generations=GENERATIONS_PER_QUESTION,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    baseline_path = Path(baseline_results_file)
    entropy_path = Path(entropy_results_file)
    embedding_path = Path(embedding_results_file)
    augmentation_path = Path(augmentation_results_file)
    baseline_rows = _read_jsonl(baseline_path)
    entropy_rows = _read_jsonl(entropy_path)
    embedding_rows = _read_jsonl(embedding_path)
    augmentation_rows = _read_jsonl(augmentation_path)
    paired_rows = _verify_fixed_sample(baseline_rows, entropy_rows, embedding_rows)
    _load_pope_annotations(Path(annotations_dir), paired_rows)
    joined_rows = _join_augmentation_rows(paired_rows, augmentation_rows)

    first_baseline = baseline_rows[0]
    generation = _generation_settings(first_baseline)
    if adapter is None:
        resolved_device = None if device in {None, "auto"} else device
        try:
            adapter = TransformersVLMAdapter.from_pretrained(
                model_id=str(first_baseline["model_identifier"]),
                revision=str(first_baseline["model_revision"]),
                device=resolved_device,
                generation=generation,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            raise AnswerDisagreementRunError(
                f"could not load the fixed baseline model: {type(exc).__name__}: {exc}"
            ) from exc
    _verify_model_settings(first_baseline, adapter)

    evaluated_rows = []
    for index, row in enumerate(joined_rows):
        per_question_seed = seed + index * GENERATIONS_PER_QUESTION
        settings = SamplingSettings(
            num_generations=GENERATIONS_PER_QUESTION,
            temperature=sampling_template.temperature,
            top_p=sampling_template.top_p,
            seed=per_question_seed,
            max_new_tokens=sampling_template.max_new_tokens,
        )
        evaluated_rows.append(
            evaluate_sampled_row(
                row,
                image_dir=image_dir,
                adapter=adapter,
                settings=settings,
            )
        )
        print(
            f"Sampled answers for {index + 1}/{len(joined_rows)} questions.",
            flush=True,
        )

    comparison = compare_error_detection_scores(evaluated_rows)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = f"pope_answer_disagreement_{EXPECTED_QUESTION_COUNT}_seed_{seed}"
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
        writer.writerows(comparison["comparison_table"])

    model_metadata = adapter.metadata.to_dict()
    category_counts = dict(sorted(Counter(row["category"] for row in evaluated_rows).items()))
    summary = {
        **comparison,
        "sample": {
            "question_count": len(evaluated_rows),
            "distinct_image_count": len({row["image_id"] for row in evaluated_rows}),
            "category_counts": category_counts,
            "question_image_pairs_match_baseline_entropy_embedding_augmentation": True,
            "original_baseline_predictions_and_error_labels_reused_unchanged": True,
            "baseline_results_file": str(baseline_path),
            "entropy_results_file": str(entropy_path),
            "embedding_results_file": str(embedding_path),
            "augmentation_results_file": str(augmentation_path),
        },
        "sampling": {
            "num_generations_per_question": GENERATIONS_PER_QUESTION,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "base_seed": seed,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "num_beams": 1,
            "per_question_seed_strategy": (
                "rows sorted lexicographically by question ID and normalized image ID; base_seed + "
                "sample_index * num_generations; each response seed adds its zero-based index"
            ),
            "total_sampled_generations": sum(
                len(row["sampled_answers"]) for row in evaluated_rows
            ),
            "model_calls_on_original_baseline_predictions": 0,
        },
        "model_metadata": model_metadata,
        "outputs": {
            "scores_jsonl": str(scores_path),
            "comparison_json": str(summary_path),
            "comparison_csv": str(table_path),
            "risk_coverage_svg": str(curve_path),
        },
        "interpretation_note": (
            "Sampled-answer disagreement is empirical response variation, not a calibrated "
            "probability of error or hallucination. A stable incorrect answer can score zero."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    curve_path.write_text(
        render_answer_disagreement_risk_coverage_svg(
            comparison["risk_coverage_curves"],
            overall_error_rate=comparison["error_prevalence"],
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "sample": summary["sample"],
                "sampling": summary["sampling"],
                "comparison_table": comparison["comparison_table"],
                "outputs": summary["outputs"],
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
            "Evaluate five sampled responses on the exact paired 30-question POPE pilot."
        )
    )
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--entropy-results", type=Path, default=DEFAULT_ENTROPY_RESULTS)
    parser.add_argument("--embedding-results", type=Path, default=DEFAULT_EMBEDDING_RESULTS)
    parser.add_argument(
        "--augmentation-results", type=Path, default=DEFAULT_AUGMENTATION_RESULTS
    )
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_answer_disagreement_evaluation(
            baseline_results_file=args.baseline_results,
            entropy_results_file=args.entropy_results,
            embedding_results_file=args.embedding_results,
            augmentation_results_file=args.augmentation_results,
            annotations_dir=args.annotations_dir,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            device=None if args.device == "auto" else args.device,
        )
    except (
        AnswerDisagreementRunError,
        AnswerDisagreementEvaluationError,
        PopeDataError,
        ModelAdapterError,
        OSError,
        ValueError,
    ) as exc:
        print(
            f"answer-disagreement evaluation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
