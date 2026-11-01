"""Run repeated sampled answers for one saved POPE baseline pair."""

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

from halluood.data import PopeDataError
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
    AugmentationEvaluationError,
    _generation_settings,
    _read_jsonl,
    _select_annotation,
    _select_baseline_row,
    _verify_model_settings,
)
from scripts.run_baseline import normalize_prediction

DEFAULT_QUESTION_ID = "2813"


class AnswerDisagreementRunError(ValueError):
    """Raised when a sampled-answer experiment cannot be completed safely."""


def run_answer_disagreement(
    *,
    baseline_results_file: str | Path = DEFAULT_BASELINE_RESULTS,
    annotations_dir: str | Path = DEFAULT_ANNOTATIONS_DIR,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    output_file: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    question_id: str = DEFAULT_QUESTION_ID,
    settings: SamplingSettings | None = None,
    device: str | None = None,
    adapter: VLMAdapter | None = None,
) -> dict[str, Any]:
    """Evaluate sampled-answer disagreement on one existing baseline question.

    The saved baseline response is included for context but is not regenerated
    or counted in the empirical sampled-answer distribution.
    """

    if settings is None:
        sampling = SamplingSettings()
    elif isinstance(settings, SamplingSettings):
        sampling = settings
    else:
        raise AnswerDisagreementRunError("settings must be a SamplingSettings instance")
    baseline_rows = _read_jsonl(Path(baseline_results_file))
    baseline_row = _select_baseline_row(baseline_rows, str(question_id))
    annotation = _select_annotation(Path(annotations_dir), baseline_row)
    image_path = Path(image_dir) / annotation.image
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise AnswerDisagreementRunError(
            f"could not read POPE image {image_path}: {exc}"
        ) from exc

    if adapter is None:
        generation = _generation_settings(baseline_row)
        resolved_device = None if device in {None, "auto"} else device
        try:
            adapter = TransformersVLMAdapter.from_pretrained(
                model_id=str(baseline_row["model_identifier"]),
                revision=str(baseline_row["model_revision"]),
                device=resolved_device,
                generation=generation,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            raise AnswerDisagreementRunError(
                f"could not load the baseline model checkpoint: {type(exc).__name__}: {exc}"
            ) from exc
    _verify_model_settings(baseline_row, adapter)

    start_time = time.perf_counter()
    try:
        raw_answers = adapter.generate_sampled_answers(
            image_path,
            annotation.text,
            sampling,
        )
    except Exception as exc:
        raise AnswerDisagreementRunError(
            f"sampled inference failed for question {annotation.question_id!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    inference_time = time.perf_counter() - start_time
    if not isinstance(raw_answers, (list, tuple)):
        raise AnswerDisagreementRunError("sampled inference must return a list of answer strings")
    if len(raw_answers) != sampling.num_generations:
        raise AnswerDisagreementRunError(
            f"sampled inference returned {len(raw_answers)} answers; "
            f"expected {sampling.num_generations}"
        )
    if any(not isinstance(answer, str) for answer in raw_answers):
        raise AnswerDisagreementRunError("sampled inference returned a non-string answer")

    sampled_rows = []
    normalized_answers = []
    for index, raw_answer in enumerate(raw_answers):
        normalized, status = normalize_prediction(raw_answer)
        label = normalized if normalized in {"yes", "no"} else "ambiguous"
        normalized_answers.append(label)
        sampled_rows.append(
            {
                "generation_index": index,
                "seed": sampling.generation_seeds[index],
                "raw_answer": raw_answer,
                "normalized_answer": label,
                "parser_status": status,
            }
        )

    score = answer_disagreement_score(normalized_answers)
    result = {
        "question_id": annotation.question_id,
        "image_id": baseline_row["image_id"],
        "image_filename": annotation.image,
        "category": annotation.category,
        "question": annotation.text,
        "ground_truth_label": annotation.normalized_label,
        "original_baseline_prediction": {
            "raw_answer": baseline_row.get("raw_prediction"),
            "normalized_answer": baseline_row.get("normalized_prediction"),
            "parser_status": baseline_row.get("prediction_status"),
            "source": "unchanged saved baseline result",
        },
        "model_metadata": adapter.metadata.to_dict(),
        "sampling_settings": sampling.to_dict(),
        "sampled_answers": sampled_rows,
        "answer_disagreement": score.to_dict(),
        "inference_time_seconds": inference_time,
        "baseline_results_file": str(baseline_results_file),
    }
    output_path = (
        Path(output_file)
        if output_file is not None
        else Path("data/results")
        / f"answer_disagreement_q{annotation.question_id}_seed_{sampling.seed}.json"
    )
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise AnswerDisagreementRunError(
            f"could not save sampled-answer results to {output_path}: {exc}"
        ) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled sampled-answer generations for one existing POPE baseline pair."
        )
    )
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--question-id", default=DEFAULT_QUESTION_ID)
    parser.add_argument("--num-generations", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = SamplingSettings(
            num_generations=args.num_generations,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
        run_answer_disagreement(
            baseline_results_file=args.baseline_results,
            annotations_dir=args.annotations_dir,
            image_dir=args.image_dir,
            output_file=args.output,
            cache_dir=args.cache_dir,
            question_id=args.question_id,
            settings=settings,
            device=None if args.device == "auto" else args.device,
        )
    except (
        AnswerDisagreementRunError,
        AugmentationEvaluationError,
        ModelAdapterError,
        PopeDataError,
        OSError,
        ValueError,
    ) as exc:
        print(
            f"answer-disagreement run failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
