"""Run controlled image shifts on a fixed held-out subset from the 1,800-row run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluood.distribution_shift_evaluation import (
    render_severity_plot,
    summarize_distribution_shift,
)
from halluood.model import (
    RECOMMENDED_MODEL_ID,
    RECOMMENDED_MODEL_REVISION,
    GenerationSettings,
    TransformersVLMAdapter,
    VLMAdapter,
)
from halluood.predictions import normalize_prediction
from halluood.robustness import image_grouped_bootstrap
from halluood.shifts import SHIFT_PARAMETERS, apply_distribution_shift, derive_shift_seed
from halluood.uncertainty import score_binary_answer_entropy

DEFAULT_FULL = Path("data/results/robustness/full_pope_predictions_seed_2027.jsonl")
DEFAULT_SPLIT = Path("data/results/robustness/expanded_image_split_seed_2027.json")
DEFAULT_IMAGE_DIR = Path("data/images/val2014")
DEFAULT_OUTPUT = Path("data/results/robustness/expanded_shift_predictions_seed_2027.jsonl")
DEFAULT_CACHE = Path.home() / ".cache" / "huggingface" / "halluood"


class ExpandedShiftError(ValueError):
    """Raised when shift inputs do not match the frozen full evaluation."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExpandedShiftError(f"invalid JSONL line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ExpandedShiftError(f"JSONL line {line_number} is not an object")
        rows.append(row)
    return rows


def _sync(device: str) -> None:
    if device == "mps":
        import torch

        torch.mps.synchronize()


def _key(row: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(row["category"]),
        str(row["question_id"]),
        str(row["image_id"]),
        str(row["perturbation_type"]),
        int(row["severity"]),
    )


def run_expanded_shifts(
    *,
    full_file: str | Path = DEFAULT_FULL,
    split_file: str | Path = DEFAULT_SPLIT,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    output_file: str | Path = DEFAULT_OUTPUT,
    cache_dir: str | Path = DEFAULT_CACHE,
    device: str = "mps",
    experiment_seed: int = 2027,
    subset_seed: int = 2028,
    image_count: int = 3,
    max_new_tokens: int = 32,
    adapter: VLMAdapter | None = None,
) -> dict[str, Any]:
    """Evaluate 12 existing shift conditions on all 18 questions from each sampled image."""

    full_path = Path(full_file)
    image_root = Path(image_dir)
    output = Path(output_file)
    full_rows = _read_jsonl(full_path)
    if len(full_rows) != 1800:
        raise ExpandedShiftError(f"expected 1,800 clean rows; received {len(full_rows)}")
    split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    calibration_ids = {str(value) for value in split["calibration_image_ids"]}
    evaluation_ids = sorted(str(value) for value in split["evaluation_image_ids"])
    reference_ids = {str(value) for value in split["reference_image_ids"]}
    if (
        calibration_ids & set(evaluation_ids)
        or calibration_ids & reference_ids
        or set(evaluation_ids) & reference_ids
    ):
        raise ExpandedShiftError(
            "calibration, evaluation, and reference image IDs must be disjoint"
        )
    if image_count < 1 or image_count > len(evaluation_ids):
        raise ExpandedShiftError("image_count must be between one and the held-out image count")
    rng = np.random.Generator(np.random.PCG64(subset_seed))
    selected_ids = sorted(
        str(value)
        for value in rng.choice(
            np.asarray(evaluation_ids, dtype=object), size=image_count, replace=False
        )
    )
    if set(selected_ids) & (calibration_ids | reference_ids):
        raise ExpandedShiftError("selected shift images overlap calibration or reference images")
    clean_rows = [row for row in full_rows if str(row["image_id"]) in set(selected_ids)]
    identities = {
        (str(row["category"]), str(row["question_id"]), str(row["image_id"])) for row in clean_rows
    }
    expected_clean_count = image_count * 18
    if len(clean_rows) != expected_clean_count or len(identities) != expected_clean_count:
        raise ExpandedShiftError(
            f"expected {expected_clean_count} clean rows for the selected images; "
            f"found {len(clean_rows)}"
        )
    if any(row.get("normalized_entropy") is None for row in clean_rows):
        raise ExpandedShiftError("selected clean rows need entropy scores")

    settings = GenerationSettings(max_new_tokens=max_new_tokens, seed=experiment_seed)
    if adapter is None:
        adapter = TransformersVLMAdapter.from_pretrained(
            model_id=RECOMMENDED_MODEL_ID,
            revision=RECOMMENDED_MODEL_REVISION,
            device=device,
            generation=settings,
            cache_dir=cache_dir,
        )
    metadata = adapter.metadata.to_dict()
    revision = metadata.get("resolved_revision") or metadata.get("requested_revision")
    if metadata.get("model_id") != RECOMMENDED_MODEL_ID or revision != RECOMMENDED_MODEL_REVISION:
        raise ExpandedShiftError(
            "adapter model identifier or revision differs from clean evaluation"
        )
    if metadata.get("generation", {}).get("seed") != experiment_seed:
        raise ExpandedShiftError("adapter seed differs from clean evaluation")

    output.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    if output.is_file():
        lines = output.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if line_number == len(lines):
                    break
                raise ExpandedShiftError(f"invalid resume JSONL line {line_number}: {exc}") from exc
            key = _key(row)
            if key in completed:
                raise ExpandedShiftError(f"duplicate resumed shift row: {key}")
            completed[key] = row
    completed = {
        key: row
        for key, row in completed.items()
        if row.get("prediction_status") != "error"
        and row.get("normalized_entropy") is not None
        and row.get("model_revision") == revision
        and row.get("experiment_seed") == experiment_seed
    }
    expected = {
        (*identity, perturbation_type, severity)
        for identity in identities
        for perturbation_type in SHIFT_PARAMETERS
        for severity in (1, 2, 3)
    }
    unexpected = set(completed) - expected
    if unexpected:
        raise ExpandedShiftError(
            f"resume file has rows outside this selected experiment: {list(unexpected)[:3]}"
        )
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in completed.values()
        ),
        encoding="utf-8",
    )
    done = len(completed)
    started = time.perf_counter()
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for perturbation_type in SHIFT_PARAMETERS:
            for severity in (1, 2, 3):
                for image_id in selected_ids:
                    image_rows = [row for row in clean_rows if str(row["image_id"]) == image_id]
                    if not image_rows:
                        raise ExpandedShiftError(f"selected image has no POPE rows: {image_id}")
                    seed = derive_shift_seed(experiment_seed, image_id, perturbation_type, severity)
                    shifted = apply_distribution_shift(
                        image_root / image_rows[0]["image_filename"],
                        image_id=image_id,
                        perturbation_type=perturbation_type,
                        severity=severity,
                        random_seed=seed,
                    )
                    try:
                        for clean in image_rows:
                            identity = (
                                str(clean["category"]),
                                str(clean["question_id"]),
                                str(clean["image_id"]),
                            )
                            key = (*identity, perturbation_type, severity)
                            if key in completed:
                                continue
                            _sync(metadata["device"])
                            start = time.perf_counter()
                            raw_prediction: str | None = None
                            normalized_prediction: str | None = None
                            prediction_status = "error"
                            prediction_error: str | None = None
                            try:
                                raw_prediction = adapter.generate_answer(
                                    shifted.image, clean["question"]
                                )
                                normalized_prediction, prediction_status = normalize_prediction(
                                    raw_prediction
                                )
                            except Exception as exc:
                                prediction_error = f"{type(exc).__name__}: {exc}"
                            finally:
                                _sync(metadata["device"])
                            inference_seconds = time.perf_counter() - start
                            uncertainty: dict[str, Any] | None = None
                            uncertainty_error: str | None = None
                            uncertainty_seconds: float | None = None
                            if prediction_status != "error":
                                _sync(metadata["device"])
                                entropy_start = time.perf_counter()
                                try:
                                    uncertainty = score_binary_answer_entropy(
                                        adapter, shifted.image, clean["question"]
                                    ).to_dict()
                                except Exception as exc:
                                    uncertainty_error = f"{type(exc).__name__}: {exc}"
                                finally:
                                    _sync(metadata["device"])
                                uncertainty_seconds = time.perf_counter() - entropy_start
                            is_correct = (
                                prediction_status == "parsed"
                                and normalized_prediction == clean["normalized_ground_truth_label"]
                            )
                            row: dict[str, Any] = {
                                "question_id": clean["question_id"],
                                "image_id": image_id,
                                "image_filename": clean["image_filename"],
                                "category": clean["category"],
                                "question": clean["question"],
                                "ground_truth_label": clean["ground_truth_label"],
                                "normalized_ground_truth_label": clean[
                                    "normalized_ground_truth_label"
                                ],
                                "raw_prediction": raw_prediction,
                                "normalized_prediction": normalized_prediction,
                                "prediction_status": prediction_status,
                                "is_correct": is_correct,
                                "is_error": not is_correct,
                                "normalized_entropy": uncertainty["normalized_entropy"]
                                if uncertainty
                                else None,
                                "uncertainty_score": uncertainty,
                                "model_identifier": metadata["model_id"],
                                "model_revision": revision,
                                "device": metadata["device"],
                                "seed": experiment_seed,
                                "experiment_seed": experiment_seed,
                                "generation_settings": metadata["generation"],
                                "inference_time_seconds": inference_seconds,
                                "uncertainty_time_seconds": uncertainty_seconds,
                                "perturbation_type": perturbation_type,
                                "severity": severity,
                                "condition_id": shifted.condition_id,
                                "perturbation_parameters": dict(shifted.parameters),
                                "perturbation_seed": seed,
                                "transformed_width": shifted.image.width,
                                "transformed_height": shifted.image.height,
                            }
                            if prediction_error:
                                row["prediction_error"] = prediction_error
                            if uncertainty_error:
                                row["uncertainty_error"] = uncertainty_error
                            completed[key] = row
                            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                            stream.flush()
                            done += 1
                            if done == 1 or done % 25 == 0:
                                print(
                                    f"[{done}/{len(expected)}] "
                                    f"elapsed={time.perf_counter() - started:.1f}s",
                                    flush=True,
                                )
                    finally:
                        shifted.image.close()

    shifted_rows = [completed[key] for key in sorted(completed)]
    if len(shifted_rows) != len(expected):
        raise ExpandedShiftError(
            f"expected {len(expected)} shifted rows, found {len(shifted_rows)}"
        )
    summary = summarize_distribution_shift(
        clean_rows, shifted_rows, expected_count=expected_clean_count
    )
    bootstrap_by_condition = {}
    for condition in summary["conditions"]:
        condition_id = condition["condition_id"]
        if condition_id == "clean":
            condition_rows = clean_rows
        else:
            condition_rows = [row for row in shifted_rows if row["condition_id"] == condition_id]
        intervals = image_grouped_bootstrap(condition_rows, replicates=1000, seed=experiment_seed)
        condition["confidence_interval"] = intervals["intervals"]["overall"]
        bootstrap_by_condition[condition_id] = intervals
    summary.update(
        {
            "subset_selection_seed": subset_seed,
            "experiment_seed": experiment_seed,
            "selected_evaluation_image_ids": selected_ids,
            "calibration_image_ids_excluded": sorted(calibration_ids),
            "reference_image_ids_excluded": sorted(reference_ids),
            "image_count": image_count,
            "questions_per_image": 18,
            "heldout_clean_pair_count": expected_clean_count,
            "shifted_inference_count": len(shifted_rows),
            "model_metadata": metadata,
            "raw_results_file": str(output),
            "image_grouped_bootstrap_by_condition": bootstrap_by_condition,
        }
    )
    out_dir = output.parent
    summary_path = out_dir / "expanded_shift_summary_seed_2027.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for metric, filename, title, label in (
        (
            "accuracy",
            "expanded_shift_accuracy_vs_severity.svg",
            "Expanded subset accuracy by shift severity",
            "Accuracy",
        ),
        (
            "mean_normalized_entropy",
            "expanded_shift_entropy_vs_severity.svg",
            "Expanded subset entropy by shift severity",
            "Mean normalized binary entropy",
        ),
    ):
        (out_dir / filename).write_text(
            render_severity_plot(summary["conditions"], metric=metric, title=title, y_label=label),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-results", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--subset-seed", type=int, default=2028)
    parser.add_argument("--image-count", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_expanded_shifts(
            full_file=args.full_results,
            split_file=args.split_file,
            image_dir=args.image_dir,
            output_file=args.output,
            cache_dir=args.cache_dir,
            device=args.device,
            experiment_seed=args.seed,
            subset_seed=args.subset_seed,
            image_count=args.image_count,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as exc:
        print(f"expanded shift evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
