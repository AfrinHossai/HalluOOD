"""Evaluate the fixed POPE entropy pilot under controlled image shifts."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    sys.path.insert(1, str(project_root / "src"))

from halluood.distribution_shift_evaluation import (
    render_severity_plot,
    summarize_distribution_shift,
)
from halluood.model import (
    RECOMMENDED_MODEL_ID,
    RECOMMENDED_MODEL_REVISION,
    GenerationSettings,
    TransformersVLMAdapter,
)
from halluood.shifts import (
    SHIFT_PARAMETERS,
    apply_distribution_shift,
    derive_shift_seed,
)
from halluood.uncertainty import score_binary_answer_entropy
from scripts.evaluate_entropy import DEFAULT_OUTPUT as DEFAULT_CLEAN_EVALUATION
from scripts.run_baseline import DEFAULT_OUTPUT as DEFAULT_BASELINE
from scripts.run_baseline import normalize_prediction

DEFAULT_IMAGE_DIR = Path("data/images/val2014")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "halluood"
DEFAULT_OUTPUT_DIR = Path("data/results/distribution_shift")
EXPECTED_SAMPLE_SIZE = 30
MODEL_FIELDS = (
    "model_identifier",
    "model_revision",
    "device",
    "precision",
    "torch_version",
    "transformers_version",
    "seed",
    "generation_settings",
)


class ShiftExperimentError(ValueError):
    """Raised when the clean sample or shift experiment is inconsistent."""


def _read_jsonl(path: Path, *, allow_missing: bool = False) -> list[dict[str, Any]]:
    if allow_missing and not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ShiftExperimentError(f"could not read {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ShiftExperimentError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ShiftExperimentError(f"expected a JSON object in {path}:{line_number}")
        rows.append(row)
    return rows


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    try:
        return str(row["category"]), str(row["question_id"]), str(row["image_id"])
    except KeyError as exc:
        raise ShiftExperimentError(f"clean row is missing identity field {exc.args[0]!r}") from exc


def _assert_constant_metadata(rows: Sequence[Mapping[str, Any]], file_label: str) -> dict[str, Any]:
    if not rows:
        raise ShiftExperimentError(f"{file_label} contains no rows")
    metadata = {field: rows[0].get(field) for field in MODEL_FIELDS}
    for field in MODEL_FIELDS:
        if any(row.get(field) != metadata[field] for row in rows):
            raise ShiftExperimentError(f"{field} varies within {file_label}")
        if metadata[field] is None:
            raise ShiftExperimentError(f"{file_label} is missing model metadata field {field!r}")
    if metadata["model_identifier"] != RECOMMENDED_MODEL_ID:
        raise ShiftExperimentError("the clean sample is not from the pinned SmolVLM checkpoint")
    if metadata["model_revision"] != RECOMMENDED_MODEL_REVISION:
        raise ShiftExperimentError("the clean sample uses a different model revision")
    return metadata


def _validate_clean_sample(
    baseline_rows: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    if len(baseline_rows) != EXPECTED_SAMPLE_SIZE or len(clean_rows) != EXPECTED_SAMPLE_SIZE:
        raise ShiftExperimentError(
            f"expected exactly {EXPECTED_SAMPLE_SIZE} baseline and entropy rows; got "
            f"{len(baseline_rows)} and {len(clean_rows)}"
        )
    baseline_by_id: dict[tuple[str, str, str], dict[str, Any]] = {}
    clean_by_id: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rows, destination, name in (
        (baseline_rows, baseline_by_id, "baseline"),
        (clean_rows, clean_by_id, "clean entropy evaluation"),
    ):
        for row in rows:
            key = _identity(row)
            if key in destination:
                raise ShiftExperimentError(f"duplicate {name} row identity: {key}")
            destination[key] = row
    if set(baseline_by_id) != set(clean_by_id):
        raise ShiftExperimentError("baseline and clean entropy rows do not contain the same pairs")

    baseline_metadata = _assert_constant_metadata(baseline_rows, "baseline results")
    clean_metadata = _assert_constant_metadata(clean_rows, "clean entropy results")
    if baseline_metadata != clean_metadata:
        raise ShiftExperimentError("clean entropy run metadata differs from the baseline")
    required_equal_fields = (
        "image_filename",
        "question",
        "ground_truth_label",
        "normalized_ground_truth_label",
        "raw_prediction",
        "normalized_prediction",
        "prediction_status",
    )
    for key, baseline in baseline_by_id.items():
        clean = clean_by_id[key]
        if clean.get("reference_prediction_match") is not True:
            raise ShiftExperimentError(f"clean entropy result does not match the baseline: {key}")
        for field in required_equal_fields:
            if clean.get(field) != baseline.get(field):
                raise ShiftExperimentError(f"clean row {key} differs from baseline field {field!r}")
        if not isinstance(clean.get("is_error"), bool) or not isinstance(
            clean.get("is_correct"), bool
        ):
            raise ShiftExperimentError(f"clean row {key} is missing correctness labels")
        if clean["is_error"] == clean["is_correct"]:
            raise ShiftExperimentError(f"clean row {key} has inconsistent correctness labels")
        if not isinstance(clean.get("normalized_entropy"), (int, float)):
            raise ShiftExperimentError(f"clean row {key} is missing normalized entropy")
    if len({row["image_id"] for row in baseline_rows}) != EXPECTED_SAMPLE_SIZE:
        raise ShiftExperimentError("the baseline sample must contain 30 distinct images")
    return baseline_by_id, baseline_metadata


def _validate_image(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"COCO image is missing: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise ShiftExperimentError(f"COCO image is unreadable: {path}: {exc}") from exc


def _sync(device: str) -> None:
    if device == "mps":
        import torch

        torch.mps.synchronize()


def _job_key(
    category: str,
    question_id: str,
    image_id: str,
    perturbation_type: str,
    severity: int,
) -> tuple[str, str, str, str, int]:
    return category, question_id, image_id, perturbation_type, severity


def _ordered_jobs(
    clean_by_id: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[tuple[tuple[str, str, str, str, int], Mapping[str, Any], str, int]]:
    jobs = []
    for perturbation_type in SHIFT_PARAMETERS:
        for severity in (1, 2, 3):
            for (category, question_id, image_id), row in clean_by_id.items():
                key = _job_key(category, question_id, image_id, perturbation_type, severity)
                jobs.append((key, row, perturbation_type, severity))
    return jobs


def _is_complete_result(row: Mapping[str, Any]) -> bool:
    return (
        row.get("prediction_status") in {"parsed", "ambiguous"}
        and isinstance(row.get("uncertainty_score"), Mapping)
        and isinstance(row.get("normalized_entropy"), (int, float))
    )


def _validate_existing_rows(
    rows: list[dict[str, Any]],
    jobs: list[tuple[tuple[str, str, str, str, int], Mapping[str, Any], str, int]],
    clean_metadata: Mapping[str, Any],
    experiment_seed: int,
) -> dict[tuple[str, str, str, str, int], dict[str, Any]]:
    expected = {key: (clean, kind, severity) for key, clean, kind, severity in jobs}
    existing: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        try:
            key = _job_key(
                str(row["category"]),
                str(row["question_id"]),
                str(row["image_id"]),
                str(row["perturbation_type"]),
                row["severity"],
            )
        except KeyError as exc:
            raise ShiftExperimentError(
                f"existing result row {index} lacks {exc.args[0]!r}"
            ) from exc
        if key not in expected:
            raise ShiftExperimentError(f"existing results contain an unexpected row: {key}")
        if key in existing:
            raise ShiftExperimentError(f"existing results contain a duplicate row: {key}")
        clean, kind, severity = expected[key]
        if row.get("condition_id") != f"{kind}_severity_{severity}":
            raise ShiftExperimentError(f"existing result row {key} has a mismatched condition ID")
        if row.get("parameters") != SHIFT_PARAMETERS[kind][severity - 1]:
            raise ShiftExperimentError(f"existing result row {key} has different shift parameters")
        expected_seed = derive_shift_seed(experiment_seed, key[2], kind, severity)
        if row.get("random_seed") != expected_seed:
            raise ShiftExperimentError(f"existing result row {key} has a different shift seed")
        for field in MODEL_FIELDS:
            if row.get(field) != clean_metadata[field]:
                raise ShiftExperimentError(f"existing result row {key} differs in {field}")
        for field in (
            "image_filename",
            "question",
            "ground_truth_label",
            "normalized_ground_truth_label",
        ):
            if row.get(field) != clean.get(field):
                raise ShiftExperimentError(f"existing result row {key} changed {field}")
        existing[key] = row
    return existing


def _adapter_metadata_matches(
    adapter: TransformersVLMAdapter,
    clean_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = adapter.metadata.to_dict()
    mapping = {
        "model_identifier": metadata["model_id"],
        "model_revision": metadata["resolved_revision"] or metadata["requested_revision"],
        "device": metadata["device"],
        "precision": metadata["precision"],
        "torch_version": metadata["torch_version"],
        "transformers_version": metadata["transformers_version"],
        "seed": metadata["generation"]["seed"],
        "generation_settings": metadata["generation"],
    }
    for field, value in mapping.items():
        if value != clean_metadata[field]:
            raise ShiftExperimentError(
                f"shift inference settings differ from the clean baseline in {field}"
            )
    return metadata


def _run_one_job(
    *,
    adapter: TransformersVLMAdapter,
    clean_row: Mapping[str, Any],
    image_path: Path,
    perturbation_type: str,
    severity: int,
    experiment_seed: int,
    model_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    image_id = str(clean_row["image_id"])
    perturbation_seed = derive_shift_seed(
        experiment_seed, image_id, perturbation_type, severity
    )
    transform_started = time.perf_counter()
    shifted = apply_distribution_shift(
        image_path,
        image_id=image_id,
        perturbation_type=perturbation_type,
        severity=severity,
        random_seed=perturbation_seed,
    )
    transform_time = time.perf_counter() - transform_started
    raw_prediction: str | None = None
    normalized_prediction: str | None = None
    prediction_status = "error"
    prediction_error: str | None = None
    generation_time: float | None = None
    uncertainty_time: float | None = None
    uncertainty: dict[str, Any] | None = None
    uncertainty_error: str | None = None
    try:
        _sync(adapter.metadata.device)
        started = time.perf_counter()
        try:
            raw_prediction = adapter.generate_answer(shifted.image, str(clean_row["question"]))
            normalized_prediction, prediction_status = normalize_prediction(raw_prediction)
        except Exception as exc:
            prediction_error = f"{type(exc).__name__}: {exc}"
            prediction_status = "error"
        finally:
            try:
                _sync(adapter.metadata.device)
            except Exception as exc:
                prediction_error = prediction_error or f"{type(exc).__name__}: {exc}"
                prediction_status = "error"
                normalized_prediction = None
            generation_time = time.perf_counter() - started

        if prediction_status != "error":
            try:
                _sync(adapter.metadata.device)
                started = time.perf_counter()
                uncertainty = score_binary_answer_entropy(
                    adapter, shifted.image, str(clean_row["question"])
                ).to_dict()
            except Exception as exc:
                uncertainty_error = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    _sync(adapter.metadata.device)
                except Exception as exc:
                    uncertainty_error = uncertainty_error or f"{type(exc).__name__}: {exc}"
                if "started" in locals():
                    uncertainty_time = time.perf_counter() - started
    finally:
        shifted.image.close()

    is_correct = (
        prediction_status == "parsed"
        and normalized_prediction == clean_row["normalized_ground_truth_label"]
    )
    is_error = not is_correct
    score_value = uncertainty.get("normalized_entropy") if uncertainty is not None else None
    row = {
        "category": clean_row["category"],
        "question_id": clean_row["question_id"],
        "image_id": image_id,
        "image_filename": clean_row["image_filename"],
        "question": clean_row["question"],
        "ground_truth_label": clean_row["ground_truth_label"],
        "normalized_ground_truth_label": clean_row["normalized_ground_truth_label"],
        "raw_prediction": raw_prediction,
        "normalized_prediction": normalized_prediction,
        "prediction_status": prediction_status,
        "is_correct": is_correct,
        "is_error": is_error,
        "perturbation_type": perturbation_type,
        "severity": severity,
        "condition_id": f"{perturbation_type}_severity_{severity}",
        "parameters": dict(SHIFT_PARAMETERS[perturbation_type][severity - 1]),
        "random_seed": perturbation_seed,
        "experiment_seed": experiment_seed,
        "normalized_entropy": score_value,
        "uncertainty_score": uncertainty,
        "clean_raw_prediction": clean_row.get("raw_prediction"),
        "clean_normalized_prediction": clean_row.get("normalized_prediction"),
        "clean_is_correct": clean_row.get("is_correct"),
        "clean_normalized_entropy": clean_row.get("normalized_entropy"),
        "transform_time_seconds": transform_time,
        "inference_time_seconds": generation_time,
        "uncertainty_time_seconds": uncertainty_time,
        "total_model_time_seconds": sum(
            value for value in (generation_time, uncertainty_time) if value is not None
        ),
        "model_identifier": model_metadata["model_id"],
        "model_revision": model_metadata["resolved_revision"]
        or model_metadata["requested_revision"],
        "device": model_metadata["device"],
        "precision": model_metadata["precision"],
        "torch_version": model_metadata["torch_version"],
        "transformers_version": model_metadata["transformers_version"],
        "seed": model_metadata["generation"]["seed"],
        "generation_settings": model_metadata["generation"],
        "transformed_image_persisted": False,
    }
    if prediction_error is not None:
        row["prediction_error"] = prediction_error
    if uncertainty_error is not None:
        row["uncertainty_error"] = uncertainty_error
    return row


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_answerability_sheets(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    image_dir: Path,
    output_dir: Path,
    experiment_seed: int,
) -> list[str]:
    """Create prediction-blind clean/severity-3 contact sheets for visual review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    columns, rows_per_sheet = 5, 6
    card_width, card_height = 400, 205
    thumb_width, thumb_height = 182, 118
    font = ImageFont.load_default()
    outputs = []
    for perturbation_type in SHIFT_PARAMETERS:
        canvas = Image.new(
            "RGB",
            (columns * card_width, rows_per_sheet * card_height),
            (248, 249, 250),
        )
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(baseline_rows):
            column, row_number = index % columns, index // columns
            x, y = column * card_width, row_number * card_height
            draw.rectangle(
                (x + 2, y + 2, x + card_width - 3, y + card_height - 3),
                outline="#9aa0a6",
            )
            title = (
                f"q={row['question_id']}  image={row['image_id']}  "
                f"{row['category']}  GT={row['normalized_ground_truth_label']}"
            )
            draw.text((x + 8, y + 7), title, fill="#202124", font=font)
            question_lines = textwrap.wrap(str(row["question"]), width=66)[:2]
            draw.multiline_text(
                (x + 8, y + 21),
                "\n".join(question_lines),
                fill="#202124",
                font=font,
                spacing=1,
            )
            clean_path = image_dir / str(row["image_filename"])
            shifted = apply_distribution_shift(
                clean_path,
                image_id=str(row["image_id"]),
                perturbation_type=perturbation_type,
                severity=3,
                random_seed=derive_shift_seed(
                    experiment_seed, str(row["image_id"]), perturbation_type, 3
                ),
            )
            try:
                with Image.open(clean_path) as opened:
                    opened.load()
                    clean_image = opened.convert("RGB")
                draw.text((x + 55, y + 42), "clean", fill="#5f6368", font=font)
                draw.text((x + 236, y + 42), "severity 3", fill="#5f6368", font=font)
                clean_thumb = ImageOps.contain(clean_image, (thumb_width, thumb_height))
                shifted_thumb = ImageOps.contain(shifted.image, (thumb_width, thumb_height))
                draw_center_x = x + 8 + (thumb_width - clean_thumb.width) // 2
                draw_center_y = y + 58 + (thumb_height - clean_thumb.height) // 2
                canvas.paste(clean_thumb, (draw_center_x, draw_center_y))
                draw_center_x = x + 204 + (thumb_width - shifted_thumb.width) // 2
                draw_center_y = y + 58 + (thumb_height - shifted_thumb.height) // 2
                canvas.paste(shifted_thumb, (draw_center_x, draw_center_y))
                clean_image.close()
            finally:
                shifted.image.close()
        output_path = output_dir / f"{perturbation_type}_severity_3_review.png"
        canvas.save(output_path, format="PNG")
        canvas.close()
        outputs.append(str(output_path))
    return outputs


def run_distribution_shift_evaluation(
    *,
    baseline_file: str | Path = DEFAULT_BASELINE,
    clean_evaluation_file: str | Path = DEFAULT_CLEAN_EVALUATION,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    experiment_seed: int = 2027,
    device: str | None = None,
    answerability_sheets_only: bool = False,
) -> dict[str, Any]:
    """Run or resume the 30-pair, 12-condition image-shift evaluation."""

    if (
        isinstance(experiment_seed, bool)
        or not isinstance(experiment_seed, int)
        or experiment_seed < 0
    ):
        raise ShiftExperimentError("experiment_seed must be a non-negative integer")
    baseline_rows = _read_jsonl(Path(baseline_file))
    clean_rows = _read_jsonl(Path(clean_evaluation_file))
    clean_by_id, clean_metadata = _validate_clean_sample(baseline_rows, clean_rows)
    clean_reference_seed = clean_metadata["seed"]
    if isinstance(clean_reference_seed, bool) or not isinstance(clean_reference_seed, int):
        raise ShiftExperimentError("baseline seed must be an integer")
    generation = clean_metadata["generation_settings"]
    if (
        not isinstance(generation, Mapping)
        or generation.get("do_sample") is not False
        or generation.get("num_beams") != 1
        or generation.get("seed") != clean_reference_seed
    ):
        raise ShiftExperimentError(
            "the clean baseline must use its recorded deterministic settings"
        )
    if device is not None and device != clean_metadata["device"]:
        raise ShiftExperimentError("requested device differs from the clean baseline device")

    images_root = Path(image_dir)
    for row in baseline_rows:
        _validate_image(images_root / str(row["image_filename"]))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    review_paths = _make_answerability_sheets(
        baseline_rows=baseline_rows,
        image_dir=images_root,
        output_dir=output_root / "answerability_review",
        experiment_seed=experiment_seed,
    )
    if answerability_sheets_only:
        result = {
            "answerability_review_sheets": review_paths,
            "question_count": len(baseline_rows),
            "severity_reviewed": 3,
            "perturbation_types": list(SHIFT_PARAMETERS),
            "model_inference_run": False,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    jobs = _ordered_jobs(clean_by_id)
    raw_path = output_root / f"pope_controlled_shifts_predictions_seed_{experiment_seed}.jsonl"
    summary_path = output_root / f"pope_controlled_shifts_evaluation_seed_{experiment_seed}.json"
    accuracy_plot_path = output_root / "accuracy_vs_severity.svg"
    entropy_plot_path = output_root / "entropy_vs_severity.svg"
    existing_rows = _read_jsonl(raw_path, allow_missing=True)
    existing = _validate_existing_rows(existing_rows, jobs, clean_metadata, experiment_seed)
    completed = {key: row for key, row in existing.items() if _is_complete_result(row)}
    # Discard failed partial rows so the same deterministic pair can be retried.
    ordered_completed = [
        completed[key]
        for key, _, _, _ in jobs
        if key in completed
    ]
    _write_jsonl(raw_path, ordered_completed)

    adapter: TransformersVLMAdapter | None = None
    model_metadata: dict[str, Any] | None = None
    if len(completed) < len(jobs):
        settings = GenerationSettings(
            max_new_tokens=int(generation["max_new_tokens"]),
            seed=clean_reference_seed,
            num_beams=int(generation["num_beams"]),
        )
        adapter = TransformersVLMAdapter.from_pretrained(
            model_id=RECOMMENDED_MODEL_ID,
            revision=RECOMMENDED_MODEL_REVISION,
            device=str(clean_metadata["device"]),
            generation=settings,
            cache_dir=cache_dir,
        )
        model_metadata = _adapter_metadata_matches(adapter, clean_metadata)

    finished = len(completed)
    for key, clean_row, perturbation_type, severity in jobs:
        if key in completed:
            continue
        if adapter is None or model_metadata is None:
            raise ShiftExperimentError("adapter was not initialized for incomplete results")
        result = _run_one_job(
            adapter=adapter,
            clean_row=clean_row,
            image_path=images_root / str(clean_row["image_filename"]),
            perturbation_type=perturbation_type,
            severity=severity,
            experiment_seed=experiment_seed,
            model_metadata=model_metadata,
        )
        result_key = _job_key(
            str(result["category"]),
            str(result["question_id"]),
            str(result["image_id"]),
            perturbation_type,
            severity,
        )
        completed[result_key] = result
        _write_jsonl(
            raw_path,
            [completed[item_key] for item_key, _, _, _ in jobs if item_key in completed],
        )
        finished += 1
        if finished % 10 == 0 or finished == len(jobs):
            print(f"Completed {finished}/{len(jobs)} shifted pairs", flush=True)

    shifted_rows = [completed[key] for key, _, _, _ in jobs]
    metrics = summarize_distribution_shift(
        clean_rows,
        shifted_rows,
        expected_count=EXPECTED_SAMPLE_SIZE,
    )
    accuracy_plot_path.write_text(
        render_severity_plot(
            metrics["conditions"],
            metric="accuracy",
            title="POPE accuracy under controlled image shifts",
            y_label="Accuracy",
        ),
        encoding="utf-8",
    )
    entropy_plot_path.write_text(
        render_severity_plot(
            metrics["conditions"],
            metric="mean_normalized_entropy",
            title="Binary predictive entropy under controlled image shifts",
            y_label="Mean normalized binary entropy",
        ),
        encoding="utf-8",
    )
    clean_condition_rows = []
    for row in clean_rows:
        clean_copy = dict(row)
        clean_copy.update(
            {
                "condition_id": "clean",
                "perturbation_type": "none",
                "severity": 0,
                "parameters": {},
                "random_seed": None,
            }
        )
        clean_condition_rows.append(clean_copy)
    all_predictions_path = output_root / (
        f"pope_controlled_shifts_all_conditions_seed_{experiment_seed}.jsonl"
    )
    _write_jsonl(all_predictions_path, [*clean_condition_rows, *shifted_rows])

    evaluation_summary = {
        **metrics,
        "baseline_results_file": str(baseline_file),
        "clean_entropy_evaluation_file": str(clean_evaluation_file),
        "shifted_predictions_file": str(raw_path),
        "all_conditions_predictions_file": str(all_predictions_path),
        "accuracy_plot_file": str(accuracy_plot_path),
        "entropy_plot_file": str(entropy_plot_path),
        "answerability_review_sheets": review_paths,
        "experiment_seed": experiment_seed,
        "perturbation_seed_procedure": (
            "derive a per-image/condition seed from the first 32 bits of "
            "SHA-256(seed:image_id:type:severity)"
        ),
        "parameters_by_condition": {
            kind: {str(level): SHIFT_PARAMETERS[kind][level - 1] for level in (1, 2, 3)}
            for kind in SHIFT_PARAMETERS
        },
        "model_metadata": model_metadata or _model_metadata_from_rows(shifted_rows[0]),
        "reused_clean_sample": {
            "question_count": len(clean_rows),
            "distinct_image_count": len({row["image_id"] for row in clean_rows}),
            "prediction_matches_baseline": sum(
                row.get("reference_prediction_match") is True for row in clean_rows
            ),
        },
        "limitations": [
            "POPE uses object-presence labels; this is not a general hallucination annotation.",
            "The 30-pair pilot is small and estimates are imprecise.",
            "The same official labels are retained for all shifts; visually unanswerable "
            "items are not silently relabeled.",
            "Synthetic corruption does not represent every OOD or domain shift.",
        ],
    }
    summary_path.write_text(
        json.dumps(evaluation_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evaluation_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return evaluation_summary


def _model_metadata_from_rows(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": row["model_identifier"],
        "requested_revision": row["model_revision"],
        "resolved_revision": row["model_revision"],
        "device": row["device"],
        "precision": row["precision"],
        "torch_version": row["torch_version"],
        "transformers_version": row["transformers_version"],
        "generation": row["generation_settings"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed 30-question POPE sample across four image shifts at three severities."
        )
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--clean-evaluation", type=Path, default=DEFAULT_CLEAN_EVALUATION)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", choices=("mps", "cpu"), default=None)
    parser.add_argument(
        "--answerability-sheets-only",
        action="store_true",
        help="create clean/severity-3 visual review sheets and exit without model inference",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_distribution_shift_evaluation(
            baseline_file=args.baseline,
            clean_evaluation_file=args.clean_evaluation,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            experiment_seed=args.seed,
            device=args.device,
            answerability_sheets_only=args.answerability_sheets_only,
        )
    except Exception as exc:
        print(f"distribution-shift evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
