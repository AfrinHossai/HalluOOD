"""Publication-ready, dependency-light figures and failure summaries for HalluOOD."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from halluood.evaluation import EvaluationError, binary_auprc, binary_auroc, risk_coverage_curve
from halluood.robustness import summarize_expanded_rows

PILOT_SCORE_FILES = {
    "predictive_entropy": "pope_entropy_30_seed_2027_evaluated.jsonl",
    "cosine_distance": ("embedding_distance/pope_embedding_entropy_30_seed_2027_scores.jsonl"),
    "nearest_neighbor_distance": (
        "embedding_distance/pope_embedding_entropy_30_seed_2027_scores.jsonl"
    ),
    "augmentation_consistency": (
        "augmentation_evaluation/pope_augmentation_consistency_30_seed_2027_scores.jsonl"
    ),
    "answer_disagreement": (
        "answer_disagreement_evaluation/pope_answer_disagreement_30_seed_2027_scores.jsonl"
    ),
}
PILOT_SCORE_FIELDS = {
    "predictive_entropy": "normalized_entropy",
    "cosine_distance": "cosine_distance",
    "nearest_neighbor_distance": "nearest_neighbor_distance",
    "augmentation_consistency": "augmentation_consistency_score",
    "answer_disagreement": "answer_disagreement_score",
}
METHOD_LABELS = {
    "predictive_entropy": "Predictive entropy",
    "cosine_distance": "Cosine distance",
    "nearest_neighbor_distance": "Nearest-neighbor distance",
    "augmentation_consistency": "Augmentation consistency",
    "answer_disagreement": "Sampled answer disagreement",
}
METHOD_COLORS = {
    "predictive_entropy": "#315A9A",
    "cosine_distance": "#168C8C",
    "nearest_neighbor_distance": "#D36B3D",
    "augmentation_consistency": "#7C5AA6",
    "answer_disagreement": "#C58A16",
}
SHIFT_COLORS = {
    "gaussian_blur": "#2878B5",
    "brightness_reduction": "#D17A22",
    "gaussian_noise": "#7B5BA7",
    "contrast_reduction": "#2A9D72",
}
SHIFT_LABELS = {
    "gaussian_blur": "Gaussian blur",
    "brightness_reduction": "Brightness reduction",
    "gaussian_noise": "Gaussian noise",
    "contrast_reduction": "Contrast reduction",
}


class FinalFigureError(ValueError):
    """Raised when recorded experiment outputs are missing or inconsistent."""


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a local experiment output."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalFigureError(f"could not read JSON results at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FinalFigureError(f"expected a JSON object in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read local JSONL result rows while reporting the invalid line."""

    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FinalFigureError(f"could not read result rows at {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FinalFigureError(f"invalid JSON in {path} line {line_number}") from exc
        if not isinstance(row, dict):
            raise FinalFigureError(f"expected an object in {path} line {line_number}")
        rows.append(row)
    if not rows:
        raise FinalFigureError(f"no result rows found in {path}")
    return rows


def _identity(row: Mapping[str, Any], source: str, index: int) -> tuple[str, str, str]:
    try:
        return (str(row["category"]), str(row["question_id"]), str(row["image_id"]))
    except KeyError as exc:
        raise FinalFigureError(
            f"{source} row {index} is missing identity field {exc.args[0]!r}"
        ) from exc


def _recorded_error(row: Mapping[str, Any], source: str, index: int) -> bool:
    value = row.get("is_error")
    if not isinstance(value, bool):
        correct = row.get("is_correct")
        if not isinstance(correct, bool):
            raise FinalFigureError(f"{source} row {index} must have a boolean error label")
        value = not correct
    if isinstance(row.get("is_correct"), bool) and value == row["is_correct"]:
        raise FinalFigureError(f"{source} row {index} has inconsistent correctness labels")
    return value


def image_grouped_metric_intervals(
    rows: Sequence[Mapping[str, Any]],
    score_field: str,
    *,
    replicates: int = 2000,
    seed: int = 2027,
) -> dict[str, Any]:
    """Compute percentile AUROC/AP intervals by resampling whole image groups."""

    if not rows:
        raise FinalFigureError("bootstrap intervals require at least one result row")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise FinalFigureError("bootstrap replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FinalFigureError("bootstrap seed must be a non-negative integer")

    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    labels: list[int] = []
    for index, row in enumerate(rows):
        identity = _identity(row, "bootstrap", index)
        try:
            score = float(row[score_field])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FinalFigureError(
                f"bootstrap row {index} requires numeric {score_field!r}"
            ) from exc
        if not isfinite(score):
            raise FinalFigureError(f"bootstrap row {index} has a non-finite score")
        labels.append(int(_recorded_error(row, "bootstrap", index)))
        grouped[identity[2]].append((index, score))

    image_ids = sorted(grouped)
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = generator.choice(image_ids, size=(replicates, len(image_ids)), replace=True)
    auroc_values: list[float] = []
    auprc_values: list[float] = []
    score_values = [float(row[score_field]) for row in rows]
    for sampled_images in draws:
        selected_indices = [
            index for image_id in sampled_images for index, _score in grouped[str(image_id)]
        ]
        sampled_labels = [labels[index] for index in selected_indices]
        sampled_scores = [score_values[index] for index in selected_indices]
        try:
            auroc_values.append(binary_auroc(sampled_labels, sampled_scores))
        except EvaluationError:
            pass
        try:
            auprc_values.append(binary_auprc(sampled_labels, sampled_scores))
        except EvaluationError:
            pass

    def interval(values: Sequence[float]) -> dict[str, float | int | None]:
        if not values:
            return {
                "lower": None,
                "upper": None,
                "valid_replicates": 0,
                "requested_replicates": replicates,
            }
        return {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "valid_replicates": len(values),
            "requested_replicates": replicates,
        }

    return {
        "method": "image_grouped_percentile_bootstrap",
        "confidence_level": 0.95,
        "seed": seed,
        "unit_of_resampling": "image_id; every question from an image is sampled together",
        "image_count": len(image_ids),
        "auroc": interval(auroc_values),
        "auprc_average_precision": interval(auprc_values),
    }


def image_grouped_risk_coverage_intervals(
    rows: Sequence[Mapping[str, Any]],
    score_field: str,
    *,
    targets: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    replicates: int = 1000,
    seed: int = 2027,
) -> dict[str, Any]:
    """Bootstrap fixed-score-cutoff coverage and risk by resampling images."""

    if not rows:
        raise FinalFigureError("risk-coverage intervals require at least one result row")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise FinalFigureError("bootstrap replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FinalFigureError("bootstrap seed must be a non-negative integer")
    if not targets or any(
        not isfinite(float(target)) or not 0.0 < target <= 1.0 for target in targets
    ):
        raise FinalFigureError("risk-coverage targets must be finite values in (0, 1]")

    image_groups: dict[str, list[int]] = defaultdict(list)
    labels: list[int] = []
    scores: list[float] = []
    for index, row in enumerate(rows):
        identity = _identity(row, "risk-coverage bootstrap", index)
        try:
            score = float(row[score_field])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FinalFigureError(
                f"risk-coverage row {index} requires numeric {score_field!r}"
            ) from exc
        if not isfinite(score):
            raise FinalFigureError(f"risk-coverage row {index} has a non-finite score")
        image_groups[identity[2]].append(index)
        labels.append(int(_recorded_error(row, "risk-coverage bootstrap", index)))
        scores.append(score)

    image_ids = sorted(image_groups)
    score_values = np.asarray(scores, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int8)
    groups = [np.asarray(image_groups[image_id], dtype=np.int64) for image_id in image_ids]
    group_total = np.asarray([len(indices) for indices in groups], dtype=np.float64)
    curve = risk_coverage_curve(labels, scores)
    positive_coverage_points = [point for point in curve if float(point["coverage"]) > 0.0]
    generator = np.random.Generator(np.random.PCG64(seed))
    multiplicities = generator.multinomial(
        len(image_ids), [1.0 / len(image_ids)] * len(image_ids), size=replicates
    )
    points: list[dict[str, Any]] = []

    def interval(values: np.ndarray) -> dict[str, float | int | None]:
        if values.size == 0:
            return {
                "lower": None,
                "upper": None,
                "valid_replicates": 0,
                "requested_replicates": replicates,
            }
        return {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "valid_replicates": int(values.size),
            "requested_replicates": replicates,
        }

    for target in targets:
        observed = min(
            positive_coverage_points,
            key=lambda point: (
                abs(float(point["coverage"]) - float(target)),
                float(point["coverage"]),
            ),
        )
        threshold = float(observed["max_retained_score"])
        retained = score_values <= threshold
        group_retained = np.asarray(
            [float(np.sum(retained[indices])) for indices in groups], dtype=np.float64
        )
        group_errors = np.asarray(
            [float(np.sum(label_values[indices] * retained[indices])) for indices in groups],
            dtype=np.float64,
        )
        denominator = multiplicities @ group_total
        retained_count = multiplicities @ group_retained
        error_count = multiplicities @ group_errors
        boot_coverage = retained_count / denominator
        valid = retained_count > 0
        boot_risk = error_count[valid] / retained_count[valid]
        points.append(
            {
                "target_coverage": float(target),
                "threshold": threshold,
                "observed": {
                    "coverage": float(observed["coverage"]),
                    "risk": float(observed["risk"]),
                },
                "confidence_intervals": {
                    "coverage": interval(boot_coverage),
                    "risk": interval(boot_risk),
                },
            }
        )
    return {
        "method": "image_grouped_percentile_bootstrap_at_frozen_empirical_cutoffs",
        "confidence_level": 0.95,
        "replicates": replicates,
        "seed": seed,
        "unit_of_resampling": "image_id; every question from a sampled image travels together",
        "image_count": len(image_ids),
        "tie_policy": "fixed empirical score thresholds retain or exclude complete tie groups",
        "points": points,
    }


def image_grouped_selective_intervals(
    outcomes: Sequence[Mapping[str, Any]], *, replicates: int = 2000, seed: int = 2027
) -> dict[str, Any]:
    """Bootstrap coverage and selective risk for fixed, already-selected decisions."""

    if not outcomes:
        raise FinalFigureError("selective intervals require recorded evaluation outcomes")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise FinalFigureError("bootstrap replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FinalFigureError("bootstrap seed must be a non-negative integer")

    grouped: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for index, row in enumerate(outcomes):
        try:
            image_id = str(row["image_id"])
            is_error = row["is_error"]
            answered = row["decision"] == "answer"
        except KeyError as exc:
            raise FinalFigureError(
                f"selective outcome {index} lacks required field {exc.args[0]!r}"
            ) from exc
        if not isinstance(is_error, bool) or row["decision"] not in {"answer", "abstain"}:
            raise FinalFigureError(f"selective outcome {index} has an invalid decision or label")
        grouped[image_id].append((answered, is_error))
    image_ids = sorted(grouped)
    group_total = np.asarray([len(grouped[image_id]) for image_id in image_ids], dtype=np.float64)
    group_answered = np.asarray(
        [sum(answered for answered, _error in grouped[image_id]) for image_id in image_ids],
        dtype=np.float64,
    )
    group_answered_errors = np.asarray(
        [
            sum(int(answered and error) for answered, error in grouped[image_id])
            for image_id in image_ids
        ],
        dtype=np.float64,
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    multiplicities = generator.multinomial(
        len(image_ids), [1.0 / len(image_ids)] * len(image_ids), size=replicates
    )
    denominator = multiplicities @ group_total
    answered_count = multiplicities @ group_answered
    error_count = multiplicities @ group_answered_errors
    boot_coverage = answered_count / denominator
    valid = answered_count > 0
    boot_risk = error_count[valid] / answered_count[valid]

    def interval(values: np.ndarray) -> dict[str, float | int | None]:
        return {
            "lower": float(np.quantile(values, 0.025)) if values.size else None,
            "upper": float(np.quantile(values, 0.975)) if values.size else None,
            "valid_replicates": int(values.size),
            "requested_replicates": replicates,
        }

    return {
        "method": "image_grouped_percentile_bootstrap_with_frozen_decisions",
        "confidence_level": 0.95,
        "replicates": replicates,
        "seed": seed,
        "unit_of_resampling": "image_id; all evaluation rows for a sampled image travel together",
        "image_count": len(image_ids),
        "coverage": interval(boot_coverage),
        "selective_risk": interval(boot_risk),
    }


def _metrics_for_score(
    rows: Sequence[Mapping[str, Any]], score_field: str, source_name: str
) -> dict[str, float | int]:
    labels = [int(_recorded_error(row, source_name, index)) for index, row in enumerate(rows)]
    scores: list[float] = []
    for index, row in enumerate(rows):
        try:
            value = float(row[score_field])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FinalFigureError(
                f"{source_name} row {index} requires numeric {score_field!r}"
            ) from exc
        if not isfinite(value):
            raise FinalFigureError(f"{source_name} row {index} has a non-finite score")
        scores.append(value)
    try:
        auroc = binary_auroc(labels, scores)
        auprc = binary_auprc(labels, scores)
    except EvaluationError as exc:
        raise FinalFigureError(f"{source_name} cannot produce both ranking metrics: {exc}") from exc
    return {
        "question_count": len(rows),
        "image_count": len({str(row["image_id"]) for row in rows}),
        "error_count": sum(labels),
        "error_prevalence": float(np.mean(labels)),
        "auroc": auroc,
        "auprc_average_precision": auprc,
    }


def _load_pilot_score_rows(results_dir: Path) -> dict[str, list[dict[str, Any]]]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    cache: dict[str, list[dict[str, Any]]] = {}
    for method, relative_path in PILOT_SCORE_FILES.items():
        if relative_path not in cache:
            cache[relative_path] = read_jsonl(results_dir / relative_path)
        loaded[method] = cache[relative_path]

    identity_sets: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for method, rows in loaded.items():
        by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
        for index, row in enumerate(rows):
            identity = _identity(row, method, index)
            if identity in by_identity:
                raise FinalFigureError(
                    f"{method} contains a duplicate question identity {identity}"
                )
            _recorded_error(row, method, index)
            by_identity[identity] = row
        identity_sets[method] = by_identity

    reference_method = "predictive_entropy"
    reference_rows = identity_sets[reference_method]
    for method, by_identity in identity_sets.items():
        if set(by_identity) != set(reference_rows):
            raise FinalFigureError(f"{method} does not contain the exact entropy pilot sample")
        for index, (identity, row) in enumerate(by_identity.items()):
            ref = reference_rows[identity]
            if _recorded_error(row, method, index) != _recorded_error(ref, reference_method, index):
                raise FinalFigureError(f"{method} has a changed correctness label for {identity}")
            for field in (
                "question",
                "normalized_ground_truth_label",
                "normalized_prediction",
                "raw_prediction",
                "prediction_status",
                "normalized_entropy",
            ):
                if row.get(field) != ref.get(field):
                    raise FinalFigureError(f"{method} changed {field} for {identity}")
    if len(reference_rows) != 30:
        raise FinalFigureError(f"expected 30 pilot examples, found {len(reference_rows)}")
    if len({identity[2] for identity in reference_rows}) != 30:
        raise FinalFigureError("the pilot sample is expected to contain 30 distinct images")
    return loaded


def _make_pilot_comparison(
    method_rows: Mapping[str, Sequence[Mapping[str, Any]]], *, replicates: int, seed: int
) -> list[dict[str, Any]]:
    comparison: list[dict[str, Any]] = []
    for method, score_field in PILOT_SCORE_FIELDS.items():
        rows = method_rows[method]
        metrics = _metrics_for_score(rows, score_field, method)
        comparison.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "cohort": "30-question pilot",
                "cohort_label": "pilot (30 questions / 30 images)",
                **metrics,
                "confidence_intervals": image_grouped_metric_intervals(
                    rows, score_field, replicates=replicates, seed=seed
                ),
            }
        )
    return comparison


def _verify_metric_close(
    observed: float, recorded: Any, label: str, tolerance: float = 1e-9
) -> None:
    try:
        expected = float(recorded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalFigureError(f"saved report lacks numeric {label}") from exc
    if not isfinite(expected) or abs(observed - expected) > tolerance:
        raise FinalFigureError(f"row-level {label} does not match the saved aggregate report")


def _failure_example(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "category": row.get("category"),
        "question_id": row.get("question_id"),
        "image_id": row.get("image_id"),
        "question": row.get("question"),
        "ground_truth": row.get("normalized_ground_truth_label"),
        "raw_prediction": row.get("raw_prediction"),
        "normalized_prediction": row.get("normalized_prediction"),
        "prediction_status": row.get("prediction_status"),
        "normalized_entropy": row.get("normalized_entropy"),
    }


def _find_failure_examples(
    full_rows: Sequence[Mapping[str, Any]], disagreement_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    parsed_errors = [
        row
        for row in full_rows
        if row.get("is_error") is True and row.get("prediction_status") == "parsed"
    ]
    if not parsed_errors:
        raise FinalFigureError("no parsed errors were available for the failure analysis")
    low_entropy_error = min(parsed_errors, key=lambda row: float(row["normalized_entropy"]))

    ambiguous = [row for row in full_rows if row.get("prediction_status") == "ambiguous"]
    if not ambiguous:
        raise FinalFigureError("no ambiguous recorded response was available")
    low_entropy_ambiguous = min(ambiguous, key=lambda row: float(row["normalized_entropy"]))

    unanimous_wrong: list[dict[str, Any]] = []
    for row in disagreement_rows:
        samples = row.get("sampled_answers")
        if not isinstance(samples, list) or not row.get("is_error"):
            continue
        answers = [
            sample.get("normalized_answer") for sample in samples if isinstance(sample, dict)
        ]
        gold = row.get("normalized_ground_truth_label")
        if len(answers) == 5 and answers[0] in {"yes", "no"} and len(set(answers)) == 1:
            if answers[0] != gold:
                unanimous_wrong.append(
                    {
                        **_failure_example(row),
                        "answer_disagreement_score": row.get("answer_disagreement_score"),
                        "sampled_normalized_answers": answers,
                        "sampled_raw_answers": [sample.get("raw_answer") for sample in samples],
                    }
                )
    if not unanimous_wrong:
        raise FinalFigureError("no unanimously wrong five-generation example was available")
    unanimous_wrong.sort(key=lambda row: float(row["normalized_entropy"]))

    return {
        "selection_rule": (
            "Choose the lowest-entropy parsed error, the lowest-entropy ambiguous response, "
            "and the lowest-entropy baseline error with five unanimous wrong sampled answers."
        ),
        "low_entropy_parsed_error": _failure_example(low_entropy_error),
        "low_entropy_ambiguous_response": _failure_example(low_entropy_ambiguous),
        "unanimously_wrong_repeated_answer_count": len(unanimous_wrong),
        "unanimously_wrong_repeated_answer_examples": unanimous_wrong[:3],
    }


def load_final_analysis(
    results_dir: Path,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 2027,
) -> dict[str, Any]:
    """Load and cross-check recorded runs, metrics, splits, and failure cases."""

    method_rows = _load_pilot_score_rows(results_dir)
    pilot_metrics = _make_pilot_comparison(method_rows, replicates=bootstrap_replicates, seed=seed)
    saved_method_table = read_json(
        results_dir
        / "answer_disagreement_evaluation/pope_answer_disagreement_30_seed_2027_comparison.json"
    )["comparison_table"]
    saved_metrics_by_name = {row["score"]: row for row in saved_method_table}
    for result in pilot_metrics:
        saved = saved_metrics_by_name.get(result["method"])
        if saved is None:
            raise FinalFigureError(f"saved method report is missing {result['method']}")
        _verify_metric_close(result["auroc"], saved["auroc"], f"pilot {result['method']} AUROC")
        _verify_metric_close(
            result["auprc_average_precision"],
            saved["auprc_average_precision"],
            f"pilot {result['method']} AUPRC",
        )

    full_path = results_dir / "robustness/full_pope_predictions_seed_2027.jsonl"
    full_rows = read_jsonl(full_path)
    full_report = read_json(results_dir / "robustness/expanded_robustness_summary_seed_2027.json")
    full_recomputed = summarize_expanded_rows(full_rows)["overall"]
    full_recorded = full_report["full_clean_metrics"]["overall"]
    for metric in ("accuracy", "mean_normalized_entropy", "auroc", "auprc_average_precision"):
        _verify_metric_close(full_recomputed[metric], full_recorded[metric], f"expanded {metric}")
    full_image_count = len({str(row["image_id"]) for row in full_rows})
    if len(full_rows) != 1800 or full_image_count != 100:
        raise FinalFigureError(
            f"expanded clean result must be 1,800 questions on 100 images; "
            f"found {len(full_rows)} questions on {full_image_count} images"
        )

    expanded_entropy = {
        "method": "predictive_entropy",
        "label": METHOD_LABELS["predictive_entropy"],
        "cohort": "expanded clean evaluation",
        "cohort_label": "expanded (1,800 questions / 100 images)",
        "question_count": len(full_rows),
        "image_count": full_image_count,
        "error_count": int(full_recorded["question_count"] - full_recorded["correct_count"]),
        "error_prevalence": float(full_recorded["error_prevalence_among_scored"]),
        "auroc": float(full_recorded["auroc"]),
        "auprc_average_precision": float(full_recorded["auprc_average_precision"]),
        "confidence_intervals": {
            "method": "image_grouped_percentile_bootstrap",
            "confidence_level": 0.95,
            "seed": full_report["seed"],
            "unit_of_resampling": (
                "image_id; every question for each sampled image travels together"
            ),
            "image_count": full_image_count,
            "auroc": full_recorded["confidence_interval"]["auroc"],
            "auprc_average_precision": full_recorded["confidence_interval"][
                "auprc_average_precision"
            ],
        },
    }

    shift_report = read_json(results_dir / "robustness/expanded_shift_summary_seed_2027.json")
    shift_rows = shift_report.get("conditions")
    if not isinstance(shift_rows, list) or len(shift_rows) != 13:
        raise FinalFigureError(
            "expanded shift summary must contain one clean and 12 shifted conditions"
        )
    shift_images = len(shift_report.get("selected_evaluation_image_ids", []))
    if shift_images != 3 or shift_report.get("paired_image_question_count") != 54:
        raise FinalFigureError(
            "expected the recorded paired shift subset of 54 questions on 3 images"
        )
    shift_raw_rows = read_jsonl(
        results_dir / "robustness/expanded_shift_predictions_seed_2027.jsonl"
    )
    if len(shift_raw_rows) != 648:
        raise FinalFigureError(f"expected 648 recorded shifted rows, found {len(shift_raw_rows)}")
    shifted_image_ids = set(map(str, shift_report["selected_evaluation_image_ids"]))
    if {str(row.get("image_id")) for row in shift_raw_rows} != shifted_image_ids:
        raise FinalFigureError("shift rows do not match the saved three-image evaluation subset")
    shifted_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in shift_raw_rows:
        shifted_groups[str(row.get("condition_id"))].append(row)
    paired_clean_rows = [row for row in full_rows if str(row["image_id"]) in shifted_image_ids]
    if len(paired_clean_rows) != 54:
        raise FinalFigureError("clean shift controls do not contain the same 54 paired questions")
    clean_by_identity = {
        _identity(row, "paired clean", index): row for index, row in enumerate(paired_clean_rows)
    }
    for condition in shift_rows:
        condition_id = str(condition["condition_id"])
        selected_rows = (
            paired_clean_rows if condition_id == "clean" else shifted_groups.get(condition_id, [])
        )
        selected_by_identity = {
            _identity(row, condition_id, index): row for index, row in enumerate(selected_rows)
        }
        if set(selected_by_identity) != set(clean_by_identity):
            raise FinalFigureError(
                f"shift condition {condition_id} is not paired to the clean rows"
            )
        for identity, row in selected_by_identity.items():
            clean_row = clean_by_identity[identity]
            if row.get("question") != clean_row.get("question") or row.get(
                "normalized_ground_truth_label"
            ) != clean_row.get("normalized_ground_truth_label"):
                raise FinalFigureError(
                    f"shift condition {condition_id} changed question or label {identity}"
                )
        computed_summary = summarize_expanded_rows(selected_rows)
        computed = computed_summary["overall"]
        if len(selected_rows) != 54 or computed_summary["image_count"] != shift_images:
            raise FinalFigureError(
                f"shift condition {condition_id} does not retain the full paired sample"
            )
        for metric, recorded_metric in (
            ("accuracy", "accuracy"),
            ("mean_normalized_entropy", "mean_normalized_entropy"),
            ("auroc", "auroc"),
            ("auprc_average_precision", "auprc"),
        ):
            _verify_metric_close(
                computed[metric],
                condition[recorded_metric],
                f"shift {condition_id} {metric}",
            )

    split = read_json(results_dir / "robustness/expanded_image_split_seed_2027.json")
    abstention = full_report["abstention"]
    calibration_ids = set(map(str, abstention["calibration_image_ids"]))
    evaluation_ids = set(map(str, abstention["evaluation_image_ids"]))
    reference_ids = set(map(str, split["reference_image_ids"]))
    if (
        calibration_ids & evaluation_ids
        or calibration_ids & reference_ids
        or evaluation_ids & reference_ids
    ):
        raise FinalFigureError("calibration, reference, and held-out image partitions overlap")
    if abstention.get("image_overlap_count") != 0 or abstention.get("reference_overlap_count") != 0:
        raise FinalFigureError("saved abstention summary reports a nonzero image overlap")

    pilot_entropy_report = read_json(results_dir / "pope_entropy_30_seed_2027_evaluation.json")
    pilot_entropy_rows = method_rows["predictive_entropy"]
    pilot_curve = risk_coverage_curve(
        [
            int(_recorded_error(row, "pilot entropy", index))
            for index, row in enumerate(pilot_entropy_rows)
        ],
        [float(row["normalized_entropy"]) for row in pilot_entropy_rows],
    )
    recorded_pilot_curve = pilot_entropy_report["risk_coverage_curve"]
    if len(pilot_curve) != len(recorded_pilot_curve):
        raise FinalFigureError(
            "recomputed pilot risk-coverage curve differs in length from saved report"
        )
    for current, recorded in zip(pilot_curve, recorded_pilot_curve, strict=True):
        if current["coverage"] != recorded["coverage"] or current["risk"] != recorded["risk"]:
            raise FinalFigureError("recomputed pilot risk-coverage curve differs from saved report")
    pilot_risk_intervals = image_grouped_risk_coverage_intervals(
        pilot_entropy_rows,
        "normalized_entropy",
        replicates=bootstrap_replicates,
        seed=seed,
    )

    full_entropy_curve = risk_coverage_curve(
        [
            int(_recorded_error(row, "expanded entropy", index))
            for index, row in enumerate(full_rows)
        ],
        [float(row["normalized_entropy"]) for row in full_rows],
    )
    recorded_full_curve = full_report["full_clean_risk_coverage_curve"]
    if len(full_entropy_curve) != len(recorded_full_curve):
        raise FinalFigureError(
            "recomputed expanded risk-coverage curve differs in length from report"
        )
    for current, recorded in zip(full_entropy_curve, recorded_full_curve, strict=True):
        if current["coverage"] != recorded["coverage"] or current["risk"] != recorded["risk"]:
            raise FinalFigureError(
                "recomputed expanded risk-coverage curve differs from saved report"
            )
    expanded_risk_intervals = image_grouped_risk_coverage_intervals(
        full_rows,
        "normalized_entropy",
        replicates=1000,
        seed=seed,
    )

    heldout_ids = evaluation_ids
    heldout_rows = [row for row in full_rows if str(row["image_id"]) in heldout_ids]
    if len(heldout_rows) != int(abstention["evaluation_questions"]):
        raise FinalFigureError("held-out rows do not match the saved abstention evaluation count")
    heldout_curve = risk_coverage_curve(
        [
            int(_recorded_error(row, "held-out entropy", index))
            for index, row in enumerate(heldout_rows)
        ],
        [float(row["normalized_entropy"]) for row in heldout_rows],
    )
    abstention_thresholds = abstention["thresholds"]
    if len(abstention_thresholds) != 4:
        raise FinalFigureError("expected four saved calibrated abstention operating points")
    if any(
        item["calibration"].get("selection_uses_correctness_labels") is not False
        for item in abstention_thresholds
    ):
        raise FinalFigureError("abstention threshold selection is not documented as label-free")

    pilot_abstention_report = read_json(
        results_dir / "abstention/pope_abstention_20cal_30eval_seed_2027.json"
    )
    pilot_calibration_ids = set(map(str, pilot_abstention_report["calibration_image_ids"]))
    pilot_evaluation_ids = set(map(str, pilot_abstention_report["evaluation_image_ids"]))
    if pilot_calibration_ids & pilot_evaluation_ids:
        raise FinalFigureError("pilot abstention calibration and evaluation images overlap")
    if pilot_abstention_report.get("image_id_overlap_count") != 0:
        raise FinalFigureError("pilot abstention report records a nonzero image overlap")
    if len(pilot_calibration_ids) != 20 or len(pilot_evaluation_ids) != 30:
        raise FinalFigureError("pilot abstention must use 20 calibration and 30 evaluation images")
    pilot_abstention_thresholds: list[dict[str, Any]] = []
    for item in pilot_abstention_report["targets"]:
        calibration = item["calibration"]
        evaluation = item["evaluation"]
        if calibration.get("selection_uses_correctness_labels") is not False:
            raise FinalFigureError("pilot abstention threshold is not documented as label-free")
        outcomes = evaluation["outcomes"]
        if len(outcomes) != 30 or len({str(row["image_id"]) for row in outcomes}) != 30:
            raise FinalFigureError("pilot abstention must contain 30 held-out images")
        if not any(row["decision"] == "answer" for row in outcomes):
            raise FinalFigureError("pilot abstention has no answered outcomes")
        intervals = image_grouped_selective_intervals(
            outcomes, replicates=bootstrap_replicates, seed=seed
        )
        answered = [row for row in outcomes if row["decision"] == "answer"]
        observed_coverage = len(answered) / len(outcomes)
        observed_risk = sum(bool(row["is_error"]) for row in answered) / len(answered)
        _verify_metric_close(observed_coverage, evaluation["coverage"], "pilot selective coverage")
        _verify_metric_close(observed_risk, evaluation["selective_risk"], "pilot selective risk")
        pilot_abstention_thresholds.append(
            {
                "target_coverage": float(evaluation["target_coverage"]),
                "threshold": float(calibration["threshold"]),
                "evaluation": {
                    "coverage": float(evaluation["coverage"]),
                    "selective_risk": float(evaluation["selective_risk"]),
                    "target_coverage": float(evaluation["target_coverage"]),
                },
                "evaluation_confidence_intervals": {"intervals": intervals},
            }
        )

    disagreement_rows = method_rows["answer_disagreement"]
    return {
        "bootstrap": {
            "pilot_method": "image_grouped_percentile_bootstrap",
            "pilot_replicates": bootstrap_replicates,
            "seed": seed,
            "pilot_unit": "image_id; each pilot image contains one question",
            "expanded_clean_source": "saved 1,000-replicate image-group bootstrap",
            "shift_source": "saved 1,000-replicate image-group bootstrap; three image groups",
        },
        "method_comparison": {
            "pilot": pilot_metrics,
            "expanded_predictive_entropy": expanded_entropy,
            "interpretation": (
                "Only predictive entropy has an expanded 1,800-question result. "
                "The other four scores have results only on the 30-question pilot."
            ),
        },
        "expanded_clean": {
            "question_count": len(full_rows),
            "image_count": full_image_count,
            "error_count": int(full_recorded["question_count"] - full_recorded["correct_count"]),
            "accuracy": float(full_recorded["accuracy"]),
            "mean_normalized_entropy": float(full_recorded["mean_normalized_entropy"]),
            "error_prevalence": float(full_recorded["error_prevalence_among_scored"]),
            "confidence_intervals": full_recorded["confidence_interval"],
        },
        "distribution_shift": {
            "paired_questions": 54,
            "distinct_images": shift_images,
            "conditions": shift_rows,
            "full_clean_reference": {
                "question_count": len(full_rows),
                "image_count": full_image_count,
                "accuracy": float(full_recorded["accuracy"]),
                "mean_normalized_entropy": float(full_recorded["mean_normalized_entropy"]),
                "confidence_intervals": full_recorded["confidence_interval"],
            },
            "image_ids": shift_report["selected_evaluation_image_ids"],
            "bootstrap": shift_report["image_grouped_bootstrap_by_condition"],
        },
        "risk_coverage": {
            "pilot_entropy": pilot_curve,
            "expanded_entropy": full_report["full_clean_risk_coverage_curve"],
            "heldout_entropy": heldout_curve,
            "pilot_bootstrap": pilot_risk_intervals,
            "expanded_bootstrap": expanded_risk_intervals,
            "pilot_question_count": len(pilot_entropy_rows),
            "expanded_question_count": len(full_rows),
            "heldout_question_count": len(heldout_rows),
            "heldout_image_count": len(evaluation_ids),
            "pilot_calibration_question_count": int(
                pilot_abstention_report["calibration_question_count"]
            ),
            "pilot_calibration_image_count": len(pilot_calibration_ids),
            "pilot_evaluation_question_count": len(pilot_evaluation_ids),
            "pilot_evaluation_image_count": len(pilot_evaluation_ids),
            "pilot_abstention_thresholds": pilot_abstention_thresholds,
            "calibration_question_count": int(abstention["calibration_questions"]),
            "calibration_image_count": len(calibration_ids),
            "abstention_thresholds": abstention_thresholds,
            "reference_image_count": len(reference_ids),
            "image_overlaps": {
                "pilot_calibration_evaluation": len(pilot_calibration_ids & pilot_evaluation_ids),
                "calibration_evaluation": len(calibration_ids & evaluation_ids),
                "calibration_reference": len(calibration_ids & reference_ids),
                "evaluation_reference": len(evaluation_ids & reference_ids),
            },
        },
        "failure_analysis": _find_failure_examples(full_rows, disagreement_rows),
        "source_files": [
            "robustness/full_pope_predictions_seed_2027.jsonl",
            "robustness/expanded_robustness_summary_seed_2027.json",
            "robustness/expanded_shift_summary_seed_2027.json",
            "robustness/expanded_image_split_seed_2027.json",
            "pope_entropy_30_seed_2027_evaluated.jsonl",
            "embedding_distance/pope_embedding_entropy_30_seed_2027_scores.jsonl",
            "augmentation_evaluation/pope_augmentation_consistency_30_seed_2027_scores.jsonl",
            "answer_disagreement_evaluation/pope_answer_disagreement_30_seed_2027_scores.jsonl",
            "abstention/pope_abstention_20cal_30eval_seed_2027.json",
        ],
    }


def _text(
    x: float, y: float, value: str, *, size: int = 14, color: str = "#28323c", **attrs: str
) -> str:
    extra = " ".join(f'{key.replace("_", "-")}="{escape(item)}"' for key, item in attrs.items())
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,Helvetica,sans-serif" '
        f'font-size="{size}" fill="{color}" {extra}>{escape(value)}</text>'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = "#d9e0e7",
    width: float = 1,
    dash: str | None = None,
) -> str:
    dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attribute}/>'
    )


def _circle(x: float, y: float, *, radius: float, fill: str, stroke: str = "white") -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.6"/>'
    )


def _polyline(points: Sequence[tuple[float, float]], *, color: str, width: float = 2.5) -> str:
    if not points:
        return ""
    coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def _svg_header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-variant-numeric:tabular-nums} .muted{fill:#647282}</style>",
        _text(
            width / 2, 34, title, size=23, color="#172b4d", text_anchor="middle", font_weight="bold"
        ),
        _text(width / 2, 59, subtitle, size=13, color="#5e6c7b", text_anchor="middle"),
    ]


def _draw_metric_panel(
    elements: list[str],
    *,
    y: float,
    title: str,
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    pilot_prevalence: float,
    expanded_prevalence: float,
) -> None:
    left, right = 440.0, 1345.0
    top, bottom = y + 73, y + 318
    elements.append(_text(68, y + 24, title, size=18, color="#172b4d", font_weight="bold"))
    for tick in range(6):
        value = tick / 5
        x = left + value * (right - left)
        elements.append(_line(x, top, x, bottom, color="#e6ebf0" if tick else "#aab5c0"))
        elements.append(
            _text(x, bottom + 23, f"{value:.1f}", size=12, color="#607080", text_anchor="middle")
        )
    elements.append(_line(left, bottom, right, bottom, color="#8997a5", width=1.3))

    if metric == "auroc":
        baseline_items = [(0.5, "AUROC chance", "#747d87", "5 4")]
    else:
        baseline_items = [
            (pilot_prevalence, f"pilot prevalence {pilot_prevalence:.3f}", "#9a694f", "4 4"),
            (
                expanded_prevalence,
                f"expanded prevalence {expanded_prevalence:.3f}",
                "#527a82",
                "2 4",
            ),
        ]
    for baseline_index, (value, label, color, dash) in enumerate(baseline_items):
        x = left + value * (right - left)
        elements.append(_line(x, top - 8, x, bottom, color=color, width=1.4, dash=dash))
        if metric == "auprc":
            elements.append(
                _text(x + 5, top - 11 + baseline_index * 14, label, size=10, color=color)
            )
        else:
            elements.append(_text(x + 7, top - 12, label, size=10, color=color))

    row_gap = 33.0
    first_center = top + 24
    for index, row in enumerate(rows):
        center = first_center + index * row_gap
        method = str(row["method"])
        color = METHOD_COLORS[method]
        value = float(row[metric])
        interval = row["confidence_intervals"][metric]
        lower = interval.get("lower")
        upper = interval.get("upper")
        label = (
            str(row["cohort_label"])
            if row["cohort"] != "30-question pilot"
            else "pilot · 30 questions / 30 images"
        )
        elements.append(
            _text(68, center + 4, f"{row['label']}  |  {label}", size=12, color="#28323c")
        )
        if lower is not None and upper is not None:
            x_low = left + float(lower) * (right - left)
            x_high = left + float(upper) * (right - left)
            elements.append(_line(x_low, center, x_high, center, color=color, width=2.4))
            elements.append(_line(x_low, center - 6, x_low, center + 6, color=color, width=1.8))
            elements.append(_line(x_high, center - 6, x_high, center + 6, color=color, width=1.8))
        x_value = left + value * (right - left)
        if row["cohort"] == "expanded clean evaluation":
            elements.append(
                f'<rect x="{x_value - 5:.1f}" y="{center - 5:.1f}" width="10" height="10" '
                f'fill="{color}" stroke="white" stroke-width="1.5"/>'
            )
        else:
            elements.append(_circle(x_value, center, radius=5.4, fill=color))
        text_x = min(right - 36, x_value + 11)
        elements.append(
            _text(text_x, center + 4, f"{value:.3f}", size=11, color=color, font_weight="bold")
        )
    elements.append(
        _text(
            (left + right) / 2,
            bottom + 46,
            "Error-ranking score (higher is better)",
            size=12,
            color="#526170",
            text_anchor="middle",
        )
    )


def render_method_comparison_svg(analysis: Mapping[str, Any]) -> str:
    """Render pilot methods and the expanded entropy result with image CIs."""

    pilot = list(analysis["method_comparison"]["pilot"])
    expanded = analysis["method_comparison"]["expanded_predictive_entropy"]
    ordered = [pilot[0], expanded, *pilot[1:]]
    pilot_error_prevalence = float(pilot[0]["error_prevalence"])
    expanded_error_prevalence = float(expanded["error_prevalence"])
    elements = _svg_header(
        1440,
        880,
        "Uncertainty methods: error-ranking performance",
        "POPE answer-error proxy; whiskers show 95% image-group bootstrap intervals.",
    )
    _draw_metric_panel(
        elements,
        y=75,
        title="A   AUROC",
        rows=ordered,
        metric="auroc",
        pilot_prevalence=pilot_error_prevalence,
        expanded_prevalence=expanded_error_prevalence,
    )
    _draw_metric_panel(
        elements,
        y=455,
        title="B   AUPRC (non-interpolated average precision)",
        rows=ordered,
        metric="auprc_average_precision",
        pilot_prevalence=pilot_error_prevalence,
        expanded_prevalence=expanded_error_prevalence,
    )
    elements.append(
        _text(
            68,
            866,
            "Non-entropy scores are pilot-only; expanded results are entropy-only.",
            size=11,
            color="#5e6c7b",
        )
    )
    elements.append("</svg>")
    return "\n".join(elements)


def _metric_interval(
    condition: Mapping[str, Any], metric: str
) -> tuple[float | None, float | None]:
    interval = condition.get("confidence_interval", {}).get(metric, {})
    lower = interval.get("lower")
    upper = interval.get("upper")
    return (None if lower is None else float(lower), None if upper is None else float(upper))


def _draw_errorbar(
    elements: list[str], x: float, center_y: float, lower_y: float, upper_y: float, *, color: str
) -> None:
    elements.append(_line(x, lower_y, x, upper_y, color=color, width=1.5))
    elements.append(_line(x - 5, lower_y, x + 5, lower_y, color=color, width=1.5))
    elements.append(_line(x - 5, upper_y, x + 5, upper_y, color=color, width=1.5))


def render_shift_severity_svg(analysis: Mapping[str, Any], metric: str) -> str:
    """Render four paired clean-to-shift panels with saved image-cluster CIs."""

    if metric not in {"accuracy", "mean_normalized_entropy"}:
        raise FinalFigureError("shift figure metric must be accuracy or mean_normalized_entropy")
    shift = analysis["distribution_shift"]
    conditions = shift["conditions"]
    condition_by_id = {str(row["condition_id"]): row for row in conditions}
    clean = condition_by_id.get("clean")
    if clean is None:
        raise FinalFigureError("shift summary does not contain its paired clean condition")
    full = shift["full_clean_reference"]
    metric_label = "Accuracy" if metric == "accuracy" else "Mean normalized binary entropy"
    title = f"{metric_label}: clean versus controlled image shifts"
    subtitle = (
        "Paired shift subset: 54 question rows from 3 images; severity 1–3. "
        "Dashed reference uses the expanded clean set: 1,800 questions from 100 images."
    )
    elements = _svg_header(1440, 840, title, subtitle)
    panels = [
        ("gaussian_blur", 72.0, 105.0),
        ("brightness_reduction", 735.0, 105.0),
        ("gaussian_noise", 72.0, 465.0),
        ("contrast_reduction", 735.0, 465.0),
    ]
    value_max = 1.0 if metric == "accuracy" else 0.5
    clean_value = float(clean[metric])
    clean_low, clean_high = _metric_interval(clean, metric)
    full_value = float(full[metric])
    full_ci = full["confidence_intervals"][metric]
    full_low = full_ci.get("lower")
    full_high = full_ci.get("upper")

    for perturbation, panel_x, panel_y in panels:
        panel_width, panel_height = 625.0, 315.0
        plot_left, plot_right = panel_x + 55, panel_x + panel_width - 20
        plot_top, plot_bottom = panel_y + 53, panel_y + 250
        elements.append(
            _text(
                panel_x,
                panel_y + 20,
                SHIFT_LABELS[perturbation],
                size=16,
                color="#172b4d",
                font_weight="bold",
            )
        )
        elements.append(
            _text(
                panel_x,
                panel_y + 40,
                "n = 3 images / 54 paired questions",
                size=10,
                color="#687786",
            )
        )
        for tick in range(5):
            value = value_max * tick / 4
            y = plot_bottom - (value / value_max) * (plot_bottom - plot_top)
            elements.append(_line(plot_left, y, plot_right, y, color="#e6ebf0"))
            elements.append(
                _text(
                    plot_left - 9,
                    y + 4,
                    f"{value:.2f}",
                    size=10,
                    color="#607080",
                    text_anchor="end",
                )
            )
        x_positions = [plot_left + (plot_right - plot_left) * step / 3 for step in range(4)]
        labels = ["Clean", "Level 1", "Level 2", "Level 3"]
        for x, label in zip(x_positions, labels, strict=True):
            elements.append(_line(x, plot_top, x, plot_bottom, color="#f0f3f6"))
            elements.append(
                _text(x, plot_bottom + 20, label, size=10, color="#526170", text_anchor="middle")
            )
        elements.append(
            _line(plot_left, plot_bottom, plot_right, plot_bottom, color="#8997a5", width=1.1)
        )

        if full_low is not None and full_high is not None:
            y_low = plot_bottom - (float(full_low) / value_max) * (plot_bottom - plot_top)
            y_high = plot_bottom - (float(full_high) / value_max) * (plot_bottom - plot_top)
            elements.append(
                f'<rect x="{plot_left:.1f}" y="{y_high:.1f}" width="{plot_right - plot_left:.1f}" '
                f'height="{max(0.0, y_low - y_high):.1f}" fill="#61758a" fill-opacity="0.08"/>'
            )
        full_y = plot_bottom - (full_value / value_max) * (plot_bottom - plot_top)
        elements.append(
            _line(plot_left, full_y, plot_right, full_y, color="#667788", width=1.7, dash="7 5")
        )
        elements.append(
            _text(
                plot_right - 2,
                full_y - 6,
                "full clean · 100 images",
                size=9,
                color="#566778",
                text_anchor="end",
            )
        )

        clean_y = plot_bottom - (clean_value / value_max) * (plot_bottom - plot_top)
        if clean_low is not None and clean_high is not None:
            lower_y = plot_bottom - (clean_low / value_max) * (plot_bottom - plot_top)
            upper_y = plot_bottom - (clean_high / value_max) * (plot_bottom - plot_top)
            _draw_errorbar(elements, x_positions[0], clean_y, upper_y, lower_y, color="#45515d")

        shifted: list[tuple[float, float, float | None, float | None]] = []
        for severity in (1, 2, 3):
            condition_id = f"{perturbation}_severity_{severity}"
            condition = condition_by_id.get(condition_id)
            if condition is None:
                raise FinalFigureError(f"shift condition is missing: {condition_id}")
            value = float(condition[metric])
            lower, upper = _metric_interval(condition, metric)
            shifted.append((value, x_positions[severity], lower, upper))
        color = SHIFT_COLORS[perturbation]
        points = [(x_positions[0], clean_y)] + [
            (x, plot_bottom - (value / value_max) * (plot_bottom - plot_top))
            for value, x, _lower, _upper in shifted
        ]
        elements.append(_polyline(points, color=color, width=2.4))
        elements.append(_circle(x_positions[0], clean_y, radius=5, fill="#45515d"))
        for value, x, lower, upper in shifted:
            y = plot_bottom - (value / value_max) * (plot_bottom - plot_top)
            if lower is not None and upper is not None:
                lower_y = plot_bottom - (lower / value_max) * (plot_bottom - plot_top)
                upper_y = plot_bottom - (upper / value_max) * (plot_bottom - plot_top)
                _draw_errorbar(elements, x, y, upper_y, lower_y, color=color)
            elements.append(_circle(x, y, radius=4.5, fill=color))

        elements.append(
            _text(
                panel_x,
                panel_y + panel_height,
                "Image corruption severity",
                size=10,
                color="#526170",
            )
        )

    elements.append(
        _text(
            72,
            824,
            "Whiskers: 95% bootstrap intervals; each shifted condition resamples 3 images.",
            size=11,
            color="#5e6c7b",
        )
    )
    elements.append("</svg>")
    return "\n".join(elements)


def _curve_points(curve: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in curve:
        coverage, risk = row.get("coverage"), row.get("risk")
        if coverage is None or risk is None:
            continue
        points.append((float(coverage), float(risk)))
    return points


def _chart_axes(elements: list[str], left: float, top: float, right: float, bottom: float) -> None:
    for tick in range(6):
        value = tick / 5
        x = left + value * (right - left)
        y = bottom - value * (bottom - top)
        elements.append(_line(x, top, x, bottom, color="#e5eaf0"))
        elements.append(_line(left, y, right, y, color="#e5eaf0"))
        elements.append(
            _text(x, bottom + 22, f"{value:.1f}", size=10, color="#607080", text_anchor="middle")
        )
        elements.append(
            _text(left - 9, y + 4, f"{value:.1f}", size=10, color="#607080", text_anchor="end")
        )
    elements.append(_line(left, bottom, right, bottom, color="#8997a5", width=1.2))
    elements.append(_line(left, top, left, bottom, color="#8997a5", width=1.2))


def _plot_curve(
    elements: list[str],
    curve: Sequence[Mapping[str, Any]],
    *,
    left: float,
    top: float,
    right: float,
    bottom: float,
    color: str,
    width: float = 2.5,
) -> None:
    normalized = _curve_points(curve)
    points = [
        (left + coverage * (right - left), bottom - risk * (bottom - top))
        for coverage, risk in normalized
    ]
    elements.append(_polyline(points, color=color, width=width))


def _plot_curve_intervals(
    elements: list[str],
    bootstrap: Mapping[str, Any],
    *,
    left: float,
    top: float,
    right: float,
    bottom: float,
    color: str,
) -> None:
    """Add sparse pointwise whiskers at frozen empirical score cutoffs."""

    for point in bootstrap["points"]:
        observed = point["observed"]
        coverage = float(observed["coverage"])
        risk_value = float(observed["risk"])
        x = left + coverage * (right - left)
        y = bottom - risk_value * (bottom - top)
        coverage_ci = point["confidence_intervals"]["coverage"]
        risk_ci = point["confidence_intervals"]["risk"]
        if coverage_ci.get("lower") is not None and coverage_ci.get("upper") is not None:
            x_low = left + float(coverage_ci["lower"]) * (right - left)
            x_high = left + float(coverage_ci["upper"]) * (right - left)
            elements.append(_line(x_low, y, x_high, y, color=color, width=1.0))
            elements.append(_line(x_low, y - 3, x_low, y + 3, color=color, width=1.0))
            elements.append(_line(x_high, y - 3, x_high, y + 3, color=color, width=1.0))
        if risk_ci.get("lower") is not None and risk_ci.get("upper") is not None:
            y_low = bottom - float(risk_ci["upper"]) * (bottom - top)
            y_high = bottom - float(risk_ci["lower"]) * (bottom - top)
            elements.append(_line(x, y_low, x, y_high, color=color, width=1.0))
            elements.append(_line(x - 3, y_low, x + 3, y_low, color=color, width=1.0))
            elements.append(_line(x - 3, y_high, x + 3, y_high, color=color, width=1.0))
        elements.append(_circle(x, y, radius=3.1, fill=color))


def _draw_abstention_marker(
    elements: list[str],
    evaluation: Mapping[str, Any],
    intervals: Mapping[str, Any],
    *,
    left: float,
    top: float,
    right: float,
    bottom: float,
    color: str,
    label_prefix: str,
    label_index: int,
) -> None:
    coverage = float(evaluation["coverage"])
    selective_risk = float(evaluation["selective_risk"])
    x = left + coverage * (right - left)
    y = bottom - selective_risk * (bottom - top)
    coverage_ci, risk_ci = intervals["coverage"], intervals["selective_risk"]
    x_low = left + float(coverage_ci["lower"]) * (right - left)
    x_high = left + float(coverage_ci["upper"]) * (right - left)
    y_low = bottom - float(risk_ci["upper"]) * (bottom - top)
    y_high = bottom - float(risk_ci["lower"]) * (bottom - top)
    elements.append(_line(x_low, y, x_high, y, color=color, width=1.3))
    elements.append(_line(x_low, y - 4, x_low, y + 4, color=color, width=1.3))
    elements.append(_line(x_high, y - 4, x_high, y + 4, color=color, width=1.3))
    elements.append(_line(x, y_low, x, y_high, color=color, width=1.3))
    elements.append(_line(x - 4, y_low, x + 4, y_low, color=color, width=1.3))
    elements.append(_line(x - 4, y_high, x + 4, y_high, color=color, width=1.3))
    elements.append(
        f'<polygon points="{x:.1f},{y - 6:.1f} {x + 6:.1f},{y:.1f} {x:.1f},{y + 6:.1f} '
        f'{x - 6:.1f},{y:.1f}" fill="{color}" stroke="white" stroke-width="1.2"/>'
    )
    label_x = min(right - 38, max(left + 2, x + 8))
    label_y = y - 8 if label_index % 2 == 0 else y + 17
    target = int(round(float(evaluation["target_coverage"]) * 100))
    elements.append(
        _text(
            label_x,
            label_y,
            f"{label_prefix}{target}%",
            size=9,
            color=color,
            font_weight="bold",
        )
    )


def render_risk_coverage_svg(analysis: Mapping[str, Any]) -> str:
    """Render entropy risk curves and held-out calibrated-abstention points."""

    risk = analysis["risk_coverage"]
    elements = _svg_header(
        1440,
        700,
        "Selective answering: risk versus coverage",
        "Risk is error fraction among answered questions; highest entropy is excluded first.",
    )
    panels = [
        (75.0, "A   Predictive entropy ranking", 655.0),
        (750.0, "B   Calibrated abstention on held-out images", 1365.0),
    ]
    top, bottom = 130.0, 515.0
    for left_edge, title, right_edge in panels:
        elements.append(_text(left_edge, 103, title, size=17, color="#172b4d", font_weight="bold"))
        plot_left, plot_right = left_edge + 58, right_edge - 18
        _chart_axes(elements, plot_left, top, plot_right, bottom)
        elements.append(
            _text(
                (plot_left + plot_right) / 2,
                bottom + 47,
                "Coverage retained",
                size=11,
                color="#526170",
                text_anchor="middle",
            )
        )
        elements.append(
            _text(
                left_edge + 13,
                (top + bottom) / 2,
                "Selective risk",
                size=11,
                color="#526170",
                text_anchor="middle",
                transform=f"rotate(-90 {left_edge + 13:.1f} {(top + bottom) / 2:.1f})",
            )
        )

    left_l, left_r = panels[0][0] + 58, panels[0][2] - 18
    _plot_curve(
        elements,
        risk["pilot_entropy"],
        left=left_l,
        top=top,
        right=left_r,
        bottom=bottom,
        color="#7C5AA6",
        width=2.2,
    )
    _plot_curve(
        elements,
        risk["expanded_entropy"],
        left=left_l,
        top=top,
        right=left_r,
        bottom=bottom,
        color="#2878B5",
        width=2.6,
    )
    _plot_curve_intervals(
        elements,
        risk["pilot_bootstrap"],
        left=left_l,
        top=top,
        right=left_r,
        bottom=bottom,
        color="#7C5AA6",
    )
    _plot_curve_intervals(
        elements,
        risk["expanded_bootstrap"],
        left=left_l,
        top=top,
        right=left_r,
        bottom=bottom,
        color="#2878B5",
    )
    point_colors = ["#2A9D72", "#2878B5", "#D17A22", "#7B5BA7"]
    for index, item in enumerate(risk["pilot_abstention_thresholds"]):
        _draw_abstention_marker(
            elements,
            item["evaluation"],
            item["evaluation_confidence_intervals"]["intervals"],
            left=left_l,
            top=top,
            right=left_r,
            bottom=bottom,
            color=point_colors[index],
            label_prefix="P",
            label_index=index,
        )
    elements.append(_line(left_l + 10, 560, left_l + 38, 560, color="#7C5AA6", width=2.5))
    elements.append(
        _text(left_l + 46, 564, "Pilot · 30 questions / 30 images", size=11, color="#394653")
    )
    elements.append(_line(left_l + 290, 560, left_l + 318, 560, color="#2878B5", width=2.5))
    elements.append(
        _text(
            left_l + 326, 564, "Expanded · 1,800 questions / 100 images", size=11, color="#394653"
        )
    )
    elements.append(
        _text(
            left_l + 10,
            584,
            "P/E markers: pilot/expanded abstention; curve whiskers use 20–100% coverage cutoffs.",
            size=10,
            color="#5e6c7b",
        )
    )

    right_l, right_r = panels[1][0] + 58, panels[1][2] - 18
    _plot_curve(
        elements,
        risk["heldout_entropy"],
        left=right_l,
        top=top,
        right=right_r,
        bottom=bottom,
        color="#526778",
        width=2.2,
    )
    for index, item in enumerate(risk["abstention_thresholds"]):
        evaluation = item["evaluation"]
        intervals = item["evaluation_confidence_intervals"]["intervals"]
        _draw_abstention_marker(
            elements,
            evaluation,
            intervals,
            left=right_l,
            top=top,
            right=right_r,
            bottom=bottom,
            color=point_colors[index],
            label_prefix="E",
            label_index=index,
        )
    elements.append(_line(right_l + 10, 560, right_l + 38, 560, color="#526778", width=2.5))
    elements.append(
        _text(
            right_l + 46,
            564,
            "Held-out entropy curve · 720 questions / 40 images",
            size=10,
            color="#394653",
        )
    )
    elements.append(
        _text(
            right_l + 46,
            584,
            "Whiskers: 95% image-group bootstrap intervals.",
            size=10,
            color="#5e6c7b",
        )
    )
    elements.append(
        _text(
            75,
            650,
            "Calibration: 180 / 10 images; held-out: 720 / 40 images; reference: 50 images.",
            size=11,
            color="#5e6c7b",
        )
    )
    elements.append("</svg>")
    return "\n".join(elements)


def generate_final_figures(
    results_dir: Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 2027,
) -> dict[str, Any]:
    """Generate the final SVG figure set and a JSON source-data audit."""

    analysis = load_final_analysis(
        results_dir, bootstrap_replicates=bootstrap_replicates, seed=seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "method_comparison": output_dir / "uncertainty_method_comparison.svg",
        "shift_accuracy": output_dir / "clean_shift_accuracy.svg",
        "shift_entropy": output_dir / "clean_shift_entropy.svg",
        "risk_coverage": output_dir / "risk_coverage_entropy_abstention.svg",
        "summary": output_dir / f"final_figure_data_seed_{seed}.json",
    }
    artifacts["method_comparison"].write_text(
        render_method_comparison_svg(analysis), encoding="utf-8"
    )
    artifacts["shift_accuracy"].write_text(
        render_shift_severity_svg(analysis, "accuracy"), encoding="utf-8"
    )
    artifacts["shift_entropy"].write_text(
        render_shift_severity_svg(analysis, "mean_normalized_entropy"), encoding="utf-8"
    )
    artifacts["risk_coverage"].write_text(render_risk_coverage_svg(analysis), encoding="utf-8")
    analysis["generated_artifacts"] = {
        key: str(path) for key, path in artifacts.items() if key != "summary"
    }
    artifacts["summary"].write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"analysis": analysis, "artifacts": {key: str(path) for key, path in artifacts.items()}}
