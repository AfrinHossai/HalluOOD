"""Image-disjoint threshold selection and predictive-entropy abstention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

from halluood.embeddings import EmbeddingError, normalize_coco_image_id


class SelectivePredictionError(ValueError):
    """Raised when calibration or selective prediction inputs are invalid."""


@dataclass(frozen=True)
class AbstentionThreshold:
    """A score-only threshold plus the split provenance used to choose it."""

    target_coverage: float
    threshold: float
    calibration_coverage: float
    calibration_dataset_id: str
    evaluation_dataset_id: str
    calibration_seed: int
    calibration_question_count: int
    calibration_image_ids: tuple[str, ...]
    evaluation_image_ids: tuple[str, ...]
    score_name: str = "normalized_binary_predictive_entropy"
    selection_procedure: str = (
        "Choose the observed calibration-entropy cutoff whose empirical coverage is "
        "closest to the target; ties at a score cutoff are kept together and an equally "
        "close under-target coverage is preferred. No correctness labels are used."
    )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable threshold and split provenance."""

        return {
            "target_coverage": self.target_coverage,
            "threshold": self.threshold,
            "calibration_coverage": self.calibration_coverage,
            "calibration_dataset_id": self.calibration_dataset_id,
            "evaluation_dataset_id": self.evaluation_dataset_id,
            "calibration_seed": self.calibration_seed,
            "calibration_question_count": self.calibration_question_count,
            "calibration_image_count": len(self.calibration_image_ids),
            "calibration_image_ids": list(self.calibration_image_ids),
            "evaluation_image_ids": list(self.evaluation_image_ids),
            "image_id_overlap_count": 0,
            "score_name": self.score_name,
            "decision_rule": "answer if normalized_entropy <= threshold; otherwise abstain",
            "tie_policy": "all equal uncertainty scores receive the same decision",
            "selection_procedure": self.selection_procedure,
            "selection_uses_correctness_labels": False,
        }


def _normalized_image_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise SelectivePredictionError("image_id must be a non-empty string or integer")
    text = str(value).strip()
    if not text:
        raise SelectivePredictionError("image_id must be a non-empty string or integer")
    try:
        return normalize_coco_image_id(text)
    except EmbeddingError as exc:
        if Path(text).name.casefold().startswith("coco_"):
            raise SelectivePredictionError(f"invalid COCO image_id {text!r}") from exc
        return text.casefold()


def _validated_entropy(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SelectivePredictionError(f"{name} must be a finite number in [0, 1]")
    score = float(value)
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise SelectivePredictionError(f"{name} must be a finite number in [0, 1]")
    return score


def _read_entropy(row: Mapping[str, Any], index: int) -> float:
    try:
        value = row["normalized_entropy"]
    except KeyError as exc:
        raise SelectivePredictionError(
            f"row {index} is missing normalized_entropy"
        ) from exc
    return _validated_entropy(value, f"row {index} normalized_entropy")


def _collect_image_ids(rows: Sequence[Mapping[str, Any]], name: str) -> tuple[str, ...]:
    if not rows:
        raise SelectivePredictionError(f"{name} must contain at least one image ID")
    image_ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SelectivePredictionError(f"{name} row {index} must be a mapping")
        if "image_id" not in row:
            raise SelectivePredictionError(f"{name} row {index} is missing image_id")
        image_ids.add(_normalized_image_id(row["image_id"]))
    return tuple(sorted(image_ids))


def _validate_dataset_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SelectivePredictionError(f"{name} must be a non-empty string")
    return value.strip()


def calibrate_abstention_threshold(
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    evaluation_image_ids: Sequence[str | int],
    target_coverage: float,
    calibration_dataset_id: str,
    evaluation_dataset_id: str,
    calibration_seed: int,
) -> AbstentionThreshold:
    """Choose an entropy threshold from calibration scores only.

    The evaluation input is intentionally limited to image IDs, so its labels
    and scores cannot affect threshold selection. An observed score cutoff is
    selected by nearest attainable empirical coverage. All tied scores at a
    cutoff are handled together; when two coverages are equally close, the
    lower coverage is selected.
    """

    if (
        isinstance(calibration_rows, (str, bytes))
        or not isinstance(calibration_rows, Sequence)
        or not calibration_rows
    ):
        raise SelectivePredictionError("calibration_rows must not be empty")
    if (
        isinstance(target_coverage, bool)
        or not isinstance(target_coverage, Real)
        or not isfinite(target_coverage)
        or not 0.0 < target_coverage <= 1.0
    ):
        raise SelectivePredictionError("target_coverage must be finite and in (0, 1]")
    if (
        isinstance(calibration_seed, bool)
        or not isinstance(calibration_seed, int)
        or calibration_seed < 0
    ):
        raise SelectivePredictionError("calibration_seed must be a non-negative integer")
    calibration_dataset = _validate_dataset_id(
        calibration_dataset_id, "calibration_dataset_id"
    )
    evaluation_dataset = _validate_dataset_id(
        evaluation_dataset_id, "evaluation_dataset_id"
    )
    if calibration_dataset == evaluation_dataset:
        raise SelectivePredictionError(
            "calibration_dataset_id and evaluation_dataset_id must be different"
        )

    if isinstance(evaluation_image_ids, (str, bytes)) or not isinstance(
        evaluation_image_ids, Sequence
    ):
        raise SelectivePredictionError("evaluation_image_ids must be a non-empty sequence")
    calibration_ids = _collect_image_ids(calibration_rows, "calibration_rows")
    evaluation_ids = tuple(
        sorted({_normalized_image_id(image_id) for image_id in evaluation_image_ids})
    )
    if not evaluation_ids:
        raise SelectivePredictionError("evaluation_image_ids must not be empty")
    overlap = set(calibration_ids) & set(evaluation_ids)
    if overlap:
        preview = sorted(overlap)[:5]
        raise SelectivePredictionError(
            f"calibration and evaluation image IDs overlap: {preview}"
        )

    scores = [_read_entropy(row, index) for index, row in enumerate(calibration_rows)]
    candidates = []
    for score in sorted(set(scores)):
        coverage = sum(value <= score for value in scores) / len(scores)
        candidates.append((score, coverage))
    threshold, calibration_coverage = min(
        candidates,
        key=lambda item: (
            abs(item[1] - float(target_coverage)),
            item[1] > float(target_coverage),
            item[1],
        ),
    )
    return AbstentionThreshold(
        target_coverage=float(target_coverage),
        threshold=float(threshold),
        calibration_coverage=float(calibration_coverage),
        calibration_dataset_id=calibration_dataset,
        evaluation_dataset_id=evaluation_dataset,
        calibration_seed=calibration_seed,
        calibration_question_count=len(calibration_rows),
        calibration_image_ids=calibration_ids,
        evaluation_image_ids=evaluation_ids,
    )


def should_abstain(normalized_entropy: float, threshold: float) -> bool:
    """Abstain above the entropy threshold; ties are answered consistently."""

    score = _validated_entropy(normalized_entropy, "normalized_entropy")
    cutoff = _validated_entropy(threshold, "threshold")
    return score > cutoff


def _row_is_error(row: Mapping[str, Any], index: int) -> bool:
    is_error = row.get("is_error")
    is_correct = row.get("is_correct")
    if is_error is not None and not isinstance(is_error, bool):
        raise SelectivePredictionError(f"row {index} is_error must be boolean")
    if is_correct is not None and not isinstance(is_correct, bool):
        raise SelectivePredictionError(f"row {index} is_correct must be boolean")
    if is_error is None and is_correct is None:
        raise SelectivePredictionError(
            f"row {index} requires boolean is_error or is_correct label"
        )
    if is_error is not None and is_correct is not None and is_error == is_correct:
        raise SelectivePredictionError(
            f"row {index} has inconsistent is_error and is_correct labels"
        )
    return is_error if is_error is not None else not is_correct


def evaluate_selective_predictions(
    evaluation_rows: Sequence[Mapping[str, Any]],
    calibration: AbstentionThreshold,
    *,
    evaluation_dataset_id: str,
) -> dict[str, Any]:
    """Measure coverage, error risk among answers, and abstention on held-out rows."""

    if not isinstance(calibration, AbstentionThreshold):
        raise SelectivePredictionError("calibration must be an AbstentionThreshold")
    dataset_id = _validate_dataset_id(evaluation_dataset_id, "evaluation_dataset_id")
    if dataset_id != calibration.evaluation_dataset_id:
        raise SelectivePredictionError("evaluation_dataset_id differs from calibrated metadata")
    evaluation_ids = _collect_image_ids(evaluation_rows, "evaluation_rows")
    if evaluation_ids != calibration.evaluation_image_ids:
        raise SelectivePredictionError(
            "evaluation rows do not match the image IDs recorded during calibration"
        )
    overlap = set(calibration.calibration_image_ids) & set(evaluation_ids)
    if overlap:
        raise SelectivePredictionError("calibration and evaluation image IDs overlap")

    outcomes = []
    answered_errors = 0
    answered_count = 0
    abstained_count = 0
    for index, row in enumerate(evaluation_rows):
        image_id = _normalized_image_id(row["image_id"])
        entropy = _read_entropy(row, index)
        is_error = _row_is_error(row, index)
        abstain = should_abstain(entropy, calibration.threshold)
        if abstain:
            abstained_count += 1
        else:
            answered_count += 1
            answered_errors += int(is_error)
        outcomes.append(
            {
                "question_id": row.get("question_id"),
                "image_id": image_id,
                "category": row.get("category"),
                "normalized_entropy": entropy,
                "is_error": is_error,
                "decision": "abstain" if abstain else "answer",
            }
        )

    total = len(evaluation_rows)
    coverage = answered_count / total
    return {
        "evaluation_dataset_id": dataset_id,
        "evaluation_question_count": total,
        "evaluation_image_count": len(evaluation_ids),
        "threshold": calibration.threshold,
        "target_coverage": calibration.target_coverage,
        "calibration_coverage": calibration.calibration_coverage,
        "coverage": coverage,
        "selective_risk": answered_errors / answered_count if answered_count else None,
        "abstention_rate": abstained_count / total,
        "answered_count": answered_count,
        "abstained_count": abstained_count,
        "answered_error_count": answered_errors,
        "outcomes": outcomes,
    }
