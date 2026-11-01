"""Uncertainty scores for model probabilities and answer stability.

The binary POPE score below compares model likelihoods for two fixed answer
strings (``Yes`` and ``No``). It is distinct from full-vocabulary next-token
entropy, which is also available through :func:`entropy_from_logits`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from numbers import Real
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


class UncertaintyScoreError(ValueError):
    """Raised when an uncertainty score cannot be computed safely."""


@dataclass(frozen=True)
class CandidateSequenceLogProbability:
    """Model log-likelihood for one answer string and its actual tokenization."""

    answer_text: str
    token_ids: tuple[int, ...]
    log_probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.answer_text, str) or not self.answer_text.strip():
            raise UncertaintyScoreError("candidate answer text must be non-empty")
        if not self.token_ids or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in self.token_ids
        ):
            raise UncertaintyScoreError("candidate token_ids must contain integer token IDs")
        if isinstance(self.log_probability, bool) or not isinstance(self.log_probability, Real):
            raise UncertaintyScoreError("candidate log probability must be numeric")
        if np.isnan(self.log_probability) or np.isposinf(self.log_probability):
            raise UncertaintyScoreError("candidate log probability must not be NaN or +infinity")


@dataclass(frozen=True)
class BinaryAnswerEntropyScore:
    """Entropy of the normalized Yes/No candidate-sequence distribution."""

    yes_candidate: CandidateSequenceLogProbability
    no_candidate: CandidateSequenceLogProbability
    yes_probability: float
    no_probability: float
    entropy_nats: float
    normalized_entropy: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable score with tokenization provenance."""

        return {
            "method": "conditional_binary_answer_sequence_entropy",
            "entropy_type": "binary_yes_no_answer_entropy",
            "probabilities": {
                "yes": self.yes_probability,
                "no": self.no_probability,
            },
            "entropy_nats": self.entropy_nats,
            "normalized_entropy": self.normalized_entropy,
            "candidate_log_probabilities": {
                "yes": self.yes_candidate.log_probability,
                "no": self.no_candidate.log_probability,
            },
            "candidate_token_ids": {
                "yes": list(self.yes_candidate.token_ids),
                "no": list(self.no_candidate.token_ids),
            },
            "candidate_token_counts": {
                "yes": len(self.yes_candidate.token_ids),
                "no": len(self.no_candidate.token_ids),
            },
            "candidate_texts": {
                "yes": self.yes_candidate.answer_text,
                "no": self.no_candidate.answer_text,
            },
        }


@dataclass(frozen=True)
class AugmentationConsistencyScore:
    """Normalized answer disagreement between one original and its views."""

    original_answer: str | None
    transformed_answers: tuple[str | None, ...]
    disagreement_count: int

    @property
    def transformed_view_count(self) -> int:
        return len(self.transformed_answers)

    @property
    def uncertainty_score(self) -> float:
        """Fraction of transformed views whose answer differs from the original."""

        return self.disagreement_count / self.transformed_view_count

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable score with the compared answer labels."""

        return {
            "method": "normalized_answer_disagreement_from_original",
            "uncertainty_score": self.uncertainty_score,
            "score_direction": "higher_means_less_augmentation_consistency",
            "original_normalized_answer": self.original_answer,
            "transformed_normalized_answers": list(self.transformed_answers),
            "disagreement_count": self.disagreement_count,
            "transformed_view_count": self.transformed_view_count,
            "normalization_note": (
                "Parsed answers are lowercase yes/no; ambiguous outputs are None and count "
                "as a separate answer state."
            ),
        }


def augmentation_consistency_score(
    original_answer: str | None,
    transformed_answers: Sequence[str | None],
) -> AugmentationConsistencyScore:
    """Score the share of transformed answers that disagree with the original.

    Inputs must already be normalized to ``yes``, ``no``, or ``None``. ``None``
    represents an ambiguous/unparseable answer and is compared as its own state.
    Thus this measures answer instability under the specified views, not the
    model's absolute uncertainty or probability of hallucination.
    """

    valid_answers = {"yes", "no", None}
    if original_answer not in valid_answers:
        raise UncertaintyScoreError("original_answer must be 'yes', 'no', or None")
    if not transformed_answers:
        raise UncertaintyScoreError("at least one transformed answer is required")
    answers = tuple(transformed_answers)
    if any(answer not in valid_answers for answer in answers):
        raise UncertaintyScoreError("transformed answers must be 'yes', 'no', or None")
    disagreements = sum(answer != original_answer for answer in answers)
    return AugmentationConsistencyScore(
        original_answer=original_answer,
        transformed_answers=answers,
        disagreement_count=disagreements,
    )


class CandidateScoringAdapter(Protocol):
    """The part of a VLM adapter required for candidate-answer scoring."""

    def score_candidate_answers(
        self,
        image: Any,
        question: str,
        candidate_answers: Sequence[str],
    ) -> Mapping[str, CandidateSequenceLogProbability]: ...


def _numeric_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UncertaintyScoreError(f"{name} must be a numeric vector") from exc
    if array.ndim != 1 or array.size == 0:
        raise UncertaintyScoreError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all():
        raise UncertaintyScoreError(f"{name} must contain only finite values")
    return array


def entropy_from_probabilities(probabilities: Sequence[float] | np.ndarray) -> float:
    """Compute Shannon entropy in nats from finite nonnegative weights.

    The weights are normalized to sum to one. Zeros are allowed and contribute
    zero to entropy; negative, non-finite, empty, or all-zero inputs are rejected.
    """

    values = _numeric_vector(probabilities, "probabilities")
    if np.any(values < 0):
        raise UncertaintyScoreError("probabilities must be nonnegative")
    scale = float(values.max())
    if scale == 0:
        raise UncertaintyScoreError("probabilities must have a positive total")

    scaled = values / scale
    normalized = scaled / scaled.sum()
    positive = normalized > 0
    return float(-np.sum(normalized[positive] * np.log(normalized[positive])))


def entropy_from_logits(logits: Sequence[float] | np.ndarray) -> float:
    """Compute full-vocabulary next-token entropy in nats from logits.

    The softmax is evaluated after subtracting the maximum logit, avoiding
    overflow for large finite values. This score covers the complete supplied
    vocabulary vector, unlike the binary POPE answer entropy below.
    """

    values = _numeric_vector(logits, "logits")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        shifted = values - values.max()
        log_normalizer = float(np.log(np.exp(shifted).sum()))
    log_probabilities = shifted - log_normalizer
    probabilities = np.exp(log_probabilities)
    positive = probabilities > 0
    return max(0.0, float(-np.sum(probabilities[positive] * log_probabilities[positive])))


def binary_entropy_from_log_scores(
    yes_log_score: float,
    no_log_score: float,
) -> dict[str, float]:
    """Normalize two candidate sequence log scores and compute binary entropy.

    The scores may be very small (large negative log-likelihoods). A stable
    two-element softmax yields a distribution conditioned on the candidate set
    ``{Yes, No}``. The returned normalized entropy is ``H / log(2)`` in [0, 1].
    """

    try:
        scores = np.asarray([yes_log_score, no_log_score], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UncertaintyScoreError("binary log scores must be numeric") from exc
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise UncertaintyScoreError("binary log scores must not be NaN or +infinity")
    maximum = float(np.max(scores))
    if not isfinite(maximum):
        raise UncertaintyScoreError("at least one binary log score must be finite")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        weights = np.exp(scores - maximum)
    probabilities = weights / weights.sum()
    entropy_nats = entropy_from_probabilities(probabilities)
    normalized_entropy = min(1.0, max(0.0, entropy_nats / log(2.0)))
    return {
        "yes_probability": float(probabilities[0]),
        "no_probability": float(probabilities[1]),
        "entropy_nats": entropy_nats,
        "normalized_entropy": normalized_entropy,
    }


def score_binary_answer_entropy(
    adapter: CandidateScoringAdapter,
    image: Any,
    question: str,
) -> BinaryAnswerEntropyScore:
    """Score the answer-position likelihoods for the fixed strings ``Yes``/``No``.

    The adapter must return actual sequence log probabilities from model logits;
    generated-answer text is not used to create this score.
    """

    scores = adapter.score_candidate_answers(image, question, ("Yes", "No"))
    if "Yes" not in scores or "No" not in scores:
        raise UncertaintyScoreError("adapter must score both 'Yes' and 'No' candidates")
    yes_candidate = scores["Yes"]
    no_candidate = scores["No"]
    if not isinstance(yes_candidate, CandidateSequenceLogProbability) or not isinstance(
        no_candidate, CandidateSequenceLogProbability
    ):
        raise UncertaintyScoreError("adapter returned an invalid candidate score")

    distribution = binary_entropy_from_log_scores(
        yes_candidate.log_probability,
        no_candidate.log_probability,
    )
    return BinaryAnswerEntropyScore(
        yes_candidate=yes_candidate,
        no_candidate=no_candidate,
        **distribution,
    )
