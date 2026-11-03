"""Conservative normalization of free-form POPE answers."""

from __future__ import annotations

import re

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


def normalize_prediction(raw_prediction: str) -> tuple[str | None, str]:
    """Extract a leading yes/no answer while preserving ambiguity explicitly."""

    if not isinstance(raw_prediction, str) or not raw_prediction.strip():
        return None, "ambiguous"
    if _UNCERTAINTY.search(raw_prediction):
        return None, "ambiguous"
    match = _ANSWER_PREFIX.match(raw_prediction)
    answer_words = {word.casefold() for word in _ANSWER_WORD.findall(raw_prediction)}
    if match is not None:
        first_answer = match.group(1).casefold()
        if (
            len(answer_words) != 1
            or first_answer == "no"
            and re.match(r"^\s*no\s+doubt\b", raw_prediction, flags=re.IGNORECASE)
        ):
            return None, "ambiguous"
        return first_answer, "parsed"
    if _DIRECT_NEGATIVE.search(raw_prediction) and "yes" not in answer_words:
        return "no", "parsed"
    if _DIRECT_POSITIVE.search(raw_prediction) and "no" not in answer_words:
        return "yes", "parsed"
    return None, "ambiguous"
