"""Loading and deterministic sampling for POPE question annotations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class PopeDataError(ValueError):
    """Raised when a POPE annotation file is malformed or incomplete."""


@dataclass(frozen=True)
class PopeRecord:
    """A validated POPE annotation with both raw and normalized labels."""

    question_id: int | str
    image: str
    text: str
    label: str
    normalized_label: str
    category: str | None
    raw_record: dict[str, Any] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return source fields unchanged, plus the normalized label."""

        result = dict(self.raw_record)
        result["normalized_label"] = self.normalized_label
        if self.category is not None:
            if result.get("category") is None:
                result["category"] = self.category
        return result


def normalize_yes_no(value: Any) -> str:
    """Normalize a POPE label to lowercase yes or no.

    Leading/trailing whitespace, case, and terminal sentence punctuation are
    ignored. The input value itself is retained on PopeRecord.
    """

    if not isinstance(value, str):
        raise PopeDataError("ground-truth label must be a string containing yes or no")

    normalized = value.strip().casefold()
    normalized = re.sub(r"[.!?]+$", "", normalized).strip()
    if normalized not in {"yes", "no"}:
        raise PopeDataError(f"ground-truth label must be yes or no; received {value!r}")
    return normalized


def _category_from_filename(source: Path) -> str | None:
    """Read a known POPE sampling strategy from a standard filename."""

    match = re.search(
        r"(?:^|[_-])(random|popular|adversarial)(?:$|[_-])",
        source.stem,
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else None


def _required_text(record: dict[str, Any], key: str, row_number: int, source: Path) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PopeDataError(
            f"{source}: record {row_number} requires a non-empty string field {key!r}"
        )
    return value


def _validate_record(
    value: Any,
    row_number: int,
    source: Path,
    filename_category: str | None,
) -> PopeRecord:
    if not isinstance(value, dict):
        raise PopeDataError(f"{source}: record {row_number} must be a JSON object")

    question_id = value.get("question_id")
    valid_question_id = (
        isinstance(question_id, int) and not isinstance(question_id, bool)
    ) or (isinstance(question_id, str) and bool(question_id.strip()))
    if not valid_question_id:
        raise PopeDataError(
            f"{source}: record {row_number} requires a non-empty string or integer "
            "field 'question_id'"
        )

    image = _required_text(value, "image", row_number, source)
    text = _required_text(value, "text", row_number, source)
    if "label" not in value:
        raise PopeDataError(f"{source}: record {row_number} is missing required field 'label'")

    label = value["label"]
    try:
        normalized_label = normalize_yes_no(label)
    except PopeDataError as exc:
        raise PopeDataError(f"{source}: record {row_number}: {exc}") from exc

    category = value.get("category")
    if category is not None:
        if not isinstance(category, str) or not category.strip():
            raise PopeDataError(
                f"{source}: record {row_number} field 'category' must be a non-empty string"
            )
    else:
        category = filename_category

    return PopeRecord(
        question_id=question_id,
        image=image,
        text=text,
        label=label,
        normalized_label=normalized_label,
        category=category,
        raw_record=dict(value),
    )


def _decode_records(contents: str, source: Path) -> list[Any]:
    """Decode a JSON array/object or newline-delimited JSON objects."""

    if not contents.strip():
        raise PopeDataError(f"{source}: file is empty")

    try:
        decoded = json.loads(contents)
    except json.JSONDecodeError:
        decoded = []
        for line_number, line in enumerate(contents.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                decoded.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PopeDataError(
                    f"{source}: invalid JSON on line {line_number}: {exc.msg}"
                ) from exc
    else:
        if isinstance(decoded, dict):
            decoded = [decoded]
        elif not isinstance(decoded, list):
            raise PopeDataError(
                f"{source}: top-level JSON value must be an object or an array of objects"
            )

    if not decoded:
        raise PopeDataError(f"{source}: no annotation records found")
    return decoded


def load_pope_records(input_file: str | Path) -> list[PopeRecord]:
    """Load POPE annotations from a JSON object/array or JSONL file.

    The verified source fields are question_id, image, text, and label. A
    record-level category is preserved when present. For the authors' standard
    filenames, a missing category is derived from the random, popular, or
    adversarial filename suffix.
    """

    source = Path(input_file)
    try:
        contents = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise PopeDataError(f"could not read {source}: {exc}") from exc

    values = _decode_records(contents, source)
    filename_category = _category_from_filename(source)
    return [
        _validate_record(value, row_number, source, filename_category)
        for row_number, value in enumerate(values, start=1)
    ]


def sample_pilot_subset(
    records: list[PopeRecord],
    sample_size: int,
    random_seed: int = 2027,
) -> list[PopeRecord]:
    """Sample distinct image groups with NumPy's PCG64 generator.

    sample_size counts distinct image identifiers, not question rows. Every
    record belonging to a selected image is kept, preventing questions from one
    image crossing a later calibration/test split.
    """

    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")

    image_ids = sorted({record.image for record in records})
    if sample_size > len(image_ids):
        raise ValueError(
            f"sample_size ({sample_size}) exceeds the {len(image_ids)} unique images available"
        )

    generator = np.random.Generator(np.random.PCG64(random_seed))
    sampled_indices = generator.choice(len(image_ids), size=sample_size, replace=False)
    sampled_images = [image_ids[int(index)] for index in sampled_indices]
    records_by_image = {image_id: [] for image_id in sampled_images}
    for record in records:
        if record.image in records_by_image:
            records_by_image[record.image].append(record)
    return [
        record
        for image_id in sampled_images
        for record in records_by_image[image_id]
    ]
