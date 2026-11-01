"""Create a deterministic local preview of controlled COCO image shifts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from PIL import __version__ as PILLOW_VERSION

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluood.data import PopeDataError, load_pope_records
from halluood.embeddings import EmbeddingError, normalize_coco_image_id
from halluood.shifts import (
    SHIFT_PARAMETERS,
    DistributionShiftError,
    apply_distribution_shift,
)

CATEGORIES = ("random", "popular", "adversarial")
DEFAULT_ANNOTATIONS_DIR = Path("data/processed")
DEFAULT_IMAGE_DIR = Path("data/images/val2014")
DEFAULT_OUTPUT_DIR = Path("data/images/shifts_preview")
SAMPLING_METHOD = "PCG64 over sorted shared POPE pilot image IDs, without replacement"
SEED_DERIVATION = "first 32 bits of SHA-256(sample_seed:image_id:type:severity)"


class ShiftPreparationError(ValueError):
    """Raised when pilot images cannot be sampled or prepared safely."""


def _pilot_image_names(annotations_dir: Path) -> dict[str, str]:
    per_category: dict[str, dict[str, str]] = {}
    for category in CATEGORIES:
        records = load_pope_records(
            annotations_dir / f"coco_pope_{category}_pilot.jsonl"
        )
        image_names: dict[str, str] = {}
        for record in records:
            try:
                image_id = normalize_coco_image_id(record.image)
            except EmbeddingError as exc:
                raise ShiftPreparationError(
                    f"unsupported COCO image identifier {record.image!r}"
                ) from exc
            previous = image_names.setdefault(image_id, record.image)
            if previous != record.image:
                raise ShiftPreparationError(
                    f"pilot annotations contain multiple filenames for image {image_id}"
                )
        per_category[category] = image_names

    shared_ids = set.intersection(*(set(mapping) for mapping in per_category.values()))
    if not shared_ids:
        raise ShiftPreparationError("POPE pilot annotation categories have no shared images")
    for category, image_names in per_category.items():
        for image_id in shared_ids:
            names = {mapping[image_id] for mapping in per_category.values()}
            if len(names) != 1:
                raise ShiftPreparationError(
                    f"pilot categories disagree on the filename for image {image_id}"
                )
        if not image_names:
            raise ShiftPreparationError(f"no images were found in {category} annotations")
    return {
        image_id: per_category[CATEGORIES[0]][image_id]
        for image_id in sorted(shared_ids)
    }


def _derive_perturbation_seed(seed: int, image_id: str, kind: str, severity: int) -> int:
    payload = f"{seed}:{image_id}:{kind}:{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _validate_source(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"pilot image does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise ShiftPreparationError(f"pilot image is unreadable: {path}: {exc}") from exc


def prepare_shift_preview(
    *,
    annotations_dir: str | Path = DEFAULT_ANNOTATIONS_DIR,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sample_size: int = 1,
    random_seed: int = 2027,
) -> dict[str, Any]:
    """Sample pilot images and save clean references plus all shift severities."""

    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ShiftPreparationError("sample_size must be a positive integer")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ShiftPreparationError("random_seed must be a non-negative integer")

    image_names = _pilot_image_names(Path(annotations_dir))
    if sample_size > len(image_names):
        raise ShiftPreparationError(
            f"sample_size ({sample_size}) exceeds the {len(image_names)} shared pilot images"
        )
    sampler = np.random.Generator(np.random.PCG64(random_seed))
    selected = sorted(
        str(image_id)
        for image_id in sampler.choice(
            np.asarray(sorted(image_names), dtype=object),
            size=sample_size,
            replace=False,
        )
    )

    image_root = Path(image_dir)
    output_root = Path(output_dir)
    resolved_image_root = image_root.resolve()
    resolved_output_root = output_root.resolve()
    if resolved_output_root == resolved_image_root or resolved_image_root in (
        resolved_output_root,
        *resolved_output_root.parents,
    ):
        raise ShiftPreparationError(
            "output_dir must be separate from and outside the original COCO image directory"
        )

    manifest_rows: list[dict[str, Any]] = []
    for image_id in selected:
        filename = image_names[image_id]
        source_path = image_root / filename
        width, height = _validate_source(source_path)
        manifest_rows.append(
            {
                "image_id": image_id,
                "image_filename": filename,
                "condition_id": "clean",
                "perturbation_type": "none",
                "severity": 0,
                "parameters": {},
                "random_seed": None,
                "sampling_seed": random_seed,
                "sampling_method": SAMPLING_METHOD,
                "source_image": str(source_path),
                "output_image": str(source_path),
                "is_original": True,
                "width": width,
                "height": height,
                "output_format": "original",
                "pillow_version": PILLOW_VERSION,
                "numpy_version": np.__version__,
            }
        )
        for perturbation_type in SHIFT_PARAMETERS:
            for severity in (1, 2, 3):
                perturbation_seed = _derive_perturbation_seed(
                    random_seed, image_id, perturbation_type, severity
                )
                shifted = apply_distribution_shift(
                    source_path,
                    image_id=image_id,
                    perturbation_type=perturbation_type,
                    severity=severity,
                    random_seed=perturbation_seed,
                )
                try:
                    relative_output = Path(perturbation_type) / f"severity_{severity}" / (
                        f"COCO_val2014_{image_id}.png"
                    )
                    output_path = output_root / relative_output
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shifted.image.save(output_path, format="PNG")
                    row = shifted.to_dict()
                    row.update(
                        {
                            "image_filename": filename,
                            "sampling_seed": random_seed,
                            "sampling_method": SAMPLING_METHOD,
                            "seed_derivation": SEED_DERIVATION,
                            "source_image": str(source_path),
                            "output_image": str(relative_output),
                            "is_original": False,
                            "output_format": "PNG",
                            "pillow_version": PILLOW_VERSION,
                            "numpy_version": np.__version__,
                        }
                    )
                    manifest_rows.append(row)
                finally:
                    shifted.image.close()

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in manifest_rows
        ),
        encoding="utf-8",
    )
    summary = {
        "sample_seed": random_seed,
        "sample_size": sample_size,
        "image_ids": selected,
        "perturbation_types": list(SHIFT_PARAMETERS),
        "severity_levels": [1, 2, 3],
        "sampling_method": SAMPLING_METHOD,
        "perturbation_seed_derivation": SEED_DERIVATION,
        "shifted_image_count": sample_size * len(SHIFT_PARAMETERS) * 3,
        "clean_reference_count": sample_size,
        "manifest": str(manifest_path),
        "output_dir": str(output_root),
        "pillow_version": PILLOW_VERSION,
        "numpy_version": np.__version__,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create local PNG previews and a manifest for COCO pilot image shifts."
    )
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2027)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepare_shift_preview(
            annotations_dir=args.annotations_dir,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            random_seed=args.seed,
        )
    except (DistributionShiftError, PopeDataError, OSError, ValueError) as exc:
        print(f"prepare_shifts.py: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
