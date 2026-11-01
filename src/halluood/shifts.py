"""Deterministic image perturbations for controlled visual distribution shifts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError


class DistributionShiftError(ValueError):
    """Raised when a shift request or source image is invalid."""


SHIFT_PARAMETERS: dict[str, tuple[dict[str, Any], ...]] = {
    "gaussian_blur": (
        {"radius_pixels": 0.5},
        {"radius_pixels": 1.5},
        {"radius_pixels": 3.0},
    ),
    "brightness_reduction": (
        {"factor": 0.8},
        {"factor": 0.6},
        {"factor": 0.4},
    ),
    "gaussian_noise": (
        {"standard_deviation_0_255": 5.0},
        {"standard_deviation_0_255": 15.0},
        {"standard_deviation_0_255": 30.0},
    ),
    "contrast_reduction": (
        {"factor": 0.8},
        {"factor": 0.6},
        {"factor": 0.4},
    ),
}


def derive_shift_seed(
    sampling_seed: int,
    image_id: str | int,
    perturbation_type: str,
    severity: int,
) -> int:
    """Derive a stable per-image/condition 32-bit seed from an experiment seed."""

    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
        raise DistributionShiftError("sampling_seed must be a non-negative integer")
    if sampling_seed < 0:
        raise DistributionShiftError("sampling_seed must be a non-negative integer")
    if isinstance(image_id, bool) or not isinstance(image_id, (str, int)):
        raise DistributionShiftError("image_id must be a non-empty string or integer")
    normalized_image_id = str(image_id).strip()
    if not normalized_image_id:
        raise DistributionShiftError("image_id must be a non-empty string or integer")
    if perturbation_type not in SHIFT_PARAMETERS:
        raise DistributionShiftError("unknown perturbation_type")
    if isinstance(severity, bool) or not isinstance(severity, int) or severity not in (1, 2, 3):
        raise DistributionShiftError("severity must be an integer from 1 through 3")
    payload = (
        f"{sampling_seed}:{normalized_image_id}:{perturbation_type}:{severity}".encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


@dataclass(frozen=True)
class ShiftedImage:
    """A transformed RGB image and the metadata needed to reproduce it."""

    image_id: str
    perturbation_type: str
    severity: int
    parameters: dict[str, Any]
    random_seed: int
    image: Image.Image = field(repr=False, compare=False)

    @property
    def condition_id(self) -> str:
        """Stable condition identifier for a perturbation and severity."""

        return f"{self.perturbation_type}_severity_{self.severity}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize perturbation metadata without embedding image bytes."""

        return {
            "image_id": self.image_id,
            "condition_id": self.condition_id,
            "perturbation_type": self.perturbation_type,
            "severity": self.severity,
            "parameters": dict(self.parameters),
            "random_seed": self.random_seed,
            "width": self.image.width,
            "height": self.image.height,
        }


def _load_rgb(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    if not isinstance(image, (str, Path)):
        raise DistributionShiftError("image must be a Pillow image or local path")
    try:
        with Image.open(image) as opened:
            opened.load()
            return opened.convert("RGB").copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise DistributionShiftError(f"could not read image {image!s}: {exc}") from exc


def apply_distribution_shift(
    image: Image.Image | str | Path,
    *,
    image_id: str | int,
    perturbation_type: str,
    severity: int,
    random_seed: int = 2027,
) -> ShiftedImage:
    """Apply one deterministic perturbation without changing the source image.

    Severity levels are 1 (mild), 2 (moderate), and 3 (strong). Gaussian noise
    uses a NumPy PCG64 generator seeded by ``random_seed`` and independent
    zero-mean pixel/channel samples. Other transforms are deterministic but
    record the same seed for a uniform experiment manifest.
    """

    if isinstance(image_id, bool) or not isinstance(image_id, (str, int)):
        raise DistributionShiftError("image_id must be a non-empty string or integer")
    normalized_image_id = str(image_id).strip()
    if not normalized_image_id:
        raise DistributionShiftError("image_id must be a non-empty string or integer")
    if perturbation_type not in SHIFT_PARAMETERS:
        valid = ", ".join(SHIFT_PARAMETERS)
        raise DistributionShiftError(f"unknown perturbation_type; expected one of: {valid}")
    if isinstance(severity, bool) or not isinstance(severity, int) or severity not in (1, 2, 3):
        raise DistributionShiftError("severity must be an integer from 1 through 3")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise DistributionShiftError("random_seed must be a non-negative integer")

    parameters = dict(SHIFT_PARAMETERS[perturbation_type][severity - 1])
    original = _load_rgb(image)
    original_size = original.size
    try:
        if perturbation_type == "gaussian_blur":
            transformed = original.filter(
                ImageFilter.GaussianBlur(radius=parameters["radius_pixels"])
            )
        elif perturbation_type == "brightness_reduction":
            transformed = ImageEnhance.Brightness(original).enhance(parameters["factor"])
        elif perturbation_type == "contrast_reduction":
            transformed = ImageEnhance.Contrast(original).enhance(parameters["factor"])
        else:
            pixels = np.asarray(original, dtype=np.float32)
            rng = np.random.Generator(np.random.PCG64(random_seed))
            noise = rng.normal(
                loc=0.0,
                scale=parameters["standard_deviation_0_255"],
                size=pixels.shape,
            )
            noisy_pixels = np.clip(np.rint(pixels + noise), 0, 255).astype(np.uint8)
            transformed = Image.fromarray(noisy_pixels)
    finally:
        original.close()

    if transformed.size != original_size:
        transformed.close()
        raise DistributionShiftError("a perturbation unexpectedly changed image dimensions")
    return ShiftedImage(
        image_id=normalized_image_id,
        perturbation_type=perturbation_type,
        severity=severity,
        parameters=parameters,
        random_seed=random_seed,
        image=transformed,
    )
