"""Deterministic, mild Pillow transforms for augmentation-consistency checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageStat, UnidentifiedImageError


class AugmentationError(ValueError):
    """Raised when an input image or transform configuration is invalid."""


@dataclass(frozen=True)
class AugmentedImage:
    """One transformed RGB image and the parameters needed to reproduce it."""

    name: str
    parameters: dict[str, Any]
    image: Image.Image = field(repr=False, compare=False)


def _load_rgb(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if not isinstance(image, (str, Path)):
        raise TypeError("image must be a Pillow image or a local path")
    try:
        with Image.open(image) as opened:
            opened.load()
            return opened.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise AugmentationError(f"could not read image {image!s}: {exc}") from exc


def create_mild_augmentations(
    image: Image.Image | str | Path,
    *,
    random_seed: int = 2027,
    shift_fraction: float = 0.01,
) -> list[AugmentedImage]:
    """Create fixed brightness/contrast views and seeded 1%-scale translations.

    Image dimensions are preserved. A horizontal or vertical translation is
    included when its selected fraction yields at least one pixel. The sign is
    sampled with NumPy PCG64 using the provided seed. Translation fills the thin
    exposed strip with the image's mean RGB color. The default shift is one
    percent of a dimension, which is intended to perturb location slightly
    without removing most of an object.

    The caller owns and should close each returned view. This function never
    mutates the input image and does not write transformed image files.
    """

    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise AugmentationError("random_seed must be a non-negative integer")
    if (
        isinstance(shift_fraction, bool)
        or not isinstance(shift_fraction, (int, float))
        or not np.isfinite(shift_fraction)
        or not 0 < shift_fraction <= 0.05
    ):
        raise AugmentationError("shift_fraction must be greater than 0 and at most 0.05")

    rgb = _load_rgb(image)
    try:
        width, height = rgb.size
        views: list[AugmentedImage] = []
        for name, kind, factor in (
            ("brightness_0.90", "brightness", 0.9),
            ("brightness_1.10", "brightness", 1.1),
            ("contrast_0.90", "contrast", 0.9),
            ("contrast_1.10", "contrast", 1.1),
        ):
            enhancer = ImageEnhance.Brightness if kind == "brightness" else ImageEnhance.Contrast
            views.append(
                AugmentedImage(
                    name=name,
                    parameters={"kind": kind, "factor": factor},
                    image=enhancer(rgb).enhance(factor),
                )
            )

        rng = np.random.Generator(np.random.PCG64(random_seed))
        fill_color = tuple(int(round(channel)) for channel in ImageStat.Stat(rgb).mean[:3])
        shift_specs: list[tuple[str, int, int, int]] = []
        dx = int(width * shift_fraction)
        if dx > 0:
            dx *= -1 if int(rng.integers(0, 2)) == 0 else 1
            shift_specs.append(("shift_horizontal", dx, 0, width))
        dy = int(height * shift_fraction)
        if dy > 0:
            dy *= -1 if int(rng.integers(0, 2)) == 0 else 1
            shift_specs.append(("shift_vertical", 0, dy, height))

        for name, dx, dy, dimension in shift_specs:
            shifted = rgb.transform(
                rgb.size,
                Image.Transform.AFFINE,
                (1, 0, -dx, 0, 1, -dy),
                resample=Image.Resampling.BICUBIC,
                fillcolor=fill_color,
            )
            views.append(
                AugmentedImage(
                    name=name,
                    parameters={
                        "kind": "translation",
                        "dx_pixels": dx,
                        "dy_pixels": dy,
                        "shift_fraction": shift_fraction,
                        "dimension_pixels": dimension,
                        "fill_rgb": list(fill_color),
                    },
                    image=shifted,
                )
            )
        return views
    except Exception:
        # If a transform fails midway, release any images already created.
        for view in locals().get("views", []):
            view.image.close()
        raise
    finally:
        rgb.close()
