"""Synthetic-image tests for controlled distribution shifts."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from halluood.shifts import (
    SHIFT_PARAMETERS,
    DistributionShiftError,
    apply_distribution_shift,
    derive_shift_seed,
)


def _pattern_image() -> Image.Image:
    x = np.arange(48, dtype=np.uint8)[None, :, None]
    y = np.arange(32, dtype=np.uint8)[:, None, None]
    pixels = np.concatenate(
        (
            np.broadcast_to(x * 3, (32, 48, 1)),
            np.broadcast_to(y * 5, (32, 48, 1)),
            np.broadcast_to((x + y) * 2, (32, 48, 1)),
        ),
        axis=2,
    )
    return Image.fromarray(pixels)


@pytest.mark.parametrize("perturbation_type", tuple(SHIFT_PARAMETERS))
@pytest.mark.parametrize("severity", (1, 2, 3))
def test_each_shift_is_deterministic_and_preserves_image_identity_and_size(
    perturbation_type: str, severity: int
) -> None:
    source = _pattern_image()
    before = np.asarray(source).copy()
    first = apply_distribution_shift(
        source,
        image_id="000000123456",
        perturbation_type=perturbation_type,
        severity=severity,
        random_seed=2027,
    )
    second = apply_distribution_shift(
        source,
        image_id="000000123456",
        perturbation_type=perturbation_type,
        severity=severity,
        random_seed=2027,
    )
    try:
        assert first.image.size == source.size == (48, 32)
        assert first.image_id == "000000123456"
        assert first.perturbation_type == perturbation_type
        assert first.severity == severity
        assert first.random_seed == 2027
        assert first.parameters == SHIFT_PARAMETERS[perturbation_type][severity - 1]
        assert first.condition_id == f"{perturbation_type}_severity_{severity}"
        np.testing.assert_array_equal(np.asarray(first.image), np.asarray(second.image))
        np.testing.assert_array_equal(np.asarray(source), before)
    finally:
        first.image.close()
        second.image.close()
        source.close()


def test_gaussian_noise_changes_with_seed() -> None:
    source = _pattern_image()
    first = apply_distribution_shift(
        source,
        image_id="synthetic-1",
        perturbation_type="gaussian_noise",
        severity=2,
        random_seed=1,
    )
    second = apply_distribution_shift(
        source,
        image_id="synthetic-1",
        perturbation_type="gaussian_noise",
        severity=2,
        random_seed=2,
    )
    try:
        assert not np.array_equal(np.asarray(first.image), np.asarray(second.image))
    finally:
        source.close()
        first.image.close()
        second.image.close()


def test_per_image_condition_seed_is_stable_and_distinct() -> None:
    first = derive_shift_seed(2027, "000000123456", "gaussian_noise", 2)

    assert first == derive_shift_seed(2027, "000000123456", "gaussian_noise", 2)
    assert first != derive_shift_seed(2027, "000000123456", "gaussian_noise", 3)
    assert first != derive_shift_seed(2027, "000000123457", "gaussian_noise", 2)
    assert 0 <= first < 2**32


def test_path_input_is_read_only_and_grayscale_becomes_rgb(tmp_path) -> None:
    path = tmp_path / "gray.png"
    Image.new("L", (13, 9), color=120).save(path)
    original_bytes = path.read_bytes()

    shifted = apply_distribution_shift(
        path,
        image_id=123,
        perturbation_type="brightness_reduction",
        severity=1,
        random_seed=3,
    )

    try:
        assert shifted.image.mode == "RGB"
        assert shifted.image.size == (13, 9)
        assert shifted.image_id == "123"
        assert path.read_bytes() == original_bytes
    finally:
        shifted.image.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"perturbation_type": "rotate", "severity": 1, "random_seed": 1}, "unknown"),
        (
            {"perturbation_type": "gaussian_blur", "severity": 0, "random_seed": 1},
            "severity",
        ),
        (
            {"perturbation_type": "gaussian_blur", "severity": 4, "random_seed": 1},
            "severity",
        ),
        (
            {"perturbation_type": "gaussian_blur", "severity": True, "random_seed": 1},
            "severity",
        ),
        (
            {"perturbation_type": "gaussian_noise", "severity": 1, "random_seed": -1},
            "random_seed",
        ),
    ),
)
def test_invalid_shift_settings_are_rejected(kwargs, message: str) -> None:
    source = _pattern_image()
    with pytest.raises(DistributionShiftError, match=message):
        apply_distribution_shift(source, image_id="x", **kwargs)
    source.close()


def test_invalid_image_id_and_image_source_are_rejected() -> None:
    source = _pattern_image()
    with pytest.raises(DistributionShiftError, match="image_id"):
        apply_distribution_shift(
            source,
            image_id=" ",
            perturbation_type="gaussian_blur",
            severity=1,
        )
    source.close()
    with pytest.raises(DistributionShiftError, match="could not read image"):
        apply_distribution_shift(
            "/path/that/does/not/exist.jpg",
            image_id="synthetic-1",
            perturbation_type="gaussian_blur",
            severity=1,
        )
