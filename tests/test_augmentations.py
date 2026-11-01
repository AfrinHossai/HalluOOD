from __future__ import annotations

import pytest
from PIL import Image, ImageChops, ImageDraw

from halluood.augmentations import AugmentationError, create_mild_augmentations


def _synthetic_image(width=200, height=160):
    image = Image.new("RGB", (width, height), color=(80, 110, 140))
    draw = ImageDraw.Draw(image)
    draw.rectangle((width // 2 - 12, height // 2 - 12, width // 2 + 12, height // 2 + 12),
                   fill=(220, 40, 30))
    return image


def test_mild_augmentations_are_seeded_same_size_and_do_not_mutate_input():
    image = _synthetic_image()
    original_pixels = image.tobytes()

    first = create_mild_augmentations(image, random_seed=31)
    second = create_mild_augmentations(image, random_seed=31)
    try:
        assert [view.name for view in first] == [view.name for view in second]
        assert [view.parameters for view in first] == [view.parameters for view in second]
        assert len(first) == 6
        assert all(view.image.size == image.size and view.image.mode == "RGB" for view in first)
        assert all(
            ImageChops.difference(left.image, right.image).getbbox() is None
            for left, right in zip(first, second, strict=True)
        )
        assert all(ImageChops.difference(image, view.image).getbbox() for view in first)
        assert image.tobytes() == original_pixels
        for view in first[-2:]:
            assert abs(view.parameters["dx_pixels"]) <= round(image.width * 0.01) or (
                view.parameters["dx_pixels"] == 0
            )
            assert abs(view.parameters["dy_pixels"]) <= round(image.height * 0.01) or (
                view.parameters["dy_pixels"] == 0
            )
        for view in first:
            red, green, blue = view.image.getpixel((100, 80))
            assert red > green * 2 and red > blue * 2
    finally:
        for view in [*first, *second]:
            view.image.close()
        image.close()


def test_small_images_skip_translations_and_keep_dimensions():
    image = Image.new("RGB", (16, 12), color=(10, 20, 30))
    views = create_mild_augmentations(image)
    try:
        assert [view.name for view in views] == [
            "brightness_0.90",
            "brightness_1.10",
            "contrast_0.90",
            "contrast_1.10",
        ]
        assert all(view.image.size == image.size for view in views)
    finally:
        for view in views:
            view.image.close()
        image.close()


def test_augmentation_builder_accepts_a_local_image_path(tmp_path):
    image_path = tmp_path / "synthetic.png"
    _synthetic_image().save(image_path)

    views = create_mild_augmentations(image_path, random_seed=9)
    try:
        assert len(views) == 6
        assert all(view.image.size == (200, 160) for view in views)
    finally:
        for view in views:
            view.image.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"random_seed": -1}, "random_seed"),
        ({"random_seed": True}, "random_seed"),
        ({"shift_fraction": 0}, "shift_fraction"),
        ({"shift_fraction": 0.1}, "shift_fraction"),
    ],
)
def test_augmentation_builder_rejects_invalid_parameters(kwargs, message):
    image = Image.new("RGB", (64, 64), color=(0, 0, 0))
    try:
        with pytest.raises(AugmentationError, match=message):
            create_mild_augmentations(image, **kwargs)
    finally:
        image.close()
