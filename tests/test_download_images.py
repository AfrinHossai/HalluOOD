import io
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_images import (  # noqa: E402
    COCO_IMAGE_BASE_URL,
    ImageDownloadError,
    ImagePreparationError,
    download_image,
    load_pilot_image_filenames,
    validate_pilot_images,
)

CATEGORIES = ("random", "popular", "adversarial")
IMAGE_NAMES = (
    "COCO_val2014_000000000123.jpg",
    "COCO_val2014_000000000456.jpg",
)


def _jpeg_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), color=(40, 80, 120)).save(stream, format="JPEG")
    return stream.getvalue()


def _write_pilot_fixture(tmp_path, filenames=IMAGE_NAMES, missing_category=None):
    pilot_dir = tmp_path / "processed"
    pilot_dir.mkdir()
    metadata = {
        "sampling": {"selected_image_ids_in_sampling_order": list(filenames)},
        "totals": {"distinct_image_ids": len(filenames)},
        "pilot_files": {},
    }
    for category in CATEGORIES:
        output_name = f"coco_pope_{category}_pilot.jsonl"
        metadata["pilot_files"][category] = {"filename": output_name}
        names = filenames[:-1] if category == missing_category else filenames
        rows = [
            {
                "question_id": index,
                "image": image,
                "text": "Is there a chair in the image?",
                "label": "yes",
            }
            for index, image in enumerate(names, start=1)
        ]
        (pilot_dir / output_name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    metadata_path = pilot_dir / "pope_pilot_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return pilot_dir, metadata_path


class _MockResponse(io.BytesIO):
    def __init__(self, payload, url):
        super().__init__(payload)
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def geturl(self):
        return self.url


def test_loads_only_image_filenames_shared_by_all_pilot_files(tmp_path):
    pilot_dir, metadata_path = _write_pilot_fixture(tmp_path)

    names = load_pilot_image_filenames(pilot_dir, metadata_path)

    assert names == list(IMAGE_NAMES)


def test_rejects_strategy_file_that_does_not_match_metadata(tmp_path):
    pilot_dir, metadata_path = _write_pilot_fixture(
        tmp_path, missing_category="adversarial"
    )

    with pytest.raises(ImagePreparationError, match="image IDs differ"):
        load_pilot_image_filenames(pilot_dir, metadata_path)


def test_skips_existing_readable_image_without_network_request(tmp_path):
    image_root = tmp_path / "images"
    target = image_root / "val2014" / IMAGE_NAMES[0]
    target.parent.mkdir(parents=True)
    target.write_bytes(_jpeg_bytes())
    open_url = Mock()

    result = download_image(IMAGE_NAMES[0], image_root, open_url=open_url)

    assert result == "skipped"
    open_url.assert_not_called()
    validate_pilot_images([IMAGE_NAMES[0]], image_root)


def test_retries_timeout_then_downloads_official_jpeg(tmp_path):
    url = COCO_IMAGE_BASE_URL + IMAGE_NAMES[0]
    open_url = Mock(
        side_effect=[
            TimeoutError("temporary timeout"),
            _MockResponse(_jpeg_bytes(), url),
        ]
    )
    sleeps = []

    result = download_image(
        IMAGE_NAMES[0],
        tmp_path,
        retries=1,
        timeout=7.5,
        open_url=open_url,
        sleep=sleeps.append,
    )

    assert result == "downloaded"
    assert open_url.call_count == 2
    request, = open_url.call_args.args
    assert request.full_url == url
    assert open_url.call_args.kwargs == {"timeout": 7.5}
    assert sleeps == [0.5]
    validate_pilot_images([IMAGE_NAMES[0]], tmp_path)


def test_rejects_unreadable_bytes_and_does_not_leave_partial_file(tmp_path):
    url = COCO_IMAGE_BASE_URL + IMAGE_NAMES[0]
    open_url = Mock(return_value=_MockResponse(b"not a jpeg", url))

    with pytest.raises(ImageDownloadError, match="not a readable JPEG"):
        download_image(IMAGE_NAMES[0], tmp_path, retries=0, open_url=open_url)

    assert not (tmp_path / "val2014" / IMAGE_NAMES[0]).exists()
    assert list((tmp_path / "val2014").iterdir()) == []


def test_rejects_non_validation_filename():
    with pytest.raises(ImagePreparationError, match="COCO_val2014"):
        download_image("COCO_train2014_000000000123.jpg", "/tmp/images")


def test_validation_reports_missing_image(tmp_path):
    with pytest.raises(ImagePreparationError, match="missing or unreadable"):
        validate_pilot_images([IMAGE_NAMES[0]], tmp_path)
