"""Download only the COCO 2014 validation images used by the POPE pilot."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image

from halluood.data import load_pope_records

COCO_IMAGE_BASE_URL = "http://images.cocodataset.org/val2014/"
COCO_IMAGE_HOST = "images.cocodataset.org"
COCO_SPLIT = "val2014"
POPE_CATEGORIES = ("random", "popular", "adversarial")
IMAGE_FILENAME_PATTERN = re.compile(r"COCO_val2014_\d{12}\.jpg\Z")
MAX_IMAGE_BYTES = 50 * 1024 * 1024
OpenUrl = Callable[..., object]


class ImagePreparationError(ValueError):
    """Raised when pilot metadata or image files fail validation."""


class ImageDownloadError(RuntimeError):
    """Raised when an official COCO image cannot be downloaded and validated."""


def validate_image_filename(filename: object) -> str:
    """Accept only COCO 2014 validation JPEG filenames from POPE annotations."""

    if not isinstance(filename, str) or not IMAGE_FILENAME_PATTERN.fullmatch(filename):
        raise ImagePreparationError(
            f"expected a COCO_val2014_<12-digit-id>.jpg filename; received {filename!r}"
        )
    return filename


def load_pilot_image_filenames(
    pilot_dir: str | Path,
    metadata_path: str | Path,
) -> list[str]:
    """Read the pilot manifest and ensure all three POPE strategies agree."""

    pilot_dir = Path(pilot_dir)
    metadata_path = Path(metadata_path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        names = metadata["sampling"]["selected_image_ids_in_sampling_order"]
        expected_count = metadata["totals"]["distinct_image_ids"]
        pilot_files = metadata["pilot_files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        message = f"could not read pilot metadata {metadata_path}: {exc}"
        raise ImagePreparationError(message) from exc

    if not isinstance(names, list) or not names:
        raise ImagePreparationError("pilot metadata must list selected image filenames")
    filenames = [validate_image_filename(name) for name in names]
    if len(set(filenames)) != len(filenames):
        raise ImagePreparationError("pilot metadata contains duplicate image filenames")
    if expected_count != len(filenames):
        raise ImagePreparationError(
            f"metadata declares {expected_count} images but lists {len(filenames)}"
        )

    expected_names = set(filenames)
    if not isinstance(pilot_files, dict):
        raise ImagePreparationError("pilot metadata must describe the POPE strategy files")

    for category in POPE_CATEGORIES:
        try:
            filename = pilot_files[category]["filename"]
        except (KeyError, TypeError) as exc:
            raise ImagePreparationError(
                f"pilot metadata is missing the {category} annotation filename"
            ) from exc
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ImagePreparationError(f"invalid {category} annotation filename: {filename!r}")

        annotation_path = pilot_dir / filename
        try:
            records = load_pope_records(annotation_path)
        except (OSError, ValueError) as exc:
            raise ImagePreparationError(f"could not validate {category} pilot file: {exc}") from exc

        found_names = {record.image for record in records}
        if found_names != expected_names:
            missing = sorted(expected_names - found_names)
            unexpected = sorted(found_names - expected_names)
            raise ImagePreparationError(
                f"{category} pilot image IDs differ from metadata "
                f"(missing={missing[:3]}, unexpected={unexpected[:3]})"
            )
        if any(record.category != category for record in records):
            raise ImagePreparationError(
                f"{category} pilot file contains a different question category"
            )

    return filenames


def _check_official_url(url: str, filename: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    expected_path = f"/{COCO_SPLIT}/{filename}"
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname == COCO_IMAGE_HOST
        and parsed.port in (None, 80, 443)
        and parsed.path == expected_path
        and not parsed.query
        and not parsed.fragment
        and not parsed.username
        and not parsed.password
    )


class _OfficialRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow redirects only when they stay on the exact official COCO image URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target_url = urllib.parse.urljoin(req.full_url, newurl)
        filename = Path(urllib.parse.urlsplit(req.full_url).path).name
        if not _check_official_url(target_url, filename):
            raise urllib.error.HTTPError(
                req.full_url,
                code,
                "refusing redirect away from the official COCO validation image host",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, target_url)


def _open_official_url(request: urllib.request.Request, timeout: float):
    opener = urllib.request.build_opener(_OfficialRedirectHandler())
    return opener.open(request, timeout=timeout)


def _validate_image_bytes(payload: bytes) -> None:
    if not payload:
        raise ImagePreparationError("downloaded image is empty")
    if len(payload) > MAX_IMAGE_BYTES:
        raise ImagePreparationError(
            f"downloaded file exceeds the {MAX_IMAGE_BYTES}-byte image size limit"
        )
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "JPEG":
                raise ImagePreparationError(f"expected JPEG image, received {image.format!r}")
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
    except ImagePreparationError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImagePreparationError(f"downloaded file is not a readable JPEG: {exc}") from exc


def is_readable_jpeg(path: str | Path) -> bool:
    """Return whether a local file is a bounded, fully decodable JPEG."""

    path = Path(path)
    try:
        if path.stat().st_size > MAX_IMAGE_BYTES:
            return False
        _validate_image_bytes(path.read_bytes())
    except (OSError, ImagePreparationError):
        return False
    return True


def validate_pilot_images(filenames: Sequence[str], output_dir: str | Path) -> None:
    """Require one readable image for every selected POPE filename."""

    output_dir = Path(output_dir) / COCO_SPLIT
    missing_or_invalid = [
        filename
        for filename in filenames
        if not is_readable_jpeg(output_dir / validate_image_filename(filename))
    ]
    if missing_or_invalid:
        raise ImagePreparationError(
            "missing or unreadable pilot images: " + ", ".join(missing_or_invalid[:10])
        )


def download_image(
    filename: str,
    output_dir: str | Path,
    *,
    retries: int = 3,
    timeout: float = 30.0,
    open_url: OpenUrl | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Download one official COCO validation JPEG, retrying failures safely."""

    filename = validate_image_filename(filename)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    destination = Path(output_dir) / COCO_SPLIT / filename
    if is_readable_jpeg(destination):
        return "skipped"
    destination.parent.mkdir(parents=True, exist_ok=True)

    url = urllib.parse.urljoin(COCO_IMAGE_BASE_URL, filename)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "HalluOOD-research-pilot/0.1 (COCO image retrieval)"},
    )
    open_url = open_url or _open_official_url
    max_attempts = retries + 1

    for attempt in range(1, max_attempts + 1):
        temporary_path: Path | None = None
        try:
            with open_url(request, timeout=timeout) as response:
                final_url = response.geturl()
                if not _check_official_url(final_url, filename):
                    raise ImagePreparationError(
                        f"refusing non-official image URL after redirect: {final_url}"
                    )
                payload = response.read(MAX_IMAGE_BYTES + 1)
            _validate_image_bytes(payload)

            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{filename}.",
                suffix=".part",
                dir=destination.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
            return "downloaded"
        except (OSError, TimeoutError, ImagePreparationError) as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if attempt == max_attempts:
                raise ImageDownloadError(
                    f"failed to download {filename} after {max_attempts} attempt(s): {exc}"
                ) from exc
            sleep(min(0.5 * (2 ** (attempt - 1)), 8.0))

    raise AssertionError("unreachable retry loop")


def download_pilot_images(
    filenames: Sequence[str],
    output_dir: str | Path,
    *,
    retries: int = 3,
    timeout: float = 30.0,
    open_url: OpenUrl | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    """Download the specified pilot files and validate the complete set."""

    if len(set(filenames)) != len(filenames):
        raise ImagePreparationError("image filename list contains duplicates")
    results = {
        filename: download_image(
            filename,
            output_dir,
            retries=retries,
            timeout=timeout,
            open_url=open_url,
            sleep=sleep,
        )
        for filename in filenames
    }
    validate_pilot_images(filenames, output_dir)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download only the COCO val2014 images referenced by the POPE pilot."
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("data/processed"),
        help="directory containing the three prepared POPE pilot JSONL files",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/pope_pilot_metadata.json"),
        help="pilot metadata manifest that lists the selected image filenames",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/images"),
        help="local image root; downloaded files go under val2014/",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="number of retries after the initial request (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="per-request timeout in seconds (default: 30)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.retries < 0:
            raise ImagePreparationError("--retries must be non-negative")
        if args.timeout <= 0:
            raise ImagePreparationError("--timeout must be positive")
        filenames = load_pilot_image_filenames(args.pilot_dir, args.metadata)
        results = download_pilot_images(
            filenames,
            args.output_dir,
            retries=args.retries,
            timeout=args.timeout,
        )
    except (ImagePreparationError, ImageDownloadError, OSError, ValueError) as exc:
        print(f"download_images.py: error: {exc}", file=sys.stderr)
        return 2

    downloaded = sum(result == "downloaded" for result in results.values())
    skipped = sum(result == "skipped" for result in results.values())
    print(
        f"Validated {len(filenames)} COCO val2014 pilot images under "
        f"{args.output_dir / COCO_SPLIT} ({downloaded} downloaded, {skipped} already valid)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
