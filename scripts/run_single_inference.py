"""Run one reproducible POPE question with the pinned SmolVLM checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

# Allow direct execution from a source checkout without an editable install.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluood.data import PopeDataError, PopeRecord, load_pope_records
from halluood.model import (
    RECOMMENDED_MODEL_ID,
    RECOMMENDED_MODEL_REVISION,
    GenerationSettings,
    TransformersVLMAdapter,
    VLMAdapter,
)

DEFAULT_INPUT = Path("data/processed/coco_pope_random_pilot.jsonl")
DEFAULT_IMAGE_DIR = Path("data/images/val2014")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "halluood"
DEFAULT_QUESTION_ID = "49"


def _select_question(records: list[PopeRecord], question_id: str) -> PopeRecord:
    matches = [record for record in records if str(record.question_id) == question_id]
    if not matches:
        raise PopeDataError(f"question_id {question_id!r} was not found in the annotation file")
    if len(matches) != 1:
        raise PopeDataError(
            f"question_id {question_id!r} occurs {len(matches)} times; expected exactly one row"
        )
    return matches[0]


def _validate_image(image_path: Path) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"pilot image does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"pilot image is not readable: {image_path}: {exc}") from exc


def _synchronize_mps(device: str) -> None:
    """Wait for queued MPS work before stopping the wall-clock timer."""

    if device == "mps":
        import torch

        torch.mps.synchronize()


def run_single_inference(
    *,
    input_file: str | Path = DEFAULT_INPUT,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    question_id: str = DEFAULT_QUESTION_ID,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    device: str | None = None,
    generation: GenerationSettings | None = None,
    adapter: VLMAdapter | None = None,
) -> dict[str, Any]:
    """Run and print exactly one selected image-question pair.

    Supplying ``adapter`` is intended for unit tests; the CLI always constructs
    the real Transformers adapter and may download weights into ``cache_dir``.
    """

    records = load_pope_records(input_file)
    record = _select_question(records, question_id)
    image_path = Path(image_dir) / record.image
    _validate_image(image_path)

    settings = generation or GenerationSettings()
    if adapter is None:
        adapter = TransformersVLMAdapter.from_pretrained(
            model_id=RECOMMENDED_MODEL_ID,
            revision=RECOMMENDED_MODEL_REVISION,
            device=device,
            generation=settings,
            cache_dir=cache_dir,
        )

    _synchronize_mps(adapter.metadata.device)
    start_time = time.perf_counter()
    generated_response = adapter.generate_answer(image_path, record.text)
    _synchronize_mps(adapter.metadata.device)
    inference_time_seconds = time.perf_counter() - start_time

    result = {
        "question_id": record.question_id,
        "image": record.image,
        "question": record.text,
        "ground_truth_answer": record.label,
        "normalized_ground_truth_answer": record.normalized_label,
        "generated_response": generated_response,
        "model_identifier": adapter.metadata.model_id,
        "model_revision": (
            adapter.metadata.resolved_revision or adapter.metadata.requested_revision
        ),
        "device": adapter.metadata.device,
        "inference_time_seconds": inference_time_seconds,
        "generation_settings": adapter.metadata.to_dict()["generation"],
        "cache_dir": str(Path(cache_dir).expanduser()),
        "model_metadata": adapter.metadata.to_dict(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one POPE pilot image-question pair with SmolVLM."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--question-id",
        default=DEFAULT_QUESTION_ID,
        help="Unique POPE question_id to run (default: %(default)s).",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    generation = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        num_beams=1,
        do_sample=False,
    )
    try:
        run_single_inference(
            input_file=args.input,
            image_dir=args.image_dir,
            question_id=args.question_id,
            cache_dir=args.cache_dir,
            device=None if args.device == "auto" else args.device,
            generation=generation,
        )
    except Exception as exc:
        print(f"single inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
