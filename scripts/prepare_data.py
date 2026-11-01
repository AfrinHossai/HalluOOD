"""Validate POPE annotations and optionally sample whole image groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from halluood.data import load_pope_records, sample_pilot_subset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and prepare a local POPE annotation file."
    )
    parser.add_argument("--input", required=True, type=Path, help="input JSON or JSONL file")
    parser.add_argument("--output", required=True, type=Path, help="output JSONL file")
    parser.add_argument(
        "--sample-size",
        type=int,
        help="number of distinct image groups to sample; all questions for each image are kept",
    )
    parser.add_argument(
        "--random-seed",
        "--seed",
        dest="random_seed",
        type=int,
        default=2027,
        help="non-negative PCG64 seed used when --sample-size is set (default: 2027)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")

    try:
        records = load_pope_records(args.input)
        if args.sample_size is None:
            prepared = records
        else:
            prepared = sample_pilot_subset(records, args.sample_size, args.random_seed)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
            for record in prepared:
                output_file.write(
                    json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                )
    except (OSError, ValueError) as exc:
        print(f"prepare_data.py: error: {exc}", file=sys.stderr)
        return 2

    image_count = len({record.image for record in prepared})
    print(f"Wrote {len(prepared)} questions from {image_count} image groups to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
