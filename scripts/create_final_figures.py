"""Generate final HalluOOD comparison figures from local recorded outputs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from halluood.final_figures import generate_final_figures

DEFAULT_RESULTS = Path("data/results")
DEFAULT_OUTPUT = Path("data/results/final_figures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate_final_figures(
            args.results_dir,
            args.output_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Final figure generation failed: {exc}")
        return 2
    print(
        json.dumps(
            {
                "created": result["artifacts"],
                "pilot_image_count": result["analysis"]["method_comparison"]["pilot"][0][
                    "image_count"
                ],
                "expanded_question_count": result["analysis"]["expanded_clean"]["question_count"],
                "shift_image_count": result["analysis"]["distribution_shift"]["distinct_images"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
