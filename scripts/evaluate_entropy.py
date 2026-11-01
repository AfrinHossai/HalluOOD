"""Summarize binary entropy as an answer-error risk score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluood.evaluation import EvaluationError, evaluate_entropy_rows, render_risk_coverage_svg

DEFAULT_INPUT = Path("data/results/pope_entropy_30_seed_2027.jsonl")
DEFAULT_REFERENCE = Path("data/results/pope_baseline_smollm_500m_30_seed_2027.jsonl")
DEFAULT_OUTPUT = Path("data/results/pope_entropy_30_seed_2027_evaluated.jsonl")
DEFAULT_SUMMARY = Path("data/results/pope_entropy_30_seed_2027_evaluation.json")
DEFAULT_CURVE = Path("data/results/pope_entropy_30_seed_2027_risk_coverage.svg")
_COMPARISON_FIELDS = (
    "model_identifier",
    "model_revision",
    "device",
    "precision",
    "torch_version",
    "transformers_version",
    "seed",
    "generation_settings",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationError(f"could not read results file {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise EvaluationError(f"expected a JSON object on {path}:{line_number}")
        rows.append(row)
    return rows


def _identity(row: dict[str, Any]) -> tuple[str, str, str]:
    try:
        return (str(row["category"]), str(row["question_id"]), str(row["image_id"]))
    except KeyError as exc:
        raise EvaluationError(f"result row lacks identity field {exc.args[0]!r}") from exc


def _ensure_comparable(
    evaluation_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(evaluation_rows) != expected_count or len(reference_rows) != expected_count:
        raise EvaluationError(
            f"expected {expected_count} entropy rows and reference rows; got "
            f"{len(evaluation_rows)} and {len(reference_rows)}"
        )
    eval_keys = [_identity(row) for row in evaluation_rows]
    ref_keys = [_identity(row) for row in reference_rows]
    if len(set(eval_keys)) != len(eval_keys) or len(set(ref_keys)) != len(ref_keys):
        raise EvaluationError("results contain duplicate category/question/image identities")
    if set(eval_keys) != set(ref_keys):
        missing = sorted(set(ref_keys) - set(eval_keys))
        unexpected = sorted(set(eval_keys) - set(ref_keys))
        raise EvaluationError(
            f"entropy run does not match the reference question sample; "
            f"missing={missing[:3]!r}, unexpected={unexpected[:3]!r}"
        )

    reference_by_key = {_identity(row): row for row in reference_rows}
    for field in _COMPARISON_FIELDS:
        reference_values = {json.dumps(row.get(field), sort_keys=True) for row in reference_rows}
        evaluation_values = {json.dumps(row.get(field), sort_keys=True) for row in evaluation_rows}
        if len(reference_values) != 1 or len(evaluation_values) != 1:
            raise EvaluationError(f"{field} varies within a result file")
        if reference_values != evaluation_values:
            raise EvaluationError(
                f"the entropy run's {field} does not match the reference baseline"
            )

    enriched: list[dict[str, Any]] = []
    prediction_agreements = 0
    for row in evaluation_rows:
        reference = reference_by_key[_identity(row)]
        agrees = all(
            row.get(field) == reference.get(field)
            for field in ("normalized_prediction", "prediction_status", "raw_prediction")
        )
        prediction_agreements += int(agrees)
        updated = dict(row)
        updated["reference_prediction_match"] = agrees
        enriched.append(updated)

    return enriched, {
        "same_question_set": True,
        "question_count": expected_count,
        "prediction_agreement_count": prediction_agreements,
        "prediction_disagreement_count": expected_count - prediction_agreements,
    }


def evaluate_files(
    *,
    input_file: str | Path = DEFAULT_INPUT,
    reference_file: str | Path = DEFAULT_REFERENCE,
    output_file: str | Path = DEFAULT_OUTPUT,
    summary_file: str | Path = DEFAULT_SUMMARY,
    curve_file: str | Path = DEFAULT_CURVE,
    expected_count: int = 30,
) -> dict[str, Any]:
    """Validate the fixed baseline sample, then save rows, metrics, and SVG curve."""

    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise ValueError("expected_count must be a positive integer")
    entropy_rows = _read_jsonl(Path(input_file))
    baseline_rows = _read_jsonl(Path(reference_file))
    matched_rows, reference_comparison = _ensure_comparable(
        entropy_rows,
        baseline_rows,
        expected_count,
    )
    evaluated_rows, metrics = evaluate_entropy_rows(matched_rows)
    model_metadata = {
        field: evaluated_rows[0].get(field)
        for field in _COMPARISON_FIELDS
    }
    summary = {
        **metrics,
        "reference_comparison": reference_comparison,
        "reference_results_file": str(reference_file),
        "input_results_file": str(input_file),
        "evaluated_results_file": str(output_file),
        "risk_curve_file": str(curve_file),
        "model_and_inference_settings": model_metadata,
        "score_definition": (
            "Normalized entropy of the Yes/No candidate-sequence distribution; "
            "higher scores are treated as higher risk of any incorrect or ambiguous answer."
        ),
    }

    output_path = Path(output_file)
    summary_path = Path(summary_file)
    curve_path = Path(curve_file)
    for path in (output_path, summary_path, curve_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in evaluated_rows
        ),
        encoding="utf-8",
    )
    summary["summary_file"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    curve_path.write_text(
        render_risk_coverage_svg(
            metrics["risk_coverage_curve"],
            overall_error_rate=metrics["error_rate"],
            title=f"POPE answer error risk vs. coverage (n={metrics['question_count']})",
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate binary predictive entropy on an existing POPE result JSONL."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--curve", type=Path, default=DEFAULT_CURVE)
    parser.add_argument("--expected-count", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evaluate_files(
            input_file=args.input,
            reference_file=args.reference,
            output_file=args.output,
            summary_file=args.summary,
            curve_file=args.curve,
            expected_count=args.expected_count,
        )
    except Exception as exc:
        print(f"entropy evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
