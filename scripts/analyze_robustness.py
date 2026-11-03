"""Summarize full POPE entropy results with cluster intervals and held-out abstention."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluood.embeddings import normalize_coco_image_id
from halluood.robustness import (
    CATEGORIES,
    deterministic_image_split,
    entropy_risk_coverage,
    image_grouped_bootstrap,
    image_grouped_selective_bootstrap,
    render_metric_bar_svg,
    render_risk_coverage_svg,
    summarize_expanded_rows,
)
from halluood.selective import calibrate_abstention_threshold, evaluate_selective_predictions

DEFAULT_FULL = Path("data/results/robustness/full_pope_predictions_seed_2027.jsonl")
DEFAULT_PILOT = Path("data/results/pope_entropy_30_seed_2027_evaluated.jsonl")
DEFAULT_REFERENCE = Path("data/results/embedding_distance/coco_val2014_reference_manifest.json")
DEFAULT_OUTPUT_DIR = Path("data/results/robustness")


class AnalysisError(ValueError):
    """Raised when source outputs or provenance cannot be compared safely."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"{path}: invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise AnalysisError(f"{path}: line {line_number} is not a JSON object")
        rows.append(row)
    return rows


def _identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["category"]), str(row["question_id"]), str(row["image_id"]))


def _validate_full(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 1800:
        raise AnalysisError(f"expected 1,800 full rows; received {len(rows)}")
    identities = [_identity(row) for row in rows]
    if len(set(identities)) != len(identities):
        raise AnalysisError("full results contain duplicate category/question/image identities")
    image_ids = {identity[2] for identity in identities}
    if len(image_ids) != 100:
        raise AnalysisError(f"expected 100 images; received {len(image_ids)}")
    for category in CATEGORIES:
        category_rows = [row for row in rows if row["category"] == category]
        if len(category_rows) != 600 or len({str(row["image_id"]) for row in category_rows}) != 100:
            raise AnalysisError(f"{category} must contain 600 rows across the same 100 images")
        if any(row.get("normalized_entropy") is None for row in category_rows):
            raise AnalysisError(f"{category} has incomplete binary entropy scores")
        if any(not isinstance(row.get("is_correct"), bool) for row in category_rows):
            raise AnalysisError(f"{category} has missing correctness labels")


def _compare_pilot(
    full_rows: list[dict[str, Any]], pilot_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    full_by_id = {_identity(row): row for row in full_rows}
    pilot_by_id = {_identity(row): row for row in pilot_rows}
    if len(pilot_by_id) != 30:
        raise AnalysisError(f"expected the original 30-row pilot; received {len(pilot_by_id)}")
    differences = []
    for key, pilot in pilot_by_id.items():
        full = full_by_id.get(key)
        if full is None:
            raise AnalysisError(f"original pilot identity is missing from full run: {key}")
        for field in (
            "question",
            "ground_truth_label",
            "normalized_ground_truth_label",
            "raw_prediction",
            "normalized_prediction",
            "prediction_status",
            "is_correct",
        ):
            if full.get(field) != pilot.get(field):
                differences.append({"identity": key, "field": field})
        pilot_entropy = pilot.get("normalized_entropy")
        if pilot_entropy is None and isinstance(pilot.get("uncertainty_score"), dict):
            pilot_entropy = pilot["uncertainty_score"].get("normalized_entropy")
        if (
            pilot_entropy is None
            or abs(float(full["normalized_entropy"]) - float(pilot_entropy)) > 1e-7
        ):
            differences.append({"identity": key, "field": "normalized_entropy"})
    if differences:
        raise AnalysisError(f"full run does not reproduce original pilot rows: {differences[:5]}")
    full_summary = summarize_expanded_rows(full_rows)
    pilot_summary = summarize_expanded_rows(pilot_rows)
    metrics = (
        "accuracy",
        "ambiguous_response_rate",
        "mean_normalized_entropy",
        "auroc",
        "auprc_average_precision",
    )
    return {
        "original_pilot_question_count": len(pilot_rows),
        "original_pilot_image_count": len({row["image_id"] for row in pilot_rows}),
        "pilot_rows_exactly_reproduced_in_full_run": True,
        "comparison_is_nested_not_independent": True,
        "overall": {
            metric: {
                "pilot_30": pilot_summary["overall"][metric],
                "expanded_1800": full_summary["overall"][metric],
                "expanded_minus_pilot": full_summary["overall"][metric]
                - pilot_summary["overall"][metric],
            }
            for metric in metrics
        },
        "by_category": {
            category: {
                metric: {
                    "pilot_30": pilot_summary["by_category"][category][metric],
                    "expanded_1800": full_summary["by_category"][category][metric],
                    "expanded_minus_pilot": full_summary["by_category"][category][metric]
                    - pilot_summary["by_category"][category][metric],
                }
                for metric in metrics
            }
            for category in CATEGORIES
        },
        "interpretation_note": (
            "The 30 rows are a subset of the 1,800; differences are descriptive, "
            "not independent-sample tests."
        ),
    }


def analyze_robustness(
    *,
    full_file: str | Path = DEFAULT_FULL,
    pilot_file: str | Path = DEFAULT_PILOT,
    reference_manifest: str | Path = DEFAULT_REFERENCE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seed: int = 2027,
    bootstrap_replicates: int = 1000,
    target_coverages: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9),
) -> dict[str, Any]:
    full_path = Path(full_file)
    pilot_path = Path(pilot_file)
    reference_path = Path(reference_manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    full_rows = _read_jsonl(full_path)
    pilot_rows = _read_jsonl(pilot_path)
    _validate_full(full_rows)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_ids = {
        normalize_coco_image_id(str(image_id)) for image_id in reference["reference_image_ids"]
    }
    image_ids = {str(row["image_id"]) for row in full_rows}
    reference_overlap = image_ids & reference_ids
    pilot_image_ids = {str(row["image_id"]) for row in pilot_rows}
    pilot_reference_overlap = pilot_image_ids & reference_ids
    if pilot_reference_overlap:
        raise AnalysisError(
            "the original 30-question pilot overlaps the reference bank: "
            f"{sorted(pilot_reference_overlap)[:5]}"
        )

    full_summary = summarize_expanded_rows(full_rows)
    bootstrap = image_grouped_bootstrap(full_rows, replicates=bootstrap_replicates, seed=seed)
    for scope in ("overall", *CATEGORIES):
        target = (
            full_summary["overall"] if scope == "overall" else full_summary["by_category"][scope]
        )
        target["confidence_interval"] = bootstrap["intervals"][scope]
    full_summary["image_grouped_bootstrap"] = {
        key: value for key, value in bootstrap.items() if key != "intervals"
    }
    full_summary["method_scale"] = {
        "expanded_to_1800": ["deterministic answer prediction", "binary predictive entropy"],
        "held_out_calibration_and_evaluation": ["entropy-based calibrated abstention"],
        "prior_30_question_pilot_only": [
            "embedding distance",
            "augmentation consistency",
            "sampled answer disagreement",
            "the earlier 30-question controlled-shift run",
        ],
        "not_expanded_here": ["other uncertainty methods and distribution-shift scores"],
    }
    pilot_comparison = _compare_pilot(full_rows, pilot_rows)

    eligible_image_ids = sorted(image_ids - reference_ids)
    if len(eligible_image_ids) < 2:
        raise AnalysisError("fewer than two POPE images remain after excluding reference images")
    calibration_ids, evaluation_ids = deterministic_image_split(
        eligible_image_ids, calibration_fraction=0.2, seed=seed
    )
    split_overlap = set(calibration_ids) & set(evaluation_ids)
    if split_overlap or (set(calibration_ids) | set(evaluation_ids)) != set(eligible_image_ids):
        raise AnalysisError("image-group split is incomplete or overlapping")
    if (set(calibration_ids) | set(evaluation_ids)) & reference_ids:
        raise AnalysisError("calibration/evaluation split overlaps the embedding reference bank")
    calibration_dataset_id = f"pope-pilot-calibration-{len(calibration_ids)}-images-seed-{seed}"
    evaluation_dataset_id = (
        f"pope-pilot-heldout-evaluation-{len(evaluation_ids)}-images-seed-{seed}"
    )
    calibration_id_set = set(calibration_ids)
    evaluation_id_set = set(evaluation_ids)
    calibration_rows = [
        {
            "question_id": row["question_id"],
            "image_id": row["image_id"],
            "category": row["category"],
            "normalized_entropy": row["normalized_entropy"],
        }
        for row in full_rows
        if str(row["image_id"]) in calibration_id_set
    ]
    evaluation_rows = [row for row in full_rows if str(row["image_id"]) in evaluation_id_set]
    selective_results = []
    for coverage in target_coverages:
        calibration = calibrate_abstention_threshold(
            calibration_rows,
            evaluation_image_ids=evaluation_ids,
            target_coverage=coverage,
            calibration_dataset_id=calibration_dataset_id,
            evaluation_dataset_id=evaluation_dataset_id,
            calibration_seed=seed,
        )
        metrics = evaluate_selective_predictions(
            evaluation_rows,
            calibration,
            evaluation_dataset_id=evaluation_dataset_id,
        )
        selective_bootstrap = image_grouped_selective_bootstrap(
            evaluation_rows,
            calibration.threshold,
            replicates=bootstrap_replicates,
            seed=seed,
        )
        selective_results.append(
            {
                "calibration": calibration.to_dict(),
                "evaluation": {key: value for key, value in metrics.items() if key != "outcomes"},
                "evaluation_outcomes": metrics["outcomes"],
                "evaluation_confidence_intervals": selective_bootstrap,
            }
        )

    risk_curve = entropy_risk_coverage(full_rows)
    shift_summary_path = out / f"expanded_shift_summary_seed_{seed}.json"
    if shift_summary_path.is_file():
        shift_summary = json.loads(shift_summary_path.read_text(encoding="utf-8"))
        distribution_shift_summary = {
            "status": "completed",
            "selected_evaluation_image_ids": shift_summary["selected_evaluation_image_ids"],
            "image_count": shift_summary["image_count"],
            "paired_clean_question_count": shift_summary["paired_image_question_count"],
            "shifted_evaluation_count": shift_summary["shifted_inference_count"],
            "condition_count": shift_summary["condition_count"],
            "subset_selection_seed": shift_summary["subset_selection_seed"],
            "results_file": shift_summary["raw_results_file"],
            "summary_file": str(shift_summary_path),
        }
    else:
        distribution_shift_summary = {
            "status": "pending_shift_run",
            "planned_fixed_image_count": 3,
            "planned_questions_per_image": 18,
            "planned_pairs": 54,
            "conditions": 12,
            "planned_shifted_inferences": 648,
            "subset_selection_seed": seed + 1,
            "eligible_images": (
                f"the {len(evaluation_ids)} held-out evaluation images, disjoint from "
                "calibration and reference images"
            ),
        }
    summary = {
        "experiment": "HalluOOD expanded clean POPE and held-out abstention analysis",
        "seed": seed,
        "full_clean_results_file": str(full_path),
        "original_30_question_pilot_file": str(pilot_path),
        "reference_manifest_file": str(reference_path),
        "reference_set_id": reference.get("reference_set_id"),
        "reference_image_count": len(reference_ids),
        "reference_evaluation_image_overlap_count": len(reference_overlap),
        "reference_original_pilot_overlap_count": len(pilot_reference_overlap),
        "reference_overlap_note": (
            "The full 100-image entropy-only summary contains reference-bank image IDs, "
            "but does not load or use reference embeddings. The original 30-row comparison, "
            "abstention split, and expanded shift evaluation are disjoint from the reference bank."
        ),
        "full_clean_metrics": full_summary,
        "full_clean_risk_coverage_curve": risk_curve,
        "original_pilot_comparison": pilot_comparison,
        "abstention": {
            "split_method": (
                "Exclude all reference image IDs, then apply a PCG64 permutation to "
                "the sorted remaining POPE image IDs; first 20% calibrate and the "
                "remainder is held out"
            ),
            "calibration_image_ids": calibration_ids,
            "evaluation_image_ids": evaluation_ids,
            "image_overlap_count": len(split_overlap),
            "reference_overlap_count": 0,
            "calibration_questions": len(calibration_rows),
            "evaluation_questions": len(evaluation_rows),
            "thresholds": selective_results,
            "selection_note": (
                "Each threshold uses only calibration entropy and the held-out image IDs; "
                "held-out labels and scores are not passed to threshold selection."
            ),
        },
        "distribution_shift": distribution_shift_summary,
    }
    for metric, filename, label in (
        ("accuracy", "accuracy_by_category.svg", "Accuracy"),
        ("ambiguous_response_rate", "ambiguous_rate_by_category.svg", "Ambiguous response rate"),
        ("auroc", "auroc_by_category.svg", "AUROC for detecting incorrect answers"),
        (
            "auprc_average_precision",
            "auprc_by_category.svg",
            "AUPRC for detecting incorrect answers",
        ),
    ):
        plotted = {
            category.title(): full_summary["by_category"][category] for category in CATEGORIES
        }
        plotted["Overall"] = full_summary["overall"]
        (out / filename).write_text(
            render_metric_bar_svg(plotted, metric=metric, title=label, y_label=label),
            encoding="utf-8",
        )
    (out / "risk_coverage.svg").write_text(
        render_risk_coverage_svg(risk_curve, title="Full POPE: entropy risk vs. coverage"),
        encoding="utf-8",
    )
    summary_path = out / "expanded_robustness_summary_seed_2027.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bootstrap_path = out / "expanded_image_bootstrap_ci_seed_2027.json"
    bootstrap_path.write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    split_path = out / "expanded_image_split_seed_2027.json"
    split_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "reference_image_ids": sorted(reference_ids),
                "calibration_image_ids": calibration_ids,
                "evaluation_image_ids": evaluation_ids,
                "calibration_evaluation_overlap_count": 0,
                "reference_calibration_overlap_count": 0,
                "reference_evaluation_overlap_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-results", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--pilot-results", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--reference-manifest", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--target-coverages", default="0.5,0.7,0.8,0.9")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        coverages = tuple(float(value) for value in args.target_coverages.split(","))
        analyze_robustness(
            full_file=args.full_results,
            pilot_file=args.pilot_results,
            reference_manifest=args.reference_manifest,
            output_dir=args.output_dir,
            seed=args.seed,
            bootstrap_replicates=args.bootstrap_replicates,
            target_coverages=coverages,
        )
    except Exception as exc:
        print(f"robustness analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
