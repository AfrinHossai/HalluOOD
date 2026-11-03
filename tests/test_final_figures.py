"""Tests for local-result figure analysis and SVG rendering."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from halluood.final_figures import (
    METHOD_LABELS,
    image_grouped_metric_intervals,
    image_grouped_risk_coverage_intervals,
    image_grouped_selective_intervals,
    render_method_comparison_svg,
    render_risk_coverage_svg,
    render_shift_severity_svg,
)


def _method(method: str, cohort: str = "30-question pilot") -> dict:
    return {
        "method": method,
        "label": METHOD_LABELS[method],
        "cohort": cohort,
        "cohort_label": "expanded (1,800 questions / 100 images)"
        if cohort == "expanded clean evaluation"
        else "pilot (30 questions / 30 images)",
        "question_count": 30,
        "image_count": 30,
        "error_count": 6,
        "error_prevalence": 0.2,
        "auroc": 0.7,
        "auprc_average_precision": 0.4,
        "confidence_intervals": {
            "auroc": {"lower": 0.55, "upper": 0.85},
            "auprc_average_precision": {"lower": 0.2, "upper": 0.65},
        },
    }


def _shift_condition(condition_id: str, kind: str | None, severity: int) -> dict:
    return {
        "condition_id": condition_id,
        "perturbation_type": kind,
        "severity": severity,
        "accuracy": 0.8 if severity == 0 else 0.8 - severity * 0.03,
        "mean_normalized_entropy": 0.16 if severity == 0 else 0.16 + severity * 0.02,
        "confidence_interval": {
            "accuracy": {"lower": 0.7, "upper": 0.9},
            "mean_normalized_entropy": {"lower": 0.1, "upper": 0.25},
        },
    }


def test_image_grouped_metric_intervals_are_seeded_and_group_questions() -> None:
    rows = [
        {
            "category": "random",
            "question_id": index,
            "image_id": image,
            "is_error": error,
            "score": score,
        }
        for index, (image, error, score) in enumerate(
            (
                ("im-a", True, 0.8),
                ("im-a", False, 0.1),
                ("im-b", False, 0.2),
                ("im-b", False, 0.3),
                ("im-c", True, 0.7),
                ("im-c", True, 0.9),
            )
        )
    ]
    first = image_grouped_metric_intervals(rows, "score", replicates=200, seed=73)
    second = image_grouped_metric_intervals(rows, "score", replicates=200, seed=73)

    assert first == second
    assert first["image_count"] == 3
    assert first["auroc"]["valid_replicates"] <= 200
    assert first["auroc"]["lower"] <= first["auroc"]["upper"]
    assert first["auprc_average_precision"]["lower"] <= first["auprc_average_precision"]["upper"]


def test_image_grouped_risk_coverage_intervals_are_seeded_and_tie_safe() -> None:
    rows = [
        {
            "category": "random",
            "question_id": index,
            "image_id": image,
            "is_error": error,
            "score": score,
        }
        for index, (image, error, score) in enumerate(
            (
                ("im-a", True, 0.8),
                ("im-a", False, 0.1),
                ("im-b", False, 0.2),
                ("im-b", False, 0.3),
                ("im-c", True, 0.7),
                ("im-c", True, 0.9),
            )
        )
    ]
    first = image_grouped_risk_coverage_intervals(
        rows, "score", targets=(0.5, 1.0), replicates=200, seed=42
    )
    second = image_grouped_risk_coverage_intervals(
        rows, "score", targets=(0.5, 1.0), replicates=200, seed=42
    )

    assert first == second
    assert first["image_count"] == 3
    assert first["tie_policy"].startswith("fixed empirical score thresholds")
    for point in first["points"]:
        assert point["confidence_intervals"]["coverage"]["lower"] <= point["observed"]["coverage"]
        assert point["confidence_intervals"]["risk"]["lower"] <= point["observed"]["risk"]


def test_image_grouped_selective_intervals_keep_decisions_frozen() -> None:
    outcomes = [
        {"image_id": image, "decision": decision, "is_error": error}
        for image, decision, error in (
            ("im-a", "answer", False),
            ("im-a", "abstain", True),
            ("im-b", "answer", True),
            ("im-b", "answer", False),
            ("im-c", "abstain", False),
            ("im-c", "answer", False),
        )
    ]
    first = image_grouped_selective_intervals(outcomes, replicates=300, seed=101)
    second = image_grouped_selective_intervals(outcomes, replicates=300, seed=101)

    assert first == second
    assert first["image_count"] == 3
    assert first["method"].endswith("frozen_decisions")
    assert first["coverage"]["lower"] <= first["coverage"]["upper"]
    assert first["selective_risk"]["lower"] <= first["selective_risk"]["upper"]


def test_method_comparison_svg_is_well_formed_and_separates_cohorts() -> None:
    pilot = [
        _method(method)
        for method in (
            "predictive_entropy",
            "cosine_distance",
            "nearest_neighbor_distance",
            "augmentation_consistency",
            "answer_disagreement",
        )
    ]
    expanded = _method("predictive_entropy", "expanded clean evaluation")
    analysis = {
        "method_comparison": {
            "pilot": pilot,
            "expanded_predictive_entropy": expanded,
        }
    }

    svg = render_method_comparison_svg(analysis)
    root = ET.fromstring(svg)

    assert root.tag.endswith("svg")
    assert "pilot · 30 questions / 30 images" in svg
    assert "expanded (1,800 questions / 100 images)" in svg
    assert "AUPRC (non-interpolated average precision)" in svg
    assert "whiskers" in svg


def test_shift_severity_svgs_show_image_counts_and_confidence_intervals() -> None:
    kinds = (
        "gaussian_blur",
        "brightness_reduction",
        "gaussian_noise",
        "contrast_reduction",
    )
    conditions = [_shift_condition("clean", None, 0)]
    for kind in kinds:
        for severity in (1, 2, 3):
            conditions.append(_shift_condition(f"{kind}_severity_{severity}", kind, severity))
    analysis = {
        "distribution_shift": {
            "conditions": conditions,
            "paired_questions": 54,
            "distinct_images": 3,
            "full_clean_reference": {
                "question_count": 1800,
                "image_count": 100,
                "accuracy": 0.84,
                "mean_normalized_entropy": 0.16,
                "confidence_intervals": {
                    "accuracy": {"lower": 0.82, "upper": 0.86},
                    "mean_normalized_entropy": {"lower": 0.14, "upper": 0.18},
                },
            },
        }
    }

    accuracy_svg = render_shift_severity_svg(analysis, "accuracy")
    entropy_svg = render_shift_severity_svg(analysis, "mean_normalized_entropy")

    assert ET.fromstring(accuracy_svg).tag.endswith("svg")
    assert ET.fromstring(entropy_svg).tag.endswith("svg")
    assert "54 question rows from 3 images" in accuracy_svg
    assert "full clean · 100 images" in entropy_svg
    assert "95% bootstrap intervals" in accuracy_svg


def test_risk_coverage_svg_shows_calibrated_points_and_split_sizes() -> None:
    curve = [
        {"coverage": 0.0, "risk": None},
        {"coverage": 0.5, "risk": 0.05},
        {"coverage": 1.0, "risk": 0.2},
    ]
    thresholds = []
    for index, target in enumerate((0.5, 0.7, 0.8, 0.9)):
        thresholds.append(
            {
                "evaluation": {
                    "coverage": target,
                    "selective_risk": 0.1,
                    "target_coverage": target,
                },
                "evaluation_confidence_intervals": {
                    "intervals": {
                        "coverage": {"lower": target - 0.05, "upper": target + 0.05},
                        "selective_risk": {"lower": 0.05, "upper": 0.15},
                    }
                },
            }
        )
    bootstrap = {
        "points": [
            {
                "observed": {"coverage": 0.5, "risk": 0.05},
                "confidence_intervals": {
                    "coverage": {"lower": 0.4, "upper": 0.6},
                    "risk": {"lower": 0.02, "upper": 0.12},
                },
            }
        ]
    }
    analysis = {
        "risk_coverage": {
            "pilot_entropy": curve,
            "expanded_entropy": curve,
            "heldout_entropy": curve,
            "pilot_bootstrap": bootstrap,
            "expanded_bootstrap": bootstrap,
            "pilot_abstention_thresholds": thresholds,
            "abstention_thresholds": thresholds,
        }
    }

    svg = render_risk_coverage_svg(analysis)

    assert ET.fromstring(svg).tag.endswith("svg")
    assert "Predictive entropy ranking" in svg
    assert "Calibrated abstention on held-out images" in svg
    assert "P50%" in svg and "E50%" in svg
    assert "P90%" in svg and "E90%" in svg
    assert "Calibration: 180 / 10 images" in svg
    assert "P/E markers: pilot/expanded abstention" in svg
