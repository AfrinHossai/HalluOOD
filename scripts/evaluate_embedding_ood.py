"""Evaluate embedding distances against entropy on the fixed POPE sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluood.embedding_evaluation import (
    EmbeddingEvaluationError,
    compare_embedding_and_entropy,
    join_paired_rows,
    render_comparison_risk_coverage_svg,
)
from halluood.embeddings import (
    embedding_distance_scores,
    normalize_coco_image_id,
    reference_set_identifier,
    select_disjoint_reference_images,
)
from halluood.model import TransformersVLMAdapter

DEFAULT_ENTROPY_RESULTS = Path(
    "data/results/pope_entropy_30_seed_2027_evaluated.jsonl"
)
DEFAULT_BASELINE_RESULTS = Path(
    "data/results/pope_baseline_smollm_500m_30_seed_2027.jsonl"
)
DEFAULT_BANK = Path("data/results/embedding_distance/coco_val2014_reference_embeddings.npz")
DEFAULT_MANIFEST = Path(
    "data/results/embedding_distance/coco_val2014_reference_manifest.json"
)
DEFAULT_IMAGE_DIR = Path("data/images/val2014")
DEFAULT_OUTPUT_DIR = Path("data/results/embedding_distance")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "halluood"
EXPECTED_COUNT = 30
EXPECTED_SAMPLE_SEED = 2027
_SCORE_TABLE_FIELDS = (
    "score",
    "question_count",
    "incorrect_count",
    "incorrect_prevalence",
    "auroc",
    "auprc_average_precision",
    "score_mean_correct",
    "score_median_correct",
    "score_mean_incorrect",
    "score_median_incorrect",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingEvaluationError(f"could not read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EmbeddingEvaluationError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EmbeddingEvaluationError(f"could not read JSONL file {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EmbeddingEvaluationError(
                f"invalid JSON in {path} at line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise EmbeddingEvaluationError(f"expected an object in {path} at line {line_number}")
        rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _verify_reference_bank(
    bank_path: Path,
    manifest: dict[str, Any],
    evaluation_image_ids: set[str],
    image_dir: Path,
) -> tuple[list[str], np.ndarray]:
    reference_ids = [
        normalize_coco_image_id(str(image_id))
        for image_id in manifest.get("reference_image_ids", [])
    ]
    if manifest.get("reference_count") != 50 or len(reference_ids) != 50:
        raise EmbeddingEvaluationError("the saved reference bank must contain exactly 50 images")
    if manifest.get("reference_selection", {}).get("random_seed") != EXPECTED_SAMPLE_SEED:
        raise EmbeddingEvaluationError(
            f"reference bank seed must be {EXPECTED_SAMPLE_SEED}"
        )
    if manifest.get("evaluation_image_ids_excluded") != sorted(evaluation_image_ids):
        raise EmbeddingEvaluationError(
            "reference manifest does not exclude exactly the current evaluation image IDs"
        )
    overlap = set(reference_ids).intersection(evaluation_image_ids)
    if overlap or manifest.get("reference_evaluation_overlap") != []:
        raise EmbeddingEvaluationError(
            f"reference bank overlaps evaluation images: {sorted(overlap)!r}"
        )
    if len(set(reference_ids)) != 50:
        raise EmbeddingEvaluationError("reference bank contains duplicate image IDs")
    local_pilot_images = sorted(image_dir.glob("COCO_val2014_*.jpg"))
    expected_references = select_disjoint_reference_images(
        local_pilot_images,
        evaluation_image_ids,
        sample_size=50,
        random_seed=EXPECTED_SAMPLE_SEED,
    )
    expected_reference_ids = [image_id for image_id, _ in expected_references]
    if reference_ids != expected_reference_ids:
        raise EmbeddingEvaluationError(
            "reference image IDs do not match deterministic PCG64 sampling from local images"
        )
    candidate_count = len(
        {
            normalize_coco_image_id(path.name)
            for path in local_pilot_images
            if normalize_coco_image_id(path.name) not in evaluation_image_ids
        }
    )
    if manifest.get("reference_selection", {}).get("candidate_count_after_exclusion") != (
        candidate_count
    ):
        raise EmbeddingEvaluationError(
            "reference candidate count differs from the recorded selection metadata"
        )

    try:
        expected_sha256 = manifest["feature_storage"]["sha256"]
        if _sha256_file(bank_path) != expected_sha256:
            raise EmbeddingEvaluationError(
                "reference embedding bank checksum does not match manifest"
            )
        with np.load(bank_path, allow_pickle=False) as stored:
            stored_ids = [str(image_id) for image_id in stored["image_ids"].tolist()]
            embeddings = np.asarray(stored["embeddings"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, EmbeddingEvaluationError):
            raise
        raise EmbeddingEvaluationError(f"could not load reference embedding bank: {exc}") from exc
    if stored_ids != reference_ids:
        raise EmbeddingEvaluationError("bank image ID order differs from its reference manifest")
    if embeddings.ndim != 2 or embeddings.shape != (50, 960):
        raise EmbeddingEvaluationError(
            f"unexpected reference embedding shape: {embeddings.shape!r}"
        )
    if not np.isfinite(embeddings).all():
        raise EmbeddingEvaluationError("reference bank contains non-finite embeddings")

    encoder = manifest.get("encoder", {})
    reference_id = reference_set_identifier(
        reference_ids,
        embeddings,
        encoder_identifier=str(encoder.get("model_identifier", "")),
        encoder_revision=str(encoder.get("model_revision", "")),
        random_seed=EXPECTED_SAMPLE_SEED,
    )
    if reference_id != manifest.get("reference_set_id"):
        raise EmbeddingEvaluationError("reference-set identifier does not match bank contents")

    expected_image_hashes = manifest.get("reference_image_sha256", {})
    for filename, image_id in zip(
        manifest.get("reference_filenames", []), reference_ids, strict=True
    ):
        if normalize_coco_image_id(Path(filename).name) != image_id:
            raise EmbeddingEvaluationError(
                f"reference filename does not match image ID {image_id}"
            )
        image_path = image_dir / Path(filename).name
        if not image_path.is_file():
            raise FileNotFoundError(f"reference COCO image is missing: {image_path}")
        if _sha256_file(image_path) != expected_image_hashes.get(image_id):
            raise EmbeddingEvaluationError(
                f"reference COCO image checksum changed for {image_id}"
            )
    return reference_ids, embeddings


def _verify_same_sample_settings(
    entropy_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> set[str]:
    if len(entropy_rows) != EXPECTED_COUNT or len(baseline_rows) != EXPECTED_COUNT:
        raise EmbeddingEvaluationError(
            "the comparison must use exactly 30 entropy and baseline rows"
        )
    for label, rows in (("entropy", entropy_rows), ("baseline", baseline_rows)):
        if any(row.get("seed") != EXPECTED_SAMPLE_SEED for row in rows):
            raise EmbeddingEvaluationError(
                f"{label} results do not use sample/generation seed {EXPECTED_SAMPLE_SEED}"
            )
        if any(row.get("generation_settings", {}).get("max_new_tokens") != 32 for row in rows):
            raise EmbeddingEvaluationError(f"{label} generation settings differ from baseline")
        if any(
            row.get("generation_settings", {}).get("do_sample") is not False
            or row.get("generation_settings", {}).get("num_beams") != 1
            for row in rows
        ):
            raise EmbeddingEvaluationError(
                f"{label} generation is not deterministic greedy decoding"
            )
    if any(row.get("reference_prediction_match") is not True for row in entropy_rows):
        raise EmbeddingEvaluationError(
            "the entropy results are not verified against baseline outputs"
        )
    return {
        normalize_coco_image_id(str(row["image_id"]))
        for row in entropy_rows
    }


def evaluate_embedding_ood(
    *,
    entropy_results_file: str | Path = DEFAULT_ENTROPY_RESULTS,
    baseline_results_file: str | Path = DEFAULT_BASELINE_RESULTS,
    bank_file: str | Path = DEFAULT_BANK,
    manifest_file: str | Path = DEFAULT_MANIFEST,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    device: str | None = None,
    adapter: TransformersVLMAdapter | None = None,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 2027,
) -> dict[str, Any]:
    """Score the exact entropy sample against its existing reference bank."""

    entropy_rows = _read_jsonl(Path(entropy_results_file))
    baseline_rows = _read_jsonl(Path(baseline_results_file))
    evaluation_ids = _verify_same_sample_settings(entropy_rows, baseline_rows)
    if len(evaluation_ids) != EXPECTED_COUNT:
        raise EmbeddingEvaluationError("the fixed evaluation set must contain 30 distinct images")

    manifest = _read_json(Path(manifest_file))
    local_image_dir = Path(image_dir)
    reference_ids, reference_embeddings = _verify_reference_bank(
        Path(bank_file),
        manifest,
        evaluation_ids,
        local_image_dir,
    )
    encoder_manifest = manifest.get("encoder", {})
    if adapter is None:
        adapter = TransformersVLMAdapter.from_pretrained(
            model_id=str(encoder_manifest["model_identifier"]),
            revision=str(encoder_manifest["model_revision"]),
            device=device or str(encoder_manifest["device"]),
            cache_dir=cache_dir,
        )
    current_encoder = adapter.vision_encoder_metadata
    if current_encoder != encoder_manifest:
        raise EmbeddingEvaluationError(
            "loaded encoder metadata differs from the encoder used to build the reference bank"
        )

    embedding_rows: list[dict[str, Any]] = []
    evaluation_vectors: list[np.ndarray] = []
    for index, entropy_row in enumerate(entropy_rows, start=1):
        image_id = normalize_coco_image_id(str(entropy_row["image_id"]))
        image_filename = Path(str(entropy_row["image_filename"])).name
        if normalize_coco_image_id(image_filename) != image_id:
            raise EmbeddingEvaluationError(
                f"image filename does not match image ID for question {entropy_row['question_id']}"
            )
        image_path = local_image_dir / image_filename
        if not image_path.is_file():
            raise FileNotFoundError(f"evaluation COCO image is missing: {image_path}")
        embedding = adapter.extract_image_embedding(image_path)
        if embedding.shape != (reference_embeddings.shape[1],):
            raise EmbeddingEvaluationError(
                f"feature dimension changed for evaluation image {image_id}"
            )
        score = embedding_distance_scores(
            embedding,
            reference_embeddings,
            reference_ids,
        )
        embedding_rows.append(
            {
                "question_id": entropy_row["question_id"],
                "image_id": image_id,
                "category": entropy_row["category"],
                "image_filename": image_filename,
                "reference_set_id": manifest["reference_set_id"],
                **score,
            }
        )
        evaluation_vectors.append(embedding.astype(np.float32, copy=False))
        if index % 5 == 0 or index == len(entropy_rows):
            print(f"Encoded {index}/{len(entropy_rows)} evaluation images.", flush=True)

    joined_rows = join_paired_rows(
        entropy_rows,
        baseline_rows,
        embedding_rows,
        expected_count=EXPECTED_COUNT,
    )
    comparison = compare_embedding_and_entropy(
        joined_rows,
        bootstrap_replicates=bootstrap_replicates,
        random_seed=bootstrap_seed,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = "pope_embedding_entropy_30_seed_2027"
    scores_path = output_path / f"{stem}_scores.jsonl"
    vectors_path = output_path / f"{stem}_evaluation_embeddings.npz"
    summary_path = output_path / f"{stem}_comparison.json"
    table_path = output_path / f"{stem}_comparison.csv"
    curve_path = output_path / f"{stem}_risk_coverage.svg"

    scores_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in joined_rows),
        encoding="utf-8",
    )
    np.savez_compressed(
        vectors_path,
        question_ids=np.asarray([str(row["question_id"]) for row in entropy_rows], dtype="U32"),
        image_ids=np.asarray(
            [normalize_coco_image_id(str(row["image_id"])) for row in entropy_rows],
            dtype="U12",
        ),
        embeddings=np.stack(evaluation_vectors).astype(np.float32, copy=False),
    )
    summary = {
        **comparison,
        "sample": {
            "question_count": len(entropy_rows),
            "distinct_image_count": len(evaluation_ids),
            "sampling_seed": EXPECTED_SAMPLE_SEED,
            "question_image_pairs_match_baseline": True,
            "baseline_results_file": str(baseline_results_file),
            "entropy_results_file": str(entropy_results_file),
        },
        "reference_bank": {
            "reference_set_id": manifest["reference_set_id"],
            "reference_count": len(reference_ids),
            "sampling_seed": manifest["reference_selection"]["random_seed"],
            "evaluation_overlap_count": len(set(reference_ids).intersection(evaluation_ids)),
            "bank_sha256": manifest["feature_storage"]["sha256"],
            "manifest_file": str(manifest_file),
        },
        "encoder": encoder_manifest,
        "score_definitions": manifest["distance_definitions"],
        "output_files": {
            "paired_scores_jsonl": str(scores_path),
            "evaluation_embeddings_npz": str(vectors_path),
            "comparison_json": str(summary_path),
            "comparison_csv": str(table_path),
            "risk_coverage_svg": str(curve_path),
        },
        "incremental_check_note": (
            "Rank fusion is an exploratory, label-independent diagnostic on this fixed sample, "
            "not a calibrated predictor or evidence of generalization."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with table_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_SCORE_TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(comparison["score_comparison_table"])
    curve_path.write_text(
        render_comparison_risk_coverage_svg(
            comparison["risk_coverage_curves"],
            overall_error_rate=comparison["incorrect_prevalence"],
            title="POPE error risk–coverage by uncertainty score (n=30)",
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "sample": summary["sample"],
                "reference_bank": summary["reference_bank"],
                "score_comparison_table": summary["score_comparison_table"],
                "embedding_entropy_spearman": summary["embedding_entropy_spearman"],
                "incremental_rank_fusion_check": summary["incremental_rank_fusion_check"],
                "output_files": summary["output_files"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare embedding distances with predictive entropy on the fixed 30-question "
            "POPE sample using the existing reference embedding bank."
        )
    )
    parser.add_argument("--entropy-results", type=Path, default=DEFAULT_ENTROPY_RESULTS)
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", choices=("mps", "cpu"), default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evaluate_embedding_ood(
            entropy_results_file=args.entropy_results,
            baseline_results_file=args.baseline_results,
            bank_file=args.bank,
            manifest_file=args.manifest,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            device=args.device,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    except Exception as exc:
        print(
            f"embedding comparison failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
