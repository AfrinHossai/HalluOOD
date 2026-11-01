"""Build a disjoint COCO reference bank and score one evaluation image."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halluood.embeddings import (
    EmbeddingError,
    embedding_distance_scores,
    normalize_coco_image_id,
    reference_set_identifier,
    select_disjoint_reference_images,
)
from halluood.model import (
    RECOMMENDED_MODEL_ID,
    RECOMMENDED_MODEL_REVISION,
    TransformersVLMAdapter,
)

DEFAULT_IMAGE_DIR = Path("data/images/val2014")
DEFAULT_PILOT_METADATA = Path("data/processed/pope_pilot_metadata.json")
DEFAULT_EVALUATION_RESULTS = Path("data/results/pope_entropy_30_seed_2027_evaluated.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/results/embedding_distance")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "halluood"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EmbeddingError(f"invalid JSON in {path} at line {line_number}") from exc
        if not isinstance(row, dict):
            raise EmbeddingError(f"expected an object in {path} at line {line_number}")
        rows.append(row)
    if not rows:
        raise EmbeddingError(f"evaluation results contain no rows: {path}")
    return rows


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def build_reference_bank_and_smoke_score(
    *,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    pilot_metadata_file: str | Path = DEFAULT_PILOT_METADATA,
    evaluation_results_file: str | Path = DEFAULT_EVALUATION_RESULTS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    reference_count: int = 50,
    random_seed: int = 2027,
    score_image_id: str | None = None,
    device: str | None = None,
    adapter: TransformersVLMAdapter | None = None,
) -> dict[str, Any]:
    """Build a deterministic disjoint bank and score exactly one pilot image."""

    if isinstance(reference_count, bool) or not isinstance(reference_count, int):
        raise ValueError("reference_count must be a positive integer")
    if reference_count < 1:
        raise ValueError("reference_count must be a positive integer")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")

    metadata_path = Path(pilot_metadata_file)
    try:
        pilot_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pilot_filenames = pilot_metadata["sampling"]["selected_image_ids_in_sampling_order"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise EmbeddingError(f"invalid POPE pilot metadata at {metadata_path}: {exc}") from exc
    if not isinstance(pilot_filenames, list) or not pilot_filenames:
        raise EmbeddingError("pilot metadata has no selected image identifiers")

    local_image_dir = Path(image_dir)
    pilot_paths: list[Path] = []
    pilot_image_ids: set[str] = set()
    for filename in pilot_filenames:
        image_id = normalize_coco_image_id(str(filename))
        if image_id in pilot_image_ids:
            raise EmbeddingError(f"duplicate image identifier in pilot metadata: {image_id}")
        pilot_image_ids.add(image_id)
        path = local_image_dir / Path(str(filename)).name
        if not path.is_file():
            raise FileNotFoundError(f"pilot COCO image is missing: {path}")
        pilot_paths.append(path)

    evaluation_path = Path(evaluation_results_file)
    evaluation_rows = _read_jsonl(evaluation_path)
    evaluation_ids = {
        normalize_coco_image_id(str(row["image_id"]))
        for row in evaluation_rows
        if "image_id" in row
    }
    if not evaluation_ids or len(evaluation_ids) != len(
        {str(row.get("image_id")) for row in evaluation_rows}
    ):
        raise EmbeddingError("evaluation results must provide image_id on every row")
    unexpected_ids = evaluation_ids - pilot_image_ids
    if unexpected_ids:
        raise EmbeddingError(
            "evaluation images are outside the local pilot image set: "
            + ", ".join(sorted(unexpected_ids)[:5])
        )
    chosen_evaluation_id = (
        normalize_coco_image_id(score_image_id)
        if score_image_id is not None
        else normalize_coco_image_id(str(evaluation_rows[0]["image_id"]))
    )
    if chosen_evaluation_id not in evaluation_ids:
        raise EmbeddingError(
            f"smoke-test image {chosen_evaluation_id} is not in the evaluation image set"
        )

    selected_reference = select_disjoint_reference_images(
        pilot_paths,
        evaluation_ids,
        sample_size=reference_count,
        random_seed=random_seed,
    )
    reference_ids = [image_id for image_id, _ in selected_reference]
    if set(reference_ids).intersection(evaluation_ids):
        raise EmbeddingError("selected reference bank overlaps evaluation image IDs")

    if adapter is None:
        adapter = TransformersVLMAdapter.from_pretrained(
            model_id=RECOMMENDED_MODEL_ID,
            revision=RECOMMENDED_MODEL_REVISION,
            device=device,
            cache_dir=cache_dir,
        )
    encoder_metadata = adapter.vision_encoder_metadata
    print(
        f"Encoding {len(selected_reference)} disjoint reference images "
        f"with {encoder_metadata['model_identifier']} on {encoder_metadata['device']}...",
        flush=True,
    )
    embedding_rows: list[np.ndarray] = []
    for index, (_, path) in enumerate(selected_reference, start=1):
        embedding = adapter.extract_image_embedding(path)
        if embedding.ndim != 1 or not np.isfinite(embedding).all():
            raise EmbeddingError(f"encoder returned an invalid embedding for {path}")
        if embedding_rows and embedding.shape != embedding_rows[0].shape:
            raise EmbeddingError("encoder returned inconsistent feature dimensions")
        embedding_rows.append(embedding.astype(np.float32, copy=False))
        if index % 10 == 0 or index == len(selected_reference):
            print(f"Encoded {index}/{len(selected_reference)} reference images.", flush=True)

    reference_embeddings = np.stack(embedding_rows)
    reference_revision = str(encoder_metadata["model_revision"])
    reference_id = reference_set_identifier(
        reference_ids,
        reference_embeddings,
        encoder_identifier=str(encoder_metadata["model_identifier"]),
        encoder_revision=reference_revision,
        random_seed=random_seed,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bank_path = output_path / "coco_val2014_reference_embeddings.npz"
    manifest_path = output_path / "coco_val2014_reference_manifest.json"
    score_path = output_path / "embedding_ood_smoke_scores.jsonl"
    np.savez_compressed(
        bank_path,
        image_ids=np.asarray(reference_ids, dtype="U12"),
        embeddings=reference_embeddings,
    )
    image_hashes = {
        image_id: _sha256_file(path) for image_id, path in selected_reference
    }
    manifest = {
        "schema_version": 1,
        "dataset": "MS COCO 2014 validation pilot images",
        "pilot_metadata_file": str(metadata_path),
        "evaluation_results_file": str(evaluation_path),
        "pilot_image_count": len(pilot_image_ids),
        "evaluation_image_count": len(evaluation_ids),
        "evaluation_image_ids_excluded": sorted(evaluation_ids),
        "reference_set_id": reference_id,
        "reference_count": len(reference_ids),
        "reference_image_ids": reference_ids,
        "reference_filenames": [path.name for _, path in selected_reference],
        "reference_image_sha256": image_hashes,
        "reference_selection": {
            "method": "sample without replacement from sorted local pilot images "
            "after excluding every evaluation image ID",
            "random_generator": "NumPy PCG64",
            "random_seed": random_seed,
            "candidate_count_after_exclusion": len(pilot_image_ids - evaluation_ids),
        },
        "reference_evaluation_overlap": sorted(set(reference_ids).intersection(evaluation_ids)),
        "encoder": encoder_metadata,
        "feature_storage": {
            "file": str(bank_path),
            "sha256": _sha256_file(bank_path),
            "array_dtype": str(reference_embeddings.dtype),
            "array_shape": list(reference_embeddings.shape),
            "embedding_normalization": (
                "raw mean-pooled vectors stored; distances normalize as documented"
            ),
        },
        "distance_definitions": {
            "cosine_distance": (
                "minimum 1 - cosine similarity after L2-normalizing query and each "
                "reference embedding"
            ),
            "nearest_neighbor_distance": (
                "minimum Euclidean distance in raw mean-pooled embedding space"
            ),
            "score_direction": "higher values mean farther from the reference bank",
        },
        "image_id_disjointness_note": (
            "Reference and evaluation image IDs are disjoint. No separate scene-level "
            "or perceptual near-duplicate audit was performed."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    score_image = local_image_dir / f"COCO_val2014_{chosen_evaluation_id}.jpg"
    if not score_image.is_file():
        raise FileNotFoundError(f"evaluation COCO image is missing: {score_image}")
    query_embedding = adapter.extract_image_embedding(score_image)
    score_values = embedding_distance_scores(
        query_embedding,
        reference_embeddings,
        reference_ids,
    )
    score_row = {
        "image_id": chosen_evaluation_id,
        "image_filename": score_image.name,
        "reference_set_id": reference_id,
        "reference_count": len(reference_ids),
        "reference_random_seed": random_seed,
        "encoder": encoder_metadata,
        **score_values,
    }
    score_path.write_text(
        json.dumps(score_row, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "reference_manifest": str(manifest_path),
        "embedding_bank": str(bank_path),
        "smoke_scores": str(score_path),
        "smoke_score": score_row,
        "reference_image_count": len(reference_ids),
        "available_disjoint_image_count": len(pilot_image_ids - evaluation_ids),
        "evaluation_image_count": len(evaluation_ids),
        "reference_evaluation_overlap_count": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a seeded COCO reference embedding bank excluding all evaluation images, "
            "then score one evaluation image."
        )
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--pilot-metadata", type=Path, default=DEFAULT_PILOT_METADATA)
    parser.add_argument("--evaluation-results", type=Path, default=DEFAULT_EVALUATION_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--reference-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--score-image-id")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        build_reference_bank_and_smoke_score(
            image_dir=args.image_dir,
            pilot_metadata_file=args.pilot_metadata,
            evaluation_results_file=args.evaluation_results,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            reference_count=args.reference_count,
            random_seed=args.seed,
            score_image_id=args.score_image_id,
            device=None if args.device == "auto" else args.device,
        )
    except Exception as exc:
        print(
            f"embedding OOD smoke evaluation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
