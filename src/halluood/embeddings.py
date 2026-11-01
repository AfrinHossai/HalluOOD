"""Frozen image embeddings and nearest-reference distance scores."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class EmbeddingError(ValueError):
    """Raised when an image embedding or reference selection is invalid."""


_COCO_IMAGE_ID = re.compile(r"^COCO_(?:train|val)2014_(\d{12})$", re.IGNORECASE)


def normalize_coco_image_id(value: str | int) -> str:
    """Return the 12-digit COCO identifier from a numeric ID or image filename."""

    value_string = str(value)
    if value_string.isdecimal() and len(value_string) <= 12:
        return value_string.zfill(12)
    name = Path(value_string).name
    stem = Path(name).stem
    match = _COCO_IMAGE_ID.fullmatch(stem)
    if match is None:
        raise EmbeddingError(f"not a supported COCO 2014 image identifier: {value_string!r}")
    return match.group(1)


def select_disjoint_reference_images(
    image_paths: Sequence[str | Path],
    evaluation_image_ids: set[str] | Sequence[str],
    *,
    sample_size: int,
    random_seed: int,
) -> list[tuple[str, Path]]:
    """Seededly select local COCO images that do not occur in evaluation."""

    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise EmbeddingError("sample_size must be a positive integer")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise EmbeddingError("random_seed must be a non-negative integer")
    evaluation_ids = {normalize_coco_image_id(image_id) for image_id in evaluation_image_ids}
    candidates: dict[str, Path] = {}
    for image_path in image_paths:
        path = Path(image_path)
        image_id = normalize_coco_image_id(path.name)
        if image_id in candidates:
            raise EmbeddingError(f"duplicate local COCO image ID: {image_id}")
        if path.is_file() and image_id not in evaluation_ids:
            candidates[image_id] = path

    sorted_ids = sorted(candidates)
    if sample_size > len(sorted_ids):
        raise EmbeddingError(
            f"requested {sample_size} reference images, but only {len(sorted_ids)} "
            "local images are disjoint from evaluation"
        )
    rng = np.random.Generator(np.random.PCG64(random_seed))
    chosen_indices = sorted(
        int(index) for index in rng.choice(len(sorted_ids), size=sample_size, replace=False)
    )
    selected = [(sorted_ids[index], candidates[sorted_ids[index]]) for index in chosen_indices]
    if evaluation_ids.intersection(image_id for image_id, _ in selected):
        raise EmbeddingError("reference selection overlaps evaluation image IDs")
    return selected


def _embedding_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EmbeddingError(f"{name} must be numeric") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise EmbeddingError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(vector).all():
        raise EmbeddingError(f"{name} must contain only finite values")
    if np.linalg.norm(vector) == 0:
        raise EmbeddingError(f"{name} must not be a zero vector")
    return vector


def _embedding_matrix(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EmbeddingError("reference_embeddings must be a numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise EmbeddingError("reference_embeddings must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise EmbeddingError("reference_embeddings must contain only finite values")
    if np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise EmbeddingError("reference_embeddings must not contain zero vectors")
    return matrix


def embedding_distance_scores(
    query_embedding: Sequence[float] | np.ndarray,
    reference_embeddings: Sequence[Sequence[float]] | np.ndarray,
    reference_image_ids: Sequence[str],
) -> dict[str, Any]:
    """Measure nearest-bank cosine distance and raw-feature Euclidean distance.

    Cosine distance is 1 minus cosine similarity after L2 normalization, using
    the closest cosine reference. The nearest-neighbor score is the minimum
    Euclidean distance between unnormalized mean-pooled feature vectors.
    Larger values indicate a more distant image; neither score is a probability.
    """

    query = _embedding_vector(query_embedding, "query_embedding")
    bank = _embedding_matrix(reference_embeddings)
    if bank.shape[1] != query.shape[0]:
        raise EmbeddingError(
            f"embedding dimensions differ: query has {query.shape[0]}, "
            f"reference bank has {bank.shape[1]}"
        )
    if len(reference_image_ids) != bank.shape[0]:
        raise EmbeddingError("reference_image_ids must match the number of reference embeddings")
    ids = [normalize_coco_image_id(image_id) for image_id in reference_image_ids]
    if len(set(ids)) != len(ids):
        raise EmbeddingError("reference_image_ids must be unique")

    normalized_query = query / np.linalg.norm(query)
    normalized_bank = bank / np.linalg.norm(bank, axis=1, keepdims=True)
    cosine_similarity = np.clip(normalized_bank @ normalized_query, -1.0, 1.0)
    cosine_distances = 1.0 - cosine_similarity
    cosine_index = int(np.argmin(cosine_distances))
    euclidean_distances = np.linalg.norm(bank - query, axis=1)
    euclidean_index = int(np.argmin(euclidean_distances))
    return {
        "cosine_distance": float(cosine_distances[cosine_index]),
        "cosine_nearest_reference_image_id": ids[cosine_index],
        "nearest_neighbor_distance": float(euclidean_distances[euclidean_index]),
        "nearest_neighbor_reference_image_id": ids[euclidean_index],
        "score_direction": "higher_means_more_distant_from_reference_bank",
    }


def reference_set_identifier(
    image_ids: Sequence[str],
    embeddings: Sequence[Sequence[float]] | np.ndarray,
    *,
    encoder_identifier: str,
    encoder_revision: str,
    random_seed: int,
) -> str:
    """Create a stable SHA-256 identifier for the selected and encoded bank."""

    matrix = _embedding_matrix(embeddings).astype("<f4", copy=False)
    normalized_ids = [normalize_coco_image_id(image_id) for image_id in image_ids]
    if len(normalized_ids) != matrix.shape[0] or len(set(normalized_ids)) != len(normalized_ids):
        raise EmbeddingError("image_ids must uniquely match the embedding-bank rows")
    hasher = hashlib.sha256()
    hasher.update(encoder_identifier.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(encoder_revision.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(str(random_seed).encode("ascii"))
    hasher.update(b"\0")
    hasher.update("\n".join(normalized_ids).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(matrix.tobytes(order="C"))
    return hasher.hexdigest()
