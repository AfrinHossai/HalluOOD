from __future__ import annotations

import numpy as np
import pytest

from halluood.embeddings import (
    EmbeddingError,
    embedding_distance_scores,
    reference_set_identifier,
    select_disjoint_reference_images,
)


def test_identical_query_has_zero_cosine_and_nearest_neighbor_distances():
    scores = embedding_distance_scores(
        [3.0, 4.0],
        [[3.0, 4.0], [8.0, 6.0]],
        ["000000000001", "000000000002"],
    )

    assert scores["cosine_distance"] == pytest.approx(0.0)
    assert scores["nearest_neighbor_distance"] == pytest.approx(0.0)
    assert scores["cosine_nearest_reference_image_id"] == "000000000001"
    assert scores["nearest_neighbor_reference_image_id"] == "000000000001"


def test_dissimilar_query_has_positive_distances_and_identifies_nearest_rows():
    scores = embedding_distance_scores(
        [1.0, 0.0],
        [[0.0, 1.0], [-1.0, 0.0]],
        ["000000000010", "000000000011"],
    )

    assert scores["cosine_distance"] == pytest.approx(1.0)
    assert scores["nearest_neighbor_distance"] == pytest.approx(np.sqrt(2.0))
    assert scores["cosine_nearest_reference_image_id"] == "000000000010"
    assert scores["nearest_neighbor_reference_image_id"] == "000000000010"


@pytest.mark.parametrize(
    ("query", "bank", "message"),
    [
        ([0.0, 0.0], [[1.0, 0.0]], "zero vector"),
        ([float("nan"), 1.0], [[1.0, 0.0]], "finite"),
        ([1.0, 0.0], [[0.0, 0.0]], "zero vectors"),
        ([1.0, 0.0], [[float("inf"), 0.0]], "finite"),
        ([1.0, 0.0], [[1.0]], "dimensions differ"),
        ([1.0, 0.0], [], "non-empty"),
    ],
)
def test_distance_scores_reject_invalid_vectors(query, bank, message):
    with pytest.raises(EmbeddingError, match=message):
        embedding_distance_scores(query, bank, ["000000000001"] * len(bank))


def test_reference_selection_is_seeded_and_disjoint(tmp_path):
    image_paths = []
    for image_id in range(1, 8):
        path = tmp_path / f"COCO_val2014_{image_id:012d}.jpg"
        path.touch()
        image_paths.append(path)

    first = select_disjoint_reference_images(
        image_paths,
        {"000000000001", "COCO_val2014_000000000002.jpg"},
        sample_size=3,
        random_seed=73,
    )
    second = select_disjoint_reference_images(
        image_paths,
        {"000000000001", "COCO_val2014_000000000002.jpg"},
        sample_size=3,
        random_seed=73,
    )

    assert [(image_id, path.name) for image_id, path in first] == [
        (image_id, path.name) for image_id, path in second
    ]
    assert len(first) == 3
    assert {image_id for image_id, _ in first}.isdisjoint(
        {"000000000001", "000000000002"}
    )


def test_reference_selection_fails_if_disjoint_set_is_too_small(tmp_path):
    path = tmp_path / "COCO_val2014_000000000001.jpg"
    path.touch()

    with pytest.raises(EmbeddingError, match="only 0 local images"):
        select_disjoint_reference_images(
            [path],
            {"000000000001"},
            sample_size=1,
            random_seed=2027,
        )


def test_reference_set_identifier_is_stable_and_tracks_encoder_and_seed():
    embeddings = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arguments = {
        "image_ids": ["000000000001", "000000000002"],
        "embeddings": embeddings,
        "encoder_identifier": "model/image-encoder",
        "encoder_revision": "abc123",
        "random_seed": 2027,
    }

    first = reference_set_identifier(**arguments)
    second = reference_set_identifier(**arguments)
    changed_seed = reference_set_identifier(**{**arguments, "random_seed": 2028})

    assert first == second
    assert first != changed_seed
