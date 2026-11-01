import json
import subprocess
import sys
from pathlib import Path

import pytest

from halluood.data import (
    PopeDataError,
    load_pope_records,
    normalize_yes_no,
    sample_pilot_subset,
)


def _record(question_id, image, label="yes", **extra):
    return {
        "question_id": question_id,
        "image": image,
        "text": "Is there a bicycle in the image?",
        "label": label,
        **extra,
    }


def _load_values(tmp_path, values, filename="questions.json"):
    source = tmp_path / filename
    source.write_text(json.dumps(values), encoding="utf-8")
    return load_pope_records(source)


def test_load_json_array_preserves_raw_label_and_infers_strategy(tmp_path):
    records = _load_values(
        tmp_path,
        [_record(1, "COCO_val2014_0001.jpg", " YES. ")],
        filename="coco_pope_random.json",
    )

    assert len(records) == 1
    assert records[0].label == " YES. "
    assert records[0].normalized_label == "yes"
    assert records[0].category == "random"
    assert records[0].to_dict()["label"] == " YES. "
    assert records[0].to_dict()["normalized_label"] == "yes"
    assert records[0].to_dict()["category"] == "random"


def test_load_jsonl_and_record_category(tmp_path):
    source = tmp_path / "questions.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(_record(1, "image-a.jpg", "yes", category="adversarial")),
                json.dumps(_record(2, "image-b.jpg", "no", category="popular")),
            ]
        ),
        encoding="utf-8",
    )

    records = load_pope_records(source)

    assert [record.normalized_label for record in records] == ["yes", "no"]
    assert [record.category for record in records] == ["adversarial", "popular"]


def test_load_single_json_object(tmp_path):
    source = tmp_path / "single.json"
    source.write_text(json.dumps(_record(1, "image-a.jpg", "no")), encoding="utf-8")

    records = load_pope_records(source)

    assert len(records) == 1
    assert records[0].normalized_label == "no"


@pytest.mark.parametrize(
    ("label", "expected"),
    [("yes", "yes"), (" NO ", "no"), ("Yes!", "yes")],
)
def test_normalize_yes_no(label, expected):
    assert normalize_yes_no(label) == expected


@pytest.mark.parametrize("label", ["maybe", "yes and no", "", None, True])
def test_normalize_rejects_non_binary_labels(label):
    with pytest.raises(PopeDataError):
        normalize_yes_no(label)


@pytest.mark.parametrize("missing_field", ["question_id", "image", "text", "label"])
def test_loader_rejects_missing_required_fields(tmp_path, missing_field):
    item = _record(1, "image-a.jpg")
    del item[missing_field]
    source = tmp_path / "bad.json"
    source.write_text(json.dumps([item]), encoding="utf-8")

    with pytest.raises(PopeDataError, match=missing_field):
        load_pope_records(source)


def test_loader_rejects_invalid_optional_category(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text(
        json.dumps([_record(1, "image-a.jpg", category=3)]),
        encoding="utf-8",
    )

    with pytest.raises(PopeDataError, match="category"):
        load_pope_records(source)


def test_sampler_is_deterministic_and_keeps_image_groups_together(tmp_path):
    records = _load_values(
        tmp_path,
        [
            _record(1, "image-a.jpg"),
            _record(2, "image-a.jpg", "no"),
            _record(3, "image-b.jpg"),
            _record(4, "image-b.jpg", "no"),
            _record(5, "image-c.jpg"),
            _record(6, "image-c.jpg", "no"),
            _record(7, "image-d.jpg"),
            _record(8, "image-d.jpg", "no"),
        ],
    )

    first = sample_pilot_subset(records, sample_size=2, random_seed=2027)
    second = sample_pilot_subset(records, sample_size=2, random_seed=2027)
    selected_images = {record.image for record in first}

    assert [record.image for record in first] == [record.image for record in second]
    assert len(selected_images) == 2
    assert len(first) == 4
    assert all(sum(record.image == image for record in first) == 2 for image in selected_images)
    image_group_order = [
        record.image
        for index, record in enumerate(first)
        if index == 0 or record.image != first[index - 1].image
    ]
    assert len(image_group_order) == len(selected_images)


def test_sampler_validates_size_and_seed(tmp_path):
    records = _load_values(tmp_path, [_record(1, "image-a.jpg")])

    with pytest.raises(ValueError, match="sample_size"):
        sample_pilot_subset(records, sample_size=0)
    with pytest.raises(ValueError, match="exceeds"):
        sample_pilot_subset(records, sample_size=2)
    with pytest.raises(ValueError, match="random_seed"):
        sample_pilot_subset(records, sample_size=1, random_seed=-1)


def test_prepare_data_cli_writes_normalized_grouped_jsonl(tmp_path):
    source = tmp_path / "input.json"
    output = tmp_path / "prepared" / "output.jsonl"
    source.write_text(
        json.dumps(
            [
                _record(1, "image-a.jpg", "YES"),
                _record(2, "image-a.jpg", "no"),
                _record(3, "image-b.jpg", "yes"),
                _record(4, "image-b.jpg", "no"),
            ]
        ),
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--output",
            str(output),
            "--sample-size",
            "1",
            "--random-seed",
            "2027",
        ],
        cwd=script.parent.parent,
        capture_output=True,
        check=False,
        text=True,
    )

    prepared = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result.returncode == 0, result.stderr
    assert len(prepared) == 2
    assert len({record["image"] for record in prepared}) == 1
    assert {record["normalized_label"] for record in prepared} == {"yes", "no"}
