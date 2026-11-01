import json
from collections import Counter

from PIL import Image

from halluood.model import MockVLMAdapter
from scripts.run_baseline import (
    CATEGORIES,
    normalize_prediction,
    run_baseline,
    select_balanced_questions,
)


def _write_pilot(tmp_path, image_count=8):
    annotations_dir = tmp_path / "processed"
    annotations_dir.mkdir()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    records_by_category = {}

    for category_index, category in enumerate(CATEGORIES):
        records = []
        for image_index in range(image_count):
            filename = f"COCO_val2014_{image_index:012d}.jpg"
            label = "yes" if image_index % 2 == 0 else "no"
            for replicate, raw_label in enumerate((label, "no" if label == "yes" else "yes")):
                records.append(
                    {
                        "question_id": category_index * 1000 + image_index * 2 + replicate,
                        "image": filename,
                        "text": f"Is object {image_index} visible?",
                        "label": raw_label,
                        "category": category,
                    }
                )
            if category_index == 0:
                Image.new("RGB", (4, 3), color=(20, image_index, 80)).save(
                    image_dir / filename
                )

        (annotations_dir / f"coco_pope_{category}_pilot.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        records_by_category[category] = records
    return annotations_dir, image_dir


def test_prediction_normalizer_preserves_ambiguous_cases():
    assert normalize_prediction("Yes, there is a backpack.") == ("yes", "parsed")
    assert normalize_prediction("No.") == ("no", "parsed")
    assert normalize_prediction("There is no chair present in the image.") == ("no", "parsed")
    assert normalize_prediction("I think there may be one.") == (None, "ambiguous")
    assert normalize_prediction("There is no mention of a parking meter.") == (
        None,
        "ambiguous",
    )
    assert normalize_prediction("Yes, no backpack is visible.") == (None, "ambiguous")
    assert normalize_prediction("No doubt a backpack is present.") == (None, "ambiguous")


def test_sampler_is_seeded_balanced_and_prefers_distinct_images(tmp_path):
    _, image_dir = _write_pilot(tmp_path)
    from halluood.data import load_pope_records

    records_by_category = {
        category: load_pope_records(
            tmp_path / "processed" / f"coco_pope_{category}_pilot.jsonl"
        )
        for category in CATEGORIES
    }

    first = select_balanced_questions(records_by_category, sample_size=6, random_seed=91)
    second = select_balanced_questions(records_by_category, sample_size=6, random_seed=91)

    assert [(category, record.question_id) for category, record in first] == [
        (category, record.question_id) for category, record in second
    ]
    assert Counter(category for category, _ in first) == Counter(
        {category: 2 for category in CATEGORIES}
    )
    assert len({record.image for _, record in first}) == 6
    assert all((image_dir / record.image).is_file() for _, record in first)


def test_baseline_writes_rows_and_category_metrics_with_mock(tmp_path):
    annotations_dir, image_dir = _write_pilot(tmp_path)
    output_file = tmp_path / "results" / "baseline.jsonl"
    adapter = MockVLMAdapter(answer="Yes, visible.")

    summary, rows = run_baseline(
        annotations_dir=annotations_dir,
        image_dir=image_dir,
        output_file=output_file,
        sample_size=6,
        random_seed=19,
        adapter=adapter,
    )

    saved_rows = [json.loads(line) for line in output_file.read_text().splitlines()]
    summary_file = output_file.with_name("baseline_summary.json")
    saved_summary = json.loads(summary_file.read_text(encoding="utf-8"))
    expected_correct = sum(row["ground_truth_label"] == "yes" for row in rows)

    assert len(rows) == len(saved_rows) == len(adapter.calls) == 6
    assert summary["selection_category_counts"] == {category: 2 for category in CATEGORIES}
    assert summary["distinct_image_count"] == 6
    assert summary["overall"]["correct_count"] == expected_correct
    assert summary["overall"]["ambiguous_count"] == 0
    assert saved_summary["overall"] == summary["overall"]
    assert all(row["model_identifier"] == "halluood/mock-vlm" for row in saved_rows)
    assert all(row["normalized_prediction"] == "yes" for row in saved_rows)
    assert all(row["inference_time_seconds"] >= 0 for row in saved_rows)
    assert set(saved_rows[0]) >= {
        "question_id",
        "image_id",
        "category",
        "ground_truth_label",
        "raw_prediction",
        "normalized_prediction",
        "model_identifier",
        "inference_time_seconds",
    }
    assert set(summary["by_category"]) == set(CATEGORIES)
