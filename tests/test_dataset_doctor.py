import json
from pathlib import Path

from PIL import Image

from scripts.dataset_doctor import inspect_coco_inputs, inspect_manifest


def test_inspect_manifest_reports_missing_images_and_empty_captions(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    image_dir = raw_dir / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (320, 240), color=(80, 120, 160)).save(image_dir / "a.jpg")
    manifest = raw_dir / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                '{"image_id":"a","image_path":"images/a.jpg","caption":"A valid caption.","source":"demo"}',
                '{"image_id":"b","image_path":"images/missing.jpg","caption":"","source":"demo"}',
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_manifest(manifest, raw_dir)

    assert report["manifest_exists"] is True
    assert report["total_rows"] == 2
    assert report["missing_images"] == 1
    assert report["empty_captions"] == 1
    assert report["ready_for_pipeline"] is False


def test_inspect_coco_inputs_detects_annotation_and_image_dir(tmp_path: Path):
    annotations = tmp_path / "captions_val2017.json"
    annotations.write_text(json.dumps({"images": [], "annotations": []}), encoding="utf-8")
    image_dir = tmp_path / "val2017"
    image_dir.mkdir()

    report = inspect_coco_inputs(annotations, image_dir)

    assert report["annotations_exists"] is True
    assert report["source_image_dir_exists"] is True
    assert report["ready_for_subset_build"] is True
