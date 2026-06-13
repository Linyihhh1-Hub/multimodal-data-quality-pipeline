import json
from pathlib import Path

from PIL import Image

from scripts.prepare_coco_subset import prepare_coco_subset


def test_prepare_coco_subset_writes_manifest_and_copies_selected_images(tmp_path: Path):
    annotations = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg"},
            {"id": 2, "file_name": "000000000002.jpg"},
        ],
        "annotations": [
            {"image_id": 1, "caption": "A person rides a bicycle on a street."},
            {"image_id": 2, "caption": "A dog sits on a sofa."},
        ],
    }
    annotation_path = tmp_path / "captions.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    image_dir = tmp_path / "source_images"
    image_dir.mkdir()
    Image.new("RGB", (320, 240), color=(80, 120, 160)).save(image_dir / "000000000001.jpg")
    Image.new("RGB", (320, 240), color=(160, 120, 80)).save(image_dir / "000000000002.jpg")

    manifest_path = prepare_coco_subset(
        annotations_path=annotation_path,
        source_image_dir=image_dir,
        output_raw_dir=tmp_path / "raw",
        limit=1,
    )

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "image_id": "1",
            "image_path": "images/000000000001.jpg",
            "caption": "A person rides a bicycle on a street.",
            "source": "coco",
        }
    ]
    assert (tmp_path / "raw" / "images" / "000000000001.jpg").exists()
    assert not (tmp_path / "raw" / "images" / "000000000002.jpg").exists()
