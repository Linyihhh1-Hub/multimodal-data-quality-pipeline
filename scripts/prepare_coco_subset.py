from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def prepare_coco_subset(
    annotations_path: str | Path,
    source_image_dir: str | Path,
    output_raw_dir: str | Path,
    limit: int = 5000,
) -> Path:
    annotations_file = Path(annotations_path)
    source_dir = Path(source_image_dir)
    output_dir = Path(output_raw_dir)
    image_output_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.jsonl"

    payload = json.loads(annotations_file.read_text(encoding="utf-8"))
    image_lookup = {
        str(item["id"]): item.get("file_name", f"{int(item['id']):012d}.jpg")
        for item in payload.get("images", [])
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for annotation in payload.get("annotations", []):
            image_id = str(annotation["image_id"])
            file_name = image_lookup.get(image_id, f"{int(image_id):012d}.jpg")
            source_image = source_dir / file_name
            if not source_image.exists():
                continue

            target_image = image_output_dir / file_name
            if not target_image.exists():
                shutil.copy2(source_image, target_image)

            row = {
                "image_id": image_id,
                "image_path": f"images/{file_name}",
                "caption": annotation.get("caption", ""),
                "source": "coco",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written >= limit:
                break

    if written == 0:
        raise ValueError("No COCO samples were written. Check annotations and image directory.")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local COCO captions subset manifest.")
    parser.add_argument("--annotations", required=True, help="Path to captions_train2017.json or captions_val2017.json")
    parser.add_argument("--source-image-dir", required=True, help="Directory containing COCO image files")
    parser.add_argument("--output-raw-dir", default="data/raw", help="Output raw data directory")
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    manifest_path = prepare_coco_subset(
        annotations_path=args.annotations,
        source_image_dir=args.source_image_dir,
        output_raw_dir=args.output_raw_dir,
        limit=args.limit,
    )
    print(f"Wrote COCO subset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
