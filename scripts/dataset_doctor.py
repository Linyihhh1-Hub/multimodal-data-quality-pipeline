from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def inspect_manifest(manifest_path: str | Path, raw_data_dir: str | Path) -> dict[str, Any]:
    manifest = Path(manifest_path)
    raw_dir = Path(raw_data_dir)
    report: dict[str, Any] = {
        "manifest_path": str(manifest),
        "raw_data_dir": str(raw_dir),
        "manifest_exists": manifest.exists(),
        "raw_data_dir_exists": raw_dir.exists(),
        "total_rows": 0,
        "missing_images": 0,
        "empty_captions": 0,
        "duplicate_image_ids": 0,
        "ready_for_pipeline": False,
    }
    if not manifest.exists():
        return report

    rows = _load_jsonl(manifest)
    image_ids = [str(row.get("image_id", "")) for row in rows]
    report["total_rows"] = len(rows)
    report["duplicate_image_ids"] = len(image_ids) - len(set(image_ids))

    missing_images = 0
    empty_captions = 0
    for row in rows:
        image_path = Path(str(row.get("image_path", "")))
        resolved = image_path if image_path.is_absolute() else raw_dir / image_path
        if not resolved.exists():
            missing_images += 1
        if not str(row.get("caption", "")).strip():
            empty_captions += 1

    report["missing_images"] = missing_images
    report["empty_captions"] = empty_captions
    report["ready_for_pipeline"] = (
        report["raw_data_dir_exists"]
        and report["total_rows"] > 0
        and missing_images == 0
        and empty_captions < report["total_rows"]
    )
    return report


def inspect_coco_inputs(annotations_path: str | Path, source_image_dir: str | Path) -> dict[str, Any]:
    annotations = Path(annotations_path)
    image_dir = Path(source_image_dir)
    report = {
        "annotations_path": str(annotations),
        "source_image_dir": str(image_dir),
        "annotations_exists": annotations.exists(),
        "source_image_dir_exists": image_dir.exists(),
        "annotation_count": 0,
        "image_count_in_annotations": 0,
        "ready_for_subset_build": False,
    }
    if annotations.exists():
        payload = json.loads(annotations.read_text(encoding="utf-8"))
        report["annotation_count"] = len(payload.get("annotations", []))
        report["image_count_in_annotations"] = len(payload.get("images", []))
    report["ready_for_subset_build"] = report["annotations_exists"] and report["source_image_dir_exists"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local data readiness for the multimodal pipeline.")
    parser.add_argument("--manifest", default="data/raw/manifest.jsonl")
    parser.add_argument("--raw-data-dir", default="data/raw")
    parser.add_argument("--coco-annotations", default="")
    parser.add_argument("--coco-image-dir", default="")
    parser.add_argument("--output", default="data/processed/dataset_doctor.json")
    args = parser.parse_args()

    report = {
        "manifest": inspect_manifest(args.manifest, args.raw_data_dir),
    }
    if args.coco_annotations and args.coco_image_dir:
        report["coco"] = inspect_coco_inputs(args.coco_annotations, args.coco_image_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
