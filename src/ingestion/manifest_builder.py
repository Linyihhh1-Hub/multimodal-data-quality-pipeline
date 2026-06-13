from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = ["image_id", "image_path", "caption", "source"]


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(payload)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported manifest format: {path.suffix}")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["sample_id"] = df.apply(lambda row: f"{row['source']}_{row['image_id']}", axis=1)
    return df[["sample_id", *REQUIRED_COLUMNS]]


def build_manifest_from_coco_annotations(
    annotations_path: str | Path,
    images_dir: str | Path,
    source: str = "coco",
    limit: int | None = None,
) -> pd.DataFrame:
    payload = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    image_lookup = {
        str(item["id"]): item.get("file_name", f"{item['id']}.jpg")
        for item in payload.get("images", [])
    }
    rows: list[dict] = []
    for ann in payload.get("annotations", []):
        image_id = str(ann["image_id"])
        file_name = image_lookup.get(image_id, f"{int(image_id):012d}.jpg")
        rows.append(
            {
                "sample_id": f"{source}_{image_id}",
                "image_id": image_id,
                "image_path": str(Path(images_dir) / file_name),
                "caption": ann.get("caption", ""),
                "source": source,
            }
        )
        if limit and len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def write_manifest_jsonl(rows: Iterable[dict], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
