from __future__ import annotations

from pathlib import Path


def compute_perceptual_hash(image_path: str | Path) -> str:
    try:
        import imagehash
        from PIL import Image

        with Image.open(image_path) as image:
            return str(imagehash.phash(image))
    except Exception:
        return ""


def mark_duplicate_hashes(rows: list[dict], hash_field: str = "perceptual_hash") -> list[dict]:
    seen: set[str] = set()
    output: list[dict] = []
    for row in rows:
        item = dict(row)
        value = item.get(hash_field, "")
        item["is_duplicate"] = bool(value and value in seen)
        if value:
            seen.add(value)
        output.append(item)
    return output
