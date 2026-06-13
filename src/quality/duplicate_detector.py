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


def annotate_duplicate_hashes(rows: list[dict], hash_field: str = "perceptual_hash") -> list[dict]:
    group_sizes: dict[str, int] = {}
    for row in rows:
        value = str(row.get(hash_field, "") or "")
        if value:
            group_sizes[value] = group_sizes.get(value, 0) + 1

    seen: set[str] = set()
    output: list[dict] = []
    for row in rows:
        item = dict(row)
        value = str(item.get(hash_field, "") or "")
        item["duplicate_group_size"] = group_sizes.get(value, 0) if value else 0
        item["is_duplicate_image"] = bool(value and value in seen)
        if value:
            seen.add(value)
        output.append(item)
    return output
