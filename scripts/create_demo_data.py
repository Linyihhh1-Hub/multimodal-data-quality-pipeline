from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


def make_textured_image(path: Path, color: tuple[int, int, int], stripe: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (320, 240), color=color)
    draw = ImageDraw.Draw(image)
    for x in range(0, 320, 24):
        draw.line((x, 0, x, 239), fill=stripe, width=3)
    draw.rectangle((80, 60, 240, 180), outline=(255, 255, 255), width=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    raw_dir = Path("data/raw")
    image_dir = raw_dir / "images"
    make_textured_image(image_dir / "demo_1.jpg", (90, 130, 180), (20, 30, 40))
    make_textured_image(image_dir / "demo_2.jpg", (170, 120, 90), (50, 20, 20))
    Image.new("RGB", (100, 100), color=(120, 120, 120)).save(image_dir / "demo_small.jpg")

    rows = [
        {
            "image_id": "demo_1",
            "image_path": "images/demo_1.jpg",
            "caption": "A person rides a bicycle on a city street.",
            "source": "demo",
        },
        {
            "image_id": "demo_2",
            "image_path": "images/demo_2.jpg",
            "caption": "A dog sits near a wooden table indoors.",
            "source": "demo",
        },
        {
            "image_id": "demo_small",
            "image_path": "images/demo_small.jpg",
            "caption": "small",
            "source": "demo",
        },
        {
            "image_id": "demo_missing_caption",
            "image_path": "images/demo_1.jpg",
            "caption": "",
            "source": "demo",
        },
    ]
    manifest = raw_dir / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote demo manifest: {manifest}")


if __name__ == "__main__":
    main()
