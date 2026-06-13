from pathlib import Path

from PIL import Image

from src.quality.image_quality import assess_image


def test_assess_image_accepts_clear_valid_image(tmp_path: Path):
    image_path = tmp_path / "valid.jpg"
    image = Image.new("RGB", (320, 240), color=(120, 160, 200))
    for x in range(0, 320, 20):
        for y in range(240):
            image.putpixel((x, y), (20, 20, 20))
    image.save(image_path)

    result = assess_image(image_path, min_width=224, min_height=224)

    assert result["image_valid"] is True
    assert result["width"] == 320
    assert result["height"] == 240
    assert result["filter_status"] == "accepted"
    assert result["image_quality_score"] > 0.5


def test_assess_image_rejects_low_resolution_image(tmp_path: Path):
    image_path = tmp_path / "small.jpg"
    Image.new("RGB", (100, 100), color=(120, 120, 120)).save(image_path)

    result = assess_image(image_path, min_width=224, min_height=224)

    assert result["image_valid"] is True
    assert result["filter_status"] == "rejected"
    assert "low_resolution" in result["filter_reasons"]
