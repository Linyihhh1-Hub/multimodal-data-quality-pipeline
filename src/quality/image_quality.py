from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageStat


def _laplacian_variance(image: Image.Image) -> float:
    try:
        import cv2
        import numpy as np

        gray = np.array(image.convert("L"))
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        stat = ImageStat.Stat(image.convert("L"))
        return float(stat.var[0])


def _score_from_penalties(base: float, penalties: list[float]) -> float:
    return round(max(0.0, min(1.0, base - sum(penalties))), 4)


def assess_image(
    image_path: str | Path,
    min_width: int = 224,
    min_height: int = 224,
    min_blur_variance: float = 20.0,
    min_brightness: float = 25.0,
    max_brightness: float = 235.0,
    max_aspect_ratio: float = 3.5,
) -> dict:
    path = Path(image_path)
    reasons: list[str] = []
    penalties: list[float] = []

    if not path.exists():
        return {
            "image_valid": False,
            "width": 0,
            "height": 0,
            "file_size_bytes": 0,
            "aspect_ratio": 0.0,
            "blur_variance": 0.0,
            "brightness": 0.0,
            "image_quality_score": 0.0,
            "filter_status": "rejected",
            "filter_reasons": ["missing_image"],
        }

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            image = img.convert("RGB")
            width, height = image.size
            file_size = path.stat().st_size
            aspect_ratio = round(max(width / height, height / width), 4) if width and height else 0.0
            blur_variance = round(_laplacian_variance(image), 4)
            brightness = round(float(ImageStat.Stat(image.convert("L")).mean[0]), 4)
    except Exception:
        return {
            "image_valid": False,
            "width": 0,
            "height": 0,
            "file_size_bytes": 0,
            "aspect_ratio": 0.0,
            "blur_variance": 0.0,
            "brightness": 0.0,
            "image_quality_score": 0.0,
            "filter_status": "rejected",
            "filter_reasons": ["corrupt_image"],
        }

    status = "accepted"
    if width < min_width or height < min_height:
        reasons.append("low_resolution")
        penalties.append(0.55)
        status = "rejected"

    if aspect_ratio > max_aspect_ratio:
        reasons.append("abnormal_aspect_ratio")
        penalties.append(0.25)
        if status != "rejected":
            status = "review"

    if blur_variance < min_blur_variance:
        reasons.append("blurry_image")
        penalties.append(0.2)
        if status != "rejected":
            status = "review"

    if brightness < min_brightness or brightness > max_brightness:
        reasons.append("abnormal_brightness")
        penalties.append(0.15)
        if status != "rejected":
            status = "review"

    return {
        "image_valid": True,
        "width": width,
        "height": height,
        "file_size_bytes": file_size,
        "aspect_ratio": aspect_ratio,
        "blur_variance": blur_variance,
        "brightness": brightness,
        "image_quality_score": _score_from_penalties(1.0, penalties),
        "filter_status": status,
        "filter_reasons": reasons,
    }
