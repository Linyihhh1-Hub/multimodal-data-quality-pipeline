from __future__ import annotations

from pathlib import Path
from typing import Any


def probe_video(video_path: str | Path) -> dict[str, Any]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for video metadata probing") from exc

    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
    finally:
        capture.release()

    duration = round(frame_count / fps, 4) if fps > 0 else 0.0
    return {
        "video_id": video.stem,
        "video_path": str(video),
        "fps": round(fps, 4),
        "frame_count": frame_count,
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "file_size_bytes": video.stat().st_size,
    }
