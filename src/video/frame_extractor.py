from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractedFrame:
    video_id: str
    frame_index: int
    timestamp_seconds: float
    frame_path: Path


def extract_keyframes(
    video_path: str | Path,
    output_dir: str | Path,
    sample_interval_seconds: float = 1.0,
    image_extension: str = ".jpg",
) -> list[ExtractedFrame]:
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be greater than 0")

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for video keyframe extraction") from exc

    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")

    frames_dir = Path(output_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_id = video.stem

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 1.0
    step = max(1, int(round(fps * sample_interval_seconds)))

    extracted: list[ExtractedFrame] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step == 0:
                frame_name = f"{video_id}_frame_{frame_index:06d}{image_extension}"
                frame_path = frames_dir / frame_name
                if not cv2.imwrite(str(frame_path), frame):
                    raise IOError(f"Failed to write extracted frame: {frame_path}")
                extracted.append(
                    ExtractedFrame(
                        video_id=video_id,
                        frame_index=frame_index,
                        timestamp_seconds=round(frame_index / fps, 4),
                        frame_path=frame_path,
                    )
                )
            frame_index += 1
    finally:
        capture.release()

    return extracted
