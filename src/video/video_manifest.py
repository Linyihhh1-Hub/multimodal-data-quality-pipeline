from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.quality.image_quality import assess_image
from src.video.frame_extractor import extract_keyframes


def build_video_frame_manifest(
    video_path: str | Path,
    frame_output_dir: str | Path,
    manifest_path: str | Path,
    raw_data_dir: str | Path | None = None,
    sample_interval_seconds: float = 1.0,
    source: str = "video_demo",
    prompt_template: str = "Describe the key frame at {timestamp_seconds:.1f}s.",
) -> list[dict[str, Any]]:
    frames = extract_keyframes(
        video_path=video_path,
        output_dir=frame_output_dir,
        sample_interval_seconds=sample_interval_seconds,
    )
    video = Path(video_path)
    frame_dir = Path(frame_output_dir)
    path_base = Path(raw_data_dir) if raw_data_dir is not None else frame_dir.parent
    rows: list[dict[str, Any]] = []

    for frame in frames:
        quality = assess_image(frame.frame_path)
        try:
            relative_frame_path = frame.frame_path.relative_to(path_base)
        except ValueError:
            relative_frame_path = frame.frame_path
        rows.append(
            {
                "sample_id": f"{frame.video_id}_frame_{frame.frame_index:06d}",
                "video_id": frame.video_id,
                "video_path": str(video),
                "frame_path": str(relative_frame_path).replace("\\", "/"),
                "caption": prompt_template.format(
                    video_id=frame.video_id,
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                ),
                "source": source,
                "frame_index": frame.frame_index,
                "timestamp_seconds": frame.timestamp_seconds,
                **quality,
            }
        )

    output = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows
