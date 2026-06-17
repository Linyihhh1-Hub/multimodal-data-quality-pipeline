from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.quality.image_quality import assess_image
from src.video.frame_extractor import extract_keyframes
from src.video.metadata import probe_video


def build_video_frame_manifest(
    video_path: str | Path,
    frame_output_dir: str | Path,
    manifest_path: str | Path,
    raw_data_dir: str | Path | None = None,
    sample_interval_seconds: float = 1.0,
    max_frames: int | None = None,
    source: str = "video_demo",
    prompt_template: str = "Describe the key frame at {timestamp_seconds:.1f}s.",
) -> list[dict[str, Any]]:
    frames = extract_keyframes(
        video_path=video_path,
        output_dir=frame_output_dir,
        sample_interval_seconds=sample_interval_seconds,
        max_frames=max_frames,
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


def _relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def summarize_video_quality(frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(frame_rows)
    valid_rows = [row for row in frame_rows if row.get("filter_status") == "accepted"]
    rejected_rows = [row for row in frame_rows if row.get("filter_status") == "rejected"]
    review_rows = [row for row in frame_rows if row.get("filter_status") == "review"]
    scores = [float(row.get("image_quality_score", 0.0)) for row in frame_rows]
    return {
        "sampled_frame_count": total,
        "valid_frame_count": len(valid_rows),
        "review_frame_count": len(review_rows),
        "rejected_frame_count": len(rejected_rows),
        "video_quality_score": round(sum(scores) / total, 4) if total else 0.0,
    }


def build_video_dataset(
    video_path: str | Path,
    frame_output_dir: str | Path,
    video_manifest_path: str | Path,
    frame_manifest_path: str | Path,
    raw_data_dir: str | Path | None = None,
    sample_interval_seconds: float = 1.0,
    max_frames: int | None = None,
    source: str = "video_demo",
    prompt_template: str = "Key frame from {video_id} at {timestamp_seconds:.1f}s for visual content review.",
) -> dict[str, Any]:
    raw_base = Path(raw_data_dir) if raw_data_dir is not None else Path(frame_output_dir).parent
    frame_rows = build_video_frame_manifest(
        video_path=video_path,
        frame_output_dir=frame_output_dir,
        manifest_path=frame_manifest_path,
        raw_data_dir=raw_base,
        sample_interval_seconds=sample_interval_seconds,
        max_frames=max_frames,
        source=source,
        prompt_template=prompt_template,
    )
    metadata = probe_video(video_path)
    video = Path(video_path)
    video_row = {
        **metadata,
        "video_path": _relative_path(video, raw_base),
        "source": source,
        **summarize_video_quality(frame_rows),
    }

    output = Path(video_manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(video_row, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "video_manifest_path": str(output),
        "frame_manifest_path": str(Path(frame_manifest_path)),
        "video_rows": [video_row],
        "frame_rows": frame_rows,
        "frame_count": len(frame_rows),
    }
