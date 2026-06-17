from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


IMAGE_CREATIVE_INSTRUCTION = "请根据这张图片生成适合短视频创作的镜头描述和创作提示。"
VIDEO_CREATIVE_INSTRUCTION = "请根据这些视频关键帧生成适合智能创作模型使用的短视频场景提示。"


def export_image_creative_sft(df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = df[df["filter_status"] == "accepted"]
    with path.open("w", encoding="utf-8") as handle:
        for _, row in rows.iterrows():
            caption = str(row["caption"])
            payload = {
                "sample_id": row.get("sample_id", row.get("image_id", "")),
                "task_type": "image_to_video_prompt",
                "quality_score": float(row.get("final_quality_score", 0.0)),
                "messages": [
                    {"role": "user", "content": f"<image>\n{IMAGE_CREATIVE_INSTRUCTION}"},
                    {"role": "assistant", "content": f"画面内容：{caption}\n创作提示：可围绕主体、场景和动作生成短视频镜头描述。"},
                ],
                "images": [row["image_path"]],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def _accepted_video_rows(frame_rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        if row.get("filter_status") == "accepted":
            grouped[str(row.get("video_id", ""))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: float(item.get("timestamp_seconds", 0.0)))
    return grouped


def export_video_creative_sft(frame_rows: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = _accepted_video_rows(frame_rows)
    with path.open("w", encoding="utf-8") as handle:
        for video_id, rows in grouped.items():
            images = [str(row["frame_path"]) for row in rows]
            timestamps = ", ".join(f"{float(row.get('timestamp_seconds', 0.0)):.1f}s" for row in rows)
            scores = [float(row.get("image_quality_score", 0.0)) for row in rows]
            quality_score = round(sum(scores) / len(scores), 4) if scores else 0.0
            payload = {
                "video_id": video_id,
                "task_type": "video_keyframes_to_creation_prompt",
                "quality_score": quality_score,
                "messages": [
                    {"role": "user", "content": f"<image>\n{VIDEO_CREATIVE_INSTRUCTION}"},
                    {
                        "role": "assistant",
                        "content": f"关键帧时间点：{timestamps}\n创作提示：根据关键帧变化组织短视频场景、主体动作和镜头节奏。",
                    },
                ],
                "images": images,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path
