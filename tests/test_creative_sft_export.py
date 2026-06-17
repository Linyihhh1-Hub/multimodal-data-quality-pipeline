import json
from pathlib import Path

import pandas as pd

from src.pipeline.export_creative_sft import export_image_creative_sft, export_video_creative_sft


def test_export_image_creative_sft_uses_accepted_samples(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "sample_id": "demo_1",
                "image_path": "images/demo_1.jpg",
                "caption": "A person rides a bicycle on a city street.",
                "filter_status": "accepted",
                "final_quality_score": 0.86,
            },
            {
                "sample_id": "demo_2",
                "image_path": "images/demo_2.jpg",
                "caption": "bad",
                "filter_status": "rejected",
                "final_quality_score": 0.2,
            },
        ]
    )
    output = tmp_path / "creative_image_sft.jsonl"

    export_image_creative_sft(df, output)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "demo_1"
    assert rows[0]["task_type"] == "image_to_video_prompt"
    assert rows[0]["images"] == ["images/demo_1.jpg"]
    assert "短视频" in rows[0]["messages"][0]["content"]
    assert "A person rides a bicycle" in rows[0]["messages"][1]["content"]


def test_export_video_creative_sft_groups_keyframes_by_video(tmp_path: Path):
    frame_rows = [
        {
            "sample_id": "demo_frame_000000",
            "video_id": "demo",
            "frame_path": "video_frames/demo/demo_frame_000000.jpg",
            "timestamp_seconds": 0.0,
            "filter_status": "accepted",
            "image_quality_score": 0.9,
        },
        {
            "sample_id": "demo_frame_000012",
            "video_id": "demo",
            "frame_path": "video_frames/demo/demo_frame_000012.jpg",
            "timestamp_seconds": 1.0,
            "filter_status": "accepted",
            "image_quality_score": 0.8,
        },
        {
            "sample_id": "demo_frame_000024",
            "video_id": "demo",
            "frame_path": "video_frames/demo/demo_frame_000024.jpg",
            "timestamp_seconds": 2.0,
            "filter_status": "rejected",
            "image_quality_score": 0.1,
        },
    ]
    output = tmp_path / "creative_video_sft.jsonl"

    export_video_creative_sft(frame_rows, output)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["video_id"] == "demo"
    assert rows[0]["task_type"] == "video_keyframes_to_creation_prompt"
    assert rows[0]["images"] == [
        "video_frames/demo/demo_frame_000000.jpg",
        "video_frames/demo/demo_frame_000012.jpg",
    ]
    assert rows[0]["quality_score"] == 0.85
