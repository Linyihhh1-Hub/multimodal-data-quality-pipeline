import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from src.video.frame_extractor import extract_keyframes
from src.video.metadata import probe_video
from src.video.video_manifest import build_video_dataset, build_video_frame_manifest


def _write_demo_video(path: Path, frame_count: int = 6, fps: float = 2.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((240, 320, 3), (40 + index * 20, 90, 160), dtype=np.uint8)
        cv2.putText(frame, f"frame-{index}", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def test_extract_keyframes_writes_frames_with_timestamps(tmp_path: Path):
    video_path = tmp_path / "demo.mp4"
    output_dir = tmp_path / "frames"
    _write_demo_video(video_path, frame_count=6, fps=2.0)

    frames = extract_keyframes(video_path, output_dir, sample_interval_seconds=1.0)

    assert [frame.frame_index for frame in frames] == [0, 2, 4]
    assert [frame.timestamp_seconds for frame in frames] == [0.0, 1.0, 2.0]
    assert all(frame.frame_path.exists() for frame in frames)


def test_extract_keyframes_respects_max_frames(tmp_path: Path):
    video_path = tmp_path / "demo.mp4"
    output_dir = tmp_path / "frames"
    _write_demo_video(video_path, frame_count=10, fps=2.0)

    frames = extract_keyframes(video_path, output_dir, sample_interval_seconds=1.0, max_frames=3)

    assert [frame.frame_index for frame in frames] == [0, 2, 4]


def test_probe_video_returns_core_metadata(tmp_path: Path):
    video_path = tmp_path / "demo.mp4"
    _write_demo_video(video_path, frame_count=6, fps=2.0)

    metadata = probe_video(video_path)

    assert metadata["video_id"] == "demo"
    assert metadata["frame_count"] == 6
    assert metadata["fps"] == 2.0
    assert metadata["duration_seconds"] == 3.0
    assert metadata["width"] == 320
    assert metadata["height"] == 240
    assert metadata["file_size_bytes"] > 0


def test_build_video_frame_manifest_adds_quality_fields(tmp_path: Path):
    video_path = tmp_path / "demo.mp4"
    raw_dir = tmp_path / "raw"
    frame_dir = raw_dir / "video_frames" / "demo"
    manifest_path = tmp_path / "video_frames.jsonl"
    _write_demo_video(video_path, frame_count=4, fps=2.0)

    rows = build_video_frame_manifest(
        video_path=video_path,
        frame_output_dir=frame_dir,
        manifest_path=manifest_path,
        raw_data_dir=raw_dir,
        sample_interval_seconds=1.0,
        source="unit_test",
        prompt_template="Describe key frame at {timestamp_seconds:.1f}s.",
    )

    written_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert rows == written_rows
    assert len(rows) == 2
    assert rows[0]["sample_id"] == "demo_frame_000000"
    assert rows[0]["video_id"] == "demo"
    assert rows[0]["frame_path"] == "video_frames/demo/demo_frame_000000.jpg"
    assert rows[0]["frame_index"] == 0
    assert rows[0]["caption"] == "Describe key frame at 0.0s."
    assert rows[0]["image_valid"] is True
    assert rows[0]["width"] == 320
    assert rows[0]["height"] == 240
    assert "image_quality_score" in rows[0]


def test_build_video_dataset_writes_video_and_frame_manifests(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    video_path = raw_dir / "videos" / "demo.mp4"
    video_path.parent.mkdir(parents=True)
    frame_dir = raw_dir / "video_frames" / "demo"
    processed_dir = tmp_path / "processed"
    _write_demo_video(video_path, frame_count=6, fps=2.0)

    result = build_video_dataset(
        video_path=video_path,
        frame_output_dir=frame_dir,
        video_manifest_path=processed_dir / "video_manifest.jsonl",
        frame_manifest_path=processed_dir / "video_frame_manifest.jsonl",
        raw_data_dir=raw_dir,
        sample_interval_seconds=1.0,
        max_frames=2,
        source="unit_test",
    )

    assert result["video_manifest_path"] == str(processed_dir / "video_manifest.jsonl")
    assert result["frame_manifest_path"] == str(processed_dir / "video_frame_manifest.jsonl")
    assert result["frame_count"] == 2
    assert len(result["video_rows"]) == 1
    video_row = result["video_rows"][0]
    assert video_row["video_id"] == "demo"
    assert video_row["video_path"] == "videos/demo.mp4"
    assert video_row["valid_frame_count"] == 2
    assert video_row["rejected_frame_count"] == 0
    assert video_row["video_quality_score"] > 0

    frame_rows = [json.loads(line) for line in (processed_dir / "video_frame_manifest.jsonl").read_text().splitlines()]
    assert len(frame_rows) == 2
    assert frame_rows[0]["frame_path"] == "video_frames/demo/demo_frame_000000.jpg"


def test_process_video_demo_script_runs_from_project_root(tmp_path: Path):
    video_path = tmp_path / "demo.mp4"
    frame_dir = tmp_path / "frames"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_demo_video(video_path, frame_count=4, fps=2.0)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/process_video_demo.py",
            "--video",
            str(video_path),
            "--frame-output-dir",
            str(frame_dir),
            "--manifest",
            str(manifest_path),
            "--interval",
            "1.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert manifest_path.exists()
