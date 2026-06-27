import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_demo_manifest(raw_dir: Path) -> Path:
    image_dir = raw_dir / "images"
    image_dir.mkdir(parents=True)
    for name, color in [("demo_1.jpg", (90, 130, 180)), ("demo_2.jpg", (170, 120, 90))]:
        Image.new("RGB", (320, 240), color=color).save(image_dir / name)
    manifest = raw_dir / "manifest.jsonl"
    rows = [
        {"image_id": "demo_1", "image_path": "images/demo_1.jpg", "caption": "A person rides a bicycle.", "source": "test"},
        {"image_id": "demo_2", "image_path": "images/demo_2.jpg", "caption": "A dog sits indoors.", "source": "test"},
    ]
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return manifest


def _write_pipeline_config(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    export_dir = tmp_path / "exports"
    manifest = _write_demo_manifest(raw_dir)
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        f"""
data:
  manifest_path: {manifest.as_posix()}
  raw_data_dir: {raw_dir.as_posix()}
  processed_dir: {processed_dir.as_posix()}
  export_dir: {export_dir.as_posix()}

pipeline:
  version: cli_test
  use_clip: false
  clip_batch_size: 2

runs:
  archive: true
  runs_dir: {(tmp_path / "outputs" / "runs").as_posix()}

report:
  output: {(processed_dir / "quality_report.md").as_posix()}

creative_sft:
  image_output: {(export_dir / "creative_image_sft.jsonl").as_posix()}
  video_frame_manifest_path: {(processed_dir / "video_frame_manifest.jsonl").as_posix()}
  video_output: {(export_dir / "creative_video_sft.jsonl").as_posix()}
""".strip(),
        encoding="utf-8",
    )
    return config


def test_cli_doctor_run_and_report_use_pipeline_config(tmp_path: Path):
    config = _write_pipeline_config(tmp_path)
    processed_dir = tmp_path / "processed"
    export_dir = tmp_path / "exports"

    doctor = subprocess.run(
        [sys.executable, "-m", "src.cli", "doctor", "--config", str(config)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert doctor.returncode == 0, doctor.stderr
    assert '"ready_for_pipeline": true' in doctor.stdout

    run = subprocess.run(
        [sys.executable, "-m", "src.cli", "run", "--config", str(config)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert (processed_dir / "processed_metadata_cli_test.parquet").exists()
    assert (export_dir / "train_sft.jsonl").exists()

    report = subprocess.run(
        [sys.executable, "-m", "src.cli", "report", "--config", str(config)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert report.returncode == 0, report.stderr
    assert (processed_dir / "quality_report.md").exists()

    (processed_dir / "video_frame_manifest.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "demo_frame_000000",
                "video_id": "demo",
                "frame_path": "video_frames/demo/demo_frame_000000.jpg",
                "timestamp_seconds": 0.0,
                "filter_status": "accepted",
                "image_quality_score": 0.9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    creative = subprocess.run(
        [sys.executable, "-m", "src.cli", "creative-sft", "--config", str(config)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert creative.returncode == 0, creative.stderr
    assert (export_dir / "creative_image_sft.jsonl").exists()
    assert (export_dir / "creative_video_sft.jsonl").exists()
