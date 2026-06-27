import json
from pathlib import Path

import pandas as pd
from PIL import Image

from src.pipeline.run_pipeline import run_pipeline
from src.utils.run_manager import RunManager, create_run_id


def test_run_manager_writes_manifest_and_summary(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "image_path": "images/1.jpg",
                "caption": "A clear caption.",
                "filter_status": "accepted",
                "filter_reason": "",
                "split": "train",
                "final_quality_score": 0.91,
                "image_text_similarity": 0.82,
            },
            {
                "sample_id": "s2",
                "image_path": "images/2.jpg",
                "caption": "",
                "filter_status": "rejected",
                "filter_reason": "empty_caption",
                "split": "val",
                "final_quality_score": 0.1,
                "image_text_similarity": 0.0,
            },
        ]
    )
    manager = RunManager(tmp_path / "outputs" / "runs", run_id="unit-run")

    outputs = manager.archive(
        scored=df,
        config={"pipeline": {"version": "vtest", "use_clip": False}},
        input_paths={"manifest": "data/raw/manifest.jsonl"},
        output_paths={"metadata": "data/processed/processed_metadata_vtest.parquet"},
        quality_rules={"score": {"accept_threshold": 0.75}},
        scorer_backend="heuristic",
    )

    assert manager.run_dir == tmp_path / "outputs" / "runs" / "unit-run"
    assert outputs["config_snapshot"].exists()
    assert outputs["manifest"].exists()
    assert outputs["quality_report"].exists()
    assert outputs["filtered_samples"].exists()
    assert outputs["run_summary"].exists()

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["run_id"] == "unit-run"
    assert manifest["samples"]["total"] == 2
    assert manifest["samples"]["accepted"] == 1
    assert manifest["samples"]["rejected"] == 1
    assert manifest["quality_rules"]["score"]["accept_threshold"] == 0.75

    quality_report = json.loads(outputs["quality_report"].read_text(encoding="utf-8"))
    assert quality_report["status_counts"]["accepted"] == 1
    assert quality_report["filter_reason_counts"]["empty_caption"] == 1

    filtered_lines = outputs["filtered_samples"].read_text(encoding="utf-8").splitlines()
    assert len(filtered_lines) == 1
    assert json.loads(filtered_lines[0])["sample_id"] == "s1"

    summary = outputs["run_summary"].read_text(encoding="utf-8")
    assert "# Pipeline Run unit-run" in summary
    assert "| Accepted samples | 1 |" in summary


def test_create_run_id_is_unique_and_filesystem_safe():
    first = create_run_id()
    second = create_run_id()

    assert first != second
    assert "/" not in first
    assert "\\" not in first
    assert ":" not in first


def test_run_pipeline_archives_run_outputs(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    image_dir = raw_dir / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (320, 240), color=(80, 120, 160)).save(image_dir / "1.jpg")
    Image.new("RGB", (100, 100), color=(80, 80, 80)).save(image_dir / "2.jpg")
    manifest = raw_dir / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                '{"image_id":"1","image_path":"images/1.jpg","caption":"A person rides a bicycle on a street.","source":"demo"}',
                '{"image_id":"2","image_path":"images/2.jpg","caption":"","source":"demo"}',
            ]
        ),
        encoding="utf-8",
    )

    result = run_pipeline(
        manifest_path=manifest,
        raw_data_dir=raw_dir,
        processed_dir=tmp_path / "processed",
        export_dir=tmp_path / "exports",
        version="vtest",
        use_clip=False,
        run_id="pipeline-run",
        runs_dir=tmp_path / "outputs" / "runs",
    )

    assert result.run_id == "pipeline-run"
    assert result.run_dir == tmp_path / "outputs" / "runs" / "pipeline-run"
    assert (result.run_dir / "config_snapshot.yaml").exists()
    assert (result.run_dir / "manifest.json").exists()
    assert (result.run_dir / "quality_report.json").exists()
    assert (result.run_dir / "filtered_samples.jsonl").exists()
    assert (result.run_dir / "run_summary.md").exists()
