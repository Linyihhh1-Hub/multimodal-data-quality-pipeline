from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.distributed.duckdb_quality_agg import aggregate_quality_metadata, write_quality_summary


ROOT = Path(".tmp/duckdb-quality-agg-tests")


def _reset_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def _write_parquet(name: str, rows: list[dict]) -> Path:
    _reset_dir(ROOT)
    path = ROOT / name
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_aggregate_quality_metadata_reads_parquet_and_computes_summary():
    path = _write_parquet(
        "quality.parquet",
        [
            {
                "sample_id": "a",
                "final_status": "accepted",
                "filter_reason": "",
                "final_quality_score": 0.9,
                "clip_score": 0.8,
                "is_duplicate_image": False,
                "source": "demo",
                "split": "train",
                "caption": "a clear image caption",
            },
            {
                "sample_id": "b",
                "final_status": "review",
                "filter_reason": "borderline_quality_score",
                "final_quality_score": 0.5,
                "clip_score": 0.4,
                "is_duplicate_image": True,
                "source": "demo",
                "split": "excluded",
                "caption": "short caption",
            },
            {
                "sample_id": "c",
                "final_status": "rejected",
                "filter_reason": "low_resolution;empty_caption",
                "final_quality_score": 0.1,
                "clip_score": 0.2,
                "is_duplicate_image": False,
                "source": "coco",
                "split": "excluded",
                "caption": "",
            },
        ],
    )

    summary = aggregate_quality_metadata(path, filter_topn=3)

    assert summary["total_samples"] == 3
    assert summary["status_counts"] == {"accepted": 1, "review": 1, "rejected": 1}
    assert summary["status_ratios"] == {"accepted": 0.3333, "review": 0.3333, "rejected": 0.3333}
    assert summary["avg_quality_score"] == 0.5
    assert summary["min_quality_score"] == 0.1
    assert summary["max_quality_score"] == 0.9
    assert summary["avg_clip_score"] == 0.4667
    assert summary["duplicate_count"] == 1
    assert summary["duplicate_ratio"] == 0.3333
    assert summary["source_distribution"] == {"demo": 2, "coco": 1}
    assert summary["split_distribution"] == {"excluded": 2, "train": 1}
    assert summary["avg_caption_length"] == 2.0
    assert summary["filter_reason_topn"] == [
        {"filter_reason": "borderline_quality_score", "samples": 1},
        {"filter_reason": "empty_caption", "samples": 1},
        {"filter_reason": "low_resolution", "samples": 1},
    ]
    assert summary["clip_score_buckets"]


def test_aggregate_quality_metadata_handles_missing_review_status():
    path = _write_parquet(
        "no_review.parquet",
        [
            {"status": "accepted", "final_quality_score": 0.9},
            {"status": "rejected", "final_quality_score": 0.2},
        ],
    )

    summary = aggregate_quality_metadata(path)

    assert summary["status_counts"]["accepted"] == 1
    assert summary["status_counts"]["review"] == 0
    assert summary["status_counts"]["rejected"] == 1


def test_aggregate_quality_metadata_handles_missing_clip_field():
    path = _write_parquet(
        "missing_clip.parquet",
        [
            {"final_status": "accepted", "final_quality_score": 0.9},
            {"final_status": "rejected", "final_quality_score": 0.2},
        ],
    )

    summary = aggregate_quality_metadata(path)

    assert summary["avg_clip_score"] is None
    assert summary["clip_score_buckets"] == []


def test_aggregate_quality_metadata_counts_filter_reasons_and_duplicate_group_size():
    path = _write_parquet(
        "reasons_duplicates.parquet",
        [
            {"filter_status": "accepted", "filter_reason": "low_resolution;text_too_short", "duplicate_group_size": 2},
            {"filter_status": "accepted", "filter_reason": "low_resolution", "duplicate_group_size": 1},
            {"filter_status": "rejected", "filter_reason": "", "duplicate_group_size": None},
        ],
    )

    summary = aggregate_quality_metadata(path, filter_topn=2)

    assert summary["duplicate_count"] == 1
    assert summary["duplicate_ratio"] == 0.3333
    assert summary["filter_reason_topn"] == [
        {"filter_reason": "low_resolution", "samples": 2},
        {"filter_reason": "text_too_short", "samples": 1},
    ]


def test_write_quality_summary_json_and_cli_are_usable():
    path = _write_parquet(
        "cli.parquet",
        [
            {"image_id": "1", "status": "accepted", "similarity_score": 0.8, "caption": "hello world"},
            {"image_id": "2", "status": "accepted", "similarity_score": 0.6, "caption": "short"},
        ],
    )
    output = ROOT / "summary.json"

    write_quality_summary(aggregate_quality_metadata(path), output)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["total_samples"] == 2

    cli_output = ROOT / "summary-cli.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.distributed.duckdb_quality_agg",
            "--input",
            str(path),
            "--output",
            str(cli_output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cli_summary = json.loads(cli_output.read_text(encoding="utf-8"))
    assert cli_summary["total_samples"] == 2
    assert cli_summary["avg_clip_score"] == 0.7
