from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.review.apply_review_feedback import apply_review_feedback, apply_review_feedback_frame


ROOT = Path(".tmp/review-feedback-tests")


def _reset_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def _write_metadata(name: str, rows: list[dict]) -> Path:
    _reset_dir(ROOT)
    path = ROOT / name
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _write_feedback(name: str, rows: list[dict]) -> Path:
    path = ROOT / name
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def test_review_samples_can_be_accepted_rejected_and_kept_review():
    metadata = pd.DataFrame(
        [
            {"sample_id": "a", "final_status": "review", "caption": "a"},
            {"sample_id": "b", "final_status": "review", "caption": "b"},
            {"sample_id": "c", "final_status": "review", "caption": "c"},
        ]
    )
    feedback = pd.DataFrame(
        [
            {"sample_id": "a", "decision": "accepted", "reviewer": "human_001", "reason": "ok", "review_time": "2026-06-26T10:00:00"},
            {"sample_id": "b", "decision": "rejected", "reviewer": "human_001", "reason": "bad", "review_time": "2026-06-26T10:01:00"},
            {"sample_id": "c", "decision": "keep_review", "reviewer": "human_001", "reason": "unclear", "review_time": "2026-06-26T10:02:00"},
        ]
    )

    reviewed, summary = apply_review_feedback_frame(metadata, feedback, version="v1.2")

    assert reviewed.set_index("sample_id")["final_status"].to_dict() == {"a": "accepted", "b": "rejected", "c": "review"}
    assert reviewed.set_index("sample_id")["status_before_review"].to_dict() == {"a": "review", "b": "review", "c": "review"}
    assert reviewed.set_index("sample_id")["status_after_review"].to_dict() == {"a": "accepted", "b": "rejected", "c": "review"}
    assert reviewed.set_index("sample_id")["review_applied"].to_dict() == {"a": True, "b": True, "c": True}
    assert reviewed["version"].tolist() == ["v1.2", "v1.2", "v1.2"]
    assert summary == {
        "total_review_feedback": 3,
        "matched_feedback": 3,
        "applied_feedback": 3,
        "accepted_after_review": 1,
        "rejected_after_review": 1,
        "kept_review": 1,
        "unmatched_feedback": 0,
        "skipped_non_review": 0,
        "invalid_feedback": 0,
    }


def test_non_review_samples_are_not_overwritten_by_default():
    metadata = pd.DataFrame(
        [
            {"sample_id": "accepted_row", "status": "accepted"},
            {"sample_id": "rejected_row", "status": "rejected"},
            {"sample_id": "review_row", "status": "review"},
        ]
    )
    feedback = pd.DataFrame(
        [
            {"sample_id": "accepted_row", "decision": "rejected"},
            {"sample_id": "rejected_row", "decision": "accepted"},
            {"sample_id": "review_row", "decision": "accepted"},
        ]
    )

    reviewed, summary = apply_review_feedback_frame(metadata, feedback, version="v1.2")

    assert reviewed.set_index("sample_id")["status"].to_dict() == {
        "accepted_row": "accepted",
        "rejected_row": "rejected",
        "review_row": "accepted",
    }
    assert reviewed.set_index("sample_id")["review_applied"].to_dict() == {
        "accepted_row": False,
        "rejected_row": False,
        "review_row": True,
    }
    assert summary["applied_feedback"] == 1
    assert summary["skipped_non_review"] == 2


def test_unmatched_feedback_and_missing_optional_fields_do_not_crash():
    metadata = pd.DataFrame([{"image_id": "img_1", "status": "review"}])
    feedback = pd.DataFrame(
        [
            {"image_id": "img_1", "decision": "accepted"},
            {"image_id": "missing", "decision": "rejected"},
        ]
    )

    reviewed, summary = apply_review_feedback_frame(metadata, feedback, version="v1.2")

    row = reviewed.iloc[0]
    assert row["status"] == "accepted"
    assert row["review_decision"] == "accepted"
    assert pd.isna(row["reviewer"])
    assert pd.isna(row["review_reason"])
    assert pd.isna(row["review_time"])
    assert summary["unmatched_feedback"] == 1
    assert summary["applied_feedback"] == 1


def test_apply_review_feedback_writes_new_version_parquet():
    metadata_path = _write_metadata(
        "metadata.parquet",
        [
            {"sample_id": "a", "filter_status": "review", "split": "excluded"},
            {"sample_id": "b", "filter_status": "accepted", "split": "train"},
        ],
    )
    feedback_path = _write_feedback(
        "feedback.jsonl",
        [
            {"sample_id": "a", "decision": "accepted", "reviewer": "human_001"},
            {"sample_id": "b", "decision": "rejected", "reviewer": "human_001"},
            {"sample_id": "missing", "decision": "accepted"},
        ],
    )
    output_path = ROOT / "metadata_reviewed.parquet"

    summary = apply_review_feedback(metadata_path, feedback_path, output_path, version="v1.2")
    reviewed = pd.read_parquet(output_path)

    assert output_path.exists()
    assert reviewed.set_index("sample_id").loc["a", "filter_status"] == "accepted"
    assert reviewed.set_index("sample_id").loc["b", "filter_status"] == "accepted"
    assert summary["total_review_feedback"] == 3
    assert summary["applied_feedback"] == 1
    assert summary["unmatched_feedback"] == 1
    assert summary["skipped_non_review"] == 1


def test_review_feedback_cli_prints_summary_and_writes_output():
    metadata_path = _write_metadata("cli_metadata.parquet", [{"sample_id": "a", "status": "review"}])
    feedback_path = _write_feedback("cli_feedback.jsonl", [{"sample_id": "a", "decision": "rejected"}])
    output_path = ROOT / "cli_reviewed.parquet"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.review.apply_review_feedback",
            "--metadata",
            str(metadata_path),
            "--feedback",
            str(feedback_path),
            "--output",
            str(output_path),
            "--version",
            "v1.2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    assert "total_review_feedback: 1" in result.stdout
    assert "rejected_after_review: 1" in result.stdout
