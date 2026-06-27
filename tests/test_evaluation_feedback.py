from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.evaluation_feedback import (
    build_error_type_summary,
    build_second_round_manifest,
    join_evaluation_feedback,
    load_evaluation_errors,
    run_evaluation_feedback,
)


def test_load_evaluation_errors_reads_csv_and_jsonl(tmp_path: Path):
    csv_path = tmp_path / "errors.csv"
    csv_path.write_text(
        "sample_id,prediction,ground_truth,error_type,error_reason\n"
        "s1,a cat,a dog,semantic_mismatch,wrong object\n",
        encoding="utf-8",
    )
    jsonl_path = tmp_path / "errors.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "sample_id": "s2",
                "prediction": "red car",
                "ground_truth": "blue car",
                "error_type": "attribute_error",
                "error_reason": "wrong color",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    csv_errors = load_evaluation_errors(csv_path)
    jsonl_errors = load_evaluation_errors(jsonl_path)

    assert csv_errors.iloc[0]["sample_id"] == "s1"
    assert jsonl_errors.iloc[0]["error_type"] == "attribute_error"


def test_join_evaluation_feedback_normalizes_quality_fields():
    errors = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "prediction": "a cat",
                "ground_truth": "a dog",
                "error_type": "semantic_mismatch",
                "error_reason": "wrong object",
            },
            {
                "sample_id": "missing",
                "prediction": "x",
                "ground_truth": "y",
                "error_type": "unknown",
                "error_reason": "not in metadata",
            },
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "image_filter_status": "accepted",
                "text_filter_status": "accepted",
                "image_text_similarity": 0.88,
                "duplicate_group_size": 2,
                "perceptual_hash": "hash-a",
                "filter_status": "accepted",
                "review_decision": "accepted",
            }
        ]
    )

    joined = join_evaluation_feedback(errors, metadata)
    row = joined.set_index("sample_id").loc["s1"]
    missing = joined.set_index("sample_id").loc["missing"]

    assert row["image_quality_status"] == "accepted"
    assert row["text_quality_status"] == "accepted"
    assert row["clip_score"] == 0.88
    assert row["duplicate_group_id"] == "hash-a"
    assert row["final_quality_label"] == "accepted"
    assert row["human_review_label"] == "accepted"
    assert row["quality_metadata_matched"] is True
    assert missing["quality_metadata_matched"] is False


def test_error_type_summary_counts_quality_feature_distribution():
    joined = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "error_type": "semantic_mismatch",
                "image_quality_status": "accepted",
                "text_quality_status": "accepted",
                "clip_score": 0.88,
                "duplicate_group_id": "hash-a",
                "final_quality_label": "accepted",
                "human_review_label": "accepted",
            },
            {
                "sample_id": "s2",
                "error_type": "semantic_mismatch",
                "image_quality_status": "rejected",
                "text_quality_status": "accepted",
                "clip_score": 0.2,
                "duplicate_group_id": "",
                "final_quality_label": "rejected",
                "human_review_label": "",
            },
        ]
    )

    summary = build_error_type_summary(joined)

    assert summary["semantic_mismatch"]["total_errors"] == 2
    assert summary["semantic_mismatch"]["final_quality_label_counts"]["accepted"] == 1
    assert summary["semantic_mismatch"]["final_quality_label_counts"]["rejected"] == 1
    assert summary["semantic_mismatch"]["duplicate_group_errors"] == 1


def test_second_round_manifest_recommends_actions_and_runner_writes_outputs(tmp_path: Path):
    errors_path = tmp_path / "errors.csv"
    errors_path.write_text(
        "sample_id,prediction,ground_truth,error_type,error_reason\n"
        "s1,a cat,a dog,semantic_mismatch,wrong object\n"
        "s2,bad caption,clean caption,quality_defect,empty generated detail\n"
        "s3,rare sign,rare street sign,coverage_gap,model misses rare object\n"
        "missing,x,y,unknown,not found\n",
        encoding="utf-8",
    )
    metadata_path = tmp_path / "metadata.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "image_filter_status": "accepted",
                "text_filter_status": "accepted",
                "image_text_similarity": 0.86,
                "filter_status": "accepted",
            },
            {
                "sample_id": "s2",
                "image_filter_status": "rejected",
                "text_filter_status": "accepted",
                "image_text_similarity": 0.3,
                "filter_status": "rejected",
            },
            {
                "sample_id": "s3",
                "image_filter_status": "accepted",
                "text_filter_status": "accepted",
                "image_text_similarity": 0.91,
                "filter_status": "accepted",
            },
        ]
    ).to_parquet(metadata_path, index=False)

    result = run_evaluation_feedback(
        errors_path=errors_path,
        metadata_path=metadata_path,
        output_dir=tmp_path / "feedback",
    )
    manifest_rows = [json.loads(line) for line in result["second_round_manifest"].read_text(encoding="utf-8").splitlines()]
    action_by_id = {row["sample_id"]: row["recommendation"] for row in manifest_rows}

    assert action_by_id == {
        "s1": "keep",
        "s2": "remove",
        "s3": "augment_candidate",
        "missing": "review",
    }
    assert result["report"].exists()
    assert "## Error Type: semantic_mismatch" in result["report"].read_text(encoding="utf-8")

    direct_manifest = build_second_round_manifest(pd.read_json(result["second_round_manifest"], lines=True))
    assert len(direct_manifest) == 4
