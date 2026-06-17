import pandas as pd

from src.dashboard.metrics import build_dashboard_summary, explode_filter_reasons, load_jsonl_rows, score_columns


def test_build_dashboard_summary_counts_statuses_and_rates():
    df = pd.DataFrame(
        [
            {"filter_status": "accepted", "final_quality_score": 0.9, "image_text_similarity": 0.8},
            {"filter_status": "accepted", "final_quality_score": 0.8, "image_text_similarity": 0.7},
            {"filter_status": "review", "final_quality_score": 0.5, "image_text_similarity": 0.4},
            {"filter_status": "rejected", "final_quality_score": 0.1, "image_text_similarity": 0.2},
        ]
    )

    summary = build_dashboard_summary(df)

    assert summary["total"] == 4
    assert summary["accepted"] == 2
    assert summary["review"] == 1
    assert summary["rejected"] == 1
    assert summary["acceptance_rate"] == 0.5
    assert summary["avg_quality_score"] == 0.575
    assert summary["avg_similarity"] == 0.525


def test_explode_filter_reasons_splits_semicolon_values():
    df = pd.DataFrame({"filter_reason": ["low_resolution;text_too_short", "", None, "low_resolution"]})

    reasons = explode_filter_reasons(df)

    assert reasons.to_dict() == {"low_resolution": 2, "text_too_short": 1}


def test_score_columns_returns_available_numeric_scores():
    df = pd.DataFrame(
        {
            "image_quality_score": [1.0],
            "text_quality_score": [0.9],
            "caption": ["demo"],
        }
    )

    assert score_columns(df) == ["image_quality_score", "text_quality_score"]


def test_load_jsonl_rows_parses_json_objects(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"video_id": "demo", "video_quality_score": 1.0}\n', encoding="utf-8")

    df = load_jsonl_rows(path)

    assert df.to_dict("records") == [{"video_id": "demo", "video_quality_score": 1.0}]
