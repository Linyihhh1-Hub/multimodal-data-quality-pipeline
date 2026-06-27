import pandas as pd
from pathlib import Path

from src.dashboard.metrics import (
    ALERT_THRESHOLDS,
    build_governance_summary,
    build_quality_alerts,
    build_quality_pipeline,
    build_dashboard_summary,
    build_status_breakdown,
    caption_lengths,
    diagnostic_samples,
    export_inventory,
    explode_filter_reasons,
    filter_reason_topn,
    load_jsonl_rows,
    localize_status,
    review_queue,
    score_columns,
)


def _reset_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


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


def test_build_governance_summary_counts_statuses_rates_and_duplicates():
    export_dir = Path(".tmp/dashboard-metrics-tests/governance-exports")
    _reset_dir(export_dir)
    (export_dir / "train.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    df = pd.DataFrame(
        [
            {"final_status": "accepted", "final_quality_score": 0.9, "clip_score": 0.8, "is_duplicate_image": False},
            {"final_status": "accepted", "final_quality_score": 0.8, "clip_score": 0.7, "is_duplicate_image": True},
            {"final_status": "review", "final_quality_score": 0.5, "clip_score": 0.4, "is_duplicate_image": False},
            {"final_status": "rejected", "final_quality_score": 0.1, "clip_score": 0.2, "is_duplicate_image": False},
        ]
    )

    summary = build_governance_summary(df, export_dir)

    assert summary["total"] == 4
    assert summary["accepted"] == 2
    assert summary["review"] == 1
    assert summary["rejected"] == 1
    assert summary["accepted_rate"] == 0.5
    assert summary["review_rate"] == 0.25
    assert summary["rejected_rate"] == 0.25
    assert summary["avg_quality_score"] == 0.575
    assert summary["avg_clip_score"] == 0.525
    assert summary["near_duplicate_rate"] == 0.25
    assert summary["export_asset_count"] == 1


def test_build_governance_summary_handles_missing_review_and_clip_score():
    df = pd.DataFrame(
        [
            {"status": "accepted", "final_quality_score": 0.9},
            {"status": "rejected", "final_quality_score": 0.2},
        ]
    )

    summary = build_governance_summary(df)

    assert summary["accepted"] == 1
    assert summary["review"] == 0
    assert summary["rejected"] == 1
    assert summary["avg_clip_score"] is None


def test_build_governance_summary_uses_similarity_score_as_clip_score():
    df = pd.DataFrame(
        [
            {"status": "accepted", "similarity_score": 0.9},
            {"status": "review", "similarity_score": 0.7},
        ]
    )

    summary = build_governance_summary(df)

    assert summary["avg_clip_score"] == 0.8
    assert summary["clip_score_column"] == "similarity_score"


def test_explode_filter_reasons_splits_semicolon_values():
    df = pd.DataFrame({"filter_reason": ["low_resolution;text_too_short", "", None, "low_resolution"]})

    reasons = explode_filter_reasons(df)

    assert reasons.to_dict() == {"low_resolution": 2, "text_too_short": 1}


def test_filter_reason_topn_returns_labeled_dataframe():
    df = pd.DataFrame({"filter_reason": ["low_resolution;text_too_short", "", None, "low_resolution"]})

    topn = filter_reason_topn(df, limit=1)

    assert topn.to_dict("records") == [{"filter_reason": "low_resolution", "samples": 2}]


def test_caption_lengths_prefers_existing_word_count_and_falls_back_to_caption_text():
    with_word_count = pd.DataFrame({"caption_word_count": [3, None], "caption": ["ignored", "fallback text"]})
    without_word_count = pd.DataFrame({"caption": ["A short caption", "", None]})

    assert caption_lengths(with_word_count).tolist() == [3, 0]
    assert caption_lengths(without_word_count).tolist() == [3, 0, 0]


def test_score_columns_returns_available_numeric_scores():
    df = pd.DataFrame(
        {
            "image_quality_score": [1.0],
            "text_quality_score": [0.9],
            "similarity_score": [0.8],
            "caption": ["demo"],
        }
    )

    assert score_columns(df) == ["image_quality_score", "text_quality_score", "similarity_score"]


def test_load_jsonl_rows_parses_json_objects():
    path = Path(".tmp/dashboard-metrics-tests/rows.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"video_id": "demo", "video_quality_score": 1.0}\n', encoding="utf-8")

    df = load_jsonl_rows(path)

    assert df.to_dict("records") == [{"video_id": "demo", "video_quality_score": 1.0}]


def test_build_status_breakdown_returns_counts_and_rates():
    df = pd.DataFrame(
        {
            "filter_status": ["accepted", "accepted", "review", "rejected"],
            "final_quality_score": [0.9, 0.8, 0.5, 0.1],
        }
    )

    breakdown = build_status_breakdown(df)

    assert breakdown.to_dict("records") == [
        {"status": "accepted", "status_label": "通过", "samples": 2, "rate": 0.5},
        {"status": "review", "status_label": "待复核", "samples": 1, "rate": 0.25},
        {"status": "rejected", "status_label": "拒绝", "samples": 1, "rate": 0.25},
    ]


def test_build_quality_pipeline_uses_available_fields_and_export_rows():
    export_dir = Path(".tmp/dashboard-metrics-tests/pipeline-exports")
    _reset_dir(export_dir)
    (export_dir / "train.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (export_dir / "eval.jsonl").write_text("{}\n", encoding="utf-8")
    (export_dir / "train_sft.jsonl").write_text("{}\n{}\n{}\n", encoding="utf-8")
    df = pd.DataFrame(
        [
            {"image_valid": True, "text_valid": True, "filter_status": "accepted"},
            {"image_valid": True, "text_valid": True, "filter_status": "review"},
            {"image_valid": False, "text_valid": True, "filter_status": "rejected"},
            {"image_valid": True, "text_valid": False, "filter_status": "rejected"},
        ]
    )

    pipeline = build_quality_pipeline(df, export_dir)

    assert pipeline[["stage", "samples"]].to_dict("records") == [
        {"stage": "原始样本", "samples": 4},
        {"stage": "图片可读 / 图片质量通过", "samples": 3},
        {"stage": "文本有效", "samples": 3},
        {"stage": "图文一致性达标或进入复核", "samples": 2},
        {"stage": "Accepted", "samples": 1},
        {"stage": "Review", "samples": 1},
        {"stage": "Rejected", "samples": 2},
        {"stage": "已导出 Train", "samples": 2},
        {"stage": "已导出 Eval", "samples": 1},
        {"stage": "已导出 SFT", "samples": 3},
    ]


def test_build_quality_alerts_returns_specific_reason_and_action():
    df = pd.DataFrame(
        [
            {"filter_status": "review", "image_text_similarity": 0.4},
            {"filter_status": "rejected", "image_text_similarity": 0.3},
            {"filter_status": "rejected", "image_text_similarity": 0.2},
            {"filter_status": "accepted", "image_text_similarity": 0.5},
        ]
    )

    thresholds = {**ALERT_THRESHOLDS, "review_ratio_max": 0.2}
    alerts = build_quality_alerts(build_governance_summary(df), thresholds)

    assert [alert["alert"] for alert in alerts] == [
        "Accepted 比例低于阈值",
        "Review 样本占比过高",
        "Rejected 样本占比过高",
        "平均图文相似度偏低",
    ]
    assert "CLIP 阈值过高" in alerts[0]["possible_causes"]
    assert "导出 review_samples.jsonl" in alerts[1]["action"]


def test_localize_status_falls_back_to_unknown_values():
    assert localize_status("accepted") == "通过"
    assert localize_status("needs_manual_check") == "needs_manual_check"


def test_review_queue_prioritizes_non_accepted_low_quality_samples():
    df = pd.DataFrame(
        [
            {"sample_id": "a", "filter_status": "accepted", "final_quality_score": 0.9},
            {"sample_id": "b", "filter_status": "review", "final_quality_score": 0.4},
            {"sample_id": "c", "filter_status": "rejected", "final_quality_score": 0.2},
        ]
    )

    queue = review_queue(df)

    assert queue["sample_id"].tolist() == ["b"]


def test_diagnostic_samples_returns_low_quality_rows_with_fallback_status_labels():
    df = pd.DataFrame(
        [
            {"sample_id": "high", "filter_status": "accepted", "caption": "good", "final_quality_score": 0.9},
            {"sample_id": "low", "filter_status": "rejected", "caption": "bad", "final_quality_score": 0.1},
            {"sample_id": "mid", "filter_status": "review", "caption": "maybe", "final_quality_score": 0.4},
        ]
    )

    samples = diagnostic_samples(df, limit=2)

    assert samples["sample_id"].tolist() == ["low", "mid"]
    assert samples["status_label"].tolist() == ["拒绝", "待复核"]


def test_export_inventory_lists_jsonl_files():
    path = Path(".tmp/dashboard-metrics-tests/exports")
    _reset_dir(path)
    (path / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (path / "notes.txt").write_text("ignore", encoding="utf-8")

    inventory = export_inventory(path)

    rows = inventory.set_index("file")
    assert bool(rows.loc["train.jsonl", "exists"]) is True
    assert rows.loc["train.jsonl", "rows"] == 1
    assert rows.loc["train.jsonl", "purpose"] == "普通训练集"
    assert bool(rows.loc["review_samples.jsonl", "exists"]) is False
    assert rows.loc["review_samples.jsonl", "status_label"] == "不存在"
