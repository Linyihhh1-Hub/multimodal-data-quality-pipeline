from pathlib import Path

import pandas as pd

from src.analysis.quality_report import build_quality_report


def test_build_quality_report_contains_key_metrics(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "caption": "A strong caption.",
                "filter_status": "accepted",
                "filter_reason": "",
                "image_text_similarity": 0.82,
                "final_quality_score": 0.88,
            },
            {
                "sample_id": "b",
                "caption": "bad",
                "filter_status": "rejected",
                "filter_reason": "short_caption;low_quality_score",
                "image_text_similarity": 0.2,
                "final_quality_score": 0.35,
            },
        ]
    )

    report = build_quality_report(df, version="vtest")

    assert "# Multimodal Data Quality Report - vtest" in report
    assert "Total samples | 2" in report
    assert "Acceptance rate | 50.00%" in report
    assert "short_caption" in report
    assert "low_quality_score" in report
    assert "b | rejected" in report
