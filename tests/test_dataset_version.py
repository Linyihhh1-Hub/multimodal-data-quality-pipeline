import pandas as pd

from src.storage.dataset_version import compare_versions


def test_compare_versions_reports_status_and_score_changes():
    old = pd.DataFrame(
        [
            {"sample_id": "a", "filter_status": "accepted", "final_quality_score": 0.8},
            {"sample_id": "b", "filter_status": "rejected", "final_quality_score": 0.3},
        ]
    )
    new = pd.DataFrame(
        [
            {"sample_id": "a", "filter_status": "review", "final_quality_score": 0.6},
            {"sample_id": "c", "filter_status": "accepted", "final_quality_score": 0.9},
        ]
    )

    summary = compare_versions(old, new)

    assert summary["old_total_samples"] == 2
    assert summary["new_total_samples"] == 2
    assert summary["added_samples"] == 1
    assert summary["removed_samples"] == 1
    assert summary["status_changed_samples"] == 1
    assert summary["old_acceptance_rate"] == 0.5
    assert summary["new_acceptance_rate"] == 0.5
    assert summary["avg_quality_score_delta"] == 0.2
