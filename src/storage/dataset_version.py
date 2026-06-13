from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def build_version_record(version: str, total_samples: int, accepted_samples: int) -> dict:
    return {
        "version": version,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": int(total_samples),
        "accepted_samples": int(accepted_samples),
        "acceptance_rate": round(accepted_samples / total_samples, 4) if total_samples else 0.0,
    }


def _acceptance_rate(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    return round(float((df["filter_status"] == "accepted").sum() / len(df)), 4)


def compare_versions(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    old_ids = set(old["sample_id"])
    new_ids = set(new["sample_id"])
    shared_ids = old_ids & new_ids

    old_status = old.set_index("sample_id")["filter_status"].to_dict()
    new_status = new.set_index("sample_id")["filter_status"].to_dict()
    status_changed = sum(1 for sample_id in shared_ids if old_status[sample_id] != new_status[sample_id])

    old_avg = float(old["final_quality_score"].mean()) if len(old) else 0.0
    new_avg = float(new["final_quality_score"].mean()) if len(new) else 0.0

    return {
        "old_total_samples": int(len(old)),
        "new_total_samples": int(len(new)),
        "added_samples": int(len(new_ids - old_ids)),
        "removed_samples": int(len(old_ids - new_ids)),
        "shared_samples": int(len(shared_ids)),
        "status_changed_samples": int(status_changed),
        "old_acceptance_rate": _acceptance_rate(old),
        "new_acceptance_rate": _acceptance_rate(new),
        "old_avg_quality_score": round(old_avg, 4),
        "new_avg_quality_score": round(new_avg, 4),
        "avg_quality_score_delta": round(new_avg - old_avg, 4),
    }
