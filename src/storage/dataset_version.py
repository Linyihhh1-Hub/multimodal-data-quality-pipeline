from __future__ import annotations

from datetime import datetime, timezone


def build_version_record(version: str, total_samples: int, accepted_samples: int) -> dict:
    return {
        "version": version,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": int(total_samples),
        "accepted_samples": int(accepted_samples),
        "acceptance_rate": round(accepted_samples / total_samples, 4) if total_samples else 0.0,
    }
