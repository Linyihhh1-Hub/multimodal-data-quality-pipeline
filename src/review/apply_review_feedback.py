from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ID_COLUMNS = ("sample_id", "image_id")
STATUS_COLUMNS = ("final_status", "status", "filter_status")
VALID_DECISIONS = {"accepted", "rejected", "review", "keep_review"}
OUTPUT_COLUMNS = [
    "review_decision",
    "reviewer",
    "review_reason",
    "review_time",
    "status_before_review",
    "status_after_review",
    "review_applied",
]


def _first_existing(columns: pd.Index | list[str], candidates: tuple[str, ...]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _normalize_decision(decision: object) -> str | None:
    value = "" if decision is None else str(decision).strip()
    if value not in VALID_DECISIONS:
        return None
    return "review" if value == "keep_review" else value


def load_review_feedback(path: str | Path) -> pd.DataFrame:
    feedback_path = Path(path)
    if not feedback_path.exists():
        raise FileNotFoundError(f"Review feedback file not found: {feedback_path}")
    rows: list[dict[str, Any]] = []
    for line in feedback_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _empty_summary(total_feedback: int = 0) -> dict[str, int]:
    return {
        "total_review_feedback": total_feedback,
        "matched_feedback": 0,
        "applied_feedback": 0,
        "accepted_after_review": 0,
        "rejected_after_review": 0,
        "kept_review": 0,
        "unmatched_feedback": 0,
        "skipped_non_review": 0,
        "invalid_feedback": 0,
    }


def apply_review_feedback_frame(
    metadata: pd.DataFrame,
    feedback: pd.DataFrame,
    version: str,
    overwrite_non_review: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    result = metadata.copy()
    total_feedback = len(feedback)
    summary = _empty_summary(total_feedback)

    metadata_id_column = _first_existing(result.columns, ID_COLUMNS)
    feedback_id_column = _first_existing(feedback.columns, ID_COLUMNS)
    status_column = _first_existing(result.columns, STATUS_COLUMNS)

    for column in OUTPUT_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result["review_applied"] = False
    result["version"] = version

    if metadata_id_column is None or feedback_id_column is None or status_column is None or feedback.empty:
        summary["unmatched_feedback"] = total_feedback
        return result, summary

    feedback_by_id: dict[str, dict[str, Any]] = {}
    invalid_feedback = 0
    for record in feedback.to_dict("records"):
        sample_id = record.get(feedback_id_column)
        decision = record.get("decision")
        if sample_id is None or _normalize_decision(decision) is None:
            invalid_feedback += 1
            continue
        feedback_by_id[str(sample_id)] = record
    summary["invalid_feedback"] = invalid_feedback

    metadata_ids = result[metadata_id_column].astype(str)
    matched_ids = set(metadata_ids).intersection(feedback_by_id)
    summary["matched_feedback"] = len(matched_ids)
    summary["unmatched_feedback"] = max(0, total_feedback - len(matched_ids) - invalid_feedback)

    for row_index, sample_id in metadata_ids.items():
        record = feedback_by_id.get(sample_id)
        if record is None:
            continue

        before_status = result.at[row_index, status_column]
        before_status = "" if pd.isna(before_status) else str(before_status)
        normalized_decision = _normalize_decision(record.get("decision"))
        if normalized_decision is None:
            continue

        result.at[row_index, "review_decision"] = record.get("decision")
        result.at[row_index, "reviewer"] = record.get("reviewer", pd.NA)
        result.at[row_index, "review_reason"] = record.get("reason", pd.NA)
        result.at[row_index, "review_time"] = record.get("review_time", pd.NA)
        result.at[row_index, "status_before_review"] = before_status

        can_apply = overwrite_non_review or before_status == "review"
        after_status = normalized_decision if can_apply else before_status
        result.at[row_index, "status_after_review"] = after_status

        if not can_apply:
            summary["skipped_non_review"] += 1
            continue

        result.at[row_index, status_column] = after_status
        result.at[row_index, "review_applied"] = True
        summary["applied_feedback"] += 1
        if after_status == "accepted":
            summary["accepted_after_review"] += 1
        elif after_status == "rejected":
            summary["rejected_after_review"] += 1
        elif after_status == "review":
            summary["kept_review"] += 1

    return result, summary


def apply_review_feedback(
    metadata_path: str | Path,
    feedback_path: str | Path,
    output_path: str | Path,
    version: str,
    overwrite_non_review: bool = False,
) -> dict[str, int]:
    metadata = pd.read_parquet(metadata_path)
    feedback = load_review_feedback(feedback_path)
    reviewed, summary = apply_review_feedback_frame(
        metadata,
        feedback,
        version=version,
        overwrite_non_review=overwrite_non_review,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    reviewed.to_parquet(output, index=False)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply human review decisions to quality metadata.")
    parser.add_argument("--metadata", required=True, help="Input quality metadata Parquet path.")
    parser.add_argument("--feedback", required=True, help="Human review decisions JSONL path.")
    parser.add_argument("--output", required=True, help="Output reviewed metadata Parquet path.")
    parser.add_argument("--version", required=True, help="New dataset version label, for example v1.2.")
    parser.add_argument(
        "--overwrite-non-review",
        action="store_true",
        help="Also apply feedback to accepted/rejected rows. Disabled by default.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = apply_review_feedback(
        metadata_path=args.metadata,
        feedback_path=args.feedback,
        output_path=args.output,
        version=args.version,
        overwrite_non_review=args.overwrite_non_review,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
