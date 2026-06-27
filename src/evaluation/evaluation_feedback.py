from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.storage.metadata_store import read_metadata


REQUIRED_ERROR_COLUMNS = {"sample_id", "prediction", "ground_truth", "error_type", "error_reason"}
AUGMENT_ERROR_TYPES = {"coverage_gap", "rare_case", "domain_gap", "long_tail_error"}


def _first_existing(columns: pd.Index | list[str], candidates: tuple[str, ...]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _safe_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_safe(record: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in record.items():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if pd.isna(value):
            value = None
        safe[key] = value
    return safe


def load_evaluation_errors(path: str | Path) -> pd.DataFrame:
    error_path = Path(path)
    if not error_path.exists():
        raise FileNotFoundError(f"Evaluation errors file not found: {error_path}")

    suffix = error_path.suffix.lower()
    if suffix == ".csv":
        errors = pd.read_csv(error_path)
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in error_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        errors = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported evaluation errors format: {suffix}")

    missing = REQUIRED_ERROR_COLUMNS.difference(errors.columns)
    if missing:
        raise ValueError(f"Evaluation errors missing required columns: {sorted(missing)}")
    errors = errors.copy()
    errors["sample_id"] = errors["sample_id"].astype(str)
    return errors


def _duplicate_group_value(row: pd.Series) -> str:
    if "duplicate_group_id" in row and not pd.isna(row["duplicate_group_id"]):
        return str(row["duplicate_group_id"])
    group_size = _safe_float(row.get("duplicate_group_size"))
    if group_size is not None and group_size > 1:
        return _safe_string(row.get("perceptual_hash"))
    return ""


def join_evaluation_feedback(errors: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in metadata.columns:
        raise ValueError("Quality metadata must contain sample_id for evaluation feedback joins.")

    left = errors.copy()
    right = metadata.copy()
    left["sample_id"] = left["sample_id"].astype(str)
    right["sample_id"] = right["sample_id"].astype(str)
    joined = left.merge(right, on="sample_id", how="left", suffixes=("", "_quality"))

    image_status_col = _first_existing(joined.columns, ("image_quality_status", "image_filter_status"))
    text_status_col = _first_existing(joined.columns, ("text_quality_status", "text_filter_status"))
    clip_col = _first_existing(joined.columns, ("clip_score", "similarity_score", "image_text_similarity"))
    final_label_col = _first_existing(joined.columns, ("final_quality_label", "filter_status", "final_status", "status"))
    human_label_col = _first_existing(
        joined.columns,
        ("human_review_label", "status_after_review", "review_decision", "review_label"),
    )

    joined["image_quality_status"] = joined[image_status_col].map(_safe_string) if image_status_col else ""
    joined["text_quality_status"] = joined[text_status_col].map(_safe_string) if text_status_col else ""
    joined["clip_score"] = joined[clip_col].map(_safe_float) if clip_col else None
    joined["duplicate_group_id"] = joined.apply(_duplicate_group_value, axis=1)
    joined["final_quality_label"] = joined[final_label_col].map(_safe_string) if final_label_col else ""
    joined["human_review_label"] = joined[human_label_col].map(_safe_string) if human_label_col else ""
    joined["quality_metadata_matched"] = joined[final_label_col].notna().map(bool) if final_label_col else False
    joined["quality_metadata_matched"] = joined["quality_metadata_matched"].astype(object)
    return joined


def _value_counts(series: pd.Series) -> dict[str, int]:
    cleaned = series.fillna("").astype(str)
    cleaned = cleaned[cleaned != ""]
    return {str(key): int(value) for key, value in cleaned.value_counts().sort_index().items()}


def build_error_type_summary(joined: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for error_type, group in joined.groupby("error_type", dropna=False):
        clip_scores = pd.to_numeric(group.get("clip_score"), errors="coerce")
        duplicate_ids = group.get("duplicate_group_id", pd.Series(dtype=str)).fillna("").astype(str)
        summary[str(error_type)] = {
            "total_errors": int(len(group)),
            "image_quality_status_counts": _value_counts(group.get("image_quality_status", pd.Series(dtype=str))),
            "text_quality_status_counts": _value_counts(group.get("text_quality_status", pd.Series(dtype=str))),
            "final_quality_label_counts": _value_counts(group.get("final_quality_label", pd.Series(dtype=str))),
            "human_review_label_counts": _value_counts(group.get("human_review_label", pd.Series(dtype=str))),
            "average_clip_score": float(clip_scores.mean()) if clip_scores.notna().any() else None,
            "low_clip_score_errors": int((clip_scores < 0.55).sum()),
            "duplicate_group_errors": int((duplicate_ids != "").sum()),
            "matched_quality_metadata": int(group.get("quality_metadata_matched", pd.Series(dtype=bool)).sum()),
        }
    return summary


def _recommendation(row: pd.Series) -> tuple[str, str]:
    image_status = _safe_string(row.get("image_quality_status"))
    text_status = _safe_string(row.get("text_quality_status"))
    final_label = _safe_string(row.get("final_quality_label"))
    human_label = _safe_string(row.get("human_review_label"))
    clip_score = _safe_float(row.get("clip_score"))
    error_type = _safe_string(row.get("error_type"))

    if row.get("quality_metadata_matched") is False:
        return "review", "evaluation error sample is missing quality metadata"
    if "rejected" in {image_status, text_status, final_label, human_label}:
        return "remove", "existing quality or human review label is rejected"
    if "review" in {final_label, human_label}:
        return "review", "existing quality label requires human review"
    if clip_score is not None and clip_score < 0.55:
        return "review", "low image-text similarity on an evaluation error sample"
    if error_type in AUGMENT_ERROR_TYPES:
        return "augment_candidate", "high-quality error suggests missing coverage in training data"
    return "keep", "quality metadata does not indicate a data defect"


def build_second_round_manifest(joined: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in joined.iterrows():
        recommendation, reason = _recommendation(row)
        rows.append(
            {
                "sample_id": _safe_string(row.get("sample_id")),
                "prediction": _safe_string(row.get("prediction")),
                "ground_truth": _safe_string(row.get("ground_truth")),
                "error_type": _safe_string(row.get("error_type")),
                "error_reason": _safe_string(row.get("error_reason")),
                "image_quality_status": _safe_string(row.get("image_quality_status")),
                "text_quality_status": _safe_string(row.get("text_quality_status")),
                "clip_score": _safe_float(row.get("clip_score")),
                "duplicate_group_id": _safe_string(row.get("duplicate_group_id")),
                "final_quality_label": _safe_string(row.get("final_quality_label")),
                "human_review_label": _safe_string(row.get("human_review_label")),
                "recommendation": recommendation,
                "recommendation_reason": reason,
            }
        )
    return rows


def build_feedback_report(summary: dict[str, dict[str, Any]]) -> str:
    total_errors = sum(item["total_errors"] for item in summary.values())
    lines = [
        "# Evaluation Feedback Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total evaluation errors | {total_errors} |",
        f"| Error types | {len(summary)} |",
        "",
    ]

    for error_type, stats in summary.items():
        lines.extend(
            [
                f"## Error Type: {error_type}",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Total errors | {stats['total_errors']} |",
                f"| Average clip score | {stats['average_clip_score'] if stats['average_clip_score'] is not None else 'n/a'} |",
                f"| Low clip score errors | {stats['low_clip_score_errors']} |",
                f"| Duplicate group errors | {stats['duplicate_group_errors']} |",
                f"| Matched quality metadata | {stats['matched_quality_metadata']} |",
                "",
                "### Final Quality Labels",
                "",
                "| Label | Count |",
                "| --- | ---: |",
            ]
        )
        final_counts = stats["final_quality_label_counts"]
        if final_counts:
            for label, count in final_counts.items():
                lines.append(f"| {label} | {count} |")
        else:
            lines.append("| none | 0 |")
        lines.extend(["", "### Image Quality Status", "", "| Status | Count |", "| --- | ---: |"])
        for label, count in (stats["image_quality_status_counts"] or {"none": 0}).items():
            lines.append(f"| {label} | {count} |")
        lines.extend(["", "### Text Quality Status", "", "| Status | Count |", "| --- | ---: |"])
        for label, count in (stats["text_quality_status_counts"] or {"none": 0}).items():
            lines.append(f"| {label} | {count} |")
        lines.append("")

    return "\n".join(lines)


def write_second_round_manifest(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
    return path


def run_evaluation_feedback(
    errors_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    errors = load_evaluation_errors(errors_path)
    metadata = read_metadata(metadata_path)
    joined = join_evaluation_feedback(errors, metadata)
    summary = build_error_type_summary(joined)
    second_round_rows = build_second_round_manifest(joined)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "evaluation_feedback_report.md"
    manifest_path = output / "second_round_manifest.jsonl"
    report_path.write_text(build_feedback_report(summary), encoding="utf-8")
    write_second_round_manifest(second_round_rows, manifest_path)
    return {"report": report_path, "second_round_manifest": manifest_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze model evaluation errors against quality metadata.")
    parser.add_argument("--errors", required=True, help="Evaluation errors CSV or JSONL path.")
    parser.add_argument("--metadata", required=True, help="Quality metadata Parquet or CSV path.")
    parser.add_argument("--output-dir", default="data/processed/evaluation_feedback", help="Output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run_evaluation_feedback(
        errors_path=args.errors,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
