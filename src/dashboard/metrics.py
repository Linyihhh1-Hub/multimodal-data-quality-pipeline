from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _count_status(df: pd.DataFrame, status: str) -> int:
    if "filter_status" not in df.columns:
        return 0
    return int((df["filter_status"] == status).sum())


def _mean_or_zero(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    return round(float(pd.to_numeric(df[column], errors="coerce").fillna(0).mean()), 4)


def build_dashboard_summary(df: pd.DataFrame) -> dict[str, float | int]:
    total = len(df)
    accepted = _count_status(df, "accepted")
    review = _count_status(df, "review")
    rejected = _count_status(df, "rejected")
    return {
        "total": total,
        "accepted": accepted,
        "review": review,
        "rejected": rejected,
        "acceptance_rate": round(accepted / total, 4) if total else 0.0,
        "review_rate": round(review / total, 4) if total else 0.0,
        "rejection_rate": round(rejected / total, 4) if total else 0.0,
        "avg_quality_score": _mean_or_zero(df, "final_quality_score"),
        "avg_similarity": _mean_or_zero(df, "image_text_similarity"),
    }


def explode_filter_reasons(df: pd.DataFrame) -> pd.Series:
    if "filter_reason" not in df.columns:
        return pd.Series(dtype="int64")
    return (
        df["filter_reason"]
        .fillna("")
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
    )


def score_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "image_quality_score",
        "text_quality_score",
        "image_text_similarity",
        "final_quality_score",
    ]
    return [column for column in candidates if column in df.columns and pd.api.types.is_numeric_dtype(df[column])]


def top_tags_from_column(df: pd.DataFrame, column: str = "tags", limit: int = 20) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="int64")
    return (
        df[column]
        .fillna("")
        .astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.split(",")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(limit)
    )


def load_jsonl_rows(path: str | Path) -> pd.DataFrame:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return pd.DataFrame()
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return pd.DataFrame(rows)
