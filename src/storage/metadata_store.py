from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_metadata(df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False, encoding="utf-8")
        return fallback
    return path


def read_metadata(path: str | Path) -> pd.DataFrame:
    metadata_path = Path(path)
    if metadata_path.suffix.lower() == ".parquet":
        return pd.read_parquet(metadata_path)
    if metadata_path.suffix.lower() == ".csv":
        return pd.read_csv(metadata_path)
    raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")
