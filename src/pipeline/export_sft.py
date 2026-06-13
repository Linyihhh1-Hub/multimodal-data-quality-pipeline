from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _iter_accepted_split(df: pd.DataFrame, split: str):
    rows = df[(df["filter_status"] == "accepted") & (df["split"] == split)]
    for _, row in rows.iterrows():
        yield row


def export_caption_jsonl(df: pd.DataFrame, output_path: str | Path, split: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in _iter_accepted_split(df, split):
            payload = {"image": row["image_path"], "caption": row["caption"]}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def export_sft_jsonl(df: pd.DataFrame, output_path: str | Path, split: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in _iter_accepted_split(df, split):
            payload = {
                "messages": [
                    {"role": "user", "content": "<image>\nPlease describe this image."},
                    {"role": "assistant", "content": row["caption"]},
                ],
                "images": [row["image_path"]],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def export_status_jsonl(df: pd.DataFrame, output_path: str | Path, status: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = df[df["filter_status"] == status]
    with path.open("w", encoding="utf-8") as handle:
        for _, row in rows.iterrows():
            payload = row.to_dict()
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return path
