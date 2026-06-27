from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


LEAKAGE_KEYS = ("sample_id", "image_id", "image_path", "perceptual_hash", "duplicate_group_id")


def _clean_key_values(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _row_value(row: pd.Series, field: str) -> Any:
    if field not in row.index:
        return None
    return _json_safe(row[field])


def _example_from_rows(key: str, value: str, train_row: pd.Series, eval_row: pd.Series) -> dict[str, Any]:
    return {
        "leakage_key": key,
        "leakage_value": value,
        "train_sample_id": _row_value(train_row, "sample_id") or _row_value(train_row, "image_id"),
        "eval_sample_id": _row_value(eval_row, "sample_id") or _row_value(eval_row, "image_id"),
        "train_image_path": _row_value(train_row, "image_path"),
        "eval_image_path": _row_value(eval_row, "image_path"),
        "train_split": _row_value(train_row, "split"),
        "eval_split": _row_value(eval_row, "split"),
    }


def _empty_report(checked_keys: list[str], missing_keys: list[str]) -> dict[str, Any]:
    return {
        "has_leakage": False,
        "total_leakage_count": 0,
        "leakage_by_key": {key: 0 for key in LEAKAGE_KEYS},
        "checked_keys": checked_keys,
        "missing_keys": missing_keys,
        "leakage_examples": [],
    }


def detect_split_leakage(
    df: pd.DataFrame,
    train_split: str = "train",
    eval_splits: Sequence[str] = ("val", "eval"),
    max_examples: int = 20,
) -> dict[str, Any]:
    """Detect train/eval leakage by identifiers and visual duplicate keys.

    Counts are based on unique key values that appear in both the train split and
    any evaluation split. Examples keep one train/eval pair per leaked value so a
    repeated COCO-style image with many captions does not explode the report.
    """

    checked_keys = [key for key in LEAKAGE_KEYS if key in df.columns]
    missing_keys = [key for key in LEAKAGE_KEYS if key not in df.columns]
    if df.empty or "split" not in df.columns:
        return _empty_report(checked_keys, missing_keys)

    split_values = _clean_key_values(df["split"])
    train_df = df.loc[split_values == train_split].copy()
    eval_df = df.loc[split_values.isin({str(split) for split in eval_splits})].copy()

    leakage_by_key = {key: 0 for key in LEAKAGE_KEYS}
    examples: list[dict[str, Any]] = []

    for key in LEAKAGE_KEYS:
        if key not in df.columns:
            continue

        train_work = train_df.copy()
        eval_work = eval_df.copy()
        train_work["_leakage_value"] = _clean_key_values(train_work[key])
        eval_work["_leakage_value"] = _clean_key_values(eval_work[key])
        train_work = train_work[train_work["_leakage_value"] != ""]
        eval_work = eval_work[eval_work["_leakage_value"] != ""]

        if train_work.empty or eval_work.empty:
            continue

        train_by_value = train_work.groupby("_leakage_value", sort=True).first()
        eval_by_value = eval_work.groupby("_leakage_value", sort=True).first()
        leaked_values = sorted(set(train_by_value.index).intersection(eval_by_value.index), key=str)
        leakage_by_key[key] = len(leaked_values)

        for value in leaked_values:
            if len(examples) >= max_examples:
                break
            examples.append(_example_from_rows(key, value, train_by_value.loc[value], eval_by_value.loc[value]))

    total_leakage_count = sum(leakage_by_key.values())
    return {
        "has_leakage": total_leakage_count > 0,
        "total_leakage_count": total_leakage_count,
        "leakage_by_key": leakage_by_key,
        "checked_keys": checked_keys,
        "missing_keys": missing_keys,
        "leakage_examples": examples,
    }


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_leakage_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_metadata(args: argparse.Namespace) -> pd.DataFrame:
    if args.metadata:
        return pd.read_parquet(args.metadata)

    if args.train and args.eval_path:
        train_records = load_jsonl_records(args.train)
        eval_records = load_jsonl_records(args.eval_path)
        train_df = pd.DataFrame(train_records)
        eval_df = pd.DataFrame(eval_records)
        train_df["split"] = args.train_split
        eval_df["split"] = args.eval_splits[0] if args.eval_splits else "eval"
        return pd.concat([train_df, eval_df], ignore_index=True)

    raise ValueError("需要提供 --metadata，或同时提供 --train 和 --eval。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect train/eval data leakage in multimodal metadata.")
    parser.add_argument("--metadata", help="Input quality metadata Parquet path.")
    parser.add_argument("--train", help="Optional train JSONL path.")
    parser.add_argument("--eval", dest="eval_path", help="Optional eval JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--train-split", default="train", help="Train split label in metadata.")
    parser.add_argument(
        "--eval-splits",
        nargs="*",
        default=["val", "eval"],
        help="Evaluation split labels in metadata.",
    )
    parser.add_argument("--max-examples", type=int, default=20, help="Maximum leakage examples in the report.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        df = _load_metadata(args)
    except Exception as exc:
        parser.error(str(exc))

    report = detect_split_leakage(
        df,
        train_split=args.train_split,
        eval_splits=tuple(args.eval_splits),
        max_examples=args.max_examples,
    )
    write_leakage_report(report, args.output)

    print(f"has_leakage: {report['has_leakage']}")
    print(f"total_leakage_count: {report['total_leakage_count']}")
    for key, count in report["leakage_by_key"].items():
        print(f"{key}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
