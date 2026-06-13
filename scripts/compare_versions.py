from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage.dataset_version import compare_versions
from src.storage.metadata_store import read_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two processed metadata versions.")
    parser.add_argument("--old", required=True, help="Old metadata parquet/csv path")
    parser.add_argument("--new", required=True, help="New metadata parquet/csv path")
    parser.add_argument("--output", default="data/processed/version_compare.json")
    args = parser.parse_args()

    old_df = read_metadata(args.old)
    new_df = read_metadata(args.new)
    summary = compare_versions(old_df, new_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
