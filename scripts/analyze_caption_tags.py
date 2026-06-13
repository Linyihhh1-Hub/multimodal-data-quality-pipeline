from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.keyword_tags import summarize_tag_distribution
from src.storage.metadata_store import read_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze lightweight caption keyword/tag distribution.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="data/processed/tag_distribution.json")
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    df = read_metadata(args.metadata)
    summary = summarize_tag_distribution(df, top_k=args.top_k)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
