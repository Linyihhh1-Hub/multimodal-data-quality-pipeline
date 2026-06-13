from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.quality_report import build_quality_report
from src.storage.metadata_store import read_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown quality report from processed metadata.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", default="data/processed/quality_report.md")
    args = parser.parse_args()

    df = read_metadata(args.metadata)
    report = build_quality_report(df, version=args.version)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote quality report: {output_path}")


if __name__ == "__main__":
    main()
