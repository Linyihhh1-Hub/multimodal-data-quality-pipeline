from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.sample_gallery import write_sample_gallery
from src.storage.metadata_store import read_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a static HTML gallery from processed metadata.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="data/processed/sample_gallery.html")
    parser.add_argument("--title", default="Multimodal Data Sample Gallery")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    df = read_metadata(args.metadata)
    path = write_sample_gallery(df, args.output, title=args.title, limit=args.limit)
    print(f"Wrote sample gallery: {path}")


if __name__ == "__main__":
    main()
