from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _status_count(df: pd.DataFrame, status: str) -> int:
    if "filter_status" not in df.columns:
        return 0
    return int((df["filter_status"] == status).sum())


def _duplicate_count(df: pd.DataFrame) -> int:
    if "is_duplicate_image" not in df.columns:
        return 0
    return int(df["is_duplicate_image"].sum())


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_project_showcase(
    heuristic_metadata: str | Path,
    clip_v1_metadata: str | Path,
    clip_v11_metadata: str | Path,
    tag_distribution: str | Path,
    version_compare: str | Path,
) -> str:
    heuristic = pd.read_parquet(heuristic_metadata)
    clip_v1 = pd.read_parquet(clip_v1_metadata)
    clip_v11 = pd.read_parquet(clip_v11_metadata)
    tags = _read_json(tag_distribution)
    compare = _read_json(version_compare)

    top_tags = ", ".join(f"{item['tag']}: {item['count']}" for item in tags.get("top_tags", [])[:10])

    return "\n".join(
        [
            "# Project Showcase",
            "",
            "## Real COCO Run",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Total COCO captions | {len(heuristic)} |",
            f"| Heuristic accepted | {_status_count(heuristic, 'accepted')} |",
            f"| Heuristic rejected | {_status_count(heuristic, 'rejected')} |",
            f"| Duplicate image samples | {_duplicate_count(heuristic)} |",
            f"| CLIP v1.0 accepted | {_status_count(clip_v1, 'accepted')} |",
            f"| CLIP v1.0 rejected | {_status_count(clip_v1, 'rejected')} |",
            f"| CLIP v1.1 accepted | {_status_count(clip_v11, 'accepted')} |",
            f"| CLIP v1.1 review | {_status_count(clip_v11, 'review')} |",
            f"| CLIP v1.1 rejected | {_status_count(clip_v11, 'rejected')} |",
            f"| v1.0 to v1.1 status changed | {int(compare.get('status_changed_samples', 0))} |",
            "",
            "## Tag Distribution",
            "",
            f"Top caption tags: {top_tags}",
            "",
            "## Local Artifacts",
            "",
            "- `data/processed_coco/quality_report_coco_v1.0.md`",
            "- `data/processed_coco/sample_gallery_coco_v1.0.html`",
            "- `data/processed_clip_coco/quality_report_coco_clip_v1.1.md`",
            "- `data/processed_clip_coco/sample_gallery_coco_clip_v1.1.html`",
            "- `data/processed_clip_coco/version_compare_coco_clip_v1.0_v1.1.json`",
            "",
            "## Interview Talking Point",
            "",
            "The project now demonstrates a full AI-data workflow: data readiness checks, real COCO ingestion, rule-based quality checks, model-assisted CLIP scoring, duplicate-image analysis, caption tag distribution, versioned rule iteration, and export to train/eval/SFT JSONL formats.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a README-ready project showcase from local artifacts.")
    parser.add_argument("--heuristic-metadata", default="data/processed_coco/processed_metadata_coco_v1.0.parquet")
    parser.add_argument("--clip-v1-metadata", default="data/processed_clip_coco/processed_metadata_coco_clip_v1.0.parquet")
    parser.add_argument("--clip-v11-metadata", default="data/processed_clip_coco/processed_metadata_coco_clip_v1.1.parquet")
    parser.add_argument("--tag-distribution", default="data/processed_coco/tag_distribution_coco_v1.0.json")
    parser.add_argument("--version-compare", default="data/processed_clip_coco/version_compare_coco_clip_v1.0_v1.1.json")
    parser.add_argument("--output", default="docs/project_showcase.md")
    args = parser.parse_args()

    markdown = build_project_showcase(
        heuristic_metadata=args.heuristic_metadata,
        clip_v1_metadata=args.clip_v1_metadata,
        clip_v11_metadata=args.clip_v11_metadata,
        tag_distribution=args.tag_distribution,
        version_compare=args.version_compare,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote project showcase: {output_path}")


if __name__ == "__main__":
    main()
