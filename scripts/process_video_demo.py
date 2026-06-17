from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.video_manifest import build_video_frame_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract video keyframes and build a quality-aware frame manifest.")
    parser.add_argument("--video", default="data/raw/videos/demo_video.mp4")
    parser.add_argument("--frame-output-dir", default="data/raw/video_frames/demo_video")
    parser.add_argument("--manifest", default="data/processed/video_frame_manifest.jsonl")
    parser.add_argument("--raw-data-dir", default="data/raw")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--source", default="video_demo")
    args = parser.parse_args()

    rows = build_video_frame_manifest(
        video_path=Path(args.video),
        frame_output_dir=Path(args.frame_output_dir),
        manifest_path=Path(args.manifest),
        raw_data_dir=Path(args.raw_data_dir),
        sample_interval_seconds=args.interval,
        source=args.source,
        prompt_template="Key frame from {video_id} at {timestamp_seconds:.1f}s for visual content review.",
    )
    print(f"Wrote {len(rows)} keyframe rows: {args.manifest}")


if __name__ == "__main__":
    main()
