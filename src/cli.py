from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.dataset_doctor import inspect_manifest
from src.analysis.quality_report import build_quality_report
from src.pipeline.quality_rules import _fallback_parse_yaml
from src.pipeline.export_creative_sft import export_image_creative_sft, export_video_creative_sft
from src.pipeline.run_pipeline import run_pipeline
from src.storage.metadata_store import read_metadata
from src.video.video_manifest import build_video_dataset


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except Exception:
        return _fallback_parse_yaml(text)


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _metadata_path(config: dict[str, Any]) -> Path:
    data = _section(config, "data")
    pipeline = _section(config, "pipeline")
    version = pipeline.get("version", "v1.0")
    return Path(data.get("processed_dir", "data/processed")) / f"processed_metadata_{version}.parquet"


def run_doctor(config: dict[str, Any]) -> dict[str, Any]:
    data = _section(config, "data")
    return inspect_manifest(
        manifest_path=data.get("manifest_path", "data/raw/manifest.jsonl"),
        raw_data_dir=data.get("raw_data_dir", "data/raw"),
    )


def run_configured_pipeline(config: dict[str, Any]):
    data = _section(config, "data")
    pipeline = _section(config, "pipeline")
    runs = _section(config, "runs")
    return run_pipeline(
        manifest_path=data.get("manifest_path", "data/raw/manifest.jsonl"),
        raw_data_dir=data.get("raw_data_dir", "data/raw"),
        processed_dir=data.get("processed_dir", "data/processed"),
        export_dir=data.get("export_dir", "data/exports"),
        version=pipeline.get("version", "v1.0"),
        use_clip=bool(pipeline.get("use_clip", False)),
        quality_rules_path=pipeline.get("quality_rules_path"),
        model_cache_dir=pipeline.get("model_cache_dir", "data/models/huggingface"),
        clip_batch_size=int(pipeline.get("clip_batch_size", 16)),
        run_id=runs.get("run_id"),
        runs_dir=runs.get("runs_dir", "outputs/runs"),
        archive_run=bool(runs.get("archive", True)),
        config_snapshot=config,
    )


def run_report(config: dict[str, Any]) -> Path:
    pipeline = _section(config, "pipeline")
    report = _section(config, "report")
    metadata = Path(report.get("metadata_path") or _metadata_path(config))
    output = Path(report.get("output", Path(metadata).with_name("quality_report.md")))
    df = read_metadata(metadata)
    text = build_quality_report(df, version=str(pipeline.get("version", "v1.0")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def run_video(config: dict[str, Any]) -> dict[str, Any]:
    video = _section(config, "video")
    data = _section(config, "data")
    return build_video_dataset(
        video_path=video.get("video_path", "data/raw/videos/demo_video.mp4"),
        frame_output_dir=video.get("frame_output_dir", "data/raw/video_frames/demo_video"),
        video_manifest_path=video.get("video_manifest_path", "data/processed/video_manifest.jsonl"),
        frame_manifest_path=video.get("frame_manifest_path", "data/processed/video_frame_manifest.jsonl"),
        raw_data_dir=video.get("raw_data_dir", data.get("raw_data_dir", "data/raw")),
        sample_interval_seconds=float(video.get("sample_interval_seconds", 1.0)),
        max_frames=video.get("max_frames"),
        source=video.get("source", "video_demo"),
    )


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_creative_sft(config: dict[str, Any]) -> dict[str, str]:
    creative = _section(config, "creative_sft")
    outputs: dict[str, str] = {}
    image_output = creative.get("image_output")
    if image_output:
        metadata = Path(creative.get("metadata_path") or _metadata_path(config))
        export_image_creative_sft(read_metadata(metadata), image_output)
        outputs["image_output"] = str(image_output)

    video_output = creative.get("video_output")
    if video_output:
        frame_manifest = creative.get(
            "video_frame_manifest_path",
            _section(config, "video").get("frame_manifest_path", "data/processed/video_frame_manifest.jsonl"),
        )
        export_video_creative_sft(_load_jsonl(frame_manifest), video_output)
        outputs["video_output"] = str(video_output)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Configuration-driven multimodal data pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["doctor", "run", "report", "video", "creative-sft"]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args(argv)

    config = load_pipeline_config(args.config)
    if args.command == "doctor":
        print(json.dumps(run_doctor(config), ensure_ascii=False, indent=2))
    elif args.command == "run":
        print(run_configured_pipeline(config))
    elif args.command == "report":
        print(f"Wrote quality report: {run_report(config)}")
    elif args.command == "video":
        result = run_video(config)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    elif args.command == "creative-sft":
        print(json.dumps(run_creative_sft(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
