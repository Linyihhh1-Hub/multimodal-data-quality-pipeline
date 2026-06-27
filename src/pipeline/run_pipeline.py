from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.manifest_builder import load_manifest
from src.models.clip_scorer import ClipScorer
from src.pipeline.build_dataset import assign_splits
from src.pipeline.export_sft import export_caption_jsonl, export_sft_jsonl, export_status_jsonl
from src.pipeline.quality_rules import QualityRules, load_quality_rules
from src.quality.image_quality import assess_image
from src.quality.duplicate_detector import annotate_duplicate_hashes, compute_perceptual_hash
from src.quality.quality_scorer import combine_quality
from src.quality.text_quality import assess_caption
from src.storage.metadata_store import write_metadata
from src.utils.run_manager import RunManager


@dataclass
class PipelineResult:
    metadata_path: Path
    train_jsonl: Path
    val_jsonl: Path
    eval_jsonl: Path
    train_sft_jsonl: Path
    review_jsonl: Path
    rejected_jsonl: Path
    total_samples: int
    accepted_samples: int
    review_samples: int
    rejected_samples: int
    scorer_backend: str
    run_id: str | None = None
    run_dir: Path | None = None
    run_artifacts: dict[str, Path] | None = None


def _resolve_image_path(raw_data_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return raw_data_dir / path


def run_pipeline(
    manifest_path: str | Path,
    raw_data_dir: str | Path,
    processed_dir: str | Path,
    export_dir: str | Path,
    version: str = "v1.0",
    use_clip: bool = True,
    quality_rules_path: str | Path | None = None,
    model_cache_dir: str | Path | None = "data/models/huggingface",
    clip_batch_size: int = 16,
    run_id: str | None = None,
    runs_dir: str | Path = "outputs/runs",
    archive_run: bool = True,
    config_snapshot: dict[str, Any] | None = None,
) -> PipelineResult:
    raw_dir = Path(raw_data_dir)
    processed = Path(processed_dir)
    exports = Path(export_dir)
    df = load_manifest(manifest_path)
    scorer = ClipScorer(use_clip=use_clip, model_cache_dir=model_cache_dir)
    rules = load_quality_rules(quality_rules_path)

    prepared: list[dict] = []
    for _, row in df.iterrows():
        absolute_image_path = _resolve_image_path(raw_dir, str(row["image_path"]))
        image_result = assess_image(absolute_image_path, **rules.image)
        text_kwargs = dict(rules.text)
        if isinstance(text_kwargs.get("sensitive_words"), list):
            text_kwargs["sensitive_words"] = set(text_kwargs["sensitive_words"])
        text_result = assess_caption(row["caption"], **text_kwargs)
        prepared.append(
            {
                "row": row.to_dict(),
                "absolute_image_path": absolute_image_path,
                "image_result": image_result,
                "text_result": text_result,
            }
        )

    similarities = scorer.score_batch(
        [item["absolute_image_path"] for item in prepared],
        [str(item["row"]["caption"]) for item in prepared],
        batch_size=clip_batch_size,
    )

    records: list[dict] = []
    hash_cache: dict[str, str] = {}
    for item, similarity in zip(prepared, similarities):
        row = item["row"]
        absolute_image_path = item["absolute_image_path"]
        image_result = item["image_result"]
        text_result = item["text_result"]
        resolved_key = str(absolute_image_path)
        if resolved_key not in hash_cache:
            hash_cache[resolved_key] = compute_perceptual_hash(absolute_image_path)
        combined = combine_quality(
            image_status=image_result["filter_status"],
            text_status=text_result["filter_status"],
            image_reasons=image_result["filter_reasons"],
            text_reasons=text_result["filter_reasons"],
            image_quality_score=image_result["image_quality_score"],
            text_quality_score=text_result["text_quality_score"],
            image_text_similarity=similarity,
            accept_threshold=rules.score.get("accept_threshold", 0.75),
            review_threshold=rules.score.get("review_threshold", 0.55),
        )
        records.append(
            {
                **row,
                "image_path": str(row["image_path"]),
                "resolved_image_path": str(absolute_image_path),
                "image_valid": image_result["image_valid"],
                "width": image_result["width"],
                "height": image_result["height"],
                "file_size_bytes": image_result["file_size_bytes"],
                "aspect_ratio": image_result["aspect_ratio"],
                "blur_variance": image_result["blur_variance"],
                "brightness": image_result["brightness"],
                "image_quality_score": image_result["image_quality_score"],
                "text_valid": text_result["text_valid"],
                "caption_word_count": text_result["caption_word_count"],
                "text_quality_score": text_result["text_quality_score"],
                "image_filter_status": image_result["filter_status"],
                "text_filter_status": text_result["filter_status"],
                "image_filter_reasons": ";".join(image_result["filter_reasons"]),
                "text_filter_reasons": ";".join(text_result["filter_reasons"]),
                "image_text_similarity": similarity,
                "final_quality_score": combined["final_quality_score"],
                "filter_status": combined["filter_status"],
                "filter_reason": combined["filter_reason"],
                "version": version,
                "perceptual_hash": hash_cache[resolved_key],
            }
        )

    records = annotate_duplicate_hashes(records)
    scored = assign_splits(pd.DataFrame(records))
    metadata_path = write_metadata(scored, processed / f"processed_metadata_{version}.parquet")
    train_jsonl = export_caption_jsonl(scored, exports / "train.jsonl", "train")
    val_jsonl = export_caption_jsonl(scored, exports / "val.jsonl", "val")
    eval_jsonl = export_caption_jsonl(scored, exports / "eval.jsonl", "eval")
    train_sft_jsonl = export_sft_jsonl(scored, exports / "train_sft.jsonl", "train")
    review_jsonl = export_status_jsonl(scored, exports / "review_samples.jsonl", "review")
    rejected_jsonl = export_status_jsonl(scored, exports / "rejected_samples.jsonl", "rejected")
    run_artifacts: dict[str, Path] | None = None
    final_run_id: str | None = None
    run_dir: Path | None = None
    if archive_run:
        manager = RunManager(runs_dir=runs_dir, run_id=run_id)
        final_run_id = manager.run_id
        run_dir = manager.run_dir
        snapshot = deepcopy(config_snapshot) if config_snapshot is not None else {
            "data": {
                "manifest_path": str(manifest_path),
                "raw_data_dir": str(raw_data_dir),
                "processed_dir": str(processed_dir),
                "export_dir": str(export_dir),
            },
            "pipeline": {
                "version": version,
                "use_clip": use_clip,
                "quality_rules_path": str(quality_rules_path) if quality_rules_path is not None else None,
                "model_cache_dir": str(model_cache_dir) if model_cache_dir is not None else None,
                "clip_batch_size": clip_batch_size,
            },
            "runs": {
                "run_id": manager.run_id,
                "runs_dir": str(runs_dir),
            },
        }
        snapshot.setdefault("runs", {})
        snapshot["runs"]["run_id"] = manager.run_id
        snapshot["runs"]["runs_dir"] = str(runs_dir)
        run_artifacts = manager.archive(
            scored=scored,
            config=snapshot,
            input_paths={
                "manifest": manifest_path,
                "raw_data_dir": raw_data_dir,
                "quality_rules": quality_rules_path,
            },
            output_paths={
                "metadata": metadata_path,
                "train_jsonl": train_jsonl,
                "val_jsonl": val_jsonl,
                "eval_jsonl": eval_jsonl,
                "train_sft_jsonl": train_sft_jsonl,
                "review_jsonl": review_jsonl,
                "rejected_jsonl": rejected_jsonl,
            },
            quality_rules={
                "image": rules.image,
                "text": rules.text,
                "score": rules.score,
            },
            scorer_backend=scorer.backend,
        )

    return PipelineResult(
        metadata_path=metadata_path,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        eval_jsonl=eval_jsonl,
        train_sft_jsonl=train_sft_jsonl,
        review_jsonl=review_jsonl,
        rejected_jsonl=rejected_jsonl,
        total_samples=len(scored),
        accepted_samples=int((scored["filter_status"] == "accepted").sum()),
        review_samples=int((scored["filter_status"] == "review").sum()),
        rejected_samples=int((scored["filter_status"] == "rejected").sum()),
        scorer_backend=scorer.backend,
        run_id=final_run_id,
        run_dir=run_dir,
        run_artifacts=run_artifacts,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run multimodal data quality pipeline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--export-dir", default="data/exports")
    parser.add_argument("--version", default="v1.0")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--quality-rules", default=None)
    parser.add_argument("--model-cache-dir", default="data/models/huggingface")
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--no-run-archive", action="store_true")
    args = parser.parse_args()

    result = run_pipeline(
        manifest_path=args.manifest,
        raw_data_dir=args.raw_data_dir,
        processed_dir=args.processed_dir,
        export_dir=args.export_dir,
        version=args.version,
        use_clip=not args.no_clip,
        quality_rules_path=args.quality_rules,
        model_cache_dir=args.model_cache_dir,
        clip_batch_size=args.clip_batch_size,
        run_id=args.run_id,
        runs_dir=args.runs_dir,
        archive_run=not args.no_run_archive,
    )
    print(result)


if __name__ == "__main__":
    main()
