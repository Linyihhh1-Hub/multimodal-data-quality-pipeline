from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ingestion.manifest_builder import load_manifest
from src.models.clip_scorer import ClipScorer
from src.pipeline.build_dataset import assign_splits
from src.pipeline.export_sft import export_caption_jsonl, export_sft_jsonl, export_status_jsonl
from src.quality.image_quality import assess_image
from src.quality.quality_scorer import combine_quality
from src.quality.text_quality import assess_caption
from src.storage.metadata_store import write_metadata


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
) -> PipelineResult:
    raw_dir = Path(raw_data_dir)
    processed = Path(processed_dir)
    exports = Path(export_dir)
    df = load_manifest(manifest_path)
    scorer = ClipScorer(use_clip=use_clip)

    records: list[dict] = []
    for _, row in df.iterrows():
        absolute_image_path = _resolve_image_path(raw_dir, str(row["image_path"]))
        image_result = assess_image(absolute_image_path)
        text_result = assess_caption(row["caption"])
        similarity = scorer.score(absolute_image_path, row["caption"])
        combined = combine_quality(
            image_status=image_result["filter_status"],
            text_status=text_result["filter_status"],
            image_reasons=image_result["filter_reasons"],
            text_reasons=text_result["filter_reasons"],
            image_quality_score=image_result["image_quality_score"],
            text_quality_score=text_result["text_quality_score"],
            image_text_similarity=similarity,
        )
        records.append(
            {
                **row.to_dict(),
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
            }
        )

    scored = assign_splits(pd.DataFrame(records))
    metadata_path = write_metadata(scored, processed / f"processed_metadata_{version}.parquet")
    train_jsonl = export_caption_jsonl(scored, exports / "train.jsonl", "train")
    val_jsonl = export_caption_jsonl(scored, exports / "val.jsonl", "val")
    eval_jsonl = export_caption_jsonl(scored, exports / "eval.jsonl", "eval")
    train_sft_jsonl = export_sft_jsonl(scored, exports / "train_sft.jsonl", "train")
    review_jsonl = export_status_jsonl(scored, exports / "review_samples.jsonl", "review")
    rejected_jsonl = export_status_jsonl(scored, exports / "rejected_samples.jsonl", "rejected")

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
    args = parser.parse_args()

    result = run_pipeline(
        manifest_path=args.manifest,
        raw_data_dir=args.raw_data_dir,
        processed_dir=args.processed_dir,
        export_dir=args.export_dir,
        version=args.version,
        use_clip=not args.no_clip,
    )
    print(result)


if __name__ == "__main__":
    main()
