# Scaling Design

## Goal

The local pipeline is designed for development and validation on small to medium datasets. This document describes how the same data model can scale from thousands of samples to millions of images and videos without changing the core quality semantics.

## Current Local Mode

Local mode uses:

- Python scripts and `src.cli`.
- Pandas for metadata assembly and analysis.
- OpenCV/Pillow for image and video processing.
- Parquet for processed metadata.
- JSONL for training, evaluation, review, and SFT exports.
- Streamlit for quality dashboard exploration.

This mode is suitable for validating rules, debugging bad cases, and preparing a GitHub demo.

## Scaling Bottlenecks

Main bottlenecks at large scale:

- Image and video decoding are CPU and I/O heavy.
- CLIP scoring is GPU or model-inference bound.
- Perceptual hash duplicate detection requires grouping and comparison.
- Writing one huge JSONL or Parquet file is hard to retry and inspect.
- Full-table dashboard scans become slow when metadata grows.

## Distributed Batch Design

For million-scale datasets, the pipeline can be decomposed into stages:

1. Ingestion
   - Store raw manifest in partitioned Parquet.
   - Validate file paths and schema before heavy processing.

2. Media probing
   - Run image/video metadata extraction in parallel.
   - Store dimensions, duration, FPS, file size, and decode status.

3. Quality rules
   - Apply image, text, and frame quality checks as stateless map tasks.
   - Write partitioned intermediate results.

4. Model scoring
   - Batch CLIP inference by image/text pairs.
   - Use GPU workers or a model-serving endpoint.
   - Persist scores separately so failed batches can be retried.

5. Aggregation and filtering
   - Join quality fields and model scores by `sample_id`.
   - Compute final status and filter reason.
   - Export accepted, rejected, and review partitions.

6. Reporting
   - Use DuckDB or Spark SQL to aggregate Parquet metadata.
   - Build dashboards from aggregated tables rather than raw images.

## Ray Path

Ray is suitable when the workload is Python-heavy and model-inference-heavy:

- Parallel image/video decoding with Ray tasks.
- Actor-based CLIP workers for batched model inference.
- Retry failed media files independently.
- Keep the current Python quality functions with minimal rewrite.

Best fit:

- Mixed image/video processing.
- GPU model scoring.
- Rapid iteration on Python code.

## Spark Path

Spark is suitable when metadata scale and SQL-style aggregation dominate:

- Store manifests and processed metadata as partitioned Parquet.
- Use Spark DataFrames for joins, filtering, version comparison, and exports.
- Use UDFs or Pandas UDFs for selected media metadata fields.
- Keep heavy model inference outside Spark or behind a batch scoring service.

Best fit:

- Very large metadata tables.
- Scheduled batch processing.
- Integration with a data warehouse or lakehouse.

## Data Layout

Recommended large-scale layout:

```text
data_lake/
  raw_manifest/date=2026-06-17/source=coco/
  media_probe/date=2026-06-17/source=coco/
  image_quality/version=v1.0/source=coco/
  clip_scores/model=clip-vit-base-patch32/source=coco/
  final_metadata/version=v1.0/source=coco/
  exports/version=v1.0/split=train/
```

Partition keys should include `source`, `version`, and processing date. For video data, add `video_id` or hashed prefixes when frame counts are large.

## Reliability

Large-scale runs should record:

- Processing version.
- Config hash.
- Model name and checkpoint.
- Failed file paths.
- Error type.
- Batch id.
- Runtime timestamp.

This makes partial reruns possible without reprocessing the entire dataset.

## Dashboard Strategy

For large data, the dashboard should read pre-aggregated tables:

- Status counts by version.
- Filter reason counts.
- Score histograms.
- Tag distribution.
- Review sample samples.
- Version comparison summaries.

The dashboard should not scan raw media files.

## Interview Framing

The current repo proves the end-to-end semantics locally. At larger scale, the same schema and quality rules can be moved into Ray or Spark batch stages, with Parquet as the contract between stages and JSONL as the final model-data export format.
