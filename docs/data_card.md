# Data Card

## Dataset Purpose

This project builds quality-controlled image-text and video-keyframe metadata for visual language model (VLM) data preparation. The output is intended for training data filtering, supervised fine-tuning format construction, evaluation set construction, and data quality analysis.

## Data Sources

- Demo data: generated locally by `scripts/create_demo_data.py` and `scripts/create_demo_video.py`.
- COCO Captions subset: optional real-data path built from COCO val2017 captions and images.
- Video data: local video files processed into sampled keyframes and frame-level manifests.

Raw datasets, generated images, extracted frames, Parquet metadata, and JSONL exports are treated as local artifacts under `data/` and are not tracked by Git.

## Core Schemas

Image-text manifest:

```json
{
  "image_id": "demo_1",
  "image_path": "images/demo_1.jpg",
  "caption": "A person rides a bicycle on a city street.",
  "source": "demo"
}
```

Processed image-text metadata includes image quality fields, text quality fields, image-text similarity, duplicate metadata, final quality score, filter status, filter reason, split, and version.

Video manifest:

```json
{
  "video_id": "demo_video",
  "video_path": "videos/demo_video.mp4",
  "fps": 12.0,
  "frame_count": 48,
  "duration_seconds": 4.0,
  "width": 640,
  "height": 360,
  "sampled_frame_count": 4,
  "valid_frame_count": 4,
  "rejected_frame_count": 0,
  "video_quality_score": 0.92
}
```

Video-frame manifest contains `video_id`, `frame_index`, `timestamp_seconds`, `frame_path`, frame-level image quality fields, and frame filter status.

## Quality Rules

Image quality checks:

- Missing or corrupt images are rejected.
- Images below the configured minimum resolution are rejected.
- Abnormal aspect ratio, blur, and brightness are marked for review unless a stronger rejection rule applies.
- Perceptual hashes are used to annotate near-duplicate image groups.

Text quality checks:

- Empty captions are rejected.
- Very short, very long, or abnormal captions are marked for review or rejection based on configured rules.
- Sensitive word filtering is supported through configuration.

Image-text consistency:

- CLIP can be used to score image-text similarity.
- A deterministic heuristic scorer is available for local and CI environments where model download is not desired.
- Final status is assigned as `accepted`, `review`, or `rejected`.

## Versioning

The pipeline writes versioned metadata files such as `processed_metadata_v1.0.parquet`. Rule updates can be compared across versions to measure status changes, acceptance rate changes, and review volume changes.

## Intended Use

- VLM fine-tuning data preparation.
- Evaluation set construction.
- Data quality dashboards and rule iteration.
- Model-assisted sample review workflows.
- Lightweight video-to-keyframe data preparation for multimodal and creative AI workflows.

## Limitations

- The project does not train a foundation model.
- CLIP similarity is a model-assisted signal, not a final truth label.
- Video processing currently focuses on metadata and keyframes; it does not include ASR, OCR, audio understanding, or full temporal reasoning.
- Human review is still needed for ambiguous samples and high-impact filtering decisions.
