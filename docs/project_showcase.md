# Project Showcase

## Real COCO Run

| Metric | Value |
| --- | ---: |
| Total COCO captions | 5000 |
| Heuristic accepted | 4960 |
| Heuristic rejected | 40 |
| Duplicate image samples | 3984 |
| CLIP v1.0 accepted | 4960 |
| CLIP v1.0 rejected | 40 |
| CLIP v1.1 accepted | 1283 |
| CLIP v1.1 review | 3677 |
| CLIP v1.1 rejected | 40 |
| v1.0 to v1.1 status changed | 3677 |

## Tag Distribution

Top caption tags: bathroom: 546, sitting: 454, toilet: 410, man: 397, white: 389, kitchen: 365, two: 350, next: 336, street: 336, motorcycle: 291

## Local Artifacts

- `data/processed_coco/quality_report_coco_v1.0.md`
- `data/processed_coco/sample_gallery_coco_v1.0.html`
- `data/processed_clip_coco/quality_report_coco_clip_v1.1.md`
- `data/processed_clip_coco/sample_gallery_coco_clip_v1.1.html`
- `data/processed_clip_coco/version_compare_coco_clip_v1.0_v1.1.json`

## Interview Talking Point

The project now demonstrates a full AI-data workflow: data readiness checks, real COCO ingestion, rule-based quality checks, model-assisted CLIP scoring, duplicate-image analysis, caption tag distribution, versioned rule iteration, and export to train/eval/SFT JSONL formats.
