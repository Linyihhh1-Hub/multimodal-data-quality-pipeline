# Multimodal Data Quality Report - v1.1

## Summary

| Metric | Value |
| --- | --- |
| Total samples | 4 |
| Accepted samples | 1 |
| Review samples | 1 |
| Rejected samples | 2 |
| Acceptance rate | 25.00% |
| Average image-text similarity | 0.5291 |
| Average final quality score | 0.6616 |

## Filter Reason Distribution

| Reason | Count |
| --- | ---: |
| borderline_quality_score | 1 |
| low_resolution | 1 |
| blurry_image | 1 |
| short_caption | 1 |
| empty_caption | 1 |

## Lowest Quality Samples

| sample_id | status | score | similarity | reasons | caption |
| --- | --- | ---: | ---: | --- | --- |
| demo_demo_missing_caption | rejected | 0.3500 | 0.0000 | empty_caption |  |
| demo_demo_small | rejected | 0.4455 | 0.4888 | low_resolution;blurry_image;short_caption | small |
| demo_demo_2 | review | 0.9193 | 0.7983 | borderline_quality_score | A dog sits near a wooden table indoors. |
| demo_demo_1 | accepted | 0.9317 | 0.8292 |  | A person rides a bicycle on a city street. |

## Interview Notes

- This report turns raw filtering metadata into quality evidence: pass rate, failure modes, and low-quality examples.
- Use the reason distribution to explain rule iteration, such as tightening image-text similarity or moving borderline samples to review.
- Use the lowest-quality table to discuss concrete cases instead of describing cleaning rules only in abstract terms.
