import json
from pathlib import Path

import pandas as pd

from scripts.generate_project_showcase import build_project_showcase


def test_build_project_showcase_uses_real_metrics(tmp_path: Path):
    heuristic = pd.DataFrame(
        [
            {"filter_status": "accepted", "is_duplicate_image": False},
            {"filter_status": "rejected", "is_duplicate_image": True},
        ]
    )
    clip_v1 = pd.DataFrame([{"filter_status": "accepted"}])
    clip_v11 = pd.DataFrame([{"filter_status": "review"}, {"filter_status": "rejected"}])
    tags = {"top_tags": [{"tag": "person", "count": 3}]}
    compare = {"status_changed_samples": 2, "old_acceptance_rate": 1.0, "new_acceptance_rate": 0.0}

    heuristic_path = tmp_path / "heuristic.parquet"
    clip_v1_path = tmp_path / "clip_v1.parquet"
    clip_v11_path = tmp_path / "clip_v11.parquet"
    tag_path = tmp_path / "tags.json"
    compare_path = tmp_path / "compare.json"
    heuristic.to_parquet(heuristic_path)
    clip_v1.to_parquet(clip_v1_path)
    clip_v11.to_parquet(clip_v11_path)
    tag_path.write_text(json.dumps(tags), encoding="utf-8")
    compare_path.write_text(json.dumps(compare), encoding="utf-8")

    markdown = build_project_showcase(
        heuristic_metadata=heuristic_path,
        clip_v1_metadata=clip_v1_path,
        clip_v11_metadata=clip_v11_path,
        tag_distribution=tag_path,
        version_compare=compare_path,
    )

    assert "Total COCO captions | 2" in markdown
    assert "Duplicate image samples | 1" in markdown
    assert "CLIP v1.1 review | 1" in markdown
    assert "person: 3" in markdown
