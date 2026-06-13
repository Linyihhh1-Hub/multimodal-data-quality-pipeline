from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.manifest_builder import build_manifest_from_coco_annotations


def load_coco_captions_subset(
    annotations_path: str | Path,
    images_dir: str | Path,
    limit: int = 5000,
) -> pd.DataFrame:
    return build_manifest_from_coco_annotations(
        annotations_path=annotations_path,
        images_dir=images_dir,
        source="coco",
        limit=limit,
    )
