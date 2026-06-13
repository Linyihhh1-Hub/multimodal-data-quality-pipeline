from pathlib import Path

from PIL import Image
import pandas as pd

from src.pipeline.run_pipeline import run_pipeline


def test_run_pipeline_creates_metadata_and_exports(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    image_dir = raw_dir / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (320, 240), color=(80, 120, 160)).save(image_dir / "1.jpg")
    Image.new("RGB", (100, 100), color=(80, 80, 80)).save(image_dir / "2.jpg")
    manifest = raw_dir / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                '{"image_id":"1","image_path":"images/1.jpg","caption":"A person rides a bicycle on a street.","source":"demo"}',
                '{"image_id":"2","image_path":"images/2.jpg","caption":"","source":"demo"}',
            ]
        ),
        encoding="utf-8",
    )

    result = run_pipeline(
        manifest_path=manifest,
        raw_data_dir=raw_dir,
        processed_dir=tmp_path / "processed",
        export_dir=tmp_path / "exports",
        version="vtest",
        use_clip=False,
    )

    assert result.metadata_path.exists()
    assert result.train_jsonl.exists()
    assert result.review_jsonl.exists()
    assert result.rejected_jsonl.exists()
    assert result.total_samples == 2
    assert result.rejected_samples >= 1

    metadata = pd.read_parquet(result.metadata_path)
    for column in [
        "image_quality_score",
        "text_quality_score",
        "image_text_similarity",
        "final_quality_score",
        "filter_status",
        "filter_reason",
        "version",
    ]:
        assert column in metadata.columns
