from pathlib import Path

import pandas as pd
from PIL import Image

from src.analysis.sample_gallery import build_sample_gallery_html


def test_build_sample_gallery_html_embeds_sample_cards(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (64, 64), color=(80, 120, 160)).save(image_dir / "a.jpg")
    df = pd.DataFrame(
        [
            {
                "sample_id": "sample_a",
                "resolved_image_path": str(image_dir / "a.jpg"),
                "caption": "A valid caption.",
                "filter_status": "accepted",
                "filter_reason": "",
                "final_quality_score": 0.88,
                "image_text_similarity": 0.81,
            }
        ]
    )

    html = build_sample_gallery_html(df, title="Demo Gallery", limit=1)

    assert "Demo Gallery" in html
    assert "sample_a" in html
    assert "A valid caption." in html
    assert "data:image/jpeg;base64," in html
