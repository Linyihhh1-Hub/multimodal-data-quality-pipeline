from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd


def _image_data_uri(path: str | Path) -> str:
    image_path = Path(path)
    if not image_path.exists():
        return ""
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_sample_gallery_html(
    df: pd.DataFrame,
    title: str = "Multimodal Data Sample Gallery",
    limit: int = 60,
) -> str:
    rows = df.sort_values(["filter_status", "final_quality_score"], ascending=[True, True]).head(limit)
    cards: list[str] = []
    for _, row in rows.iterrows():
        status = html.escape(str(row.get("filter_status", "")))
        reason = html.escape(str(row.get("filter_reason", "")))
        caption = html.escape(str(row.get("caption", "")))
        sample_id = html.escape(str(row.get("sample_id", "")))
        image_src = _image_data_uri(row.get("resolved_image_path", row.get("image_path", "")))
        image_html = f'<img src="{image_src}" alt="{sample_id}">' if image_src else '<div class="missing">missing image</div>'
        cards.append(
            f"""
            <article class="card {status}">
              {image_html}
              <div class="body">
                <div class="topline"><strong>{sample_id}</strong><span>{status}</span></div>
                <p>{caption}</p>
                <dl>
                  <dt>score</dt><dd>{float(row.get("final_quality_score", 0.0)):.4f}</dd>
                  <dt>similarity</dt><dd>{float(row.get("image_text_similarity", 0.0)):.4f}</dd>
                  <dt>reason</dt><dd>{reason or "none"}</dd>
                </dl>
              </div>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f7fa; color: #1f2937; }}
    header {{ padding: 24px 32px; background: #111827; color: white; }}
    h1 {{ margin: 0; font-size: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; padding: 24px; }}
    .card {{ background: white; border: 1px solid #d1d5db; border-radius: 8px; overflow: hidden; }}
    .card.accepted {{ border-top: 5px solid #059669; }}
    .card.review {{ border-top: 5px solid #d97706; }}
    .card.rejected {{ border-top: 5px solid #dc2626; }}
    img, .missing {{ width: 100%; aspect-ratio: 4 / 3; object-fit: cover; background: #e5e7eb; display: block; }}
    .missing {{ display: grid; place-items: center; color: #6b7280; }}
    .body {{ padding: 12px; }}
    .topline {{ display: flex; justify-content: space-between; gap: 8px; font-size: 13px; }}
    p {{ min-height: 42px; font-size: 14px; line-height: 1.4; }}
    dl {{ display: grid; grid-template-columns: 72px 1fr; gap: 6px 10px; font-size: 13px; }}
    dt {{ color: #6b7280; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <header><h1>{html.escape(title)}</h1></header>
  <main class="grid">
    {''.join(cards)}
  </main>
</body>
</html>
"""


def write_sample_gallery(df: pd.DataFrame, output_path: str | Path, title: str, limit: int = 60) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_sample_gallery_html(df, title=title, limit=limit), encoding="utf-8")
    return path
