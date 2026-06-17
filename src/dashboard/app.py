from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.metrics import (
    build_dashboard_summary,
    explode_filter_reasons,
    load_jsonl_rows,
    score_columns,
    top_tags_from_column,
)
from src.storage.metadata_store import read_metadata


st.set_page_config(page_title="Multimodal Data Quality", layout="wide")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            max-width: 1320px;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetricLabel"] p {
            color: #475569;
            font-size: 0.86rem;
        }
        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-size: 1.6rem;
        }
        .dashboard-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.15rem;
        }
        .dashboard-subtitle {
            color: #475569;
            margin-bottom: 1.1rem;
        }
        .section-note {
            color: #64748b;
            font-size: 0.92rem;
            margin-top: -0.35rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_status_metrics(df: pd.DataFrame) -> None:
    summary = build_dashboard_summary(df)
    cols = st.columns(7)
    cols[0].metric("Total", f"{summary['total']:,}")
    cols[1].metric("Accepted", f"{summary['accepted']:,}", _format_percent(float(summary["acceptance_rate"])))
    cols[2].metric("Review", f"{summary['review']:,}", _format_percent(float(summary["review_rate"])))
    cols[3].metric("Rejected", f"{summary['rejected']:,}", _format_percent(float(summary["rejection_rate"])))
    cols[4].metric("Avg quality", f"{float(summary['avg_quality_score']):.3f}")
    cols[5].metric("Avg similarity", f"{float(summary['avg_similarity']):.3f}")
    cols[6].metric("Columns", f"{len(df.columns):,}")


def _render_overview(df: pd.DataFrame) -> None:
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Filter status")
        if "filter_status" in df.columns:
            st.bar_chart(df["filter_status"].value_counts())
        else:
            st.info("No filter_status column found.")
    with right:
        st.subheader("Filter reasons")
        reasons = explode_filter_reasons(df)
        if reasons.empty:
            st.info("No filter reasons found.")
        else:
            st.bar_chart(reasons.head(15))


def _render_quality(df: pd.DataFrame) -> None:
    scores = score_columns(df)
    if scores:
        st.subheader("Score distributions")
        st.line_chart(df[scores].reset_index(drop=True))
    else:
        st.info("No numeric quality score columns found.")

    cols = [column for column in ["width", "height", "blur_variance", "brightness", "aspect_ratio"] if column in df.columns]
    if cols:
        st.subheader("Image diagnostics")
        st.dataframe(df[cols].describe().round(3), use_container_width=True)


def _render_similarity(df: pd.DataFrame) -> None:
    if "image_text_similarity" not in df.columns:
        st.info("No image_text_similarity column found.")
        return
    st.subheader("Image-text similarity by status")
    chart_cols = ["image_text_similarity", "filter_status"]
    st.dataframe(
        df[chart_cols].groupby("filter_status").describe().round(4),
        use_container_width=True,
    )
    st.line_chart(df["image_text_similarity"].reset_index(drop=True))


def _render_review_queue(df: pd.DataFrame) -> None:
    st.subheader("Review queue")
    st.markdown('<div class="section-note">Samples are sorted by final quality score so the riskiest items appear first.</div>', unsafe_allow_html=True)
    preview_cols = [
        column
        for column in [
            "sample_id",
            "image_path",
            "caption",
            "image_text_similarity",
            "final_quality_score",
            "filter_status",
            "filter_reason",
        ]
        if column in df.columns
    ]
    if not preview_cols:
        st.info("No review queue columns found.")
        return
    status_filter = st.multiselect(
        "Filter status",
        options=sorted(df["filter_status"].dropna().unique()) if "filter_status" in df.columns else [],
        default=["review"] if "review" in set(df.get("filter_status", pd.Series(dtype=str))) else None,
    )
    table = df
    if status_filter and "filter_status" in table.columns:
        table = table[table["filter_status"].isin(status_filter)]
    sort_col = "final_quality_score" if "final_quality_score" in table.columns else preview_cols[0]
    st.dataframe(table[preview_cols].sort_values(sort_col).head(200), use_container_width=True, hide_index=True)


def _render_video(video_df: pd.DataFrame, frame_df: pd.DataFrame) -> None:
    if video_df.empty and frame_df.empty:
        st.info("No video manifest loaded. Set paths in the sidebar after running the video pipeline.")
        return
    if not video_df.empty:
        st.subheader("Video-level quality")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Videos", f"{len(video_df):,}")
        if "sampled_frame_count" in video_df.columns:
            metric_cols[1].metric("Sampled frames", f"{int(video_df['sampled_frame_count'].sum()):,}")
        if "valid_frame_count" in video_df.columns:
            metric_cols[2].metric("Valid frames", f"{int(video_df['valid_frame_count'].sum()):,}")
        if "video_quality_score" in video_df.columns:
            metric_cols[3].metric("Avg video quality", f"{video_df['video_quality_score'].mean():.3f}")
        st.dataframe(video_df.head(100), use_container_width=True, hide_index=True)
    if not frame_df.empty:
        st.subheader("Frame-level quality")
        status_col = "filter_status"
        if status_col in frame_df.columns:
            st.bar_chart(frame_df[status_col].value_counts())
        st.dataframe(frame_df.head(200), use_container_width=True, hide_index=True)


def _render_exports(export_dir: Path) -> None:
    st.subheader("Local export files")
    if not export_dir.exists():
        st.info(f"Export directory not found: {export_dir}")
        return
    files = sorted(path for path in export_dir.glob("*.jsonl"))
    if not files:
        st.info("No JSONL exports found.")
        return
    rows = [{"file": path.name, "size_kb": round(path.stat().st_size / 1024, 2)} for path in files]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


_inject_css()

st.markdown('<div class="dashboard-title">Multimodal Data Quality Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-subtitle">Quality monitoring for image-text samples, video keyframes, review queues, and dataset exports.</div>',
    unsafe_allow_html=True,
)

default_path = Path("data/processed/processed_metadata_v1.0.parquet")
with st.sidebar:
    st.header("Data sources")
    metadata_path = Path(st.text_input("Metadata path", str(default_path)))
    video_manifest_path = Path(st.text_input("Video manifest", "data/processed/video_manifest.jsonl"))
    frame_manifest_path = Path(st.text_input("Frame manifest", "data/processed/video_frame_manifest.jsonl"))
    export_dir = Path(st.text_input("Export directory", "data/exports"))

if not metadata_path.exists():
    st.warning(f"Metadata file not found: {metadata_path}")
    st.stop()

df = read_metadata(metadata_path)
video_df = load_jsonl_rows(video_manifest_path)
frame_df = load_jsonl_rows(frame_manifest_path)

_render_status_metrics(df)

tabs = st.tabs(["Overview", "Quality", "Similarity", "Tags", "Review Queue", "Video", "Exports"])
with tabs[0]:
    _render_overview(df)
with tabs[1]:
    _render_quality(df)
with tabs[2]:
    _render_similarity(df)
with tabs[3]:
    tags = top_tags_from_column(df)
    if tags.empty:
        st.info("No tags column found in current metadata.")
    else:
        st.bar_chart(tags)
with tabs[4]:
    _render_review_queue(df)
with tabs[5]:
    _render_video(video_df, frame_df)
with tabs[6]:
    _render_exports(export_dir)
