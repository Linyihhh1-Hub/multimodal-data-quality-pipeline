from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard.metrics import (
    ALERT_THRESHOLDS,
    build_governance_summary,
    build_quality_alerts,
    build_quality_pipeline,
    build_status_breakdown,
    caption_lengths,
    diagnostic_samples,
    export_inventory,
    filter_reason_topn,
    load_jsonl_rows,
    localize_status,
    review_queue,
    score_columns,
    top_tags_from_column,
)
from src.storage.metadata_store import read_metadata


DATASET_PRESETS: dict[str, dict[str, str]] = {
    "Demo 本地样例": {
        "metadata": "data/processed/processed_metadata_v1.1.parquet",
        "video_manifest": "data/processed/video_manifest.jsonl",
        "frame_manifest": "data/processed/video_frame_manifest.jsonl",
        "export_dir": "data/exports",
    },
    "COCO Heuristic v1.0": {
        "metadata": "data/processed_coco/processed_metadata_coco_v1.0.parquet",
        "video_manifest": "data/processed/video_manifest.jsonl",
        "frame_manifest": "data/processed/video_frame_manifest.jsonl",
        "export_dir": "data/exports_coco",
    },
    "COCO + CLIP v1.1": {
        "metadata": "data/processed_clip_coco/processed_metadata_coco_clip_v1.1.parquet",
        "video_manifest": "data/processed/video_manifest.jsonl",
        "frame_manifest": "data/processed/video_frame_manifest.jsonl",
        "export_dir": "data/exports_clip_coco_v1.1",
    },
    "自定义路径": {
        "metadata": "data/processed/processed_metadata_v1.1.parquet",
        "video_manifest": "data/processed/video_manifest.jsonl",
        "frame_manifest": "data/processed/video_frame_manifest.jsonl",
        "export_dir": "data/exports",
    },
}


st.set_page_config(
    page_title="AI 训练数据治理工作台",
    page_icon=":material/dataset:",
    layout="wide",
)


@st.cache_data(ttl="5m", max_entries=12)
def _load_metadata(path: str) -> pd.DataFrame:
    return read_metadata(path)


@st.cache_data(ttl="5m", max_entries=12)
def _load_jsonl(path: str) -> pd.DataFrame:
    return load_jsonl_rows(path)


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _render_header(metadata_path: Path) -> None:
    left, right = st.columns([0.72, 0.28], vertical_alignment="center")
    with left:
        st.title("AI 训练数据治理工作台", anchor=False)
        st.caption("用于 VLM 图文样本、视频关键帧、人工复核队列和训练数据导出产物的质量治理。")
    with right:
        st.badge("本地 Pipeline", icon=":material/database:", color="blue")
        st.badge(metadata_path.suffix.replace(".", "").upper(), icon=":material/table_chart:", color="gray")


def _render_path_status(paths: dict[str, Path]) -> None:
    missing = [f"{label}: `{path}`" for label, path in paths.items() if label in {"元数据", "导出目录"} and not path.exists()]
    if missing:
        st.warning(
            "当前路径不存在，请先运行 pipeline 或切换数据集版本。\n\n" + "\n".join(f"- {item}" for item in missing),
            icon=":material/warning:",
        )


def _render_kpis(df: pd.DataFrame, export_dir: Path) -> None:
    summary = build_governance_summary(df, export_dir)
    with st.container(horizontal=True):
        st.metric("总样本数", f"{summary['total']:,}", border=True)
        st.metric("Accepted 样本", f"{summary['accepted']:,} ({_pct(float(summary['accepted_rate']))})", border=True)
        st.metric("Review 样本", f"{summary['review']:,} ({_pct(float(summary['review_rate']))})", border=True)
        st.metric("Rejected 样本", f"{summary['rejected']:,} ({_pct(float(summary['rejected_rate']))})", border=True)
        st.metric("平均质量分", _score(summary["avg_quality_score"]), border=True)
        st.metric("平均 CLIP 图文相似度", _score(summary["avg_clip_score"]), border=True)
        st.metric("近重复图比例", _pct(float(summary["near_duplicate_rate"])), border=True)
        st.metric("导出资产数量", f"{summary['export_asset_count']:,}", border=True)
    st.caption(
        f"默认阈值：Accepted >= {_pct(ALERT_THRESHOLDS['accepted_ratio_min'])}，"
        f"Review <= {_pct(ALERT_THRESHOLDS['review_ratio_max'])}，"
        f"Rejected <= {_pct(ALERT_THRESHOLDS['rejected_ratio_max'])}，"
        f"CLIP 均值 >= {ALERT_THRESHOLDS['clip_score_min']:.2f}。"
    )


def _render_alerts(df: pd.DataFrame, export_dir: Path) -> None:
    alerts = build_quality_alerts(build_governance_summary(df, export_dir))
    if not alerts:
        st.success("当前数据集没有触发关键治理告警。", icon=":material/check_circle:")
        return
    for alert in alerts:
        message = (
            f"**{alert['alert']}**\n\n"
            f"- 可能原因：{alert['possible_causes']}\n"
            f"- 建议动作：{alert['action']}"
        )
        if alert["level"] == "error":
            st.error(message, icon=":material/error:")
        else:
            st.warning(message, icon=":material/warning:")


def _render_overview(df: pd.DataFrame, export_dir: Path) -> None:
    left, right = st.columns([0.62, 0.38], vertical_alignment="top")
    with left:
        with st.container(border=True):
            st.subheader("多阶段质量流程", anchor=False)
            pipeline = build_quality_pipeline(df, export_dir)
            st.bar_chart(pipeline, x="stage", y="samples", horizontal=True)
            st.dataframe(
                pipeline,
                hide_index=True,
                width="stretch",
                column_config={
                    "stage": st.column_config.TextColumn("阶段", pinned=True),
                    "samples": st.column_config.NumberColumn("样本数", format="%d"),
                    "rate": st.column_config.ProgressColumn("占原始样本比例", format="percent", min_value=0, max_value=1),
                    "note": st.column_config.TextColumn("口径说明", width="large"),
                },
            )
    with right:
        with st.container(border=True):
            st.subheader("告警 + 可能原因 + 建议动作", anchor=False)
            _render_alerts(df, export_dir)

        with st.container(border=True):
            st.subheader("状态分布", anchor=False)
            status_df = build_status_breakdown(df)
            if status_df.empty:
                st.caption("未找到 accepted/review/rejected 状态字段。")
            else:
                st.bar_chart(status_df, x="status_label", y="samples")


def _histogram_frame(series: pd.Series, bins: int = 10) -> pd.DataFrame:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return pd.DataFrame({"range": [], "samples": []})
    bucketed = pd.cut(values, bins=bins, include_lowest=True)
    counts = bucketed.value_counts().sort_index()
    return pd.DataFrame({"range": counts.index.astype(str), "samples": counts.values})


def _render_distribution(title: str, series: pd.Series | None) -> None:
    with st.container(border=True):
        st.subheader(title, anchor=False)
        if series is None:
            st.caption("当前数据集缺少该字段。")
            return
        hist = _histogram_frame(series)
        if hist.empty:
            st.caption("当前字段没有可用于绘图的数值。")
        else:
            st.bar_chart(hist, x="range", y="samples")


def _status_and_score_columns(df: pd.DataFrame) -> dict[str, Any]:
    config: dict[str, Any] = {
        "sample_id": st.column_config.TextColumn("sample_id", pinned=True),
        "image_id": st.column_config.TextColumn("image_id", pinned=True),
        "image_path": st.column_config.TextColumn("图片路径"),
        "caption": st.column_config.TextColumn("caption", width="large"),
        "status_label": st.column_config.TextColumn("状态"),
        "filter_reason": st.column_config.TextColumn("过滤原因"),
        "final_quality_score": st.column_config.ProgressColumn("质量分", format="%.3f", min_value=0, max_value=1),
        "clip_score": st.column_config.NumberColumn("CLIP score", format="%.4f"),
        "image_text_similarity": st.column_config.NumberColumn("CLIP 图文相似度", format="%.4f"),
        "similarity_score": st.column_config.NumberColumn("CLIP 相似度", format="%.4f"),
        "caption_length": st.column_config.NumberColumn("caption 长度", format="%d"),
    }
    return {column: config[column] for column in df.columns if column in config}


def _display_sample_table(df: pd.DataFrame, limit: int = 20) -> None:
    if df.empty:
        st.caption("暂无可展示样本。")
        return
    result = df.copy().head(limit)
    if "status_label" not in result.columns:
        for column in ["final_status", "status", "filter_status"]:
            if column in result.columns:
                result["status_label"] = result[column].map(localize_status)
                break
    result["caption_length"] = caption_lengths(result)
    display_cols = [
        column
        for column in [
            "sample_id",
            "image_id",
            "image_path",
            "caption",
            "status_label",
            "final_status",
            "status",
            "filter_status",
            "final_quality_score",
            "clip_score",
            "image_text_similarity",
            "similarity_score",
            "filter_reason",
            "caption_length",
        ]
        if column in result.columns
    ]
    st.dataframe(
        result[display_cols],
        hide_index=True,
        width="stretch",
        column_config=_status_and_score_columns(result[display_cols]),
    )


def _render_diagnosis(df: pd.DataFrame) -> None:
    top_left, top_right = st.columns([0.45, 0.55], vertical_alignment="top")
    with top_left:
        with st.container(border=True):
            st.subheader("filter_reason TopN", anchor=False)
            reasons = filter_reason_topn(df, limit=12)
            if reasons.empty:
                st.caption("暂无过滤原因记录。")
            else:
                st.bar_chart(reasons, x="filter_reason", y="samples", horizontal=True)
                st.dataframe(
                    reasons,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "filter_reason": st.column_config.TextColumn("过滤原因", pinned=True),
                        "samples": st.column_config.NumberColumn("样本数", format="%d"),
                    },
                )
    with top_right:
        score_cols = score_columns(df)
        with st.container(border=True):
            st.subheader("分数字段概览", anchor=False)
            if score_cols:
                st.dataframe(df[score_cols].describe().round(4), width="stretch")
            else:
                st.caption("未找到可统计的数值分数字段。")

    q1, q2, q3 = st.columns(3, vertical_alignment="top")
    with q1:
        _render_distribution("final_quality_score 分布", df["final_quality_score"] if "final_quality_score" in df.columns else None)
    with q2:
        clip_column = next(
            (column for column in ["clip_score", "image_text_similarity", "similarity_score"] if column in df.columns),
            None,
        )
        _render_distribution("CLIP 图文相似度分布", df[clip_column] if clip_column in df.columns else None)
    with q3:
        _render_distribution("caption 长度分布", caption_lengths(df) if len(df) else None)

    with st.container(border=True):
        st.subheader("低质量样本表格", anchor=False)
        _display_sample_table(diagnostic_samples(df, limit=20), limit=20)


def _render_review(df: pd.DataFrame) -> None:
    queue = review_queue(df, limit=200)
    left, right = st.columns([0.68, 0.32], vertical_alignment="top")
    with left:
        with st.container(border=True):
            st.subheader("Review 样本队列", anchor=False)
            if queue.empty:
                st.info("当前没有待复核样本。", icon=":material/check_circle:")
            else:
                _display_sample_table(queue, limit=200)
    with right:
        with st.container(border=True):
            st.subheader("Review 队列作用", anchor=False)
            st.markdown(
                "- 自动规则无法直接判断的边界样本进入人工复核。\n"
                "- 人工复核结果可以回流为新版本数据。\n"
                "- Review 样本不直接进入训练集，避免污染训练数据。"
            )


def _render_video(video_df: pd.DataFrame, frame_df: pd.DataFrame) -> None:
    if video_df.empty and frame_df.empty:
        st.info("未加载视频 manifest。请先运行 `python -m src.cli video --config configs/pipeline.yaml`。", icon=":material/info:")
        return

    if not video_df.empty:
        with st.container(horizontal=True):
            st.metric("视频数", f"{len(video_df):,}", border=True)
            if "sampled_frame_count" in video_df.columns:
                st.metric("采样帧数", f"{int(video_df['sampled_frame_count'].sum()):,}", border=True)
            if "valid_frame_count" in video_df.columns:
                st.metric("有效帧数", f"{int(video_df['valid_frame_count'].sum()):,}", border=True)
            if "video_quality_score" in video_df.columns:
                st.metric("平均视频质量分", f"{video_df['video_quality_score'].mean():.3f}", border=True)
        with st.container(border=True):
            st.subheader("视频 manifest", anchor=False)
            st.dataframe(video_df, hide_index=True, width="stretch")

    if not frame_df.empty:
        left, right = st.columns([0.38, 0.62], vertical_alignment="top")
        with left:
            with st.container(border=True):
                st.subheader("帧状态分布", anchor=False)
                if "filter_status" in frame_df.columns:
                    st.bar_chart(frame_df["filter_status"].map(localize_status).value_counts())
                else:
                    st.caption("未找到帧状态字段。")
        with right:
            with st.container(border=True):
                st.subheader("帧级 manifest", anchor=False)
                st.dataframe(frame_df.head(200), hide_index=True, width="stretch")


def _render_assets(df: pd.DataFrame, export_dir: Path) -> None:
    left, right = st.columns([0.36, 0.64], vertical_alignment="top")
    with left:
        with st.container(border=True):
            st.subheader("文本标签", anchor=False)
            tags = top_tags_from_column(df)
            if tags.empty:
                st.caption("当前元数据未找到 tags 字段。")
            else:
                st.bar_chart(tags)
    with right:
        with st.container(border=True):
            st.subheader("数据资产清单", anchor=False)
            inventory = export_inventory(export_dir)
            st.dataframe(
                inventory,
                hide_index=True,
                width="stretch",
                column_config={
                    "file": st.column_config.TextColumn("文件名", pinned=True),
                    "exists": st.column_config.CheckboxColumn("存在"),
                    "status_label": st.column_config.TextColumn("状态"),
                    "size_bytes": st.column_config.NumberColumn("文件大小 Bytes", format="%d"),
                    "size_kb": st.column_config.NumberColumn("文件大小 KB", format="%.2f"),
                    "rows": st.column_config.NumberColumn("样本行数", format="%d"),
                    "updated_at": st.column_config.TextColumn("更新时间"),
                    "purpose": st.column_config.TextColumn("用途说明", width="large"),
                },
            )


with st.sidebar:
    st.header("数据源")
    dataset_version = st.selectbox(
        "数据集版本选择",
        list(DATASET_PRESETS.keys()),
        index=0,
    )
    preset = DATASET_PRESETS[dataset_version]
    metadata_path = Path(st.text_input("元数据路径", preset["metadata"], key=f"metadata-{dataset_version}"))
    video_manifest_path = Path(st.text_input("视频 manifest", preset["video_manifest"], key=f"video-{dataset_version}"))
    frame_manifest_path = Path(st.text_input("帧级 manifest", preset["frame_manifest"], key=f"frame-{dataset_version}"))
    export_dir = Path(st.text_input("导出目录", preset["export_dir"], key=f"export-{dataset_version}"))
    st.caption("保留手动路径配置；选择预设版本会填充对应默认路径。")

_render_header(metadata_path)
_render_path_status(
    {
        "元数据": metadata_path,
        "视频 manifest": video_manifest_path,
        "帧级 manifest": frame_manifest_path,
        "导出目录": export_dir,
    }
)

if not metadata_path.exists():
    st.stop()

df = _load_metadata(str(metadata_path))
video_df = _load_jsonl(str(video_manifest_path))
frame_df = _load_jsonl(str(frame_manifest_path))

_render_kpis(df, export_dir)

overview_tab, diagnosis_tab, review_tab, video_tab, assets_tab = st.tabs(["总览", "诊断", "复核", "视频", "资产"])

with overview_tab:
    _render_overview(df, export_dir)
with diagnosis_tab:
    _render_diagnosis(df)
with review_tab:
    _render_review(df)
with video_tab:
    _render_video(video_df, frame_df)
with assets_tab:
    _render_assets(df, export_dir)
