from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


STATUS_LABELS = {
    "accepted": "通过",
    "review": "待复核",
    "rejected": "拒绝",
}

STATUS_ORDER = ["accepted", "review", "rejected"]

ALERT_THRESHOLDS = {
    "accepted_ratio_min": 0.7,
    "review_ratio_max": 0.5,
    "rejected_ratio_max": 0.2,
    "clip_score_min": 0.55,
}

CLIP_SCORE_COLUMNS = ["clip_score", "image_text_similarity", "similarity_score"]

EXPORT_ASSETS = {
    "train.jsonl": "普通训练集",
    "val.jsonl": "验证集",
    "eval.jsonl": "评测集",
    "train_sft.jsonl": "监督微调数据",
    "creative_image_sft.jsonl": "图像创意任务 SFT 数据",
    "creative_video_sft.jsonl": "视频创意任务 SFT 数据",
    "review_samples.jsonl": "人工复核队列",
    "rejected_samples.jsonl": "拒绝样本留痕",
}


def localize_status(status: object) -> str:
    value = "" if status is None else str(status)
    return STATUS_LABELS.get(value, value)


def _status_column(df: pd.DataFrame) -> str | None:
    for column in ["final_status", "status", "filter_status"]:
        if column in df.columns:
            return column
    return None


def _clip_score_column(df: pd.DataFrame) -> str | None:
    for column in CLIP_SCORE_COLUMNS:
        if column in df.columns:
            return column
    return None


def _numeric_mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or df.empty:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 4)


def _count_status(df: pd.DataFrame, status: str) -> int:
    column = _status_column(df)
    if column is None:
        return 0
    return int((df[column].fillna("").astype(str) == status).sum())


def _mean_or_zero(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    return round(float(pd.to_numeric(df[column], errors="coerce").fillna(0).mean()), 4)


def build_dashboard_summary(df: pd.DataFrame) -> dict[str, float | int]:
    summary = build_governance_summary(df)
    return {
        "total": summary["total"],
        "accepted": summary["accepted"],
        "review": summary["review"],
        "rejected": summary["rejected"],
        "acceptance_rate": summary["accepted_rate"],
        "review_rate": summary["review_rate"],
        "rejection_rate": summary["rejected_rate"],
        "avg_quality_score": summary["avg_quality_score"] or 0.0,
        "avg_similarity": summary["avg_clip_score"] or 0.0,
    }


def build_governance_summary(df: pd.DataFrame, export_dir: str | Path | None = None) -> dict[str, Any]:
    total = len(df)
    accepted = _count_status(df, "accepted")
    review = _count_status(df, "review")
    rejected = _count_status(df, "rejected")
    clip_column = _clip_score_column(df)
    duplicate_ratio = _near_duplicate_ratio(df)
    export_asset_count = 0
    if export_dir is not None:
        inventory = export_inventory(export_dir)
        export_asset_count = int(inventory["exists"].sum()) if "exists" in inventory.columns else 0
    return {
        "total": total,
        "accepted": accepted,
        "review": review,
        "rejected": rejected,
        "accepted_rate": round(accepted / total, 4) if total else 0.0,
        "review_rate": round(review / total, 4) if total else 0.0,
        "rejected_rate": round(rejected / total, 4) if total else 0.0,
        "avg_quality_score": _numeric_mean(df, "final_quality_score"),
        "avg_clip_score": _numeric_mean(df, clip_column) if clip_column else None,
        "clip_score_column": clip_column,
        "near_duplicate_rate": duplicate_ratio,
        "export_asset_count": export_asset_count,
    }


def _near_duplicate_ratio(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    if "is_duplicate_image" in df.columns:
        values = df["is_duplicate_image"].fillna(False).astype(bool)
        return round(float(values.mean()), 4)
    if "duplicate_group_size" in df.columns:
        values = pd.to_numeric(df["duplicate_group_size"], errors="coerce").fillna(1)
        return round(float((values > 1).mean()), 4)
    return 0.0


def build_status_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    column = _status_column(df)
    if column is None or df.empty:
        return pd.DataFrame({"status": [], "samples": [], "rate": []})
    counts = df[column].fillna("").astype(str).value_counts()
    rows = []
    total = len(df)
    for status in STATUS_ORDER:
        samples = int(counts.get(status, 0))
        if samples:
            rows.append(
                {
                    "status": status,
                    "status_label": localize_status(status),
                    "samples": samples,
                    "rate": round(samples / total, 4),
                }
            )
    for status, samples in counts.items():
        if status and status not in STATUS_ORDER:
            rows.append(
                {
                    "status": str(status),
                    "status_label": localize_status(status),
                    "samples": int(samples),
                    "rate": round(int(samples) / total, 4),
                }
            )
    return pd.DataFrame(rows)


def explode_filter_reasons(df: pd.DataFrame) -> pd.Series:
    if "filter_reason" not in df.columns:
        return pd.Series(dtype="int64")
    return (
        df["filter_reason"]
        .fillna("")
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
    )


def filter_reason_topn(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    reasons = explode_filter_reasons(df).head(limit)
    return pd.DataFrame(
        {
            "filter_reason": reasons.index.astype(str),
            "samples": reasons.astype(int).values,
        }
    )


def score_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "image_quality_score",
        "text_quality_score",
        "clip_score",
        "image_text_similarity",
        "similarity_score",
        "final_quality_score",
    ]
    return [column for column in candidates if column in df.columns and pd.api.types.is_numeric_dtype(df[column])]


def top_tags_from_column(df: pd.DataFrame, column: str = "tags", limit: int = 20) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="int64")
    return (
        df[column]
        .fillna("")
        .astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.split(",")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(limit)
    )


def load_jsonl_rows(path: str | Path) -> pd.DataFrame:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return pd.DataFrame()
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _empty_run_center_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": pd.Series(dtype="string"),
            "input_samples": pd.Series(dtype="int64"),
            "output_samples": pd.Series(dtype="int64"),
            "filtered_samples": pd.Series(dtype="int64"),
            "filter_rate": pd.Series(dtype="float64"),
            "quality_report_path": pd.Series(dtype="string"),
        }
    )


def load_run_center_rows(runs_dir: str | Path) -> pd.DataFrame:
    root = Path(runs_dir)
    if not root.exists() or not root.is_dir():
        return _empty_run_center_frame()

    rows: list[dict[str, Any]] = []
    for run_dir in sorted((item for item in root.iterdir() if item.is_dir()), reverse=True):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        samples = manifest.get("samples", {}) if isinstance(manifest.get("samples"), dict) else {}
        total = int(samples.get("total", 0) or 0)
        accepted = int(samples.get("accepted", 0) or 0)
        review = int(samples.get("review", 0) or 0)
        rejected = int(samples.get("rejected", 0) or 0)
        filtered = review + rejected
        rows.append(
            {
                "run_id": str(manifest.get("run_id") or run_dir.name),
                "input_samples": total,
                "output_samples": accepted,
                "filtered_samples": filtered,
                "filter_rate": round(filtered / total, 4) if total else 0.0,
                "quality_report_path": str(run_dir / "quality_report.json"),
            }
        )
    if not rows:
        return _empty_run_center_frame()
    return pd.DataFrame(rows)


def _count_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame({column: pd.Series(dtype="string"), "samples": pd.Series(dtype="int64")})
    counts = df[column].fillna("").astype(str).replace("", pd.NA).dropna().value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")
    return pd.DataFrame({column: counts.index.astype(str), "samples": counts.astype(int).values})


def load_feedback_loop_artifacts(feedback_dir: str | Path) -> dict[str, Any]:
    root = Path(feedback_dir)
    report_path = root / "evaluation_feedback_report.md"
    manifest_path = root / "second_round_manifest.jsonl"
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    manifest = load_jsonl_rows(manifest_path)
    return {
        "report_text": report_text,
        "second_round_manifest": manifest,
        "error_type_stats": _count_column(manifest, "error_type"),
        "recommendation_stats": _count_column(manifest, "recommendation"),
    }


def review_queue(df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    column = _status_column(df)
    if column is None:
        return pd.DataFrame()
    queue = df[df[column].fillna("").astype(str) == "review"].copy()
    if "final_quality_score" in queue.columns:
        queue = queue.sort_values("final_quality_score", ascending=True)
    return queue.head(limit)


def diagnostic_samples(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    result = df.copy()
    column = _status_column(result)
    if column is not None:
        status_order = pd.CategoricalDtype(["rejected", "review", "accepted"], ordered=True)
        result["_status_sort"] = result[column].fillna("").astype(str).astype(status_order)
        result["status_label"] = result[column].map(localize_status)
    else:
        result["_status_sort"] = pd.NA
        result["status_label"] = ""
    if "final_quality_score" in result.columns:
        result["_quality_sort"] = pd.to_numeric(result["final_quality_score"], errors="coerce").fillna(1)
    else:
        result["_quality_sort"] = 1
    result = result.sort_values(["_status_sort", "_quality_sort"], ascending=[True, True]).head(limit)
    return result.drop(columns=[column for column in ["_status_sort", "_quality_sort"] if column in result.columns])


def caption_lengths(df: pd.DataFrame) -> pd.Series:
    if "caption_word_count" in df.columns:
        return pd.to_numeric(df["caption_word_count"], errors="coerce").fillna(0).astype(int)
    if "caption" not in df.columns:
        return pd.Series([0] * len(df), dtype="int64")
    return df["caption"].fillna("").astype(str).str.split().map(len).astype(int)


def build_quality_pipeline(df: pd.DataFrame, export_dir: str | Path | None = None) -> pd.DataFrame:
    total = len(df)
    summary = build_governance_summary(df)

    # The source metadata records separate image/text rule results and final status,
    # but not every intermediate governance stage. Missing stages are approximated
    # from the closest available fields so the workbench stays comparable across
    # demo, heuristic COCO, and CLIP-enriched COCO versions.
    image_pass = _count_bool_or_status(df, bool_column="image_valid", status_column="image_filter_status")
    text_pass = _count_bool_or_status(df, bool_column="text_valid", status_column="text_filter_status")
    consistent_or_review = int(summary["accepted"]) + int(summary["review"])
    exported_train = exported_eval = exported_sft = 0
    if export_dir is not None:
        inventory = export_inventory(export_dir).set_index("file")
        exported_train = _inventory_rows(inventory, ["train.jsonl"])
        exported_eval = _inventory_rows(inventory, ["eval.jsonl"])
        exported_sft = _inventory_rows(
            inventory,
            ["train_sft.jsonl", "creative_image_sft.jsonl", "creative_video_sft.jsonl"],
        )

    rows = [
        ("原始样本", total, "输入 manifest 或元数据中的全部样本。"),
        ("图片可读 / 图片质量通过", image_pass, "优先使用 image_valid，否则使用 image_filter_status 近似。"),
        ("文本有效", text_pass, "优先使用 text_valid，否则使用 text_filter_status 近似。"),
        ("图文一致性达标或进入复核", consistent_or_review, "使用 accepted + review 近似图文一致性通过或边界样本。"),
        ("Accepted", int(summary["accepted"]), "可进入训练或评测导出的样本。"),
        ("Review", int(summary["review"]), "自动规则无法直接判定的人工复核样本。"),
        ("Rejected", int(summary["rejected"]), "被规则或质量分拒绝的样本。"),
        ("已导出 Train", exported_train, "train.jsonl 的样本行数。"),
        ("已导出 Eval", exported_eval, "eval.jsonl 的样本行数。"),
        ("已导出 SFT", exported_sft, "train_sft 与创意 SFT JSONL 的样本行数。"),
    ]
    return pd.DataFrame(
        [
            {
                "stage": stage,
                "samples": int(samples),
                "rate": round(samples / total, 4) if total else 0.0,
                "note": note,
            }
            for stage, samples, note in rows
        ]
    )


def _count_bool_or_status(df: pd.DataFrame, bool_column: str, status_column: str) -> int:
    if bool_column in df.columns:
        return int(df[bool_column].fillna(False).astype(bool).sum())
    if status_column in df.columns:
        return int((df[status_column].fillna("").astype(str) == "accepted").sum())
    return len(df)


def _inventory_rows(inventory: pd.DataFrame, files: list[str]) -> int:
    total = 0
    for file_name in files:
        if file_name in inventory.index:
            total += int(inventory.loc[file_name, "rows"])
    return total


def build_quality_alerts(
    summary: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, str]]:
    limits = thresholds or ALERT_THRESHOLDS
    alerts: list[dict[str, str]] = []
    if float(summary.get("accepted_rate", 0.0)) < limits["accepted_ratio_min"]:
        alerts.append(
            {
                "level": "warning",
                "alert": "Accepted 比例低于阈值",
                "possible_causes": "CLIP 阈值过高、caption 质量偏低、图片质量问题较多",
                "action": "进入诊断页查看 CLIP 分数分布和 filter_reason TopN",
            }
        )
    if float(summary.get("review_rate", 0.0)) > limits["review_ratio_max"]:
        alerts.append(
            {
                "level": "warning",
                "alert": "Review 样本占比过高",
                "possible_causes": "边界样本较多，自动规则无法直接判定",
                "action": "导出 review_samples.jsonl，进入人工复核流程",
            }
        )
    if float(summary.get("rejected_rate", 0.0)) > limits["rejected_ratio_max"]:
        alerts.append(
            {
                "level": "error",
                "alert": "Rejected 样本占比过高",
                "possible_causes": "图片损坏、低分辨率、caption 过短或为空",
                "action": "检查主要过滤原因和样本案例",
            }
        )
    clip_score = summary.get("avg_clip_score")
    if clip_score is not None and float(clip_score) < limits["clip_score_min"]:
        alerts.append(
            {
                "level": "error",
                "alert": "平均图文相似度偏低",
                "possible_causes": "图文不匹配、caption 过泛、CLIP 模型或预处理配置不一致",
                "action": "抽查低分样本并调整 review 阈值",
            }
        )
    return alerts


def export_inventory(export_dir: str | Path) -> pd.DataFrame:
    path = Path(export_dir)
    rows = []
    for file_name, purpose in EXPORT_ASSETS.items():
        item = path / file_name
        exists = item.is_file()
        stat = item.stat() if exists else None
        rows.append(
            {
                "file": file_name,
                "exists": bool(exists),
                "status_label": "存在" if exists else "不存在",
                "size_bytes": int(stat.st_size) if stat else 0,
                "size_kb": round(int(stat.st_size) / 1024, 2) if stat else 0.0,
                "rows": _count_file_lines(item) if exists else 0,
                "updated_at": pd.Timestamp(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S") if stat else "",
                "purpose": purpose,
            }
        )
    return pd.DataFrame(rows)


def _count_file_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
