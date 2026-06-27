from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


def create_run_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _to_yaml_lines(payload: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(payload, dict):
        lines: list[str] = []
        for key, value in payload.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(_to_yaml_lines(value, indent + 2))
            elif isinstance(value, list):
                if value:
                    lines.append(f"{prefix}{key}:")
                    for item in value:
                        lines.append(f"{prefix}  - {item}")
                else:
                    lines.append(f"{prefix}{key}: []")
            elif value is None:
                lines.append(f"{prefix}{key}: null")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return lines
    return [f"{prefix}{payload}"]


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        text = yaml.safe_dump(_json_safe(payload), allow_unicode=True, sort_keys=False)
    except Exception:
        text = "\n".join(_to_yaml_lines(_json_safe(payload))) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _reason_counts(df: pd.DataFrame) -> dict[str, int]:
    counter: Counter[str] = Counter()
    if "filter_reason" not in df.columns:
        return {}
    for value in df["filter_reason"].fillna(""):
        for reason in str(value).split(";"):
            reason = reason.strip()
            if reason:
                counter[reason] += 1
    return dict(counter.most_common())


def build_quality_stats(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    status_counts = {
        "accepted": int((df["filter_status"] == "accepted").sum()) if "filter_status" in df.columns else 0,
        "review": int((df["filter_status"] == "review").sum()) if "filter_status" in df.columns else 0,
        "rejected": int((df["filter_status"] == "rejected").sum()) if "filter_status" in df.columns else 0,
    }
    return {
        "total_samples": total,
        "status_counts": status_counts,
        "acceptance_rate": status_counts["accepted"] / total if total else 0.0,
        "average_final_quality_score": float(df["final_quality_score"].mean())
        if total and "final_quality_score" in df.columns
        else 0.0,
        "average_image_text_similarity": float(df["image_text_similarity"].mean())
        if total and "image_text_similarity" in df.columns
        else 0.0,
        "filter_reason_counts": _reason_counts(df),
    }


def _write_filtered_samples(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = df[df["filter_status"] == "accepted"] if "filter_status" in df.columns else df.iloc[0:0]
    with output_path.open("w", encoding="utf-8") as handle:
        for _, row in accepted.iterrows():
            handle.write(json.dumps(_json_safe(row.to_dict()), ensure_ascii=False) + "\n")
    return output_path


def _build_summary(run_id: str, manifest: dict[str, Any], quality_stats: dict[str, Any]) -> str:
    samples = manifest["samples"]
    lines = [
        f"# Pipeline Run {run_id}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total samples | {samples['total']} |",
        f"| Accepted samples | {samples['accepted']} |",
        f"| Review samples | {samples['review']} |",
        f"| Rejected samples | {samples['rejected']} |",
        f"| Acceptance rate | {quality_stats['acceptance_rate'] * 100:.2f}% |",
        f"| Average final quality score | {quality_stats['average_final_quality_score']:.4f} |",
        f"| Average image-text similarity | {quality_stats['average_image_text_similarity']:.4f} |",
        "",
        "## Inputs",
        "",
    ]
    for name, value in manifest["inputs"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Outputs", ""])
    for name, value in manifest["outputs"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Filter Reasons", "", "| Reason | Count |", "| --- | ---: |"])
    reasons = quality_stats["filter_reason_counts"]
    if reasons:
        for reason, count in reasons.items():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| none | 0 |")
    return "\n".join(lines) + "\n"


class RunManager:
    def __init__(self, runs_dir: str | Path = "outputs/runs", run_id: str | None = None) -> None:
        self.runs_dir = Path(runs_dir)
        self.run_id = run_id or create_run_id()
        self.run_dir = self.runs_dir / self.run_id

    def archive(
        self,
        scored: pd.DataFrame,
        config: dict[str, Any],
        input_paths: dict[str, Any],
        output_paths: dict[str, Any],
        quality_rules: dict[str, Any],
        scorer_backend: str,
    ) -> dict[str, Path]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        quality_stats = build_quality_stats(scored)
        status_counts = quality_stats["status_counts"]
        manifest = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": input_paths,
            "outputs": output_paths,
            "samples": {
                "total": quality_stats["total_samples"],
                "accepted": status_counts["accepted"],
                "review": status_counts["review"],
                "rejected": status_counts["rejected"],
            },
            "filtered_samples": {
                "accepted": status_counts["accepted"],
                "filtered_out": status_counts["review"] + status_counts["rejected"],
            },
            "quality_rules": quality_rules,
            "quality_summary": quality_stats,
            "scorer_backend": scorer_backend,
        }

        outputs = {
            "config_snapshot": _write_yaml(self.run_dir / "config_snapshot.yaml", config),
            "manifest": _write_json(self.run_dir / "manifest.json", manifest),
            "quality_report": _write_json(self.run_dir / "quality_report.json", quality_stats),
            "filtered_samples": _write_filtered_samples(scored, self.run_dir / "filtered_samples.jsonl"),
        }
        outputs["run_summary"] = self.run_dir / "run_summary.md"
        outputs["run_summary"].write_text(_build_summary(self.run_id, manifest, quality_stats), encoding="utf-8")
        return outputs
