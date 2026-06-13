from __future__ import annotations

from collections import Counter

import pandas as pd


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _reason_counts(df: pd.DataFrame) -> Counter:
    counter: Counter = Counter()
    for value in df["filter_reason"].fillna(""):
        for reason in str(value).split(";"):
            reason = reason.strip()
            if reason:
                counter[reason] += 1
    return counter


def build_quality_report(df: pd.DataFrame, version: str) -> str:
    total = len(df)
    accepted = int((df["filter_status"] == "accepted").sum())
    review = int((df["filter_status"] == "review").sum())
    rejected = int((df["filter_status"] == "rejected").sum())
    acceptance_rate = accepted / total if total else 0.0
    avg_similarity = float(df["image_text_similarity"].mean()) if total else 0.0
    avg_score = float(df["final_quality_score"].mean()) if total else 0.0

    lines = [
        f"# Multimodal Data Quality Report - {version}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total samples | {total} |",
        f"| Accepted samples | {accepted} |",
        f"| Review samples | {review} |",
        f"| Rejected samples | {rejected} |",
        f"| Acceptance rate | {_percent(acceptance_rate)} |",
        f"| Average image-text similarity | {avg_similarity:.4f} |",
        f"| Average final quality score | {avg_score:.4f} |",
        "",
        "## Filter Reason Distribution",
        "",
        "| Reason | Count |",
        "| --- | ---: |",
    ]

    reason_counts = _reason_counts(df)
    if reason_counts:
        for reason, count in reason_counts.most_common():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Lowest Quality Samples",
            "",
            "| sample_id | status | score | similarity | reasons | caption |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    low_quality = df.sort_values("final_quality_score").head(10)
    for _, row in low_quality.iterrows():
        caption = str(row.get("caption", "")).replace("|", "\\|")
        if len(caption) > 80:
            caption = caption[:77] + "..."
        lines.append(
            "| {sample_id} | {status} | {score:.4f} | {sim:.4f} | {reason} | {caption} |".format(
                sample_id=row.get("sample_id", ""),
                status=row.get("filter_status", ""),
                score=float(row.get("final_quality_score", 0.0)),
                sim=float(row.get("image_text_similarity", 0.0)),
                reason=str(row.get("filter_reason", "")),
                caption=caption,
            )
        )

    lines.extend(
        [
            "",
            "## Interview Notes",
            "",
            "- This report turns raw filtering metadata into quality evidence: pass rate, failure modes, and low-quality examples.",
            "- Use the reason distribution to explain rule iteration, such as tightening image-text similarity or moving borderline samples to review.",
            "- Use the lowest-quality table to discuss concrete cases instead of describing cleaning rules only in abstract terms.",
        ]
    )
    return "\n".join(lines) + "\n"
