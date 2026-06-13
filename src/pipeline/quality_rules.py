from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QualityRules:
    image: dict[str, Any] = field(default_factory=dict)
    text: dict[str, Any] = field(default_factory=dict)
    score: dict[str, Any] = field(default_factory=dict)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "[]":
        return []
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def _fallback_parse_yaml(text: str) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    current_section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            data[current_section] = {}
            continue
        if current_section and ":" in line:
            key, value = line.strip().split(":", 1)
            data[current_section][key.strip()] = _parse_scalar(value)
    return data


def load_quality_rules(path: str | Path | None) -> QualityRules:
    if path is None:
        return QualityRules()

    rules_path = Path(path)
    if not rules_path.exists():
        raise FileNotFoundError(f"Quality rules not found: {rules_path}")

    text = rules_path.read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(text) or {}
    except Exception:
        payload = _fallback_parse_yaml(text)

    return QualityRules(
        image=dict(payload.get("image", {}) or {}),
        text=dict(payload.get("text", {}) or {}),
        score=dict(payload.get("score", {}) or {}),
    )
