from __future__ import annotations

import re


_PRINTABLE_RE = re.compile(r"^[\x09\x0a\x0d\x20-\x7e]+$")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def assess_caption(
    caption: str | None,
    min_words: int = 3,
    max_words: int = 80,
    sensitive_words: set[str] | None = None,
) -> dict:
    text = (caption or "").strip()
    reasons: list[str] = []
    penalties: list[float] = []

    if not text:
        return {
            "text_valid": False,
            "caption_word_count": 0,
            "text_quality_score": 0.0,
            "filter_status": "rejected",
            "filter_reasons": ["empty_caption"],
        }

    words = _WORD_RE.findall(text)
    word_count = len(words)
    status = "accepted"

    if word_count < min_words:
        reasons.append("short_caption")
        penalties.append(0.35)
        status = "review"

    if word_count > max_words:
        reasons.append("long_caption")
        penalties.append(0.25)
        status = "review"

    if not _PRINTABLE_RE.match(text):
        reasons.append("non_printable_or_non_english_text")
        penalties.append(0.45)
        status = "rejected"

    lowered = text.lower()
    for word in sensitive_words or set():
        if word.lower() in lowered:
            reasons.append("sensitive_word")
            penalties.append(0.5)
            status = "rejected"
            break

    score = round(max(0.0, min(1.0, 1.0 - sum(penalties))), 4)
    return {
        "text_valid": status != "rejected",
        "caption_word_count": word_count,
        "text_quality_score": score,
        "filter_status": status,
        "filter_reasons": reasons,
    }
