from __future__ import annotations

import re
from collections import Counter

import pandas as pd


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "near",
    "of",
    "on",
    "the",
    "to",
    "with",
    "while",
    "another",
}

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")


def extract_caption_tags(caption: str, max_tags: int = 6) -> list[str]:
    seen: set[str] = set()
    tags: list[str] = []
    for token in TOKEN_RE.findall(str(caption).lower()):
        token = token.strip("'")
        if token in STOPWORDS or len(token) < 3 or token in seen:
            continue
        seen.add(token)
        tags.append(token)
        if len(tags) >= max_tags:
            break
    return tags


def summarize_tag_distribution(df: pd.DataFrame, top_k: int = 30) -> dict:
    counter: Counter[str] = Counter()
    tagged_samples = 0
    for caption in df["caption"].fillna(""):
        tags = extract_caption_tags(str(caption))
        if tags:
            tagged_samples += 1
        counter.update(tags)

    return {
        "total_samples": int(len(df)),
        "tagged_samples": int(tagged_samples),
        "top_tags": [{"tag": tag, "count": int(count)} for tag, count in counter.most_common(top_k)],
    }


def add_caption_tags(df: pd.DataFrame, max_tags: int = 6) -> pd.DataFrame:
    result = df.copy()
    result["caption_tags"] = result["caption"].fillna("").map(lambda text: extract_caption_tags(str(text), max_tags=max_tags))
    result["tag_count"] = result["caption_tags"].map(len)
    return result
