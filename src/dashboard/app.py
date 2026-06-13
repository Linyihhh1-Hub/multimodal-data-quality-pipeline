from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.storage.metadata_store import read_metadata


st.set_page_config(page_title="Multimodal Data Quality", layout="wide")
st.title("Multimodal Data Quality Dashboard")

default_path = Path("data/processed/processed_metadata_v1.0.parquet")
metadata_path = st.sidebar.text_input("Metadata path", str(default_path))

path = Path(metadata_path)
if not path.exists():
    st.warning(f"Metadata file not found: {path}")
    st.stop()

df = read_metadata(path)

total = len(df)
accepted = int((df["filter_status"] == "accepted").sum())
review = int((df["filter_status"] == "review").sum())
rejected = int((df["filter_status"] == "rejected").sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total samples", total)
col2.metric("Accepted", accepted)
col3.metric("Review", review)
col4.metric("Rejected", rejected)

left, right = st.columns(2)
with left:
    st.subheader("Filter status")
    st.bar_chart(df["filter_status"].value_counts())

with right:
    st.subheader("Filter reasons")
    reasons = (
        df["filter_reason"]
        .fillna("")
        .str.split(";")
        .explode()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
    )
    st.bar_chart(reasons)

st.subheader("Score distributions")
score_cols = [col for col in ["image_text_similarity", "final_quality_score"] if col in df.columns]
if score_cols:
    st.line_chart(df[score_cols])

st.subheader("Samples for review")
preview_cols = [
    "sample_id",
    "image_path",
    "caption",
    "image_text_similarity",
    "final_quality_score",
    "filter_status",
    "filter_reason",
]
st.dataframe(df[preview_cols].sort_values("final_quality_score").head(100), use_container_width=True)
