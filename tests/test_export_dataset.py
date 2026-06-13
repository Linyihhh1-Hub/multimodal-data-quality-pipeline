import json
from pathlib import Path

import pandas as pd

from src.pipeline.build_dataset import assign_splits
from src.pipeline.export_sft import export_caption_jsonl, export_sft_jsonl


def test_assign_splits_is_deterministic():
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(10)],
            "filter_status": ["accepted"] * 10,
        }
    )

    first = assign_splits(df, train_ratio=0.7, val_ratio=0.2, seed=7)
    second = assign_splits(df, train_ratio=0.7, val_ratio=0.2, seed=7)

    assert first["split"].tolist() == second["split"].tolist()
    assert set(first["split"]) == {"train", "val", "eval"}


def test_export_caption_jsonl_writes_only_accepted_rows(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "image_path": "images/a.jpg",
                "caption": "A clear caption.",
                "filter_status": "accepted",
                "split": "train",
            },
            {
                "image_path": "images/b.jpg",
                "caption": "Bad.",
                "filter_status": "rejected",
                "split": "train",
            },
        ]
    )
    output = tmp_path / "train.jsonl"

    export_caption_jsonl(df, output, split="train")

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"image": "images/a.jpg", "caption": "A clear caption."}]


def test_export_sft_jsonl_uses_vlm_message_format(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "image_path": "images/a.jpg",
                "caption": "A clear caption.",
                "filter_status": "accepted",
                "split": "train",
            }
        ]
    )
    output = tmp_path / "train_sft.jsonl"

    export_sft_jsonl(df, output, split="train")

    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row["images"] == ["images/a.jpg"]
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][1]["content"] == "A clear caption."
