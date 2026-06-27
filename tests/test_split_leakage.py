from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.quality.split_leakage import detect_split_leakage, write_leakage_report


ROOT = Path(".tmp/split-leakage-tests")


def _reset_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def _base_rows() -> list[dict]:
    return [
        {
            "sample_id": "train_1",
            "image_id": "img_train_1",
            "image_path": "images/train_1.jpg",
            "perceptual_hash": "hash_train_1",
            "duplicate_group_id": "group_train_1",
            "split": "train",
        },
        {
            "sample_id": "eval_1",
            "image_id": "img_eval_1",
            "image_path": "images/eval_1.jpg",
            "perceptual_hash": "hash_eval_1",
            "duplicate_group_id": "group_eval_1",
            "split": "eval",
        },
    ]


def _detect_with_shared_value(key: str) -> dict:
    rows = _base_rows()
    rows[0][key] = "shared_value"
    rows[1][key] = "shared_value"
    return detect_split_leakage(pd.DataFrame(rows))


def test_no_leakage_returns_false():
    report = detect_split_leakage(pd.DataFrame(_base_rows()))

    assert report["has_leakage"] is False
    assert report["total_leakage_count"] == 0
    assert report["leakage_by_key"] == {
        "sample_id": 0,
        "image_id": 0,
        "image_path": 0,
        "perceptual_hash": 0,
        "duplicate_group_id": 0,
    }
    assert report["leakage_examples"] == []


def test_sample_id_leakage_is_detected():
    report = _detect_with_shared_value("sample_id")

    assert report["has_leakage"] is True
    assert report["leakage_by_key"]["sample_id"] == 1
    assert report["leakage_examples"][0]["leakage_key"] == "sample_id"


def test_image_id_leakage_is_detected():
    report = _detect_with_shared_value("image_id")

    assert report["has_leakage"] is True
    assert report["leakage_by_key"]["image_id"] == 1


def test_image_path_leakage_is_detected():
    report = _detect_with_shared_value("image_path")

    assert report["has_leakage"] is True
    assert report["leakage_by_key"]["image_path"] == 1


def test_perceptual_hash_leakage_is_detected():
    report = _detect_with_shared_value("perceptual_hash")

    assert report["has_leakage"] is True
    assert report["leakage_by_key"]["perceptual_hash"] == 1


def test_duplicate_group_id_leakage_is_detected():
    report = _detect_with_shared_value("duplicate_group_id")

    assert report["has_leakage"] is True
    assert report["leakage_by_key"]["duplicate_group_id"] == 1


def test_missing_optional_keys_do_not_crash():
    df = pd.DataFrame(
        [
            {"sample_id": "a", "split": "train"},
            {"sample_id": "b", "split": "eval"},
        ]
    )

    report = detect_split_leakage(df)

    assert report["has_leakage"] is False
    assert report["checked_keys"] == ["sample_id"]
    assert report["missing_keys"] == ["image_id", "image_path", "perceptual_hash", "duplicate_group_id"]
    assert report["leakage_by_key"]["sample_id"] == 0


def test_leakage_examples_are_limited_to_20():
    rows: list[dict] = []
    for index in range(25):
        rows.append({"sample_id": f"same_{index}", "image_path": f"train/{index}.jpg", "split": "train"})
        rows.append({"sample_id": f"same_{index}", "image_path": f"eval/{index}.jpg", "split": "eval"})

    report = detect_split_leakage(pd.DataFrame(rows), max_examples=20)

    assert report["has_leakage"] is True
    assert report["total_leakage_count"] == 25
    assert report["leakage_by_key"]["sample_id"] == 25
    assert len(report["leakage_examples"]) == 20


def test_write_report_and_cli_parquet_are_usable():
    _reset_dir(ROOT)
    metadata_path = ROOT / "metadata.parquet"
    output_path = ROOT / "split_leakage_report.json"
    cli_output_path = ROOT / "split_leakage_report_cli.json"
    pd.DataFrame(_base_rows()).to_parquet(metadata_path, index=False)

    write_leakage_report(detect_split_leakage(pd.read_parquet(metadata_path)), output_path)
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["has_leakage"] is False

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.quality.split_leakage",
            "--metadata",
            str(metadata_path),
            "--output",
            str(cli_output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cli_report = json.loads(cli_output_path.read_text(encoding="utf-8"))
    assert cli_report["checked_keys"] == [
        "sample_id",
        "image_id",
        "image_path",
        "perceptual_hash",
        "duplicate_group_id",
    ]
    assert "has_leakage: False" in result.stdout
