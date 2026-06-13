from pathlib import Path

from src.pipeline.quality_rules import load_quality_rules


def test_load_quality_rules_reads_thresholds(tmp_path: Path):
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        "\n".join(
            [
                "image:",
                "  min_width: 300",
                "text:",
                "  min_words: 5",
                "score:",
                "  accept_threshold: 0.9",
                "  review_threshold: 0.7",
            ]
        ),
        encoding="utf-8",
    )

    rules = load_quality_rules(rules_path)

    assert rules.image["min_width"] == 300
    assert rules.text["min_words"] == 5
    assert rules.score["accept_threshold"] == 0.9
    assert rules.score["review_threshold"] == 0.7
