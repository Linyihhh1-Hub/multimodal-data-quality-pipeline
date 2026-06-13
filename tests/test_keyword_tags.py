import pandas as pd

from src.analysis.keyword_tags import extract_caption_tags, summarize_tag_distribution


def test_extract_caption_tags_filters_stopwords_and_limits_tags():
    tags = extract_caption_tags("A person riding a bicycle on a city street with another person.", max_tags=4)

    assert tags == ["person", "riding", "bicycle", "city"]


def test_summarize_tag_distribution_counts_tags_per_sample():
    df = pd.DataFrame(
        [
            {"caption": "A person riding a bicycle on a city street."},
            {"caption": "A dog sits on a city street."},
        ]
    )

    summary = summarize_tag_distribution(df, top_k=3)

    assert summary["total_samples"] == 2
    assert summary["tagged_samples"] == 2
    assert summary["top_tags"][0] == {"tag": "city", "count": 2}
