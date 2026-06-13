from src.quality.text_quality import assess_caption


def test_assess_caption_accepts_normal_caption():
    result = assess_caption("A man riding a bike down a city street.")

    assert result["text_valid"] is True
    assert result["filter_status"] == "accepted"
    assert result["text_quality_score"] > 0.8


def test_assess_caption_rejects_empty_caption():
    result = assess_caption("   ")

    assert result["text_valid"] is False
    assert result["filter_status"] == "rejected"
    assert "empty_caption" in result["filter_reasons"]


def test_assess_caption_marks_short_caption_for_review():
    result = assess_caption("bike")

    assert result["text_valid"] is True
    assert result["filter_status"] == "review"
    assert "short_caption" in result["filter_reasons"]
