from src.quality.quality_scorer import combine_quality


def test_combine_quality_accepts_high_quality_sample():
    result = combine_quality(
        image_status="accepted",
        text_status="accepted",
        image_reasons=[],
        text_reasons=[],
        image_quality_score=0.9,
        text_quality_score=0.9,
        image_text_similarity=0.8,
        accept_threshold=0.75,
        review_threshold=0.55,
    )

    assert result["filter_status"] == "accepted"
    assert result["final_quality_score"] == 0.86
    assert result["filter_reason"] == ""


def test_combine_quality_rejects_when_component_rejected():
    result = combine_quality(
        image_status="rejected",
        text_status="accepted",
        image_reasons=["low_resolution"],
        text_reasons=[],
        image_quality_score=0.2,
        text_quality_score=0.9,
        image_text_similarity=0.8,
    )

    assert result["filter_status"] == "rejected"
    assert result["filter_reason"] == "low_resolution"


def test_combine_quality_sends_borderline_sample_to_review():
    result = combine_quality(
        image_status="accepted",
        text_status="accepted",
        image_reasons=[],
        text_reasons=[],
        image_quality_score=0.7,
        text_quality_score=0.7,
        image_text_similarity=0.45,
        accept_threshold=0.75,
        review_threshold=0.55,
    )

    assert result["filter_status"] == "review"
    assert "borderline_quality_score" in result["filter_reason"]
