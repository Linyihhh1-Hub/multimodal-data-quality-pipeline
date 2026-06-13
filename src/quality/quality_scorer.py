from __future__ import annotations


def combine_quality(
    image_status: str,
    text_status: str,
    image_reasons: list[str],
    text_reasons: list[str],
    image_quality_score: float,
    text_quality_score: float,
    image_text_similarity: float,
    accept_threshold: float = 0.75,
    review_threshold: float = 0.55,
) -> dict:
    final_score = round(
        0.35 * float(image_quality_score)
        + 0.25 * float(text_quality_score)
        + 0.40 * float(image_text_similarity),
        4,
    )
    reasons = list(image_reasons) + list(text_reasons)

    if image_status == "rejected" or text_status == "rejected":
        status = "rejected"
    elif final_score >= accept_threshold and image_status == "accepted" and text_status == "accepted":
        status = "accepted"
    elif final_score >= review_threshold:
        status = "review"
        reasons.append("borderline_quality_score")
    else:
        status = "rejected"
        reasons.append("low_quality_score")

    return {
        "final_quality_score": final_score,
        "filter_status": status,
        "filter_reason": ";".join(dict.fromkeys(reasons)),
    }
