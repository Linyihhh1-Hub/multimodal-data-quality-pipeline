from src.models.clip_scorer import cosine_to_unit_score


def test_cosine_to_unit_score_maps_cosine_range_to_zero_one():
    assert cosine_to_unit_score(-1.0) == 0.0
    assert cosine_to_unit_score(0.0) == 0.5
    assert cosine_to_unit_score(1.0) == 1.0


def test_cosine_to_unit_score_clamps_out_of_range_values():
    assert cosine_to_unit_score(-2.0) == 0.0
    assert cosine_to_unit_score(2.0) == 1.0
