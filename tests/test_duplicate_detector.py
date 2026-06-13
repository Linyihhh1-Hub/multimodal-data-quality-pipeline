from src.quality.duplicate_detector import annotate_duplicate_hashes


def test_annotate_duplicate_hashes_marks_group_size_without_filtering():
    rows = [
        {"sample_id": "a", "perceptual_hash": "hash1", "filter_status": "accepted"},
        {"sample_id": "b", "perceptual_hash": "hash1", "filter_status": "accepted"},
        {"sample_id": "c", "perceptual_hash": "hash2", "filter_status": "accepted"},
    ]

    annotated = annotate_duplicate_hashes(rows)

    assert annotated[0]["duplicate_group_size"] == 2
    assert annotated[0]["is_duplicate_image"] is False
    assert annotated[1]["duplicate_group_size"] == 2
    assert annotated[1]["is_duplicate_image"] is True
    assert annotated[1]["filter_status"] == "accepted"
    assert annotated[2]["duplicate_group_size"] == 1
    assert annotated[2]["is_duplicate_image"] is False
