from pathlib import Path

import torch

from src.models.clip_scorer import ClipScorer, cosine_to_unit_score, extract_feature_tensor


def test_cosine_to_unit_score_maps_cosine_range_to_zero_one():
    assert cosine_to_unit_score(-1.0) == 0.0
    assert cosine_to_unit_score(0.0) == 0.5
    assert cosine_to_unit_score(1.0) == 1.0


def test_cosine_to_unit_score_clamps_out_of_range_values():
    assert cosine_to_unit_score(-2.0) == 0.0
    assert cosine_to_unit_score(2.0) == 1.0


def test_clip_scorer_uses_workspace_cache_dir_when_provided(tmp_path: Path):
    scorer = ClipScorer(use_clip=False, model_cache_dir=tmp_path / "hf")

    assert scorer.model_cache_dir == tmp_path / "hf"
    assert scorer.load_error is None


def test_extract_feature_tensor_supports_pooler_output_objects():
    class Output:
        pooler_output = torch.tensor([[1.0, 2.0]])

    result = extract_feature_tensor(Output())

    assert result.tolist() == [[1.0, 2.0]]


def test_score_batch_returns_one_score_per_pair(tmp_path: Path):
    image_path = tmp_path / "a.jpg"
    from PIL import Image

    Image.new("RGB", (32, 32), color=(80, 120, 160)).save(image_path)
    scorer = ClipScorer(use_clip=False)

    scores = scorer.score_batch([image_path, image_path], ["A caption.", "Another caption."])

    assert len(scores) == 2
    assert all(0.0 <= score <= 1.0 for score in scores)
