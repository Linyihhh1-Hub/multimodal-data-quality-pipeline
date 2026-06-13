from __future__ import annotations

import hashlib
from pathlib import Path


def cosine_to_unit_score(cosine_similarity: float) -> float:
    return round(max(0.0, min(1.0, (float(cosine_similarity) + 1.0) / 2.0)), 4)


class ClipScorer:
    """Optional CLIP scorer with a deterministic fallback for local demos/tests."""

    def __init__(self, use_clip: bool = True):
        self.use_clip = use_clip
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = "cpu"
        if use_clip:
            self._try_load_clip()

    @property
    def backend(self) -> str:
        return "clip" if self._model is not None else "heuristic"

    def _try_load_clip(self) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor

            self._torch = torch
            self._Image = Image
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self._device)
            self._preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception:
            self._model = None
            self._preprocess = None

    def score(self, image_path: str | Path, caption: str) -> float:
        if self._model is not None and self._preprocess is not None:
            return self._score_with_clip(image_path, caption)
        return self._score_with_heuristic(image_path, caption)

    def _score_with_clip(self, image_path: str | Path, caption: str) -> float:
        image = self._Image.open(image_path).convert("RGB")
        inputs = self._preprocess(text=[caption], images=image, return_tensors="pt", padding=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.no_grad():
            image_features = self._model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = self._model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            cosine = (image_features * text_features).sum(dim=-1).item()
        return cosine_to_unit_score(cosine)

    def _score_with_heuristic(self, image_path: str | Path, caption: str) -> float:
        text = (caption or "").strip()
        if not Path(image_path).exists() or not text:
            return 0.0
        digest = hashlib.sha256(f"{Path(image_path).name}|{text}".encode("utf-8")).digest()
        stable = int.from_bytes(digest[:2], "big") / 65535
        word_bonus = min(len(text.split()) / 12, 1.0) * 0.25
        return round(min(0.95, 0.45 + word_bonus + stable * 0.2), 4)
