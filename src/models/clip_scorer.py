from __future__ import annotations

import hashlib
from pathlib import Path


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
            outputs = self._model(**inputs)
            similarity = outputs.logits_per_image.softmax(dim=1)[0][0].item()
        return round(float(similarity), 4)

    def _score_with_heuristic(self, image_path: str | Path, caption: str) -> float:
        text = (caption or "").strip()
        if not Path(image_path).exists() or not text:
            return 0.0
        digest = hashlib.sha256(f"{Path(image_path).name}|{text}".encode("utf-8")).digest()
        stable = int.from_bytes(digest[:2], "big") / 65535
        word_bonus = min(len(text.split()) / 12, 1.0) * 0.25
        return round(min(0.95, 0.45 + word_bonus + stable * 0.2), 4)
