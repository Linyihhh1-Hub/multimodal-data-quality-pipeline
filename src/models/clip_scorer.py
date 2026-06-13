from __future__ import annotations

import hashlib
from pathlib import Path


def cosine_to_unit_score(cosine_similarity: float) -> float:
    return round(max(0.0, min(1.0, (float(cosine_similarity) + 1.0) / 2.0)), 4)


def extract_feature_tensor(model_output):
    if hasattr(model_output, "pooler_output"):
        return model_output.pooler_output
    return model_output


class ClipScorer:
    """Optional CLIP scorer with a deterministic fallback for local demos/tests."""

    def __init__(self, use_clip: bool = True, model_cache_dir: str | Path | None = "data/models/huggingface"):
        self.use_clip = use_clip
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir else None
        self.load_error: str | None = None
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
            if self.model_cache_dir:
                self.model_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_kwargs = {"cache_dir": str(self.model_cache_dir)} if self.model_cache_dir else {}
            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", **cache_kwargs).to(self._device)
            self._preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", **cache_kwargs)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            self._model = None
            self._preprocess = None

    def score(self, image_path: str | Path, caption: str) -> float:
        if self._model is not None and self._preprocess is not None:
            return self._score_with_clip(image_path, caption)
        return self._score_with_heuristic(image_path, caption)

    def score_batch(
        self,
        image_paths: list[str | Path],
        captions: list[str],
        batch_size: int = 16,
    ) -> list[float]:
        if len(image_paths) != len(captions):
            raise ValueError("image_paths and captions must have the same length")
        if self._model is None or self._preprocess is None:
            return [self._score_with_heuristic(image_path, caption) for image_path, caption in zip(image_paths, captions)]

        scores: list[float] = []
        for start in range(0, len(image_paths), batch_size):
            scores.extend(
                self._score_batch_with_clip(
                    image_paths[start : start + batch_size],
                    captions[start : start + batch_size],
                )
            )
        return scores

    def _score_with_clip(self, image_path: str | Path, caption: str) -> float:
        return self._score_batch_with_clip([image_path], [caption])[0]

    def _score_batch_with_clip(self, image_paths: list[str | Path], captions: list[str]) -> list[float]:
        valid_images = [self._Image.open(image_path).convert("RGB") for image_path in image_paths]
        try:
            inputs = self._preprocess(text=captions, images=valid_images, return_tensors="pt", padding=True)
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.no_grad():
                image_features = extract_feature_tensor(self._model.get_image_features(pixel_values=inputs["pixel_values"]))
                text_features = extract_feature_tensor(
                    self._model.get_text_features(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                )
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                cosines = (image_features * text_features).sum(dim=-1).detach().cpu().tolist()
        finally:
            for image in valid_images:
                image.close()
        return [cosine_to_unit_score(value) for value in cosines]

    def _score_with_heuristic(self, image_path: str | Path, caption: str) -> float:
        text = (caption or "").strip()
        if not Path(image_path).exists() or not text:
            return 0.0
        digest = hashlib.sha256(f"{Path(image_path).name}|{text}".encode("utf-8")).digest()
        stable = int.from_bytes(digest[:2], "big") / 65535
        word_bonus = min(len(text.split()) / 12, 1.0) * 0.25
        return round(min(0.95, 0.45 + word_bonus + stable * 0.2), 4)
