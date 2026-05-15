"""Image embedder implementations."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from outhad_contextkit.memory.temporal.base_embedders import BaseImageEmbedder

if TYPE_CHECKING:  # pragma: no cover - type-only; never imported at runtime
    from PIL import Image

logger = logging.getLogger(__name__)


class CLIPImageEmbedder(BaseImageEmbedder):
    """Image embedder using a CLIP / SigLIP image encoder.

    Phase A7 — ``model_name`` kwarg lets callers pick a larger or
    better-aligned backbone (e.g. ``openai/clip-vit-large-patch14`` or
    ``google/siglip-base-patch16-224``).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        dtype: str = "fp32",
    ):
        """Initialize CLIP image encoder."""
        self.model_name = model_name
        self.dtype = dtype
        self._clip_model = None
        self._clip_processor = None
        self._available = False
        self._init_clip()

    def _init_clip(self):
        """Load CLIP model (Phase P2 — shared module-level cache)."""
        try:
            import torch

            from outhad_contextkit.memory.temporal._clip_cache import get_clip

            logger.info(
                "Loading vision-language image encoder %s (dtype=%s, cached)...",
                self.model_name,
                self.dtype,
            )
            self._clip_model, self._clip_processor = get_clip(
                self.model_name, dtype=self.dtype
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                self._clip_model = self._clip_model.to("cuda")
                logger.info("CLIP image encoder loaded on GPU")
            else:
                logger.info("CLIP image encoder loaded on CPU")

            self._available = True
            logger.info("CLIP image embedder ready")

        except Exception as e:
            logger.error(f"Failed to load CLIP image encoder: {e}")
            logger.warning("Image embeddings will use placeholders")
            self._available = False
    
    def embed_image(self, image: Image.Image) -> List[float]:
        """Generate image embedding using CLIP."""
        if not self._available:
            logger.warning("CLIP not available, using placeholder")
            return self._placeholder_embedding(image)
        
        try:
            import torch
            
            # Process image with CLIP
            inputs = self._clip_processor(images=image, return_tensors="pt")
            
            # Move to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Get image features
            with torch.no_grad():
                image_features = self._clip_model.get_image_features(**inputs)
            
            # Convert to list
            embedding = image_features[0].cpu().tolist()
            
            # CLIP outputs 512-dim, pad to 768 for consistency
            if len(embedding) < 768:
                embedding = embedding + [0.0] * (768 - len(embedding))
            
            return embedding
            
        except Exception as e:
            logger.error(f"CLIP image embedding failed: {e}")
            return self._placeholder_embedding(image)
    
    def _placeholder_embedding(self, image: Image.Image) -> List[float]:
        """Generate placeholder based on basic image features."""
        try:
            import numpy as np
            width, height = image.size
            
            placeholder_vec = [0.0] * 768
            placeholder_vec[0] = float(width) / 1000.0
            placeholder_vec[1] = float(height) / 1000.0
            placeholder_vec[2] = 1.0 if image.mode == "RGB" else 0.5
            
            # Add average color
            try:
                img_array = np.array(image.resize((32, 32)))
                if len(img_array.shape) == 3:
                    avg_colors = img_array.mean(axis=(0, 1)) / 255.0
                    for i, val in enumerate(avg_colors[:3]):
                        placeholder_vec[3 + i] = float(val)
            except Exception:
                pass
            
            return placeholder_vec
        except Exception:
            return [0.0] * 768
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return CLIP image embedder capabilities."""
        return {
            "model": self.model_name,
            "available": self._available,
            "dimensions": 768,
            "type": "vision-language",
            "best_for": "cross-modal text-image search"
        }


