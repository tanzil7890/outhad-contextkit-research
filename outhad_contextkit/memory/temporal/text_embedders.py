"""Text embedder implementations."""
import logging
from typing import List, Dict, Any
from outhad_contextkit.memory.temporal.base_embedders import BaseTextEmbedder

logger = logging.getLogger(__name__)


class CLIPTextEmbedder(BaseTextEmbedder):
    """Text embedder using a CLIP / SigLIP text encoder.

     ``model_name`` kwarg lets callers pick a larger or
    better-aligned backbone (e.g. ``openai/clip-vit-large-patch14`` or
    ``google/siglip-base-patch16-224``) without code changes. The
    backbone loader is shared with ``CLIPImageEmbedder`` via the
    module-level cache so the same weights are reused for both
    modalities.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        dtype: str = "fp32",
    ):
        """Initialize CLIP text encoder."""
        self.model_name = model_name
        self.dtype = dtype
        self._clip_model = None
        self._clip_processor = None
        self._available = False
        self._init_clip()

    def _init_clip(self):
        """Load CLIP model (shared module-level cache)."""
        try:
            import torch

            from outhad_contextkit.memory.temporal._clip_cache import get_clip

            logger.info(
                "Loading vision-language text encoder %s (dtype=%s, cached)...",
                self.model_name,
                self.dtype,
            )
            self._clip_model, self._clip_processor = get_clip(
                self.model_name, dtype=self.dtype
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                self._clip_model = self._clip_model.to("cuda")
                logger.info("CLIP text encoder loaded on GPU")
            else:
                logger.info("CLIP text encoder loaded on CPU")

            self._available = True
            logger.info("CLIP text embedder ready")

        except Exception as e:
            logger.error(f"Failed to load CLIP text encoder: {e}")
            logger.warning("Text embeddings will use placeholders")
            self._available = False
    
    def embed_text(self, text: str) -> List[float]:
        """Generate text embedding using CLIP."""
        if not self._available:
            logger.warning("CLIP not available, using placeholder")
            return [0.0] * 768
        
        try:
            import torch
            
            # Process text with CLIP
            inputs = self._clip_processor(text=[text], return_tensors="pt", padding=True)
            
            # Move to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Get text features
            with torch.no_grad():
                text_features = self._clip_model.get_text_features(**inputs)
            
            # Convert to list
            embedding = text_features[0].cpu().tolist()
            
            # CLIP outputs 512-dim, pad to 768 for consistency
            if len(embedding) < 768:
                embedding = embedding + [0.0] * (768 - len(embedding))
            
            return embedding
            
        except Exception as e:
            logger.error(f"CLIP text embedding failed: {e}")
            return [0.0] * 768
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return CLIP text embedder capabilities."""
        return {
            "model": self.model_name,
            "available": self._available,
            "dimensions": 768,
            "type": "vision-language",
            "best_for": "cross-modal text-image search"
        }


class OpenAITextEmbedder(BaseTextEmbedder):
    """Text embedder using OpenAI's text-embedding API."""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-ada-002"):
        """
        Initialize OpenAI text embedder.
        
        Args:
            api_key: OpenAI API key (reads from env if not provided)
            model: OpenAI embedding model to use
        """
        import os
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._client = None
        self._available = False
        self._init_openai()
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        if not self.api_key:
            logger.warning("No OpenAI API key provided")
            return
        
        try:
            from openai import OpenAI
            
            self._client = OpenAI(api_key=self.api_key)
            self._available = True
            logger.info(f"✅ OpenAI text embedder ready ({self.model})")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self._available = False
    
    def embed_text(self, text: str) -> List[float]:
        """Generate text embedding using OpenAI API."""
        if not self._available:
            logger.warning("OpenAI not available, using placeholder")
            return [0.0] * 1536
        
        try:
            response = self._client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI text embedding failed: {e}")
            return [0.0] * 1536
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return OpenAI text embedder capabilities."""
        dims = 1536 if "ada-002" in self.model else 1536
        return {
            "model": self.model,
            "available": self._available,
            "dimensions": dims,
            "type": "text-only",
            "best_for": "pure text semantic search"
        }


