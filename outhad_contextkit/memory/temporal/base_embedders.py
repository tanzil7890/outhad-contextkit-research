"""Base classes for multimodal embedders - flexible plugin architecture."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:  # pragma: no cover - type-only import; never at runtime
    from PIL import Image

logger = logging.getLogger(__name__)


class BaseTextEmbedder(ABC):
    """Abstract base class for text embedders."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate text embedding.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector (list of floats)
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return embedder capabilities and metadata."""
        pass


class BaseImageEmbedder(ABC):
    """Abstract base class for image embedders."""
    
    @abstractmethod
    def embed_image(self, image: Image.Image) -> List[float]:
        """
        Generate image embedding.
        
        Args:
            image: PIL Image object
            
        Returns:
            Embedding vector (list of floats)
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return embedder capabilities and metadata."""
        pass


class BaseAudioEmbedder(ABC):
    """Abstract base class for audio embedders."""
    
    @abstractmethod
    def embed_audio(self, audio_bytes: bytes) -> List[float]:
        """
        Generate audio embedding.
        
        Args:
            audio_bytes: Audio data as bytes
            
        Returns:
            Embedding vector (list of floats)
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return embedder capabilities and metadata."""
        pass


class MultimodalEmbedderBase(ABC):
    """
    Base class for unified multimodal embedders.
    
    Coordinates text, image, and audio embedders.
    """
    
    def __init__(
        self,
        text_embedder: BaseTextEmbedder = None,
        image_embedder: BaseImageEmbedder = None,
        audio_embedder: BaseAudioEmbedder = None
    ):
        """
        Initialize with specific embedders for each modality.
        
        Args:
            text_embedder: Embedder for text
            image_embedder: Embedder for images
            audio_embedder: Embedder for audio
        """
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.audio_embedder = audio_embedder
    
    def embed_text(self, text: str) -> List[float]:
        """Embed text using configured embedder."""
        if self.text_embedder is None:
            logger.warning("No text embedder configured")
            return [0.0] * 768
        return self.text_embedder.embed_text(text)
    
    def embed_image(self, image: Image.Image) -> List[float]:
        """Embed image using configured embedder."""
        if self.image_embedder is None:
            logger.warning("No image embedder configured")
            return [0.0] * 768
        return self.image_embedder.embed_image(image)
    
    def embed_audio(self, audio_bytes: bytes) -> List[float]:
        """Embed audio using configured embedder."""
        if self.audio_embedder is None:
            logger.warning("No audio embedder configured")
            return [0.0] * 768
        return self.audio_embedder.embed_audio(audio_bytes)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of all configured embedders."""
        return {
            "text": self.text_embedder.get_capabilities() if self.text_embedder else None,
            "image": self.image_embedder.get_capabilities() if self.image_embedder else None,
            "audio": self.audio_embedder.get_capabilities() if self.audio_embedder else None,
        }


