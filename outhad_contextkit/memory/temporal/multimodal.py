"""Multimodal support for TCMGM."""
import hashlib
import logging
import base64
import io
from typing import List, Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)


class MultimodalContent:
    """Container for multimodal content."""
    
    def __init__(
        self,
        content: Union[str, bytes, Image.Image],
        modality: str,
        metadata: dict = None
    ):
        """
        Initialize multimodal content.
        
        Args:
            content: Content in any supported format (str, bytes, PIL Image)
            modality: Type of modality (text, image, audio, video, embedding)
            metadata: Optional metadata dictionary
        """
        self.content = content
        self.modality = modality
        self.metadata = metadata or {}
        self._hash = None
        self._embedding = None
    
    def compute_hash(self) -> str:
        """
        Compute content hash for deduplication and reference.
        
        Returns:
            SHA256 hash of the content as hex string
        """
        if self._hash:
            return self._hash
        
        if isinstance(self.content, str):
            self._hash = hashlib.sha256(self.content.encode()).hexdigest()
        elif isinstance(self.content, bytes):
            self._hash = hashlib.sha256(self.content).hexdigest()
        elif isinstance(self.content, Image.Image):
            # Hash image bytes
            img_bytes = io.BytesIO()
            self.content.save(img_bytes, format='PNG')
            self._hash = hashlib.sha256(img_bytes.getvalue()).hexdigest()
        else:
            raise ValueError(f"Unsupported content type: {type(self.content)}")
        
        return self._hash
    
    def set_embedding(self, embedding: List[float]):
        """
        Set embedding vector for this content.
        
        Args:
            embedding: List of floats representing the embedding
        """
        self._embedding = embedding
    
    def get_embedding(self) -> Optional[List[float]]:
        """
        Get embedding vector.
        
        Returns:
            Embedding vector or None if not set
        """
        return self._embedding
    
    def to_base64(self) -> str:
        """
        Convert content to base64 string for serialization.
        
        Returns:
            Base64 encoded string
        """
        if isinstance(self.content, str):
            return base64.b64encode(self.content.encode()).decode()
        elif isinstance(self.content, bytes):
            return base64.b64encode(self.content).decode()
        elif isinstance(self.content, Image.Image):
            img_bytes = io.BytesIO()
            self.content.save(img_bytes, format='PNG')
            return base64.b64encode(img_bytes.getvalue()).decode()
        return ""
    
    def get_size(self) -> int:
        """
        Get approximate size of content in bytes.
        
        Returns:
            Size in bytes
        """
        if isinstance(self.content, str):
            return len(self.content.encode())
        elif isinstance(self.content, bytes):
            return len(self.content)
        elif isinstance(self.content, Image.Image):
            img_bytes = io.BytesIO()
            self.content.save(img_bytes, format='PNG')
            return len(img_bytes.getvalue())
        return 0


class MultimodalEmbedder:
    """Generates embeddings for different modalities."""
    
    def __init__(self, embedding_model):
        """
        Initialize multimodal embedder.
        
        Args:
            embedding_model: Embedding model with embed() method
        """
        self.embedding_model = embedding_model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate text embedding.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        if hasattr(self.embedding_model, 'embed'):
            return self.embedding_model.embed(text)
        elif hasattr(self.embedding_model, 'embed_query'):
            return self.embedding_model.embed_query(text)
        else:
            raise ValueError("Embedding model must have embed() or embed_query() method")
    
    def embed_image(self, image: Image.Image) -> List[float]:
        """
        Generate image embedding.
        
        Note: This is a placeholder implementation. For production use,
        integrate CLIP, ViT, or similar vision models.
        
        Args:
            image: PIL Image object
            
        Returns:
            Embedding vector (placeholder implementation)
        """
        logger.warning("Image embedding using placeholder - integrate CLIP for production")
        
        # Placeholder: Create basic image descriptor
        # In production, use CLIP or similar vision encoder
        width, height = image.size
        mode = image.mode
        
        # Create simple feature vector from image properties
        # This is NOT a real embedding - use CLIP/ViT in production
        placeholder_vec = [0.0] * 768
        placeholder_vec[0] = float(width) / 1000.0  # Normalized width
        placeholder_vec[1] = float(height) / 1000.0  # Normalized height
        placeholder_vec[2] = 1.0 if mode == "RGB" else 0.5  # Color indicator
        
        return placeholder_vec
    
    def embed_audio(self, audio_bytes: bytes) -> List[float]:
        """
        Generate audio embedding.
        
        Note: This is a placeholder implementation. For production use,
        integrate Wav2Vec2, Whisper embeddings, or similar audio models.
        
        Args:
            audio_bytes: Audio data as bytes
            
        Returns:
            Embedding vector (placeholder implementation)
        """
        logger.warning("Audio embedding using placeholder - integrate Wav2Vec2 for production")
        
        # Placeholder: Create basic audio descriptor
        # In production, use Wav2Vec2, Whisper, or similar audio encoder
        audio_length = len(audio_bytes)
        
        placeholder_vec = [0.0] * 768
        placeholder_vec[0] = float(audio_length) / 100000.0  # Normalized length
        
        return placeholder_vec
    
    def embed_multimodal(self, content: MultimodalContent) -> List[float]:
        """
        Generate embedding based on content modality.
        
        Args:
            content: MultimodalContent object
            
        Returns:
            Embedding vector appropriate for the modality
            
        Raises:
            ValueError: If modality is not supported
        """
        if content.modality == "text":
            return self.embed_text(content.content)
        elif content.modality == "image":
            return self.embed_image(content.content)
        elif content.modality == "audio":
            return self.embed_audio(content.content)
        elif content.modality == "embedding":
            # Content is already an embedding
            if isinstance(content.content, list):
                return content.content
            else:
                raise ValueError("Embedding modality requires list of floats as content")
        else:
            raise ValueError(f"Unsupported modality: {content.modality}")


def create_text_content(text: str, metadata: dict = None) -> MultimodalContent:
    """
    Helper to create text content.
    
    Args:
        text: Text string
        metadata: Optional metadata
        
    Returns:
        MultimodalContent for text
    """
    return MultimodalContent(text, "text", metadata)


def create_image_content(image: Image.Image, metadata: dict = None) -> MultimodalContent:
    """
    Helper to create image content.
    
    Args:
        image: PIL Image object
        metadata: Optional metadata
        
    Returns:
        MultimodalContent for image
    """
    return MultimodalContent(image, "image", metadata)


def create_audio_content(audio_bytes: bytes, metadata: dict = None) -> MultimodalContent:
    """
    Helper to create audio content.
    
    Args:
        audio_bytes: Audio data as bytes
        metadata: Optional metadata
        
    Returns:
        MultimodalContent for audio
    """
    return MultimodalContent(audio_bytes, "audio", metadata)

