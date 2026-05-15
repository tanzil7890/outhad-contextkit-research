"""
Production-ready multimodal embedder with CLIP and Whisper integration.

DEPRECATED: This module is kept for backward compatibility.
            Please use the new flexible embedder system instead:
            
            from outhad_contextkit.memory.temporal import create_default_embedder
            embedder = create_default_embedder()
            
            For more options, see:
            - create_default_embedder() - Speech-focused (Whisper)
            - create_clap_embedder() - All audio types (CLAP)
            - create_hybrid_embedder() - Maximum flexibility
"""
from __future__ import annotations

import io
import logging
import os
import warnings
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover - type-only; never imported at runtime
    from PIL import Image

logger = logging.getLogger(__name__)


class ProductionMultimodalEmbedder:
    """
    Production-ready multimodal embedder.
    
    Features:
    - CLIP for image embeddings (vision-language model)
    - OpenAI Whisper for audio transcription
    - OpenAI text embeddings for transcribed audio
    - Automatic fallback to placeholders if models unavailable
    """
    
    def __init__(self, embedding_model=None, openai_api_key: Optional[str] = None):
        """
        Initialize production embedder.
        
        Args:
            embedding_model: Text embedding model (for text and transcribed audio)
            openai_api_key: OpenAI API key for Whisper (optional, reads from env)
        """
        self.embedding_model = embedding_model
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize CLIP for images
        self._clip_model = None
        self._clip_processor = None
        self._clip_available = False
        self._init_clip()
        
        # Initialize Whisper client for audio
        self._whisper_client = None
        self._whisper_available = False
        self._init_whisper()
        
        logger.info(f"ProductionMultimodalEmbedder initialized:")
        logger.info(f"  - CLIP available: {self._clip_available}")
        logger.info(f"  - Whisper available: {self._whisper_available}")
    
    def _init_clip(self):
        """Initialize CLIP model for image embeddings."""
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch
            
            logger.info("Loading CLIP model for image embeddings...")
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self._clip_model = self._clip_model.to("cuda")
                logger.info("CLIP model loaded on GPU")
            else:
                logger.info("CLIP model loaded on CPU")
            
            self._clip_available = True
            logger.info("✅ CLIP model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"CLIP not available: {e}")
            logger.warning("Install with: pip install transformers torch torchvision")
            logger.warning("Will use placeholder image embeddings")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            logger.warning("Will use placeholder image embeddings")
    
    def _init_whisper(self):
        """Initialize Whisper client for audio transcription."""
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found - Whisper unavailable")
            logger.warning("Set OPENAI_API_KEY environment variable")
            return
        
        try:
            from openai import OpenAI
            
            self._whisper_client = OpenAI(api_key=self.openai_api_key)
            self._whisper_available = True
            logger.info("✅ Whisper client initialized successfully")
            
        except ImportError:
            logger.warning("OpenAI package not available")
            logger.warning("Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate text embedding.
        
        Uses CLIP's text encoder if available (for cross-modal search),
        otherwise falls back to provided embedding model.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        # FIRST: Try using CLIP text encoder (for cross-modal alignment)
        if self._clip_available:
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
                
                logger.debug(f"Generated CLIP text embedding: dim={len(embedding)}")
                return embedding
                
            except Exception as e:
                logger.warning(f"CLIP text embedding failed, trying fallback: {e}")
        
        # FALLBACK: Use provided embedding model
        if self.embedding_model:
            try:
                if hasattr(self.embedding_model, 'embed'):
                    return self.embedding_model.embed(text)
                elif hasattr(self.embedding_model, 'embed_query'):
                    return self.embedding_model.embed_query(text)
                else:
                    logger.error("Embedding model missing embed() method")
            except Exception as e:
                logger.error(f"Text embedding failed: {e}")
        
        # NO EMBEDDING AVAILABLE
        logger.warning("No text embedding model available (CLIP or custom)")
        return [0.0] * 768
    
    def embed_image(self, image: Image.Image) -> List[float]:
        """
        Generate image embedding using CLIP.
        
        Args:
            image: PIL Image object
            
        Returns:
            Embedding vector (512-dim from CLIP or 768-dim placeholder)
        """
        if not self._clip_available:
            logger.warning("CLIP not available, using placeholder")
            return self._placeholder_image_embedding(image)
        
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
            
            logger.debug(f"Generated CLIP image embedding: dim={len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"CLIP image embedding failed: {e}", exc_info=True)
            return self._placeholder_image_embedding(image)
    
    def embed_audio(self, audio_bytes: bytes) -> List[float]:
        """
        Generate audio embedding using Whisper transcription + text embedding.
        
        Process:
        1. Transcribe audio using Whisper API
        2. Embed the transcription using text embedder
        
        Args:
            audio_bytes: Audio data as bytes
            
        Returns:
            Embedding vector (768-dim)
        """
        if not self._whisper_available:
            logger.warning("Whisper not available, using placeholder")
            return self._placeholder_audio_embedding(audio_bytes)
        
        try:
            # Create a file-like object from bytes
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.mp3"  # Whisper needs a filename
            
            # Transcribe with Whisper
            logger.info("Transcribing audio with Whisper...")
            transcript = self._whisper_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            
            logger.info(f"Transcription: {transcript[:100]}...")
            
            # Embed the transcription
            if isinstance(transcript, str):
                embedding = self.embed_text(transcript)
            else:
                embedding = self.embed_text(transcript.text)
            
            logger.debug(f"Generated audio embedding via Whisper: dim={len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Whisper audio embedding failed: {e}", exc_info=True)
            return self._placeholder_audio_embedding(audio_bytes)
    
    def embed_multimodal(self, content) -> List[float]:
        """
        Generate embedding based on content modality.
        
        Args:
            content: MultimodalContent object
            
        Returns:
            Embedding vector appropriate for the modality
        """
        modality = content.modality
        
        if modality == "text":
            return self.embed_text(content.content)
        elif modality == "image":
            return self.embed_image(content.content)
        elif modality == "audio":
            return self.embed_audio(content.content)
        elif modality == "embedding":
            if isinstance(content.content, list):
                return content.content
            else:
                raise ValueError("Embedding modality requires list of floats")
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def _placeholder_image_embedding(self, image: Image.Image) -> List[float]:
        """Generate placeholder image embedding based on basic features."""
        try:
            width, height = image.size
            mode = image.mode
            
            # Create simple feature vector
            placeholder_vec = [0.0] * 768
            placeholder_vec[0] = float(width) / 1000.0  # Normalized width
            placeholder_vec[1] = float(height) / 1000.0  # Normalized height
            placeholder_vec[2] = 1.0 if mode == "RGB" else 0.5  # Color indicator
            
            # Add some variation based on image content
            try:
                # Get average pixel values
                import numpy as np
                img_array = np.array(image.resize((32, 32)))
                if len(img_array.shape) == 3:
                    avg_colors = img_array.mean(axis=(0, 1)) / 255.0
                    for i, val in enumerate(avg_colors[:3]):
                        placeholder_vec[3 + i] = float(val)
            except Exception:
                pass
            
            return placeholder_vec
        except Exception as e:
            logger.error(f"Placeholder image embedding failed: {e}")
            return [0.0] * 768
    
    def _placeholder_audio_embedding(self, audio_bytes: bytes) -> List[float]:
        """Generate placeholder audio embedding based on basic features."""
        audio_length = len(audio_bytes)
        
        placeholder_vec = [0.0] * 768
        placeholder_vec[0] = float(audio_length) / 100000.0  # Normalized length
        
        # Add some variation based on byte distribution
        try:
            byte_sum = sum(audio_bytes[:1000]) / 1000.0 if len(audio_bytes) > 0 else 0.0
            placeholder_vec[1] = byte_sum / 255.0
        except Exception:
            pass
        
        return placeholder_vec
    
    def get_capabilities(self) -> dict:
        """
        Get embedder capabilities.
        
        Returns:
            Dict with availability status for each modality
        """
        return {
            "text": self.embedding_model is not None,
            "image": self._clip_available,
            "audio": self._whisper_available,
            "clip_model": "openai/clip-vit-base-patch32" if self._clip_available else None,
            "whisper_model": "whisper-1" if self._whisper_available else None
        }


def create_production_embedder(embedding_model=None, openai_api_key: Optional[str] = None):
    """
    Factory function to create production embedder.
    
    DEPRECATED: Use the new flexible embedder system instead:
                - create_default_embedder() for default configuration
                - create_clap_embedder() for CLAP audio support
                - create_hybrid_embedder() for maximum flexibility
    
    This function is kept for backward compatibility and internally uses
    the new flexible embedder system.
    
    Args:
        embedding_model: Text embedding model (deprecated, ignored)
        openai_api_key: OpenAI API key (optional, reads from env)
    
    Returns:
        FlexibleMultimodalEmbedder instance (backward compatible)
    """
    warnings.warn(
        "create_production_embedder() is deprecated. "
        "Use create_default_embedder(), create_clap_embedder(), "
        "or create_hybrid_embedder() instead. "
        "See FLEXIBLE_EMBEDDER_GUIDE.md for details.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Use new flexible embedder system
    from outhad_contextkit.memory.temporal.embedder_factory import create_default_embedder
    return create_default_embedder(openai_api_key=openai_api_key)

