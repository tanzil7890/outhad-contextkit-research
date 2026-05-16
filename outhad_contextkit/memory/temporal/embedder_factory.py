"""Factory for creating flexible multimodal embedders."""
import logging
import os
from typing import Optional, Dict, Any
from outhad_contextkit.memory.temporal.base_embedders import (
    MultimodalEmbedderBase,
    BaseTextEmbedder,
    BaseImageEmbedder,
    BaseAudioEmbedder
)

logger = logging.getLogger(__name__)


class EmbedderConfig:
    """
    Configuration for multimodal embedder selection.
    
    Allows users to easily switch between different models for each modality.
    """
    
    def __init__(
        self,
        text_embedder: str = "clip",
        image_embedder: str = "clip",
        audio_embedder: str = "whisper",
        openai_api_key: Optional[str] = None,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        clip_dtype: str = "fp32",
    ):
        """
        Configure embedders for each modality.

        Args:
            text_embedder: Text embedder to use
                - "clip": CLIP / SigLIP text encoder (best for cross-modal search)
                - "openai": OpenAI text-embedding-ada-002

            image_embedder: Image embedder to use
                - "clip": CLIP / SigLIP image encoder (recommended)

            audio_embedder: Audio embedder to use
                - "whisper": Whisper transcription (best for speech)
                - "clap": CLAP audio embeddings (best for all audio)
                - "hybrid": Try CLAP first, fallback to Whisper

            openai_api_key: OpenAI API key (for Whisper/OpenAI text)

            clip_model_name: HuggingFace model id for the
                vision-language backbone shared by ``clip`` text and
                image embedders . Examples:
                - ``"openai/clip-vit-base-patch32"`` (default, 88M params, fastest)
                - ``"openai/clip-vit-large-patch14"`` (304M params, +2-5% recall)
                - ``"google/siglip-base-patch16-224"`` (200M params, better text-image alignment)
        """
        self.text_embedder = text_embedder.lower()
        self.image_embedder = image_embedder.lower()
        self.audio_embedder = audio_embedder.lower()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.clip_model_name = clip_model_name
        self.clip_dtype = clip_dtype

        # Validate options
        valid_text = ["clip", "openai"]
        valid_image = ["clip"]
        valid_audio = ["whisper", "clap", "hybrid"]

        if self.text_embedder not in valid_text:
            raise ValueError(f"Invalid text_embedder: {text_embedder}. Choose from: {valid_text}")
        if self.image_embedder not in valid_image:
            raise ValueError(f"Invalid image_embedder: {image_embedder}. Choose from: {valid_image}")
        if self.audio_embedder not in valid_audio:
            raise ValueError(f"Invalid audio_embedder: {audio_embedder}. Choose from: {valid_audio}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "text_embedder": self.text_embedder,
            "image_embedder": self.image_embedder,
            "audio_embedder": self.audio_embedder,
            "clip_model_name": self.clip_model_name,
            "clip_dtype": self.clip_dtype,
            "openai_api_key": "***" if self.openai_api_key else None
        }


class FlexibleMultimodalEmbedder(MultimodalEmbedderBase):
    """
    Flexible multimodal embedder with easy model switching.
    
    Users can easily switch between different models for each modality
    without breaking anything.
    
    Example:
        # Use CLIP for text/image, Whisper for audio (speech-focused)
        embedder = create_embedder(
            text_embedder="clip",
            image_embedder="clip",
            audio_embedder="whisper"
        )
        
        # Use CLIP for text/image, CLAP for audio (all audio types)
        embedder = create_embedder(
            text_embedder="clip",
            image_embedder="clip",
            audio_embedder="clap"
        )
        
        # Switch anytime without breaking code!
    """
    
    def __init__(self, config: EmbedderConfig):
        """
        Initialize with configuration.
        
        Args:
            config: EmbedderConfig specifying which embedders to use
        """
        self.config = config
        
        # Create embedders based on config
        text_emb = self._create_text_embedder()
        image_emb = self._create_image_embedder()
        
        # Initialize base class FIRST (sets self.text_embedder, etc.)
        super().__init__(
            text_embedder=text_emb,
            image_embedder=image_emb,
            audio_embedder=None  # Set audio embedder after
        )
        
        # Now create audio embedder (can access self.text_embedder)
        audio_emb = self._create_audio_embedder()
        self.audio_embedder = audio_emb
        
        logger.info(f"✅ Flexible multimodal embedder initialized: {config.to_dict()}")
    
    def _create_text_embedder(self) -> BaseTextEmbedder:
        """Create text embedder based on config."""
        if self.config.text_embedder == "clip":
            from outhad_contextkit.memory.temporal.text_embedders import CLIPTextEmbedder
            return CLIPTextEmbedder(
                model_name=self.config.clip_model_name,
                dtype=self.config.clip_dtype,
            )

        elif self.config.text_embedder == "openai":
            from outhad_contextkit.memory.temporal.text_embedders import OpenAITextEmbedder
            return OpenAITextEmbedder(api_key=self.config.openai_api_key)

        else:
            raise ValueError(f"Unknown text embedder: {self.config.text_embedder}")

    def _create_image_embedder(self) -> BaseImageEmbedder:
        """Create image embedder based on config."""
        if self.config.image_embedder == "clip":
            from outhad_contextkit.memory.temporal.image_embedders import CLIPImageEmbedder
            return CLIPImageEmbedder(
                model_name=self.config.clip_model_name,
                dtype=self.config.clip_dtype,
            )

        else:
            raise ValueError(f"Unknown image embedder: {self.config.image_embedder}")
    
    def _create_audio_embedder(self) -> BaseAudioEmbedder:
        """Create audio embedder based on config."""
        if self.config.audio_embedder == "whisper":
            from outhad_contextkit.memory.temporal.audio_embedders import WhisperAudioEmbedder
            # Pass text embedder for transcription embedding
            return WhisperAudioEmbedder(
                api_key=self.config.openai_api_key,
                text_embedder=self.text_embedder
            )
        
        elif self.config.audio_embedder == "clap":
            from outhad_contextkit.memory.temporal.audio_embedders import CLAPAudioEmbedder
            return CLAPAudioEmbedder()
        
        elif self.config.audio_embedder == "hybrid":
            from outhad_contextkit.memory.temporal.audio_embedders import HybridAudioEmbedder
            return HybridAudioEmbedder(
                api_key=self.config.openai_api_key,
                text_embedder=self.text_embedder
            )
        
        else:
            raise ValueError(f"Unknown audio embedder: {self.config.audio_embedder}")


def create_embedder(
    text_embedder: str = "clip",
    image_embedder: str = "clip",
    audio_embedder: str = "whisper",
    openai_api_key: Optional[str] = None,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    clip_dtype: str = "fp32",
) -> FlexibleMultimodalEmbedder:
    """
    Factory function to create flexible multimodal embedder.
    
    This is the main entry point for users to create embedders.
    
    Args:
        text_embedder: Text embedder to use ("clip" or "openai")
        image_embedder: Image embedder to use ("clip")
        audio_embedder: Audio embedder to use ("whisper", "clap", or "hybrid")
        openai_api_key: OpenAI API key (optional, reads from env)
        clip_model_name: HuggingFace id for the CLIP / SigLIP backbone
            shared by the text + image embedders ( opt-in
            larger model, e.g. ``"openai/clip-vit-large-patch14"`` or
            ``"google/siglip-base-patch16-224"``).

    Returns:
        Configured multimodal embedder
    
    Examples:
        # Speech-focused setup (current default)
        embedder = create_embedder(
            text_embedder="clip",
            image_embedder="clip",
            audio_embedder="whisper"  # Best for speech
        )
        
        # All audio types setup
        embedder = create_embedder(
            text_embedder="clip",
            image_embedder="clip",
            audio_embedder="clap"  # Handles speech + environmental sounds
        )
        
        # Hybrid setup (intelligent fallback)
        embedder = create_embedder(
            text_embedder="clip",
            image_embedder="clip",
            audio_embedder="hybrid"  # Try CLAP, fallback to Whisper
        )
        
        # Switch embedders without breaking code!
        # Just change the parameters and everything still works
    """
    config = EmbedderConfig(
        text_embedder=text_embedder,
        image_embedder=image_embedder,
        audio_embedder=audio_embedder,
        openai_api_key=openai_api_key,
        clip_model_name=clip_model_name,
        clip_dtype=clip_dtype,
    )
    
    return FlexibleMultimodalEmbedder(config)


# Convenience aliases for common configurations
def create_default_embedder(openai_api_key: Optional[str] = None):
    """
    Create embedder with default configuration.
    
    Default: CLIP for text/images, Whisper for audio (speech)
    """
    return create_embedder(
        text_embedder="clip",
        image_embedder="clip",
        audio_embedder="whisper",
        openai_api_key=openai_api_key
    )


def create_clap_embedder(openai_api_key: Optional[str] = None):
    """
    Create embedder with CLAP for audio (all audio types).
    
    Use this for: Environmental sounds, music, effects + speech
    """
    return create_embedder(
        text_embedder="clip",
        image_embedder="clip",
        audio_embedder="clap",
        openai_api_key=openai_api_key
    )


def create_hybrid_embedder(openai_api_key: Optional[str] = None):
    """
    Create embedder with hybrid audio (CLAP first, Whisper fallback).

    Use this for: Maximum flexibility and robustness
    """
    return create_embedder(
        text_embedder="clip",
        image_embedder="clip",
        audio_embedder="hybrid",
        openai_api_key=openai_api_key
    )


def create_large_clip_embedder(
    openai_api_key: Optional[str] = None,
    audio_embedder: str = "whisper",
):
    """ opt-in CLIP-large backbone (304M params).

    +2-5% recall@10 on cross-modal benchmarks vs CLIP-base, at the cost
    of ~3.5× the RAM + ~2× the inference time. Use when retrieval
    quality matters more than throughput.
    """
    return create_embedder(
        text_embedder="clip",
        image_embedder="clip",
        audio_embedder=audio_embedder,
        openai_api_key=openai_api_key,
        clip_model_name="openai/clip-vit-large-patch14",
    )


def create_fp16_embedder(
    openai_api_key: Optional[str] = None,
    audio_embedder: str = "whisper",
    clip_model_name: str = "openai/clip-vit-base-patch32",
):
    """opt-in fp16 weights for the CLIP / SigLIP backbone.

    ~50% RAM cut + 1.5-2× faster inference on CUDA. CPU fp16 is sometimes
    slower than fp32 so this is opt-in and not the default.
    """
    return create_embedder(
        text_embedder="clip",
        image_embedder="clip",
        audio_embedder=audio_embedder,
        openai_api_key=openai_api_key,
        clip_model_name=clip_model_name,
        clip_dtype="fp16",
    )


def create_siglip_embedder(
    openai_api_key: Optional[str] = None,
    audio_embedder: str = "whisper",
):
    """ opt-in SigLIP backbone (200M params).

    SigLIP's sigmoid loss yields better text-image alignment than
    CLIP's softmax-contrastive loss, especially for short captions and
    fine-grained retrieval. ~768-dim output, plug-compatible with
    every CLIP code path through ``_clip_cache.get_clip``.
    """
    return create_embedder(
        text_embedder="clip",
        image_embedder="clip",
        audio_embedder=audio_embedder,
        openai_api_key=openai_api_key,
        clip_model_name="google/siglip-base-patch16-224",
    )

