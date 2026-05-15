"""Audio embedder implementations - Whisper, CLAP, and others."""
from __future__ import annotations

import hashlib
import io
import logging
import os
import tempfile
from collections import OrderedDict
from threading import RLock
from typing import Any, Dict, List, Optional

from outhad_contextkit.memory.temporal.base_embedders import BaseAudioEmbedder

logger = logging.getLogger(__name__)


# Phase P5 — process-local LRU keyed on SHA-256 of audio bytes.
# Repeat retrievals on the same audio (search after add, replay,
# RAG context windows) skip the Whisper API round-trip and the
# associated dollar cost.
_TRANSCRIPT_CACHE: "OrderedDict[str, str]" = OrderedDict()
_TRANSCRIPT_CACHE_CAP = 256
_TRANSCRIPT_LOCK = RLock()

# Phase P9 — CLAP embedding cache keyed on the same audio hash so a
# second pass over the bytes never re-loads tempfiles or re-runs the
# model.
_CLAP_EMBED_CACHE: "OrderedDict[str, List[float]]" = OrderedDict()
_CLAP_EMBED_CACHE_CAP = 256
_CLAP_EMBED_LOCK = RLock()


def _audio_hash(audio_bytes: bytes) -> str:
    return hashlib.sha256(audio_bytes or b"").hexdigest()


def _transcript_cache_get(key: str) -> Optional[str]:
    with _TRANSCRIPT_LOCK:
        if key not in _TRANSCRIPT_CACHE:
            return None
        _TRANSCRIPT_CACHE.move_to_end(key)
        return _TRANSCRIPT_CACHE[key]


def _transcript_cache_set(key: str, value: str) -> None:
    with _TRANSCRIPT_LOCK:
        _TRANSCRIPT_CACHE[key] = value
        _TRANSCRIPT_CACHE.move_to_end(key)
        while len(_TRANSCRIPT_CACHE) > _TRANSCRIPT_CACHE_CAP:
            _TRANSCRIPT_CACHE.popitem(last=False)


def clear_transcript_cache() -> int:
    with _TRANSCRIPT_LOCK:
        n = len(_TRANSCRIPT_CACHE)
        _TRANSCRIPT_CACHE.clear()
        return n


def _clap_cache_get(key: str) -> Optional[List[float]]:
    with _CLAP_EMBED_LOCK:
        if key not in _CLAP_EMBED_CACHE:
            return None
        _CLAP_EMBED_CACHE.move_to_end(key)
        return list(_CLAP_EMBED_CACHE[key])


def _clap_cache_set(key: str, value: List[float]) -> None:
    with _CLAP_EMBED_LOCK:
        _CLAP_EMBED_CACHE[key] = list(value)
        _CLAP_EMBED_CACHE.move_to_end(key)
        while len(_CLAP_EMBED_CACHE) > _CLAP_EMBED_CACHE_CAP:
            _CLAP_EMBED_CACHE.popitem(last=False)


def clear_clap_embed_cache() -> int:
    with _CLAP_EMBED_LOCK:
        n = len(_CLAP_EMBED_CACHE)
        _CLAP_EMBED_CACHE.clear()
        return n


class WhisperAudioEmbedder(BaseAudioEmbedder):
    """
    Audio embedder using Whisper for transcription + text embedding.
    
    Best for: Speech content (conversations, lectures, interviews)
    Not recommended for: Environmental sounds (birds, thunder, etc.)
    """
    
    def __init__(self, api_key: str = None, text_embedder=None):
        """
        Initialize Whisper audio embedder.
        
        Args:
            api_key: OpenAI API key for Whisper
            text_embedder: Text embedder to use for transcribed text
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.text_embedder = text_embedder
        self._client = None
        self._available = False
        self._init_whisper()
    
    def _init_whisper(self):
        """Initialize Whisper client."""
        if not self.api_key:
            logger.warning("No OpenAI API key for Whisper")
            return
        
        try:
            from openai import OpenAI
            
            self._client = OpenAI(api_key=self.api_key)
            self._available = True
            logger.info("✅ Whisper audio embedder ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            self._available = False
    
    def embed_audio(self, audio_bytes: bytes) -> List[float]:
        """
        Embed audio using Whisper transcription + text embedding.
        
        Process:
        1. Transcribe audio with Whisper
        2. Embed transcription with text embedder
        """
        if not self._available:
            logger.warning("Whisper not available, using placeholder")
            return self._placeholder_embedding(audio_bytes)

        try:
            # Phase P5 — SHA-256 transcript cache short-circuits the
            # Whisper API call for repeated audio.
            cache_key = _audio_hash(audio_bytes)
            transcript = _transcript_cache_get(cache_key)
            if transcript is None:
                # Create file-like object for Whisper
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "audio.mp3"

                # Transcribe with Whisper
                logger.debug("Transcribing audio with Whisper...")
                transcript = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                _transcript_cache_set(cache_key, str(transcript))
            else:
                logger.debug("Whisper transcript cache hit")

            logger.debug(f"Transcript: {str(transcript)[:100]}...")
            
            # Embed the transcription
            if self.text_embedder:
                embedding = self.text_embedder.embed_text(str(transcript))
            else:
                # Fallback: simple hash-based embedding
                logger.warning("No text embedder provided, using placeholder")
                embedding = [0.0] * 768
            
            return embedding
            
        except Exception as e:
            logger.error(f"Whisper audio embedding failed: {e}")
            return self._placeholder_embedding(audio_bytes)
    
    def _placeholder_embedding(self, audio_bytes: bytes) -> List[float]:
        """Generate placeholder based on audio length."""
        audio_length = len(audio_bytes)
        placeholder_vec = [0.0] * 768
        placeholder_vec[0] = float(audio_length) / 100000.0
        
        try:
            byte_sum = sum(audio_bytes[:1000]) / 1000.0 if len(audio_bytes) > 0 else 0.0
            placeholder_vec[1] = byte_sum / 255.0
        except Exception:
            pass
        
        return placeholder_vec
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return Whisper embedder capabilities."""
        return {
            "model": "whisper-1",
            "available": self._available,
            "dimensions": 768,
            "type": "speech-to-text",
            "best_for": "speech content (conversations, lectures)",
            "not_recommended_for": "environmental sounds"
        }


class CLAPAudioEmbedder(BaseAudioEmbedder):
    """
    Audio embedder using CLAP (Contrastive Language-Audio Pretraining).
    
    Best for: All audio types (speech + environmental sounds)
    Handles: Birds, thunder, rain, clapping, music, etc.
    """
    
    def __init__(self):
        """Initialize CLAP audio embedder."""
        self._clap_model = None
        self._available = False
        self._init_clap()
    
    def _init_clap(self):
        """Load CLAP model."""
        try:
            import laion_clap
            
            logger.info("Loading CLAP model for audio embeddings...")
            self._clap_model = laion_clap.CLAP_Module(enable_fusion=True)
            self._clap_model.load_ckpt()
            
            self._available = True
            logger.info("✅ CLAP audio embedder ready")
            
        except ImportError:
            logger.warning("CLAP not installed. Install with: pip install laion-clap")
            logger.warning("Falling back to placeholder audio embeddings")
            self._available = False
        except Exception as e:
            logger.error(f"Failed to load CLAP: {e}")
            self._available = False
    
    def embed_audio(self, audio_bytes: bytes) -> List[float]:
        """Generate audio embedding using CLAP."""
        if not self._available:
            logger.warning("CLAP not available, using placeholder")
            return self._placeholder_embedding(audio_bytes)

        # Phase P9 — cache CLAP embeddings keyed on the same SHA-256
        # hash used by Whisper so repeat passes never re-hit tempfile
        # I/O or rerun the model.
        cache_key = _audio_hash(audio_bytes)
        cached = _clap_cache_get(cache_key)
        if cached is not None:
            return cached

        temp_path: Optional[str] = None
        try:
            # CLAP needs a file path, so save to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name

            # Get CLAP embedding
            audio_embedding = self._clap_model.get_audio_embedding_from_filelist(
                [temp_path],
                use_tensor=False
            )

            # CLAP outputs 512-dim, convert to list and pad
            embedding = (
                audio_embedding[0].tolist()
                if hasattr(audio_embedding[0], 'tolist')
                else list(audio_embedding[0])
            )

            # Pad to 768 for consistency
            if len(embedding) < 768:
                embedding = embedding + [0.0] * (768 - len(embedding))

            _clap_cache_set(cache_key, embedding)
            return embedding

        except Exception as e:
            logger.error(f"CLAP audio embedding failed: {e}")
            return self._placeholder_embedding(audio_bytes)
        finally:
            # Cleanup tempfile in finally so CLAP errors don't leak files.
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
    
    def _placeholder_embedding(self, audio_bytes: bytes) -> List[float]:
        """Generate placeholder based on audio features."""
        audio_length = len(audio_bytes)
        placeholder_vec = [0.0] * 768
        placeholder_vec[0] = float(audio_length) / 100000.0
        
        try:
            byte_sum = sum(audio_bytes[:1000]) / 1000.0 if len(audio_bytes) > 0 else 0.0
            placeholder_vec[1] = byte_sum / 255.0
        except Exception:
            pass
        
        return placeholder_vec
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return CLAP embedder capabilities."""
        return {
            "model": "LAION-CLAP",
            "available": self._available,
            "dimensions": 768,
            "type": "audio-language",
            "best_for": "all audio types (speech + environmental sounds)",
            "handles": ["speech", "music", "environmental sounds", "effects"]
        }


class HybridAudioEmbedder(BaseAudioEmbedder):
    """
    Hybrid audio embedder that intelligently chooses between Whisper and CLAP.
    
    Strategy:
    - Try CLAP first (handles everything)
    - Fallback to Whisper for speech if CLAP unavailable
    - Placeholder if both unavailable
    """
    
    def __init__(self, api_key: str = None, text_embedder=None):
        """
        Initialize hybrid audio embedder.
        
        Args:
            api_key: OpenAI API key for Whisper fallback
            text_embedder: Text embedder for Whisper transcriptions
        """
        self.clap_embedder = CLAPAudioEmbedder()
        self.whisper_embedder = WhisperAudioEmbedder(api_key, text_embedder)
        
        logger.info("✅ Hybrid audio embedder initialized")
    
    def embed_audio(self, audio_bytes: bytes) -> List[float]:
        """
        Embed audio using best available method.
        
        Priority:
        1. CLAP (if available) - handles all audio types
        2. Whisper (if available) - handles speech
        3. Placeholder
        """
        # Try CLAP first (best for everything)
        if self.clap_embedder._available:
            return self.clap_embedder.embed_audio(audio_bytes)
        
        # Fallback to Whisper (good for speech)
        if self.whisper_embedder._available:
            logger.info("CLAP unavailable, using Whisper for audio")
            return self.whisper_embedder.embed_audio(audio_bytes)
        
        # Last resort: placeholder
        logger.warning("No audio embedder available, using placeholder")
        return [0.0] * 768
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return hybrid embedder capabilities."""
        return {
            "primary": self.clap_embedder.get_capabilities(),
            "fallback": self.whisper_embedder.get_capabilities(),
            "strategy": "CLAP first, Whisper fallback"
        }


