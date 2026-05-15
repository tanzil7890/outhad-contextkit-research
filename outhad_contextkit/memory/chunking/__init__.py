"""Document chunking module for Outhad_ContextKit.

This module provides various strategies for splitting long documents
into smaller, semantically coherent chunks for embedding and retrieval.

Example:
    >>> from outhad_contextkit.memory.chunking import ChunkingConfig, ChunkerFactory
    >>> config = ChunkingConfig(enabled=True, strategy="token", chunk_size=512)
    >>> chunker = ChunkerFactory.create(config)
    >>> chunks = chunker.split_text("Your long document here...")
"""

from .base import Chunk, ChunkerBase, ChunkMerger
from .config import ChunkingConfig
from .factory import ChunkerFactory

# Import text_splitters to trigger registration
from . import text_splitters  # noqa: F401

__all__ = [
    "Chunk",
    "ChunkerBase",
    "ChunkMerger",
    "ChunkingConfig",
    "ChunkerFactory",
]
