"""Factory for creating chunker instances."""

from typing import Dict, Type
from .base import ChunkerBase
from .config import ChunkingConfig


class ChunkerFactory:
    """Factory for creating chunker instances based on strategy.

    Example:
        >>> config = ChunkingConfig(strategy="token", chunk_size=512)
        >>> chunker = ChunkerFactory.create(config)
    """

    _chunkers: Dict[str, Type[ChunkerBase]] = {}

    @classmethod
    def register(cls, strategy: str, chunker_class: Type[ChunkerBase]):
        """Register a chunker implementation.

        Args:
            strategy: Strategy name (e.g., "token")
            chunker_class: Chunker class implementing ChunkerBase
        """
        cls._chunkers[strategy] = chunker_class

    @classmethod
    def create(cls, config: ChunkingConfig) -> ChunkerBase:
        """Create a chunker instance based on configuration.

        Args:
            config: ChunkingConfig instance

        Returns:
            Chunker instance

        Raises:
            ValueError: If strategy is not registered
        """
        strategy = config.strategy

        if strategy not in cls._chunkers:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Available: {list(cls._chunkers.keys())}"
            )

        chunker_class = cls._chunkers[strategy]

        # Pass relevant config based on chunker type
        kwargs = {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap
        }

        # Add strategy-specific config
        if strategy == "character":
            kwargs["separators"] = config.separators
        elif strategy == "token":
            kwargs["model_name"] = config.tokenizer_model
        elif strategy == "semantic":
            kwargs["threshold"] = config.semantic_threshold
        elif strategy == "hierarchical":
            kwargs["respect_markdown"] = config.respect_markdown_headers

        return chunker_class(**kwargs)

    @classmethod
    def list_strategies(cls) -> list:
        """List all registered chunking strategies."""
        return list(cls._chunkers.keys())
