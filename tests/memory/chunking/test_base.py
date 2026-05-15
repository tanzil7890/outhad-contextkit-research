"""Tests for chunking base classes."""

import pytest
from outhad_contextkit.memory.chunking import Chunk, ChunkerBase, ChunkingConfig


def test_chunk_creation():
    """Test Chunk dataclass creation."""
    chunk = Chunk(
        content="Test content",
        chunk_id="123",
        document_id="doc1",
        chunk_index=0,
        total_chunks=3,
        metadata={"key": "value"}
    )

    assert chunk.content == "Test content"
    assert chunk.chunk_index == 0
    assert chunk.total_chunks == 3
    assert chunk.metadata["key"] == "value"


def test_chunk_validation_empty_content():
    """Test Chunk validation for empty content."""
    with pytest.raises(ValueError, match="cannot be empty"):
        Chunk(
            content="   ",
            chunk_id="123",
            document_id="doc1",
            chunk_index=0,
            total_chunks=1,
            metadata={}
        )


def test_chunk_validation_invalid_index():
    """Test Chunk validation for invalid index."""
    with pytest.raises(ValueError, match="index exceeds total"):
        Chunk(
            content="Test",
            chunk_id="123",
            document_id="doc1",
            chunk_index=5,
            total_chunks=3,
            metadata={}
        )


def test_chunk_validation_negative_index():
    """Test Chunk validation for negative index."""
    with pytest.raises(ValueError, match="must be non-negative"):
        Chunk(
            content="Test",
            chunk_id="123",
            document_id="doc1",
            chunk_index=-1,
            total_chunks=3,
            metadata={}
        )


def test_chunk_validation_zero_total_chunks():
    """Test Chunk validation for zero total chunks."""
    with pytest.raises(ValueError, match="must be positive"):
        Chunk(
            content="Test",
            chunk_id="123",
            document_id="doc1",
            chunk_index=0,
            total_chunks=0,
            metadata={}
        )


def test_chunking_config_valid():
    """Test ChunkingConfig with valid values."""
    config = ChunkingConfig(
        enabled=True,
        strategy="token",
        chunk_size=512,
        chunk_overlap=50
    )
    assert config.chunk_size == 512
    assert config.chunk_overlap == 50
    assert config.strategy == "token"
    assert config.enabled is True


def test_chunking_config_invalid_overlap():
    """Test ChunkingConfig validation for invalid overlap."""
    with pytest.raises(ValueError, match="must be less than chunk_size"):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)


def test_chunking_config_overlap_larger_than_chunk_size():
    """Test ChunkingConfig validation for overlap > chunk_size."""
    with pytest.raises(ValueError, match="must be less than chunk_size"):
        ChunkingConfig(chunk_size=100, chunk_overlap=150)


def test_chunking_config_defaults():
    """Test ChunkingConfig default values."""
    config = ChunkingConfig()
    assert config.enabled is False
    assert config.strategy == "token"
    assert config.chunk_size == 512
    assert config.chunk_overlap == 50
    assert config.merge_chunks_on_retrieval is True
    assert config.chunk_context_window == 1


def test_chunking_config_chunk_size_validation():
    """Test ChunkingConfig chunk_size bounds."""
    # Too small
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10)  # Below minimum of 50

    # Too large
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10000)  # Above maximum of 8000

    # Valid boundaries
    config_min = ChunkingConfig(chunk_size=50)
    assert config_min.chunk_size == 50

    config_max = ChunkingConfig(chunk_size=8000)
    assert config_max.chunk_size == 8000


def test_chunking_config_strategies():
    """Test all valid chunking strategies."""
    strategies = ["character", "token", "semantic", "hierarchical"]

    for strategy in strategies:
        config = ChunkingConfig(strategy=strategy)
        assert config.strategy == strategy


def test_chunking_config_invalid_strategy():
    """Test ChunkingConfig with invalid strategy."""
    with pytest.raises(ValueError):
        ChunkingConfig(strategy="invalid_strategy")
