"""Integration tests for chunking in Memory class."""

import pytest
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.chunking import ChunkingConfig


def test_memory_without_chunking():
    """Test Memory works without chunking (default behavior)."""
    memory = Memory()

    result = memory.add(
        "This is a test message",
        user_id="alice",
        infer=False
    )

    assert len(result["results"]) == 1
    assert "chunk_index" not in result["results"][0]


def test_memory_with_chunking_short_text():
    """Test Memory with chunking enabled but short text (no chunking needed)."""
    config = MemoryConfig(
        chunking=ChunkingConfig(
            enabled=True,
            strategy="token",
            chunk_size=500
        )
    )
    memory = Memory(config)

    result = memory.add(
        "Short message",
        user_id="alice",
        infer=False
    )

    assert len(result["results"]) == 1
    assert "chunk_index" not in result["results"][0]


def test_memory_with_chunking_long_text():
    """Test Memory chunks long text."""
    config = MemoryConfig(
        chunking=ChunkingConfig(
            enabled=True,
            strategy="token",
            chunk_size=50,
            chunk_overlap=10
        )
    )
    memory = Memory(config)

    # Create long text
    long_text = " ".join(["This is a test sentence."] * 50)

    result = memory.add(
        long_text,
        user_id="alice",
        infer=False
    )

    # Should create multiple chunks
    assert len(result["results"]) > 1

    # All chunks should have same document_id
    doc_ids = [r["document_id"] for r in result["results"]]
    assert len(set(doc_ids)) == 1

    # Check chunk indices
    for i, chunk_result in enumerate(result["results"]):
        assert chunk_result["chunk_index"] == i
        assert chunk_result["total_chunks"] == len(result["results"])


def test_chunking_with_infer_true():
    """Test that infer=True still works (chunking not applied during fact extraction)."""
    config = MemoryConfig(
        chunking=ChunkingConfig(enabled=True, chunk_size=100, chunk_overlap=20)
    )
    memory = Memory(config)

    result = memory.add(
        "The sky is blue. The grass is green.",
        user_id="alice",
        infer=True  # LLM fact extraction
    )

    # Should work without errors
    assert "results" in result


def test_chunking_preserves_metadata():
    """Test that chunking preserves and enhances metadata."""
    config = MemoryConfig(
        chunking=ChunkingConfig(
            enabled=True,
            strategy="token",
            chunk_size=50,
            chunk_overlap=10
        )
    )
    memory = Memory(config)

    # Create long text
    long_text = " ".join(["Test content for chunking metadata."] * 30)

    result = memory.add(
        long_text,
        user_id="alice",
        metadata={"source": "test_doc", "author": "test_user"},
        infer=False
    )

    # Check that all chunks have the metadata
    assert len(result["results"]) > 1
    for chunk in result["results"]:
        assert "chunk_index" in chunk
        assert "total_chunks" in chunk
        assert "document_id" in chunk


def test_chunking_with_character_strategy():
    """Test chunking with character strategy."""
    config = MemoryConfig(
        chunking=ChunkingConfig(
            enabled=True,
            strategy="character",
            chunk_size=100,
            chunk_overlap=20
        )
    )
    memory = Memory(config)

    # Create long text
    long_text = "A" * 500

    result = memory.add(
        long_text,
        user_id="alice",
        infer=False
    )

    # Should create multiple chunks
    assert len(result["results"]) > 1


def test_chunking_disabled_by_default():
    """Test that chunking is disabled by default."""
    config = MemoryConfig()
    memory = Memory(config)

    # Chunker should be None
    assert memory._chunker is None

    # Long text should be stored as single item
    long_text = " ".join(["Test sentence."] * 100)
    result = memory.add(
        long_text,
        user_id="alice",
        infer=False
    )

    assert len(result["results"]) == 1
    assert "chunk_index" not in result["results"][0]


def test_chunking_with_mixed_messages():
    """Test chunking with a mix of short and long messages."""
    config = MemoryConfig(
        chunking=ChunkingConfig(
            enabled=True,
            strategy="token",
            chunk_size=50,
            chunk_overlap=10
        )
    )
    memory = Memory(config)

    # Add short message
    short_result = memory.add(
        "Short message",
        user_id="alice",
        infer=False
    )
    assert len(short_result["results"]) == 1
    assert "chunk_index" not in short_result["results"][0]

    # Add long message
    long_text = " ".join(["This is a long test sentence."] * 30)
    long_result = memory.add(
        long_text,
        user_id="alice",
        infer=False
    )
    assert len(long_result["results"]) > 1
    assert "chunk_index" in long_result["results"][0]
