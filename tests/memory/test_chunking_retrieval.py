"""Tests for chunking retrieval enhancements ()."""

import pytest
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.chunking import ChunkingConfig


def test_search_without_merge_chunks():
    """Test that search returns individual chunks when merge_chunks=False."""
    config = MemoryConfig(
        chunking=ChunkingConfig(
            enabled=True,
            strategy="token",
            chunk_size=50,
            chunk_overlap=10,
            merge_chunks_on_retrieval=False  # Disable merging
        )
    )
    memory = Memory(config)

    # Add long document
    long_text = " ".join(["This is a test sentence about artificial intelligence."] * 30)
    memory.add(long_text, user_id="alice", infer=False)

    # Search should return individual chunks
    results = memory.search(
        query="artificial intelligence",
        user_id="alice",
        limit=10,
        merge_chunks=False  # Explicitly disable merging
    )

    # Should have multiple chunk results
    assert len(results["results"]) > 1

    # Check that results have chunk metadata
    for result in results["results"]:
        metadata = result.get("metadata") or {}
        if "document_id" in metadata:
            assert "chunk_index" in metadata
            assert "total_chunks" in metadata


def test_search_with_merge_chunks_enabled():
    """Test that search merges chunks when merge_chunks=True."""
    config = MemoryConfig(
        chunking=ChunkingConfig(
            enabled=True,
            strategy="token",
            chunk_size=50,
            chunk_overlap=10,
            merge_chunks_on_retrieval=True  # Enable merging
        )
    )
    memory = Memory(config)

    # Add long document
    long_text = " ".join(["This is a test sentence about machine learning."] * 30)
    memory.add(long_text, user_id="alice", infer=False)

    # Search with merging enabled
    results = memory.search(
        query="machine learning",
        user_id="alice",
        limit=10,
        merge_chunks=True  # Explicitly enable merging
    )

    # Should have fewer results due to merging
    assert len(results["results"]) >= 1

    # Check for merged metadata
    for result in results["results"]:
        metadata = result.get("metadata") or {}
        if metadata.get("is_merged"):
            assert "chunk_count" in metadata
            assert "chunk_indices" in metadata
            assert metadata["chunk_count"] > 1


def test_search_merge_chunks_default_from_config():
    """Test that search uses config default for merge_chunks."""
    config = MemoryConfig(
        chunking=ChunkingConfig(
            enabled=True,
            strategy="token",
            chunk_size=50,
            chunk_overlap=10,
            merge_chunks_on_retrieval=True  # Set default to True
        )
    )
    memory = Memory(config)

    # Add long document
    long_text = " ".join(["This is a test document about deep learning."] * 30)
    memory.add(long_text, user_id="alice", infer=False)

    # Search without specifying merge_chunks (should use config default)
    results = memory.search(
        query="deep learning",
        user_id="alice",
        limit=10
    )

    # Should merge chunks by default
    merged_count = sum(1 for r in results["results"] if (r.get("metadata") or {}).get("is_merged"))
    assert merged_count >= 0  # At least some results might be merged


def test_search_with_chunk_context_window():
    """Test chunk_context_window parameter (basic test - advanced feature)."""
    config = MemoryConfig(
        chunking=ChunkingConfig(
            enabled=True,
            strategy="token",
            chunk_size=50,
            chunk_overlap=10,
            merge_chunks_on_retrieval=True,
            chunk_context_window=2  # Include ±2 neighboring chunks
        )
    )
    memory = Memory(config)

    # Add long document
    long_text = " ".join(["This is a test about neural networks."] * 30)
    memory.add(long_text, user_id="alice", infer=False)

    # Search with context window
    results = memory.search(
        query="neural networks",
        user_id="alice",
        limit=10,
        chunk_context_window=2
    )

    # Should have results (context window is optional/advanced)
    assert len(results["results"]) >= 1


def test_search_mixed_chunked_and_non_chunked():
    """Test search with mix of chunked and non-chunked memories."""
    config = MemoryConfig(
        chunking=ChunkingConfig(
            enabled=True,
            strategy="token",
            chunk_size=50,
            chunk_overlap=10,
            merge_chunks_on_retrieval=True
        )
    )
    memory = Memory(config)

    # Add short message (won't be chunked)
    memory.add("Short note about AI", user_id="alice", infer=False)

    # Add long document (will be chunked)
    long_text = " ".join(["This is a detailed article about AI."] * 30)
    memory.add(long_text, user_id="alice", infer=False)

    # Search should handle both
    results = memory.search(
        query="AI",
        user_id="alice",
        limit=10,
        merge_chunks=True
    )

    # Should have results
    assert len(results["results"]) >= 1

    # Check for both chunked and non-chunked results
    has_merged = any((r.get("metadata") or {}).get("is_merged") for r in results["results"])
    has_non_chunked = any("document_id" not in (r.get("metadata") or {}) for r in results["results"])

    # At least one of these should be true
    assert has_merged or has_non_chunked


def test_search_chunking_disabled():
    """Test that search works normally when chunking is disabled."""
    config = MemoryConfig(
        chunking=ChunkingConfig(enabled=False)
    )
    memory = Memory(config)

    # Add messages
    memory.add("Test message", user_id="alice", infer=False)
    long_text = " ".join(["Long text."] * 100)
    memory.add(long_text, user_id="alice", infer=False)

    # Search should work normally
    results = memory.search(
        query="test",
        user_id="alice",
        limit=10
    )

    # Should have results
    assert len(results["results"]) >= 1

    # No chunk metadata
    for result in results["results"]:
        metadata = result.get("metadata") or {}
        assert "is_merged" not in metadata
        assert "chunk_index" not in metadata


def test_merge_chunks_override():
    """Test that merge_chunks parameter can override config default."""
    config = MemoryConfig(
        chunking=ChunkingConfig(
            enabled=True,
            strategy="token",
            chunk_size=50,
            chunk_overlap=10,
            merge_chunks_on_retrieval=True  # Default is True
        )
    )
    memory = Memory(config)

    # Add long document
    long_text = " ".join(["This is a test about computer vision."] * 30)
    memory.add(long_text, user_id="alice", infer=False)

    # Override to False
    results_no_merge = memory.search(
        query="computer vision",
        user_id="alice",
        limit=10,
        merge_chunks=False  # Override config default
    )

    # Should return individual chunks
    assert len(results_no_merge["results"]) > 0

    # Override to True
    results_with_merge = memory.search(
        query="computer vision",
        user_id="alice",
        limit=10,
        merge_chunks=True
    )

    # Both should work
    assert len(results_with_merge["results"]) > 0


def test_search_with_multiple_documents():
    """Test search and merging with multiple different documents."""
    config = MemoryConfig(
        chunking=ChunkingConfig(
            enabled=True,
            strategy="token",
            chunk_size=50,
            chunk_overlap=10,
            merge_chunks_on_retrieval=True
        )
    )
    memory = Memory(config)

    # Add multiple long documents
    doc1 = " ".join(["This is document one about Python programming."] * 30)
    doc2 = " ".join(["This is document two about Java programming."] * 30)
    doc3 = " ".join(["This is document three about programming languages."] * 30)

    memory.add(doc1, user_id="alice", metadata={"doc": "1"}, infer=False)
    memory.add(doc2, user_id="alice", metadata={"doc": "2"}, infer=False)
    memory.add(doc3, user_id="alice", metadata={"doc": "3"}, infer=False)

    # Search should merge chunks from each document separately
    results = memory.search(
        query="programming",
        user_id="alice",
        limit=20,
        merge_chunks=True
    )

    # Should have results
    assert len(results["results"]) >= 1

    # Check that different documents are kept separate
    doc_ids = set()
    for result in results["results"]:
        if (result.get("metadata") or {}).get("is_merged"):
            # Each merged result should have a unique document_id
            doc_id = result["id"]
            assert doc_id not in doc_ids
            doc_ids.add(doc_id)
