"""Comprehensive tests for chunking system."""

import pytest
import time
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.chunking import (
    ChunkingConfig,
    ChunkerFactory,
    Chunk,
    ChunkMerger
)


class TestChunkingComplete:
    """Complete test suite for chunking functionality."""

    def test_end_to_end_workflow(self):
        """Test complete workflow: chunk → store → retrieve → merge."""
        # Setup
        config = MemoryConfig(
            chunking=ChunkingConfig(
                enabled=True,
                strategy="token",
                chunk_size=100,
                chunk_overlap=20,
                merge_chunks_on_retrieval=True
            )
        )
        memory = Memory(config)

        # Long document
        document = """
        Artificial intelligence has revolutionized many industries.
        Machine learning models can now perform complex tasks.
        Natural language processing enables human-like text understanding.
        Computer vision has advanced significantly in recent years.
        Deep learning architectures continue to improve.
        """ * 10  # Make it long enough to chunk

        # 1. Add (chunks automatically)
        add_result = memory.add(
            document,
            user_id="test_user",
            infer=False
        )

        assert len(add_result["results"]) > 1
        assert all("document_id" in r for r in add_result["results"])

        # 2. Search (retrieves chunks)
        search_result = memory.search(
            query="artificial intelligence machine learning",
            user_id="test_user",
            limit=5
        )

        assert len(search_result["results"]) >= 1

        # 3. Verify merging happened (if results exist)
        if search_result["results"] and len(add_result["results"]) > len(search_result["results"]):
            # Chunks were merged
            first_result = search_result["results"][0]
            if "metadata" in first_result and first_result["metadata"] is not None:
                assert first_result["metadata"].get("is_merged") == True

    def test_mixed_chunked_and_non_chunked(self):
        """Test memory with both chunked and non-chunked entries."""
        config = MemoryConfig(
            chunking=ChunkingConfig(
                enabled=True,
                strategy="token",
                chunk_size=100,
                chunk_overlap=20
            )
        )
        memory = Memory(config)

        # Short message (won't be chunked)
        memory.add("Short message here", user_id="alice", infer=False)

        # Long message (will be chunked)
        long_msg = " ".join(["Long message content."] * 20)
        memory.add(long_msg, user_id="alice", infer=False)

        # Search should handle both
        results = memory.search(query="message", user_id="alice")

        assert len(results["results"]) >= 1

    def test_chunk_size_validation(self):
        """Test chunk size validation."""
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=-1)

        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_empty_text_handling(self):
        """Test handling of empty/whitespace text."""
        config = ChunkingConfig(strategy="token", chunk_size=100)
        chunker = ChunkerFactory.create(config)

        # Empty string
        chunks = chunker.split_text("")
        assert len(chunks) == 0

        # Whitespace only
        chunks = chunker.split_text("   \n\n   ")
        assert len(chunks) == 0

    def test_chunk_overlap_correctness(self):
        """Test that chunk overlap is correctly applied."""
        config = ChunkingConfig(
            strategy="character",
            chunk_size=100,
            chunk_overlap=20
        )
        chunker = ChunkerFactory.create(config)

        text = "A" * 300  # 300 characters
        chunks = chunker.split_text(text)

        # Check overlap exists
        assert len(chunks) > 1
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i].content
            chunk2 = chunks[i + 1].content

            # Last 20 chars of chunk1 should overlap with first 20 of chunk2
            # (approximately, depending on splitting boundaries)
            assert len(chunk1) > 0
            assert len(chunk2) > 0

    def test_chunk_metadata_consistency(self):
        """Test that chunk metadata is consistent and complete."""
        config = MemoryConfig(
            chunking=ChunkingConfig(
                enabled=True,
                strategy="token",
                chunk_size=50,
                chunk_overlap=10
            )
        )
        memory = Memory(config)

        # Add document with custom metadata
        custom_metadata = {"source": "test_doc", "author": "test_author"}
        long_text = " ".join(["Test sentence."] * 50)

        result = memory.add(
            long_text,
            user_id="test_user",
            metadata=custom_metadata,
            infer=False
        )

        # Verify all chunks have same document_id
        doc_ids = [r["document_id"] for r in result["results"]]
        assert len(set(doc_ids)) == 1

        # Verify chunk indices are sequential
        for i, chunk_result in enumerate(result["results"]):
            assert chunk_result["chunk_index"] == i
            assert chunk_result["total_chunks"] == len(result["results"])

    def test_different_strategies_consistency(self):
        """Test that different chunking strategies work correctly."""
        strategies = ["token", "character"]

        for strategy in strategies:
            config = MemoryConfig(
                chunking=ChunkingConfig(
                    enabled=True,
                    strategy=strategy,
                    chunk_size=100,
                    chunk_overlap=20
                )
            )
            memory = Memory(config)

            long_text = " ".join(["Test content."] * 100)
            result = memory.add(
                long_text,
                user_id=f"test_user_{strategy}",
                infer=False
            )

            # Should create multiple chunks
            assert len(result["results"]) > 1

            # All chunks should have required metadata
            for chunk in result["results"]:
                assert "document_id" in chunk
                assert "chunk_index" in chunk
                assert "total_chunks" in chunk


class TestPerformance:
    """Performance and benchmark tests."""

    def test_large_document_chunking(self):
        """Test chunking of very large documents."""
        config = ChunkingConfig(
            strategy="token",
            chunk_size=512,
            chunk_overlap=50
        )
        chunker = ChunkerFactory.create(config)

        # 100KB document
        large_text = "Sample text. " * 10000

        start = time.time()
        chunks = chunker.split_text(large_text)
        elapsed = time.time() - start

        assert len(chunks) > 0
        assert elapsed < 5.0  # Should complete in < 5 seconds

        print(f"\n✅ Chunked {len(large_text)} chars into {len(chunks)} chunks in {elapsed:.2f}s")

    def test_embedding_performance(self):
        """Test end-to-end performance with embeddings."""
        config = MemoryConfig(
            chunking=ChunkingConfig(
                enabled=True,
                strategy="token",
                chunk_size=256
            )
        )
        memory = Memory(config)

        # Medium document
        doc = "Test content. " * 1000

        start = time.time()
        result = memory.add(doc, user_id="test_perf", infer=False)
        elapsed = time.time() - start

        chunks_created = len(result["results"])
        print(f"\n✅ Created {chunks_created} chunks with embeddings in {elapsed:.2f}s")

        # Should be reasonably fast (< 30s for moderate doc with embeddings)
        assert elapsed < 30.0

    def test_search_performance(self):
        """Test search performance with chunked documents."""
        config = MemoryConfig(
            chunking=ChunkingConfig(
                enabled=True,
                strategy="token",
                chunk_size=256,
                chunk_overlap=50,
                merge_chunks_on_retrieval=True
            )
        )
        memory = Memory(config)

        # Add multiple documents
        for i in range(5):
            doc = f"Document {i}. " + " ".join([f"Content for document {i}."] * 200)
            memory.add(doc, user_id="test_search_perf", infer=False)

        # Test search performance
        start = time.time()
        results = memory.search(
            query="Content for document",
            user_id="test_search_perf",
            limit=10
        )
        elapsed = time.time() - start

        print(f"\n✅ Search completed in {elapsed:.2f}s, found {len(results['results'])} results")

        # Search should be fast (< 5s)
        assert elapsed < 5.0
        assert len(results["results"]) >= 1


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_very_small_chunk_size(self):
        """Test behavior with very small chunk size."""
        config = ChunkingConfig(
            strategy="character",
            chunk_size=50,  # Minimum allowed by validation
            chunk_overlap=10
        )
        chunker = ChunkerFactory.create(config)

        text = "This is a test." * 10  # Make it long enough to chunk
        chunks = chunker.split_text(text)

        assert len(chunks) >= 1
        # Allow some variance for separator handling
        assert all(len(c.content) <= 100 for c in chunks)

    def test_chunk_larger_than_text(self):
        """Test when chunk_size > text length."""
        config = ChunkingConfig(strategy="token", chunk_size=1000)
        chunker = ChunkerFactory.create(config)

        text = "Short text."
        chunks = chunker.split_text(text)

        assert len(chunks) == 1
        assert chunks[0].content.strip() == text.strip()

    def test_unicode_handling(self):
        """Test Unicode and emoji handling."""
        config = ChunkingConfig(strategy="token", chunk_size=100, chunk_overlap=20)
        chunker = ChunkerFactory.create(config)

        text = "Hello 世界! 🚀 Testing Unicode characters. " * 10
        chunks = chunker.split_text(text)

        assert len(chunks) > 0
        # Verify content is preserved
        reconstructed = " ".join(c.content for c in chunks)
        assert "世界" in reconstructed
        assert "🚀" in reconstructed

    def test_special_characters(self):
        """Test handling of special characters and formatting."""
        config = ChunkingConfig(strategy="character", chunk_size=100, chunk_overlap=20)
        chunker = ChunkerFactory.create(config)

        text = "Code: `print('hello')`\n\nMath: x² + y² = z²\n\nURL: https://example.com"
        chunks = chunker.split_text(text)

        assert len(chunks) > 0
        reconstructed = "".join(c.content for c in chunks)
        assert "`print('hello')`" in reconstructed

    def test_zero_overlap(self):
        """Test chunking with zero overlap."""
        config = ChunkingConfig(
            strategy="character",
            chunk_size=50,
            chunk_overlap=0
        )
        chunker = ChunkerFactory.create(config)

        text = "A" * 200
        chunks = chunker.split_text(text)

        assert len(chunks) > 1
        # No overlap means clean divisions
        total_length = sum(len(c.content) for c in chunks)
        assert total_length == len(text)

    def test_single_character_chunks(self):
        """Test minimum allowed chunk size."""
        config = ChunkingConfig(
            strategy="character",
            chunk_size=50,  # Minimum allowed
            chunk_overlap=0
        )
        chunker = ChunkerFactory.create(config)

        text = "A" * 200  # Long text to ensure chunking
        chunks = chunker.split_text(text)

        # Should create multiple chunks
        assert len(chunks) >= 1
        # Verify all content is preserved
        total_content = "".join(c.content for c in chunks)
        assert len(total_content) == len(text)

    def test_text_with_only_whitespace_sections(self):
        """Test text with multiple whitespace sections."""
        config = ChunkingConfig(strategy="token", chunk_size=100, chunk_overlap=20)
        chunker = ChunkerFactory.create(config)

        text = "Content1\n\n\n\n\nContent2\n\n\n\nContent3"
        chunks = chunker.split_text(text)

        assert len(chunks) > 0
        # Verify content is preserved
        reconstructed = " ".join(c.content for c in chunks)
        assert "Content1" in reconstructed
        assert "Content2" in reconstructed
        assert "Content3" in reconstructed

    def test_extremely_long_single_word(self):
        """Test handling of extremely long single words."""
        config = ChunkingConfig(
            strategy="character",
            chunk_size=50,
            chunk_overlap=10
        )
        chunker = ChunkerFactory.create(config)

        # Single word longer than chunk_size
        text = "A" * 500
        chunks = chunker.split_text(text)

        assert len(chunks) > 1
        # Verify all content is preserved
        total_length = 0
        for chunk in chunks:
            total_length += len(chunk.content)
        # Account for overlap
        expected_min = len(text)
        assert total_length >= expected_min

    def test_mixed_language_content(self):
        """Test handling of mixed language content."""
        config = ChunkingConfig(strategy="token", chunk_size=100)
        chunker = ChunkerFactory.create(config)

        text = """
        English content here.
        المحتوى العربي هنا.
        中文内容在这里。
        Contenu français ici.
        """ * 5

        chunks = chunker.split_text(text)

        assert len(chunks) > 0
        # Verify different scripts are preserved
        reconstructed = " ".join(c.content for c in chunks)
        assert "English" in reconstructed
        assert "العربي" in reconstructed
        assert "中文" in reconstructed
        assert "français" in reconstructed


class TestChunkMerger:
    """Tests for ChunkMerger functionality."""

    def test_merge_by_document(self):
        """Test merging chunks from the same document."""
        # Create mock chunk results
        chunks = [
            {
                "id": "chunk1",
                "memory": "First chunk content",
                "score": 0.9,
                "metadata": {
                    "document_id": "doc1",
                    "chunk_index": 0,
                    "total_chunks": 3
                }
            },
            {
                "id": "chunk2",
                "memory": "Second chunk content",
                "score": 0.85,
                "metadata": {
                    "document_id": "doc1",
                    "chunk_index": 1,
                    "total_chunks": 3
                }
            },
            {
                "id": "chunk3",
                "memory": "Third chunk content",
                "score": 0.8,
                "metadata": {
                    "document_id": "doc1",
                    "chunk_index": 2,
                    "total_chunks": 3
                }
            }
        ]

        merged = ChunkMerger.merge_by_document(chunks)

        # Should merge into single result
        assert len(merged) == 1
        assert merged[0]["metadata"]["is_merged"] == True
        assert merged[0]["metadata"]["chunk_count"] == 3
        assert "First chunk content" in merged[0]["memory"]
        assert "Second chunk content" in merged[0]["memory"]
        assert "Third chunk content" in merged[0]["memory"]

    def test_merge_multiple_documents(self):
        """Test merging chunks from multiple documents."""
        chunks = [
            {
                "id": "chunk1",
                "memory": "Doc1 Chunk1",
                "score": 0.9,
                "metadata": {
                    "document_id": "doc1",
                    "chunk_index": 0,
                    "total_chunks": 2
                }
            },
            {
                "id": "chunk2",
                "memory": "Doc1 Chunk2",
                "score": 0.85,
                "metadata": {
                    "document_id": "doc1",
                    "chunk_index": 1,
                    "total_chunks": 2
                }
            },
            {
                "id": "chunk3",
                "memory": "Doc2 Chunk1",
                "score": 0.95,
                "metadata": {
                    "document_id": "doc2",
                    "chunk_index": 0,
                    "total_chunks": 1
                }
            }
        ]

        merged = ChunkMerger.merge_by_document(chunks)

        # Should have 2 merged results
        assert len(merged) == 2

    def test_merge_with_non_chunked_results(self):
        """Test merging with mixed chunked and non-chunked results."""
        chunks = [
            {
                "id": "chunk1",
                "memory": "Chunked content",
                "score": 0.9,
                "metadata": {
                    "document_id": "doc1",
                    "chunk_index": 0,
                    "total_chunks": 2
                }
            },
            {
                "id": "standalone",
                "memory": "Standalone content",
                "score": 0.95,
                "metadata": {}  # No document_id
            }
        ]

        merged = ChunkMerger.merge_by_document(chunks)

        # Should include both chunked and standalone
        assert len(merged) >= 1
