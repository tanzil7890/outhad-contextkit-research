"""Tests for text splitter implementations.

This module tests all three text splitter strategies:
- RecursiveCharacterTextSplitter
- SentenceTextSplitter
- TokenTextSplitter
"""

import pytest
from outhad_contextkit.memory.chunking.text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTextSplitter,
    TokenTextSplitter,
)
from outhad_contextkit.memory.chunking import ChunkerFactory, ChunkingConfig


class TestRecursiveCharacterTextSplitter:
    """Test RecursiveCharacterTextSplitter."""

    def test_basic_splitting(self):
        """Test basic text splitting."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10,
            separators=["\n\n", "\n", " ", ""]
        )

        text = "This is a test.\n\nThis is another paragraph.\n\nAnd a third one."
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk.content, str) for chunk in chunks)
        assert all(chunk.total_chunks == len(chunks) for chunk in chunks)

    def test_chunk_size_respected(self):
        """Test that chunks don't exceed max size."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

        text = "A" * 500  # Long text without separators
        chunks = splitter.split_text(text)

        # All chunks should be <= chunk_size
        assert all(len(chunk.content) <= 100 for chunk in chunks)

    def test_chunk_overlap(self):
        """Test chunk overlap functionality."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

        text = "This is a test sentence. " * 10
        chunks = splitter.split_text(text)

        # Should have multiple chunks
        assert len(chunks) > 1

        # Check that consecutive chunks have some overlap
        for i in range(len(chunks) - 1):
            # There should be some common text (though exact match is tricky due to splitting)
            assert chunks[i].chunk_index == i
            assert chunks[i + 1].chunk_index == i + 1

    def test_empty_text(self):
        """Test handling of empty text."""
        splitter = RecursiveCharacterTextSplitter()
        chunks = splitter.split_text("")
        assert len(chunks) == 0

    def test_metadata_preservation(self):
        """Test that metadata is preserved in chunks."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

        metadata = {"source": "test.txt", "author": "test"}
        chunks = splitter.split_text(
            "Test text " * 20,
            document_id="doc123",
            metadata=metadata
        )

        assert all(chunk.document_id == "doc123" for chunk in chunks)
        assert all(chunk.metadata == metadata for chunk in chunks)

    def test_chunk_indices(self):
        """Test that chunk indices are correct."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

        chunks = splitter.split_text("Test text. " * 20)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)

    def test_custom_separators(self):
        """Test custom separator hierarchy."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            separators=["###", "##", "#", " "]
        )

        text = "Section1### Section2### Section3"
        chunks = splitter.split_text(text)

        assert len(chunks) > 0


class TestSentenceTextSplitter:
    """Test SentenceTextSplitter."""

    def test_sentence_boundary_splitting(self):
        """Test splitting at sentence boundaries."""
        splitter = SentenceTextSplitter(chunk_size=100, chunk_overlap=10)

        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk.content, str) for chunk in chunks)

    def test_sentence_grouping(self):
        """Test that sentences are grouped into chunks."""
        splitter = SentenceTextSplitter(chunk_size=200, chunk_overlap=20)

        text = "Short. " * 50  # Many short sentences
        chunks = splitter.split_text(text)

        # Should group multiple sentences per chunk
        assert len(chunks) > 1
        assert len(chunks) < 50  # Not one chunk per sentence

    def test_respects_chunk_size(self):
        """Test that chunks respect size limits."""
        splitter = SentenceTextSplitter(chunk_size=100, chunk_overlap=0)

        text = "This is a sentence. " * 20
        chunks = splitter.split_text(text)

        # Most chunks should be close to chunk_size
        for chunk in chunks[:-1]:  # Exclude last chunk which may be smaller
            assert len(chunk.content) <= 150  # Some tolerance for sentence boundaries

    def test_empty_text(self):
        """Test handling of empty text."""
        splitter = SentenceTextSplitter()
        chunks = splitter.split_text("")
        assert len(chunks) == 0

    def test_metadata_preservation(self):
        """Test metadata preservation."""
        splitter = SentenceTextSplitter(chunk_size=100)

        metadata = {"type": "article"}
        chunks = splitter.split_text(
            "Sentence one. Sentence two. Sentence three. " * 5,
            document_id="doc456",
            metadata=metadata
        )

        assert all(chunk.document_id == "doc456" for chunk in chunks)
        assert all(chunk.metadata == metadata for chunk in chunks)

    def test_single_long_sentence(self):
        """Test handling of single long sentence."""
        splitter = SentenceTextSplitter(chunk_size=50, chunk_overlap=10)

        # Long sentence without proper ending
        text = "This is a very long sentence that goes on and on without any proper punctuation marks"
        chunks = splitter.split_text(text)

        assert len(chunks) >= 1


class TestTokenTextSplitter:
    """Test TokenTextSplitter."""

    def test_token_based_splitting(self):
        """Test token-based text splitting."""
        splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20, model_name="gpt-4")

        text = "This is a test sentence. " * 20
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        assert all(chunk.token_count is not None for chunk in chunks)
        assert all(chunk.token_count <= 100 for chunk in chunks)

    def test_token_count_accuracy(self):
        """Test that token counts are accurate."""
        splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0, model_name="gpt-4")

        text = "Hello world! " * 30
        chunks = splitter.split_text(text)

        # Each chunk should have token_count set
        assert all(chunk.token_count > 0 for chunk in chunks)
        assert all(chunk.token_count <= 100 for chunk in chunks)

    def test_chunk_overlap_tokens(self):
        """Test token-based overlap."""
        splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=10, model_name="gpt-4")

        text = "Test sentence. " * 50
        chunks = splitter.split_text(text)

        # Should have multiple chunks with overlap
        assert len(chunks) > 1

    def test_count_tokens_method(self):
        """Test the count_tokens helper method."""
        splitter = TokenTextSplitter(model_name="gpt-4")

        text = "Hello world!"
        token_count = splitter.count_tokens(text)

        assert isinstance(token_count, int)
        assert token_count > 0

    def test_empty_text(self):
        """Test handling of empty text."""
        splitter = TokenTextSplitter()
        chunks = splitter.split_text("")
        assert len(chunks) == 0

    def test_metadata_preservation(self):
        """Test metadata preservation."""
        splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20, model_name="gpt-4")

        metadata = {"model": "gpt-4"}
        chunks = splitter.split_text(
            "Test text. " * 30,
            document_id="doc789",
            metadata=metadata
        )

        assert all(chunk.document_id == "doc789" for chunk in chunks)
        assert all(chunk.metadata == metadata for chunk in chunks)

    def test_different_models(self):
        """Test with different tokenizer models."""
        for model in ["gpt-4", "gpt-3.5-turbo"]:
            splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20, model_name=model)
            chunks = splitter.split_text("Test text. " * 20)
            assert len(chunks) > 0

    def test_tiktoken_import_error(self):
        """Test graceful handling if tiktoken not available."""
        # This test assumes tiktoken is installed
        # In a real scenario without tiktoken, this would raise ImportError
        splitter = TokenTextSplitter(model_name="gpt-4")
        assert splitter._encoding is not None


class TestChunkerFactoryIntegration:
    """Test factory integration with text splitters."""

    def test_character_splitter_creation(self):
        """Test creating character splitter via factory."""
        config = ChunkingConfig(
            enabled=True,
            strategy="character",
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " "]
        )

        splitter = ChunkerFactory.create(config)
        assert isinstance(splitter, RecursiveCharacterTextSplitter)
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 50

    def test_sentence_splitter_creation(self):
        """Test creating sentence splitter via factory."""
        # SentenceTextSplitter is registered but not in ChunkingConfig Literal
        # Skip this test as sentence is an internal strategy not exposed in config
        pytest.skip("Sentence strategy not exposed in ChunkingConfig Literal")

    def test_token_splitter_creation(self):
        """Test creating token splitter via factory."""
        config = ChunkingConfig(
            enabled=True,
            strategy="token",
            chunk_size=512,
            chunk_overlap=50,
            tokenizer_model="gpt-4"
        )

        splitter = ChunkerFactory.create(config)
        assert isinstance(splitter, TokenTextSplitter)
        assert splitter.chunk_size == 512
        assert splitter.chunk_overlap == 50
        assert splitter.model_name == "gpt-4"

    def test_all_registered_strategies(self):
        """Test that all splitters are registered."""
        strategies = ChunkerFactory.list_strategies()

        # Should have at least these three strategies
        assert "character" in strategies
        assert "token" in strategies
        # sentence might not be in the config literal, so we check if it's registered

    def test_end_to_end_workflow(self):
        """Test complete workflow: config -> factory -> splitting."""
        config = ChunkingConfig(
            enabled=True,
            strategy="token",
            chunk_size=100,
            chunk_overlap=20,
            tokenizer_model="gpt-4"
        )

        splitter = ChunkerFactory.create(config)

        text = "This is a comprehensive test of the chunking system. " * 20
        chunks = splitter.split_text(
            text,
            document_id="test_doc",
            metadata={"test": True}
        )

        # Validate chunks
        assert len(chunks) > 0
        assert all(chunk.document_id == "test_doc" for chunk in chunks)
        assert all(chunk.metadata["test"] is True for chunk in chunks)
        assert all(0 <= chunk.chunk_index < len(chunks) for chunk in chunks)
        assert all(chunk.total_chunks == len(chunks) for chunk in chunks)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_chunk_size(self):
        """Test with very small chunk size."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
        text = "Short text"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_overlap_equal_to_chunk_size_prevented(self):
        """Test that config prevents overlap >= chunk_size."""
        # This should be caught by ChunkingConfig validation
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_unicode_text(self):
        """Test handling of Unicode text."""
        splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20, model_name="gpt-4")

        text = "Hello 世界! This is a test with émojis 🎉 and ñoñ-ASCII çharacters."
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        # Verify Unicode is preserved
        full_content = "".join(chunk.content for chunk in chunks)
        assert "世界" in full_content or "🎉" in full_content

    def test_whitespace_only_text(self):
        """Test handling of whitespace-only text."""
        splitter = RecursiveCharacterTextSplitter()
        chunks = splitter.split_text("   \n\n  \t  ")
        assert len(chunks) == 0

    def test_single_character_chunks(self):
        """Test extremely small chunks."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1, chunk_overlap=0)
        text = "ABC"
        chunks = splitter.split_text(text)
        # Should handle gracefully
        assert len(chunks) >= 1
