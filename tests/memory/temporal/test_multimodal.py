"""Tests for multimodal support (TCMGM Phase 3)."""
import pytest
from PIL import Image
import io
import hashlib
from unittest.mock import MagicMock

from outhad_contextkit.memory.temporal.multimodal import (
    MultimodalContent,
    MultimodalEmbedder,
    create_text_content,
    create_image_content,
    create_audio_content
)
from outhad_contextkit.memory.temporal.cross_modal import (
    cross_modal_search,
    find_image_mentions_in_text,
    find_audio_mentions_in_text,
    compute_multimodal_relevance,
    group_by_modality
)
from outhad_contextkit.memory.temporal.enums import ModalityType


class TestMultimodalContent:
    """Test MultimodalContent class."""
    
    def test_text_content(self):
        """Test text modality."""
        content = MultimodalContent("Hello world", "text")
        content_hash = content.compute_hash()
        
        assert content.modality == "text"
        assert len(content_hash) == 64  # SHA256 hex
        assert content.get_size() > 0
    
    def test_image_content(self):
        """Test image modality."""
        # Create simple image
        img = Image.new('RGB', (100, 100), color='red')
        content = MultimodalContent(img, "image")
        
        content_hash = content.compute_hash()
        assert content.modality == "image"
        assert len(content_hash) == 64
        assert content.get_size() > 0
    
    def test_bytes_content(self):
        """Test bytes modality (e.g., audio)."""
        audio_bytes = b"fake audio data"
        content = MultimodalContent(audio_bytes, "audio")
        
        content_hash = content.compute_hash()
        assert content.modality == "audio"
        assert len(content_hash) == 64
        assert content.get_size() == len(audio_bytes)
    
    def test_content_hashing_consistency(self):
        """Test that same content produces same hash."""
        text = "Test content"
        content1 = MultimodalContent(text, "text")
        content2 = MultimodalContent(text, "text")
        
        assert content1.compute_hash() == content2.compute_hash()
    
    def test_different_content_different_hash(self):
        """Test that different content produces different hashes."""
        content1 = MultimodalContent("Text A", "text")
        content2 = MultimodalContent("Text B", "text")
        
        assert content1.compute_hash() != content2.compute_hash()
    
    def test_embedding_storage(self):
        """Test embedding get/set."""
        content = MultimodalContent("Test", "text")
        embedding = [0.1, 0.2, 0.3]
        
        content.set_embedding(embedding)
        retrieved = content.get_embedding()
        
        assert retrieved == embedding
    
    def test_embedding_initially_none(self):
        """Test that embedding is None before being set."""
        content = MultimodalContent("Test", "text")
        assert content.get_embedding() is None
    
    def test_to_base64_text(self):
        """Test base64 encoding of text."""
        content = MultimodalContent("Hello", "text")
        b64 = content.to_base64()
        
        assert isinstance(b64, str)
        assert len(b64) > 0
    
    def test_to_base64_image(self):
        """Test base64 encoding of image."""
        img = Image.new('RGB', (10, 10), color='blue')
        content = MultimodalContent(img, "image")
        b64 = content.to_base64()
        
        assert isinstance(b64, str)
        assert len(b64) > 0
    
    def test_metadata_storage(self):
        """Test metadata storage."""
        metadata = {"source": "camera", "location": "home"}
        content = MultimodalContent("Test", "text", metadata)
        
        assert content.metadata == metadata
        assert content.metadata["source"] == "camera"
    
    def test_hash_caching(self):
        """Test that hash is computed once and cached."""
        content = MultimodalContent("Test", "text")
        
        # First call computes hash
        hash1 = content.compute_hash()
        # Second call should return cached hash
        hash2 = content.compute_hash()
        
        assert hash1 == hash2
        assert content._hash is not None


class TestMultimodalEmbedder:
    """Test MultimodalEmbedder class."""
    
    def test_embedder_initialization(self):
        """Test embedder initialization."""
        mock_model = MagicMock()
        embedder = MultimodalEmbedder(mock_model)
        
        assert embedder.embedding_model == mock_model
    
    def test_embed_text(self):
        """Test text embedding."""
        mock_model = MagicMock()
        mock_model.embed.return_value = [0.1, 0.2, 0.3]
        
        embedder = MultimodalEmbedder(mock_model)
        embedding = embedder.embed_text("Test text")
        
        assert embedding == [0.1, 0.2, 0.3]
        mock_model.embed.assert_called_once_with("Test text")
    
    def test_embed_text_with_embed_query(self):
        """Test text embedding with embed_query method."""
        mock_model = MagicMock()
        del mock_model.embed  # Remove embed method
        mock_model.embed_query.return_value = [0.4, 0.5, 0.6]
        
        embedder = MultimodalEmbedder(mock_model)
        embedding = embedder.embed_text("Test text")
        
        assert embedding == [0.4, 0.5, 0.6]
    
    def test_embed_image_placeholder(self):
        """Test image embedding (placeholder)."""
        mock_model = MagicMock()
        embedder = MultimodalEmbedder(mock_model)
        
        img = Image.new('RGB', (100, 100), color='red')
        embedding = embedder.embed_image(img)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 768  # Placeholder dimension
    
    def test_embed_audio_placeholder(self):
        """Test audio embedding (placeholder)."""
        mock_model = MagicMock()
        embedder = MultimodalEmbedder(mock_model)
        
        audio_bytes = b"fake audio"
        embedding = embedder.embed_audio(audio_bytes)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 768
    
    def test_embed_multimodal_text(self):
        """Test multimodal embedding for text."""
        mock_model = MagicMock()
        mock_model.embed.return_value = [0.1, 0.2]
        
        embedder = MultimodalEmbedder(mock_model)
        content = MultimodalContent("Test", "text")
        
        embedding = embedder.embed_multimodal(content)
        assert embedding == [0.1, 0.2]
    
    def test_embed_multimodal_image(self):
        """Test multimodal embedding for image."""
        mock_model = MagicMock()
        embedder = MultimodalEmbedder(mock_model)
        
        img = Image.new('RGB', (50, 50))
        content = MultimodalContent(img, "image")
        
        embedding = embedder.embed_multimodal(content)
        assert isinstance(embedding, list)
        assert len(embedding) == 768
    
    def test_embed_multimodal_audio(self):
        """Test multimodal embedding for audio."""
        mock_model = MagicMock()
        embedder = MultimodalEmbedder(mock_model)
        
        content = MultimodalContent(b"audio", "audio")
        
        embedding = embedder.embed_multimodal(content)
        assert isinstance(embedding, list)
        assert len(embedding) == 768
    
    def test_embed_multimodal_unsupported(self):
        """Test unsupported modality raises error."""
        mock_model = MagicMock()
        embedder = MultimodalEmbedder(mock_model)
        
        content = MultimodalContent("test", "unknown_modality")
        
        with pytest.raises(ValueError, match="Unsupported modality"):
            embedder.embed_multimodal(content)


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_create_text_content(self):
        """Test create_text_content helper."""
        content = create_text_content("Hello", {"key": "value"})
        
        assert content.modality == "text"
        assert content.content == "Hello"
        assert content.metadata == {"key": "value"}
    
    def test_create_image_content(self):
        """Test create_image_content helper."""
        img = Image.new('RGB', (10, 10))
        content = create_image_content(img, {"source": "test"})
        
        assert content.modality == "image"
        assert content.content == img
        assert content.metadata == {"source": "test"}
    
    def test_create_audio_content(self):
        """Test create_audio_content helper."""
        audio = b"audio data"
        content = create_audio_content(audio, {"format": "wav"})
        
        assert content.modality == "audio"
        assert content.content == audio
        assert content.metadata == {"format": "wav"}


class TestCrossModalSearch:
    """Test cross-modal search functionality."""
    
    def test_cross_modal_search_basic(self):
        """Test basic cross-modal search."""
        query_emb = [1.0, 0.0, 0.0]
        candidates = [
            {
                "embedding": [1.0, 0.0, 0.0],
                "modality": "text",
                "content": "exact match"
            },
            {
                "embedding": [0.9, 0.1, 0.0],
                "modality": "text",
                "content": "close match"
            },
            {
                "embedding": [0.0, 1.0, 0.0],
                "modality": "image",
                "content": "different"
            }
        ]
        
        results = cross_modal_search(query_emb, candidates, top_k=2)
        
        assert len(results) == 2
        assert results[0]['content'] == "exact match"
        assert results[0]['similarity'] > results[1]['similarity']
    
    def test_cross_modal_search_with_modality_filter(self):
        """Test cross-modal search with modality filter."""
        query_emb = [1.0, 0.0, 0.0]
        candidates = [
            {"embedding": [1.0, 0.0, 0.0], "modality": "text", "content": "text1"},
            {"embedding": [0.9, 0.0, 0.0], "modality": "image", "content": "img1"},
            {"embedding": [0.8, 0.0, 0.0], "modality": "image", "content": "img2"}
        ]
        
        results = cross_modal_search(
            query_emb,
            candidates,
            modality_filter="image",
            top_k=5
        )
        
        assert len(results) == 2
        assert all(r['modality'] == "image" for r in results)
    
    def test_cross_modal_search_empty_candidates(self):
        """Test with empty candidates."""
        results = cross_modal_search([1.0, 0.0], [], top_k=5)
        assert len(results) == 0
    
    def test_cross_modal_search_min_similarity(self):
        """Test minimum similarity threshold."""
        query_emb = [1.0, 0.0, 0.0]
        candidates = [
            {"embedding": [1.0, 0.0, 0.0], "modality": "text", "content": "high"},
            {"embedding": [0.0, 1.0, 0.0], "modality": "text", "content": "low"}
        ]
        
        results = cross_modal_search(
            query_emb,
            candidates,
            top_k=10,
            min_similarity=0.9
        )
        
        assert len(results) == 1
        assert results[0]['content'] == "high"


class TestImageMentions:
    """Test finding image mentions in text."""
    
    def test_find_image_mentions_basic(self):
        """Test finding basic image references."""
        text_events = [
            {"content": "Look at this image"},
            {"content": "Here is a screenshot"},
            {"content": "No reference here"}
        ]
        
        mentions = find_image_mentions_in_text("abc123", text_events)
        
        assert len(mentions) == 2
        assert mentions[0]['event']['content'] == "Look at this image"
        assert mentions[1]['event']['content'] == "Here is a screenshot"
    
    def test_find_image_mentions_with_keywords(self):
        """Test various image keywords."""
        text_events = [
            {"content": "The photo shows"},
            {"content": "See this picture"},
            {"content": "Attached diagram"},
            {"content": "Just text"}
        ]
        
        mentions = find_image_mentions_in_text("hash", text_events)
        
        assert len(mentions) == 3
        assert all('confidence' in m for m in mentions)
    
    def test_find_image_mentions_no_matches(self):
        """Test with no image references."""
        text_events = [
            {"content": "Just some text"},
            {"content": "More text here"}
        ]
        
        mentions = find_image_mentions_in_text("hash", text_events)
        assert len(mentions) == 0


class TestAudioMentions:
    """Test finding audio mentions in text."""
    
    def test_find_audio_mentions_basic(self):
        """Test finding basic audio references."""
        text_events = [
            {"content": "Listen to this audio"},
            {"content": "The recording shows"},
            {"content": "No reference"}
        ]
        
        mentions = find_audio_mentions_in_text("xyz789", text_events)
        
        assert len(mentions) == 2
        assert all('confidence' in m for m in mentions)
    
    def test_find_audio_mentions_keywords(self):
        """Test various audio keywords."""
        text_events = [
            {"content": "The voice said"},
            {"content": "Sound of music"},
            {"content": "Speaking clearly"}
        ]
        
        mentions = find_audio_mentions_in_text("hash", text_events)
        
        assert len(mentions) == 3


class TestMultimodalRelevance:
    """Test multimodal relevance scoring."""
    
    def test_same_modality_bonus(self):
        """Test same modality gets bonus."""
        score = compute_multimodal_relevance("text", "text", 0.8)
        assert score > 0.8  # Should get bonus
    
    def test_cross_modal_penalty(self):
        """Test cross-modal gets penalty."""
        score = compute_multimodal_relevance("text", "audio", 0.8)
        assert score < 0.8  # Should have penalty
    
    def test_session_bonus(self):
        """Test same session bonus."""
        score1 = compute_multimodal_relevance("text", "image", 0.8, same_session=True)
        score2 = compute_multimodal_relevance("text", "image", 0.8, same_session=False)
        
        assert score1 > score2
    
    def test_score_clamping(self):
        """Test scores are clamped to [0, 1]."""
        score = compute_multimodal_relevance("text", "text", 0.95, same_session=True)
        assert 0.0 <= score <= 1.0


class TestGroupByModality:
    """Test grouping results by modality."""
    
    def test_group_by_modality_basic(self):
        """Test basic grouping."""
        results = [
            {"modality": "text", "content": "t1"},
            {"modality": "image", "content": "i1"},
            {"modality": "text", "content": "t2"},
            {"modality": "image", "content": "i2"}
        ]
        
        grouped = group_by_modality(results)
        
        assert len(grouped) == 2
        assert len(grouped["text"]) == 2
        assert len(grouped["image"]) == 2
    
    def test_group_by_modality_single(self):
        """Test grouping with single modality."""
        results = [
            {"modality": "text", "content": "t1"},
            {"modality": "text", "content": "t2"}
        ]
        
        grouped = group_by_modality(results)
        
        assert len(grouped) == 1
        assert len(grouped["text"]) == 2
    
    def test_group_by_modality_empty(self):
        """Test grouping empty results."""
        grouped = group_by_modality([])
        assert grouped == {}


class TestModalityTypeEnum:
    """Test ModalityType enumeration."""
    
    def test_modality_values(self):
        """Test all modality type values."""
        assert ModalityType.TEXT.value == "text"
        assert ModalityType.IMAGE.value == "image"
        assert ModalityType.AUDIO.value == "audio"
        assert ModalityType.VIDEO.value == "video"
        assert ModalityType.EMBEDDING.value == "embedding"
    
    def test_modality_count(self):
        """Test number of modality types."""
        from enum import Enum
        assert isinstance(ModalityType.TEXT, Enum)
        assert len(ModalityType) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

