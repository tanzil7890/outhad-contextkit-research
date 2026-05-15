"""Tests for temporal attributes in TCMGM."""
import pytest
from datetime import datetime, timedelta

from outhad_contextkit.memory.temporal.types import (
    TemporalEvent,
    TemporalRelation,
    TimeWindow,
    CausalLink,
)
from outhad_contextkit.memory.temporal.enums import (
    ModalityType,
    CausalType,
    TemporalRelationType,
)


class TestTemporalEvent:
    """Test TemporalEvent data model."""
    
    def test_temporal_event_creation(self):
        """Test creating a temporal event with all attributes."""
        now = datetime.utcnow()
        event = TemporalEvent(
            id="event_001",
            content="User logged in to the system",
            timestamp=now,
            confidence=0.95,
            modality="text",
            user_id="user_123",
            agent_id="agent_456"
        )
        
        assert event.id == "event_001"
        assert event.content == "User logged in to the system"
        assert event.timestamp == now
        assert event.confidence == 0.95
        assert event.modality == "text"
        assert event.user_id == "user_123"
        assert event.agent_id == "agent_456"
        assert event.embedding is None
        assert event.image_hash is None
        assert event.audio_hash is None
        assert event.metadata == {}
    
    def test_temporal_event_with_image(self):
        """Test temporal event with image modality."""
        event = TemporalEvent(
            id="event_002",
            content="User uploaded profile picture",
            timestamp=datetime.utcnow(),
            confidence=0.9,
            modality="image",
            image_hash="a1b2c3d4e5f6",
            metadata={"image_size": "1024x768"}
        )
        
        assert event.modality == "image"
        assert event.image_hash == "a1b2c3d4e5f6"
        assert event.metadata["image_size"] == "1024x768"
    
    def test_temporal_event_with_audio(self):
        """Test temporal event with audio modality."""
        event = TemporalEvent(
            id="event_003",
            content="Voice command recorded",
            timestamp=datetime.utcnow(),
            confidence=0.85,
            modality="audio",
            audio_hash="x9y8z7w6v5u4",
            metadata={"duration_seconds": 5}
        )
        
        assert event.modality == "audio"
        assert event.audio_hash == "x9y8z7w6v5u4"
        assert event.metadata["duration_seconds"] == 5
    
    def test_confidence_validation_valid(self):
        """Test that valid confidence values are accepted."""
        event = TemporalEvent(
            id="test",
            content="test",
            timestamp=datetime.utcnow(),
            confidence=0.5
        )
        assert event.confidence == 0.5
        
        event = TemporalEvent(
            id="test",
            content="test",
            timestamp=datetime.utcnow(),
            confidence=0.0
        )
        assert event.confidence == 0.0
        
        event = TemporalEvent(
            id="test",
            content="test",
            timestamp=datetime.utcnow(),
            confidence=1.0
        )
        assert event.confidence == 1.0
    
    def test_confidence_validation_invalid(self):
        """Test that invalid confidence values are rejected."""
        with pytest.raises(ValueError):
            TemporalEvent(
                id="test",
                content="test",
                timestamp=datetime.utcnow(),
                confidence=1.5  # Invalid: > 1.0
            )
        
        with pytest.raises(ValueError):
            TemporalEvent(
                id="test",
                content="test",
                timestamp=datetime.utcnow(),
                confidence=-0.1  # Invalid: < 0.0
            )


class TestTimeWindow:
    """Test TimeWindow data model."""
    
    def test_time_window_contains_current_time(self):
        """Test that time window correctly identifies if timestamp is within range."""
        now = datetime.utcnow()
        window = TimeWindow(
            start=now - timedelta(hours=1),
            end=now + timedelta(hours=1)
        )
        
        # Should contain current time
        assert window.contains(now)
    
    def test_time_window_excludes_past_time(self):
        """Test that time window correctly excludes time before start."""
        now = datetime.utcnow()
        window = TimeWindow(
            start=now - timedelta(hours=1),
            end=now + timedelta(hours=1)
        )
        
        # Should not contain time outside window (before start)
        past = now - timedelta(hours=2)
        assert not window.contains(past)
    
    def test_time_window_excludes_future_time(self):
        """Test that time window correctly excludes time after end."""
        now = datetime.utcnow()
        window = TimeWindow(
            start=now - timedelta(hours=1),
            end=now + timedelta(hours=1)
        )
        
        # Should not contain time outside window (after end)
        future = now + timedelta(hours=2)
        assert not window.contains(future)
    
    def test_time_window_no_start(self):
        """Test time window with no start bound (open start)."""
        now = datetime.utcnow()
        window = TimeWindow(
            end=now + timedelta(hours=1)
        )
        
        # Should accept any time before end
        past = now - timedelta(days=365)
        assert window.contains(past)
        assert window.contains(now)
        
        # Should reject time after end
        future = now + timedelta(hours=2)
        assert not window.contains(future)
    
    def test_time_window_no_end(self):
        """Test time window with no end bound (open end)."""
        now = datetime.utcnow()
        window = TimeWindow(
            start=now - timedelta(hours=1)
        )
        
        # Should accept any time after start
        future = now + timedelta(days=365)
        assert window.contains(future)
        assert window.contains(now)
        
        # Should reject time before start
        past = now - timedelta(hours=2)
        assert not window.contains(past)
    
    def test_time_window_unbounded(self):
        """Test completely unbounded time window."""
        window = TimeWindow()
        
        # Should accept any time
        now = datetime.utcnow()
        past = now - timedelta(days=365)
        future = now + timedelta(days=365)
        
        assert window.contains(past)
        assert window.contains(now)
        assert window.contains(future)


class TestTemporalRelation:
    """Test TemporalRelation data model."""
    
    def test_temporal_relation_creation(self):
        """Test creating a temporal relation."""
        now = datetime.utcnow()
        relation = TemporalRelation(
            source_id="event_001",
            target_id="event_002",
            relation_type="mentions",
            timestamp=now,
            confidence=0.9,
            temporal_type="before",
            causal_type="leads_to"
        )
        
        assert relation.source_id == "event_001"
        assert relation.target_id == "event_002"
        assert relation.relation_type == "mentions"
        assert relation.confidence == 0.9
        assert relation.temporal_type == "before"
        assert relation.causal_type == "leads_to"
    
    def test_temporal_relation_with_metadata(self):
        """Test temporal relation with additional metadata."""
        relation = TemporalRelation(
            source_id="event_A",
            target_id="event_B",
            relation_type="triggers",
            timestamp=datetime.utcnow(),
            confidence=0.85,
            metadata={"strength": "strong", "verified": True}
        )
        
        assert relation.metadata["strength"] == "strong"
        assert relation.metadata["verified"] is True


class TestCausalLink:
    """Test CausalLink data model."""
    
    def test_causal_link_creation(self):
        """Test creating a causal link."""
        now = datetime.utcnow()
        link = CausalLink(
            cause_id="event_001",
            effect_id="event_002",
            causal_type="caused_by",
            confidence=0.9,
            evidence="Event 001 directly triggered Event 002",
            timestamp=now
        )
        
        assert link.cause_id == "event_001"
        assert link.effect_id == "event_002"
        assert link.causal_type == "caused_by"
        assert link.confidence == 0.9
        assert link.evidence == "Event 001 directly triggered Event 002"
        assert link.timestamp == now
    
    def test_causal_types(self):
        """Test all causal types are valid."""
        causal_types = [
            CausalType.CAUSED_BY,
            CausalType.LEADS_TO,
            CausalType.ENABLES,
            CausalType.PREVENTS,
            CausalType.CORRELATES_WITH
        ]
        
        for causal_type in causal_types:
            link = CausalLink(
                cause_id="e1",
                effect_id="e2",
                causal_type=causal_type.value,
                confidence=0.8,
                timestamp=datetime.utcnow()
            )
            assert link.causal_type == causal_type.value


class TestTemporalEnums:
    """Test temporal enumerations."""
    
    def test_modality_types(self):
        """Test all modality types."""
        assert ModalityType.TEXT.value == "text"
        assert ModalityType.IMAGE.value == "image"
        assert ModalityType.AUDIO.value == "audio"
        assert ModalityType.VIDEO.value == "video"
        assert ModalityType.EMBEDDING.value == "embedding"
    
    def test_causal_types(self):
        """Test all causal types."""
        assert CausalType.CAUSED_BY.value == "caused_by"
        assert CausalType.LEADS_TO.value == "leads_to"
        assert CausalType.ENABLES.value == "enables"
        assert CausalType.PREVENTS.value == "prevents"
        assert CausalType.CORRELATES_WITH.value == "correlates_with"
    
    def test_temporal_relation_types(self):
        """Test all temporal relation types."""
        assert TemporalRelationType.BEFORE.value == "before"
        assert TemporalRelationType.AFTER.value == "after"
        assert TemporalRelationType.DURING.value == "during"
        assert TemporalRelationType.OVERLAPS.value == "overlaps"
        assert TemporalRelationType.CONCURRENT.value == "concurrent"
        assert TemporalRelationType.IMMEDIATELY_AFTER.value == "immediately_after"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

