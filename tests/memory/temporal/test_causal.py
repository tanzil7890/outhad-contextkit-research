"""Tests for causal relationships in TCMGM ."""
import pytest
from datetime import datetime

from outhad_contextkit.memory.temporal.types import CausalLink
from outhad_contextkit.memory.temporal.enums import CausalType
from outhad_contextkit.memory.temporal.causal_extractor import CausalExtractor


class TestCausalLink:
    """Test CausalLink data model."""
    
    def test_causal_link_creation(self):
        """Test creating a causal link."""
        now = datetime.utcnow()
        link = CausalLink(
            cause_id="event1",
            effect_id="event2",
            causal_type=CausalType.CAUSED_BY.value,
            confidence=0.9,
            evidence="Event1 triggered Event2",
            timestamp=now
        )
        
        assert link.cause_id == "event1"
        assert link.effect_id == "event2"
        assert link.causal_type == CausalType.CAUSED_BY.value
        assert link.confidence == 0.9
        assert link.evidence == "Event1 triggered Event2"
        assert link.timestamp == now
    
    def test_causal_link_without_evidence(self):
        """Test creating causal link without evidence."""
        link = CausalLink(
            cause_id="e1",
            effect_id="e2",
            causal_type=CausalType.LEADS_TO.value,
            confidence=0.8,
            timestamp=datetime.utcnow()
        )
        
        assert link.cause_id == "e1"
        assert link.effect_id == "e2"
        assert link.evidence is None
    
    def test_all_causal_types(self):
        """Test all causal type values."""
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
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence values
        for conf in [0.0, 0.5, 1.0]:
            link = CausalLink(
                cause_id="e1",
                effect_id="e2",
                causal_type=CausalType.LEADS_TO.value,
                confidence=conf,
                timestamp=datetime.utcnow()
            )
            assert link.confidence == conf
        
        # Invalid confidence values
        with pytest.raises(ValueError):
            CausalLink(
                cause_id="e1",
                effect_id="e2",
                causal_type=CausalType.LEADS_TO.value,
                confidence=1.5,  # > 1.0
                timestamp=datetime.utcnow()
            )
        
        with pytest.raises(ValueError):
            CausalLink(
                cause_id="e1",
                effect_id="e2",
                causal_type=CausalType.LEADS_TO.value,
                confidence=-0.1,  # < 0.0
                timestamp=datetime.utcnow()
            )


class TestCausalExtractor:
    """Test CausalExtractor for extracting causal relationships."""
    
    def test_causal_extractor_creation(self):
        """Test creating CausalExtractor."""
        extractor = CausalExtractor(llm=None)
        assert extractor.llm is None
    
    def test_rule_based_extraction_no_events(self):
        """Test rule-based extraction with empty event list."""
        extractor = CausalExtractor(llm=None)
        links = extractor.extract_causal_links([], use_llm=False)
        assert links == []
    
    def test_rule_based_extraction_caused_by(self):
        """Test rule-based extraction with 'caused_by' keywords."""
        extractor = CausalExtractor(llm=None)
        
        events = [
            {"id": "event1", "content": "User clicked button", "timestamp": "2025-01-14T10:00:00"},
            {"id": "event2", "content": "Page loaded because user clicked", "timestamp": "2025-01-14T10:00:01"}
        ]
        
        links = extractor.extract_causal_links(events, use_llm=False)
        
        # Should find at least one link
        assert len(links) > 0
        
        # Check first link
        link = links[0]
        assert link.cause_id == "event1"
        assert link.effect_id == "event2"
        assert link.causal_type == CausalType.CAUSED_BY.value
        assert 0.0 <= link.confidence <= 1.0
        assert "because" in link.evidence.lower()
    
    def test_rule_based_extraction_leads_to(self):
        """Test rule-based extraction with 'leads_to' keywords."""
        extractor = CausalExtractor(llm=None)
        
        events = [
            {"id": "event1", "content": "Error occurred", "timestamp": "2025-01-14T10:00:00"},
            {"id": "event2", "content": "Error led to system crash", "timestamp": "2025-01-14T10:00:02"}
        ]
        
        links = extractor.extract_causal_links(events, use_llm=False)
        
        assert len(links) > 0
        link = links[0]
        assert link.causal_type == CausalType.LEADS_TO.value
    
    def test_rule_based_extraction_enables(self):
        """Test rule-based extraction with 'enables' keywords."""
        extractor = CausalExtractor(llm=None)
        
        events = [
            {"id": "event1", "content": "User logged in", "timestamp": "2025-01-14T10:00:00"},
            {"id": "event2", "content": "Login enabled access to dashboard", "timestamp": "2025-01-14T10:00:01"}
        ]
        
        links = extractor.extract_causal_links(events, use_llm=False)
        
        assert len(links) > 0
        link = links[0]
        assert link.causal_type == CausalType.ENABLES.value
    
    def test_rule_based_extraction_prevents(self):
        """Test rule-based extraction with 'prevents' keywords."""
        extractor = CausalExtractor(llm=None)
        
        events = [
            {"id": "event1", "content": "Security check activated", "timestamp": "2025-01-14T10:00:00"},
            {"id": "event2", "content": "Security prevented unauthorized access", "timestamp": "2025-01-14T10:00:01"}
        ]
        
        links = extractor.extract_causal_links(events, use_llm=False)
        
        assert len(links) > 0
        # Find the prevents link (there might be multiple due to keyword matching)
        prevents_links = [link for link in links if link.causal_type == CausalType.PREVENTS.value]
        assert len(prevents_links) > 0, "Should find at least one 'prevents' causal link"
        link = prevents_links[0]
        assert link.causal_type == CausalType.PREVENTS.value
    
    def test_rule_based_extraction_deduplication(self):
        """Test that duplicate links are removed."""
        extractor = CausalExtractor(llm=None)
        
        events = [
            {"id": "event1", "content": "Error occurred", "timestamp": "2025-01-14T10:00:00"},
            {"id": "event2", "content": "Error caused crash and because of error system failed", "timestamp": "2025-01-14T10:00:01"}
        ]
        
        links = extractor.extract_causal_links(events, use_llm=False)
        
        # Should have unique cause-effect pairs
        cause_effect_pairs = [(link.cause_id, link.effect_id) for link in links]
        unique_pairs = set(cause_effect_pairs)
        assert len(unique_pairs) == len(links), "Links should be deduplicated"
    
    def test_rule_based_extraction_no_keywords(self):
        """Test rule-based extraction with no causal keywords."""
        extractor = CausalExtractor(llm=None)
        
        events = [
            {"id": "event1", "content": "User viewed page", "timestamp": "2025-01-14T10:00:00"},
            {"id": "event2", "content": "User scrolled down", "timestamp": "2025-01-14T10:00:01"}
        ]
        
        links = extractor.extract_causal_links(events, use_llm=False)
        
        # Should find no causal links
        assert len(links) == 0
    
    def test_llm_extraction_fallback_to_rules(self):
        """Test that LLM extraction falls back to rules when LLM is not available."""
        extractor = CausalExtractor(llm=None)  # No LLM
        
        events = [
            {"id": "event1", "content": "Action performed", "timestamp": "2025-01-14T10:00:00"},
            {"id": "event2", "content": "Result occurred because of action", "timestamp": "2025-01-14T10:00:01"}
        ]
        
        links = extractor.extract_causal_links(events, use_llm=True)  # Request LLM
        
        # Should fall back to rule-based
        assert len(links) > 0  # Rule-based should find links


class TestCausalEnums:
    """Test causal relationship enumerations."""
    
    def test_causal_type_values(self):
        """Test all CausalType enum values."""
        assert CausalType.CAUSED_BY.value == "caused_by"
        assert CausalType.LEADS_TO.value == "leads_to"
        assert CausalType.ENABLES.value == "enables"
        assert CausalType.PREVENTS.value == "prevents"
        assert CausalType.CORRELATES_WITH.value == "correlates_with"
    
    def test_causal_type_enumeration(self):
        """Test iterating over CausalType enum."""
        causal_types = list(CausalType)
        assert len(causal_types) == 5
        assert CausalType.CAUSED_BY in causal_types
        assert CausalType.LEADS_TO in causal_types
        assert CausalType.ENABLES in causal_types
        assert CausalType.PREVENTS in causal_types
        assert CausalType.CORRELATES_WITH in causal_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

