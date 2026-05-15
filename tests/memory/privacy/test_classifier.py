"""Tests for PPMF classifiers."""
import pytest

from outhad_contextkit.memory.privacy.classifier import MemorySanitizer
from outhad_contextkit.memory.privacy.config import PPMFConfig
from outhad_contextkit.memory.privacy.enums import PrivacyLevel


class TestRuleBasedClassifier:
    """Test rule-based classifier."""
    
    def test_email_detection(self):
        """Test email detection."""
        config = PPMFConfig(enabled=True, use_llm_classifier=False)
        classifier = MemorySanitizer(config)
        
        text = "Contact me at john.doe@example.com for details."
        result = classifier.classify(text)
        
        assert result.has_sensitive_content
        assert len(result.sensitive_spans) == 1
        assert result.sensitive_spans[0].text == "john.doe@example.com"
        assert result.overall_privacy_level == PrivacyLevel.CONFIDENTIAL
    
    def test_phone_detection(self):
        """Test phone number detection."""
        config = PPMFConfig(enabled=True, use_llm_classifier=False)
        classifier = MemorySanitizer(config)
        
        text = "Call me at 555-123-4567."
        result = classifier.classify(text)
        
        assert result.has_sensitive_content
        assert len(result.sensitive_spans) == 1
    
    def test_api_key_detection(self):
        """Test API key detection."""
        config = PPMFConfig(enabled=True, use_llm_classifier=False)
        classifier = MemorySanitizer(config)
        
        text = "Use this key: sk-1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL"
        result = classifier.classify(text)
        
        assert result.has_sensitive_content
        assert result.overall_privacy_level == PrivacyLevel.RESTRICTED
    
    def test_no_sensitive_content(self):
        """Test text with no sensitive content."""
        config = PPMFConfig(enabled=True, use_llm_classifier=False)
        classifier = MemorySanitizer(config)
        
        text = "This is a regular message about the weather."
        result = classifier.classify(text)
        
        assert not result.has_sensitive_content
        assert result.overall_privacy_level == PrivacyLevel.PUBLIC
    
    def test_multiple_sensitive_spans(self):
        """Test text with multiple sensitive spans."""
        config = PPMFConfig(enabled=True, use_llm_classifier=False)
        classifier = MemorySanitizer(config)
        
        text = "Email me at john@example.com or call 555-123-4567."
        result = classifier.classify(text)
        
        assert result.has_sensitive_content
        assert len(result.sensitive_spans) == 2
    
    def test_classification_performance(self):
        """Test that classification completes quickly."""
        config = PPMFConfig(enabled=True, use_llm_classifier=False)
        classifier = MemorySanitizer(config)
        
        text = "Contact john.doe@example.com or 555-123-4567"
        result = classifier.classify(text)
        
        # Should complete in under 100ms for rule-based
        assert result.processing_time_ms < 100
        assert result.has_sensitive_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

