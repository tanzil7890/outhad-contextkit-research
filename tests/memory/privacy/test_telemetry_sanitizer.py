"""Tests for telemetry sanitizer."""
import pytest

from outhad_contextkit.memory.privacy.classifier import MemorySanitizer
from outhad_contextkit.memory.privacy.config import PPMFConfig
from outhad_contextkit.memory.privacy.telemetry_sanitizer import TelemetrySanitizer


class TestTelemetrySanitizer:
    """Test telemetry sanitization."""
    
    def test_sanitize_simple_string(self):
        """Test sanitizing a simple string with email."""
        config = PPMFConfig(enabled=True, use_llm_classifier=False, redact_telemetry=True)
        classifier = MemorySanitizer(config)
        sanitizer = TelemetrySanitizer(config, classifier)
        
        event_data = {
            "message": "Contact john.doe@example.com",
            "count": 5
        }
        
        sanitized = sanitizer.sanitize_event_data(event_data)
        
        assert "john.doe@example.com" not in sanitized["message"]
        assert "[PII_EMAIL]" in sanitized["message"]
        assert sanitized["count"] == 5
    
    def test_sanitize_nested_dict(self):
        """Test sanitizing nested dictionaries."""
        config = PPMFConfig(enabled=True, use_llm_classifier=False, redact_telemetry=True)
        classifier = MemorySanitizer(config)
        sanitizer = TelemetrySanitizer(config, classifier)
        
        event_data = {
            "user": {
                "email": "test@example.com",
                "name": "John Doe"
            },
            "action": "login"
        }
        
        sanitized = sanitizer.sanitize_event_data(event_data)
        
        assert "test@example.com" not in sanitized["user"]["email"]
        assert "[PII_EMAIL]" in sanitized["user"]["email"]
        assert sanitized["action"] == "login"
    
    def test_sanitize_list_of_strings(self):
        """Test sanitizing lists containing strings."""
        config = PPMFConfig(enabled=True, use_llm_classifier=False, redact_telemetry=True)
        classifier = MemorySanitizer(config)
        sanitizer = TelemetrySanitizer(config, classifier)
        
        event_data = {
            "emails": ["john@example.com", "jane@example.com"],
            "count": 2
        }
        
        sanitized = sanitizer.sanitize_event_data(event_data)
        
        assert all("[PII_EMAIL]" in email for email in sanitized["emails"])
        assert sanitized["count"] == 2
    
    def test_disabled_sanitization(self):
        """Test that sanitization can be disabled."""
        config = PPMFConfig(enabled=True, use_llm_classifier=False, redact_telemetry=False)
        classifier = MemorySanitizer(config)
        sanitizer = TelemetrySanitizer(config, classifier)
        
        event_data = {
            "message": "Contact john.doe@example.com"
        }
        
        sanitized = sanitizer.sanitize_event_data(event_data)
        
        # Should not be sanitized when disabled
        assert sanitized["message"] == "Contact john.doe@example.com"
    
    def test_non_sensitive_data_unchanged(self):
        """Test that non-sensitive data passes through unchanged."""
        config = PPMFConfig(enabled=True, use_llm_classifier=False, redact_telemetry=True)
        classifier = MemorySanitizer(config)
        sanitizer = TelemetrySanitizer(config, classifier)
        
        event_data = {
            "message": "Normal message about the weather",
            "count": 42,
            "nested": {
                "value": "Another normal string"
            }
        }
        
        sanitized = sanitizer.sanitize_event_data(event_data)
        
        assert sanitized == event_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

