"""Sanitize telemetry data to prevent sensitive information leakage."""
import logging
from typing import Any, Dict

from outhad_contextkit.memory.privacy.classifier import MemorySanitizer
from outhad_contextkit.memory.privacy.config import PPMFConfig
from outhad_contextkit.memory.privacy.encryption import redact_text
from outhad_contextkit.memory.privacy.enums import RedactionMode

logger = logging.getLogger(__name__)


class TelemetrySanitizer:
    """Sanitizes telemetry data before transmission."""
    
    def __init__(self, config: PPMFConfig, classifier: MemorySanitizer):
        self.config = config
        self.classifier = classifier
        self.enabled = config.redact_telemetry
    
    def sanitize_event_data(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize event data dictionary.
        
        Recursively scans all string values and redacts sensitive content.
        """
        if not self.enabled:
            return event_data
        
        sanitized = {}
        
        for key, value in event_data.items():
            if isinstance(value, str):
                sanitized[key] = self._sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_event_data(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_string(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize a single string value."""
        if not text or len(text) < 3:
            return text
        
        try:
            # Classify for sensitive content
            result = self.classifier.classify(text)
            
            # If sensitive content found, redact it
            if result.has_sensitive_content:
                # Use REPLACE mode for telemetry (clearer than MASK)
                redacted = redact_text(text, result.sensitive_spans, RedactionMode.REPLACE)
                logger.debug(f"Redacted telemetry: '{text}' -> '{redacted}'")
                return redacted
            
            return text
        
        except Exception as e:
            logger.error(f"Error sanitizing telemetry string: {e}")
            # On error, return truncated version to be safe
            return text[:10] + "..."

