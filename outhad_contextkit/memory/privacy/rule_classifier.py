"""Rule-based classifier using regex patterns."""
import re
import time
from typing import List

from outhad_contextkit.memory.privacy.base_classifier import (
    BaseClassifier,
    ClassificationResult,
    SensitiveSpan,
)
from outhad_contextkit.memory.privacy.config import PPMFConfig, get_default_sensitivity_rules
from outhad_contextkit.memory.privacy.enums import PrivacyLevel


class RuleBasedClassifier(BaseClassifier):
    """Rule-based classifier using regex patterns and heuristics."""
    
    def __init__(self, config: PPMFConfig):
        self.config = config
        self.rules = config.sensitivity_rules or get_default_sensitivity_rules()
        
        # Compile regex patterns for performance
        self.compiled_patterns = {}
        for rule in self.rules:
            if rule.patterns:
                self.compiled_patterns[rule.span_type] = [
                    re.compile(pattern) for pattern in rule.patterns
                ]
    
    def classify(self, text: str) -> ClassificationResult:
        """Classify text using rule-based patterns."""
        start_time = time.time()
        
        sensitive_spans = []
        
        for rule in self.rules:
            if rule.span_type in self.compiled_patterns:
                for pattern in self.compiled_patterns[rule.span_type]:
                    for match in pattern.finditer(text):
                        span = SensitiveSpan(
                            text=match.group(),
                            start=match.start(),
                            end=match.end(),
                            span_type=rule.span_type,
                            privacy_level=rule.privacy_level,
                            confidence=1.0,  # Rule-based is definitive
                            metadata={"rule": str(pattern.pattern)}
                        )
                        sensitive_spans.append(span)
        
        # Determine overall privacy level (highest among spans)
        overall_level = PrivacyLevel.PUBLIC
        if sensitive_spans:
            privacy_levels = {
                PrivacyLevel.PUBLIC: 0,
                PrivacyLevel.INTERNAL: 1,
                PrivacyLevel.CONFIDENTIAL: 2,
                PrivacyLevel.SENSITIVE: 3,
                PrivacyLevel.RESTRICTED: 4,
            }
            max_level = max(privacy_levels[span.privacy_level] for span in sensitive_spans)
            overall_level = [k for k, v in privacy_levels.items() if v == max_level][0]
        
        processing_time = (time.time() - start_time) * 1000
        
        return ClassificationResult(
            original_text=text,
            has_sensitive_content=len(sensitive_spans) > 0,
            overall_privacy_level=overall_level,
            sensitive_spans=sensitive_spans,
            processing_time_ms=processing_time
        )
    
    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts."""
        return [self.classify(text) for text in texts]

