"""Main MemorySanitizer - hybrid classifier combining rules and LLM."""
import logging
from typing import List

from outhad_contextkit.memory.privacy.base_classifier import (
    BaseClassifier,
    ClassificationResult,
)
from outhad_contextkit.memory.privacy.config import PPMFConfig
from outhad_contextkit.memory.privacy.enums import PrivacyLevel
from outhad_contextkit.memory.privacy.llm_classifier import LLMClassifier
from outhad_contextkit.memory.privacy.rule_classifier import RuleBasedClassifier

logger = logging.getLogger(__name__)


class MemorySanitizer(BaseClassifier):
    """
    Hybrid classifier that combines rule-based and LLM-based detection.
    
    Strategy:
    1. Run fast rule-based classifier first
    2. If high-confidence sensitive content found, return immediately
    3. Otherwise, run LLM classifier for nuanced detection
    4. Merge results from both classifiers
    """
    
    def __init__(self, config: PPMFConfig, llm=None):
        self.config = config
        self.rule_classifier = RuleBasedClassifier(config)
        self.llm_classifier = LLMClassifier(config, llm) if config.use_llm_classifier and llm else None
    
    def classify(self, text: str) -> ClassificationResult:
        """Classify text using hybrid approach."""
        
        # Step 1: Rule-based classification (fast)
        rule_result = self.rule_classifier.classify(text)
        
        # If no LLM classifier or rule-based found high-confidence sensitive data, return
        if not self.llm_classifier or (
            rule_result.has_sensitive_content and 
            any(span.confidence >= 0.9 for span in rule_result.sensitive_spans)
        ):
            return rule_result
        
        # Step 2: LLM-based classification (slower, more nuanced)
        try:
            llm_result = self.llm_classifier.classify(text)
            
            # Merge results (combine unique spans)
            merged_spans = list(rule_result.sensitive_spans)
            
            for llm_span in llm_result.sensitive_spans:
                # Check if this span overlaps with rule-based spans
                overlap = False
                for rule_span in rule_result.sensitive_spans:
                    if self._spans_overlap(llm_span, rule_span):
                        overlap = True
                        break
                
                if not overlap:
                    merged_spans.append(llm_span)
            
            # Use highest privacy level
            privacy_levels = {
                PrivacyLevel.PUBLIC: 0,
                PrivacyLevel.INTERNAL: 1,
                PrivacyLevel.CONFIDENTIAL: 2,
                PrivacyLevel.SENSITIVE: 3,
                PrivacyLevel.RESTRICTED: 4,
            }
            
            rule_level_value = privacy_levels[rule_result.overall_privacy_level]
            llm_level_value = privacy_levels[llm_result.overall_privacy_level]
            max_value = max(rule_level_value, llm_level_value)
            overall_level = [k for k, v in privacy_levels.items() if v == max_value][0]
            
            return ClassificationResult(
                original_text=text,
                has_sensitive_content=len(merged_spans) > 0,
                overall_privacy_level=overall_level,
                sensitive_spans=merged_spans,
                processing_time_ms=rule_result.processing_time_ms + llm_result.processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"LLM classification failed, falling back to rule-based: {e}")
            return rule_result
    
    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts."""
        return [self.classify(text) for text in texts]
    
    @staticmethod
    def _spans_overlap(span1, span2) -> bool:
        """Check if two spans overlap."""
        return not (span1.end <= span2.start or span2.end <= span1.start)

