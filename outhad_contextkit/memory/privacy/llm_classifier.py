"""LLM-based classifier for nuanced sensitivity detection."""
import json
import logging
import time
from typing import List

from outhad_contextkit.memory.privacy.base_classifier import (
    BaseClassifier,
    ClassificationResult,
    SensitiveSpan,
)
from outhad_contextkit.memory.privacy.config import PPMFConfig
from outhad_contextkit.memory.privacy.enums import PrivacyLevel, SensitiveSpanType

logger = logging.getLogger(__name__)


SENSITIVITY_CLASSIFICATION_PROMPT = """You are a privacy expert analyzing text for sensitive information.

Identify ALL sensitive spans in the text and classify them by type and privacy level.

Sensitive Types:
- PII (Personally Identifiable Information): names, emails, phone numbers, SSNs, addresses
- PHI (Protected Health Information): medical records, diagnoses, medications
- Secrets: API keys, passwords, tokens, private keys
- Financial: credit cards, bank accounts, routing numbers
- Location: GPS coordinates, home addresses
- Biometric: fingerprints, facial recognition data

Privacy Levels:
- public: No sensitive information
- internal: Internal information, not public
- confidential: Should be protected, limited access
- sensitive: Highly sensitive, strict access controls
- restricted: Most sensitive, maximum protection

Return JSON format:
{{
  "has_sensitive_content": true/false,
  "overall_privacy_level": "public|internal|confidential|sensitive|restricted",
  "sensitive_spans": [
    {{
      "text": "exact text span",
      "span_type": "pii_email|pii_phone|secret_api_key|etc",
      "privacy_level": "confidential",
      "confidence": 0.95,
      "reasoning": "why this is sensitive"
    }}
  ]
}}

Text to analyze:
{text}

Output ONLY valid JSON:"""


class LLMClassifier(BaseClassifier):
    """LLM-based classifier for contextual sensitivity detection."""
    
    def __init__(self, config: PPMFConfig, llm=None):
        self.config = config
        self.llm = llm  # Will be injected from Memory instance
        self.threshold = config.classification_threshold
    
    def classify(self, text: str) -> ClassificationResult:
        """Classify text using LLM."""
        start_time = time.time()
        
        if not self.llm:
            raise ValueError("LLM not configured for LLMClassifier")
        
        prompt = SENSITIVITY_CLASSIFICATION_PROMPT.format(text=text)
        
        try:
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result_data = json.loads(response)
            
            # Parse spans
            sensitive_spans = []
            for span_data in result_data.get("sensitive_spans", []):
                # Find span positions in text
                text_span = span_data["text"]
                start_pos = text.find(text_span)
                if start_pos == -1:
                    continue
                
                # Validate confidence threshold
                confidence = span_data.get("confidence", 1.0)
                if confidence < self.threshold:
                    continue
                
                span = SensitiveSpan(
                    text=text_span,
                    start=start_pos,
                    end=start_pos + len(text_span),
                    span_type=SensitiveSpanType(span_data["span_type"]),
                    privacy_level=PrivacyLevel(span_data["privacy_level"]),
                    confidence=confidence,
                    metadata={"reasoning": span_data.get("reasoning")}
                )
                sensitive_spans.append(span)
            
            processing_time = (time.time() - start_time) * 1000
            
            return ClassificationResult(
                original_text=text,
                has_sensitive_content=result_data["has_sensitive_content"],
                overall_privacy_level=PrivacyLevel(result_data["overall_privacy_level"]),
                sensitive_spans=sensitive_spans,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Fallback to PUBLIC
            return ClassificationResult(
                original_text=text,
                has_sensitive_content=False,
                overall_privacy_level=PrivacyLevel.PUBLIC,
                sensitive_spans=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts."""
        return [self.classify(text) for text in texts]

