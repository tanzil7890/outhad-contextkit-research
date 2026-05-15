"""Base classifier interface for PPMF."""
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel

from outhad_contextkit.memory.privacy.enums import PrivacyLevel, SensitiveSpanType


class SensitiveSpan(BaseModel):
    """Represents a detected sensitive span in text."""
    text: str
    start: int
    end: int
    span_type: SensitiveSpanType
    privacy_level: PrivacyLevel
    confidence: float
    metadata: Optional[dict] = None


class ClassificationResult(BaseModel):
    """Result of sensitivity classification."""
    original_text: str
    has_sensitive_content: bool
    overall_privacy_level: PrivacyLevel
    sensitive_spans: List[SensitiveSpan]
    processing_time_ms: float


class BaseClassifier(ABC):
    """Abstract base classifier for sensitivity detection."""
    
    @abstractmethod
    def classify(self, text: str) -> ClassificationResult:
        """Classify text for sensitive content."""
        pass
    
    @abstractmethod
    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts."""
        pass

