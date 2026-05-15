"""Data types for TCMGM (Temporal-Causal Multimodal Graph Memory)."""
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TemporalEvent(BaseModel):
    """An event with temporal and modal attributes."""
    id: str
    content: str
    timestamp: datetime
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")
    modality: str = Field(default="text", description="Modality type: text, image, audio, video")
    embedding: Optional[List[float]] = Field(default=None, description="Embedding vector for the event")
    image_hash: Optional[str] = Field(default=None, description="Hash for image modality")
    audio_hash: Optional[str] = Field(default=None, description="Hash for audio modality")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    agent_id: Optional[str] = Field(default=None, description="Agent identifier")


class TemporalRelation(BaseModel):
    """A relationship between two events with temporal context."""
    source_id: str = Field(description="Source event ID")
    target_id: str = Field(description="Target event ID")
    relation_type: str = Field(description="Type of relationship")
    timestamp: datetime = Field(description="When the relationship was established")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")
    temporal_type: Optional[str] = Field(default=None, description="Temporal relationship: before, after, during, etc.")
    causal_type: Optional[str] = Field(default=None, description="Causal relationship: caused_by, leads_to, etc.")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")


class TimeWindow(BaseModel):
    """A time window for querying events."""
    start: Optional[datetime] = Field(default=None, description="Start time of window")
    end: Optional[datetime] = Field(default=None, description="End time of window")
    duration_minutes: Optional[int] = Field(default=None, description="Duration in minutes")
    
    def contains(self, timestamp: datetime) -> bool:
        """
        Check if timestamp is within window.
        
        Args:
            timestamp: Timestamp to check
            
        Returns:
            True if timestamp is within window, False otherwise
        """
        if self.start and timestamp < self.start:
            return False
        if self.end and timestamp > self.end:
            return False
        return True


class CausalLink(BaseModel):
    """A causal link between events."""
    cause_id: str = Field(description="ID of the causing event")
    effect_id: str = Field(description="ID of the effect event")
    causal_type: str = Field(description="Type of causality: caused_by, leads_to, enables, prevents, correlates_with")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")
    evidence: Optional[str] = Field(default=None, description="Evidence or explanation for the causal link")
    timestamp: datetime = Field(description="When the causal link was identified")

