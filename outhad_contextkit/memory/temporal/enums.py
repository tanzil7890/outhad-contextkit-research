"""Enumerations for TCMGM (Temporal-Causal Multimodal Graph Memory)."""
from enum import Enum


class ModalityType(str, Enum):
    """Types of modalities supported."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    EMBEDDING = "embedding"


class CausalType(str, Enum):
    """Types of causal relationships."""
    CAUSED_BY = "caused_by"
    LEADS_TO = "leads_to"
    ENABLES = "enables"
    PREVENTS = "prevents"
    CORRELATES_WITH = "correlates_with"


class TemporalRelationType(str, Enum):
    """Types of temporal relationships."""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    CONCURRENT = "concurrent"
    IMMEDIATELY_AFTER = "immediately_after"

