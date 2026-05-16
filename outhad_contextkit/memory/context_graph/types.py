"""Core dataclasses for the Context-Graph Layer (CGL)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class EdgeType(str, Enum):
    """Structural edge taxonomy between ``MemoryNode``s."""

    REPLY_TO = "REPLY_TO"
    TOPIC_SIMILAR = "TOPIC_SIMILAR"
    DOCUMENT_LINK = "DOCUMENT_LINK"
    TEMPORAL_NEXT = "TEMPORAL_NEXT"
    UPDATED_FROM = "UPDATED_FROM"
    CAUSAL = "CAUSAL"
    #  LLM-inferred semantic edges.
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    REFINES = "REFINES"
    ELABORATES = "ELABORATES"


def _utcnow() -> datetime:
    return datetime.utcnow()


@dataclass
class MemoryNode:
    """A first-class citizen of the context graph.

    ``id`` is the vector-store memory UUID, so the CGL is keyed to the same
    primary identifier used by every other subsystem.
    """

    id: str
    hash: str
    created_at: datetime
    updated_at: datetime
    version: int = 1
    prev_version_id: Optional[str] = None
    relevance: float = 1.0
    relevance_floor: float = 0.0
    archived: bool = False
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    last_accessed_at: Optional[datetime] = None
    #MSPR counters. Default 0/None so existing serialised
    # nodes (pickle / JSONL / Neo4j rows) deserialise with safe zero-state.
    access_count: int = 0
    helpful_count: int = 0
    unhelpful_count: int = 0
    last_feedback_at: Optional[datetime] = None
    #  tenant routing fields. NULL on legacy nodes; the
    # resolver treats NULL as the default tenant so reads keep working.
    tenant_id: Optional[str] = None
    sub_tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEdge:
    """A weighted, typed relationship between two memory nodes."""

    src: str
    dst: str
    type: EdgeType
    weight: float = 1.0
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    evidence: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangeEvent:
    """A record appended to the change timeline.

    ``event_type`` is one of:
    ``node_added``, ``node_updated``, ``node_archived``, ``node_deleted``,
    ``edge_added``, ``edge_decayed``, ``edge_pruned``.
    """

    event_type: str
    target_id: str
    timestamp: datetime
    user_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
