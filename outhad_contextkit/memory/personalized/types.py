"""Dataclasses shared across the MSPR subsystem.

Kept minimal on purpose — complex behaviour belongs in ``feedback_store``,
``pipeline``, etc., not on the dataclasses themselves. All dataclasses
are JSON-serialisable via ``dataclasses.asdict()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class FeedbackEvent:
    """One user-supplied verdict on a retrieved memory.

    Written to ``feedback_events.db`` by ``FeedbackStore.append()``.
    ``id`` is the SQLite rowid; ``None`` until the row is persisted.
    ``delta_applied`` is the *post-clamp* value actually handed to
    ``ContextGraph.bump_relevance`` so later aggregations don't have to
    re-derive the clamp.
    """

    memory_id: str
    helpful: bool
    delta_applied: float
    timestamp: datetime
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    query: Optional[str] = None
    query_hash: Optional[str] = None
    id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoleContext:
    """Identity bundle passed through the retrieval pipeline.

    Only ``tenant_id`` and ``role`` are new for MSPR; the session ids
    were already honoured by ``Memory.search``. Present on this struct
    as a single source of truth for role/tenant-aware hooks (Phase F6).
    """

    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    tenant_id: Optional[str] = None
    role: Optional[str] = None
    sub_tenant_id: Optional[str] = None

    def ids(self) -> Dict[str, Optional[str]]:
        return {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
        }


__all__ = ["FeedbackEvent", "RoleContext"]
