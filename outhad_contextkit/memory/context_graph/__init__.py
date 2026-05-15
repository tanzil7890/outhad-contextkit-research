"""Context-Graph Layer (CGL) for Outhad_ContextKit.

Enabling the CGL turns every memory into a first-class graph node with typed,
weighted, and decaying edges. See
``doc_extra/CONTEXT_GRAPH/CONTEXT_GRAPH_IMPLEMENTATION_GUIDE.md`` for the full
design.

This module only re-exports lightweight symbols so ``import outhad_contextkit``
remains cheap when the feature flag is off.
"""
from __future__ import annotations

from outhad_contextkit.memory.context_graph.config import (
    ContextGraphConfig,
    DecayConfig,
    EdgeSynthesisConfig,
    RetrievalConfig,
)
from outhad_contextkit.memory.context_graph.types import (
    ChangeEvent,
    EdgeType,
    MemoryEdge,
    MemoryNode,
)

__all__ = [
    "ContextGraphConfig",
    "DecayConfig",
    "EdgeSynthesisConfig",
    "RetrievalConfig",
    "EdgeType",
    "MemoryNode",
    "MemoryEdge",
    "ChangeEvent",
    "build_context_graph",
]


def build_context_graph(config: ContextGraphConfig, *, neo4j_driver=None):
    """Construct a :class:`ContextGraph` from a config.

    Imports are kept local so the optional ``networkx`` dependency is only
    required when the feature is actually enabled.
    """
    if not config.enabled:
        return None

    from outhad_contextkit.memory.context_graph.changelog import ContextChangeLog
    from outhad_contextkit.memory.context_graph.facade import ContextGraph

    backend = None
    if config.backend == "neo4j":
        if neo4j_driver is None:
            raise ValueError(
                "ContextGraphConfig.backend='neo4j' requires a driver. "
                "Enable the entity graph (graph_store) first so the driver can be reused."
            )
        from outhad_contextkit.memory.context_graph.backends.neo4j_backend import (
            Neo4jBackend,
        )

        backend = Neo4jBackend(neo4j_driver)
    else:
        from outhad_contextkit.memory.context_graph.backends.networkx_backend import (
            NetworkXBackend,
        )

        backend = NetworkXBackend()

    changelog = None
    if config.log_changes:
        db_path = _resolve_changelog_path(config)
        if db_path:
            changelog = ContextChangeLog(db_path)

    return ContextGraph(config=config, backend=backend, changelog=changelog)


def _resolve_changelog_path(config: ContextGraphConfig):
    import os

    if config.persist_path:
        base_dir = os.path.dirname(os.path.abspath(config.persist_path))
    else:
        base_dir = os.path.join(os.path.expanduser("~"), ".outhad_contextkit")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, "context_graph_changes.db")
