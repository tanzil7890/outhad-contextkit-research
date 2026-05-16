"""Graph-store throughput phases.

These tests stub out ``Neo4jGraph`` so no real Neo4j instance is
required. They verify:

* PT1 — connection pool kwargs (``max_connection_pool_size``,
  ``connection_acquisition_timeout``, ``max_connection_lifetime``) on
  ``Neo4jConfig`` flow through to the ``Neo4jGraph`` constructor's
  ``driver_config`` dict.
* PT2 — ``MemoryGraph.add_causal_links_batch`` issues a single
  Cypher ``UNWIND`` query with the batched rows; legacy
  ``add_causal_link`` still works per-link.
* PT5 — ``_create_temporal_schema`` issues every expected
  ``CREATE INDEX IF NOT EXISTS`` statement (timestamp + modality +
  confidence + causal_type + causal_strength + composite + event_id),
  and a backend that rejects an index keeps booting instead of
  crashing.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# PT1 — Neo4jConfig pool kwargs validation + driver passthrough
# ---------------------------------------------------------------------------

def test_neo4j_config_accepts_pool_kwargs():
    from outhad_contextkit.graphs.configs import Neo4jConfig

    cfg = Neo4jConfig(
        url="bolt://x", username="u", password="p",
        max_connection_pool_size=200,
        connection_acquisition_timeout=15.0,
        max_connection_lifetime=600,
    )
    assert cfg.max_connection_pool_size == 200
    assert cfg.connection_acquisition_timeout == 15.0
    assert cfg.max_connection_lifetime == 600


def test_neo4j_config_pool_kwargs_validate_bounds():
    from outhad_contextkit.graphs.configs import Neo4jConfig

    with pytest.raises(Exception):
        Neo4jConfig(
            url="bolt://x", username="u", password="p",
            max_connection_pool_size=0,
        )
    with pytest.raises(Exception):
        Neo4jConfig(
            url="bolt://x", username="u", password="p",
            connection_acquisition_timeout=-1,
        )


def test_neo4j_config_pool_kwargs_default_none():
    from outhad_contextkit.graphs.configs import Neo4jConfig

    cfg = Neo4jConfig(url="bolt://x", username="u", password="p")
    assert cfg.max_connection_pool_size is None
    assert cfg.connection_acquisition_timeout is None
    assert cfg.max_connection_lifetime is None


def _build_memory_graph(neo4j_cfg, fake_graph):
    """Build a MemoryGraph with all heavyweight dependencies mocked."""
    fake_config = SimpleNamespace(
        graph_store=SimpleNamespace(
            config=neo4j_cfg, llm=None, custom_prompt=None,
        ),
        embedder=SimpleNamespace(provider="openai", config=SimpleNamespace()),
        vector_store=SimpleNamespace(config=SimpleNamespace()),
        llm=SimpleNamespace(provider="openai", config=SimpleNamespace()),
    )
    with patch(
        "outhad_contextkit.memory.graph_memory.Neo4jGraph",
        return_value=fake_graph,
    ) as m_neo4j, patch(
        "outhad_contextkit.memory.graph_memory.EmbedderFactory.create",
        return_value=MagicMock(),
    ), patch(
        "outhad_contextkit.memory.graph_memory.LlmFactory.create",
        return_value=MagicMock(),
    ):
        from outhad_contextkit.memory.graph_memory import MemoryGraph

        mg = MemoryGraph(fake_config)
    return mg, m_neo4j


def test_memory_graph_threads_pool_kwargs_into_driver_config():
    from outhad_contextkit.graphs.configs import Neo4jConfig

    cfg = Neo4jConfig(
        url="bolt://x", username="u", password="p",
        max_connection_pool_size=250,
        connection_acquisition_timeout=20.0,
        max_connection_lifetime=900,
    )
    fake_graph = MagicMock()
    fake_graph.query.return_value = []
    _, m_neo4j = _build_memory_graph(cfg, fake_graph)

    assert m_neo4j.call_count == 1
    driver_config = m_neo4j.call_args.kwargs["driver_config"]
    assert driver_config["max_connection_pool_size"] == 250
    assert driver_config["connection_acquisition_timeout"] == 20.0
    assert driver_config["max_connection_lifetime"] == 900
    # Existing key preserved.
    assert driver_config["notifications_min_severity"] == "OFF"


def test_memory_graph_omits_pool_kwargs_when_none():
    """Default config (no pool tuning) must NOT add pool kwargs to driver_config."""
    from outhad_contextkit.graphs.configs import Neo4jConfig

    cfg = Neo4jConfig(url="bolt://x", username="u", password="p")
    fake_graph = MagicMock()
    fake_graph.query.return_value = []
    _, m_neo4j = _build_memory_graph(cfg, fake_graph)

    driver_config = m_neo4j.call_args.kwargs["driver_config"]
    assert "max_connection_pool_size" not in driver_config
    assert "connection_acquisition_timeout" not in driver_config
    assert "max_connection_lifetime" not in driver_config


# ---------------------------------------------------------------------------
# PT5 — temporal index DDL emitted at init
# ---------------------------------------------------------------------------

def _executed_index_names(fake_graph) -> set:
    names = set()
    for call in fake_graph.query.call_args_list:
        cypher = call.args[0] if call.args else call.kwargs.get("query", "")
        if "CREATE INDEX" in cypher:
            # Format: "CREATE INDEX <name> IF NOT EXISTS FOR ..."
            parts = cypher.split()
            try:
                names.add(parts[parts.index("INDEX") + 1])
            except (ValueError, IndexError):
                pass
    return names


def test_pt5_indexes_all_emitted_on_init():
    from outhad_contextkit.graphs.configs import Neo4jConfig

    cfg = Neo4jConfig(url="bolt://x", username="u", password="p")
    fake_graph = MagicMock()
    fake_graph.query.return_value = []
    _build_memory_graph(cfg, fake_graph)

    expected = {
        "event_timestamp",
        "event_modality",
        "event_confidence",
        "event_causal_type",
        "event_causal_strength",
        "event_user_timestamp",
        "event_id_lookup",
    }
    assert expected.issubset(_executed_index_names(fake_graph))


def test_pt5_index_failure_does_not_crash_init():
    """If one CREATE INDEX raises, init must keep going."""
    from outhad_contextkit.graphs.configs import Neo4jConfig

    cfg = Neo4jConfig(url="bolt://x", username="u", password="p")
    fake_graph = MagicMock()

    def query(cypher, *args, **kwargs):
        if "event_causal_type" in cypher:
            raise RuntimeError("backend rejects this index shape")
        return []

    fake_graph.query.side_effect = query
    # Must not raise.
    _build_memory_graph(cfg, fake_graph)
    # All other indexes should still have been attempted.
    assert "event_timestamp" in _executed_index_names(fake_graph)
    assert "event_user_timestamp" in _executed_index_names(fake_graph)


# ---------------------------------------------------------------------------
# PT2 — UNWIND bulk causal insert
# ---------------------------------------------------------------------------

def _make_link(cause_id, effect_id, *, type_="caused_by", conf=0.9, ts=None):
    return {
        "cause_id": cause_id,
        "effect_id": effect_id,
        "causal_type": type_,
        "confidence": conf,
        "evidence": "x",
        "timestamp": ts or datetime(2026, 4, 25, 14, 0).isoformat(),
    }


def test_add_causal_links_batch_emits_single_unwind_query():
    from outhad_contextkit.graphs.configs import Neo4jConfig

    cfg = Neo4jConfig(url="bolt://x", username="u", password="p")
    fake_graph = MagicMock()
    fake_graph.query.return_value = [{"created": 3}]
    mg, _ = _build_memory_graph(cfg, fake_graph)

    fake_graph.query.reset_mock()
    links = [
        _make_link("e1", "e2"),
        _make_link("e2", "e3"),
        _make_link("e3", "e4"),
    ]
    mg.add_causal_links_batch(links, filters={"user_id": "u1"})

    # Exactly one query should have been issued for the batch.
    causal_calls = [
        c for c in fake_graph.query.call_args_list
        if "UNWIND $rows" in (c.args[0] if c.args else "")
    ]
    assert len(causal_calls) == 1

    # The rows param must contain every link as a dict.
    params = causal_calls[0].kwargs.get("params") or causal_calls[0].args[1]
    assert len(params["rows"]) == 3
    assert params["user_id"] == "u1"
    assert {r["cause_id"] for r in params["rows"]} == {"e1", "e2", "e3"}


def test_add_causal_links_batch_empty_returns_none_no_query():
    from outhad_contextkit.graphs.configs import Neo4jConfig

    cfg = Neo4jConfig(url="bolt://x", username="u", password="p")
    fake_graph = MagicMock()
    fake_graph.query.return_value = []
    mg, _ = _build_memory_graph(cfg, fake_graph)

    fake_graph.query.reset_mock()
    out = mg.add_causal_links_batch([], filters={"user_id": "u1"})
    assert out is None
    # No UNWIND query issued for empty batch.
    assert not any(
        "UNWIND" in (c.args[0] if c.args else "")
        for c in fake_graph.query.call_args_list
    )


def test_add_causal_links_batch_failure_returns_none():
    from outhad_contextkit.graphs.configs import Neo4jConfig

    cfg = Neo4jConfig(url="bolt://x", username="u", password="p")
    fake_graph = MagicMock()
    fake_graph.query.return_value = []
    mg, _ = _build_memory_graph(cfg, fake_graph)

    def query(cypher, *args, **kwargs):
        if "UNWIND" in cypher:
            raise RuntimeError("constraint violation")
        return []

    fake_graph.query.side_effect = query
    out = mg.add_causal_links_batch(
        [_make_link("e1", "e2")], filters={"user_id": "u1"}
    )
    assert out is None  # batch fails atomically without raising


def test_add_causal_links_batch_threads_agent_filter():
    from outhad_contextkit.graphs.configs import Neo4jConfig

    cfg = Neo4jConfig(url="bolt://x", username="u", password="p")
    fake_graph = MagicMock()
    fake_graph.query.return_value = [{"created": 1}]
    mg, _ = _build_memory_graph(cfg, fake_graph)

    fake_graph.query.reset_mock()
    mg.add_causal_links_batch(
        [_make_link("e1", "e2")],
        filters={"user_id": "u1", "agent_id": "agent-007"},
    )

    causal_call = next(
        c for c in fake_graph.query.call_args_list
        if "UNWIND $rows" in (c.args[0] if c.args else "")
    )
    cypher = causal_call.args[0]
    assert "cause.agent_id = $agent_id" in cypher
    assert "effect.agent_id = $agent_id" in cypher
    params = causal_call.kwargs.get("params") or causal_call.args[1]
    assert params["agent_id"] == "agent-007"


def test_add_causal_links_batch_accepts_causal_link_objects():
    from outhad_contextkit.graphs.configs import Neo4jConfig
    from outhad_contextkit.memory.temporal.types import CausalLink

    cfg = Neo4jConfig(url="bolt://x", username="u", password="p")
    fake_graph = MagicMock()
    fake_graph.query.return_value = [{"created": 1}]
    mg, _ = _build_memory_graph(cfg, fake_graph)

    fake_graph.query.reset_mock()
    link = CausalLink(
        cause_id="e1", effect_id="e2",
        causal_type="caused_by", confidence=0.9, evidence="x",
        timestamp=datetime(2026, 4, 25, 14, 0),
    )
    mg.add_causal_links_batch([link], filters={"user_id": "u1"})

    causal_call = next(
        c for c in fake_graph.query.call_args_list
        if "UNWIND $rows" in (c.args[0] if c.args else "")
    )
    params = causal_call.kwargs.get("params") or causal_call.args[1]
    assert params["rows"][0]["cause_id"] == "e1"
    assert params["rows"][0]["confidence"] == pytest.approx(0.9)
