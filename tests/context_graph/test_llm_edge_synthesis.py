""" LLM-driven semantic edge synthesis tests (mock LLM only).

Covers:
* each label (SUPPORTS / CONTRADICTS / REFINES / ELABORATES) → real edge,
* ``NONE`` / low-confidence → no edge,
* daily_token_budget=0 with LLMBudget constructed at 0 capacity,
* schema-violation retry path (first reply garbage, second reply JSON),
* CONTRADICTS stickiness under tick_decay,
* disabled-CGL no-op path still records zero LLM calls.
"""
from __future__ import annotations

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.memory.context_graph import build_context_graph
from outhad_contextkit.memory.context_graph.budget import LLMBudget
from outhad_contextkit.memory.context_graph.builder import IncrementalGraphBuilder
from outhad_contextkit.memory.context_graph.config import (
    ContextGraphConfig,
    EdgeSynthesisConfig,
    LLMEdgeConfig,
)
from outhad_contextkit.memory.context_graph.llm_edges import (
    LLMEdgeSynthesizer,
)
from outhad_contextkit.memory.context_graph.types import EdgeType


class ScriptedLLM:
    """LLM stand-in whose ``generate_response`` returns a scripted queue."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_response(self, messages, **_kwargs):
        self.calls.append(messages)
        if not self.replies:
            raise RuntimeError("scripted LLM ran out of replies")
        return self.replies.pop(0)


def _cfg(enabled: bool = True, **overrides) -> ContextGraphConfig:
    base = ContextGraphConfig(
        enabled=enabled,
        backend="networkx",
        log_changes=True,
    )
    if "edges" in overrides:
        base.edges = overrides["edges"]
    return base


def _builder(cfg: ContextGraphConfig, llm, budget: LLMBudget):
    cg = build_context_graph(cfg)
    builder = IncrementalGraphBuilder(cg, cfg.edges, llm=llm, llm_budget=budget)
    return cg, builder


def _seed_pair(builder, cg):
    """Insert two memories with identical embeddings so the topic edge fires."""
    builder.on_memory_created(
        "m1", "Dark roast coffee is bitter.", embedding=[1.0, 0.0]
    )
    builder.on_memory_created(
        "m2", "Arabica light roast is bright.", embedding=[1.0, 0.0]
    )


def _llm_edges(cg):
    return [
        e
        for e in cg.backend.all_edges()
        if e.type
        in {
            EdgeType.SUPPORTS,
            EdgeType.CONTRADICTS,
            EdgeType.REFINES,
            EdgeType.ELABORATES,
        }
    ]


@pytest.mark.parametrize(
    "label, expected_type",
    [
        ("SUPPORTS", EdgeType.SUPPORTS),
        ("CONTRADICTS", EdgeType.CONTRADICTS),
        ("REFINES", EdgeType.REFINES),
        ("ELABORATES", EdgeType.ELABORATES),
    ],
)
def test_each_label_creates_matching_edge(label, expected_type):
    reply = (
        f'{{"label": "{label}", "confidence": 0.82, '
        f'"evidence": "agent detected"}}'
    )
    llm = ScriptedLLM([reply])
    cfg = _cfg(
        edges=EdgeSynthesisConfig(
            llm=LLMEdgeConfig(
                enabled=True,
                daily_token_budget=0,  # unlimited for parametrized path
                max_pairs_per_insert=1,
                min_topic_similarity=0.5,
            ),
            topic_top_k=1,
            topic_min_similarity=0.5,
        )
    )
    budget = LLMBudget(
        daily_token_budget=cfg.edges.llm.daily_token_budget,
        cooldown_seconds=0,
    )
    cg, builder = _builder(cfg, llm, budget)
    _seed_pair(builder, cg)
    edges = _llm_edges(cg)
    assert len(edges) == 1
    assert edges[0].type is expected_type
    assert 0.0 <= edges[0].weight <= 1.0
    assert edges[0].metadata.get("source") == "llm"


def test_none_label_does_not_create_edge():
    llm = ScriptedLLM(['{"label": "NONE", "confidence": 0.9, "evidence": "n/a"}'])
    cfg = _cfg(
        edges=EdgeSynthesisConfig(
            llm=LLMEdgeConfig(
                enabled=True, daily_token_budget=0, max_pairs_per_insert=1,
                min_topic_similarity=0.5,
            ),
            topic_top_k=1,
            topic_min_similarity=0.5,
        )
    )
    budget = LLMBudget(0, cooldown_seconds=0)
    cg, builder = _builder(cfg, llm, budget)
    _seed_pair(builder, cg)
    assert _llm_edges(cg) == []


def test_low_confidence_is_dropped():
    llm = ScriptedLLM(
        ['{"label": "SUPPORTS", "confidence": 0.3, "evidence": "weak"}']
    )
    cfg = _cfg(
        edges=EdgeSynthesisConfig(
            llm=LLMEdgeConfig(
                enabled=True,
                daily_token_budget=0,
                max_pairs_per_insert=1,
                min_topic_similarity=0.5,
                min_confidence=0.6,
            ),
            topic_top_k=1,
            topic_min_similarity=0.5,
        )
    )
    budget = LLMBudget(0, cooldown_seconds=0)
    cg, builder = _builder(cfg, llm, budget)
    _seed_pair(builder, cg)
    assert _llm_edges(cg) == []


def test_zero_budget_blocks_llm_entirely():
    llm = ScriptedLLM([])  # would raise if called
    cfg = _cfg(
        edges=EdgeSynthesisConfig(
            llm=LLMEdgeConfig(
                enabled=True,
                daily_token_budget=1,  # tiny
                tokens_per_call_estimate=999,
                max_pairs_per_insert=1,
                min_topic_similarity=0.5,
            ),
            topic_top_k=1,
            topic_min_similarity=0.5,
        )
    )
    budget = LLMBudget(
        daily_token_budget=1, cooldown_seconds=0
    )
    cg, builder = _builder(cfg, llm, budget)
    _seed_pair(builder, cg)
    assert llm.calls == []
    assert _llm_edges(cg) == []


def test_schema_violation_retry_then_success():
    llm = ScriptedLLM(
        [
            "not json at all",
            '{"label": "SUPPORTS", "confidence": 0.8, "evidence": "retry worked"}',
        ]
    )
    cfg = _cfg(
        edges=EdgeSynthesisConfig(
            llm=LLMEdgeConfig(
                enabled=True,
                daily_token_budget=0,
                retry_on_schema_violation=True,
                max_pairs_per_insert=1,
                min_topic_similarity=0.5,
            ),
            topic_top_k=1,
            topic_min_similarity=0.5,
        )
    )
    budget = LLMBudget(0, cooldown_seconds=0)
    cg, builder = _builder(cfg, llm, budget)
    _seed_pair(builder, cg)
    edges = _llm_edges(cg)
    assert len(edges) == 1
    assert edges[0].type is EdgeType.SUPPORTS
    assert len(llm.calls) == 2


def test_schema_violation_no_retry_skips_silently():
    llm = ScriptedLLM(["still not json"])
    cfg = _cfg(
        edges=EdgeSynthesisConfig(
            llm=LLMEdgeConfig(
                enabled=True,
                daily_token_budget=0,
                retry_on_schema_violation=False,
                max_pairs_per_insert=1,
                min_topic_similarity=0.5,
            ),
            topic_top_k=1,
            topic_min_similarity=0.5,
        )
    )
    budget = LLMBudget(0, cooldown_seconds=0)
    cg, builder = _builder(cfg, llm, budget)
    _seed_pair(builder, cg)
    assert _llm_edges(cg) == []
    assert len(llm.calls) == 1


def test_contradicts_survives_tick_decay():
    llm = ScriptedLLM(
        [
            '{"label": "CONTRADICTS", "confidence": 0.9, "evidence": "opposed"}',
            '{"label": "SUPPORTS", "confidence": 0.9, "evidence": "aligned"}',
        ]
    )
    cfg = _cfg(
        edges=EdgeSynthesisConfig(
            llm=LLMEdgeConfig(
                enabled=True,
                daily_token_budget=0,
                max_pairs_per_insert=1,
                min_topic_similarity=0.5,
            ),
            topic_top_k=1,
            topic_min_similarity=0.5,
        )
    )
    budget = LLMBudget(0, cooldown_seconds=0)
    cg, builder = _builder(cfg, llm, budget)

    # First pair → CONTRADICTS
    builder.on_memory_created("m1", "A", embedding=[1.0, 0.0])
    builder.on_memory_created("m2", "B", embedding=[1.0, 0.0])
    # Second pair → SUPPORTS (on m3, against m2)
    builder.on_memory_created("m3", "C", embedding=[1.0, 0.0])

    # Force an aggressive decay: zero half-life is invalid (gt=0), so use
    # something tiny plus fake 'now' far in the future.
    import datetime as _dt
    future = _dt.datetime.utcnow() + _dt.timedelta(days=365 * 5)
    cg.config.decay.half_life_days = 0.01
    cg.config.decay.min_edge_weight = 0.05
    cg.tick_decay(now=future)

    types_remaining = {e.type for e in cg.backend.all_edges()}
    assert EdgeType.CONTRADICTS in types_remaining, (
        "CONTRADICTS edge must be sticky under decay"
    )
    # SUPPORTS is not sticky, so it should be gone after an aggressive decay.
    assert EdgeType.SUPPORTS not in types_remaining


def test_llm_synth_disabled_when_flag_off():
    llm = ScriptedLLM([])
    cfg = _cfg(
        edges=EdgeSynthesisConfig(
            llm=LLMEdgeConfig(enabled=False),  # off
            topic_top_k=1,
            topic_min_similarity=0.5,
        )
    )
    budget = LLMBudget(0, cooldown_seconds=0)
    cg, builder = _builder(cfg, llm, budget)
    assert builder._llm_synth is None
    _seed_pair(builder, cg)
    assert llm.calls == []
    assert _llm_edges(cg) == []


def test_direct_synthesizer_handles_missing_text_candidates():
    llm = ScriptedLLM([])
    cfg = LLMEdgeConfig(enabled=True, daily_token_budget=0, max_pairs_per_insert=2)
    budget = LLMBudget(0, cooldown_seconds=0)
    synth = LLMEdgeSynthesizer(llm, cfg, budget)
    # candidate whose similarity is below threshold -> filtered out
    edges = synth.infer_edges("src", "text", [("cid", "ctext", 0.2)], user_id=None)
    assert edges == []
    assert llm.calls == []


def test_cooldown_blocks_followup_calls():
    llm = ScriptedLLM(
        [
            '{"label": "SUPPORTS", "confidence": 0.9, "evidence": "x"}',
        ]
    )
    cfg = LLMEdgeConfig(
        enabled=True,
        daily_token_budget=100,
        tokens_per_call_estimate=60,
        min_topic_similarity=0.5,
        max_pairs_per_insert=5,
    )
    budget = LLMBudget(daily_token_budget=100, cooldown_seconds=60)
    synth = LLMEdgeSynthesizer(llm, cfg, budget)
    edges = synth.infer_edges(
        "src",
        "text",
        [("c1", "a", 0.9), ("c2", "b", 0.9)],
        user_id="u1",
    )
    # First call consumes the whole budget (60 of 100, then +60 trips over).
    assert len(edges) == 1
    # Cooldown now engaged; second attempt skipped without calling LLM again.
    edges2 = synth.infer_edges(
        "src",
        "text",
        [("c3", "c", 0.9)],
        user_id="u1",
    )
    assert edges2 == []
    assert len(llm.calls) == 1


def test_changelog_records_edge_added_with_source_llm(tmp_path):
    llm = ScriptedLLM(
        ['{"label": "REFINES", "confidence": 0.9, "evidence": "clarify"}']
    )
    cfg = _cfg(
        edges=EdgeSynthesisConfig(
            llm=LLMEdgeConfig(
                enabled=True,
                daily_token_budget=0,
                max_pairs_per_insert=1,
                min_topic_similarity=0.5,
            ),
            topic_top_k=1,
            topic_min_similarity=0.5,
        )
    )
    cfg.persist_path = str(tmp_path / "cg.pkl")
    budget = LLMBudget(0, cooldown_seconds=0)
    cg, builder = _builder(cfg, llm, budget)
    _seed_pair(builder, cg)

    rows = list(cg.changelog.all_events())
    edge_events = [
        ev
        for ev in rows
        if ev.event_type == "edge_added" and ev.payload.get("type") == "REFINES"
    ]
    assert edge_events, "expected an edge_added event with REFINES label"


def test_parser_tolerates_fenced_json():
    llm = ScriptedLLM(
        ["```json\n{\"label\": \"SUPPORTS\", \"confidence\": 0.75, \"evidence\": \"x\"}\n```"]
    )
    cfg = LLMEdgeConfig(
        enabled=True, daily_token_budget=0, max_pairs_per_insert=1, min_topic_similarity=0.5
    )
    budget = LLMBudget(0, cooldown_seconds=0)
    synth = LLMEdgeSynthesizer(llm, cfg, budget)
    edges = synth.infer_edges("src", "text", [("cid", "ctext", 0.9)], user_id=None)
    assert len(edges) == 1
    assert edges[0].type is EdgeType.SUPPORTS


def test_cgl_disabled_means_builder_never_constructed():
    cfg = ContextGraphConfig(enabled=False)
    # With CGL off, build_context_graph returns None and the builder is never
    # created in Memory._init_context_graph. Simulate that contract directly.
    cg = build_context_graph(cfg)
    assert cg is None
