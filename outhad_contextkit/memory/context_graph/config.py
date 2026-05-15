"""Configuration models for the Context-Graph Layer (CGL).

All fields have safe defaults. When ``ContextGraphConfig.enabled`` is ``False``
(the default), every CGL code path short-circuits to a no-op so existing
behaviour of ``Memory`` is byte-identical to the pre-feature main branch.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DecayConfig(BaseModel):
    """Edge weight + node relevance decay parameters."""

    enabled: bool = Field(default=True)
    half_life_days: float = Field(
        default=30.0, gt=0, description="t½ for exponential edge weight decay"
    )
    min_edge_weight: float = Field(
        default=0.05, ge=0, le=1, description="Edges below this weight are pruned"
    )
    min_node_relevance: float = Field(
        default=0.02, ge=0, le=1, description="Nodes below this relevance are archived"
    )
    tick_on_read: bool = Field(
        default=True, description="Lazily apply decay during search traversals"
    )


class LLMEdgeConfig(BaseModel):
    """Phase C — LLM-driven semantic edge synthesis parameters."""

    enabled: bool = Field(
        default=False,
        description="Master switch. Mirror of EdgeSynthesisConfig.use_llm_inference "
        "(either flag set opts in); kept as its own model so the LLM knobs cluster.",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Optional LLM provider override. When None the synthesiser "
        "reuses ``Memory.llm`` (the extraction/update LLM).",
    )
    max_pairs_per_insert: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Hard cap on LLM classification calls per new memory.",
    )
    min_topic_similarity: float = Field(
        default=0.80,
        ge=0,
        le=1,
        description="Only LLM-classify pairs whose topic cosine ≥ this floor.",
    )
    min_confidence: float = Field(
        default=0.60,
        ge=0,
        le=1,
        description="Drop LLM-inferred edges with confidence below this threshold.",
    )
    daily_token_budget: int = Field(
        default=100_000,
        ge=0,
        description="0 disables the budget. Enforced per ``user_id`` per UTC day.",
    )
    tokens_per_call_estimate: int = Field(
        default=500,
        ge=1,
        description="Conservative token estimate used for pre-call budget checks.",
    )
    timeout_seconds: float = Field(
        default=8.0,
        gt=0,
        description="Per-call LLM timeout. Exhausted calls fail-soft to no edges.",
    )
    retry_on_schema_violation: bool = Field(
        default=True,
        description="Retry once when the LLM returns malformed JSON.",
    )
    cooldown_seconds: int = Field(
        default=900,
        ge=0,
        description="After budget exhaustion, cool down for N seconds before retrying.",
    )


class EdgeSynthesisConfig(BaseModel):
    """Controls which structural edges get synthesised on memory add/update."""

    enable_reply_to: bool = Field(default=True)
    enable_topic_similar: bool = Field(default=True)
    enable_document_link: bool = Field(default=True)
    enable_temporal_next: bool = Field(default=True)
    topic_top_k: int = Field(
        default=5, ge=1, le=50, description="Max TOPIC_SIMILAR edges per insert"
    )
    topic_min_similarity: float = Field(
        default=0.75, ge=0, le=1, description="Cosine similarity floor for topic edges"
    )
    reply_to_window_seconds: int = Field(
        default=1800,
        ge=1,
        description="Max seconds between memories in the same run to create REPLY_TO",
    )
    use_llm_inference: bool = Field(
        default=False,
        description="Opt-in LLM edge synthesis (Phase C). Equivalent to ``llm.enabled``.",
    )
    llm: LLMEdgeConfig = Field(default_factory=LLMEdgeConfig)


class RetrievalConfig(BaseModel):
    """Graph-first retrieval parameters."""

    enabled: bool = Field(
        default=True,
        description="Use graph-first retrieval when context_graph.enabled is True",
    )
    seed_top_k: int = Field(default=10, ge=1, le=100)
    expansion_depth: int = Field(default=2, ge=0, le=4)
    max_candidates: int = Field(default=50, ge=1, le=500)
    edge_weight_floor: float = Field(default=0.1, ge=0, le=1)
    alpha_dense: float = Field(default=0.55, ge=0, le=1)
    beta_bm25: float = Field(default=0.15, ge=0, le=1)
    gamma_graph: float = Field(default=0.30, ge=0, le=1)
    # Phase D — retriever dispatcher + Personalised PageRank knobs.
    algorithm: Literal["bfs", "ppr"] = Field(
        default="bfs",
        description="Which retriever to use. 'bfs' (default) preserves pre-Phase-D "
        "behaviour; 'ppr' swaps in the PersonalisedPageRankRetriever.",
    )
    ppr_damping: float = Field(
        default=0.85,
        gt=0,
        lt=1,
        description="PageRank damping factor (1-teleport prob).",
    )
    ppr_max_iter: int = Field(
        default=200,
        ge=5,
        le=500,
        description="Maximum power-iteration steps.",
    )
    ppr_tolerance: float = Field(
        default=1e-6,
        gt=0,
        description="Convergence tolerance for power iteration.",
    )
    # Phase F1 — MSPR additive scoring terms. All default to 0.0 so the
    # retriever collapses to the pre-MSPR formula (α·dense + β·lex + γ·graph)
    # when the operator has not opted in.
    delta_personal: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Weight of the personalised feedback term "
        "(δ·personal_boost). 0 = disabled.",
    )
    epsilon_frequency: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Weight of the log-scaled access_count term "
        "(ε·frequency). 0 = disabled.",
    )
    zeta_success: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Weight of the (query_hash, memory_id) success-rate "
        "term (ζ·success). 0 = disabled.",
    )


class ContextGraphConfig(BaseModel):
    """Master config for the Context-Graph Layer."""

    enabled: bool = Field(
        default=False,
        description="Master switch. Off = zero behaviour change.",
    )
    backend: Literal["networkx", "neo4j"] = Field(default="networkx")
    persist_path: Optional[str] = Field(
        default=None,
        description="Snapshot path for networkx backend (defaults to "
        "{outhad_contextkit_dir}/context_graph.pkl when None)",
    )
    decay: DecayConfig = Field(default_factory=DecayConfig)
    edges: EdgeSynthesisConfig = Field(default_factory=EdgeSynthesisConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    log_changes: bool = Field(
        default=True,
        description="Write mutation events to context_graph_changes SQLite table",
    )
