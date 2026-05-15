"""Multi-Stage Personalized Retrieval (MSPR) configuration.

All fields default OFF. When ``MSPRConfig.enabled`` is ``False`` (the default)
every MSPR code path short-circuits so behaviour of ``Memory`` is
byte-identical to the pre-feature main branch.

See ``doc_extra/PERSONALIZED_RETRIEVAL/MSPR_IMPLEMENTATION_GUIDE.md`` for the
full design and phase map.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class IntentConfig(BaseModel):
    """Phase F5 — Query intent classifier parameters."""

    enabled: bool = Field(default=False)
    strategy: Literal["regex", "llm", "hybrid"] = Field(default="regex")
    llm_timeout_seconds: float = Field(default=2.0, gt=0)
    cache_size: int = Field(
        default=1024,
        ge=0,
        description="LRU cache size for classified queries. 0 disables cache.",
    )
    enable_regex_fastpath: bool = Field(
        default=True,
        description="Run regex rules before any LLM call. Hits short-circuit.",
    )


class FeedbackConfig(BaseModel):
    """Phase F3/F4 — `record_feedback` + δ·personal_boost parameters."""

    enabled: bool = Field(default=False)
    boost_delta: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description="Positive relevance delta applied on helpful feedback.",
    )
    penalty_delta: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="Absolute value of the negative delta for unhelpful feedback.",
    )
    max_boost_per_memory: float = Field(
        default=0.60,
        ge=0,
        le=1,
        description="Cap on cumulative boost a single memory can accrue.",
    )
    pin_floor_on_boost: bool = Field(
        default=True,
        description="Pin relevance_floor after a helpful boost so decay cannot "
        "erase the signal.",
    )
    sqlite_path: Optional[str] = Field(
        default=None,
        description="Override path for feedback_events.db. None → default dir.",
    )


class FrequencyConfig(BaseModel):
    """Phase F2 — access_count + ε·frequency parameters."""

    enabled: bool = Field(default=False)
    epsilon_weight: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Weight applied to the normalised frequency term. "
        "Mirrors RetrievalConfig.epsilon_frequency for convenience; the "
        "retriever reads from RetrievalConfig, this field is a presentation hint.",
    )
    normalise: Literal["log1p", "zscore", "none"] = Field(default="log1p")


class RoleConfig(BaseModel):
    """Phase F6 — role / tenant-aware retrieval parameters."""

    enabled: bool = Field(default=False)
    tenant_field: str = Field(
        default="tenant_id",
        description="Key in memory metadata that holds the tenant identifier.",
    )
    role_field: str = Field(
        default="role",
        description="Key in memory metadata that holds the actor role.",
    )
    hard_filter: bool = Field(
        default=True,
        description="When True, deny cross-role/tenant reads pre-rerank. "
        "When False, merely deboost via RolePolicy.reweight().",
    )


class SuccessConfig(BaseModel):
    """Phase F7 — historical (query_hash, memory_id) success parameters."""

    enabled: bool = Field(default=False)
    zeta_weight: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="Mirror of RetrievalConfig.zeta_success for presentation.",
    )
    query_hash_algo: Literal["sha256", "minhash"] = Field(default="sha256")
    sqlite_path: Optional[str] = Field(default=None)
    min_sample_size: int = Field(
        default=3,
        ge=1,
        description="Only apply the success boost once we have this many samples.",
    )


class MSPRConfig(BaseModel):
    """Master config for Multi-Stage Personalized Retrieval."""

    enabled: bool = Field(
        default=False,
        description="Master switch. Off = zero behaviour change, zero imports.",
    )
    intent: IntentConfig = Field(default_factory=IntentConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    frequency: FrequencyConfig = Field(default_factory=FrequencyConfig)
    role: RoleConfig = Field(default_factory=RoleConfig)
    success: SuccessConfig = Field(default_factory=SuccessConfig)
