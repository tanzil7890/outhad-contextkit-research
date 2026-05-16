"""Temporal-Causal Multimodal Graph Memory (TCMGM) module.

every public symbol resolves through ``__getattr__`` so
``import outhad_contextkit.memory.temporal`` itself is free.
Heavyweight imports (``transformers``, ``torch``, ``librosa``, ``PIL``)
only land when the caller actually touches a symbol that needs them.

The list of exports is identical to the eager version; we just defer
the underlying ``importlib.import_module`` until the first attribute
access. Intentionally idempotent: every resolved symbol is cached on
this module so subsequent lookups skip the dispatch.
"""
from __future__ import annotations

import importlib
from typing import Any

# {public_symbol: submodule_path} — extend here when adding a new export.
_LAZY_EXPORTS: dict = {
    # Causal extraction
    "CausalExtractor": ".causal_extractor",
    "CAUSAL_MIN_CONFIDENCE": ".causal_extractor",
    "DEFAULT_EXTRACTION_MODE": ".causal_extractor",
    "CASCADE_MIN_CONFIDENCE": ".causal_extractor",
    "CASCADE_MIN_LINKS": ".causal_extractor",
    # Circuit breaker (C4)
    "CircuitBreaker": "._circuit_breaker",
    "CircuitOpenError": "._circuit_breaker",
    "get_breaker": "._circuit_breaker",
    "reset_breaker": "._circuit_breaker",
    "clear_breaker_registry": "._circuit_breaker",
    # Dedup (A3)
    "dedupe_events": ".dedup",
    "fingerprint": ".dedup",
    "DEFAULT_DEDUP_THRESHOLD": ".dedup",
    # Causal queries
    "find_root_causes": ".causal_queries",
    "get_causal_chain": ".causal_queries",
    "get_causal_chains_batch": ".causal_queries",
    "get_causal_subgraph": ".causal_queries",
    # Cross-modal
    "cross_modal_search": ".cross_modal",
    "find_audio_mentions_in_text": ".cross_modal",
    "find_image_mentions_in_text": ".cross_modal",
    "compute_multimodal_relevance": ".cross_modal",
    "group_by_modality": ".cross_modal",
    "MODALITY_THRESHOLDS": ".cross_modal",
    # Reranker (A6)
    "rerank_results": ".reranker",
    "get_reranker": ".reranker",
    "RerankerUnavailable": ".reranker",
    "DEFAULT_RERANKER_MODEL": ".reranker",
    # Enums
    "CausalType": ".enums",
    "ModalityType": ".enums",
    "TemporalRelationType": ".enums",
    # Multimodal
    "MultimodalContent": ".multimodal",
    "MultimodalEmbedder": ".multimodal",
    "create_audio_content": ".multimodal",
    "create_image_content": ".multimodal",
    "create_text_content": ".multimodal",
    # Production embedder (deprecated, kept for back-compat)
    "ProductionMultimodalEmbedder": ".production_embedder",
    "create_production_embedder": ".production_embedder",
    # Types
    "CausalLink": ".types",
    "TemporalEvent": ".types",
    "TemporalRelation": ".types",
    "TimeWindow": ".types",
    # Flexible embedder system
    "BaseTextEmbedder": ".base_embedders",
    "BaseImageEmbedder": ".base_embedders",
    "BaseAudioEmbedder": ".base_embedders",
    "MultimodalEmbedderBase": ".base_embedders",
    "CLIPTextEmbedder": ".text_embedders",
    "OpenAITextEmbedder": ".text_embedders",
    "CLIPImageEmbedder": ".image_embedders",
    "WhisperAudioEmbedder": ".audio_embedders",
    "CLAPAudioEmbedder": ".audio_embedders",
    "HybridAudioEmbedder": ".audio_embedders",
    "EmbedderConfig": ".embedder_factory",
    "FlexibleMultimodalEmbedder": ".embedder_factory",
    "create_embedder": ".embedder_factory",
    "create_default_embedder": ".embedder_factory",
    "create_clap_embedder": ".embedder_factory",
    "create_hybrid_embedder": ".embedder_factory",
    "create_large_clip_embedder": ".embedder_factory",
    "create_siglip_embedder": ".embedder_factory",
    "create_fp16_embedder": ".embedder_factory",
    # Timeline + orchestrator
    "TimelineQueries": ".timeline_queries",
    "RetrievalOrchestrator": ".orchestrator",
}


def __getattr__(name: str) -> Any:
    """Resolve ``name`` lazily then cache it on the module."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target, package=__name__)
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list:  # pragma: no cover - mostly for IDE autocomplete
    return sorted(set(list(globals().keys()) + list(_LAZY_EXPORTS.keys())))


__all__ = list(_LAZY_EXPORTS.keys())
