"""Phase P2/P3/P6 — cache correctness tests.

Each cache must:
- Hit on second identical call (no second backend call).
- Miss on differing input.
- Evict when capacity exceeded (no unbounded growth).
- Be thread-safe — N threads with same key produce ≤ 1 backend call.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# P6 — query embedding cache
# ---------------------------------------------------------------------------

def test_query_embed_cache_hits_on_repeat():
    from outhad_contextkit.memory.temporal.orchestrator import (
        _cached_query_embed,
        clear_query_embed_cache,
    )

    clear_query_embed_cache()
    em = MagicMock()
    em.embed.return_value = [0.1, 0.2, 0.3]

    a = _cached_query_embed(em, "hello", "search")
    b = _cached_query_embed(em, "hello", "search")
    assert a == b == [0.1, 0.2, 0.3]
    # Second call must not re-embed.
    assert em.embed.call_count == 1


def test_query_embed_cache_misses_on_different_query():
    from outhad_contextkit.memory.temporal.orchestrator import (
        _cached_query_embed,
        clear_query_embed_cache,
    )

    clear_query_embed_cache()
    em = MagicMock()
    em.embed.side_effect = lambda q, m: [hash(q) % 100 / 100.0]

    _cached_query_embed(em, "alpha", "search")
    _cached_query_embed(em, "beta", "search")
    assert em.embed.call_count == 2


def test_query_embed_cache_evicts_above_capacity():
    """Drop the smallest cap to assert eviction without filling 1024 entries."""
    from outhad_contextkit.memory.temporal import orchestrator

    orchestrator.clear_query_embed_cache()
    orig_cap = orchestrator._QUERY_EMBED_CACHE_CAP
    orchestrator._QUERY_EMBED_CACHE_CAP = 4
    try:
        em = MagicMock()
        em.embed.side_effect = lambda q, m: [float(hash(q) % 100) / 100.0]
        for i in range(10):
            orchestrator._cached_query_embed(em, f"q{i}", "search")
        # Cache holds at most cap entries.
        assert len(orchestrator._QUERY_EMBED_CACHE) <= 4
    finally:
        orchestrator._QUERY_EMBED_CACHE_CAP = orig_cap


# ---------------------------------------------------------------------------
# P3 — causal LLM cache
# ---------------------------------------------------------------------------

def test_causal_llm_cache_short_circuits_repeat_extraction():
    from outhad_contextkit.memory.temporal.causal_extractor import (
        CausalExtractor,
        clear_causal_llm_cache,
    )

    clear_causal_llm_cache()

    llm = MagicMock()
    llm.generate_response.return_value = (
        '{"causal_links": [{"cause_id": "e1", "effect_id": "e2", '
        '"causal_type": "leads_to", "confidence": 0.9, '
        '"evidence": "test"}]}'
    )
    ex = CausalExtractor(llm=llm)
    events = [
        {"id": "e1", "timestamp": "2026-01-01T00:00:00", "content": "deploy"},
        {"id": "e2", "timestamp": "2026-01-01T00:01:00", "content": "lag"},
    ]
    a = ex.extract_causal_links(events, use_llm=True)
    b = ex.extract_causal_links(events, use_llm=True)
    assert len(a) == 1 == len(b)
    # Second call hits cache → no second LLM round-trip.
    assert llm.generate_response.call_count == 1


def test_causal_llm_cache_misses_when_events_change():
    from outhad_contextkit.memory.temporal.causal_extractor import (
        CausalExtractor,
        clear_causal_llm_cache,
    )

    clear_causal_llm_cache()
    llm = MagicMock()
    llm.generate_response.return_value = '{"causal_links": []}'
    ex = CausalExtractor(llm=llm)

    ex.extract_causal_links(
        [{"id": "e1", "content": "a"}, {"id": "e2", "content": "b"}],
        use_llm=True,
    )
    ex.extract_causal_links(
        [{"id": "e1", "content": "a"}, {"id": "e2", "content": "X"}],
        use_llm=True,
    )
    assert llm.generate_response.call_count == 2


# ---------------------------------------------------------------------------
# P2 — CLIP cache (skipped when transformers / torch unavailable)
# ---------------------------------------------------------------------------

def test_clip_cache_returns_same_instances():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import (
        clear_clip_cache,
        get_clip,
    )

    clear_clip_cache()
    # Patch from_pretrained so test never downloads weights.
    from unittest.mock import patch

    sentinel_model = object()
    sentinel_proc = object()
    with patch(
        "transformers.CLIPModel.from_pretrained",
        return_value=sentinel_model,
    ), patch(
        "transformers.CLIPProcessor.from_pretrained",
        return_value=sentinel_proc,
    ):
        m1, p1 = get_clip("openai/clip-vit-base-patch32")
        m2, p2 = get_clip("openai/clip-vit-base-patch32")
    assert m1 is m2 is sentinel_model
    assert p1 is p2 is sentinel_proc


# ---------------------------------------------------------------------------
# P5 — Whisper transcript cache
# ---------------------------------------------------------------------------

def test_whisper_transcript_cache_round_trip():
    from outhad_contextkit.memory.temporal.audio_embedders import (
        _audio_hash,
        _transcript_cache_get,
        _transcript_cache_set,
        clear_transcript_cache,
    )

    clear_transcript_cache()
    audio_bytes = b"fake-audio-1"
    key = _audio_hash(audio_bytes)
    assert _transcript_cache_get(key) is None
    _transcript_cache_set(key, "hello world")
    assert _transcript_cache_get(key) == "hello world"


def test_whisper_transcript_cache_hits_on_repeat_call():
    """Mock client; second embed_audio with same bytes must NOT re-call API."""
    from outhad_contextkit.memory.temporal.audio_embedders import (
        WhisperAudioEmbedder,
        clear_transcript_cache,
    )

    clear_transcript_cache()
    obj = object.__new__(WhisperAudioEmbedder)
    obj._available = True
    obj.text_embedder = None
    obj._client = MagicMock()
    obj._client.audio.transcriptions.create.return_value = "transcript"
    audio_bytes = b"deterministic-audio"
    obj.embed_audio(audio_bytes)
    obj.embed_audio(audio_bytes)
    # Second call must hit cache → only 1 API call.
    assert obj._client.audio.transcriptions.create.call_count == 1


# ---------------------------------------------------------------------------
# P9 — CLAP embedding cache
# ---------------------------------------------------------------------------

def test_clap_embedding_cache_round_trip():
    from outhad_contextkit.memory.temporal.audio_embedders import (
        _audio_hash,
        _clap_cache_get,
        _clap_cache_set,
        clear_clap_embed_cache,
    )

    clear_clap_embed_cache()
    audio_bytes = b"fake-audio-2"
    key = _audio_hash(audio_bytes)
    assert _clap_cache_get(key) is None
    vec = [0.1, 0.2, 0.3] + [0.0] * 765
    _clap_cache_set(key, vec)
    cached = _clap_cache_get(key)
    assert cached == vec
    # Must return a copy so callers can't mutate the cache in place.
    cached.append(99.9)
    assert _clap_cache_get(key) == vec


def test_clip_cache_thread_safe():
    pytest.importorskip("transformers")
    from outhad_contextkit.memory.temporal._clip_cache import (
        clear_clip_cache,
        get_clip,
    )

    clear_clip_cache()
    from unittest.mock import patch

    load_calls = {"n": 0}

    def _slow_load(*a, **kw):
        time.sleep(0.05)
        load_calls["n"] += 1
        return object()

    with patch(
        "transformers.CLIPModel.from_pretrained",
        side_effect=_slow_load,
    ), patch(
        "transformers.CLIPProcessor.from_pretrained",
        side_effect=_slow_load,
    ):
        results = []
        threads = [
            threading.Thread(
                target=lambda: results.append(get_clip("dummy/clip"))
            )
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    # Despite 8 concurrent callers, the model should load exactly once.
    # Two separate from_pretrained calls expected: one for model + one
    # for processor.
    assert load_calls["n"] == 2
    # All threads got the same instances.
    first = results[0]
    for r in results[1:]:
        assert r == first
