"""(cross-modal prefilter) + A4 (calibrated thresholds) +
A6 (cross-encoder reranker) tests.

P8 — prefilter:
- modality_filter drops non-matching candidates before cosine loop.
- time_window drops out-of-window candidates before cosine loop.
- prefilter_max_candidates caps the candidate pool.

A4 — calibrated thresholds:
- text↔text uses calibrated 0.55 floor, drops 0.30 cosine result.
- text↔image uses lower 0.22 floor, keeps 0.30 cosine result.
- use_calibrated_thresholds=False reverts to flat min_similarity.

A6 — reranker:
- get_reranker returns None when sentence-transformers absent
  (strict=False).
- rerank_results returns input unchanged when model is None.
- rerank_results reorders by cross-encoder score when mock model
  installed.
- fused_search rerank=True triggers reranker but fails-soft on
  missing dependency.
"""
from __future__ import annotations

from datetime import datetime
import numpy as np
import pytest

from outhad_contextkit.memory.temporal.cross_modal import (
    MODALITY_THRESHOLDS,
    cross_modal_search,
)
from outhad_contextkit.memory.temporal.reranker import (
    DEFAULT_RERANKER_MODEL,
    RerankerUnavailable,
    clear_reranker_cache,
    get_reranker,
    rerank_results,
)


# ---------------------------------------------------------------------------
# P8 — prefilter
# ---------------------------------------------------------------------------

def _vec(seed: int, dim: int = 8) -> list:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return list(v / np.linalg.norm(v))


def test_modality_filter_drops_non_matching():
    qv = _vec(1)
    candidates = [
        {"id": "a", "embedding": _vec(2), "modality": "image", "content": "img"},
        {"id": "b", "embedding": _vec(3), "modality": "text", "content": "txt"},
    ]
    out = cross_modal_search(
        qv, candidates, modality_filter="image", use_calibrated_thresholds=False
    )
    assert all(r["modality"] == "image" for r in out)
    assert {r["id"] for r in out} == {"a"}


def test_time_window_drops_out_of_window():
    qv = _vec(1)
    candidates = [
        {
            "id": "in_window",
            "embedding": _vec(2),
            "modality": "text",
            "content": "in",
            "timestamp": "2024-06-15T12:00:00",
        },
        {
            "id": "out_of_window",
            "embedding": _vec(3),
            "modality": "text",
            "content": "out",
            "timestamp": "2020-01-01T00:00:00",
        },
    ]
    out = cross_modal_search(
        qv,
        candidates,
        time_window=(datetime(2024, 1, 1), datetime(2024, 12, 31)),
        use_calibrated_thresholds=False,
    )
    assert {r["id"] for r in out} == {"in_window"}


def test_prefilter_max_candidates_caps_pool():
    qv = _vec(1)
    candidates = [
        {"id": str(i), "embedding": _vec(i + 10), "modality": "text", "content": f"c{i}"}
        for i in range(20)
    ]
    out = cross_modal_search(
        qv,
        candidates,
        prefilter_max_candidates=5,
        use_calibrated_thresholds=False,
        top_k=20,
    )
    # At most 5 made it past the prefilter.
    assert len(out) <= 5


# ---------------------------------------------------------------------------
# A4 — calibrated thresholds
# ---------------------------------------------------------------------------

def test_text_text_threshold_drops_low_cosine():
    """Synthesise a candidate with cosine ≈ 0.3 — below text↔text 0.55 floor."""
    # Build query/candidate vectors with a known cosine.
    a = np.array([1.0, 0.0])
    b = np.array([0.3, 0.95394])  # cosine ≈ 0.3
    out = cross_modal_search(
        list(a),
        [{"id": "low", "embedding": list(b), "modality": "text", "content": "x"}],
        query_modality="text",
        use_calibrated_thresholds=True,
    )
    assert out == []


def test_text_image_threshold_keeps_modest_cosine():
    """Same cosine ≈ 0.3 — above text↔image 0.22 floor → kept."""
    a = np.array([1.0, 0.0])
    b = np.array([0.3, 0.95394])  # cosine ≈ 0.3
    out = cross_modal_search(
        list(a),
        [{"id": "kept", "embedding": list(b), "modality": "image", "content": "x"}],
        query_modality="text",
        use_calibrated_thresholds=True,
    )
    assert len(out) == 1
    assert out[0]["id"] == "kept"


def test_threshold_disabled_reverts_to_global_floor():
    a = np.array([1.0, 0.0])
    b = np.array([0.3, 0.95394])
    out = cross_modal_search(
        list(a),
        [{"id": "x", "embedding": list(b), "modality": "text", "content": "x"}],
        query_modality="text",
        use_calibrated_thresholds=False,
        min_similarity=0.0,
    )
    assert len(out) == 1


def test_modality_thresholds_table_well_formed():
    # Symmetric pairs should mostly use the same threshold.
    assert MODALITY_THRESHOLDS[("text", "image")] == MODALITY_THRESHOLDS[("image", "text")]
    assert MODALITY_THRESHOLDS[("text", "audio")] == MODALITY_THRESHOLDS[("audio", "text")]
    # Same-modality thresholds higher than cross-modal.
    assert MODALITY_THRESHOLDS[("text", "text")] > MODALITY_THRESHOLDS[("text", "image")]


# ---------------------------------------------------------------------------
# A6 — cross-encoder reranker
# ---------------------------------------------------------------------------

def test_get_reranker_returns_none_when_dep_missing(monkeypatch):
    """Force the import to fail; default strict=False returns None."""
    clear_reranker_cache()

    import sys

    real_module = sys.modules.pop("sentence_transformers", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    try:
        assert get_reranker() is None
    finally:
        if real_module is not None:
            sys.modules["sentence_transformers"] = real_module
        clear_reranker_cache()


def test_get_reranker_strict_raises_when_missing(monkeypatch):
    """strict=True must raise RerankerUnavailable on ImportError."""
    clear_reranker_cache()

    import sys

    real_module = sys.modules.pop("sentence_transformers", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    try:
        with pytest.raises(RerankerUnavailable):
            get_reranker(strict=True)
    finally:
        if real_module is not None:
            sys.modules["sentence_transformers"] = real_module
        clear_reranker_cache()


# ---------------------------------------------------------------------------
# Live rerank tests via a fake sentence_transformers stub.
#
# Note on the torch segfault: importing the real ``sentence_transformers``
# pulls torch's C extension, which segfaults during pytest's interpreter
# shutdown on Python 3.11 + macOS arm64 + torch 2.x. To avoid that path
# while still exercising the real ``rerank_results`` code, we install a
# minimal in-memory stub for ``sentence_transformers`` BEFORE any test in
# this module touches the reranker. The stub's CrossEncoder.predict
# returns deterministic scores so we can assert the rerank ordering.
# ---------------------------------------------------------------------------

class _FakeCrossEncoder:
    """In-process double for ``sentence_transformers.CrossEncoder``.

    ``score_map`` maps the candidate text to a deterministic score so
    tests can assert exact rerank ordering. Anything missing falls back
    to a stable hash-derived score so behaviour is deterministic across
    runs.
    """

    score_map: dict = {}

    def __init__(self, model_name: str):
        self.model_name = model_name

    def predict(self, pairs, show_progress_bar: bool = False):
        out = []
        for _query, candidate in pairs:
            score = type(self).score_map.get(
                candidate, (hash(candidate) % 1000) / 1000.0
            )
            out.append(float(score))
        return out


@pytest.fixture
def stub_sentence_transformers(monkeypatch):
    """Install a fake ``sentence_transformers`` so torch is never imported."""
    import sys
    import types

    from outhad_contextkit.memory.temporal.reranker import clear_reranker_cache

    clear_reranker_cache()

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = _FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    # Reset the per-test score map so previous test data does not leak.
    _FakeCrossEncoder.score_map = {}
    yield fake_module
    clear_reranker_cache()


def test_rerank_results_returns_input_when_model_none_via_stub(
    stub_sentence_transformers, monkeypatch
):
    """get_reranker → None path: input list returned unchanged."""
    # Patch the module attribute via the imported module object — the
    # parent package uses a lazy ``__getattr__`` so ``monkeypatch.setattr``
    # by dotted string can't traverse ``temporal.reranker``.
    from outhad_contextkit.memory.temporal import reranker as _reranker_mod

    monkeypatch.setattr(
        _reranker_mod, "get_reranker", lambda *args, **kwargs: None
    )
    out = rerank_results(
        query="anything",
        results=[
            {"id": "a", "content": "alpha", "final_score": 0.5},
            {"id": "b", "content": "beta", "final_score": 0.4},
        ],
    )
    assert [r["id"] for r in out] == ["a", "b"]


def test_rerank_results_reorders_with_stubbed_cross_encoder(
    stub_sentence_transformers,
):
    """Real rerank_results path against the fake CrossEncoder."""
    from outhad_contextkit.memory.temporal.reranker import (
        DEFAULT_RERANKER_MODEL,
        get_reranker,
    )

    # Plug in deterministic scores for our two candidates.
    _FakeCrossEncoder.score_map = {"alpha": 0.1, "beta": 0.9}

    # Sanity — the cache returns our fake instance, not None.
    model = get_reranker(DEFAULT_RERANKER_MODEL)
    assert isinstance(model, _FakeCrossEncoder)

    out = rerank_results(
        query="q",
        results=[
            {"id": "a", "content": "alpha"},
            {"id": "b", "content": "beta"},
        ],
    )
    assert [r["id"] for r in out] == ["b", "a"]
    assert out[0]["rerank_score"] == pytest.approx(0.9)
    assert out[0]["final_score"] == pytest.approx(0.9)


def test_rerank_predict_failure_falls_back_to_input_order(
    stub_sentence_transformers, monkeypatch
):
    """Cross-encoder ``predict`` exception must not break retrieval."""

    class _BrokenEncoder(_FakeCrossEncoder):
        def predict(self, pairs, show_progress_bar=False):
            raise RuntimeError("inference failure")

    monkeypatch.setattr(
        stub_sentence_transformers, "CrossEncoder", _BrokenEncoder
    )
    from outhad_contextkit.memory.temporal.reranker import clear_reranker_cache

    clear_reranker_cache()  # force a fresh model load

    out = rerank_results(
        query="q",
        results=[
            {"id": "a", "content": "alpha"},
            {"id": "b", "content": "beta"},
        ],
    )
    # Predict raised → original ordering preserved.
    assert [r["id"] for r in out] == ["a", "b"]


def test_rerank_top_k_truncates_after_reorder(
    stub_sentence_transformers,
):
    _FakeCrossEncoder.score_map = {"alpha": 0.1, "beta": 0.9, "gamma": 0.5}
    out = rerank_results(
        query="q",
        results=[
            {"id": "a", "content": "alpha"},
            {"id": "b", "content": "beta"},
            {"id": "c", "content": "gamma"},
        ],
        top_k=2,
    )
    assert len(out) == 2
    # Top-2 by reranked score: beta (0.9) → gamma (0.5).
    assert [r["id"] for r in out] == ["b", "c"]


def test_rerank_handles_empty_input():
    assert rerank_results(query="q", results=[]) == []


def test_default_reranker_model_constant():
    assert "ms-marco" in DEFAULT_RERANKER_MODEL.lower()
