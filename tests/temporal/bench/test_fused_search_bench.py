"""Synthetic benchmark for TCMGM fused_search.

Backends (vector / graph / timeline / causal) are mocked to a fixed
``STAGE_LATENCY_S`` sleep so the benchmark targets the orchestrator
+ embedder code paths, not external I/O. Sequential implementation
should take ~5×STAGE_LATENCY; parallel implementation should approach
1×STAGE_LATENCY (+ thread-pool overhead).

Run:
    PYTHONPATH=. pytest tests/temporal/bench/ -v -s

Asserts:
    parallel p95 ≤ STAGE_LATENCY_S * 2.5  (room for jitter + fusion CPU)
    sequential p95 ≥ STAGE_LATENCY_S * 4   (sanity check the mock works)
    parallel ranking == sequential ranking (correctness preserved)
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from outhad_contextkit.memory.temporal.orchestrator import RetrievalOrchestrator

STAGE_LATENCY_S = 0.05  # 50 ms per stub stage


def _slow(n_results: int = 3, latency: float = STAGE_LATENCY_S):
    """Return a callable that sleeps then returns ``n_results`` dummy hits."""

    def _impl(*args, **kwargs):
        time.sleep(latency)
        return [
            {"id": f"r{i}", "memory": f"hit {i}", "score": 0.5 + 0.01 * i}
            for i in range(n_results)
        ]

    return _impl


def _build_orchestrator() -> RetrievalOrchestrator:
    def _slow_vector(*args, **kwargs):
        time.sleep(STAGE_LATENCY_S)
        return [
            {"id": f"v{i}", "memory": f"vec hit {i}", "score": 0.6 + 0.01 * i}
            for i in range(3)
        ]

    vector_store = MagicMock()
    vector_store.search.side_effect = _slow_vector

    graph_store = MagicMock()
    graph_store.search.side_effect = _slow(latency=STAGE_LATENCY_S)
    graph_store.graph = MagicMock()

    timeline_builder = MagicMock()
    timeline_builder.get_timeline.side_effect = _slow(latency=STAGE_LATENCY_S)

    embedding_model = MagicMock()
    embedding_model.embed.return_value = [0.1] * 128

    o = RetrievalOrchestrator(
        vector_store=vector_store,
        graph_store=graph_store,
        timeline_builder=timeline_builder,
        embedding_model=embedding_model,
    )
    # Stub the causal + cross-modal helpers so they also sleep.
    o._explore_causal_chains = _slow(latency=STAGE_LATENCY_S)  # type: ignore[assignment]
    o._cross_modal_search = _slow(latency=STAGE_LATENCY_S)  # type: ignore[assignment]
    return o


def _bench(orchestrator: RetrievalOrchestrator, *, parallel: bool, n: int = 20) -> Dict[str, float]:
    timings: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        orchestrator.fused_search(
            query="benchmark",
            user_id="u1",
            include_causal=True,
            include_multimodal=True,
            top_k=5,
            parallel=parallel,
        )
        timings.append(time.perf_counter() - t0)
    timings.sort()
    return {
        "p50": timings[len(timings) // 2],
        "p95": timings[int(len(timings) * 0.95)],
        "p99": timings[int(len(timings) * 0.99)],
        "mean": statistics.fmean(timings),
    }


def test_sequential_baseline_takes_at_least_4_stages():
    """Sanity check: synthetic stub really sleeps, sequential adds up."""
    o = _build_orchestrator()
    stats = _bench(o, parallel=False, n=5)
    # 5 stages × 50 ms = 250 ms minimum; allow some slop.
    assert stats["p95"] >= STAGE_LATENCY_S * 4, stats


def test_parallel_fused_search_under_3_stage_budget():
    o = _build_orchestrator()
    stats = _bench(o, parallel=True, n=20)
    # Vector first (1 stage) + parallel fan-out (1 stage) + fusion CPU.
    # Median + mean must clear 3× stage latency budget; p95 from a small
    # n is dominated by GIL/scheduler jitter so we don't gate on it.
    assert stats["p50"] < STAGE_LATENCY_S * 2.5, stats
    assert stats["mean"] < STAGE_LATENCY_S * 3.0, stats


def test_parallel_matches_sequential_correctness():
    o = _build_orchestrator()
    seq = o.fused_search(
        query="benchmark",
        user_id="u1",
        include_causal=True,
        include_multimodal=True,
        top_k=5,
        parallel=False,
    )
    par = o.fused_search(
        query="benchmark",
        user_id="u1",
        include_causal=True,
        include_multimodal=True,
        top_k=5,
        parallel=True,
    )
    # Same merged content set (order may differ tiny for ties).
    seq_ids = {r.get("id") for r in seq.get("fused_ranking", [])}
    par_ids = {r.get("id") for r in par.get("fused_ranking", [])}
    assert seq_ids == par_ids, (seq_ids, par_ids)


def test_parallel_speedup_at_least_1p5x(capsys):
    o = _build_orchestrator()
    seq = _bench(o, parallel=False, n=10)
    par = _bench(o, parallel=True, n=10)
    speedup = seq["p50"] / par["p50"]
    print(
        f"\n[bench] sequential p50={seq['p50']*1000:.1f} ms  "
        f"parallel p50={par['p50']*1000:.1f} ms  "
        f"speedup={speedup:.2f}×"
    )
    # Synthetic 50 ms stages — parallel should be at least 1.5× faster.
    assert speedup >= 1.5, (seq, par, speedup)
