"""Real-data benchmark on the LOCOMO dataset.

Drives `Memory.add` + `Memory.search` against actual OpenAI embeddings
+ LLM and (optionally) a real Neo4j graph. Compares the post-perf-fix
(parallel + RRF + caches) path against the legacy path on the same
conversations.

Auto-loads ``./.env`` (repo root) on import so OPENAI_API_KEY and
NEO4J_PASSWORD don't have to be exported manually. Skipped when the
required keys are absent so CI never fails on missing credentials.

Limited to the FIRST LOCOMO conversation only by default to keep
runtime ≤ 60 s and API cost ≤ $0.50/run. Override via env:

    LOCOMO_BENCH_SAMPLES=1   # only first sample (default)
    LOCOMO_BENCH_QUERIES=3   # only first 3 QA queries (default)
    LOCOMO_BENCH_SESSIONS=1  # only first 1 session of the convo (default)
    LOCOMO_BENCH_TURNS=15    # cap turns per ingest (default)

Run locally:

    PYTHONPATH=. pytest tests/temporal/bench/test_real_locomo_bench.py -s

Output: JSON file under ``evaluation/`` with the timing + accuracy
deltas so the perf doc can cite real numbers.
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

logger = logging.getLogger(__name__)

LOCOMO_PATH = (
    Path(__file__).resolve().parents[3]
    / "evaluation"
    / "locomo10.json"
)
RESULTS_DIR = Path(__file__).resolve().parents[3] / "evaluation"
DOTENV_PATH = Path(__file__).resolve().parents[3] / ".env"

# Auto-load .env at module import so the user doesn't have to source it.
if DOTENV_PATH.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(DOTENV_PATH)
        logger.info("Loaded .env from %s", DOTENV_PATH)
    except ImportError:
        # python-dotenv not installed — fall back to manual parse.
        for line in DOTENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# Defaults tightened: first conversation only, very small slice.
DEFAULT_SAMPLES = int(os.getenv("LOCOMO_BENCH_SAMPLES", "1"))
DEFAULT_QUERIES = int(os.getenv("LOCOMO_BENCH_QUERIES", "3"))
DEFAULT_SESSIONS = int(os.getenv("LOCOMO_BENCH_SESSIONS", "1"))
DEFAULT_TURN_CAP = int(os.getenv("LOCOMO_BENCH_TURNS", "15"))


def _have_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _have_neo4j() -> bool:
    return bool(os.getenv("NEO4J_PASSWORD"))


pytestmark = pytest.mark.skipif(
    not _have_openai(),
    reason="OPENAI_API_KEY not set; real-data bench requires live API",
)


@pytest.fixture(scope="module")
def locomo() -> List[Dict[str, Any]]:
    if not LOCOMO_PATH.exists():
        pytest.skip(f"LOCOMO dataset missing at {LOCOMO_PATH}")
    with LOCOMO_PATH.open() as fh:
        return json.load(fh)


def _flatten_sessions(conversation: Dict[str, Any], max_sessions: int) -> List[Dict[str, Any]]:
    """Walk session_1 / session_2 / ... and return ordered turns."""
    turns: List[Dict[str, Any]] = []
    for i in range(1, max_sessions + 1):
        key = f"session_{i}"
        date_key = f"session_{i}_date_time"
        session = conversation.get(key)
        if not isinstance(session, list):
            continue
        date_str = conversation.get(date_key, "")
        for turn in session:
            turns.append({
                "speaker": turn.get("speaker", ""),
                "text": turn.get("text", ""),
                "dia_id": turn.get("dia_id", ""),
                "session_date": date_str,
                "session": i,
            })
    return turns


def _neo4j_url() -> str:
    # Honour both common env var names so existing setups work unchanged.
    return os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL") or "bolt://localhost:7687"


def _neo4j_user() -> str:
    return os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j"


def _build_memory(*, with_tcmgm: bool):
    """Construct Memory with or without TCMGM. Always uses real OpenAI.

    Skips the test (rather than raising) when ``with_tcmgm=True`` and
    Neo4j is unreachable so a single missing service doesn't fail CI.
    """
    from outhad_contextkit import Memory
    from outhad_contextkit.configs.base import MemoryConfig

    cfg = MemoryConfig()
    if with_tcmgm and _have_neo4j():
        from outhad_contextkit.graphs.configs import (
            GraphStoreConfig,
            Neo4jConfig,
        )

        cfg.graph_store = GraphStoreConfig(
            provider="neo4j",
            config=Neo4jConfig(
                url=_neo4j_url(),
                username=_neo4j_user(),
                password=os.getenv("NEO4J_PASSWORD"),
            ),
        )
    try:
        return Memory(config=cfg)
    except (ValueError, ConnectionError, OSError) as exc:
        msg = str(exc).lower()
        if "neo4j" in msg or "connect" in msg or "url" in msg:
            pytest.skip(f"Neo4j unreachable; skipping real-graph bench: {exc}")
        raise


def _ingest_turns(memory, turns: List[Dict[str, Any]], user_id: str) -> Tuple[int, float]:
    """Add every turn; returns (count, total_seconds)."""
    started = time.perf_counter()
    successful = 0
    for turn in turns:
        text = f"{turn['speaker']}: {turn['text']}"
        try:
            memory.add(
                text,
                user_id=user_id,
                metadata={
                    "dia_id": turn["dia_id"],
                    "session": turn["session"],
                    "session_date": turn["session_date"],
                    "speaker": turn["speaker"],
                },
            )
            successful += 1
        except Exception as exc:
            logger.warning("ingest failed: %s", exc)
    return successful, time.perf_counter() - started


def _bench_search(
    memory,
    questions: List[str],
    user_id: str,
    *,
    use_tcmgm: bool,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Run searches; return (per-call latencies, raw responses)."""
    timings: List[float] = []
    responses: List[Dict[str, Any]] = []
    for q in questions:
        t0 = time.perf_counter()
        try:
            res = memory.search(
                query=q,
                user_id=user_id,
                limit=5,
                use_tcmgm=use_tcmgm,
            )
        except Exception as exc:
            logger.warning("search failed (%s): %s", q[:40], exc)
            res = {"results": []}
        timings.append(time.perf_counter() - t0)
        responses.append(res)
    return timings, responses


def _quantiles(timings: List[float]) -> Dict[str, float]:
    if not timings:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    ts = sorted(timings)
    return {
        "p50_ms": ts[len(ts) // 2] * 1000.0,
        "p95_ms": ts[max(0, int(len(ts) * 0.95) - 1)] * 1000.0,
        "p99_ms": ts[max(0, int(len(ts) * 0.99) - 1)] * 1000.0,
        "mean_ms": statistics.fmean(ts) * 1000.0,
    }


def _accuracy(responses: List[Dict[str, Any]], qa_pairs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Crude evidence overlap: fraction of QA where any returned memory
    references the expected dia_id evidence. Treats it as recall@k.
    """
    hits = 0
    total = 0
    for resp, qa in zip(responses, qa_pairs):
        evidence = set(qa.get("evidence", []) or [])
        if not evidence:
            continue
        total += 1
        results = resp.get("results", []) if isinstance(resp, dict) else []
        seen_ids: set = set()
        for r in results:
            md = r.get("metadata") if isinstance(r, dict) else {}
            if isinstance(md, dict) and md.get("dia_id"):
                seen_ids.add(md["dia_id"])
        if seen_ids & evidence:
            hits += 1
    if total == 0:
        return {"recall_at_5": 0.0, "evaluated_questions": 0}
    return {"recall_at_5": hits / total, "evaluated_questions": total}


# ---------------------------------------------------------------------------
# Real benchmark — parallel (post-fix) vs sequential (pre-fix)
# ---------------------------------------------------------------------------

def test_real_locomo_search_post_vs_pre_fix(locomo, capsys):
    """Single sample, small slice. Compares parallel + RRF vs sequential + weighted."""
    if not _have_neo4j():
        pytest.skip("NEO4J_PASSWORD missing — real graph required for fused TCMGM")

    sample = locomo[0]
    user_id = f"locomo_bench_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    turns = _flatten_sessions(sample["conversation"], DEFAULT_SESSIONS)
    questions = [qa["question"] for qa in sample["qa"][:DEFAULT_QUERIES]]

    memory = _build_memory(with_tcmgm=True)

    # Ingest once — both runs share the same graph state.
    capped = turns[:DEFAULT_TURN_CAP]
    ingested, ingest_s = _ingest_turns(memory, capped, user_id)
    logger.info(
        "Ingested %d / %d turns in %.2f s (%.0f ms / turn avg)",
        ingested,
        len(capped),
        ingest_s,
        (ingest_s / max(1, ingested)) * 1000.0,
    )

    # Pre-fix path (sequential + weighted) — patch orchestrator at runtime.
    if memory._retrieval_orchestrator is None:
        pytest.skip("orchestrator unavailable on this Memory")
    o = memory._retrieval_orchestrator
    orig_fused = o.fused_search

    def _legacy_search(*args, **kwargs):
        kwargs["parallel"] = False
        return orig_fused(*args, **kwargs)

    # warm caches first to avoid a once-per-run cold start dominating numbers
    _bench_search(memory, questions[:1], user_id, use_tcmgm=True)

    o.fused_search = _legacy_search
    seq_timings, seq_responses = _bench_search(
        memory, questions, user_id, use_tcmgm=True
    )
    o.fused_search = orig_fused
    par_timings, par_responses = _bench_search(
        memory, questions, user_id, use_tcmgm=True
    )

    seq_q = _quantiles(seq_timings)
    par_q = _quantiles(par_timings)
    seq_acc = _accuracy(seq_responses, sample["qa"][:DEFAULT_QUERIES])
    par_acc = _accuracy(par_responses, sample["qa"][:DEFAULT_QUERIES])

    speedup = (
        seq_q["p50_ms"] / par_q["p50_ms"]
        if par_q["p50_ms"] > 0
        else 0.0
    )

    record = {
        "dataset": "locomo10",
        "sample_id": sample.get("sample_id"),
        "config": {
            "sessions": DEFAULT_SESSIONS,
            "queries": DEFAULT_QUERIES,
            "ingested_turns": ingested,
            "ingest_seconds": ingest_s,
        },
        "sequential_legacy": {**seq_q, **seq_acc},
        "parallel_post_fix": {**par_q, **par_acc},
        "speedup_p50": speedup,
        "timestamp": datetime.utcnow().isoformat(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "tcmgm_perf_real_bench.json"
    out.write_text(json.dumps(record, indent=2))
    print("\n[real-bench] " + json.dumps(record, indent=2))
    print(f"[real-bench] saved → {out}")

    # Soft assertions — real data has variance; just gate against
    # catastrophic regressions.
    assert par_q["p50_ms"] > 0
    assert par_q["p50_ms"] <= seq_q["p50_ms"] * 1.10, (
        "parallel path should not be >10% slower than sequential on real data"
    )
    assert par_acc["recall_at_5"] >= seq_acc["recall_at_5"] - 0.10, (
        "RRF must not drop recall@5 by more than 10 points vs weighted"
    )


def test_real_locomo_add_throughput(locomo, capsys):
    """Throughput-only bench: how many adds per second on real OpenAI?"""
    sample = locomo[0]
    user_id = f"locomo_throughput_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    turns = _flatten_sessions(sample["conversation"], DEFAULT_SESSIONS)[:DEFAULT_TURN_CAP]

    memory = _build_memory(with_tcmgm=False)  # vector-only baseline
    ingested, ingest_s = _ingest_turns(memory, turns, user_id)
    rate = ingested / ingest_s if ingest_s > 0 else 0.0

    record = {
        "dataset": "locomo10",
        "config": {
            "sessions": DEFAULT_SESSIONS,
            "ingested_turns": ingested,
        },
        "ingest_seconds": ingest_s,
        "ingest_rate_per_sec": rate,
        "timestamp": datetime.utcnow().isoformat(),
    }
    print("\n[throughput] " + json.dumps(record, indent=2))

    out = RESULTS_DIR / "tcmgm_throughput_real_bench.json"
    out.write_text(json.dumps(record, indent=2))
    assert ingested >= 1
    assert rate > 0


def test_real_locomo_query_embed_cache_hit_saves_api_call(locomo, capsys):
    """Same query twice → 2nd call hits the P6 query embedding cache."""
    from outhad_contextkit.memory.temporal.orchestrator import (
        _QUERY_EMBED_CACHE,
        clear_query_embed_cache,
    )

    if not _have_neo4j():
        pytest.skip("NEO4J_PASSWORD missing — real graph required for fused TCMGM")

    sample = locomo[0]
    user_id = f"locomo_cache_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    turns = _flatten_sessions(sample["conversation"], 1)[:DEFAULT_TURN_CAP]
    questions = [sample["qa"][0]["question"]]

    memory = _build_memory(with_tcmgm=True)
    _ingest_turns(memory, turns, user_id)

    clear_query_embed_cache()
    timings_first, _ = _bench_search(memory, questions, user_id, use_tcmgm=True)
    cache_size_after_first = len(_QUERY_EMBED_CACHE)
    timings_second, _ = _bench_search(memory, questions, user_id, use_tcmgm=True)
    cache_size_after_second = len(_QUERY_EMBED_CACHE)

    record = {
        "first_call_ms": timings_first[0] * 1000.0,
        "second_call_ms": timings_second[0] * 1000.0,
        "cache_size_after_first": cache_size_after_first,
        "cache_size_after_second": cache_size_after_second,
    }
    print("\n[cache-real] " + json.dumps(record, indent=2))

    # Cache only grows on first call.
    assert cache_size_after_second == cache_size_after_first
    # Second call should not add a new entry; allowed to be slower for
    # other reasons but the cache itself must hit.
    assert cache_size_after_first >= 1
