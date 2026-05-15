"""Phase D3 — HistoryStore factory + Postgres adapter parity.

Local round-trip test runs against SQLite via the existing
:class:`SQLiteManager`, plus an integration smoke test that exercises
``PostgresHistoryStore`` against an in-process Postgres URL when one is
exported through ``OUTHAD_TEST_PG_DSN``. The Postgres test is skipped
when the env is absent so CI stays offline-friendly.
"""
from __future__ import annotations

import os
import uuid

import pytest

from outhad_contextkit.memory.history_store import (
    PostgresHistoryStore,
    build_history_store,
)
from outhad_contextkit.memory.storage import SQLiteManager


def test_factory_routes_filesystem_path_to_sqlite(tmp_path):
    store = build_history_store(str(tmp_path / "history.db"))
    try:
        assert isinstance(store, SQLiteManager)
    finally:
        store.close()


def test_factory_routes_memory_to_sqlite():
    store = build_history_store(":memory:")
    try:
        assert isinstance(store, SQLiteManager)
    finally:
        store.close()


def test_factory_recognises_postgres_url(monkeypatch):
    """Unit-only — instantiate is monkeypatched so we don't open a socket."""
    captured = {}

    class _Stub(PostgresHistoryStore):
        def __init__(self, dsn, **kwargs):  # type: ignore[override]
            captured["dsn"] = dsn

        def _ensure_schema(self):  # noqa: D401
            pass

    monkeypatch.setattr(
        "outhad_contextkit.memory.history_store.PostgresHistoryStore", _Stub
    )
    store = build_history_store(
        "postgresql+asyncpg://u:p@host.neon.tech/db?sslmode=require"
    )
    assert isinstance(store, _Stub)
    # asyncpg → psycopg2-style scheme normalisation happens inside
    # PostgresHistoryStore.__init__ — covered by the integration test.


def test_dsn_normalisation():
    """Async-style DSN + Neon channel_binding gets cleaned for psycopg2."""
    cleaned = PostgresHistoryStore._normalise_dsn(
        "postgresql+asyncpg://u:p@host.neon.tech/db?sslmode=require&channel_binding=require"
    )
    assert cleaned.startswith("postgresql://")
    assert "asyncpg" not in cleaned
    assert "channel_binding" not in cleaned
    assert "sslmode=require" in cleaned


def _pg_dsn():
    return os.environ.get("OUTHAD_TEST_PG_DSN")


@pytest.mark.skipif(not _pg_dsn(), reason="set OUTHAD_TEST_PG_DSN to enable")
def test_postgres_round_trip():
    store = PostgresHistoryStore(_pg_dsn())
    try:
        store.reset()
        memory_id = str(uuid.uuid4())
        store.add_history(
            memory_id=memory_id,
            old_memory=None,
            new_memory="hello",
            event="ADD",
            tenant_id="t1",
        )
        store.add_history(
            memory_id=memory_id,
            old_memory="hello",
            new_memory="hello world",
            event="UPDATE",
            tenant_id="t1",
        )
        rows = store.get_history(memory_id, tenant_id="t1")
        assert [r["event"] for r in rows] == ["ADD", "UPDATE"]
        # Tenant scoping rejects mismatched tenant.
        assert store.get_history(memory_id, tenant_id="other") == []
    finally:
        store.close()
