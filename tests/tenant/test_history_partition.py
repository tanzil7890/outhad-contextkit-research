"""Phase T4 — history-store tenant partitioning tests."""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from outhad_contextkit.memory.storage import SQLiteManager


def _open(tmp_path) -> SQLiteManager:
    return SQLiteManager(str(tmp_path / "history.db"))


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

def test_tenant_columns_added_on_fresh_db(tmp_path):
    db = _open(tmp_path)
    cur = db.connection.execute("PRAGMA table_info(history)")
    cols = {row[1] for row in cur.fetchall()}
    assert "tenant_id" in cols
    assert "sub_tenant_id" in cols


def test_migration_idempotent(tmp_path):
    path = str(tmp_path / "history.db")
    SQLiteManager(path)
    SQLiteManager(path)
    db = SQLiteManager(path)
    cols = {row[1] for row in db.connection.execute("PRAGMA table_info(history)").fetchall()}
    assert "tenant_id" in cols
    assert "sub_tenant_id" in cols


def test_migration_indexes_created(tmp_path):
    db = _open(tmp_path)
    cur = db.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='history'"
    )
    names = {r[0] for r in cur.fetchall()}
    assert "idx_history_tenant" in names
    assert "idx_history_sub_tenant" in names


def test_migration_handles_legacy_table(tmp_path):
    path = str(tmp_path / "history.db")
    # Build a legacy-shape table that lacks tenant columns AND PPMF cols.
    raw = sqlite3.connect(path)
    raw.execute(
        """
        CREATE TABLE history (
            id           TEXT PRIMARY KEY,
            memory_id    TEXT,
            old_memory   TEXT,
            new_memory   TEXT,
            event        TEXT,
            created_at   DATETIME,
            updated_at   DATETIME,
            is_deleted   INTEGER,
            actor_id     TEXT,
            role         TEXT
        )
        """
    )
    raw.execute(
        "INSERT INTO history (id, memory_id, event) VALUES ('h1', 'm1', 'ADD')"
    )
    raw.commit()
    raw.close()

    db = SQLiteManager(path)
    # Legacy migration kept the row; tenant migration added the columns.
    rows = db.get_history("m1")
    assert len(rows) == 1
    assert rows[0]["tenant_id"] is None
    assert rows[0]["sub_tenant_id"] is None


# ---------------------------------------------------------------------------
# add_history / get_history with tenant scope
# ---------------------------------------------------------------------------

def test_add_history_persists_tenant(tmp_path):
    db = _open(tmp_path)
    db.add_history(
        "m1", None, "hello", "ADD",
        tenant_id="acme", sub_tenant_id="finance",
    )
    rows = db.get_history("m1")
    assert len(rows) == 1
    assert rows[0]["tenant_id"] == "acme"
    assert rows[0]["sub_tenant_id"] == "finance"


def test_get_history_scoped_to_tenant_returns_only_match(tmp_path):
    db = _open(tmp_path)
    db.add_history("m1", None, "v1", "ADD", tenant_id="acme")
    db.add_history("m1", None, "v1", "ADD", tenant_id="beta")
    db.add_history("m1", None, "legacy", "ADD")  # no tenant tag
    rows_acme = db.get_history("m1", tenant_id="acme")
    # Includes acme + legacy (NULL-tolerant), excludes beta.
    tenant_ids = [r["tenant_id"] for r in rows_acme]
    assert "acme" in tenant_ids
    assert None in tenant_ids
    assert "beta" not in tenant_ids


def test_get_history_default_returns_all(tmp_path):
    """Legacy callers (no tenant arg) get every row regardless of tag."""
    db = _open(tmp_path)
    db.add_history("m1", None, "a", "ADD", tenant_id="acme")
    db.add_history("m1", None, "b", "ADD", tenant_id="beta")
    rows = db.get_history("m1")
    assert len(rows) == 2


def test_get_history_sub_tenant_scope(tmp_path):
    db = _open(tmp_path)
    db.add_history(
        "m1", None, "fin", "ADD", tenant_id="acme", sub_tenant_id="finance"
    )
    db.add_history(
        "m1", None, "eng", "ADD", tenant_id="acme", sub_tenant_id="engineering"
    )
    rows = db.get_history("m1", tenant_id="acme", sub_tenant_id="finance")
    sub_tenants = {r["sub_tenant_id"] for r in rows}
    assert "finance" in sub_tenants
    assert "engineering" not in sub_tenants


def test_get_history_wrong_tenant_returns_empty(tmp_path):
    db = _open(tmp_path)
    db.add_history("m1", None, "v", "ADD", tenant_id="acme")
    rows = db.get_history("m1", tenant_id="ghost")
    # Only the acme row exists, but legacy NULL fallback doesn't match
    # since tenant_id='acme' != 'ghost'. Result should be empty.
    assert rows == []


# ---------------------------------------------------------------------------
# Memory.history end-to-end with tenant subsystem
# ---------------------------------------------------------------------------

@patch("outhad_contextkit.memory.main.capture_event")
def test_memory_history_default_path_unchanged(_capture, tmp_path):
    """tenant.enabled=False → Memory.history returns full history."""
    from outhad_contextkit.configs.base import MemoryConfig
    from outhad_contextkit.memory.main import Memory

    obj = object.__new__(Memory)
    obj.config = MemoryConfig()
    obj.collection_name = "ck"
    obj.db = _open(tmp_path)
    obj.db.add_history("m1", None, "a", "ADD", tenant_id="acme")
    obj.db.add_history("m1", None, "b", "ADD", tenant_id="beta")
    Memory._init_tenant(obj)
    rows = Memory.history(obj, "m1")
    assert len(rows) == 2


@patch("outhad_contextkit.memory.main.capture_event")
def test_memory_history_scoped_when_tenant_supplied(_capture, tmp_path):
    from outhad_contextkit.configs.base import MemoryConfig
    from outhad_contextkit.memory.main import Memory

    obj = object.__new__(Memory)
    obj.config = MemoryConfig()
    obj.config.tenant.enabled = True
    obj.config.tenant.registry.sqlite_path = str(tmp_path / "registry.db")
    obj.collection_name = "ck"
    obj.db = SQLiteManager(str(tmp_path / "history.db"))
    obj.db.add_history("m1", None, "a", "ADD", tenant_id="acme")
    obj.db.add_history("m1", None, "b", "ADD", tenant_id="beta")
    Memory._init_tenant(obj)
    rows = Memory.history(obj, "m1", tenant_id="acme")
    assert {r["tenant_id"] for r in rows} == {"acme"}
