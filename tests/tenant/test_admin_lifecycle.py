""" TenantAdmin facade, lifecycle, backfill / export.

Covers:
* Memory.tenant is None when feature flag is off; facade exists when on.
* TenantAdmin delegates to registry (tenant + sub-tenant + role bindings).
* Soft delete archives CGL nodes; status flipped to "deleted".
* Hard delete cascades through history + CGL nodes (returns counts).
* TenantAdmin.create_tenant emits outhad_contextkit.tenant.created.
* backfill_tenant stamps NULL rows + nodes with default tenant.
* export_tenant + import_tenant round-trip via tmp dir.
* migrate_tenant rewrites history + CGL + sub-tenants + role bindings.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from unittest.mock import patch

import pytest

pytest.importorskip("networkx")

from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.context_graph import build_context_graph
from outhad_contextkit.memory.context_graph.config import ContextGraphConfig
from outhad_contextkit.memory.context_graph.types import MemoryNode
from outhad_contextkit.memory.main import Memory
from outhad_contextkit.memory.storage import SQLiteManager
from outhad_contextkit.memory.tenant.admin import TenantAdmin
from outhad_contextkit.memory.tenant.migration import (
    backfill_tenant,
    export_tenant,
    import_tenant,
    migrate_tenant,
)


def _now():
    return datetime(2026, 4, 25, 12, 0, 0)


def _shell(tmp_path, *, tenant_enabled=True):
    """Build a Memory shell with tenant + CGL + history wired."""
    os.makedirs(str(tmp_path), exist_ok=True)
    obj = object.__new__(Memory)
    cfg = MemoryConfig()
    cfg.tenant.enabled = tenant_enabled
    cfg.tenant.registry.sqlite_path = str(tmp_path / "registry.db")
    cfg.context_graph = ContextGraphConfig(
        enabled=True, backend="networkx", log_changes=False
    )
    obj.config = cfg
    obj.collection_name = "ck"
    obj.vector_store = object()
    obj.db = SQLiteManager(str(tmp_path / "history.db"))
    obj._context_graph = build_context_graph(cfg.context_graph)
    obj._context_graph_builder = None
    Memory._init_tenant(obj)
    return obj


# ---------------------------------------------------------------------------
# T6 — TenantAdmin facade + Memory.tenant
# ---------------------------------------------------------------------------

def test_memory_tenant_none_when_disabled(tmp_path):
    obj = _shell(tmp_path, tenant_enabled=False)
    assert obj.tenant is None


def test_memory_tenant_facade_when_enabled(tmp_path):
    obj = _shell(tmp_path)
    assert obj.tenant is not None
    assert isinstance(obj.tenant, TenantAdmin)


def test_admin_create_tenant_round_trip(tmp_path):
    obj = _shell(tmp_path)
    t = obj.tenant.create_tenant(id="acme", name="Acme")
    assert t.id == "acme"
    listed = [t.id for t in obj.tenant.list_tenants()]
    assert "acme" in listed


def test_admin_create_tenant_emits_telemetry(tmp_path):
    obj = _shell(tmp_path)
    with patch(
        "outhad_contextkit.memory.tenant.admin.capture_event"
    ) as spy:
        obj.tenant.create_tenant(id="acme", name="Acme")
    names = [c.args[0] for c in spy.call_args_list]
    assert "outhad_contextkit.tenant.created" in names


def test_admin_assign_revoke_role(tmp_path):
    obj = _shell(tmp_path)
    obj.tenant.create_tenant(id="acme", name="Acme")
    obj.tenant.assign_role(
        tenant_id="acme", principal="alice", role="member"
    )
    assert obj.tenant.has_role(
        principal="alice", tenant_id="acme", role="member"
    )
    assert obj.tenant.revoke_role(
        tenant_id="acme", principal="alice", role="member"
    )
    assert not obj.tenant.has_role(
        principal="alice", tenant_id="acme", role="member"
    )


def test_admin_raises_when_subsystem_disabled(tmp_path):
    obj = _shell(tmp_path, tenant_enabled=False)
    admin = TenantAdmin(obj)
    with pytest.raises(RuntimeError):
        admin.create_tenant(id="acme", name="Acme")


# ---------------------------------------------------------------------------
# T7 — Lifecycle (soft / hard delete)
# ---------------------------------------------------------------------------

def test_soft_delete_archives_cgl_nodes_and_flips_status(tmp_path):
    obj = _shell(tmp_path)
    obj.tenant.create_tenant(id="acme", name="Acme")
    obj._context_graph.upsert_memory_node(
        "m1", "hello", tenant_id="acme"
    )
    obj._context_graph.upsert_memory_node(
        "m2", "other", tenant_id="other"
    )
    counts = obj.tenant.delete_tenant("acme", hard=False)
    assert counts["nodes"] == 1
    # Tenant status now "deleted"; data preserved.
    t = obj.tenant.get_tenant("acme")
    assert t.status == "deleted"
    # The acme node is archived; the other-tenant node is untouched.
    assert obj._context_graph.backend.get_node("m1").archived is True
    assert obj._context_graph.backend.get_node("m2").archived is False


def test_hard_delete_cascades_history_and_cgl(tmp_path):
    obj = _shell(tmp_path)
    obj.tenant.create_tenant(id="acme", name="Acme")
    obj._context_graph.upsert_memory_node(
        "m1", "hello", tenant_id="acme"
    )
    obj.db.add_history(
        "m1", None, "hello", "ADD", tenant_id="acme"
    )
    counts = obj.tenant.delete_tenant("acme", hard=True)
    assert counts["nodes"] == 1
    assert counts["history"] == 1
    assert counts["tenant"] == 1
    # Tenant gone from registry.
    assert obj.tenant.get_tenant("acme") is None
    # CGL node gone.
    assert obj._context_graph.backend.get_node("m1") is None
    # History row gone.
    assert obj.db.get_history("m1", tenant_id="acme") == []


# ---------------------------------------------------------------------------
# T8 — backfill_tenant
# ---------------------------------------------------------------------------

@patch("outhad_contextkit.memory.main.capture_event")
def test_backfill_tenant_stamps_legacy_rows(_capture, tmp_path):
    obj = _shell(tmp_path)
    obj.db.add_history("m1", None, "v", "ADD")  # NULL tenant
    obj.db.add_history("m1", None, "v2", "UPDATE", tenant_id="acme")
    obj._context_graph.upsert_memory_node("m1", "hello")  # NULL tenant
    obj._context_graph.upsert_memory_node(
        "m2", "tagged", tenant_id="acme"
    )
    counts = obj.backfill_tenant()
    assert counts["history"] == 1  # only the NULL row was rewritten
    assert counts["nodes"] == 1   # only m1 (m2 was already tagged)
    # m1 now stamped.
    rows = obj.db.get_history("m1")
    tenants = {r["tenant_id"] for r in rows}
    assert "__default__" in tenants


def test_backfill_tenant_raises_when_disabled(tmp_path):
    obj = _shell(tmp_path, tenant_enabled=False)
    with pytest.raises(RuntimeError):
        backfill_tenant(obj)


# ---------------------------------------------------------------------------
# T7 — export / import / migrate
# ---------------------------------------------------------------------------

def test_export_import_round_trip(tmp_path):
    src = _shell(tmp_path)
    src.tenant.create_tenant(id="acme", name="Acme Corp")
    src.tenant.create_sub_tenant(
        tenant_id="acme", name="finance", kind="department"
    )
    src.tenant.assign_role(
        tenant_id="acme", principal="alice", role="member"
    )
    src._context_graph.upsert_memory_node(
        "m1", "hello", tenant_id="acme"
    )
    src.db.add_history("m1", None, "hello", "ADD", tenant_id="acme")
    dest = tmp_path / "export"
    counts = src.export_tenant("acme", str(dest))
    assert counts["tenant"] == 1
    assert counts["sub_tenants"] == 1
    assert counts["role_bindings"] == 1
    assert counts["history"] == 1
    assert counts["nodes"] == 1
    # Import into a fresh shell.
    dst = _shell(tmp_path / "dest_shell")
    imported = dst.import_tenant(str(dest))
    assert imported["tenant"] == 1
    assert imported["sub_tenants"] == 1
    assert imported["role_bindings"] == 1
    assert imported["history"] == 1
    assert imported["nodes"] == 1
    # Verify the imported tenant is reachable.
    assert dst.tenant.get_tenant("acme") is not None
    assert dst._context_graph.backend.get_node("m1") is not None


def test_migrate_tenant_rewrites_storage_layers(tmp_path):
    obj = _shell(tmp_path)
    obj.tenant.create_tenant(id="src_t", name="Src")
    obj.tenant.create_tenant(id="dst_t", name="Dst")
    obj.tenant.create_sub_tenant(tenant_id="src_t", name="finance")
    obj.tenant.assign_role(
        tenant_id="src_t", principal="alice", role="member"
    )
    obj.db.add_history("m1", None, "hello", "ADD", tenant_id="src_t")
    obj._context_graph.upsert_memory_node(
        "m1", "hello", tenant_id="src_t"
    )
    counts = obj.migrate_tenant("src_t", "dst_t")
    assert counts["history"] == 1
    assert counts["sub_tenants"] == 1
    assert counts["role_bindings"] == 1
    assert counts["nodes"] == 1
    # Confirm the node now points at dst_t.
    assert obj._context_graph.backend.get_node("m1").tenant_id == "dst_t"
