""" TenantRegistry SQLite integration."""
from __future__ import annotations

import pytest

from outhad_contextkit.memory.tenant.registry import TenantRegistry
from outhad_contextkit.memory.tenant.types import SubTenant


def _open(tmp_path, *, cache_size: int = 16) -> TenantRegistry:
    return TenantRegistry(str(tmp_path / "registry.db"), cache_size=cache_size)


# ---------------------------------------------------------------------------
# bootstrap / tenants
# ---------------------------------------------------------------------------

def test_schema_bootstrap_idempotent(tmp_path):
    path = str(tmp_path / "registry.db")
    TenantRegistry(path)
    TenantRegistry(path)
    reopened = TenantRegistry(path)
    assert reopened.list_tenants() == []


def test_create_tenant_round_trip(tmp_path):
    r = _open(tmp_path)
    t = r.create_tenant(id="acme", name="Acme Corp", metadata={"region": "us"})
    assert t.id == "acme"
    assert t.metadata == {"region": "us"}
    fetched = r.get_tenant("acme")
    assert fetched is not None
    assert fetched.name == "Acme Corp"


def test_create_tenant_rejects_default_sentinel(tmp_path):
    r = _open(tmp_path)
    with pytest.raises(ValueError):
        r.create_tenant(id="__default__", name="x")


def test_create_tenant_duplicate_raises(tmp_path):
    r = _open(tmp_path)
    r.create_tenant(id="acme", name="Acme")
    with pytest.raises(ValueError):
        r.create_tenant(id="acme", name="Acme 2")


def test_list_tenants_filters_by_status(tmp_path):
    r = _open(tmp_path)
    r.create_tenant(id="t1", name="Active1")
    r.create_tenant(id="t2", name="Active2")
    r.update_tenant("t2", status="suspended")
    active = [t.id for t in r.list_tenants(status="active")]
    suspended = [t.id for t in r.list_tenants(status="suspended")]
    assert "t1" in active and "t2" not in active
    assert suspended == ["t2"]


def test_update_tenant_persists_metadata(tmp_path):
    r = _open(tmp_path)
    r.create_tenant(id="acme", name="Acme")
    r.update_tenant("acme", metadata={"plan": "enterprise"})
    assert r.get_tenant("acme").metadata == {"plan": "enterprise"}


def test_delete_tenant_cascades(tmp_path):
    r = _open(tmp_path)
    r.create_tenant(id="acme", name="Acme")
    st = r.create_sub_tenant(tenant_id="acme", name="finance", kind="department")
    r.assign_role(tenant_id="acme", principal="alice", role="member")
    r.assign_role(
        tenant_id="acme", sub_tenant_id=st.id, principal="bob", role="viewer"
    )
    assert r.delete_tenant("acme") is True
    # Cascades — children gone.
    assert r.get_tenant("acme") is None
    assert r.list_sub_tenants("acme") == []
    assert r.list_roles_for("alice") == []
    assert r.list_roles_for("bob") == []


# ---------------------------------------------------------------------------
# sub-tenants
# ---------------------------------------------------------------------------

def test_create_sub_tenant_idempotent(tmp_path):
    r = _open(tmp_path)
    r.create_tenant(id="acme", name="Acme")
    a = r.create_sub_tenant(tenant_id="acme", name="finance", kind="department")
    b = r.create_sub_tenant(tenant_id="acme", name="finance", kind="department")
    assert a.id == b.id == SubTenant.derive_id("acme", "finance", "department")
    assert len(r.list_sub_tenants("acme")) == 1


def test_create_sub_tenant_unknown_parent(tmp_path):
    r = _open(tmp_path)
    with pytest.raises(ValueError):
        r.create_sub_tenant(tenant_id="missing", name="x")


def test_delete_sub_tenant_removes_role_bindings(tmp_path):
    r = _open(tmp_path)
    r.create_tenant(id="acme", name="Acme")
    st = r.create_sub_tenant(tenant_id="acme", name="finance")
    r.assign_role(
        tenant_id="acme", sub_tenant_id=st.id, principal="alice", role="member"
    )
    assert r.delete_sub_tenant(st.id) is True
    bindings = r.list_roles_for("alice")
    # tenant-scoped binding doesn't exist; only sub-tenant binding existed → gone.
    assert bindings == []


# ---------------------------------------------------------------------------
# role bindings
# ---------------------------------------------------------------------------

def test_assign_role_idempotent(tmp_path):
    r = _open(tmp_path)
    r.create_tenant(id="acme", name="Acme")
    r.assign_role(tenant_id="acme", principal="alice", role="member")
    r.assign_role(tenant_id="acme", principal="alice", role="member")
    assert len(r.list_roles_for("alice")) == 1


def test_assign_role_unknown_tenant(tmp_path):
    r = _open(tmp_path)
    with pytest.raises(ValueError):
        r.assign_role(tenant_id="ghost", principal="alice", role="member")


def test_revoke_role(tmp_path):
    r = _open(tmp_path)
    r.create_tenant(id="acme", name="Acme")
    r.assign_role(tenant_id="acme", principal="alice", role="member")
    assert r.revoke_role(tenant_id="acme", principal="alice", role="member") is True
    assert r.list_roles_for("alice") == []


def test_has_role_tenant_scope(tmp_path):
    r = _open(tmp_path)
    r.create_tenant(id="acme", name="Acme")
    r.assign_role(tenant_id="acme", principal="alice", role="member")
    assert r.has_role(principal="alice", tenant_id="acme", role="member")
    assert not r.has_role(principal="alice", tenant_id="acme", role="admin")
    assert not r.has_role(principal="bob", tenant_id="acme", role="member")


def test_has_role_sub_tenant_scope_inherits_tenant_grant(tmp_path):
    r = _open(tmp_path)
    r.create_tenant(id="acme", name="Acme")
    st = r.create_sub_tenant(tenant_id="acme", name="finance")
    r.assign_role(tenant_id="acme", principal="alice", role="admin")
    # Tenant-scoped admin should be visible inside any sub-tenant scope.
    assert r.has_role(
        principal="alice", tenant_id="acme", role="admin", sub_tenant_id=st.id
    )


def test_has_role_any(tmp_path):
    """role=None checks for any binding (membership probe)."""
    r = _open(tmp_path)
    r.create_tenant(id="acme", name="Acme")
    r.assign_role(tenant_id="acme", principal="alice", role="viewer")
    assert r.has_role(principal="alice", tenant_id="acme", role=None)
    assert not r.has_role(principal="bob", tenant_id="acme", role=None)


def test_cache_returns_cached_tenant(tmp_path):
    r = _open(tmp_path, cache_size=4)
    r.create_tenant(id="acme", name="Acme")
    a = r.get_tenant("acme")
    b = r.get_tenant("acme")
    assert a is b  # cached object reuse


def test_cache_disabled_when_size_zero(tmp_path):
    r = _open(tmp_path, cache_size=0)
    r.create_tenant(id="acme", name="Acme")
    a = r.get_tenant("acme")
    b = r.get_tenant("acme")
    # No caching → distinct objects from sql each call.
    assert a is not b


def test_durability_across_instances(tmp_path):
    path = str(tmp_path / "registry.db")
    r1 = TenantRegistry(path)
    r1.create_tenant(id="acme", name="Acme")
    r1.assign_role(tenant_id="acme", principal="alice", role="member")
    r2 = TenantRegistry(path)
    assert r2.get_tenant("acme") is not None
    assert r2.has_role(principal="alice", tenant_id="acme", role="member")
