""" RolePolicy ABC + bundled policies + apply_role_policy.

Covers:
* TenantIsolationPolicy: same-tenant pass, cross-tenant deny, global visibility,
  treat_global_as_visible=False denies missing tenant.
* RoleScopePolicy: explicit role match, allowed_roles list, missing-role allow.
* CompositePolicy: AND semantics, first-deny wins.
* AllowAllPolicy: pass-through.
* apply_role_policy: hard filter drops denied; soft filter zeros score; reweight
  multiplies score; never mutates input.
* set_role_policy / get_role_policy: type validation, None disables filtering.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from outhad_contextkit.memory.personalized.role import (
    AllowAllPolicy,
    CompositePolicy,
    RolePolicy,
    RoleScopePolicy,
    TenantIsolationPolicy,
    apply_role_policy,
)
from outhad_contextkit.memory.personalized.types import RoleContext


# ---------------------------------------------------------------------------
# AllowAllPolicy
# ---------------------------------------------------------------------------

def test_allow_all_policy_always_true():
    p = AllowAllPolicy()
    assert p.allow(memory_payload={}, role_ctx=RoleContext()) is True
    assert p.allow(memory_payload={"tenant_id": "x"}, role_ctx=RoleContext(tenant_id="y")) is True


# ---------------------------------------------------------------------------
# TenantIsolationPolicy
# ---------------------------------------------------------------------------

def test_tenant_isolation_same_tenant_allowed():
    p = TenantIsolationPolicy()
    payload = {"id": "m1", "tenant_id": "acme"}
    assert p.allow(memory_payload=payload, role_ctx=RoleContext(tenant_id="acme"))


def test_tenant_isolation_cross_tenant_denied():
    p = TenantIsolationPolicy()
    payload = {"id": "m1", "tenant_id": "acme"}
    assert not p.allow(memory_payload=payload, role_ctx=RoleContext(tenant_id="other"))


def test_tenant_isolation_global_memory_visible_by_default():
    p = TenantIsolationPolicy()
    payload = {"id": "m1"}  # no tenant_id
    assert p.allow(memory_payload=payload, role_ctx=RoleContext(tenant_id="acme"))


def test_tenant_isolation_strict_mode_denies_global():
    p = TenantIsolationPolicy(treat_global_as_visible=False)
    payload = {"id": "m1"}
    assert not p.allow(memory_payload=payload, role_ctx=RoleContext(tenant_id="acme"))


def test_tenant_isolation_reads_nested_metadata():
    """vector-store payloads sometimes carry tenant_id under metadata."""
    p = TenantIsolationPolicy()
    payload = {"id": "m1", "metadata": {"tenant_id": "acme"}}
    assert p.allow(memory_payload=payload, role_ctx=RoleContext(tenant_id="acme"))
    assert not p.allow(
        memory_payload=payload, role_ctx=RoleContext(tenant_id="other")
    )


# ---------------------------------------------------------------------------
# RoleScopePolicy
# ---------------------------------------------------------------------------

def test_role_scope_explicit_role_match():
    p = RoleScopePolicy()
    payload = {"id": "m1", "role": "admin"}
    assert p.allow(memory_payload=payload, role_ctx=RoleContext(role="admin"))
    assert not p.allow(memory_payload=payload, role_ctx=RoleContext(role="member"))


def test_role_scope_allowed_roles_list():
    p = RoleScopePolicy()
    payload = {"id": "m1", "allowed_roles": ["admin", "owner"]}
    assert p.allow(memory_payload=payload, role_ctx=RoleContext(role="admin"))
    assert p.allow(memory_payload=payload, role_ctx=RoleContext(role="owner"))
    assert not p.allow(memory_payload=payload, role_ctx=RoleContext(role="guest"))


def test_role_scope_unrestricted_allowed_by_default():
    p = RoleScopePolicy()
    payload = {"id": "m1"}
    assert p.allow(memory_payload=payload, role_ctx=RoleContext(role="anyone"))


def test_role_scope_strict_mode_denies_unrestricted():
    p = RoleScopePolicy(treat_unrestricted_as_visible=False)
    payload = {"id": "m1"}
    assert not p.allow(memory_payload=payload, role_ctx=RoleContext(role="anyone"))


# ---------------------------------------------------------------------------
# CompositePolicy
# ---------------------------------------------------------------------------

def test_composite_first_deny_wins():
    p = CompositePolicy([TenantIsolationPolicy(), RoleScopePolicy()])
    payload = {"id": "m1", "tenant_id": "acme", "role": "admin"}
    assert p.allow(memory_payload=payload, role_ctx=RoleContext(tenant_id="acme", role="admin"))
    # Cross-tenant — first policy denies even though role matches.
    assert not p.allow(
        memory_payload=payload,
        role_ctx=RoleContext(tenant_id="other", role="admin"),
    )


def test_composite_requires_at_least_one_child():
    with pytest.raises(ValueError):
        CompositePolicy([])


def test_composite_reweight_multiplies():
    class _HalfWeight(RolePolicy):
        def allow(self, **_):
            return True

        def reweight(self, **_):
            return 0.5

    p = CompositePolicy([_HalfWeight(), _HalfWeight()])
    weight = p.reweight(memory_payload={}, role_ctx=RoleContext())
    assert weight == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# apply_role_policy
# ---------------------------------------------------------------------------

def test_apply_role_policy_hard_filter_drops_denied():
    p = TenantIsolationPolicy()
    candidates = [
        {"id": "m1", "tenant_id": "acme", "score": 0.9},
        {"id": "m2", "tenant_id": "other", "score": 0.8},
        {"id": "m3", "tenant_id": "acme", "score": 0.7},
    ]
    out = apply_role_policy(
        candidates,
        policy=p,
        role_ctx=RoleContext(tenant_id="acme"),
        hard_filter=True,
    )
    assert [r["id"] for r in out] == ["m1", "m3"]


def test_apply_role_policy_soft_filter_zeros_denied_score():
    p = TenantIsolationPolicy()
    candidates = [
        {"id": "m1", "tenant_id": "acme", "score": 0.9},
        {"id": "m2", "tenant_id": "other", "score": 0.8},
    ]
    out = apply_role_policy(
        candidates,
        policy=p,
        role_ctx=RoleContext(tenant_id="acme"),
        hard_filter=False,
    )
    by_id = {r["id"]: r for r in out}
    assert by_id["m1"]["score"] == pytest.approx(0.9)
    assert by_id["m2"]["score"] == 0.0


def test_apply_role_policy_does_not_mutate_input():
    p = TenantIsolationPolicy()
    candidates = [{"id": "m1", "tenant_id": "other", "score": 0.5}]
    apply_role_policy(
        candidates, policy=p, role_ctx=RoleContext(tenant_id="acme"), hard_filter=False
    )
    # Original entry untouched.
    assert candidates[0]["score"] == 0.5


def test_apply_role_policy_empty_candidates_returns_empty_list():
    out = apply_role_policy(
        [],
        policy=AllowAllPolicy(),
        role_ctx=RoleContext(),
        hard_filter=True,
    )
    assert out == []


# ---------------------------------------------------------------------------
# set_role_policy / get_role_policy on Memory shell
# ---------------------------------------------------------------------------

def test_set_role_policy_rejects_non_policy():
    from outhad_contextkit.memory.main import Memory

    obj = object.__new__(Memory)
    obj._role_policy = None
    with pytest.raises(TypeError):
        Memory.set_role_policy(obj, "not a policy")  # type: ignore[arg-type]


def test_set_role_policy_accepts_policy_and_none():
    from outhad_contextkit.memory.main import Memory

    obj = object.__new__(Memory)
    obj._role_policy = None
    p = TenantIsolationPolicy()
    Memory.set_role_policy(obj, p)
    assert Memory.get_role_policy(obj) is p
    Memory.set_role_policy(obj, None)
    assert Memory.get_role_policy(obj) is None
