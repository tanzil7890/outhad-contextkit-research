"""Phase F6 — Role-aware retrieval policy hooks.

A ``RolePolicy`` is a pluggable callable that decides whether a memory may
be returned to a given ``RoleContext``, and optionally re-weights the
final score. Policies are *not* part of Pydantic config (they are
callables / objects, not JSON-serialisable), so they are injected via
``Memory.set_role_policy(policy)``.

Two filter modes are supported:

* **Hard filter** — ``allow()`` returns False → memory is dropped from
  the candidate set *before* re-ranking. Cheapest path.
* **Soft filter** — ``reweight()`` returns a multiplier in ``[0, 1]``
  that scales the final retrieval score. Used when the operator wants
  the memory still visible but de-prioritised.

Both helpers are no-ops by default so a custom policy can opt into
either or both behaviours.

Bundled policies:

* ``AllowAllPolicy`` — explicit no-op; useful for tests.
* ``TenantIsolationPolicy`` — denies cross-tenant reads. Reads
  ``payload['tenant_id']`` (configurable).
* ``RoleScopePolicy`` — denies memories whose ``role`` does not appear
  in the requester's allowed-roles set.
* ``CompositePolicy`` — AND-combines multiple policies; first deny wins.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

from outhad_contextkit.memory.personalized.types import RoleContext

logger = logging.getLogger(__name__)


class RolePolicy(ABC):
    """Base class for all role / tenant gates.

    Subclasses must implement ``allow``; ``reweight`` is optional and
    defaults to a no-op multiplier of ``1.0``.
    """

    @abstractmethod
    def allow(
        self,
        *,
        memory_payload: Dict[str, Any],
        role_ctx: RoleContext,
    ) -> bool:
        """Return True if ``memory_payload`` is visible to ``role_ctx``."""

    def reweight(
        self,
        *,
        memory_payload: Dict[str, Any],
        role_ctx: RoleContext,
    ) -> float:
        """Multiplier applied to the final score when ``hard_filter`` is off.

        Default is 1.0 (no change). Implementations should return values
        in ``[0, 1]``; the caller clamps just in case.
        """
        return 1.0


class AllowAllPolicy(RolePolicy):
    """Pass-through policy. Useful as a sentinel and in tests."""

    def allow(self, **_: Any) -> bool:  # noqa: D401 - simple boolean
        return True


class TenantIsolationPolicy(RolePolicy):
    """Deny cross-tenant reads.

    A memory with no ``tenant_id`` is treated as global (visible to every
    tenant). When a tenant is set on the memory it must equal
    ``role_ctx.tenant_id`` for the read to succeed.

    Args:
        tenant_field: The metadata key holding the tenant identifier.
                      Defaults to ``"tenant_id"``.
        treat_global_as_visible: When False, memories without a
                                 ``tenant_id`` are also denied. Defaults
                                 to True (industry standard).
    """

    def __init__(
        self,
        *,
        tenant_field: str = "tenant_id",
        treat_global_as_visible: bool = True,
    ) -> None:
        self.tenant_field = tenant_field
        self.treat_global_as_visible = bool(treat_global_as_visible)

    def allow(
        self,
        *,
        memory_payload: Dict[str, Any],
        role_ctx: RoleContext,
    ) -> bool:
        memory_tenant = _read_field(memory_payload, self.tenant_field)
        if memory_tenant is None:
            return self.treat_global_as_visible
        return memory_tenant == role_ctx.tenant_id


class RoleScopePolicy(RolePolicy):
    """Restrict reads by ``role`` membership.

    Each memory may store a ``role`` (string) or ``allowed_roles``
    (list/set) in its metadata. The requester is allowed when their own
    ``role_ctx.role`` is in that set, or when the memory has no
    role restriction at all.

    Args:
        role_field: Metadata key holding the role attribute (default
                    ``"role"``).
        allowed_roles_field: Metadata key holding the allow-list (default
                             ``"allowed_roles"``).
        treat_unrestricted_as_visible: When False, memories without any
                                       role attribute are denied.
    """

    def __init__(
        self,
        *,
        role_field: str = "role",
        allowed_roles_field: str = "allowed_roles",
        treat_unrestricted_as_visible: bool = True,
    ) -> None:
        self.role_field = role_field
        self.allowed_roles_field = allowed_roles_field
        self.treat_unrestricted_as_visible = bool(treat_unrestricted_as_visible)

    def allow(
        self,
        *,
        memory_payload: Dict[str, Any],
        role_ctx: RoleContext,
    ) -> bool:
        allowed = _read_field(memory_payload, self.allowed_roles_field)
        if allowed is not None:
            try:
                allowed_set = {str(r) for r in allowed}
            except TypeError:
                allowed_set = {str(allowed)}
            return role_ctx.role is not None and str(role_ctx.role) in allowed_set
        memory_role = _read_field(memory_payload, self.role_field)
        if memory_role is None:
            return self.treat_unrestricted_as_visible
        return role_ctx.role is not None and str(memory_role) == str(role_ctx.role)


class CompositePolicy(RolePolicy):
    """AND-combine multiple policies. First ``allow=False`` wins.

    ``reweight`` multiplies all child results so any sub-policy can
    de-prioritise a memory.
    """

    def __init__(self, policies: Iterable[RolePolicy]) -> None:
        self._policies: List[RolePolicy] = list(policies)
        if not self._policies:
            raise ValueError("CompositePolicy requires at least one child policy")

    def allow(
        self,
        *,
        memory_payload: Dict[str, Any],
        role_ctx: RoleContext,
    ) -> bool:
        for p in self._policies:
            try:
                if not p.allow(memory_payload=memory_payload, role_ctx=role_ctx):
                    return False
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("RolePolicy.allow raised: %s — defaulting to deny", exc)
                return False
        return True

    def reweight(
        self,
        *,
        memory_payload: Dict[str, Any],
        role_ctx: RoleContext,
    ) -> float:
        product = 1.0
        for p in self._policies:
            try:
                product *= float(
                    p.reweight(memory_payload=memory_payload, role_ctx=role_ctx)
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("RolePolicy.reweight raised: %s — using 1.0", exc)
        return max(0.0, min(1.0, product))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_field(payload: Dict[str, Any], field: str) -> Optional[Any]:
    """Read ``field`` from ``payload`` or its nested ``metadata`` dict.

    Supports both shapes that the vector-store payload can take:
    * ``{"tenant_id": "x", ...}`` — flat
    * ``{"metadata": {"tenant_id": "x"}, ...}`` — nested under metadata
    """
    if not isinstance(payload, dict):
        return None
    if field in payload and payload[field] is not None:
        return payload[field]
    nested = payload.get("metadata")
    if isinstance(nested, dict) and field in nested and nested[field] is not None:
        return nested[field]
    return None


def apply_role_policy(
    candidates: List[Dict[str, Any]],
    *,
    policy: RolePolicy,
    role_ctx: RoleContext,
    hard_filter: bool,
) -> List[Dict[str, Any]]:
    """Apply ``policy`` to ``candidates`` in-place semantics.

    * ``hard_filter=True``  → drop denied entries.
    * ``hard_filter=False`` → keep all entries but multiply their
      ``score`` by ``policy.reweight``. Denied entries get score 0.0.

    Returns a *new* list; the input is never mutated.
    """
    if not candidates:
        return list(candidates)
    out: List[Dict[str, Any]] = []
    for entry in candidates:
        try:
            allowed = policy.allow(memory_payload=entry, role_ctx=role_ctx)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("RolePolicy.allow raised: %s — defaulting to deny", exc)
            allowed = False
        if hard_filter:
            if allowed:
                out.append(entry)
            continue
        # Soft filter: copy entry and adjust score.
        adj = dict(entry)
        if not allowed:
            adj["score"] = 0.0
        else:
            try:
                weight = float(
                    policy.reweight(memory_payload=entry, role_ctx=role_ctx)
                )
            except Exception:  # pragma: no cover - defensive
                weight = 1.0
            weight = max(0.0, min(1.0, weight))
            if "score" in adj:
                adj["score"] = float(adj["score"]) * weight
        out.append(adj)
    return out


__all__ = [
    "RolePolicy",
    "AllowAllPolicy",
    "TenantIsolationPolicy",
    "RoleScopePolicy",
    "CompositePolicy",
    "apply_role_policy",
]
