""" tenant backfill / export / import / migrate helpers.

After upgrading from a pre-tenant build, operators run
:func:`backfill_tenant` once to stamp every legacy memory + CGL node
with the default tenant id. Without it, legacy reads keep working
(NULL → default tenant fallback) but new tenant-scoped queries see
only the rows explicitly tagged.

Export / import / migrate are the offline lifecycle ops that ship as
part of T7.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def backfill_tenant(
    memory: Any,
    *,
    default_tenant_id: str = "__default__",
    batch_size: int = 500,
) -> Dict[str, int]:
    """Stamp every legacy memory + CGL node with ``default_tenant_id``.

    Idempotent. Returns ``{"history": N, "nodes": M, "edges": K}``
    counts of rows actually rewritten (already-tagged rows are
    skipped).
    """
    if not getattr(memory.config, "tenant", None) or not (
        memory.config.tenant.enabled
    ):
        raise RuntimeError(
            "backfill_tenant requires MemoryConfig.tenant.enabled=True"
        )

    counts = {"history": 0, "nodes": 0, "edges": 0}

    # 1) History rows.
    try:
        with memory.db._lock:
            cur = memory.db.connection.execute(
                "UPDATE history SET tenant_id = ? WHERE tenant_id IS NULL",
                (default_tenant_id,),
            )
            counts["history"] = int(cur.rowcount or 0)
    except Exception as exc:
        logger.error("backfill_tenant history update failed: %s", exc)

    # 2) CGL nodes (only when CGL is enabled).
    cg = getattr(memory, "_context_graph", None)
    if cg is not None:
        backend = cg.backend
        try:
            for node in list(backend.iter_nodes(include_archived=True)):
                if getattr(node, "tenant_id", None) is not None:
                    continue
                node.tenant_id = default_tenant_id
                try:
                    backend.upsert_node(node)
                    counts["nodes"] += 1
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "backfill node upsert failed for %s: %s", node.id, exc
                    )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("backfill_tenant CGL iter failed: %s", exc)

    logger.info(
        "backfill_tenant: stamped %d history rows + %d CGL nodes with %r",
        counts["history"],
        counts["nodes"],
        default_tenant_id,
    )
    return counts


def export_tenant(memory: Any, tenant_id: str, dest_dir: str) -> Dict[str, int]:
    """ write every storage row for ``tenant_id`` to ``dest_dir``.

    Output files (all under ``dest_dir``):
    * ``tenant.json``        — registry row.
    * ``sub_tenants.json``   — every child sub-tenant.
    * ``role_bindings.json`` — every role binding.
    * ``history.jsonl``      — one history row per line.
    * ``cgl_nodes.jsonl``    — one CGL node per line (tenant-scoped).
    * ``cgl_edges.jsonl``    — one CGL edge per line (where both endpoints belong to the tenant).

    Vector-store rows are intentionally NOT exported: re-ingestion
    requires re-embedding which is provider-specific. Use the source
    Memory's vector-store admin helper if needed.

    Returns counts dict.
    """
    os.makedirs(dest_dir, exist_ok=True)
    registry = memory._tenant_registry
    if registry is None:
        raise RuntimeError("export_tenant requires the tenant subsystem")
    tenant = registry.get_tenant(tenant_id)
    if tenant is None:
        raise ValueError(f"Unknown tenant {tenant_id!r}")

    counts = {
        "tenant": 1,
        "sub_tenants": 0,
        "role_bindings": 0,
        "history": 0,
        "nodes": 0,
        "edges": 0,
    }

    # 1) tenant.json
    with open(os.path.join(dest_dir, "tenant.json"), "w") as fh:
        json.dump(_dataclass_to_dict(tenant), fh, indent=2, default=str)

    # 2) sub_tenants.json
    sub_tenants = registry.list_sub_tenants(tenant_id)
    with open(os.path.join(dest_dir, "sub_tenants.json"), "w") as fh:
        json.dump(
            [_dataclass_to_dict(s) for s in sub_tenants], fh, indent=2, default=str
        )
    counts["sub_tenants"] = len(sub_tenants)

    # 3) role_bindings.json
    bindings: list = []
    try:
        with registry._lock, registry._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM role_bindings WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchall()
            bindings = [registry._row_to_role(r) for r in rows]
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("export role_bindings failed: %s", exc)
    with open(os.path.join(dest_dir, "role_bindings.json"), "w") as fh:
        json.dump(
            [_dataclass_to_dict(b) for b in bindings], fh, indent=2, default=str
        )
    counts["role_bindings"] = len(bindings)

    # 4) history.jsonl
    try:
        with memory.db._lock:
            rows = memory.db.connection.execute(
                "SELECT id, memory_id, old_memory, new_memory, event, "
                "       created_at, updated_at, is_deleted, actor_id, role, "
                "       sensitive_blob, encryption_metadata, privacy_level, "
                "       tenant_id, sub_tenant_id "
                "FROM history WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchall()
        with open(os.path.join(dest_dir, "history.jsonl"), "w") as fh:
            for r in rows:
                fh.write(
                    json.dumps(
                        {
                            "id": r[0],
                            "memory_id": r[1],
                            "old_memory": r[2],
                            "new_memory": r[3],
                            "event": r[4],
                            "created_at": r[5],
                            "updated_at": r[6],
                            "is_deleted": bool(r[7]),
                            "actor_id": r[8],
                            "role": r[9],
                            "sensitive_blob": r[10],
                            "encryption_metadata": r[11],
                            "privacy_level": r[12],
                            "tenant_id": r[13],
                            "sub_tenant_id": r[14],
                        },
                        default=str,
                    )
                    + "\n"
                )
            counts["history"] = len(rows)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("export history failed: %s", exc)

    # 5) cgl_nodes.jsonl + cgl_edges.jsonl
    cg = getattr(memory, "_context_graph", None)
    if cg is not None:
        backend = cg.backend
        from outhad_contextkit.memory.context_graph.backends import (
            _edge_to_dict,
            _node_to_dict,
        )

        node_ids: set = set()
        with open(os.path.join(dest_dir, "cgl_nodes.jsonl"), "w") as fh:
            for node in backend.iter_nodes(include_archived=True):
                if getattr(node, "tenant_id", None) != tenant_id:
                    continue
                fh.write(json.dumps(_node_to_dict(node), default=str) + "\n")
                node_ids.add(node.id)
                counts["nodes"] += 1
        with open(os.path.join(dest_dir, "cgl_edges.jsonl"), "w") as fh:
            for node_id in list(node_ids):
                try:
                    edges = backend.neighbours(node_id, depth=1, min_weight=0.0)
                except Exception:
                    edges = []
                for edge in edges:
                    if edge.src in node_ids and edge.dst in node_ids:
                        fh.write(
                            json.dumps(_edge_to_dict(edge), default=str) + "\n"
                        )
                        counts["edges"] += 1

    logger.info(
        "export_tenant %s → %s: %s",
        tenant_id,
        dest_dir,
        ", ".join(f"{k}={v}" for k, v in counts.items()),
    )
    return counts


def import_tenant(memory: Any, src_dir: str) -> Dict[str, int]:
    """ reverse of :func:`export_tenant`.

    Re-creates the tenant + sub-tenants + role-bindings + history rows
    + CGL nodes/edges from a previous export. Idempotent: existing
    rows are reused, conflicts skipped with a warning. Vector-store
    rows are NOT restored (see export_tenant note).
    """
    counts = {
        "tenant": 0,
        "sub_tenants": 0,
        "role_bindings": 0,
        "history": 0,
        "nodes": 0,
        "edges": 0,
    }
    registry = memory._tenant_registry
    if registry is None:
        raise RuntimeError("import_tenant requires the tenant subsystem")

    # 1) tenant.json
    with open(os.path.join(src_dir, "tenant.json")) as fh:
        t = json.load(fh)
    try:
        registry.create_tenant(
            id=t["id"], name=t.get("name", t["id"]), metadata=t.get("metadata") or {}
        )
        counts["tenant"] = 1
    except ValueError:
        counts["tenant"] = 0  # already exists

    # 2) sub_tenants.json
    with open(os.path.join(src_dir, "sub_tenants.json")) as fh:
        for s in json.load(fh):
            registry.create_sub_tenant(
                tenant_id=s["tenant_id"],
                name=s["name"],
                kind=s.get("kind", "custom"),
                metadata=s.get("metadata") or {},
            )
            counts["sub_tenants"] += 1

    # 3) role_bindings.json
    with open(os.path.join(src_dir, "role_bindings.json")) as fh:
        for b in json.load(fh):
            registry.assign_role(
                tenant_id=b["tenant_id"],
                principal=b["principal"],
                role=b["role"],
                principal_kind=b.get("principal_kind", "user"),
                sub_tenant_id=b.get("sub_tenant_id"),
                granted_by=b.get("granted_by"),
            )
            counts["role_bindings"] += 1

    # 4) history.jsonl
    history_path = os.path.join(src_dir, "history.jsonl")
    if os.path.exists(history_path):
        with open(history_path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    memory.db.add_history(
                        row["memory_id"],
                        row.get("old_memory"),
                        row.get("new_memory"),
                        row.get("event", "ADD"),
                        created_at=row.get("created_at"),
                        updated_at=row.get("updated_at"),
                        is_deleted=int(bool(row.get("is_deleted", False))),
                        actor_id=row.get("actor_id"),
                        role=row.get("role"),
                        sensitive_blob=row.get("sensitive_blob"),
                        encryption_metadata=row.get("encryption_metadata"),
                        privacy_level=row.get("privacy_level"),
                        tenant_id=row.get("tenant_id"),
                        sub_tenant_id=row.get("sub_tenant_id"),
                    )
                    counts["history"] += 1
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("import history skip: %s", exc)

    # 5) cgl_nodes / cgl_edges
    cg = getattr(memory, "_context_graph", None)
    if cg is not None:
        from outhad_contextkit.memory.context_graph.backends import (
            _dict_to_edge,
            _dict_to_node,
        )

        nodes_path = os.path.join(src_dir, "cgl_nodes.jsonl")
        edges_path = os.path.join(src_dir, "cgl_edges.jsonl")
        if os.path.exists(nodes_path):
            with open(nodes_path) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    node = _dict_to_node(json.loads(line))
                    cg.backend.upsert_node(node)
                    counts["nodes"] += 1
        if os.path.exists(edges_path):
            with open(edges_path) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    edge = _dict_to_edge(json.loads(line))
                    cg.backend.upsert_edge(edge)
                    counts["edges"] += 1

    return counts


def migrate_tenant(memory: Any, src_id: str, dst_id: str) -> Dict[str, int]:
    """ rewrite every storage row from ``src_id`` to ``dst_id``.

    Useful for renames / re-homing. Both ids must exist in the
    registry. Vector-store collection move is left to the caller in
    ``mode='collection'`` (driver-specific). Other layers are
    rewritten in place.
    """
    registry = memory._tenant_registry
    if registry is None:
        raise RuntimeError("migrate_tenant requires the tenant subsystem")
    if registry.get_tenant(src_id) is None:
        raise ValueError(f"Unknown source tenant {src_id!r}")
    if registry.get_tenant(dst_id) is None:
        raise ValueError(f"Unknown destination tenant {dst_id!r}")

    counts = {"history": 0, "sub_tenants": 0, "role_bindings": 0, "nodes": 0}

    # History rows.
    try:
        with memory.db._lock:
            cur = memory.db.connection.execute(
                "UPDATE history SET tenant_id = ? WHERE tenant_id = ?",
                (dst_id, src_id),
            )
            counts["history"] = int(cur.rowcount or 0)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("migrate history failed: %s", exc)

    # Sub-tenants + role bindings via direct SQL.
    try:
        with registry._lock, registry._connect() as conn:
            cur1 = conn.execute(
                "UPDATE sub_tenants SET tenant_id = ? WHERE tenant_id = ?",
                (dst_id, src_id),
            )
            counts["sub_tenants"] = int(cur1.rowcount or 0)
            cur2 = conn.execute(
                "UPDATE role_bindings SET tenant_id = ? WHERE tenant_id = ?",
                (dst_id, src_id),
            )
            counts["role_bindings"] = int(cur2.rowcount or 0)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("migrate registry failed: %s", exc)
    registry._tenant_cache.invalidate()
    registry._role_cache.invalidate()

    # CGL nodes.
    cg = getattr(memory, "_context_graph", None)
    if cg is not None:
        try:
            for node in list(cg.backend.iter_nodes(include_archived=True)):
                if getattr(node, "tenant_id", None) != src_id:
                    continue
                node.tenant_id = dst_id
                cg.backend.upsert_node(node)
                counts["nodes"] += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("migrate CGL failed: %s", exc)

    return counts


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Tiny shim — supports dataclasses + Pydantic models alike."""
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


__all__ = [
    "backfill_tenant",
    "export_tenant",
    "import_tenant",
    "migrate_tenant",
]
