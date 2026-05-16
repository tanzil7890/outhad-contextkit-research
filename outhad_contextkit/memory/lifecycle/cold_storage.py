"""cold archival storage adapters.

When a memory's ``decay_score`` stays below ``archive_threshold`` for
longer than ``archive_grace_seconds``, the scheduler can demote it to
cheap storage. The vector-store row is dropped + the CGL node deleted
+ the payload is handed to a :class:`ColdStorageAdapter`.
``Memory.promote_from_cold(memory_id)`` is the inverse — re-ingests
the payload from cold storage back into the live memory layers.

Two adapters ship in-tree:

* :class:`LocalDiskAdapter` — JSON files under a configurable root.
  Zero new dependencies; default for ``cold_storage.backend='local'``.
* :class:`S3Adapter` — opt-in; imports ``boto3`` inside ``__init__``
  so installations that don't enable the S3 backend never pay the
  import cost. Raises a clear ``ImportError`` when the dependency is
  missing.

Both adapters implement the same narrow interface so future cloud
adapters (GCS, Azure Blob) plug in without touching ``Memory`` code.
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ColdStorageAdapter(ABC):
    """Narrow ABC used by ``Memory.demote_to_cold`` /
    ``Memory.promote_from_cold``.
    """

    @abstractmethod
    def put(self, memory_id: str, payload: Dict[str, Any]) -> str:
        """Persist ``payload`` keyed by ``memory_id``. Returns an opaque
        location identifier (path / S3 URI). Raises on failure."""

    @abstractmethod
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Return the payload previously stored under ``memory_id`` or
        ``None`` when missing."""

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Remove the payload. Returns True when something was deleted."""


class LocalDiskAdapter(ColdStorageAdapter):
    """Stores each cold memory as a JSON file under ``root_dir``."""

    def __init__(self, root_dir: str) -> None:
        self._root = os.path.abspath(root_dir)
        os.makedirs(self._root, exist_ok=True)

    def _path(self, memory_id: str) -> str:
        # Simple sharded layout: first 2 chars of id → subdir.
        if not memory_id:
            raise ValueError("memory_id must be non-empty")
        prefix = memory_id[:2] if len(memory_id) >= 2 else memory_id
        sub = os.path.join(self._root, prefix)
        os.makedirs(sub, exist_ok=True)
        return os.path.join(sub, f"{memory_id}.json")

    def put(self, memory_id: str, payload: Dict[str, Any]) -> str:
        path = self._path(memory_id)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=str, sort_keys=True)
        return path

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(memory_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def delete(self, memory_id: str) -> bool:
        path = self._path(memory_id)
        if not os.path.exists(path):
            return False
        try:
            os.remove(path)
            return True
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("LocalDiskAdapter.delete failed: %s", exc)
            return False


class S3Adapter(ColdStorageAdapter):
    """boto3-backed S3 adapter (opt-in). ``boto3`` imported inside
    ``__init__`` so packages that don't enable S3 cold storage never
    require the dependency.
    """

    def __init__(self, *, bucket: str, prefix: str = "outhad_contextkit/") -> None:
        try:
            import boto3  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "S3Adapter requires boto3. Install with: pip install boto3"
            ) from exc
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._client = boto3.client("s3")

    def _key(self, memory_id: str) -> str:
        if not memory_id:
            raise ValueError("memory_id must be non-empty")
        return f"{self._prefix}{memory_id}.json"

    def put(self, memory_id: str, payload: Dict[str, Any]) -> str:
        key = self._key(memory_id)
        body = json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
        self._client.put_object(Bucket=self._bucket, Key=key, Body=body)
        return f"s3://{self._bucket}/{key}"

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        key = self._key(memory_id)
        try:
            obj = self._client.get_object(Bucket=self._bucket, Key=key)
        except Exception:
            return None
        try:
            return json.loads(obj["Body"].read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("S3Adapter.get parse failed: %s", exc)
            return None

    def delete(self, memory_id: str) -> bool:
        key = self._key(memory_id)
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("S3Adapter.delete failed: %s", exc)
            return False


__all__ = ["ColdStorageAdapter", "LocalDiskAdapter", "S3Adapter"]
