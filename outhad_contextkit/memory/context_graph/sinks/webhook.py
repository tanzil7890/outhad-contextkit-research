"""Phase E — HMAC-signed webhook sink.

Posts each :class:`ChangeEvent` to a URL as JSON with an HMAC-SHA-256
signature so consumers can verify authenticity and guard against replay.

Signature scheme::

    X-CGL-Event-Id: <row_id>
    X-CGL-Timestamp: <ISO-8601 UTC>
    X-CGL-Signature: sha256=HMAC(secret, f"{timestamp}\\n{body_bytes}")

Consumers should reject requests where ``|now - timestamp| > 300s``.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from outhad_contextkit.memory.context_graph.sinks import ChangeSink

if TYPE_CHECKING:
    from outhad_contextkit.memory.context_graph.types import ChangeEvent

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE = 0.5  # seconds


class WebhookSink(ChangeSink):
    """HTTP POST sink with HMAC-SHA-256 signatures and exponential-backoff retry.

    Args:
        url: Destination endpoint.
        secret: Shared secret for HMAC signing. Pass ``None`` to skip signing.
        timeout: Per-request timeout in seconds.
        max_retries: Retry budget (default 3).
    """

    def __init__(
        self,
        url: str,
        secret: Optional[str] = None,
        *,
        timeout: float = 5.0,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self._url = url
        self._secret = secret.encode() if isinstance(secret, str) else secret
        self._timeout = timeout
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    def send(self, event: "ChangeEvent") -> None:
        body = json.dumps(
            {
                "event_type": event.event_type,
                "target_id": event.target_id,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "payload": event.payload or {},
            },
            separators=(",", ":"),
        ).encode()

        ts = datetime.now(timezone.utc).isoformat()
        headers = {
            "Content-Type": "application/json",
            "X-CGL-Timestamp": ts,
            "X-CGL-Event-Type": event.event_type,
        }
        if self._secret:
            sig = hmac.new(
                self._secret,
                f"{ts}\n".encode() + body,
                hashlib.sha256,
            ).hexdigest()
            headers["X-CGL-Signature"] = f"sha256={sig}"

        for attempt in range(self._max_retries):
            try:
                import urllib.request

                req = urllib.request.Request(
                    self._url,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout):
                    return
            except Exception as exc:
                if attempt < self._max_retries - 1:
                    time.sleep(_RETRY_BASE * (2**attempt))
                else:
                    logger.debug("WebhookSink: all retries exhausted for %s: %s", self._url, exc)

    # ------------------------------------------------------------------
    @staticmethod
    def verify_signature(
        body: bytes,
        timestamp: str,
        signature_header: str,
        secret: str,
    ) -> bool:
        """Helper for receivers: verify the X-CGL-Signature header.

        Returns True when the signature is valid and the timestamp skew is < 5 min.
        """
        try:
            ts_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            skew = abs((datetime.now(timezone.utc) - ts_dt).total_seconds())
            if skew > 300:
                return False
        except Exception:
            return False

        expected = hmac.new(
            secret.encode() if isinstance(secret, str) else secret,
            f"{timestamp}\n".encode() + body,
            hashlib.sha256,
        ).hexdigest()
        provided = signature_header.removeprefix("sha256=")
        return hmac.compare_digest(expected, provided)


__all__ = ["WebhookSink"]
