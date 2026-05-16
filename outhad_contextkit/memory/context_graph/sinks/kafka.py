""" Kafka sink (optional; requires ``confluent-kafka``).

Install with::

    pip install outhad_contextkit[context-graph-kafka]

The import guard ensures that the absence of ``confluent-kafka`` raises a
clear error only at *construction* time, not at module import time.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from outhad_contextkit.memory.context_graph.sinks import ChangeSink

if TYPE_CHECKING:
    from outhad_contextkit.memory.context_graph.types import ChangeEvent

logger = logging.getLogger(__name__)


class KafkaSink(ChangeSink):
    """Publish CGL change events to a Kafka topic.

    Args:
        bootstrap_servers: Comma-separated ``host:port`` pairs.
        topic: Kafka topic name.
        producer_config: Extra kwargs forwarded to ``confluent_kafka.Producer``.

    Raises:
        RuntimeError: At construction time when ``confluent-kafka`` is not installed.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        *,
        producer_config: Optional[dict] = None,
    ) -> None:
        try:
            from confluent_kafka import Producer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "KafkaSink requires 'confluent-kafka'. "
                "Install it with: pip install outhad_contextkit[context-graph-kafka]"
            ) from exc

        cfg = {"bootstrap.servers": bootstrap_servers}
        if producer_config:
            cfg.update(producer_config)
        self._producer = Producer(cfg)
        self._topic = topic

    def send(self, event: "ChangeEvent") -> None:
        value = json.dumps(
            {
                "event_type": event.event_type,
                "target_id": event.target_id,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "payload": event.payload or {},
            },
            separators=(",", ":"),
        ).encode()
        try:
            self._producer.produce(
                self._topic,
                value=value,
                key=event.target_id.encode(),
                on_delivery=self._on_delivery,
            )
            self._producer.poll(0)
        except Exception as exc:
            logger.debug("KafkaSink: produce failed: %s", exc)

    def close(self) -> None:
        try:
            self._producer.flush(timeout=5.0)
        except Exception as exc:
            logger.debug("KafkaSink: flush on close failed: %s", exc)

    @staticmethod
    def _on_delivery(err, msg) -> None:
        if err:
            logger.debug("KafkaSink: delivery failed for %s: %s", msg.topic(), err)


__all__ = ["KafkaSink"]
