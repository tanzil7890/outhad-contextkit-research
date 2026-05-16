""" Server-Sent Events (SSE) sink.

Produces an async generator of SSE-formatted text lines suitable for use
with FastAPI / Starlette ``StreamingResponse``::

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI()
    sse_sink = SSESink()
    changelog.subscribe(sse_sink.send)

    @app.get("/events")
    async def stream():
        return StreamingResponse(sse_sink.stream(), media_type="text/event-stream")
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
from typing import TYPE_CHECKING, AsyncIterator

from outhad_contextkit.memory.context_graph.sinks import ChangeSink

if TYPE_CHECKING:
    from outhad_contextkit.memory.context_graph.types import ChangeEvent

logger = logging.getLogger(__name__)

_QUEUE_MAXSIZE = 512


class SSESink(ChangeSink):
    """Converts CGL change events into SSE lines for a streaming HTTP endpoint.

    The :meth:`send` method is called by the change-bus dispatcher thread;
    :meth:`stream` is an async generator consumed by the HTTP handler coroutine.
    """

    def __init__(self, maxsize: int = _QUEUE_MAXSIZE) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._closed = False

    # ------------------------------------------------------------------
    def send(self, event: "ChangeEvent") -> None:
        if self._closed:
            return
        data = json.dumps(
            {
                "event_type": event.event_type,
                "target_id": event.target_id,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "payload": event.payload or {},
            }
        )
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                logger.debug("SSESink queue full, dropping event")

    def close(self) -> None:
        self._closed = True
        try:
            self._queue.put_nowait(None)  # sentinel
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    async def stream(self) -> AsyncIterator[str]:
        """Yield SSE-formatted lines. Suitable for FastAPI StreamingResponse."""
        while True:
            try:
                data = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._queue.get(timeout=1.0)
                )
                if data is None:  # close sentinel
                    break
                yield f"data: {data}\n\n"
            except Exception:
                if self._closed:
                    break


__all__ = ["SSESink"]
