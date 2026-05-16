""" Pluggable sink adapters for the CGL streaming change bus.

Every sink implements :class:`ChangeSink`. The CGL uses the sink ABC only;
concrete adapters are imported lazily so missing optional deps never crash on import.
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from outhad_contextkit.memory.context_graph.types import ChangeEvent


class ChangeSink(abc.ABC):
    """Base class for all CGL change-bus sinks.

    A sink is a consumer that receives :class:`ChangeEvent` objects produced
    by :class:`~outhad_contextkit.memory.context_graph.changelog.ContextChangeLog`.
    Concrete implementations are responsible for delivery and error handling;
    the bus only calls :meth:`send` and guarantees non-blocking queuing before
    the call.
    """

    @abc.abstractmethod
    def send(self, event: "ChangeEvent") -> None:
        """Deliver *event* to the sink.

        Implementations **must not** raise — swallow and log any errors.
        """

    def close(self) -> None:
        """Optional teardown. Called when the subscriber is removed."""


__all__ = ["ChangeSink"]
