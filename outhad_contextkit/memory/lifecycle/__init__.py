"""Time-aware decay + immutable versioning layer for Outhad_ContextKit.

This package stays inert until ``MemoryConfig.decay_v2.enabled`` is
``True``. Only lightweight config + dataclass symbols are re-exported
at module import; heavyweight modules (``scheduler``, ``versioning``,
``cold_storage``) are imported lazily from ``Memory._init_lifecycle``
so they never execute when the master switch is off.
"""
from __future__ import annotations

from outhad_contextkit.memory.lifecycle.config import (
    ColdStorageConfig,
    DecayV2Config,
    ScheduleConfig,
    VersioningConfig,
)

__all__ = [
    "DecayV2Config",
    "ScheduleConfig",
    "VersioningConfig",
    "ColdStorageConfig",
]
