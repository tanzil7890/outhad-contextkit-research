"""Multi-Stage Personalized Retrieval (MSPR) layer for Outhad_ContextKit.

This package stays inert until ``MemoryConfig.mspr.enabled`` is ``True``.
Only lightweight config + dataclass symbols are re-exported at module
import so ``import outhad_contextkit`` stays cheap when the feature flag
is off. Heavyweight modules (``feedback_store``, ``providers``) are
imported lazily from ``Memory._init_mspr`` so they never execute when
MSPR is disabled.
"""
from __future__ import annotations

from outhad_contextkit.memory.personalized.config import (
    FeedbackConfig,
    FrequencyConfig,
    IntentConfig,
    MSPRConfig,
    RoleConfig,
    SuccessConfig,
)
from outhad_contextkit.memory.personalized.role import (
    AllowAllPolicy,
    CompositePolicy,
    RolePolicy,
    RoleScopePolicy,
    TenantIsolationPolicy,
)
from outhad_contextkit.memory.personalized.types import (
    FeedbackEvent,
    RoleContext,
)


def _lazy_pipeline():
    """Re-export ``PersonalizedRetrievalPipeline`` lazily.

    Importing the pipeline at module load would pull ``time`` /
    ``logging`` regardless of whether MSPR is enabled. Lazy-loading
    keeps cold start cheap when the master switch is off.
    """
    from outhad_contextkit.memory.personalized.pipeline import (
        PersonalizedRetrievalPipeline,
    )

    return PersonalizedRetrievalPipeline


__all__ = [
    "MSPRConfig",
    "IntentConfig",
    "FeedbackConfig",
    "FrequencyConfig",
    "RoleConfig",
    "SuccessConfig",
    "FeedbackEvent",
    "RoleContext",
    "RolePolicy",
    "AllowAllPolicy",
    "TenantIsolationPolicy",
    "RoleScopePolicy",
    "CompositePolicy",
]
