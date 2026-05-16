"""time-aware decay + versioning configuration.

All fields default OFF. When ``DecayV2Config.enabled`` is ``False`` (the
default) every Part-D code path short-circuits so behaviour of
``Memory`` is byte-identical to the pre-feature main branch.

See ``doc_extra/TENANT_AND_DECAY/IMPLEMENTATION_GUIDE.md`` for the full
design.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ScheduleConfig(BaseModel):
    """Periodic decay scheduler parameters."""

    enabled: bool = Field(default=False)
    interval_seconds: float = Field(default=3600.0, gt=0,
        description="Seconds between scheduler ticks. Default: hourly.")
    jitter_seconds: float = Field(default=60.0, ge=0,
        description="Random ± jitter added to each interval to avoid "
        "thundering-herd ticks across replicas.")
    max_drift_seconds: float = Field(default=600.0, ge=0,
        description="Reserved — bounds clock skew when computing the next "
        "tick. Not enforced in D3.")


class VersioningConfig(BaseModel):
    """Immutable versioning parameters."""

    mode: Literal["overwrite", "immutable"] = Field(default="overwrite",
        description="overwrite — pre-feature in-place mutation. "
        "immutable — Memory.update creates a new vector-store row and "
        "marks the previous row metadata.status='superseded'.")
    keep_versions: int = Field(default=10, ge=1,
        description="Pruned chain length per memory. Older versions "
        "beyond this depth are dropped on insert / scheduler tick.")
    diff_provider: Literal["unified", "json", "none"] = Field(default="unified",
        description="Default mode for Memory.diff_versions.")


class ColdStorageConfig(BaseModel):
    """Cold-archival adapter parameters."""

    backend: Literal["disabled", "local", "s3"] = Field(default="disabled")
    local_root: Optional[str] = Field(default=None,
        description="Root directory for the local-disk adapter. None → "
        "{outhad_contextkit_dir}/cold_storage.")
    s3_bucket: Optional[str] = Field(default=None)
    s3_prefix: str = Field(default="outhad_contextkit/")
    archive_grace_seconds: float = Field(default=86400.0, ge=0,
        description="When the scheduler picks up archived nodes for cold "
        "demotion, only nodes archived more than this many seconds ago "
        "are moved. Defaults to 24h grace period.")


class DecayV2Config(BaseModel):
    """Master config for Part D — time-aware decay + versioning."""

    enabled: bool = Field(default=False,
        description="Master switch. Off = zero behaviour change, zero imports.")
    weight_recency: float = Field(default=0.6, ge=0, le=1,
        description="Coefficient for the exp-decay recency term in "
        "compute_decay_score.")
    weight_frequency: float = Field(default=0.3, ge=0, le=1,
        description="Coefficient for the log1p-normalised frequency term.")
    weight_feedback: float = Field(default=0.1, ge=0, le=1,
        description="Coefficient for the tanh-normalised feedback term.")
    archive_threshold: float = Field(default=0.05, ge=0, le=1,
        description="Memory.archive_low archives every node whose "
        "decay_score falls below this floor.")
    track_references: bool = Field(default=False,
        description="When True, Memory.record_reference(memory_id) bumps "
        "access_count + last_accessed_at to reflect downstream usage.")
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)
    cold_storage: ColdStorageConfig = Field(default_factory=ColdStorageConfig)

    @model_validator(mode="after")
    def _validate_weights_sum_to_at_most_one(self) -> "DecayV2Config":
        total = (
            self.weight_recency
            + self.weight_frequency
            + self.weight_feedback
        )
        if total - 1.0 > 1e-6:
            raise ValueError(
                f"DecayV2Config weights must sum to ≤ 1.0; got {total:.3f}"
            )
        return self


__all__ = [
    "DecayV2Config",
    "ScheduleConfig",
    "VersioningConfig",
    "ColdStorageConfig",
]
