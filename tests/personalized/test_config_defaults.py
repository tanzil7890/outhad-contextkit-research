""" MSPR config defaults and opt-in semantics.

MSPR must remain dormant until the operator flips ``mspr.enabled``.
These tests pin the defaults so a regression in a sub-config cannot
silently flip behaviour for existing users.
"""
from __future__ import annotations

from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.context_graph.config import RetrievalConfig
from outhad_contextkit.memory.personalized.config import (
    FeedbackConfig,
    FrequencyConfig,
    IntentConfig,
    MSPRConfig,
    RoleConfig,
    SuccessConfig,
)


def test_mspr_disabled_by_default():
    cfg = MemoryConfig()
    assert cfg.mspr.enabled is False
    assert cfg.mspr.intent.enabled is False
    assert cfg.mspr.feedback.enabled is False
    assert cfg.mspr.frequency.enabled is False
    assert cfg.mspr.role.enabled is False
    assert cfg.mspr.success.enabled is False


def test_sub_config_defaults_are_safe():
    cfg = MSPRConfig()
    assert isinstance(cfg.intent, IntentConfig)
    assert isinstance(cfg.feedback, FeedbackConfig)
    assert isinstance(cfg.frequency, FrequencyConfig)
    assert isinstance(cfg.role, RoleConfig)
    assert isinstance(cfg.success, SuccessConfig)

    assert cfg.feedback.boost_delta > 0
    assert cfg.feedback.penalty_delta > 0
    assert cfg.feedback.pin_floor_on_boost is True
    assert 0 <= cfg.feedback.max_boost_per_memory <= 1

    assert cfg.frequency.normalise == "log1p"
    assert cfg.intent.strategy == "regex"
    assert cfg.intent.enable_regex_fastpath is True
    assert cfg.role.hard_filter is True
    assert cfg.success.min_sample_size >= 1


def test_retrieval_weights_default_to_zero():
    cfg = RetrievalConfig()
    assert cfg.delta_personal == 0.0
    assert cfg.epsilon_frequency == 0.0
    assert cfg.zeta_success == 0.0


def test_retrieval_weights_enforce_bounds():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RetrievalConfig(delta_personal=1.5)
    with pytest.raises(ValidationError):
        RetrievalConfig(epsilon_frequency=-0.1)
    with pytest.raises(ValidationError):
        RetrievalConfig(zeta_success=2.0)


def test_mspr_config_roundtrip_dict():
    cfg = MSPRConfig(enabled=True)
    cfg.feedback.enabled = True
    cfg.frequency.enabled = True
    dumped = cfg.model_dump()
    restored = MSPRConfig(**dumped)
    assert restored.enabled is True
    assert restored.feedback.enabled is True
    assert restored.frequency.enabled is True
    assert restored.role.enabled is False


def test_legacy_memory_config_does_not_break():
    """A MemoryConfig constructed exactly as before MSPR must still work."""
    cfg = MemoryConfig()
    assert hasattr(cfg, "mspr")
    assert hasattr(cfg, "context_graph")
    assert cfg.context_graph.enabled is False
    assert cfg.mspr.enabled is False
