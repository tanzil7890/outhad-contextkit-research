"""Privacy-Preserving Memory Firewall (PPMF) module."""
from outhad_contextkit.memory.privacy.classifier import MemorySanitizer
from outhad_contextkit.memory.privacy.config import PPMFConfig, SensitivityRule
from outhad_contextkit.memory.privacy.encryption import EncryptionManager, redact_text
from outhad_contextkit.memory.privacy.enums import (
    EncryptionAlgorithm,
    PrivacyLevel,
    RedactionMode,
    SensitiveSpanType,
)
from outhad_contextkit.memory.privacy.telemetry_sanitizer import TelemetrySanitizer

__all__ = [
    "MemorySanitizer",
    "EncryptionManager",
    "TelemetrySanitizer",
    "redact_text",
    "PrivacyLevel",
    "SensitiveSpanType",
    "RedactionMode",
    "EncryptionAlgorithm",
    "PPMFConfig",
    "SensitivityRule",
]

