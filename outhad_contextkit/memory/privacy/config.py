"""Configuration for Privacy-Preserving Memory Firewall."""
from typing import List, Optional

from pydantic import BaseModel, Field

from outhad_contextkit.memory.privacy.enums import (
    EncryptionAlgorithm,
    PrivacyLevel,
    RedactionMode,
    SensitiveSpanType,
)


class SensitivityRule(BaseModel):
    """Rule for classifying sensitive content."""
    span_type: SensitiveSpanType
    privacy_level: PrivacyLevel
    redaction_mode: RedactionMode
    encrypt_at_rest: bool = Field(default=True)
    patterns: Optional[List[str]] = Field(default=None, description="Regex patterns for detection")


class PPMFConfig(BaseModel):
    """Privacy-Preserving Memory Firewall configuration."""
    
    # Enable/disable PPMF
    enabled: bool = Field(default=False, description="Enable PPMF")
    
    # Classification settings
    use_llm_classifier: bool = Field(
        default=True,
        description="Use LLM for sensitivity classification (fallback to rule-based)"
    )
    classifier_model: Optional[str] = Field(
        default=None,
        description="Model for classification (uses memory LLM if None)"
    )
    classification_threshold: float = Field(
        default=0.7,
        description="Confidence threshold for LLM classification"
    )
    
    # Encryption settings
    encryption_algorithm: EncryptionAlgorithm = Field(
        default=EncryptionAlgorithm.FERNET,
        description="Encryption algorithm to use"
    )
    encryption_key_path: Optional[str] = Field(
        default=None,
        description="Path to encryption key file"
    )
    key_rotation_enabled: bool = Field(
        default=False,
        description="Enable automatic key rotation"
    )
    
    # Redaction settings
    default_redaction_mode: RedactionMode = Field(
        default=RedactionMode.ENCRYPT,
        description="Default redaction mode"
    )
    sensitivity_rules: List[SensitivityRule] = Field(
        default_factory=list,
        description="Custom sensitivity rules"
    )
    
    # Telemetry settings
    redact_telemetry: bool = Field(
        default=True,
        description="Redact sensitive data from telemetry"
    )
    
    # Adversarial testing
    enable_adversarial_testing: bool = Field(
        default=False,
        description="Enable MEXTRA-style adversarial testing"
    )
    adversarial_test_interval: int = Field(
        default=100,
        description="Run adversarial tests every N memory operations"
    )
    
    # Performance
    async_processing: bool = Field(
        default=True,
        description="Process privacy operations asynchronously"
    )
    cache_classifications: bool = Field(
        default=True,
        description="Cache classification results for identical content"
    )


def get_default_sensitivity_rules() -> List[SensitivityRule]:
    """Get default sensitivity rules."""
    return [
        SensitivityRule(
            span_type=SensitiveSpanType.PII_EMAIL,
            privacy_level=PrivacyLevel.CONFIDENTIAL,
            redaction_mode=RedactionMode.ENCRYPT,
            encrypt_at_rest=True,
            patterns=[r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b']
        ),
        SensitivityRule(
            span_type=SensitiveSpanType.PII_PHONE,
            privacy_level=PrivacyLevel.CONFIDENTIAL,
            redaction_mode=RedactionMode.MASK,
            encrypt_at_rest=True,
            patterns=[r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b']
        ),
        SensitivityRule(
            span_type=SensitiveSpanType.SECRET_API_KEY,
            privacy_level=PrivacyLevel.RESTRICTED,
            redaction_mode=RedactionMode.REMOVE,
            encrypt_at_rest=True,
            patterns=[
                r'sk-[a-zA-Z0-9]{48}',  # OpenAI
                r'AIza[0-9A-Za-z-_]{35}',  # Google
                r'AKIA[0-9A-Z]{16}',  # AWS
            ]
        ),
        SensitivityRule(
            span_type=SensitiveSpanType.PHI_MEDICAL_RECORD,
            privacy_level=PrivacyLevel.RESTRICTED,
            redaction_mode=RedactionMode.ENCRYPT,
            encrypt_at_rest=True,
        ),
    ]

