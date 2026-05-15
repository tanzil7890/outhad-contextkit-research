"""Privacy-related enumerations for PPMF."""
from enum import Enum


class PrivacyLevel(str, Enum):
    """Privacy levels for memory content."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


class SensitiveSpanType(str, Enum):
    """Types of sensitive data that can be detected."""
    # Personal Identifiable Information
    PII_NAME = "pii_name"
    PII_EMAIL = "pii_email"
    PII_PHONE = "pii_phone"
    PII_SSN = "pii_ssn"
    PII_ADDRESS = "pii_address"
    PII_DOB = "pii_dob"
    
    # Protected Health Information
    PHI_MEDICAL_RECORD = "phi_medical_record"
    PHI_DIAGNOSIS = "phi_diagnosis"
    PHI_MEDICATION = "phi_medication"
    PHI_INSURANCE = "phi_insurance"
    
    # Secrets & Credentials
    SECRET_API_KEY = "secret_api_key"
    SECRET_PASSWORD = "secret_password"
    SECRET_TOKEN = "secret_token"
    SECRET_PRIVATE_KEY = "secret_private_key"
    
    # Financial
    FINANCIAL_CREDIT_CARD = "financial_credit_card"
    FINANCIAL_BANK_ACCOUNT = "financial_bank_account"
    FINANCIAL_ROUTING = "financial_routing"
    
    # Other
    BIOMETRIC = "biometric"
    LOCATION = "location"
    CUSTOM = "custom"


class RedactionMode(str, Enum):
    """Modes for redacting sensitive data."""
    MASK = "mask"  # Replace with asterisks: ***
    REPLACE = "replace"  # Replace with type: [EMAIL]
    ENCRYPT = "encrypt"  # Encrypt the span
    REMOVE = "remove"  # Remove entirely
    HASH = "hash"  # One-way hash


class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms."""
    FERNET = "fernet"  # Symmetric (default)
    NACL = "nacl"  # NaCl Box encryption
    AES_256_GCM = "aes_256_gcm"  # AES-256 GCM mode

