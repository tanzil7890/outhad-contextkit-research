"""Tests for encryption functionality."""
import pytest

from outhad_contextkit.memory.privacy.base_classifier import SensitiveSpan
from outhad_contextkit.memory.privacy.config import PPMFConfig
from outhad_contextkit.memory.privacy.encryption import EncryptionManager, redact_text
from outhad_contextkit.memory.privacy.enums import (
    EncryptionAlgorithm,
    PrivacyLevel,
    RedactionMode,
    SensitiveSpanType,
)


class TestEncryption:
    """Test encryption/decryption."""
    
    def test_fernet_encryption_decryption(self):
        """Test Fernet encryption and decryption."""
        config = PPMFConfig(
            enabled=True,
            encryption_algorithm=EncryptionAlgorithm.FERNET
        )
        manager = EncryptionManager(config)
        
        plaintext = "This is sensitive data: john.doe@example.com"
        
        # Encrypt
        encrypted, metadata = manager.encrypt(plaintext)
        assert encrypted != plaintext
        assert metadata["encrypted"] is True
        assert metadata["algorithm"] == "fernet"
        
        # Decrypt
        decrypted = manager.decrypt(encrypted, metadata)
        assert decrypted == plaintext
    
    def test_aes_encryption_decryption(self):
        """Test AES-256-GCM encryption and decryption."""
        config = PPMFConfig(
            enabled=True,
            encryption_algorithm=EncryptionAlgorithm.AES_256_GCM
        )
        manager = EncryptionManager(config)
        
        plaintext = "Secret API key: sk-abc123xyz"
        
        # Encrypt
        encrypted, metadata = manager.encrypt(plaintext)
        assert encrypted != plaintext
        assert metadata["encrypted"] is True
        
        # Decrypt
        decrypted = manager.decrypt(encrypted, metadata)
        assert decrypted == plaintext
    
    def test_hash_sensitive_span(self):
        """Test one-way hashing."""
        config = PPMFConfig(enabled=True)
        manager = EncryptionManager(config)
        
        text = "john.doe@example.com"
        hash1 = manager.hash_sensitive_span(text)
        hash2 = manager.hash_sensitive_span(text)
        
        # Same text produces same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated to 16 chars
        
        # Different text produces different hash
        hash3 = manager.hash_sensitive_span("jane.doe@example.com")
        assert hash1 != hash3
    
    def test_empty_string_encryption(self):
        """Test encrypting empty string."""
        config = PPMFConfig(enabled=True)
        manager = EncryptionManager(config)
        
        encrypted, metadata = manager.encrypt("")
        assert encrypted == ""
        assert metadata == {}


class TestRedaction:
    """Test redaction functionality."""
    
    def test_mask_redaction(self):
        """Test MASK redaction mode."""
        text = "Email: john.doe@example.com"
        spans = [
            SensitiveSpan(
                text="john.doe@example.com",
                start=7,
                end=27,
                span_type=SensitiveSpanType.PII_EMAIL,
                privacy_level=PrivacyLevel.CONFIDENTIAL,
                confidence=1.0
            )
        ]
        
        redacted = redact_text(text, spans, RedactionMode.MASK)
        assert redacted == "Email: ********************"
    
    def test_replace_redaction(self):
        """Test REPLACE redaction mode."""
        text = "Email: john.doe@example.com"
        spans = [
            SensitiveSpan(
                text="john.doe@example.com",
                start=7,
                end=27,
                span_type=SensitiveSpanType.PII_EMAIL,
                privacy_level=PrivacyLevel.CONFIDENTIAL,
                confidence=1.0
            )
        ]
        
        redacted = redact_text(text, spans, RedactionMode.REPLACE)
        assert redacted == "Email: [PII_EMAIL]"
    
    def test_remove_redaction(self):
        """Test REMOVE redaction mode."""
        text = "Email: john.doe@example.com end"
        spans = [
            SensitiveSpan(
                text="john.doe@example.com",
                start=7,
                end=27,
                span_type=SensitiveSpanType.PII_EMAIL,
                privacy_level=PrivacyLevel.CONFIDENTIAL,
                confidence=1.0
            )
        ]
        
        redacted = redact_text(text, spans, RedactionMode.REMOVE)
        assert redacted == "Email:  end"
    
    def test_hash_redaction(self):
        """Test HASH redaction mode."""
        text = "Email: john.doe@example.com"
        spans = [
            SensitiveSpan(
                text="john.doe@example.com",
                start=7,
                end=27,
                span_type=SensitiveSpanType.PII_EMAIL,
                privacy_level=PrivacyLevel.CONFIDENTIAL,
                confidence=1.0
            )
        ]
        
        redacted = redact_text(text, spans, RedactionMode.HASH)
        assert redacted.startswith("Email: [HASH:")
        assert redacted.endswith("]")
    
    def test_multiple_span_redaction(self):
        """Test redacting multiple spans."""
        text = "Email john@example.com or call 555-1234"
        spans = [
            SensitiveSpan(
                text="john@example.com",
                start=6,
                end=22,
                span_type=SensitiveSpanType.PII_EMAIL,
                privacy_level=PrivacyLevel.CONFIDENTIAL,
                confidence=1.0
            ),
            SensitiveSpan(
                text="555-1234",
                start=31,
                end=39,
                span_type=SensitiveSpanType.PII_PHONE,
                privacy_level=PrivacyLevel.CONFIDENTIAL,
                confidence=1.0
            )
        ]
        
        redacted = redact_text(text, spans, RedactionMode.REPLACE)
        assert redacted == "Email [PII_EMAIL] or call [PII_PHONE]"
    
    def test_no_spans_redaction(self):
        """Test redaction with no spans."""
        text = "Regular text with no sensitive data"
        redacted = redact_text(text, [], RedactionMode.MASK)
        assert redacted == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

