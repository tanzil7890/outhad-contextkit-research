"""Encryption manager for PPMF."""
import base64
import hashlib
import logging
import os
from typing import Dict, Optional, Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from outhad_contextkit.memory.privacy.config import PPMFConfig
from outhad_contextkit.memory.privacy.enums import EncryptionAlgorithm, RedactionMode

logger = logging.getLogger(__name__)


class EncryptionManager:
    """Manages encryption/decryption of sensitive memory content."""
    
    def __init__(self, config: PPMFConfig):
        self.config = config
        self.algorithm = config.encryption_algorithm
        
        # Load or generate encryption key
        self.key = self._load_or_generate_key()
        
        # Initialize cipher based on algorithm
        if self.algorithm == EncryptionAlgorithm.FERNET:
            self.cipher = Fernet(self.key)
        elif self.algorithm == EncryptionAlgorithm.AES_256_GCM:
            self.cipher = AESGCM(self.key)
        else:
            raise ValueError(f"Unsupported encryption algorithm: {self.algorithm}")
    
    def _load_or_generate_key(self) -> bytes:
        """Load encryption key from file or generate new one."""
        key_path = self.config.encryption_key_path
        
        if key_path and os.path.exists(key_path):
            # Load existing key
            with open(key_path, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            if self.algorithm == EncryptionAlgorithm.FERNET:
                key = Fernet.generate_key()
            elif self.algorithm == EncryptionAlgorithm.AES_256_GCM:
                key = AESGCM.generate_key(bit_length=256)
            else:
                raise ValueError(f"Cannot generate key for algorithm: {self.algorithm}")
            
            # Save key if path specified
            if key_path:
                os.makedirs(os.path.dirname(key_path), exist_ok=True)
                with open(key_path, 'wb') as f:
                    f.write(key)
                logger.info(f"Generated and saved encryption key to {key_path}")
            
            return key
    
    def encrypt(self, plaintext: str) -> Tuple[str, Dict]:
        """
        Encrypt plaintext and return ciphertext + metadata.
        
        Returns:
            Tuple of (encrypted_text, encryption_metadata)
        """
        if not plaintext:
            return plaintext, {}
        
        try:
            plaintext_bytes = plaintext.encode('utf-8')
            
            if self.algorithm == EncryptionAlgorithm.FERNET:
                encrypted_bytes = self.cipher.encrypt(plaintext_bytes)
                encrypted_text = base64.b64encode(encrypted_bytes).decode('utf-8')
                
                metadata = {
                    "encrypted": True,
                    "algorithm": self.algorithm.value,
                    "encoding": "base64",
                }
            
            elif self.algorithm == EncryptionAlgorithm.AES_256_GCM:
                nonce = os.urandom(12)
                encrypted_bytes = self.cipher.encrypt(nonce, plaintext_bytes, None)
                
                # Combine nonce + ciphertext
                combined = nonce + encrypted_bytes
                encrypted_text = base64.b64encode(combined).decode('utf-8')
                
                metadata = {
                    "encrypted": True,
                    "algorithm": self.algorithm.value,
                    "encoding": "base64",
                    "nonce_length": 12,
                }
            
            return encrypted_text, metadata
        
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            # Return original text with error metadata
            return plaintext, {"encryption_error": str(e)}
    
    def decrypt(self, encrypted_text: str, metadata: Dict) -> str:
        """
        Decrypt encrypted text using metadata.
        
        Args:
            encrypted_text: The encrypted string
            metadata: Encryption metadata containing algorithm info
        
        Returns:
            Decrypted plaintext
        """
        if not metadata.get("encrypted"):
            return encrypted_text
        
        try:
            algorithm = metadata.get("algorithm")
            
            if algorithm == EncryptionAlgorithm.FERNET.value:
                encrypted_bytes = base64.b64decode(encrypted_text.encode('utf-8'))
                plaintext_bytes = self.cipher.decrypt(encrypted_bytes)
                return plaintext_bytes.decode('utf-8')
            
            elif algorithm == EncryptionAlgorithm.AES_256_GCM.value:
                combined = base64.b64decode(encrypted_text.encode('utf-8'))
                nonce_length = metadata.get("nonce_length", 12)
                
                nonce = combined[:nonce_length]
                ciphertext = combined[nonce_length:]
                
                plaintext_bytes = self.cipher.decrypt(nonce, ciphertext, None)
                return plaintext_bytes.decode('utf-8')
            
            else:
                logger.error(f"Unknown encryption algorithm: {algorithm}")
                return encrypted_text
        
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_text
    
    def hash_sensitive_span(self, text: str) -> str:
        """
        Create one-way hash of sensitive span.
        Useful for matching without storing plaintext.
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def redact_text(text: str, spans, mode: RedactionMode) -> str:
    """
    Redact sensitive spans from text based on mode.
    
    Args:
        text: Original text
        spans: List of SensitiveSpan objects
        mode: RedactionMode (MASK, REPLACE, REMOVE, etc.)
    
    Returns:
        Redacted text
    """
    if not spans:
        return text
    
    # Sort spans by start position (reverse to maintain indices)
    sorted_spans = sorted(spans, key=lambda s: s.start, reverse=True)
    
    redacted = text
    for span in sorted_spans:
        if mode == RedactionMode.MASK:
            replacement = "*" * len(span.text)
        elif mode == RedactionMode.REPLACE:
            replacement = f"[{span.span_type.value.upper()}]"
        elif mode == RedactionMode.REMOVE:
            replacement = ""
        elif mode == RedactionMode.HASH:
            hash_val = hashlib.sha256(span.text.encode()).hexdigest()[:8]
            replacement = f"[HASH:{hash_val}]"
        else:
            replacement = span.text  # No change
        
        redacted = redacted[:span.start] + replacement + redacted[span.end:]
    
    return redacted

