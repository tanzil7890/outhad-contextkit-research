# PPMF API Reference

## Core Classes

### PPMFConfig

Configuration class for Privacy-Preserving Memory Firewall.

```python
from outhad_contextkit.memory.privacy.config import PPMFConfig
```

#### Parameters

See [Configuration Guide](02_configuration.md) for detailed parameter documentation.

#### Example

```python
config = PPMFConfig(
    enabled=True,
    use_llm_classifier=False,
    encryption_algorithm=EncryptionAlgorithm.FERNET
)
```

---

### MemorySanitizer

Hybrid classifier for detecting sensitive content.

```python
from outhad_contextkit.memory.privacy import MemorySanitizer
```

#### Methods

##### `classify(text: str) -> ClassificationResult`

Classify text for sensitive content.

**Parameters**:
- `text` (str): Text to classify

**Returns**: `ClassificationResult` with:
- `original_text` (str): Input text
- `has_sensitive_content` (bool): Whether sensitive data was found
- `overall_privacy_level` (PrivacyLevel): Highest privacy level detected
- `sensitive_spans` (List[SensitiveSpan]): Detected sensitive spans
- `processing_time_ms` (float): Processing time

**Example**:
```python
from outhad_contextkit.memory.privacy import MemorySanitizer
from outhad_contextkit.memory.privacy.config import PPMFConfig

sanitizer = MemorySanitizer(PPMFConfig(enabled=True))
result = sanitizer.classify("Email: john@example.com")

print(f"Sensitive: {result.has_sensitive_content}")
print(f"Level: {result.overall_privacy_level}")
for span in result.sensitive_spans:
    print(f"  - {span.text} ({span.span_type})")
```

##### `classify_batch(texts: List[str]) -> List[ClassificationResult]`

Classify multiple texts.

**Parameters**:
- `texts` (List[str]): List of texts to classify

**Returns**: List of `ClassificationResult`

---

### EncryptionManager

Manages encryption and decryption of sensitive data.

```python
from outhad_contextkit.memory.privacy import EncryptionManager
```

#### Methods

##### `encrypt(plaintext: str) -> Tuple[str, Dict]`

Encrypt plaintext.

**Parameters**:
- `plaintext` (str): Text to encrypt

**Returns**: Tuple of (encrypted_text, metadata)

**Example**:
```python
from outhad_contextkit.memory.privacy import EncryptionManager
from outhad_contextkit.memory.privacy.config import PPMFConfig

manager = EncryptionManager(PPMFConfig(enabled=True))
encrypted, metadata = manager.encrypt("secret data")

print(f"Encrypted: {encrypted}")
print(f"Algorithm: {metadata['algorithm']}")
```

##### `decrypt(encrypted_text: str, metadata: Dict) -> str`

Decrypt encrypted text.

**Parameters**:
- `encrypted_text` (str): Encrypted text
- `metadata` (Dict): Encryption metadata from encrypt()

**Returns**: Decrypted plaintext (str)

**Example**:
```python
decrypted = manager.decrypt(encrypted, metadata)
print(f"Decrypted: {decrypted}")
```

##### `hash_sensitive_span(text: str) -> str`

Create one-way hash of sensitive data.

**Parameters**:
- `text` (str): Text to hash

**Returns**: Hash string (16 characters)

**Example**:
```python
hash_val = manager.hash_sensitive_span("john@example.com")
print(f"Hash: {hash_val}")  # e.g., "973dfe463b15a5d2"
```

---

### TelemetrySanitizer

Sanitizes telemetry data to prevent sensitive information leakage.

```python
from outhad_contextkit.memory.privacy import TelemetrySanitizer
```

#### Methods

##### `sanitize_event_data(event_data: Dict[str, Any]) -> Dict[str, Any]`

Sanitize telemetry event data.

**Parameters**:
- `event_data` (Dict): Event data dictionary

**Returns**: Sanitized event data (Dict)

**Example**:
```python
sanitizer = TelemetrySanitizer(config, classifier)

data = {
    "user_message": "Email: test@example.com",
    "action": "add_memory"
}

sanitized = sanitizer.sanitize_event_data(data)
# Result: {"user_message": "Email: [PII_EMAIL]", "action": "add_memory"}
```

---

## Utility Functions

### redact_text()

Redact sensitive spans from text.

```python
from outhad_contextkit.memory.privacy import redact_text
from outhad_contextkit.memory.privacy.enums import RedactionMode
```

**Signature**:
```python
def redact_text(
    text: str,
    spans: List[SensitiveSpan],
    mode: RedactionMode
) -> str
```

**Parameters**:
- `text` (str): Original text
- `spans` (List[SensitiveSpan]): Sensitive spans to redact
- `mode` (RedactionMode): Redaction mode

**Returns**: Redacted text (str)

**Example**:
```python
# Get classification
result = sanitizer.classify("Email: john@example.com")

# Redact with different modes
masked = redact_text(text, result.sensitive_spans, RedactionMode.MASK)
# "Email: *****************"

replaced = redact_text(text, result.sensitive_spans, RedactionMode.REPLACE)
# "Email: [PII_EMAIL]"

removed = redact_text(text, result.sensitive_spans, RedactionMode.REMOVE)
# "Email: "

hashed = redact_text(text, result.sensitive_spans, RedactionMode.HASH)
# "Email: [HASH:973dfe46]"
```

---

## Enums

### PrivacyLevel

Privacy classification levels.

```python
from outhad_contextkit.memory.privacy.enums import PrivacyLevel

class PrivacyLevel(str, Enum):
    PUBLIC = "public"           # No sensitive information
    INTERNAL = "internal"       # Internal info, not public
    CONFIDENTIAL = "confidential"  # Protected, limited access
    SENSITIVE = "sensitive"     # Highly sensitive, strict controls
    RESTRICTED = "restricted"   # Maximum protection
```

### SensitiveSpanType

Types of sensitive data.

```python
from outhad_contextkit.memory.privacy.enums import SensitiveSpanType

# PII
SensitiveSpanType.PII_NAME
SensitiveSpanType.PII_EMAIL
SensitiveSpanType.PII_PHONE
SensitiveSpanType.PII_SSN
SensitiveSpanType.PII_ADDRESS
SensitiveSpanType.PII_DOB

# PHI
SensitiveSpanType.PHI_MEDICAL_RECORD
SensitiveSpanType.PHI_DIAGNOSIS
SensitiveSpanType.PHI_MEDICATION
SensitiveSpanType.PHI_INSURANCE

# Secrets
SensitiveSpanType.SECRET_API_KEY
SensitiveSpanType.SECRET_PASSWORD
SensitiveSpanType.SECRET_TOKEN
SensitiveSpanType.SECRET_PRIVATE_KEY

# Financial
SensitiveSpanType.FINANCIAL_CREDIT_CARD
SensitiveSpanType.FINANCIAL_BANK_ACCOUNT
SensitiveSpanType.FINANCIAL_ROUTING

# Other
SensitiveSpanType.BIOMETRIC
SensitiveSpanType.LOCATION
SensitiveSpanType.CUSTOM
```

### RedactionMode

Modes for redacting sensitive data.

```python
from outhad_contextkit.memory.privacy.enums import RedactionMode

class RedactionMode(str, Enum):
    MASK = "mask"        # Replace with asterisks: ***
    REPLACE = "replace"  # Replace with type: [EMAIL]
    ENCRYPT = "encrypt"  # Encrypt the span
    REMOVE = "remove"    # Remove entirely
    HASH = "hash"        # One-way hash
```

### EncryptionAlgorithm

Supported encryption algorithms.

```python
from outhad_contextkit.memory.privacy.enums import EncryptionAlgorithm

class EncryptionAlgorithm(str, Enum):
    FERNET = "fernet"          # Symmetric (default)
    NACL = "nacl"              # NaCl Box encryption
    AES_256_GCM = "aes_256_gcm"  # AES-256 GCM mode
```

---

## Data Models

### SensitiveSpan

Represents a detected sensitive span in text.

```python
class SensitiveSpan(BaseModel):
    text: str                    # The sensitive text
    start: int                   # Start position
    end: int                     # End position
    span_type: SensitiveSpanType # Type of sensitive data
    privacy_level: PrivacyLevel  # Privacy classification
    confidence: float            # Confidence score (0.0-1.0)
    metadata: Optional[dict]     # Additional metadata
```

### ClassificationResult

Result of sensitivity classification.

```python
class ClassificationResult(BaseModel):
    original_text: str               # Input text
    has_sensitive_content: bool      # Contains sensitive data
    overall_privacy_level: PrivacyLevel  # Highest level detected
    sensitive_spans: List[SensitiveSpan]  # Detected spans
    processing_time_ms: float        # Processing time
```

### SensitivityRule

Custom sensitivity detection rule.

```python
class SensitivityRule(BaseModel):
    span_type: SensitiveSpanType     # Type to detect
    privacy_level: PrivacyLevel      # Privacy level
    redaction_mode: RedactionMode    # How to redact
    encrypt_at_rest: bool            # Encrypt in storage
    patterns: Optional[List[str]]    # Regex patterns
```

---

## Memory Integration

PPMF integrates seamlessly with the Memory class:

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig

# Create memory with PPMF
config = MemoryConfig(ppmf=PPMFConfig(enabled=True))
memory = Memory(config=config)

# Access PPMF components
if memory.ppmf_enabled:
    # Classifier
    result = memory._memory_sanitizer.classify("text")
    
    # Encryption
    encrypted, meta = memory._encryption_manager.encrypt("secret")
    
    # Telemetry
    sanitized = memory._telemetry_sanitizer.sanitize_event_data(data)
```

---

## Next Steps

- [Examples](04_examples.md) - Real-world usage patterns
- [Advanced Topics](05_advanced.md) - Custom rules, key management
- [Configuration Guide](02_configuration.md) - Detailed configuration

