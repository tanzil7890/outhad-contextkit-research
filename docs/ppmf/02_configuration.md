# PPMF Configuration Guide

## Configuration Overview

PPMF is configured through the `PPMFConfig` class, which is part of `MemoryConfig`.

```python
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig

config = MemoryConfig(
    ppmf=PPMFConfig(
        enabled=True,
        # ... other settings
    )
)
```

## Configuration Parameters

### Basic Settings

#### `enabled: bool = False`
Enable or disable PPMF entirely.

```python
PPMFConfig(enabled=True)  # Enable PPMF
```

**Default**: `False` (disabled for backward compatibility)

---

### Classification Settings

#### `use_llm_classifier: bool = True`
Use LLM for contextual sensitivity classification. Falls back to rule-based if False.

```python
PPMFConfig(
    enabled=True,
    use_llm_classifier=False  # Fast, rule-based only
)
```

**Default**: `True`  
**Performance**: 
- `False`: < 1ms (fast, regex-based)
- `True`: 500-1000ms (accurate, context-aware)

#### `classifier_model: Optional[str] = None`
Specify which LLM model to use for classification. If None, uses the main memory LLM.

```python
PPMFConfig(
    enabled=True,
    classifier_model="gpt-3.5-turbo"  # Specific model for classification
)
```

**Default**: `None` (uses memory LLM)

#### `classification_threshold: float = 0.7`
Minimum confidence threshold for LLM classifications.

```python
PPMFConfig(
    enabled=True,
    classification_threshold=0.9  # Higher threshold = fewer false positives
)
```

**Default**: `0.7`  
**Range**: 0.0 to 1.0  
**Recommended**: 0.7-0.9

---

### Encryption Settings

#### `encryption_algorithm: EncryptionAlgorithm = FERNET`
Choose the encryption algorithm.

```python
from outhad_contextkit.memory.privacy.enums import EncryptionAlgorithm

PPMFConfig(
    enabled=True,
    encryption_algorithm=EncryptionAlgorithm.AES_256_GCM
)
```

**Options**:
- `FERNET`: Symmetric encryption (default, fast)
- `AES_256_GCM`: AES-256 in GCM mode (AEAD)
- `NACL`: NaCl encryption

**Default**: `FERNET`

#### `encryption_key_path: Optional[str] = None`
Path to encryption key file. If not specified, generates a new key in memory.

```python
PPMFConfig(
    enabled=True,
    encryption_key_path="/secure/keys/ppmf_key.bin"
)
```

**Default**: `None` (generates temporary key)  
**Production**: Always specify a secure key path!

#### `key_rotation_enabled: bool = False`
Enable automatic key rotation (future feature).

```python
PPMFConfig(
    enabled=True,
    key_rotation_enabled=True
)
```

**Default**: `False`  
**Status**: Coming soon

---

### Redaction Settings

#### `default_redaction_mode: RedactionMode = ENCRYPT`
Default mode for redacting sensitive data.

```python
from outhad_contextkit.memory.privacy.enums import RedactionMode

PPMFConfig(
    enabled=True,
    default_redaction_mode=RedactionMode.REPLACE
)
```

**Options**:
- `MASK`: Replace with asterisks (`***`)
- `REPLACE`: Replace with type labels (`[EMAIL]`)
- `ENCRYPT`: Encrypt the span
- `REMOVE`: Delete entirely
- `HASH`: One-way hash

**Default**: `ENCRYPT`

#### `sensitivity_rules: List[SensitivityRule] = []`
Custom sensitivity rules for domain-specific patterns.

```python
from outhad_contextkit.memory.privacy.config import SensitivityRule
from outhad_contextkit.memory.privacy.enums import (
    SensitiveSpanType,
    PrivacyLevel,
    RedactionMode
)

custom_rule = SensitivityRule(
    span_type=SensitiveSpanType.CUSTOM,
    privacy_level=PrivacyLevel.RESTRICTED,
    redaction_mode=RedactionMode.ENCRYPT,
    encrypt_at_rest=True,
    patterns=[r'PATIENT-\d{6}']  # Custom pattern
)

PPMFConfig(
    enabled=True,
    sensitivity_rules=[custom_rule]
)
```

**Default**: `[]` (uses built-in rules)

---

### Telemetry Settings

#### `redact_telemetry: bool = True`
Redact sensitive data from telemetry events.

```python
PPMFConfig(
    enabled=True,
    redact_telemetry=True  # Protect analytics data
)
```

**Default**: `True`

---

### Adversarial Testing Settings

#### `enable_adversarial_testing: bool = False`
Enable MEXTRA-style adversarial testing.

```python
PPMFConfig(
    enabled=True,
    enable_adversarial_testing=True
)
```

**Default**: `False`  
**Use**: Testing and validation environments

#### `adversarial_test_interval: int = 100`
Run adversarial tests every N memory operations.

```python
PPMFConfig(
    enabled=True,
    enable_adversarial_testing=True,
    adversarial_test_interval=50  # Test every 50 operations
)
```

**Default**: `100`

---

### Performance Settings

#### `async_processing: bool = True`
Process privacy operations asynchronously.

```python
PPMFConfig(
    enabled=True,
    async_processing=True  # Better performance
)
```

**Default**: `True`

#### `cache_classifications: bool = True`
Cache classification results for identical content.

```python
PPMFConfig(
    enabled=True,
    cache_classifications=True  # Avoid re-classifying same text
)
```

**Default**: `True`

---

## Configuration Presets

### Development (Fast)
```python
PPMFConfig(
    enabled=True,
    use_llm_classifier=False,
    encryption_algorithm=EncryptionAlgorithm.FERNET,
    default_redaction_mode=RedactionMode.REPLACE,
    redact_telemetry=False,
    enable_adversarial_testing=True
)
```

### Production (Balanced)
```python
PPMFConfig(
    enabled=True,
    use_llm_classifier=False,
    encryption_algorithm=EncryptionAlgorithm.FERNET,
    encryption_key_path="/secure/keys/ppmf.key",
    default_redaction_mode=RedactionMode.ENCRYPT,
    redact_telemetry=True,
    classification_threshold=0.8
)
```

### High Security (Maximum Protection)
```python
PPMFConfig(
    enabled=True,
    use_llm_classifier=True,
    encryption_algorithm=EncryptionAlgorithm.AES_256_GCM,
    encryption_key_path="/secure/keys/ppmf.key",
    default_redaction_mode=RedactionMode.ENCRYPT,
    redact_telemetry=True,
    classification_threshold=0.9,
    enable_adversarial_testing=True,
    adversarial_test_interval=50
)
```

## Environment Variables

You can configure PPMF using environment variables:

```bash
# .env file
PPMF_ENABLED=true
PPMF_USE_LLM=false
PPMF_ENCRYPTION_KEY_PATH=/secure/keys/ppmf.key
PPMF_REDACT_TELEMETRY=true
```

Then in code:
```python
import os

config = MemoryConfig(
    ppmf=PPMFConfig(
        enabled=os.getenv("PPMF_ENABLED", "false").lower() == "true",
        use_llm_classifier=os.getenv("PPMF_USE_LLM", "false").lower() == "true",
        encryption_key_path=os.getenv("PPMF_ENCRYPTION_KEY_PATH"),
        redact_telemetry=os.getenv("PPMF_REDACT_TELEMETRY", "true").lower() == "true"
    )
)
```

## Best Practices

### 1. Always Use Encryption Key Path in Production
```python
# ❌ Bad: Uses temporary key (lost on restart)
PPMFConfig(enabled=True)

# ✅ Good: Persistent key
PPMFConfig(
    enabled=True,
    encryption_key_path="/secure/keys/ppmf.key"
)
```

### 2. Start with Rule-Based, Add LLM If Needed
```python
# Start here
PPMFConfig(enabled=True, use_llm_classifier=False)

# Upgrade if you need better accuracy
PPMFConfig(enabled=True, use_llm_classifier=True)
```

### 3. Adjust Threshold Based on Your Needs
```python
# Fewer false positives (may miss some sensitive data)
PPMFConfig(enabled=True, classification_threshold=0.9)

# Catch more sensitive data (may have false positives)
PPMFConfig(enabled=True, classification_threshold=0.6)
```

### 4. Test with Adversarial Testing Enabled
```python
# In test/staging environment
PPMFConfig(
    enabled=True,
    enable_adversarial_testing=True,
    adversarial_test_interval=10
)
```

## Next Steps

- [API Reference](03_api_reference.md) - Complete API documentation
- [Examples](04_examples.md) - Real-world usage patterns
- [Advanced Topics](05_advanced.md) - Custom rules, key management

