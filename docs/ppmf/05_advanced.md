# Advanced PPMF Topics

## Key Management

### Generating Encryption Keys

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

# Generate Fernet key
fernet_key = Fernet.generate_key()
with open('/secure/keys/ppmf_fernet.key', 'wb') as f:
    f.write(fernet_key)

# Generate AES-256 key
aes_key = AESGCM.generate_key(bit_length=256)
with open('/secure/keys/ppmf_aes.key', 'wb') as f:
    f.write(aes_key)

# Set secure permissions
os.chmod('/secure/keys/ppmf_fernet.key', 0o600)
os.chmod('/secure/keys/ppmf_aes.key', 0o600)
```

### Key Rotation (Manual)

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig
from outhad_contextkit.memory.privacy import EncryptionManager

# 1. Create new key
new_key_path = "/secure/keys/ppmf_new.key"
old_key_path = "/secure/keys/ppmf_old.key"

# 2. Initialize with old key
old_config = PPMFConfig(enabled=True, encryption_key_path=old_key_path)
old_manager = EncryptionManager(old_config)

# 3. Initialize with new key
new_config = PPMFConfig(enabled=True, encryption_key_path=new_key_path)
new_manager = EncryptionManager(new_config)

# 4. Re-encrypt data
def rotate_encrypted_data(encrypted_text, old_metadata):
    # Decrypt with old key
    plaintext = old_manager.decrypt(encrypted_text, old_metadata)
    
    # Encrypt with new key
    new_encrypted, new_metadata = new_manager.encrypt(plaintext)
    
    return new_encrypted, new_metadata

# 5. Update database with new encrypted data
# ... your database update logic ...
```

### Key Storage Best Practices

1. **Use Hardware Security Modules (HSM)** for production
2. **Never commit keys to version control**
3. **Use environment variables or secret managers**
4. **Rotate keys regularly** (90-180 days)
5. **Backup keys securely** with encryption
6. **Implement key versioning** for rotation

Example with AWS Secrets Manager:

```python
import boto3
import json

def get_ppmf_key_from_aws():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='ppmf/encryption-key')
    secret = json.loads(response['SecretString'])
    return secret['key']

# Use in config
key_data = get_ppmf_key_from_aws()
# Save temporarily to file or pass directly
```

---

## Custom Sensitivity Rules

### Creating Domain-Specific Rules

```python
from outhad_contextkit.memory.privacy.config import (
    SensitivityRule,
    get_default_sensitivity_rules
)
from outhad_contextkit.memory.privacy.enums import (
    SensitiveSpanType,
    PrivacyLevel,
    RedactionMode
)

# Medical Record Numbers (MRN)
mrn_rule = SensitivityRule(
    span_type=SensitiveSpanType.PHI_MEDICAL_RECORD,
    privacy_level=PrivacyLevel.RESTRICTED,
    redaction_mode=RedactionMode.ENCRYPT,
    encrypt_at_rest=True,
    patterns=[
        r'MRN[:\s]*\d{7,10}',
        r'Medical\s+Record[:\s]*\d{7,10}'
    ]
)

# Internal Employee IDs
employee_rule = SensitivityRule(
    span_type=SensitiveSpanType.CUSTOM,
    privacy_level=PrivacyLevel.INTERNAL,
    redaction_mode=RedactionMode.HASH,
    encrypt_at_rest=True,
    patterns=[
        r'EMP-\d{6}',
        r'Employee\s+ID[:\s]*\d{6}'
    ]
)

# Customer Account Numbers
account_rule = SensitivityRule(
    span_type=SensitiveSpanType.FINANCIAL_BANK_ACCOUNT,
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    redaction_mode=RedactionMode.ENCRYPT,
    encrypt_at_rest=True,
    patterns=[
        r'ACC-\d{10,12}',
        r'Account[:\s]*\d{10,12}'
    ]
)

# Combine with defaults
all_rules = get_default_sensitivity_rules()
all_rules.extend([mrn_rule, employee_rule, account_rule])

config = PPMFConfig(
    enabled=True,
    sensitivity_rules=all_rules
)
```

### Testing Custom Rules

```python
from outhad_contextkit.memory.privacy import MemorySanitizer

classifier = MemorySanitizer(config)

# Test MRN rule
test_cases = [
    "MRN: 1234567",
    "Medical Record 98765432",
    "Employee ID 123456",
    "Account ACC-1234567890"
]

for test in test_cases:
    result = classifier.classify(test)
    print(f"\nTest: {test}")
    print(f"Sensitive: {result.has_sensitive_content}")
    if result.sensitive_spans:
        for span in result.sensitive_spans:
            print(f"  - {span.text} ({span.span_type})")
```

---

## Custom LLM Classifier

### Using Different LLM Providers

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.llms.configs import LlmConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig

# Use GPT-3.5 for classification (faster, cheaper)
config = MemoryConfig(
    llm=LlmConfig(
        provider="openai",
        config={"model": "gpt-3.5-turbo"}
    ),
    ppmf=PPMFConfig(
        enabled=True,
        use_llm_classifier=True,
        classifier_model="gpt-3.5-turbo"  # Specific for classification
    )
)

memory = Memory(config=config)
```

### Custom Classification Prompt

Currently, the LLM classification prompt is built-in. To customize it, you can extend the LLMClassifier:

```python
from outhad_contextkit.memory.privacy.llm_classifier import LLMClassifier

class CustomLLMClassifier(LLMClassifier):
    def __init__(self, config, llm):
        super().__init__(config, llm)
        
        # Override the classification prompt
        self.custom_prompt = """
        You are a medical data privacy expert.
        
        Analyze this text for HIPAA-protected information:
        {text}
        
        Return JSON with detected PHI...
        """
    
    def classify(self, text):
        # Use custom prompt
        # ... implementation ...
        pass
```

---

## Performance Optimization

### 1. Disable LLM Classification for Speed

```python
config = PPMFConfig(
    enabled=True,
    use_llm_classifier=False  # 1000x faster
)
```

**Impact**: < 1ms vs 500-1000ms

### 2. Enable Caching

```python
config = PPMFConfig(
    enabled=True,
    cache_classifications=True  # Avoid re-classifying same text
)
```

### 3. Async Processing

```python
config = PPMFConfig(
    enabled=True,
    async_processing=True  # Process privacy ops async
)
```

### 4. Batch Processing

```python
# Instead of:
for text in texts:
    result = classifier.classify(text)

# Do:
results = classifier.classify_batch(texts)  # Faster
```

### 5. Profile Performance

```python
import time

config = PPMFConfig(enabled=True, use_llm_classifier=False)
classifier = MemorySanitizer(config)

# Profile classification
start = time.time()
for _ in range(1000):
    classifier.classify("Test email: test@example.com")
elapsed = time.time() - start

print(f"1000 classifications: {elapsed:.2f}s")
print(f"Average: {elapsed/1000*1000:.2f}ms")
```

---

## Integration Patterns

### 1. Middleware Pattern

```python
class PPMFMiddleware:
    """Middleware for automatic PPMF protection."""
    
    def __init__(self, memory):
        self.memory = memory
    
    def add_with_protection(self, text, user_id, **kwargs):
        # Classify before adding
        if self.memory.ppmf_enabled:
            result = self.memory._memory_sanitizer.classify(text)
            
            # Log classification
            print(f"Privacy level: {result.overall_privacy_level}")
            
            # Add metadata
            kwargs['metadata'] = kwargs.get('metadata', {})
            kwargs['metadata']['ppmf_classified'] = True
            kwargs['metadata']['privacy_level'] = result.overall_privacy_level.value
        
        return self.memory.add(text, user_id=user_id, **kwargs)

# Usage
middleware = PPMFMiddleware(memory)
middleware.add_with_protection("Sensitive data", "user_001")
```

### 2. Decorator Pattern

```python
from functools import wraps

def ppmf_protected(func):
    """Decorator to add PPMF protection."""
    
    @wraps(func)
    def wrapper(self, text, *args, **kwargs):
        if hasattr(self, 'memory') and self.memory.ppmf_enabled:
            # Classify
            result = self.memory._memory_sanitizer.classify(text)
            
            # Check privacy level
            if result.overall_privacy_level == PrivacyLevel.RESTRICTED:
                print("⚠️ RESTRICTED data detected")
            
            # Add metadata
            kwargs['ppmf_metadata'] = {
                'privacy_level': result.overall_privacy_level.value,
                'sensitive_spans': len(result.sensitive_spans)
            }
        
        return func(self, text, *args, **kwargs)
    
    return wrapper

class MyService:
    def __init__(self, memory):
        self.memory = memory
    
    @ppmf_protected
    def process_message(self, text, user_id):
        return self.memory.add(text, user_id=user_id)
```

### 3. Observer Pattern

```python
class PrivacyObserver:
    """Observer for privacy events."""
    
    def on_sensitive_detected(self, result):
        print(f"⚠️ Sensitive data detected: {result.overall_privacy_level}")
    
    def on_encryption_complete(self, encrypted_text, metadata):
        print(f"🔒 Data encrypted with {metadata['algorithm']}")
    
    def on_telemetry_sanitized(self, before, after):
        print(f"🛡️ Telemetry sanitized")

# Use in your application
observer = PrivacyObserver()

# Classify and notify
result = classifier.classify(text)
if result.has_sensitive_content:
    observer.on_sensitive_detected(result)
```

---

## Compliance & Auditing

### GDPR Compliance

```python
from outhad_contextkit.memory.privacy.config import PPMFConfig
from outhad_contextkit.memory.privacy.enums import RedactionMode

# GDPR-compliant configuration
gdpr_config = PPMFConfig(
    enabled=True,
    use_llm_classifier=True,
    encryption_algorithm=EncryptionAlgorithm.AES_256_GCM,
    default_redaction_mode=RedactionMode.ENCRYPT,
    redact_telemetry=True,
    classification_threshold=0.85
)

# Right to be forgotten: Delete user data
def gdpr_delete_user_data(memory, user_id):
    # Implementation depends on your vector store
    # This is a placeholder
    memory.delete(user_id=user_id)
```

### HIPAA Compliance

```python
# HIPAA-compliant configuration
hipaa_config = PPMFConfig(
    enabled=True,
    use_llm_classifier=True,
    encryption_algorithm=EncryptionAlgorithm.AES_256_GCM,
    encryption_key_path="/secure/hipaa/ppmf.key",
    default_redaction_mode=RedactionMode.ENCRYPT,
    redact_telemetry=True,
    classification_threshold=0.9,
    sensitivity_rules=[
        # PHI-specific rules
        SensitivityRule(
            span_type=SensitiveSpanType.PHI_MEDICAL_RECORD,
            privacy_level=PrivacyLevel.RESTRICTED,
            redaction_mode=RedactionMode.ENCRYPT,
            encrypt_at_rest=True
        )
    ]
)
```

### Audit Logging

```python
import logging
import json
from datetime import datetime

class PPMFAuditLogger:
    """Audit logger for PPMF operations."""
    
    def __init__(self, log_file="ppmf_audit.log"):
        self.logger = logging.getLogger("ppmf_audit")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_classification(self, text, result, user_id):
        self.logger.info(json.dumps({
            "event": "classification",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "text_length": len(text),
            "has_sensitive": result.has_sensitive_content,
            "privacy_level": result.overall_privacy_level.value,
            "spans_detected": len(result.sensitive_spans)
        }))
    
    def log_encryption(self, user_id, algorithm):
        self.logger.info(json.dumps({
            "event": "encryption",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "algorithm": algorithm
        }))

# Usage
audit = PPMFAuditLogger()

result = classifier.classify(text)
audit.log_classification(text, result, "user_001")

if result.has_sensitive_content:
    encrypted, meta = manager.encrypt(text)
    audit.log_encryption("user_001", meta['algorithm'])
```

---

## Testing & Validation

### Unit Testing PPMF

```python
import pytest
from outhad_contextkit.memory.privacy import MemorySanitizer
from outhad_contextkit.memory.privacy.config import PPMFConfig

class TestPPMF:
    def setup_method(self):
        self.config = PPMFConfig(enabled=True, use_llm_classifier=False)
        self.classifier = MemorySanitizer(self.config)
    
    def test_email_detection(self):
        result = self.classifier.classify("Email: test@example.com")
        assert result.has_sensitive_content
        assert len(result.sensitive_spans) == 1
        assert result.sensitive_spans[0].span_type == SensitiveSpanType.PII_EMAIL
    
    def test_no_false_positives(self):
        result = self.classifier.classify("The weather is nice today")
        assert not result.has_sensitive_content
    
    def test_performance(self):
        import time
        start = time.time()
        for _ in range(100):
            self.classifier.classify("test@example.com")
        elapsed = time.time() - start
        assert elapsed < 1.0  # Should be fast
```

### Integration Testing

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig

def test_ppmf_e2e():
    # Setup
    config = MemoryConfig(ppmf=PPMFConfig(enabled=True))
    memory = Memory(config=config)
    
    # Add sensitive data
    result = memory.add(
        "Email: test@example.com",
        user_id="test_user"
    )
    assert result is not None
    
    # Search
    search_results = memory.search(
        "email",
        user_id="test_user"
    )
    
    # Verify protection
    memories = search_results.get("results", [])
    assert len(memories) > 0
    
    # Check if email is protected in results
    # (Implementation depends on your protection strategy)
```

---

## Troubleshooting

### Debug Mode

```python
import logging

# Enable PPMF debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("outhad_contextkit.memory.privacy")
logger.setLevel(logging.DEBUG)

# Now all PPMF operations will be logged
config = PPMFConfig(enabled=True)
memory = Memory(config=MemoryConfig(ppmf=config))
```

### Common Issues

#### 1. Slow Classification

**Problem**: Classification takes too long

**Solution**:
```python
# Disable LLM classifier
config = PPMFConfig(
    enabled=True,
    use_llm_classifier=False  # Much faster
)
```

#### 2. High False Positive Rate

**Problem**: Too many false positives

**Solution**:
```python
# Increase threshold
config = PPMFConfig(
    enabled=True,
    classification_threshold=0.9  # Higher = fewer false positives
)
```

#### 3. Key Not Found Error

**Problem**: Encryption key file not found

**Solution**:
```python
import os

key_path = "/secure/keys/ppmf.key"
if not os.path.exists(key_path):
    # Generate key
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    os.makedirs(os.path.dirname(key_path), exist_ok=True)
    with open(key_path, 'wb') as f:
        f.write(key)
    os.chmod(key_path, 0o600)
```

---

## Next Steps

- [Getting Started](01_getting_started.md)
- [Configuration Guide](02_configuration.md)
- [API Reference](03_api_reference.md)
- [Examples](04_examples.md)

