# PPMF Examples

## Basic Usage Examples

### Example 1: Simple Protection

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig

# Enable PPMF
config = MemoryConfig(ppmf=PPMFConfig(enabled=True))
memory = Memory(config=config)

# Add memory with sensitive data
memory.add(
    "Contact me at john.doe@example.com or call 555-123-4567",
    user_id="user_001"
)

# Search - sensitive data is automatically protected
results = memory.search("contact information", user_id="user_001")
print(results)
```

### Example 2: Healthcare Application

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig
from outhad_contextkit.memory.privacy.enums import (
    EncryptionAlgorithm,
    RedactionMode
)

# High-security configuration for healthcare
config = MemoryConfig(
    ppmf=PPMFConfig(
        enabled=True,
        use_llm_classifier=True,  # More accurate for medical terms
        encryption_algorithm=EncryptionAlgorithm.AES_256_GCM,
        encryption_key_path="/secure/keys/healthcare_ppmf.key",
        default_redaction_mode=RedactionMode.ENCRYPT,
        redact_telemetry=True,
        classification_threshold=0.9  # High threshold for accuracy
    )
)

memory = Memory(config=config)

# Store patient information
memory.add(
    "Patient John Doe, MRN: 12345, diagnosed with hypertension. Prescribed lisinopril 10mg.",
    user_id="doctor_001",
    metadata={"patient_id": "P12345"}
)

# All PHI is automatically encrypted
results = memory.search("patient hypertension", user_id="doctor_001")
```

### Example 3: Customer Support Chatbot

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig, SensitivityRule
from outhad_contextkit.memory.privacy.enums import (
    SensitiveSpanType,
    PrivacyLevel,
    RedactionMode
)

# Custom rule for order numbers
order_number_rule = SensitivityRule(
    span_type=SensitiveSpanType.CUSTOM,
    privacy_level=PrivacyLevel.INTERNAL,
    redaction_mode=RedactionMode.HASH,
    encrypt_at_rest=True,
    patterns=[r'ORD-\d{8}']  # Match order numbers like ORD-12345678
)

config = MemoryConfig(
    ppmf=PPMFConfig(
        enabled=True,
        use_llm_classifier=False,  # Fast for real-time chat
        sensitivity_rules=[order_number_rule],
        redact_telemetry=True
    )
)

memory = Memory(config=config)

# Store customer conversation
memory.add(
    "Customer email john@example.com asking about order ORD-12345678",
    user_id="support_agent_001"
)
```

## Direct Component Usage

### Example 4: Classify Text Without Memory

```python
from outhad_contextkit.memory.privacy import MemorySanitizer
from outhad_contextkit.memory.privacy.config import PPMFConfig

# Create standalone classifier
config = PPMFConfig(enabled=True, use_llm_classifier=False)
classifier = MemorySanitizer(config)

# Classify text
text = "My email is test@example.com and phone is 555-987-6543"
result = classifier.classify(text)

print(f"Has sensitive content: {result.has_sensitive_content}")
print(f"Privacy level: {result.overall_privacy_level}")
print(f"Processing time: {result.processing_time_ms}ms")

for span in result.sensitive_spans:
    print(f"\nDetected: '{span.text}'")
    print(f"  Type: {span.span_type}")
    print(f"  Privacy level: {span.privacy_level}")
    print(f"  Position: {span.start}-{span.end}")
```

### Example 5: Encrypt/Decrypt Data

```python
from outhad_contextkit.memory.privacy import EncryptionManager
from outhad_contextkit.memory.privacy.config import PPMFConfig
from outhad_contextkit.memory.privacy.enums import EncryptionAlgorithm

# Create encryption manager
config = PPMFConfig(
    enabled=True,
    encryption_algorithm=EncryptionAlgorithm.AES_256_GCM,
    encryption_key_path="/secure/keys/my_key.bin"
)
manager = EncryptionManager(config)

# Encrypt sensitive data
plaintext = "Credit card: 4111-1111-1111-1111"
encrypted, metadata = manager.encrypt(plaintext)

print(f"Encrypted: {encrypted[:50]}...")
print(f"Algorithm: {metadata['algorithm']}")

# Store encrypted data and metadata in your database
# ... save to DB ...

# Later, decrypt when authorized
decrypted = manager.decrypt(encrypted, metadata)
print(f"Decrypted: {decrypted}")
assert decrypted == plaintext  # ✅
```

### Example 6: Redact Text with Different Modes

```python
from outhad_contextkit.memory.privacy import MemorySanitizer, redact_text
from outhad_contextkit.memory.privacy.config import PPMFConfig
from outhad_contextkit.memory.privacy.enums import RedactionMode

classifier = MemorySanitizer(PPMFConfig(enabled=True))

text = "Email me at john.doe@example.com or call 555-123-4567"
result = classifier.classify(text)

print(f"Original: {text}\n")

# Try all redaction modes
for mode in RedactionMode:
    redacted = redact_text(text, result.sensitive_spans, mode)
    print(f"{mode.value.upper():8} → {redacted}")

# Output:
# MASK     → Email me at *********************** or call ************
# REPLACE  → Email me at [PII_EMAIL] or call [PII_PHONE]
# REMOVE   → Email me at  or call 
# HASH     → Email me at [HASH:a1b2c3d4] or call [HASH:e5f6g7h8]
# ENCRYPT  → (Would encrypt the spans)
```

## Advanced Examples

### Example 7: Custom Sensitivity Rules

```python
from outhad_contextkit.memory.privacy.config import (
    PPMFConfig,
    SensitivityRule,
    get_default_sensitivity_rules
)
from outhad_contextkit.memory.privacy.enums import (
    SensitiveSpanType,
    PrivacyLevel,
    RedactionMode
)

# Start with default rules
rules = get_default_sensitivity_rules()

# Add custom rules for your domain
employee_id_rule = SensitivityRule(
    span_type=SensitiveSpanType.CUSTOM,
    privacy_level=PrivacyLevel.INTERNAL,
    redaction_mode=RedactionMode.HASH,
    encrypt_at_rest=True,
    patterns=[r'EMP-\d{6}']  # Employee IDs
)

contract_id_rule = SensitivityRule(
    span_type=SensitiveSpanType.CUSTOM,
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    redaction_mode=RedactionMode.ENCRYPT,
    encrypt_at_rest=True,
    patterns=[r'CONTRACT-[A-Z]{3}-\d{4}']  # Contract numbers
)

rules.extend([employee_id_rule, contract_id_rule])

# Use custom rules
config = PPMFConfig(
    enabled=True,
    sensitivity_rules=rules
)
```

### Example 8: Telemetry Sanitization

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig

config = MemoryConfig(
    ppmf=PPMFConfig(
        enabled=True,
        redact_telemetry=True
    )
)
memory = Memory(config=config)

# Telemetry is automatically sanitized
event_data = {
    "user_message": "My API key is sk-test123456789",
    "timestamp": "2025-01-15T10:30:00Z",
    "action": "add_memory",
    "nested_data": {
        "email": "admin@example.com",
        "status": "success"
    }
}

sanitized = memory._telemetry_sanitizer.sanitize_event_data(event_data)

# Sensitive data is redacted:
# {
#     "user_message": "My API key is [SECRET_API_KEY]",
#     "timestamp": "2025-01-15T10:30:00Z",
#     "action": "add_memory",
#     "nested_data": {
#         "email": "[PII_EMAIL]",
#         "status": "success"
#     }
# }
```

### Example 9: Batch Classification

```python
from outhad_contextkit.memory.privacy import MemorySanitizer
from outhad_contextkit.memory.privacy.config import PPMFConfig

classifier = MemorySanitizer(PPMFConfig(enabled=True))

# Classify multiple texts at once
texts = [
    "Email: alice@example.com",
    "Phone: 555-111-2222",
    "API key: sk-abc123xyz",
    "Normal text with no sensitive data"
]

results = classifier.classify_batch(texts)

for i, result in enumerate(results):
    print(f"\nText {i+1}: {texts[i]}")
    print(f"  Sensitive: {result.has_sensitive_content}")
    print(f"  Level: {result.overall_privacy_level}")
    print(f"  Spans: {len(result.sensitive_spans)}")
```

### Example 10: Adversarial Testing

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig

config = MemoryConfig(
    ppmf=PPMFConfig(
        enabled=True,
        enable_adversarial_testing=True,
        adversarial_test_interval=10  # Test every 10 operations
    )
)
memory = Memory(config=config)

# Add test data
memory.add("My email is test@example.com", user_id="test_user")

# Adversarial tests run automatically every 10 operations
# Check last test report
if memory._adversarial_runner:
    report = memory._adversarial_runner.last_report
    if report:
        print(f"Adversarial Test Results:")
        print(f"  Total tests: {report['total_tests']}")
        print(f"  Passed: {report['passed']}")
        print(f"  Failed: {report['failed']}")
        print(f"  Leaked attacks: {report['leaked_attacks']}")
```

## Integration Examples

### Example 11: FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig

app = FastAPI()

# Initialize memory with PPMF
memory = Memory(
    config=MemoryConfig(ppmf=PPMFConfig(enabled=True))
)

class MessageRequest(BaseModel):
    text: str
    user_id: str

@app.post("/add-memory")
async def add_memory(request: MessageRequest):
    try:
        result = memory.add(request.text, user_id=request.user_id)
        return {
            "success": True,
            "message": "Memory added with PPMF protection",
            "memories_stored": len(result.get("results", []))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/{user_id}")
async def search(user_id: str, query: str):
    results = memory.search(query=query, user_id=user_id)
    return results
```

### Example 12: Environment-Based Configuration

```python
import os
from dotenv import load_dotenv
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig
from outhad_contextkit.memory.privacy.enums import EncryptionAlgorithm

load_dotenv()

# Configure based on environment
environment = os.getenv("ENVIRONMENT", "development")

if environment == "production":
    ppmf_config = PPMFConfig(
        enabled=True,
        use_llm_classifier=False,  # Fast for production
        encryption_algorithm=EncryptionAlgorithm.AES_256_GCM,
        encryption_key_path=os.getenv("PPMF_KEY_PATH"),
        redact_telemetry=True,
        classification_threshold=0.8
    )
elif environment == "staging":
    ppmf_config = PPMFConfig(
        enabled=True,
        use_llm_classifier=False,
        redact_telemetry=True,
        enable_adversarial_testing=True
    )
else:  # development
    ppmf_config = PPMFConfig(
        enabled=True,
        use_llm_classifier=False,
        redact_telemetry=False,
        enable_adversarial_testing=True,
        adversarial_test_interval=5
    )

memory = Memory(config=MemoryConfig(ppmf=ppmf_config))
print(f"PPMF initialized for {environment} environment")
```

## Error Handling

### Example 13: Handling Classification Errors

```python
from outhad_contextkit.memory.privacy import MemorySanitizer
from outhad_contextkit.memory.privacy.config import PPMFConfig

classifier = MemorySanitizer(PPMFConfig(enabled=True))

try:
    result = classifier.classify("Text to classify")
    print(f"Classification successful: {result.has_sensitive_content}")
except Exception as e:
    print(f"Classification failed: {e}")
    # Handle error appropriately
```

### Example 14: Handling Encryption Errors

```python
from outhad_contextkit.memory.privacy import EncryptionManager
from outhad_contextkit.memory.privacy.config import PPMFConfig

try:
    manager = EncryptionManager(PPMFConfig(enabled=True))
    encrypted, metadata = manager.encrypt("sensitive data")
    
    # If metadata indicates encryption error
    if "encryption_error" in metadata:
        print(f"Encryption failed: {metadata['encryption_error']}")
        # Fall back to alternative protection
    else:
        print("Encryption successful")
        
except Exception as e:
    print(f"Encryption manager error: {e}")
```

## Next Steps

- [Advanced Topics](05_advanced.md) - Key management, custom classifiers, performance tuning
- [Configuration Guide](02_configuration.md) - Detailed configuration options
- [API Reference](03_api_reference.md) - Complete API documentation

