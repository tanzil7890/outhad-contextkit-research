# Getting Started with PPMF

## What is PPMF?

The **Privacy-Preserving Memory Firewall (PPMF)** is a built-in security feature of Outhad_ContextKit that automatically protects sensitive information in your AI memory system.

## Features

- ✅ **Automatic Detection**: Identifies PII, PHI, secrets, and financial data
- ✅ **Encryption**: Industry-standard encryption (Fernet, AES-256-GCM)
- ✅ **Flexible Redaction**: Multiple modes (MASK, REPLACE, REMOVE, HASH)
- ✅ **Telemetry Protection**: Prevents sensitive data leaks in analytics
- ✅ **Adversarial Testing**: Built-in MEXTRA-style testing
- ✅ **Zero Breaking Changes**: Disabled by default, enable with one line

## Installation

PPMF is included with Outhad_ContextKit. Just install the library:

```bash
pip install outhad_contextkitai
```

The required dependencies (cryptography, pynacl) are automatically installed.

## Quick Start

### Enable PPMF (One Line!)

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig

# Enable PPMF - that's it!
config = MemoryConfig(ppmf=PPMFConfig(enabled=True))
memory = Memory(config=config)
```

### Use Normally

```python
# Add memory with sensitive data
memory.add(
    "My email is john.doe@example.com and phone is 555-123-4567",
    user_id="user_001"
)

# PPMF automatically protects sensitive information!
# Search works normally
results = memory.search("contact information", user_id="user_001")
```

## How It Works

```
Your Code → Memory.add() → PPMF → Protected Storage
                           ↓
                    1. Classify Sensitivity
                    2. Encrypt/Redact
                    3. Store Securely
```

PPMF runs **transparently** - no changes to your existing code required!

## Configuration Levels

### Level 1: Basic Protection (Fast)
```python
# Rule-based classification only
config = MemoryConfig(
    ppmf=PPMFConfig(
        enabled=True,
        use_llm_classifier=False  # Fast, regex-based
    )
)
```

**Use for**: High-throughput applications, real-time systems

### Level 2: Advanced Protection (Accurate)
```python
# Rule-based + LLM classification
config = MemoryConfig(
    ppmf=PPMFConfig(
        enabled=True,
        use_llm_classifier=True  # Slower but more accurate
    )
)
```

**Use for**: High-security applications, complex sensitive data

### Level 3: Maximum Protection
```python
from outhad_contextkit.memory.privacy.enums import (
    EncryptionAlgorithm,
    RedactionMode
)

config = MemoryConfig(
    ppmf=PPMFConfig(
        enabled=True,
        use_llm_classifier=True,
        encryption_algorithm=EncryptionAlgorithm.AES_256_GCM,
        default_redaction_mode=RedactionMode.ENCRYPT,
        redact_telemetry=True,
        enable_adversarial_testing=True
    )
)
```

**Use for**: Healthcare, finance, government applications

## What Gets Protected?

### PII (Personally Identifiable Information)
- ✅ Email addresses
- ✅ Phone numbers
- ✅ Social Security Numbers
- ✅ Physical addresses
- ✅ Names
- ✅ Dates of birth

### PHI (Protected Health Information)
- ✅ Medical record numbers
- ✅ Diagnoses
- ✅ Medications
- ✅ Insurance information

### Secrets & Credentials
- ✅ API keys (OpenAI, Google, AWS, etc.)
- ✅ Passwords
- ✅ Tokens
- ✅ Private keys

### Financial Information
- ✅ Credit card numbers
- ✅ Bank account numbers
- ✅ Routing numbers

## Performance

- **Rule-based classification**: < 1ms
- **Encryption/Decryption**: 1-3ms
- **Overall overhead**: < 5% with rule-based
- **Memory overhead**: Minimal

## Next Steps

- [Configuration Guide](02_configuration.md) - Detailed configuration options
- [API Reference](03_api_reference.md) - Complete API documentation
- [Examples](04_examples.md) - Real-world usage examples
- [Advanced Topics](05_advanced.md) - Custom rules, key management, etc.

## Need Help?

- 📖 [Full Documentation](../README.md)
- 🐛 [GitHub Issues](https://github.com/outhad/outhad_contextkit/issues)
- 💬 [Discussions](https://github.com/outhad/outhad_contextkit/discussions)

