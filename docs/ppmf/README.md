# PPMF Documentation

Welcome to the **Privacy-Preserving Memory Firewall (PPMF)** documentation! This is your complete guide to using PPMF as a built-in library feature.

## 📚 Documentation Structure

### [1. Getting Started](01_getting_started.md) ⭐
**Start here!** Learn what PPMF is, how it works, and enable it in one line of code.

**Topics**:
- What is PPMF?
- Quick start (< 5 minutes)
- What gets protected?
- Performance metrics
- Configuration levels

**Perfect for**: First-time users, quick setup

---

### [2. Configuration Guide](02_configuration.md) ⚙️
Comprehensive guide to all PPMF configuration options.

**Topics**:
- All configuration parameters
- Configuration presets (dev, prod, high-security)
- Environment variables
- Best practices

**Perfect for**: Customizing PPMF for your needs

---

### [3. API Reference](03_api_reference.md) 📖
Complete API documentation for all PPMF classes and methods.

**Topics**:
- Core classes (PPMFConfig, MemorySanitizer, EncryptionManager)
- All methods with parameters and return types
- Enums (PrivacyLevel, SensitiveSpanType, RedactionMode)
- Data models

**Perfect for**: Understanding the API, IDE integration

---

### [4. Examples](04_examples.md) 💡
Real-world code examples for common use cases.

**Topics**:
- Basic usage examples
- Healthcare, customer support, finance applications
- Direct component usage
- Advanced examples
- Integration examples (FastAPI, etc.)

**Perfect for**: Learning by example, copy-paste solutions

---

### [5. Advanced Topics](05_advanced.md) 🚀
Deep dive into advanced features and patterns.

**Topics**:
- Key management and rotation
- Custom sensitivity rules
- Performance optimization
- Integration patterns
- Compliance (GDPR, HIPAA)
- Audit logging
- Testing and troubleshooting

**Perfect for**: Production deployments, enterprise use

---

## 🚀 Quick Start

### Enable PPMF (One Line!)

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig

# Enable PPMF
config = MemoryConfig(ppmf=PPMFConfig(enabled=True))
memory = Memory(config=config)

# Use normally - PPMF protects automatically!
memory.add("My email is john@example.com", user_id="user_001")
```

That's it! PPMF automatically:
- ✅ Detects sensitive data (PII, PHI, secrets)
- ✅ Encrypts sensitive information
- ✅ Protects telemetry
- ✅ Works transparently

---

## 📖 Documentation by Use Case

### I want to...

#### Protect email addresses and phone numbers
→ [Getting Started](01_getting_started.md) - Basic protection

#### Configure encryption algorithm
→ [Configuration Guide](02_configuration.md#encryption-settings)

#### Add custom sensitivity rules for my domain
→ [Advanced Topics](05_advanced.md#custom-sensitivity-rules)

#### Build a healthcare application
→ [Examples](04_examples.md#example-2-healthcare-application)

#### Integrate with FastAPI
→ [Examples](04_examples.md#example-11-fastapi-integration)

#### Optimize performance
→ [Advanced Topics](05_advanced.md#performance-optimization)

#### Understand all configuration options
→ [Configuration Guide](02_configuration.md)

#### See all available API methods
→ [API Reference](03_api_reference.md)

#### Implement GDPR/HIPAA compliance
→ [Advanced Topics](05_advanced.md#compliance--auditing)

#### Test PPMF
→ [Advanced Topics](05_advanced.md#testing--validation)

#### Troubleshoot issues
→ [Advanced Topics](05_advanced.md#troubleshooting)

---

## 🎯 Features

### Automatic Detection
- **PII**: Email, phone, SSN, address, name, DOB
- **PHI**: Medical records, diagnoses, medications, insurance
- **Secrets**: API keys, passwords, tokens, private keys
- **Financial**: Credit cards, bank accounts, routing numbers

### Industry-Standard Encryption
- **Fernet**: Symmetric encryption (default, fast)
- **AES-256-GCM**: AEAD encryption
- **NaCl**: High-performance encryption

### Flexible Redaction
- **MASK**: Replace with asterisks (`***`)
- **REPLACE**: Type labels (`[EMAIL]`)
- **ENCRYPT**: Full encryption
- **REMOVE**: Delete entirely
- **HASH**: One-way hash

### Additional Features
- ✅ Telemetry protection
- ✅ Adversarial testing (MEXTRA-style)
- ✅ Custom sensitivity rules
- ✅ Batch processing
- ✅ Caching for performance
- ✅ Async processing

---

## 📊 Performance

- **Rule-based classification**: < 1ms
- **Encryption/Decryption**: 1-3ms
- **Overall overhead**: < 5% (with rule-based)
- **Memory overhead**: Minimal

---

## 🏗️ Architecture

```
Your Application
    ↓
Memory.add()
    ↓
PPMF (Transparent)
    ├─ 1. Classify Sensitivity
    │  ├─ Rule-based (fast)
    │  └─ LLM-based (accurate)
    ├─ 2. Encrypt/Redact
    │  ├─ Fernet/AES-256-GCM
    │  └─ Multiple redaction modes
    └─ 3. Protect Telemetry
    ↓
Protected Storage
```

---

## 💻 Installation

PPMF is built into Outhad_ContextKit:

```bash
pip install outhad_contextkitai
```

Dependencies (automatically installed):
- `cryptography>=41.0.0`
- `pynacl>=1.5.0`

---

## 🔧 Configuration Presets

### Development (Fast)
```python
PPMFConfig(
    enabled=True,
    use_llm_classifier=False,  # Fast
    redact_telemetry=False,
    enable_adversarial_testing=True
)
```

### Production (Balanced)
```python
PPMFConfig(
    enabled=True,
    use_llm_classifier=False,
    encryption_key_path="/secure/keys/ppmf.key",
    redact_telemetry=True
)
```

### High Security
```python
PPMFConfig(
    enabled=True,
    use_llm_classifier=True,
    encryption_algorithm=EncryptionAlgorithm.AES_256_GCM,
    encryption_key_path="/secure/keys/ppmf.key",
    classification_threshold=0.9
)
```

---

## 🤝 Support

- **GitHub Issues**: [Report bugs](https://github.com/outhad/outhad_contextkit/issues)
- **Discussions**: [Ask questions](https://github.com/outhad/outhad_contextkit/discussions)
- **Email**: support@outhad_contextkit.ai

---

## 📝 License

Same as Outhad_ContextKit main project.

---

## 🎓 Learning Path

1. ✅ **Start**: [Getting Started](01_getting_started.md) (5 min)
2. ✅ **Configure**: [Configuration Guide](02_configuration.md) (15 min)
3. ✅ **Practice**: [Examples](04_examples.md) (30 min)
4. ✅ **Reference**: [API Reference](03_api_reference.md) (as needed)
5. ✅ **Master**: [Advanced Topics](05_advanced.md) (1 hour)

**Total Learning Time**: ~2 hours to become proficient

---

## ✨ Quick Reference

### Enable PPMF
```python
config = MemoryConfig(ppmf=PPMFConfig(enabled=True))
```

### Classify Text
```python
result = memory._memory_sanitizer.classify(text)
```

### Encrypt Data
```python
encrypted, metadata = memory._encryption_manager.encrypt(text)
```

### Redact Text
```python
from outhad_contextkit.memory.privacy import redact_text
redacted = redact_text(text, spans, RedactionMode.REPLACE)
```

---

**Ready to get started?** → [Getting Started Guide](01_getting_started.md)

---

**Version**: 1.0.0  
**Last Updated**: November 13, 2025  
**Status**: Production Ready ✅

