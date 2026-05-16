# Troubleshooting Guide

Quick solutions to common Outhad_ContextKit issues.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Import Errors](#import-errors)
3. [Connection Errors](#connection-errors)
4. [API Key Issues](#api-key-issues)
5. [Memory Operations](#memory-operations)
6. [TCMGM-Specific Issues](#tcmgm-specific-issues)
7. [Performance Issues](#performance-issues)
8. [Platform-Specific Issues](#platform-specific-issues)

---

## Installation Issues

### Issue: `pip install` fails

**Symptoms:**
```bash
ERROR: Could not find a version that satisfies the requirement outhad_contextkitai
```

**Causes:**
- Python version too old
- pip outdated
- Network issues

**Solutions:**

1. **Check Python version**:
   ```bash
   python --version  # Should be 3.9+
   ```
   If too old, upgrade:
   ```bash
   # macOS
   brew install python@3.11

   # Ubuntu/Debian
   sudo apt install python3.11
   ```

2. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

3. **Try with explicit version**:
   ```bash
   pip install outhad_contextkitai==0.1.115
   ```

4. **Check network/proxy**:
   ```bash
   pip install --verbose outhad_contextkitai
   ```

### Issue: Dependency conflicts

**Symptoms:**
```bash
ERROR: pip's dependency resolver does not currently take into account all the packages
```

**Solution:**
```bash
# Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/Mac
# or
fresh_env\Scripts\activate  # Windows

# Install outhad_contextkit first (before other packages)
pip install outhad_contextkitai

# Then install other packages
pip install your-other-packages
```

---

## Import Errors

### Issue: `ModuleNotFoundError: No module named 'outhad_contextkit'`

**Symptoms:**
```python
>>> from outhad_contextkit import Memory
ModuleNotFoundError: No module named 'outhad_contextkit'
```

**Causes:**
- Package not installed
- Wrong Python environment
- Typo in import

**Solutions:**

1. **Verify installation**:
   ```bash
   pip list | grep outhad
   ```

2. **Check Python path**:
   ```python
   import sys
   print(sys.path)  # Should include package location
   ```

3. **Reinstall**:
   ```bash
   pip uninstall outhad_contextkitai
   pip install outhad_contextkitai
   ```

### Issue: `ImportError: cannot import name 'Memory'`

**Symptoms:**
```python
ImportError: cannot import name 'Memory' from 'outhad_contextkit'
```

**Cause:** Package version mismatch or corrupted install

**Solution:**
```bash
# Uninstall completely
pip uninstall outhad_contextkitai -y

# Clear cache
pip cache purge

# Reinstall latest
pip install --no-cache-dir outhad_contextkitai
```

---

## Connection Errors

### Issue: Neo4j connection refused

**Symptoms:**
```python
neo4j.exceptions.ServiceUnavailable: Failed to establish connection to bolt://localhost:7687
```

**Causes:**
- Neo4j not running
- Wrong port
- Firewall blocking

**Solutions:**

1. **Check if Neo4j is running**:
   ```bash
   # Desktop: Look for "Active" status

   # Docker:
   docker ps | grep neo4j

   # If not running:
   docker start neo4j-outhad
   ```

2. **Verify port**:
   ```bash
   # Neo4j uses port 7687 by default
   lsof -i :7687  # Should show neo4j process
   ```

3. **Test connection manually**:
   ```bash
   # Using curl
   curl http://localhost:7474
   ```

4. **Check credentials**:
   ```python
   from neo4j import GraphDatabase

   driver = GraphDatabase.driver(
       "bolt://localhost:7687",
       auth=("neo4j", "your-password")
   )
   driver.verify_connectivity()
   ```

### Issue: OpenAI API connection timeout

**Symptoms:**
```python
openai.error.Timeout: Request timed out
```

**Solutions:**

1. **Check internet connection**:
   ```bash
   ping api.openai.com
   ```

2. **Increase timeout**:
   ```python
   from outhad_contextkit import Memory
   from outhad_contextkit.configs.base import MemoryConfig

   config = MemoryConfig(
       llm_config={
           "timeout": 60  # Increase from default 30s
       }
   )
   memory = Memory(config=config)
   ```

3. **Check OpenAI status**:
   - Visit: [status.openai.com](https://status.openai.com)

---

## API Key Issues

### Issue: `OPENAI_API_KEY not set`

**Symptoms:**
```python
ValueError: OPENAI_API_KEY environment variable not found
```

**Solutions:**

1. **Set environment variable**:
   ```bash
   export OPENAI_API_KEY="sk-proj-..."
   ```

2. **Verify it's set**:
   ```bash
   echo $OPENAI_API_KEY | head -c 10  # Should show first 10 chars
   ```

3. **Set in Python directly** (not recommended for production):
   ```python
   import os
   os.environ["OPENAI_API_KEY"] = "sk-proj-..."

   from outhad_contextkit import Memory
   memory = Memory()
   ```

4. **Make persistent**:
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   echo 'export OPENAI_API_KEY="sk-proj-..."' >> ~/.bashrc
   source ~/.bashrc
   ```

### Issue: `Invalid API key`

**Symptoms:**
```python
openai.error.AuthenticationError: Invalid API key
```

**Solutions:**

1. **Verify key format**:
   ```bash
   # Should start with sk-proj- or sk-
   echo $OPENAI_API_KEY | head -c 10
   ```

2. **Generate new key**:
   - Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Delete old key
   - Create new key
   - Update environment variable

3. **Check for spaces**:
   ```bash
   # Remove any trailing spaces
   export OPENAI_API_KEY=$(echo $OPENAI_API_KEY | tr -d ' ')
   ```

---

## Memory Operations

### Issue: Search returns no results

**Symptoms:**
```python
results = memory.search("query", user_id="alice")
# results['results'] is empty
```

**Causes:**
- No memories added yet
- Wrong user_id
- Query doesn't match content

**Solutions:**

1. **Verify memories exist**:
   ```python
   all_memories = memory.get_all(user_id="alice")
   print(f"Found {len(all_memories['results'])} memories")
   ```

2. **Check user_id matches**:
   ```python
   # When adding:
   memory.add("content", user_id="alice")

   # When searching:
   results = memory.search("query", user_id="alice")  # Must match!
   ```

3. **Try broader query**:
   ```python
   # Instead of specific query
   results = memory.search("user information", user_id="alice")
   ```

4. **Lower similarity threshold**:
   ```python
   results = memory.search(
       query="query",
       user_id="alice",
       threshold=0.5  # Lower = more results
   )
   ```

### Issue: `add()` fails silently

**Symptoms:**
```python
memory.add("content", user_id="alice")
# No error, but memory not stored
```

**Causes:**
- Telemetry errors
- Storage backend issues
- Invalid content format

**Solutions:**

1. **Enable logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)

   memory.add("content", user_id="alice")
   # Will show detailed error messages
   ```

2. **Check storage backend**:
   ```python
   # Verify vector store is accessible
   print(memory.vector_store)
   ```

3. **Try simple content first**:
   ```python
   # Start with simple string
   memory.add("test", user_id="alice")

   # Then try complex content
   ```

---

## TCMGM-Specific Issues

### Issue: `use_tcmgm=True` returns standard results

**Symptoms:**
```python
results = memory.search(query, user_id="alice", use_tcmgm=True)
# Results don't have 'fused_ranking' key
```

**Cause:** TCMGM not initialized (Neo4j not configured)

**Solution:**
```python
# Check if TCMGM is enabled
if hasattr(memory, '_tcmgm_enabled') and memory._tcmgm_enabled:
    print("✅ TCMGM enabled")
else:
    print("❌ TCMGM disabled - configure Neo4j")

# To enable, configure graph store
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.graphs.configs import Neo4jConfig, GraphStoreConfig

config = MemoryConfig(
    graph_store=GraphStoreConfig(
        provider="neo4j",
        config=Neo4jConfig(
            url="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )
    )
)
memory = Memory(config=config)
```

### Issue: Timeline queries return empty

**Symptoms:**
```python
from outhad_contextkit.memory.temporal.timeline_queries import TimelineQueries
queries = TimelineQueries(memory._timeline_builder)
events = queries.events_last_n_days("alice", 7)
# Returns empty list
```

**Cause:** No timeline events created

**Solution:**
```python
# Build timeline from conversation
conversation = [
    {"role": "user", "content": "I went to the store"},
    {"role": "assistant", "content": "What did you buy?"},
    {"role": "user", "content": "Milk and bread"}
]

memory.add(conversation, user_id="alice")

# Now timeline queries work
events = queries.events_last_n_days("alice", 7)
```

---

## Performance Issues

### Issue: Slow search queries

**Symptoms:**
```python
# Search takes >5 seconds
results = memory.search(query, user_id="alice")
```

**Solutions:**

1. **Reduce result limit**:
   ```python
   results = memory.search(
       query,
       user_id="alice",
       limit=10  # Instead of 100
   )
   ```

2. **Disable TCMGM for simple queries**:
   ```python
   results = memory.search(
       query,
       user_id="alice",
       use_tcmgm=False  # Much faster
   )
   ```

3. **Disable causal exploration**:
   ```python
   results = memory.search(
       query,
       user_id="alice",
       use_tcmgm=True,
       include_causal=False  # Faster
   )
   ```

4. **Index Neo4j** (if using TCMGM):
   ```cypher
   // In Neo4j Browser
   CREATE INDEX user_id_index FOR (n:Memory) ON (n.user_id);
   CREATE INDEX timestamp_index FOR (n:TemporalEvent) ON (n.timestamp);
   ```

---

## Platform-Specific Issues

### macOS Issues

**Issue:** SSL certificate verification fails

**Solution:**
```bash
# Install certificates
/Applications/Python\ 3.11/Install\ Certificates.command
```

### Windows Issues

**Issue:** Path too long error

**Solution:**
```powershell
# Enable long paths in Windows
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### Linux Issues

**Issue:** Permission denied errors

**Solution:**
```bash
# Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Or install with --user
pip install --user outhad_contextkitai
```

---

## Getting More Help

If your issue isn't listed here:

1. **Check GitHub Issues**: [github.com/outhad/outhad_contextkit/issues](https://github.com/outhad/outhad_contextkit/issues)
2. **Enable Debug Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
3. **Create Minimal Reproduction**:
   ```python
   # Simplest code that shows the issue
   from outhad_contextkit import Memory
   memory = Memory()
   # ... minimal steps to reproduce
   ```
4. **Report Bug**: Include Python version, OS, error message, minimal reproduction

---

## FAQ

### Q: Do I need Neo4j to use Outhad_ContextKit?
**A:** No! Neo4j is only needed for TCMGM features. Basic memory works without it.

### Q: Can I use a different LLM than OpenAI?
**A:** Yes! We support 19 LLM providers. See [LLM Configuration](./llms/).

### Q: How do I migrate from version X to Y?
**A:** See [CHANGELOG.md](../CHANGELOG.md) for breaking changes and migration guides.

### Q: Is my data encrypted?
**A:** Yes, if you enable PPMF. See [Privacy Documentation](./ppmf/).

### Q: Can I use this in production?
**A:** Yes! The package is production-ready. Use external services (Neo4j, vector stores) for best performance.

---

**Last Updated**: December 2, 2025
**Package Version**: 0.1.115
