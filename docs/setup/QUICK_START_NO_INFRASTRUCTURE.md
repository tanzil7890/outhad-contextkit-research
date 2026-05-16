# Quick Start Without Infrastructure

Get started with Outhad_ContextKit in **5 minutes** with **no external services**.

---

## What You Get

With just Python and an API key, you can use:
- ✅ Vector memory (in-memory)
- ✅ Semantic search
- ✅ Memory operations (add/search/update/delete)
- ✅ 19 LLM providers
- ✅ PPMF privacy features
- ✅ Adaptive chunking

**What You Need Later**:
- Neo4j for TCMGM features (temporal/causal/graph)
- External vector store for production

---

## 5-Minute Setup

### Step 1: Install Package (30 seconds)

```bash
pip install outhad_contextkitai
```

**Expected Output**:
```
Successfully installed outhad_contextkitai-0.1.115
```

### Step 2: Get API Key (2 minutes)

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click "Create new secret key"
3. Copy the key (starts with `sk-`)

**Alternative LLM Providers**:
- **Anthropic**: [console.anthropic.com](https://console.anthropic.com)
- **Google Gemini**: [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
- **Groq**: [console.groq.com](https://console.groq.com)

### Step 3: Set Environment Variable (30 seconds)

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

**For persistence**, add to `~/.bashrc` or `~/.zshrc`:
```bash
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Windows PowerShell**:
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
```

### Step 4: Write Your First Script (2 minutes)

Create `my_first_memory.py`:

```python
from outhad_contextkit import Memory

# Initialize (uses in-memory vector store)
memory = Memory()

# Add memories
memory.add("I love Python programming", user_id="alice")
memory.add("I enjoy hiking on weekends", user_id="alice")
memory.add("My favorite food is pizza", user_id="alice")

# Search memories
results = memory.search(
    query="What are my hobbies?",
    user_id="alice"
)

# Print results
print("🔍 Search Results:")
for result in results['results']:
    print(f"  - {result['memory']}")
```

### Step 5: Run It!

```bash
python my_first_memory.py
```

**Output**:
```
🔍 Search Results:
  - I enjoy hiking on weekends
  - I love Python programming
```

---

## ✅ You're Done!

You now have a working memory system without any external services.

---

## What's Next?

### Try More Features

**Multi-user support**:
```python
# Different users have separate memories
memory.add("I live in New York", user_id="alice")
memory.add("I live in London", user_id="bob")

# Search only returns alice's memories
results = memory.search("Where do I live?", user_id="alice")
# Returns: "I live in New York"
```

**Update memories**:
```python
# Get all memories
all_memories = memory.get_all(user_id="alice")

# Update a specific memory
memory_id = all_memories['results'][0]['id']
memory.update(memory_id, "I love Python and Rust programming")
```

**Delete memories**:
```python
memory.delete(memory_id)  # Delete specific memory
memory.reset()  # Delete all memories (careful!)
```

### Add More Features

```bash
# Add graph memory (requires Neo4j)
pip install outhad_contextkitai[graph]

# Add more vector stores
pip install outhad_contextkitai[vector_stores]

# Add more LLMs
pip install outhad_contextkitai[llms]

# Add multimodal support
pip install outhad_contextkitai[multimodal]
```

### Setup External Services (When Ready)

- **[Neo4j Setup](./NEO4J_SETUP_GUIDE.md)** - For TCMGM features
- **[Vector Stores](./VECTOR_STORES_SETUP.md)** - For production
- **[Advanced Configuration](./ADVANCED_CONFIG.md)** - Custom setups

---

## Troubleshooting

### Import Error

**Problem**: `ModuleNotFoundError: No module named 'outhad_contextkit'`

**Solution**:
```bash
# Verify installation
pip list | grep outhad

# Reinstall if needed
pip uninstall outhad_contextkitai
pip install outhad_contextkitai
```

### API Key Error

**Problem**: `ValueError: OPENAI_API_KEY not set`

**Solution**:
```bash
# Set the environment variable
export OPENAI_API_KEY="sk-your-key-here"

# Verify it's set
echo $OPENAI_API_KEY | head -c 10
```

### OpenAI Connection Error

**Problem**: `openai.error.AuthenticationError: Invalid API key`

**Solution**:
- Verify your API key is correct
- Check if key has been revoked
- Generate a new key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

---

## Learning Resources

- **[Full Documentation](../README.md)** - Complete feature guide
- **[Examples](../../examples/)** - Real-world use cases
- **[TCMGM User Guides](../TCMGM/)** - Advanced temporal-causal memory
- **[Troubleshooting](../TROUBLESHOOTING.md)** - Common issues

---

## Next Steps

Once you're comfortable with basic memory operations:

1. **Explore TCMGM** - Add temporal reasoning and causal analysis
2. **Production Setup** - Configure external vector stores
3. **Privacy Features** - Enable PPMF for data protection
4. **Customize LLMs** - Switch to your preferred provider

---

