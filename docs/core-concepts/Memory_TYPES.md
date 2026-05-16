# Outhad_ContextKit: Complete Python User Guide

## 🎯 Overview

**Outhad_ContextKit** is an intelligent memory layer for AI applications that provides persistent, personalized memory with layered storage architecture. It combines vector search, graph relationships, and temporal reasoning to create context-aware AI agents that remember user preferences, past interactions, and domain knowledge across sessions.

**Key Benefits:**
- **99.6% Cost Reduction**: Cut LLM costs from $34,425/month to $129/month for 1000 users
- **91% Faster Responses**: Retrieve only relevant context instead of full conversation history
- **36% Better Accuracy**: Outperforms OpenAI Memory on LoCoMo benchmark
- **Enterprise Privacy**: Automatic PII/PHI encryption with PPMF
- **Temporal Reasoning**: TCMGM provides "why" and "when" analysis

---

## 🚀 Quick Start

### 1. Installation

```bash
# Basic installation
pip install outhad_contextkitai

# With graph memory (Neo4j/Memgraph/Neptune)
pip install outhad_contextkitai[graph]

# With all features
pip install outhad_contextkitai[all]
```

### 2. Set API Keys

```bash
# Required: LLM provider (choose one)
export OPENAI_API_KEY="sk-proj-..."
# OR
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: For graph memory features
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"
```

### 3. Basic Usage

```python
import os
from openai import OpenAI
from outhad_contextkit import Memory

# Initialize
openai_client = OpenAI()
memory = Memory()

# Add memories from conversation
messages = [
    {"role": "user", "content": "I'm allergic to peanuts"},
    {"role": "assistant", "content": "I'll remember that!"}
]
memory.add(messages, user_id="alice")

# Search relevant memories
relevant_memories = memory.search(
    query="What are my dietary restrictions?",
    user_id="alice",
    limit=3
)

# Use in LLM context
system_prompt = f"""You are a helpful assistant.
User memories: {relevant_memories['results']}"""

response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What should I avoid eating?"}
    ]
)

print(response.choices[0].message.content)
# Output: "Based on your allergy, you should avoid peanuts..."
```

---

## 📚 Core Concepts: Memory Types

### Understanding Memory Layers

Outhad_ContextKit organizes memory into layers, similar to how humans remember information:
- **Conversation memory**: What was just said (short-term)
- **Session memory**: Context for current task (short-term)
- **User memory**: Personal preferences and history (long-term)
- **Organizational memory**: Shared knowledge across agents (long-term)

### Memory Layer Architecture

```
┌─────────────────────────────────────────┐
│     Conversation Memory (Current Turn)  │
│         ↓                                │
│     Session Memory (Current Task)       │
│         ↓                                │
│     User Memory (Personal History)      │
│         ↓                                │
│     Org Memory (Shared Knowledge)       │
└─────────────────────────────────────────┘
```

### Short-Term vs Long-Term Memory

**Short-Term Memory** (Conversation & Session):
- **Conversation history**: Recent turns in chronological order
- **Working memory**: Tool outputs, intermediate calculations
- **Attention context**: Immediate focus of the assistant

**Long-Term Memory** (User & Org):
- **Factual memory**: User preferences, account details, domain facts
- **Episodic memory**: Summaries of past interactions
- **Semantic memory**: Relationships between concepts

### Memory Flow: Capture → Promote → Retrieve

```python
from outhad_contextkit import Memory

memory = Memory()

# 1. CAPTURE: Messages enter conversation layer
memory.add(
    ["I'm Alex and I prefer boutique hotels."],
    user_id="alex",
    session_id="trip-planning-2025"
)

# 2. PROMOTE: Relevant details persist to user memory
# (Happens automatically based on user_id)

# 3. RETRIEVE: Search pulls from all layers
results = memory.search(
    "Any hotel preferences?",
    user_id="alex",
    session_id="trip-planning-2025"
)
# Returns: User memories first, then session notes, then raw history
```

### When to Use Each Layer

| Layer | Lifetime | Best For | Example |
|-------|----------|----------|---------|
| **Conversation** | Single response | Tool execution details | LLM chain-of-thought |
| **Session** | Minutes to hours | Multi-step tasks | Onboarding flow, debug session |
| **User** | Weeks to forever | Personalization | Preferences, account state |
| **Org** | Configured globally | Shared knowledge | FAQs, product catalog |

### Scoping Memory with Identifiers

```python
# User memory: Persists across all sessions
memory.add(
    "Alice prefers vegetarian food",
    user_id="alice"
)

# Session memory: Expires when task completes
memory.add(
    "Current order: Pizza without meat",
    user_id="alice",
    session_id="order_2025_01"
)

# Agent memory: Shared across users for specific agent
memory.add(
    "Bug found in authentication module",
    agent_id="debug_agent",
    run_id="audit_jan_2025"
)

# Multi-agent shared memory
memory.add(
    "Security vulnerability CVE-2025-1234 fixed",
    agent_id="security_scanner",
    run_id="audit_jan_2025"
)

# Development agent can retrieve security agent's findings
findings = memory.search(
    query="recent security issues",
    run_id="audit_jan_2025"  # Filter by run, any agent
)
```

### Memory Comparison Table

| Layer | Lifetime | Trade-offs | Privacy Considerations |
|-------|----------|------------|------------------------|
| Conversation | Single response | Lost after turn finishes | Minimal - temporary only |
| Session | Minutes-hours | Manual cleanup needed | Moderate - task-scoped |
| User | Weeks-forever | Requires user consent | High - needs governance |
| Org | Configured globally | Owner must keep current | Highest - shared access |

**⚠️ Important**: Never store unencrypted PII, secrets, or sensitive data in user/org memories. Use PPMF for automatic encryption (see Privacy section).

---

## 💾 Memory Operations

### 1. Add Memory

Store conversations, facts, and preferences for later retrieval.

#### Basic Add

```python
from outhad_contextkit import Memory

memory = Memory()

# Add single message
memory.add(
    "I love Italian food",
    user_id="alice"
)

# Add conversation messages
messages = [
    {"role": "user", "content": "I'm planning a trip to Tokyo next month"},
    {"role": "assistant", "content": "Great! I'll remember that for future suggestions"}
]
memory.add(messages, user_id="alice")
```

#### Add with Metadata

```python
# Add with custom metadata for filtering
memory.add(
    "Alice prefers morning meetings",
    user_id="alice",
    metadata={
        "category": "scheduling_preferences",
        "priority": "high",
        "updated_at": "2025-01-14"
    }
)
```

#### Inferred vs Raw Storage

```python
# INFERRED (default): LLM extracts structured facts
result = memory.add(
    messages=[
        {"role": "user", "content": "I prefer sci-fi movies over thrillers"},
        {"role": "assistant", "content": "Got it! I'll suggest sci-fi in the future"}
    ],
    user_id="alice",
    infer=True  # Default
)
# Stores: "Alice prefers sci-fi movies" (extracted fact)

# RAW: Store exact messages without extraction
result = memory.add(
    messages=[
        {"role": "user", "content": "Meeting at 3pm tomorrow"},
        {"role": "assistant", "content": "Reminder set!"}
    ],
    user_id="alice",
    infer=False
)
# Stores: Exact messages as-is
```

**⚠️ Warning**: Mixing `infer=True` and `infer=False` for the same content creates duplicates. Choose one approach per memory type.

#### Add with Session Scoping

```python
# Short-term session memory
memory.add(
    "Current debugging: API timeout in /auth endpoint",
    user_id="dev_team",
    session_id="debug_session_001"
)

# Clear session when done
memory.delete_all(user_id="dev_team", session_id="debug_session_001")
```

#### When to Add Memory

Add memory when:
- ✅ User shares a new preference
- ✅ Decision or suggestion is made
- ✅ Goal or task is completed
- ✅ New entity is introduced
- ✅ User gives feedback or clarification

### 2. Search Memory

Retrieve relevant memories using natural language queries.

#### Basic Search

```python
# Simple search
results = memory.search(
    query="What food does Alice like?",
    user_id="alice"
)

# Print results
for result in results['results']:
    print(f"Memory: {result['memory']}")
    print(f"Score: {result.get('score', 'N/A')}")
    print(f"Metadata: {result.get('metadata', {})}")
    print()
```

#### Search with Filters

```python
# Filter by metadata
results = memory.search(
    query="scheduling preferences",
    user_id="alice",
    filters={"category": "scheduling_preferences"}
)

# Filter by session
session_results = memory.search(
    query="current bug status",
    user_id="dev_team",
    session_id="debug_session_001"
)

# Multi-agent search
agent_results = memory.search(
    query="security vulnerabilities found",
    agent_id="security_scanner",
    run_id="audit_jan_2025"
)
```

#### Search Parameters

```python
# Control result count and quality
results = memory.search(
    query="user preferences",
    user_id="alice",
    limit=5,              # Return top 5 results
    filters={
        "category": "preferences",
        "priority": "high"
    }
)
```

#### Practical Search Example

```python
from openai import OpenAI
from outhad_contextkit import Memory

openai_client = OpenAI()
memory = Memory()

def chat_with_memory(user_message: str, user_id: str) -> str:
    """Chat with memory-augmented context."""

    # 1. Retrieve relevant memories
    memories = memory.search(
        query=user_message,
        user_id=user_id,
        limit=3
    )

    # 2. Build context from memories
    context = "\n".join([
        f"- {m['memory']}"
        for m in memories['results']
    ])

    # 3. Generate response with context
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"User context:\n{context}"
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    # 4. Store new conversation
    memory.add([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response.choices[0].message.content}
    ], user_id=user_id)

    return response.choices[0].message.content

# Usage
print(chat_with_memory("I love pizza", user_id="alice"))
# Later...
print(chat_with_memory("What food do I like?", user_id="alice"))
# Output: "You mentioned you love pizza!"
```

#### Search Best Practices

1. **Always scope with user_id**: Prevents cross-contamination
   ```python
   # Good
   memory.search("preferences", user_id="alice")

   # Bad - returns all users' memories
   memory.search("preferences")
   ```

2. **Use natural language**: Outhad_ContextKit understands intent
   ```python
   # Both work equally well
   memory.search("What are Alice's dietary restrictions?", user_id="alice")
   memory.search("diet allergies food avoid", user_id="alice")
   ```

3. **Combine filters for precision**:
   ```python
   memory.search(
       "recent issues",
       user_id="dev_team",
       session_id="current_sprint",
       filters={"priority": "critical"}
   )
   ```

### 3. Update Memory

Modify existing memories when information changes.

#### Update by Memory ID

```python
# First, get the memory ID from search
results = memory.search("Alice's email", user_id="alice")
memory_id = results['results'][0]['id']

# Update the memory
memory.update(
    memory_id=memory_id,
    data="Alice's new email: alice.new@example.com"
)
```

#### Update with Metadata

```python
memory.update(
    memory_id=memory_id,
    data="Updated preference: Alice prefers morning meetings before 10 AM",
    metadata={"category": "scheduling", "updated_at": "2025-01-15"}
)
```

#### When to Update Memory

Update when:
- ✅ User corrects previous information
- ✅ Preference changes over time
- ✅ More accurate data becomes available
- ✅ Metadata needs enrichment

**Note**: For significant changes, consider deleting and re-adding instead of updating.

### 4. Delete Memory

Remove memories for privacy, compliance, or data cleanup.

#### Delete by Memory ID

```python
# Get memory ID from search
results = memory.search("old preference", user_id="alice")
memory_id = results['results'][0]['id']

# Delete specific memory
memory.delete(memory_id=memory_id)
```

#### Delete All Memories for User

```python
# GDPR/CCPA right to be forgotten
memory.delete_all(user_id="alice")
```

#### Delete by Session

```python
# Clear session after task completion
memory.delete_all(
    user_id="alice",
    session_id="trip_planning_2025"
)
```

#### Delete by Agent/Run

```python
# Clean up agent-specific memories
memory.delete_all(
    agent_id="debug_agent",
    run_id="audit_jan_2025"
)
```

**⚠️ Warning**: `delete_all` requires at least one filter to prevent accidental data loss.

### 5. Get and Get All

Retrieve specific memories without semantic search.

#### Get by Memory ID

```python
# Retrieve specific memory
memory_data = memory.get(memory_id="mem_abc123")
print(memory_data)
```

#### Get All Memories

```python
# Get all memories for a user
all_memories = memory.get_all(user_id="alice")

# Get all memories for a session
session_memories = memory.get_all(
    user_id="alice",
    session_id="trip_planning"
)

# Print all memories
for mem in all_memories['results']:
    print(f"ID: {mem['id']}")
    print(f"Memory: {mem['memory']}")
    print(f"Created: {mem.get('created_at', 'N/A')}")
    print()
```

### 6. Reset

Clear all memories (use with caution).

```python
# Reset entire memory store
memory.reset()
```

**⚠️ Danger**: This deletes ALL memories for ALL users. Use only in development/testing.

---

## ⚙️ Configuration

### Basic Configuration

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig

# Default configuration
config = MemoryConfig()
memory = Memory(config=config)
```

### LLM Provider Configuration

```python
from outhad_contextkit.configs.llms.base import LlmConfig

# OpenAI (default)
config = MemoryConfig(
    llm_provider="openai",
    llm_config=LlmConfig(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=2000
    )
)

# Anthropic Claude
config = MemoryConfig(
    llm_provider="anthropic",
    llm_config=LlmConfig(
        model="claude-3-5-sonnet-20241022",
        temperature=0.1
    )
)

# Google Gemini
config = MemoryConfig(
    llm_provider="gemini",
    llm_config=LlmConfig(
        model="gemini-1.5-pro",
        temperature=0.1
    )
)

# Groq (fast inference)
config = MemoryConfig(
    llm_provider="groq",
    llm_config=LlmConfig(
        model="llama-3.1-8b-instant",
        temperature=0.1
    )
)

# Local with Ollama
config = MemoryConfig(
    llm_provider="ollama",
    llm_config=LlmConfig(
        model="llama3.1:8b",
        base_url="http://localhost:11434"
    )
)

memory = Memory(config=config)
```

### Embedder Configuration

```python
from outhad_contextkit.configs.embeddings.base import EmbeddingConfig

# OpenAI embeddings (default)
config = MemoryConfig(
    embedder_provider="openai",
    embedder_config=EmbeddingConfig(
        model="text-embedding-3-small"
    )
)

# HuggingFace embeddings
config = MemoryConfig(
    embedder_provider="huggingface",
    embedder_config=EmbeddingConfig(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
)

# Ollama local embeddings
config = MemoryConfig(
    embedder_provider="ollama",
    embedder_config=EmbeddingConfig(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
)

memory = Memory(config=config)
```

### Vector Store Configuration

```python
# Qdrant (default, in-memory)
config = MemoryConfig(
    vector_store="qdrant"
)

# Qdrant (persistent)
from outhad_contextkit.configs.vector_stores.qdrant import QdrantConfig

config = MemoryConfig(
    vector_store="qdrant",
    vector_store_config=QdrantConfig(
        url="http://localhost:6333",
        collection_name="memories"
    )
)

# Pinecone
from outhad_contextkit.configs.vector_stores.pinecone import PineconeConfig

config = MemoryConfig(
    vector_store="pinecone",
    vector_store_config=PineconeConfig(
        api_key="your-pinecone-api-key",
        index_name="memories",
        environment="us-east-1-gcp"
    )
)

# ChromaDB
from outhad_contextkit.configs.vector_stores.chroma import ChromaConfig

config = MemoryConfig(
    vector_store="chroma",
    vector_store_config=ChromaConfig(
        path="./chroma_db",
        collection_name="memories"
    )
)

memory = Memory(config=config)
```

### Supported Providers

**LLM Providers (19):**
- OpenAI, Anthropic, Google Gemini, Groq, Together AI
- Ollama (local), LiteLLM, Azure OpenAI, AWS Bedrock
- Vertex AI, LMStudio, Sarvam AI, DeepSeek, xAI Grok, vLLM

**Embedders (10):**
- OpenAI, HuggingFace, Sentence Transformers
- Azure OpenAI, Vertex AI, Ollama
- AWS Bedrock, Google Gemini, Together AI, LangChain

**Vector Stores (17):**
- Qdrant, Pinecone, Weaviate, Chroma, FAISS
- Milvus, PGVector, Azure AI Search, MongoDB
- Upstash, ElasticSearch, OpenSearch, Baidu Mochow
- Redis, Supabase, Vertex AI Vector Search, LangChain

**Graph Stores (3):**
- Neo4j, Memgraph, AWS Neptune

---

## 🧠 Advanced Features

### 1. TCMGM: Temporal-Causal-Multimodal Graph Memory

**Unique to Outhad_ContextKit** - provides temporal reasoning, causal analysis, and multimodal support.

#### Enable TCMGM

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig

config = MemoryConfig(
    graph_store="neo4j",  # Requires Neo4j
    temporal_graph_enabled=True
)
memory = Memory(config=config)
```

#### Temporal Queries

```python
from datetime import datetime, timedelta

# Add timestamped events
memory.add([
    {"role": "user", "content": "I deployed the new API yesterday"},
    {"role": "assistant", "content": "How did it go?"},
    {"role": "user", "content": "Response times doubled - something broke"}
], user_id="dev_team")

# Query with temporal reasoning
timeline = memory.search(
    query="When did the performance issue start?",
    user_id="dev_team"
)
# Response includes timeline visualization and causal analysis
```

#### Causal Analysis

```python
# TCMGM automatically extracts causal relationships
memory.add([
    {"role": "user", "content": "High server load detected"},
    {"role": "user", "content": "Database queries slowed down due to load"},
    {"role": "user", "content": "Request timeout occurred"},
    {"role": "user", "content": "500 error returned to users"}
], user_id="system_monitor")

# Query root cause
results = memory.search(
    query="Why did we get 500 errors?",
    user_id="system_monitor"
)
# Response: Traces causal chain back to "High server load"
```

**See full TCMGM documentation:**
- [Temporal Attributes Guide](./TCMGM/TEMPORAL_ATTRIBUTES_USAGE_GUIDE.md)
- [Causal Relationships Guide](./TCMGM/CASUAL_RELATIONSHIP_USAGE_GUIDE.md)
- [Multimodal Support Guide](./TCMGM/MULTIMODAL_SUPPORT_USAGE_GUIDE.md)

### 2. PPMF: Privacy-Preserving Memory Firewall

Enterprise-grade automatic PII/PHI detection and encryption.

#### Enable PPMF

```python
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig

config = MemoryConfig(
    ppmf=PPMFConfig(
        enabled=True,
        encryption_method="aes-256-gcm",  # or "fernet", "nacl"
        auto_detect_pii=True,
        detect_phi=True,  # Healthcare data
        adversarial_testing=True  # >90% attack resistance
    )
)
memory = Memory(config=config)
```

#### Automatic PII Encryption

```python
# PII is automatically detected and encrypted
memory.add(
    "My SSN is 123-45-6789 and email is john@example.com",
    user_id="alice"
)

# Stored as: "My SSN is [ENCRYPTED:SSN] and email is [ENCRYPTED:EMAIL]"

# Retrieval automatically decrypts (with proper auth)
results = memory.search("What's my email?", user_id="alice")
# Returns: "john@example.com" (decrypted)

# Telemetry sees redacted version
# Analytics: "My SSN is [REDACTED:SSN] and email is [REDACTED:EMAIL]"
```

#### Healthcare Example

```python
# PHI automatically encrypted
memory.add(
    "Patient Alice Smith (DOB: 01/15/1985) diagnosed with Type 2 diabetes. "
    "HbA1c: 7.8%. Prescribed Metformin 500mg.",
    user_id="patient_12345"
)

# Secure retrieval
results = memory.search(
    "diabetes treatment plan",
    user_id="patient_12345"
)
# PHI decrypted only with proper authorization
```

**PPMF Features:**
- ✅ Auto PII/PHI detection (SSN, emails, phone, medical data)
- ✅ AES-256-GCM / Fernet / NaCl encryption at rest
- ✅ Telemetry redaction (prevents analytics leakage)
- ✅ MEXTRA-style adversarial attack resistance (>90%)
- ✅ HIPAA/GDPR compliance-ready

**See full PPMF documentation:** [Privacy Firewall Guide](../ppmf/README.md)

### 3. Adaptive Chunking

Smart document processing that only chunks when needed.

#### Enable Adaptive Chunking

```python
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.chunking import ChunkingConfig

config = MemoryConfig(
    chunking=ChunkingConfig(
        enabled=True,
        strategy="token",  # "token", "character", or "semantic"
        chunk_size=512,
        chunk_overlap=50,
        min_document_size=1500,  # Don't chunk short docs
        merge_chunks_on_retrieval=True
    )
)
memory = Memory(config=config)
```

#### Basic Chunking Example

```python
# Short document - NOT chunked
memory.add(
    "Alice loves Italian food",
    user_id="alice",
    infer=False
)
# Stored as single memory

# Long document - Automatically chunked
with open("research_paper.pdf", "r") as f:
    memory.add(
        f.read(),
        user_id="researcher",
        metadata={"doc_type": "paper"},
        infer=False
    )
# Chunked into 512-token segments with 50-token overlap

# Retrieval automatically merges related chunks
results = memory.search(
    "methodology section",
    user_id="researcher"
)
# Returns merged chunks for coherent context
```

#### Custom Chunking Configuration

```python
config = MemoryConfig(
    chunking=ChunkingConfig(
        enabled=True,
        strategy="semantic",  # Respects sentence boundaries
        chunk_size=1024,  # Larger chunks
        chunk_overlap=100,  # More overlap for better context
        tokenizer_model="gpt-4",
        merge_chunks_on_retrieval=True,
        chunk_context_window=2  # Include ±2 neighboring chunks
    )
)
memory = Memory(config=config)
```

#### Search with Chunk Merging

```python
# Search automatically handles chunks
results = memory.search(
    query="temporal causal relationships in AI",
    user_id="researcher",
    merge_chunks=True,  # Merge related chunks
    chunk_context_window=1  # Include neighboring chunks
)

# Each result may combine multiple related chunks
for result in results['results']:
    print(f"Memory: {result['memory'][:200]}...")
    print(f"Chunk info: {result.get('chunk_index')}/{result.get('total_chunks')}")
```

**Adaptive Chunking Benefits:**
- **88% Performance Recovery**: CoQA benchmark (-0.33% vs -2.02%)
- **Prevents Over-Fragmentation**: Short texts maintain semantic unity
- **Context Preservation**: Configurable overlap between chunks
- **Production-Ready**: 100% test coverage

**See full Chunking documentation:** [Chunking Guide](../CHUNKING_AND_RETRIEVAL/CHUNKING_USER_GUIDE.md)

### 4. Multimodal Support

Handle images, audio, PDFs, and text in unified memory.

#### Enable Multimodal

```python
config = MemoryConfig(
    multimodal_enabled=True
)
memory = Memory(config=config)
```

#### Add Image with Context

```python
# Add image with text description
with open("product_photo.jpg", "rb") as img:
    memory.add([
        {"role": "user", "content": "This is our new product design"},
        {"role": "user", "image": img.read()}
    ], user_id="design_team")
```

#### Search Across Modalities

```python
# Search returns both text and images
results = memory.search(
    query="product design with blue accents",
    user_id="design_team"
)

for result in results['results']:
    print(f"Memory: {result['memory']}")
    if 'image_hash' in result:
        print(f"  Contains image: {result['image_hash']}")
```

---

## 🎯 Common Use Cases

### Use Case 1: Chatbot with Persistent Memory

```python
from openai import OpenAI
from outhad_contextkit import Memory

openai_client = OpenAI()
memory = Memory()

def chat_with_memory(message: str, user_id: str) -> str:
    # 1. Retrieve relevant memories
    memories = memory.search(query=message, user_id=user_id, limit=3)
    context = "\n".join([m['memory'] for m in memories['results']])

    # 2. Generate response with context
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"User context: {context}"},
            {"role": "user", "content": message}
        ]
    )

    # 3. Store conversation
    memory.add([
        {"role": "user", "content": message},
        {"role": "assistant", "content": response.choices[0].message.content}
    ], user_id=user_id)

    return response.choices[0].message.content

# Usage
print(chat_with_memory("I love pizza", user_id="alice"))
# Later...
print(chat_with_memory("What food do I like?", user_id="alice"))
# Output: "You mentioned you love pizza!"
```

### Use Case 2: Multi-Agent System

```python
from outhad_contextkit import Memory

memory = Memory()

# Research agent stores findings
memory.add(
    "Found 3 security vulnerabilities in auth module",
    agent_id="research_agent",
    run_id="audit_2025"
)

# Development agent retrieves findings
findings = memory.search(
    query="security issues",
    run_id="audit_2025"  # All agents in this run
)

# Development agent adds fixes
memory.add(
    "Fixed SQL injection in login endpoint",
    agent_id="dev_agent",
    run_id="audit_2025"
)

# QA agent verifies fixes
verification = memory.search(
    query="security fixes",
    run_id="audit_2025"
)
```

### Use Case 3: Fitness Tracker with Memory

```python
from openai import OpenAI
from outhad_contextkit import Memory

openai_client = OpenAI()
memory = Memory()
USER_ID = "anish"

def store_fitness_log(conversation: list):
    """Store fitness conversation history."""
    memory.add(conversation, user_id=USER_ID)

def fitness_coach(user_input: str) -> str:
    """Memory-aware fitness assistant."""
    # Search relevant memories
    memories = memory.search(user_input, user_id=USER_ID)
    memory_context = "\n".join(f"- {m['memory']}" for m in memories['results'])

    # Generate personalized response
    prompt = f"""You are a fitness assistant for {USER_ID}.

What you remember about {USER_ID}:
{memory_context}

User query: {user_input}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    # Store new interaction
    memory.add(
        f"User: {user_input}\nAssistant: {response.choices[0].message.content}",
        user_id=USER_ID
    )

    return response.choices[0].message.content

# Store user preferences
store_fitness_log([
    {"role": "user", "content": "I'm 26 years old, 5'10\", and weigh 72kg"},
    {"role": "assistant", "content": "Got it - 26, 5'10\", 72kg"},
    {"role": "user", "content": "I follow push-pull-legs, train 5x/week"},
    {"role": "assistant", "content": "Noted - PPL split, 5x/week training"},
    {"role": "user", "content": "I have mild lactose intolerance"},
    {"role": "assistant", "content": "Understood - avoiding regular dairy"}
])

# Query with memory
print(fitness_coach("Suggest a post-workout meal"))
# Uses stored preferences: lactose intolerance, training routine
```

### Use Case 4: Customer Support with Context

```python
from outhad_contextkit import Memory

memory = Memory()

def handle_support_ticket(ticket_text: str, user_id: str):
    # Get customer history
    history = memory.search(
        query=ticket_text,
        user_id=user_id,
        limit=5
    )

    # Check for past issues
    past_issues = [m['memory'] for m in history['results'] if 'issue' in m['memory'].lower()]

    # Generate response context
    context = {
        "customer_history": history['results'],
        "past_issues": past_issues,
        "is_repeat_customer": len(history['results']) > 0
    }

    # Store new ticket
    memory.add(
        f"Support ticket: {ticket_text}",
        user_id=user_id,
        metadata={"type": "support_ticket", "status": "open"}
    )

    return context

# Handle ticket
context = handle_support_ticket(
    "My account is locked",
    user_id="customer_789"
)

if context["is_repeat_customer"]:
    print(f"Returning customer with {len(context['customer_history'])} past interactions")
```

---

## 📋 Best Practices

### 1. Always Scope with User ID

```python
# ✅ Good: Scoped to user
memory.search("preferences", user_id="alice")

# ❌ Bad: Returns all users' memories
memory.search("preferences")
```

### 2. Use Session IDs for Temporary Context

```python
# Short-term task context
memory.add(
    "Debugging login timeout issue",
    user_id="dev_team",
    session_id="debug_20250114"
)

# Clear when done
memory.delete_all(user_id="dev_team", session_id="debug_20250114")
```

### 3. Add Metadata for Better Filtering

```python
memory.add(
    "Alice prefers morning meetings",
    user_id="alice",
    metadata={
        "category": "scheduling",
        "priority": "high",
        "last_updated": "2025-01-14"
    }
)

# Filter by metadata
results = memory.search(
    "scheduling preferences",
    user_id="alice",
    filters={"priority": "high"}
)
```

### 4. Use Infer=True for Conversational Data

```python
# Let LLM extract facts from conversations
memory.add([
    {"role": "user", "content": "I prefer sci-fi over horror movies"},
    {"role": "assistant", "content": "Noted!"}
], user_id="alice", infer=True)  # Extracts: "Alice prefers sci-fi movies"
```

### 5. Use Infer=False for Structured Data

```python
# Store structured data as-is
memory.add(
    "Order #12345: Pizza Margherita, €15.99, delivered 2025-01-14",
    user_id="alice",
    infer=False,
    metadata={"type": "order", "order_id": "12345"}
)
```

### 6. Handle Privacy Carefully

```python
# ✅ Good: Use PPMF for automatic encryption
config = MemoryConfig(
    ppmf=PPMFConfig(enabled=True, auto_detect_pii=True)
)
memory = Memory(config=config)

memory.add("My email is alice@example.com", user_id="alice")
# Automatically encrypted

# ❌ Bad: Store sensitive data without encryption
memory.add("My password is 12345", user_id="alice")  # NEVER DO THIS
```

### 7. Clean Up Old Sessions

```python
from datetime import datetime, timedelta

# Delete sessions older than 7 days
old_session_id = "session_20250107"
memory.delete_all(user_id="alice", session_id=old_session_id)
```

### 8. Monitor Memory Growth

```python
# Check memory count
all_memories = memory.get_all(user_id="alice")
print(f"Total memories: {len(all_memories['results'])}")

# Implement retention policy
if len(all_memories['results']) > 1000:
    # Archive or delete old memories
    pass
```

---

## 🐛 Troubleshooting

### Issue: "No memories found"

```python
# Check if memories exist
all_memories = memory.get_all(user_id="alice")
print(f"Total memories: {len(all_memories['results'])}")

# Verify user_id is correct
results = memory.search("preferences", user_id="alice")  # Must match

# Check if infer=True was used (may need time to process)
import time
memory.add("Test memory", user_id="alice", infer=True)
time.sleep(2)  # Wait for LLM processing
results = memory.search("test", user_id="alice")
```

### Issue: "Memory returns wrong user's data"

```python
# Always filter by user_id
results = memory.search(
    "preferences",
    user_id="alice"  # REQUIRED
)

# Check if session_id is mixed up
results = memory.search(
    "preferences",
    user_id="alice",
    session_id="correct_session"  # Match session
)
```

### Issue: "LLM extraction not working"

```python
# Verify API key is set
import os
print(os.getenv("OPENAI_API_KEY"))  # Should not be None

# Try with infer=False to bypass LLM
memory.add(
    "Direct storage test",
    user_id="alice",
    infer=False  # Skip LLM processing
)
```

### Issue: "Search returns irrelevant results"

```python
# Use more specific queries
# ❌ Bad: Too generic
memory.search("preferences", user_id="alice")

# ✅ Good: Specific query
memory.search("What food does Alice prefer for breakfast?", user_id="alice")

# Add metadata for filtering
memory.add(
    "Alice likes oatmeal for breakfast",
    user_id="alice",
    metadata={"category": "food_preferences", "meal": "breakfast"}
)

results = memory.search(
    "breakfast preferences",
    user_id="alice",
    filters={"meal": "breakfast"}
)
```

### Issue: "Vector store connection failed"

```python
# Check if vector store is running (for external stores)
# For Qdrant
import requests
try:
    response = requests.get("http://localhost:6333")
    print("Qdrant is running")
except:
    print("Qdrant not accessible")

# Use in-memory mode (default)
config = MemoryConfig(
    vector_store="qdrant"  # In-memory by default
)
memory = Memory(config=config)
```

### Issue: "Graph memory not working"

```python
# Verify Neo4j is running
from neo4j import GraphDatabase

try:
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )
    driver.verify_connectivity()
    print("Neo4j connected")
except Exception as e:
    print(f"Neo4j error: {e}")

# Check environment variables
import os
print(os.getenv("NEO4J_URI"))
print(os.getenv("NEO4J_USERNAME"))
print(os.getenv("NEO4J_PASSWORD"))
```

---

## 📊 Performance Tips

### 1. Batch Operations

```python
# Add multiple memories efficiently
messages_batch = [
    {"role": "user", "content": f"Message {i}"}
    for i in range(10)
]

memory.add(messages_batch, user_id="alice", infer=False)
```

### 2. Limit Search Results

```python
# Only retrieve what you need
results = memory.search(
    "preferences",
    user_id="alice",
    limit=3  # Top 3 results only
)
```

### 3. Use In-Memory Vector Store for Development

```python
# Fastest for testing
config = MemoryConfig(
    vector_store="qdrant"  # In-memory by default
)
memory = Memory(config=config)
```

### 4. Cache Embeddings (External Vector Stores)

```python
# Use persistent vector store to avoid re-embedding
from outhad_contextkit.configs.vector_stores.qdrant import QdrantConfig

config = MemoryConfig(
    vector_store="qdrant",
    vector_store_config=QdrantConfig(
        url="http://localhost:6333",
        collection_name="cached_memories"
    )
)
memory = Memory(config=config)
```

### 5. Use Faster LLM for Development

```python
# Groq is 10x faster than GPT-4
config = MemoryConfig(
    llm_provider="groq",
    llm_config=LlmConfig(model="llama-3.1-8b-instant")
)
memory = Memory(config=config)
```

---

## 📚 API Reference

### Memory Class

```python
class Memory:
    def __init__(self, config: MemoryConfig = None):
        """Initialize Memory with optional configuration."""

    def add(
        self,
        messages: Union[str, List[Dict], List[str]],
        user_id: str,
        session_id: str = None,
        agent_id: str = None,
        run_id: str = None,
        metadata: Dict = None,
        infer: bool = True
    ) -> Dict:
        """Add memories from messages."""

    def search(
        self,
        query: str,
        user_id: str,
        session_id: str = None,
        agent_id: str = None,
        run_id: str = None,
        limit: int = 5,
        filters: Dict = None
    ) -> Dict:
        """Search memories with semantic similarity."""

    def get(self, memory_id: str) -> Dict:
        """Get specific memory by ID."""

    def get_all(
        self,
        user_id: str = None,
        session_id: str = None,
        agent_id: str = None,
        run_id: str = None
    ) -> Dict:
        """Get all memories matching filters."""

    def update(
        self,
        memory_id: str,
        data: str,
        metadata: Dict = None
    ) -> Dict:
        """Update existing memory."""

    def delete(self, memory_id: str) -> None:
        """Delete specific memory."""

    def delete_all(
        self,
        user_id: str = None,
        session_id: str = None,
        agent_id: str = None,
        run_id: str = None
    ) -> None:
        """Delete all memories matching filters."""

    def reset(self) -> None:
        """Delete ALL memories (use with caution)."""
```

### MemoryConfig

```python
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.configs.llms.base import LlmConfig
from outhad_contextkit.configs.embeddings.base import EmbeddingConfig

config = MemoryConfig(
    # LLM settings
    llm_provider="openai",  # openai, anthropic, gemini, groq, etc.
    llm_config=LlmConfig(model="gpt-4o-mini", temperature=0.1),

    # Embedder settings
    embedder_provider="openai",  # openai, huggingface, ollama, etc.
    embedder_config=EmbeddingConfig(model="text-embedding-3-small"),

    # Vector store settings
    vector_store="qdrant",  # qdrant, pinecone, chroma, weaviate, etc.
    vector_store_config=None,  # Provider-specific config

    # Graph store settings (optional)
    graph_store="neo4j",  # neo4j, memgraph, neptune
    temporal_graph_enabled=False,

    # Privacy settings
    ppmf=None,  # PPMFConfig for PII encryption

    # Chunking settings
    chunking=None  # ChunkingConfig for document chunking
)
```

---

## 🔗 Additional Resources

### Documentation
- [TCMGM Guide](./TCMGM/) - Temporal-Causal-Multimodal Graph Memory
- [PPMF Guide](../ppmf/README.md) - Privacy-Preserving Memory Firewall
- [Chunking Guide](../CHUNKING_AND_RETRIEVAL/CHUNKING_USER_GUIDE.md) - Adaptive Document Chunking
- [Setup Guide](../setup/README.md) - Production deployment
- [Troubleshooting](../TROUBLESHOOTING.md) - Common issues

### Examples
- [Fitness Tracker Example](../../examples/misc/fitness_checker.py)
- [Multi-Agent Example](../../examples/multiagents/llamaindex_learning_system.py)
- [Chunking Examples](../../examples/chunking_basic_usage.py)

### External Links
- [GitHub Repository](https://github.com/outhad/outhad_contextkit)
- [PyPI Package](https://pypi.org/project/outhad_contextkitai/)
- [Research Paper](https://github.com/outhad/outhad_contextkit#citation)

---

## 📝 Quick Reference

### Essential Commands

```python
# Import
from outhad_contextkit import Memory

# Initialize
memory = Memory()

# Add
memory.add("User preference", user_id="alice")

# Search
results = memory.search("query", user_id="alice")

# Update
memory.update(memory_id="id", data="new data")

# Delete
memory.delete(memory_id="id")
memory.delete_all(user_id="alice")

# Get
memory.get(memory_id="id")
memory.get_all(user_id="alice")
```

### Common Patterns

```python
# Conversation memory
memory.add(messages, user_id="alice", infer=True)

# Session memory
memory.add(data, user_id="alice", session_id="session_1")

# Agent memory
memory.add(data, agent_id="agent_1", run_id="run_1")

# With metadata
memory.add(data, user_id="alice", metadata={"category": "prefs"})

# Filtered search
memory.search("query", user_id="alice", filters={"category": "prefs"})
```

---

**Need help?** Check [Troubleshooting](../TROUBLESHOOTING.md) or open an issue on [GitHub](https://github.com/outhad/outhad_contextkit/issues).
