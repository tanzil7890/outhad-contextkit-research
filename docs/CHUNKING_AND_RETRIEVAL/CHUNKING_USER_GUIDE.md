# Document Chunking - User Guide

## Table of Contents
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

---

## Introduction

### What is Document Chunking?

Document chunking automatically splits long documents into smaller, manageable pieces (chunks) before storing them in the vector database. This solves several problems:

**Problems Solved:**
- ❌ **Token limit errors** - Long documents exceed embedding model limits (e.g., OpenAI: 8191 tokens)
- ❌ **Poor retrieval quality** - Entire long documents retrieved even when only a small section is relevant
- ❌ **Loss of context** - Fine-grained information buried in large text blocks
- ❌ **Inefficient search** - Semantic search struggles with large, multi-topic documents

**Benefits:**
- ✅ **Granular retrieval** - Find exact relevant sections, not entire documents
- ✅ **No token limits** - Process documents of any length
- ✅ **Better search quality** - More precise semantic matching
- ✅ **Automatic merging** - Related chunks automatically combined during retrieval
- ✅ **Context preservation** - Overlapping chunks maintain continuity

### When to Use Chunking

**✅ Use chunking when:**
- Processing documents > 1000 words (research papers, articles, books)
- Building RAG (Retrieval-Augmented Generation) systems
- Indexing long conversations or chat histories
- Working with PDFs, documentation, or knowledge bases
- Need granular fact retrieval from large texts

**❌ Don't use chunking when:**
- All documents are short (< 500 words)
- Need exact full-document retrieval (e.g., storing complete emails)
- Working with already-chunked data (e.g., individual Q&A pairs)

---

## Quick Start

### Installation

```bash
# Install Outhad_ContextKit (if not already installed)
pip install outhad_contextkit

# Install tiktoken for token-based chunking (RECOMMENDED)
pip install tiktoken
```

### Minimal Example

```python
from outhad_contextkit import Memory, MemoryConfig
from outhad_contextkit.memory.chunking import ChunkingConfig

# Enable chunking
config = MemoryConfig(
    chunking=ChunkingConfig(enabled=True)  # Uses smart defaults
)

memory = Memory(config)

# Add long document - automatically chunked!
memory.add(
    "Your very long document here..." * 100,
    user_id="user123",
    infer=False
)

# Search - chunks automatically retrieved and merged!
results = memory.search("your query", user_id="user123")
```

That's it! The system automatically:
1. Splits long documents into 512-token chunks with 50-token overlap
2. Stores each chunk with metadata linking them together
3. Retrieves relevant chunks during search
4. Merges related chunks for coherent results

---

## Configuration

### ChunkingConfig Parameters

```python
from outhad_contextkit.memory.chunking import ChunkingConfig

config = ChunkingConfig(
    # BASIC SETTINGS
    enabled=False,                    # Enable/disable chunking (default: False)
    strategy="token",                 # Chunking strategy: "token", "character", "semantic"
    chunk_size=512,                   # Target chunk size (tokens or characters)
    chunk_overlap=50,                 # Overlap between consecutive chunks

    # TOKEN STRATEGY SETTINGS
    tokenizer_model="gpt-4",          # Model for tiktoken (gpt-4, gpt-3.5-turbo, etc.)

    # CHARACTER STRATEGY SETTINGS
    separators=["\n\n", "\n", ". ", " "],  # Separators for character splitting

    # RETRIEVAL SETTINGS
    merge_chunks_on_retrieval=True,   # Auto-merge related chunks during search
    chunk_context_window=1,           # Include ±N neighboring chunks (0-5)
)
```

### Strategy Comparison

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| **`token`** ⭐ | Production use, RAG systems | Accurate token counting, matches embedding models | Requires tiktoken dependency |
| **`character`** | Development, testing | Fast, no dependencies | Less accurate, may split mid-word |
| **`semantic`** | Research, advanced use | Respects sentence/paragraph boundaries | Slower, variable chunk sizes |

**Recommendation:** Use `strategy="token"` for production systems.

---

## Basic Usage

### Example 1: Research Paper Chunking

```python
from outhad_contextkit import Memory, MemoryConfig
from outhad_contextkit.memory.chunking import ChunkingConfig

# Configure for research papers
config = MemoryConfig(
    chunking=ChunkingConfig(
        enabled=True,
        strategy="token",
        chunk_size=1024,      # Larger chunks for research papers
        chunk_overlap=100,    # More overlap to preserve context
    )
)

memory = Memory(config)

# Load research paper
with open("research_paper.txt", "r") as f:
    paper_text = f.read()

# Add to memory
result = memory.add(
    paper_text,
    user_id="researcher_123",
    metadata={
        "document_type": "research_paper",
        "title": "Temporal-Causal Memory Graphs",
        "authors": "Smith et al.",
        "year": 2024
    },
    infer=False  # Store raw text directly
)

print(f"Created {len(result['results'])} chunks from research paper")

# Search for specific information
results = memory.search(
    query="temporal memory graph architecture",
    user_id="researcher_123",
    limit=5
)

# Display results
for idx, result in enumerate(results['results']):
    print(f"\nResult {idx + 1}:")
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['memory'][:200]}...")
```

### Example 2: Multi-Document Knowledge Base

```python
from outhad_contextkit import Memory, MemoryConfig
from outhad_contextkit.memory.chunking import ChunkingConfig

# Configure for knowledge base
config = MemoryConfig(
    chunking=ChunkingConfig(
        enabled=True,
        strategy="token",
        chunk_size=512,
        chunk_overlap=50,
    )
)

memory = Memory(config)

# Add multiple documents
documents = [
    {"content": open("python_guide.txt").read(), "category": "programming"},
    {"content": open("javascript_guide.txt").read(), "category": "programming"},
    {"content": open("ai_basics.txt").read(), "category": "artificial_intelligence"},
]

for doc in documents:
    memory.add(
        doc["content"],
        user_id="kb_admin",
        metadata={"category": doc["category"]},
        infer=False
    )

# Search across all documents
results = memory.search(
    query="web development best practices",
    user_id="kb_admin",
    limit=10
)
```

### Example 3: Conversation History

```python
from outhad_contextkit import Memory, MemoryConfig
from outhad_contextkit.memory.chunking import ChunkingConfig

# Configure for conversation history
config = MemoryConfig(
    chunking=ChunkingConfig(
        enabled=True,
        strategy="token",
        chunk_size=256,       # Smaller chunks for conversations
        chunk_overlap=30,     # Minimal overlap
    )
)

memory = Memory(config)

# Add long conversation
conversation_history = """
User: Tell me about machine learning.
Assistant: Machine learning is a subset of AI...
User: What are the main types?
Assistant: The three main types are supervised, unsupervised, and reinforcement learning...
""" * 50  # Long conversation

memory.add(
    conversation_history,
    user_id="chatbot_user",
    metadata={"conversation_id": "conv_123", "session": "morning"},
    infer=False
)

# Retrieve relevant conversation parts
results = memory.search(
    query="types of machine learning",
    user_id="chatbot_user"
)
```

---

## Advanced Features

### 1. Context Window Retrieval

Include neighboring chunks for richer context:

```python
# Search with context window
results = memory.search(
    query="neural network training",
    user_id="user123",
    merge_chunks=True,           # Enable merging
    chunk_context_window=2,      # Include ±2 neighboring chunks
    limit=3
)
```

**How it works:**
- If chunk #5 matches your query
- System retrieves chunks #3, #4, **#5**, #6, #7
- All merged into a single coherent result
- Provides broader context around the match

**When to use:**
- Need full context around matched section
- Working with sequential content (books, articles)
- Query requires understanding surrounding information

### 2. Custom Chunk Merging

Control how chunks are merged:

```python
from outhad_contextkit.memory.chunking import ChunkMerger

# Manual merging (advanced use case)
search_results = memory.search(query, merge_chunks=False)  # Get raw chunks

# Custom merge logic
merged = ChunkMerger.merge_by_document(
    search_results['results'],
    include_neighbors=True,
    neighbor_window=3,
    max_merged_length=2000  # Limit merged content length
)
```

### 3. Metadata-Based Filtering

Chunks inherit document metadata:

```python
# Add document with rich metadata
memory.add(
    document_text,
    user_id="user123",
    metadata={
        "document_id": "doc_456",
        "category": "finance",
        "year": 2024,
        "confidential": False
    },
    infer=False
)

# Each chunk gets this metadata
# Search can filter by metadata (if vector store supports it)
results = memory.search(
    query="quarterly revenue",
    user_id="user123",
    filters={"category": "finance", "year": 2024}  # Depends on vector store
)
```

### 4. Chunk Information Access

Access chunk metadata in results:

```python
results = memory.search(query, user_id="user123")

for result in results['results']:
    # Access chunk-specific metadata
    chunk_info = result.get('metadata', {})

    print(f"Document ID: {chunk_info.get('document_id')}")
    print(f"Chunk Index: {chunk_info.get('chunk_index')}")
    print(f"Total Chunks: {chunk_info.get('total_chunks')}")
    print(f"Chunk ID: {chunk_info.get('chunk_id')}")
```

### 5. Performance Tuning

Optimize for your use case:

```python
# For speed (fewer, larger chunks)
config = ChunkingConfig(
    chunk_size=2048,      # Large chunks
    chunk_overlap=100,    # Minimal overlap
)

# For precision (more, smaller chunks)
config = ChunkingConfig(
    chunk_size=256,       # Small chunks
    chunk_overlap=50,     # More overlap
)

# Balanced (recommended)
config = ChunkingConfig(
    chunk_size=512,       # Medium chunks
    chunk_overlap=50,     # Balanced overlap
)
```

---

## Best Practices

### 1. Chunk Size Selection

**General Guidelines:**
- **256 tokens**: Conversations, Q&A pairs, short snippets
- **512 tokens**: ⭐ **Recommended default** - works for most use cases
- **1024 tokens**: Research papers, technical documentation
- **2048 tokens**: Books, long-form content (watch embedding limits!)

**Rule of thumb:** Chunk size should be ~1/4 to 1/2 of your embedding model's max context.

### 2. Overlap Optimization

**Guidelines:**
- **25-50 tokens**: Standard overlap for most use cases
- **50-100 tokens**: Technical content where context is critical
- **10-25 tokens**: Conversations or already-segmented content

**Too little overlap:** May lose context between chunks
**Too much overlap:** Wastes storage and creates redundancy

### 3. Strategy Selection

```python
# ✅ PRODUCTION: Token-based
config = ChunkingConfig(strategy="token")  # Most accurate

# ⚠️ DEVELOPMENT: Character-based
config = ChunkingConfig(strategy="character")  # Fast prototyping

# 🔬 RESEARCH: Semantic
config = ChunkingConfig(strategy="semantic")  # Experimental
```

### 4. Memory Integration

```python
# ✅ GOOD: Store raw text for chunking
memory.add(document, user_id="user", infer=False)

# ❌ BAD: LLM fact extraction interferes with chunking
memory.add(document, user_id="user", infer=True)
```

**Why?** When `infer=True`, the LLM extracts facts and may discard information. For document chunking, store raw text with `infer=False`.

### 5. Retrieval Best Practices

```python
# For factual QA: No merging, more results
results = memory.search(
    query,
    merge_chunks=False,
    limit=10
)

# For conversational context: Merge with context window
results = memory.search(
    query,
    merge_chunks=True,
    chunk_context_window=2,
    limit=3
)
```

---

## Troubleshooting

### Problem: Chunks are too small/large

**Solution:** Adjust `chunk_size`:
```python
# Chunks too small
config = ChunkingConfig(chunk_size=1024)  # Increase

# Chunks too large
config = ChunkingConfig(chunk_size=256)   # Decrease
```

### Problem: Lost context between chunks

**Solution:** Increase overlap or use context window:
```python
# More overlap
config = ChunkingConfig(chunk_overlap=100)

# Or use context window during retrieval
results = memory.search(query, chunk_context_window=2)
```

### Problem: `ModuleNotFoundError: No module named 'tiktoken'`

**Solution:** Install tiktoken:
```bash
pip install tiktoken
```

Or switch to character-based chunking (no dependencies):
```python
config = ChunkingConfig(strategy="character")
```

### Problem: Poor retrieval quality

**Checklist:**
1. ✅ Using `infer=False` when adding documents?
2. ✅ Chunk size appropriate for your content?
3. ✅ Sufficient overlap between chunks?
4. ✅ Using context window during retrieval?
5. ✅ Embedding model matches your use case?

### Problem: Chunking too slow

**Solutions:**
- Use character-based splitting (faster): `strategy="character"`
- Increase chunk size (fewer chunks): `chunk_size=1024`
- Batch process documents asynchronously

### Problem: Chunks split mid-sentence

**Solution:** Use semantic splitting:
```python
config = ChunkingConfig(
    strategy="semantic",
    separators=["\n\n", "\n", ". ", "! ", "? "]  # Sentence boundaries
)
```

---

## API Reference

### ChunkingConfig

```python
class ChunkingConfig(BaseModel):
    enabled: bool = False
    strategy: Literal["token", "character", "semantic"] = "token"
    chunk_size: int = 512
    chunk_overlap: int = 50
    tokenizer_model: str = "gpt-4"
    separators: List[str] = ["\n\n", "\n", ". ", " "]
    merge_chunks_on_retrieval: bool = True
    chunk_context_window: int = 1
```

### Memory.add() with Chunking

```python
result = memory.add(
    messages,                    # Text or messages to add
    user_id="user123",          # Required: user identifier
    metadata={},                # Optional: document metadata
    infer=False,                # Recommended: False for chunking
    filters=None,               # Optional: metadata filters
)

# Returns:
{
    "results": [
        {
            "id": "mem_123",
            "memory": "chunk content",
            "event": "ADD",
            "chunk_index": 0,
            "total_chunks": 5,
            "document_id": "doc_456",
            ...
        },
        ...
    ]
}
```

### Memory.search() with Chunking

```python
results = memory.search(
    query,                          # Search query
    user_id="user123",             # Required: user identifier
    limit=10,                      # Number of results
    merge_chunks=True,             # Merge related chunks
    chunk_context_window=1,        # Include ±N neighbors
    filters=None,                  # Metadata filters
)

# Returns:
{
    "results": [
        {
            "id": "mem_123",
            "memory": "retrieved content",
            "score": 0.85,
            "metadata": {
                "chunk_index": 2,
                "total_chunks": 5,
                "document_id": "doc_456",
                ...
            },
            ...
        },
        ...
    ]
}
```

### ChunkMerger (Advanced)

```python
from outhad_contextkit.memory.chunking import ChunkMerger

merged = ChunkMerger.merge_by_document(
    chunk_list,                    # List of chunk results
    include_neighbors=True,        # Include neighboring chunks
    neighbor_window=2,             # ±N chunks
    max_merged_length=2000,        # Max length after merging
)
```

---

## Examples

Complete example scripts are available:
- `examples/chunking_basic_usage.py` - Basic chunking examples
- `examples/chunking_advanced_usage.py` - Advanced features and performance
- `CHUNKING_QUICKSTART.md` - 5-minute quick start guide
- `CHUNKING_IMPLEMENTATION_GUIDE.md` - Full implementation details

---

## Support

For issues or questions:
- 📖 Read the implementation guide: `CHUNKING_IMPLEMENTATION_GUIDE.md`
- 🚀 Try quick start: `CHUNKING_QUICKSTART.md`
- 🔍 Run examples: `python examples/chunking_basic_usage.py`
- 💬 GitHub Issues: [Report a bug or request a feature]

---

## Summary

**Key Takeaways:**
1. ✅ Chunking enables processing of arbitrarily long documents
2. ✅ Use `strategy="token"` for production (most accurate)
3. ✅ Default settings (`chunk_size=512`, `overlap=50`) work for most use cases
4. ✅ Use `infer=False` when adding documents for chunking
5. ✅ Context windows provide richer retrieval results
6. ✅ Chunking is **opt-in** - existing code continues to work

**Next Steps:**
1. Install tiktoken: `pip install tiktoken`
2. Enable chunking in your config
3. Test with your documents
4. Tune parameters based on your use case
5. Monitor retrieval quality and iterate

Happy chunking! 🚀
