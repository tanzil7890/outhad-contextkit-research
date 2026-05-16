# outhad-contextkit

Production-grade memory system for AI agents. Combines vector storage, knowledge graphs, temporal reasoning, privacy enforcement, and multi-tenant isolation into a unified memory layer.

---

## What It Is

`outhad-contextkit` is a multi-layer memory engine designed to give AI agents persistent, structured, and semantically-rich memory. It goes beyond simple vector search — memories form a weighted graph, decay over time, track user feedback, and can be partitioned by tenant or privacy level.

### Core Capabilities

| Layer | What It Does |
|-------|-------------|
| **Vector Storage** | Semantic similarity search across 16+ backends (Pinecone, Weaviate, Chroma, Qdrant, etc.) |
| **Context Graph (CGL)** | Weighted knowledge graph with typed edges, exponential decay, BFS/PPR retrieval |
| **Graph Memory** | Neo4j entity graph with LLM-extracted relations and vector similarity |
| **Temporal-Causal (TCMGM)** | Timeline events, causal chains, cross-modal retrieval |
| **Privacy Firewall (PPMF)** | AES-256 encryption, rule + LLM classification, adversarial testing |
| **Personalized Retrieval (MSPR)** | Feedback-driven scoring, intent routing, query success tracking |
| **Lifecycle Management** | Decay scheduler, immutable versioning, cold storage (disk/S3) |
| **Multi-Tenancy** | SQLite-backed registry, per-tenant vector collections, full isolation |

---

## Architecture

```
Memory.add(messages)
  ├── Vector Store  ─────────────────── semantic search, fact extraction
  ├── Graph Memory (Neo4j)  ─────────── entity/relation graph
  ├── Context Graph Layer (CGL)  ────── weighted edge graph + decay
  │     ├── IncrementalGraphBuilder    edge synthesis on write
  │     ├── ContextChangeLog           mutation timeline (SQLite)
  │     └── Sinks: Webhook / Kafka / SSE
  ├── TCMGM  ─────────────────────────  temporal + causal + multimodal
  ├── PPMF   ─────────────────────────  privacy classification + encryption
  ├── MSPR   ─────────────────────────  personalized retrieval pipeline
  ├── Lifecycle  ─────────────────────  decay, versioning, cold storage
  └── Tenant Resolver  ───────────────  routing, isolation, migrations
```

### Search Fusion Formula

```
score = α·dense + β·bm25 + γ·graph [+ δ·personal + ε·frequency + ζ·success]

defaults:
  α_dense     = 0.55   # vector similarity
  β_bm25      = 0.15   # lexical BM25
  γ_graph     = 0.30   # graph centrality / PPR
  δ_personal  = 0.0    # user feedback boost (opt-in)
  ε_frequency = 0.0    # log(access_count)   (opt-in)
  ζ_success   = 0.0    # query→memory success rate (opt-in)
```

---

## Installation

```bash
pip install outhad-contextkit
```

For optional backends:

```bash
pip install outhad-contextkit[neo4j]      # Neo4j graph memory
pip install outhad-contextkit[kafka]      # Kafka sink
pip install outhad-contextkit[s3]         # S3 cold storage
pip install outhad-contextkit[all]        # Everything
```

---

## Quick Start

### Basic Memory

```python
from outhad_contextkit.memory import Memory

mem = Memory()

# Add messages
result = mem.add(
    messages=[
        {"role": "user", "content": "I prefer Python over JavaScript."},
        {"role": "assistant", "content": "Noted. I'll use Python examples."},
    ],
    user_id="user-123",
)

# Search
results = mem.search("programming language preference", user_id="user-123")
for r in results:
    print(r["memory"], r["score"])
```

### Async

```python
from outhad_contextkit.memory import AsyncMemory

mem = AsyncMemory()
await mem.add(messages=[...], user_id="user-123")
results = await mem.search("query", user_id="user-123")
```

---

## Configuration

### Full Config Structure

```python
from outhad_contextkit.memory import Memory
from outhad_contextkit.memory.context_graph import ContextGraphConfig
from outhad_contextkit.memory.context_graph.config import (
    DecayConfig,
    EdgeSynthesisConfig,
    LLMEdgeConfig,
    RetrievalConfig,
)

config = {
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4o", "api_key": "..."},
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small", "api_key": "..."},
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {"host": "localhost", "port": 6333},
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
        },
    },
    "context_graph": ContextGraphConfig(
        enabled=True,
        backend="neo4j",           # "networkx" (in-memory) or "neo4j"
        persist_path="./cgl.json", # only for networkx backend
        decay=DecayConfig(
            enabled=True,
            half_life_days=30.0,
            min_edge_weight=0.05,
            min_node_relevance=0.02,
            tick_on_read=True,
        ),
        edges=EdgeSynthesisConfig(
            enable_reply_to=True,
            enable_topic_similar=True,
            enable_document_link=True,
            enable_temporal_next=True,
            topic_min_similarity=0.75,
            reply_to_window_seconds=1800,
            use_llm_inference=False,      # LLM semantic edges (SUPPORTS, CONTRADICTS, etc.)
            llm=LLMEdgeConfig(
                enabled=False,
                daily_token_budget=100_000,
                min_confidence=0.60,
            ),
        ),
        retrieval=RetrievalConfig(
            enabled=True,
            algorithm="bfs",       # "bfs" or "ppr"
            expansion_depth=2,
            max_candidates=50,
            alpha_dense=0.55,
            beta_bm25=0.15,
            gamma_graph=0.30,
        ),
        log_changes=True,
    ),
}

mem = Memory.from_config(config)
```

---

## Feature Modules

### Context Graph Layer (CGL)

Weighted knowledge graph built on top of vector memory. Each memory becomes a node; edges are synthesized automatically on write.

**Edge Types:**

| Type | Meaning |
|------|---------|
| `REPLY_TO` | Sequential conversational reply |
| `TOPIC_SIMILAR` | Semantic similarity above threshold |
| `DOCUMENT_LINK` | Cross-document reference |
| `TEMPORAL_NEXT` | Temporal ordering |
| `UPDATED_FROM` | Version chain (old → new) |
| `CAUSAL` | Cause → Effect |
| `SUPPORTS` | Evidence-based support |
| `CONTRADICTS` | Conflicting claim |
| `REFINES` | Detail / clarification |
| `ELABORATES` | Extension of idea |

**Decay:** Edge weights decay exponentially with configurable half-life. Nodes below relevance floor get archived. `CAUSAL` and `CONTRADICTS` edges are sticky (exempt from decay).

```python
# Manual decay tick
stats = mem.context_graph.tick_decay()
# {"decayed": 12, "pruned": 3, "archived": 1}

# Bump relevance after positive feedback
mem.record_feedback(memory_id="abc", helpful=True, user_id="user-123")
```

### Graph Memory (Neo4j Entity Graph)

LLM-extracted entity/relation graph stored in Neo4j. Separate from CGL — this stores named entities and their typed relations.

```python
# Enabled automatically when graph_store configured
result = mem.add(
    messages=[{"role": "user", "content": "Alice works at Acme Corp."}],
    user_id="user-123",
)
# result["graph"]["added_entities"] → ["Alice", "Acme Corp"]
```

### Temporal-Causal Multimodal (TCMGM)

Timeline events, causal chains, and cross-modal retrieval (text / image / audio).

```python
results = mem.search(
    query="what happened before the outage?",
    user_id="user-123",
    use_tcmgm=True,
    time_window={"start": "2024-01-01", "end": "2024-01-31"},
)
```

### Privacy Firewall (PPMF)

Classification pipeline with rule-based and LLM classifiers. Sensitive memories are AES-256 encrypted before storage.

**Privacy Levels:** `PUBLIC` → `INTERNAL` → `CONFIDENTIAL` → `RESTRICTED`

```python
config = {
    "ppmf": {
        "enabled": True,
        "encryption_key": "...",  # 32-byte base64
        "default_level": "internal",
    }
}
```

### Personalized Retrieval (MSPR)

Feedback-driven retrieval that learns which memories are useful per user.

```python
# Record that memory was helpful
mem.record_feedback(
    memory_id="abc-123",
    helpful=True,
    user_id="user-123",
    query="programming language",
)

# Search uses feedback scores automatically when MSPR enabled
results = mem.search("language preference", user_id="user-123")
```

### Lifecycle Management

Versioned memories with cold storage archival.

```python
# Version history
versions = mem.versions("memory-id")

# Rollback to previous version
mem.rollback("memory-id", to_version=2, actor_id="admin")

# Archive stale memories
mem.archive_low(threshold=0.05, dry_run=True)

# Move to cold storage
mem.demote_to_cold("memory-id")
mem.promote_from_cold("memory-id")

# Background decay scheduler
mem.start_decay_scheduler()
```

### Multi-Tenancy

Per-tenant isolation with routing, migration, and admin operations.

```python
# Add memory to specific tenant
mem.add(
    messages=[...],
    user_id="user-123",
    metadata={"tenant_id": "tenant-a"},
)

# Export tenant data
mem.export_tenant("tenant-a", dest_dir="./exports/tenant-a")

# Migrate between tenants
mem.migrate_tenant(src_id="tenant-a", dst_id="tenant-b")

# Hard delete tenant (irreversible)
mem._tenant_hard_delete("tenant-a")
```

---

## Core API Reference

### Memory

```python
class Memory:
    def add(
        messages: list[dict],
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict = {},
        infer: bool = True,
        memory_type: str = "user",
        prompt: str | None = None,
    ) -> dict

    def search(
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        filters: dict = {},
        limit: int = 10,
        threshold: float = 0.0,
        use_tcmgm: bool = False,
        time_window: dict | None = None,
    ) -> list[dict]

    def get(memory_id: str) -> dict | None
    def get_all(user_id: str | None, ...) -> list[dict]
    def delete(memory_id: str) -> dict
    def delete_all(user_id: str | None, ...) -> dict
    def update(memory_id: str, data: str) -> dict
    def history(memory_id: str) -> list[dict]

    def record_feedback(
        memory_id: str,
        helpful: bool,
        user_id: str | None = None,
        query: str | None = None,
    ) -> None

    def decay_score(memory_id: str) -> float
    def versions(memory_id: str) -> list[dict]
    def rollback(memory_id: str, to_version: int, actor_id: str) -> dict
    def archive_low(threshold: float, dry_run: bool = False) -> dict
    def demote_to_cold(memory_id: str) -> None
    def promote_from_cold(memory_id: str) -> None
    def start_decay_scheduler() -> None
    def stop_decay_scheduler() -> None
```

---

## Supported Backends

### LLM Providers (15+)
OpenAI, Anthropic, Google Gemini, Azure OpenAI, AWS Bedrock, Groq, Together AI, Mistral, Cohere, Ollama, LiteLLM, and more.

### Embedding Providers (17+)
OpenAI, Cohere, Hugging Face, Sentence Transformers, Google Vertex AI, AWS Bedrock, Nomic, FastEmbed, and more.

### Vector Stores (16+)
Pinecone, Qdrant, Weaviate, Chroma, Milvus, PgVector, Redis, MongoDB Atlas, Azure AI Search, OpenSearch, Elasticsearch, and more.

### Graph Databases
Neo4j, Memgraph, AWS Neptune.

---

## Project Structure

```
outhad_contextkit/
├── memory/
│   ├── main.py                    # Memory, AsyncMemory (primary API)
│   ├── graph_memory.py            # Neo4j entity graph
│   ├── storage.py                 # SQLite history store
│   ├── history_store.py           # PostgreSQL history store
│   ├── context_graph/             # Context Graph Layer (CGL)
│   │   ├── facade.py              # ContextGraph orchestrator
│   │   ├── builder.py             # IncrementalGraphBuilder
│   │   ├── config.py              # ContextGraphConfig
│   │   ├── types.py               # MemoryNode, MemoryEdge, ChangeEvent
│   │   ├── changelog.py           # Mutation timeline
│   │   ├── retriever.py           # BFS graph retriever
│   │   ├── ppr_retriever.py       # Personalized PageRank retriever
│   │   ├── scoring.py             # Decay score computation
│   │   ├── llm_edges.py           # LLM semantic edge synthesis
│   │   ├── budget.py              # Daily token budget
│   │   ├── replay.py              # Timeline replay / debugging
│   │   ├── backends/              # NetworkX + Neo4j backends
│   │   └── sinks/                 # Webhook, Kafka, SSE
│   ├── temporal/                  # TCMGM (22 modules)
│   ├── privacy/                   # PPMF (13 modules)
│   ├── personalized/              # MSPR (8 modules)
│   ├── lifecycle/                 # Decay, versioning, cold storage
│   ├── tenant/                    # Multi-tenant registry + routing
│   └── chunking/                  # Adaptive document chunking
├── embeddings/                    # 17+ embedding providers
├── llms/                          # 15+ LLM providers
├── vector_stores/                 # 16+ vector store backends
├── graphs/                        # Graph DB configs (Neptune)
└── client/                        # Sync + async client interfaces
```

---

## Development

```bash
# Clone
git clone https://github.com/outhad-ai/contextkit
cd contextkit

# Install dev deps
pip install -e ".[dev]"

# Run tests
pytest tests/

# Type check
mypy outhad_contextkit/
```

---

## License

See [LICENSE](LICENSE) for details.
