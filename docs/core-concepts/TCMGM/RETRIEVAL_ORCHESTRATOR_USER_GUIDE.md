#  TCMGM: Retrieval Orchestrator - Usage Guide

## 🎯 Overview

 implements **TCMGM Fused Retrieval** - an intelligent orchestration layer that combines multiple search modalities into a unified, powerful retrieval system. Instead of relying on a single search method, the Retrieval Orchestrator intelligently fuses results from:

1. **Vector Search** - Semantic similarity using embeddings
2. **Graph Search** - Entity relationships and knowledge graph queries
3. **Timeline Search** - Temporal event retrieval with time-based filtering
4. **Causal Chain Exploration** - Cause-effect reasoning and relationship tracking
5. **Cross-Modal Search** - Framework for multimodal content retrieval

**Key Benefits:**
- **Comprehensive Results**: Combines semantic, structural, and temporal information
- **Intelligent Ranking**: Weighted scoring ensures best results surface first
- **Flexible Configuration**: Enable/disable features based on your needs
- **Backward Compatible**: Existing code continues to work unchanged
- **Production Ready**: Robust error handling and graceful degradation

---

## 🚀 Quick Start

### 1. Import Components

```python
from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.graphs.configs import Neo4jConfig, GraphStoreConfig
from datetime import datetime, timedelta
import os
```

### 2. Initialize Memory with TCMGM

```python
# Configure with Neo4j graph store
config = MemoryConfig(
    graph_store=GraphStoreConfig(
        provider="neo4j",
        config=Neo4jConfig(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database="neo4j"
        )
    )
)

# Initialize Memory (TCMGM auto-initializes when graph is enabled)
memory = Memory(config=config)

# Verify TCMGM is enabled
if memory._tcmgm_enabled:
    print("✅ TCMGM Retrieval Orchestrator initialized!")
else:
    print("⚠️ TCMGM not enabled (graph storage required)")
```

### 3. Your First TCMGM Search

```python
# Add some memories
memory.add(
    "I love hiking in the mountains during summer.",
    user_id="user123"
)
memory.add(
    "My favorite programming language is Python.",
    user_id="user123"
)

# Standard search (vector only)
standard_results = memory.search(
    query="What do I like?",
    user_id="user123",
    use_tcmgm=False  # Standard vector search
)

# TCMGM fused search (all sources)
fused_results = memory.search(
    query="What do I like?",
    user_id="user123",
    use_tcmgm=True,  # Enable TCMGM
    include_causal=True
)

# Access different result sources
print(f"Vector results: {len(fused_results['vector_results'])}")
print(f"Graph results: {len(fused_results['graph_results'])}")
print(f"Timeline results: {len(fused_results['timeline_results'])}")
print(f"Fused ranking: {len(fused_results['fused_ranking'])}")
```

---

## 📖 Core Features

### Feature 1: Fused Multi-Source Retrieval

The Retrieval Orchestrator combines results from multiple sources and intelligently ranks them.

#### Basic Fused Search

```python
from outhad_contextkit import Memory

memory = Memory(config_with_graph_enabled)

# Add diverse content
memory.add("I enjoy reading science fiction novels", user_id="user123")
memory.add("Yesterday I went to the bookstore", user_id="user123")
memory.add("My favorite author is Isaac Asimov", user_id="user123")

# Perform fused search
results = memory.search(
    query="books and reading",
    user_id="user123",
    use_tcmgm=True,
    limit=10
)

# Results structure
print(results.keys())
# ['vector_results', 'graph_results', 'timeline_results',
#  'causal_chains', 'multimodal_results', 'fused_ranking']
```

#### Understanding Result Structure

```python
# Examine vector results (semantic similarity)
for result in results['vector_results'][:3]:
    print(f"[Vector] Score: {result['score']:.3f}")
    print(f"Content: {result['content']}")
    print(f"Metadata: {result.get('metadata', {})}")
    print()

# Examine graph results (entity relationships)
for result in results['graph_results']:
    print(f"[Graph] Score: {result['score']:.3f}")
    print(f"Entity: {result['content']}")
    print()

# Examine timeline results (temporal events)
for result in results['timeline_results']:
    print(f"[Timeline] Score: {result['score']:.3f}")
    print(f"Event: {result['content']}")
    print(f"Time: {result.get('timestamp', 'N/A')}")
    print()

# Examine fused ranking (combined and weighted)
print("\nFused Ranking (Best Results):")
for i, result in enumerate(results['fused_ranking'][:5], 1):
    print(f"{i}. [{result['final_score']:.3f}] ({result['source']}) {result['content'][:60]}...")
```

#### Weighted Scoring

The orchestrator uses weighted scoring to rank results:

```python
# Default weights
WEIGHTS = {
    'vector': 0.4,   # 40% - Semantic similarity
    'graph': 0.3,    # 30% - Entity relationships
    'timeline': 0.3  # 30% - Temporal relevance
}

# Final score calculation
# final_score = original_score × weight

# Example:
# Vector result with score 0.95: 0.95 × 0.4 = 0.38
# Graph result with score 0.80: 0.80 × 0.3 = 0.24
# Timeline result with score 0.85: 0.85 × 0.3 = 0.255
```

---

### Feature 2: Time-Bounded Retrieval

Search within specific time windows to focus on recent or historical events.

#### Search Last N Days

```python
from datetime import datetime, timedelta

# Get events from last 7 days
now = datetime.utcnow()
time_window = {
    "start": (now - timedelta(days=7)).isoformat(),
    "end": now.isoformat()
}

results = memory.search(
    query="recent activities",
    user_id="user123",
    use_tcmgm=True,
    time_window=time_window,
    limit=10
)

print(f"Timeline results in last 7 days: {len(results['timeline_results'])}")
```

#### Search Specific Date Range

```python
# Search specific period (e.g., January 2025)
start_date = datetime(2025, 1, 1, 0, 0, 0)
end_date = datetime(2025, 1, 31, 23, 59, 59)

january_results = memory.search(
    query="what happened in January",
    user_id="user123",
    use_tcmgm=True,
    time_window={
        "start": start_date.isoformat(),
        "end": end_date.isoformat()
    }
)

print(f"Events in January: {len(january_results['timeline_results'])}")
```

#### Recent Events Only

```python
# Last 24 hours
time_window = {
    "start": (datetime.utcnow() - timedelta(hours=24)).isoformat(),
    "end": datetime.utcnow().isoformat()
}

recent = memory.search(
    query="today's activities",
    user_id="user123",
    use_tcmgm=True,
    time_window=time_window
)
```

---

### Feature 3: Causal Chain Exploration

Automatically explore cause-effect relationships in search results.

#### Enable Causal Exploration

```python
# Add memories with causal relationships
memory.add(
    "I started learning Python programming",
    user_id="user123"
)
memory.add(
    "Learning Python enabled me to build web applications",
    user_id="user123"
)
memory.add(
    "I built a portfolio website using Python",
    user_id="user123"
)

# Search with causal exploration
results = memory.search(
    query="Python programming",
    user_id="user123",
    use_tcmgm=True,
    include_causal=True  # Enable causal chain exploration
)

# Examine causal chains
print(f"Found {len(results['causal_chains'])} causal chains")

for i, chain in enumerate(results['causal_chains'], 1):
    print(f"\nCausal Chain {i}:")
    print(f"  Seed event: {chain['seed_event']['name']}")

    if chain['forward_chain']:
        print(f"  Forward effects ({len(chain['forward_chain'])} events):")
        for event in chain['forward_chain']:
            print(f"    → {event['name']}")

    if chain['backward_chain']:
        print(f"  Backward causes ({len(chain['backward_chain'])} events):")
        for event in chain['backward_chain']:
            print(f"    ← {event['name']}")
```

#### Disable Causal Exploration (Faster)

```python
# Skip causal exploration for faster queries
results = memory.search(
    query="simple query",
    user_id="user123",
    use_tcmgm=True,
    include_causal=False  # Disable causal exploration
)

# No causal chains in results
assert len(results['causal_chains']) == 0
print("✅ Faster search without causal exploration")
```

---

### Feature 4: Direct Orchestrator Access

Access the Retrieval Orchestrator directly for advanced use cases.

#### Basic Direct Access

```python
# Access orchestrator instance
orchestrator = memory._retrieval_orchestrator

# Perform fused search directly
results = orchestrator.fused_search(
    query="What are my interests?",
    user_id="user123",
    top_k=20,  # Get top 20 from each source
    include_causal=True,
    include_multimodal=True
)

print(f"Direct orchestrator results:")
print(f"  Vector: {len(results['vector_results'])}")
print(f"  Graph: {len(results['graph_results'])}")
print(f"  Timeline: {len(results['timeline_results'])}")
print(f"  Fused: {len(results['fused_ranking'])}")
```

#### Custom Configuration

```python
# Advanced configuration for specific use cases
custom_results = orchestrator.fused_search(
    query="complex query",
    user_id="user123",
    top_k=50,  # More results per source
    time_window={
        "start": (datetime.utcnow() - timedelta(days=30)).isoformat(),
        "end": datetime.utcnow().isoformat()
    },
    include_causal=True,
    include_multimodal=False  # Skip multimodal for performance
)
```

---

### Feature 5: Backward Compatible Standard Search

TCMGM is fully backward compatible - existing code continues to work.

#### Standard Search Still Works

```python
# Old code continues to work unchanged
results = memory.search(
    query="search query",
    user_id="user123",
    limit=10
)

# Returns standard format
print(results.keys())  # ['results']

# No TCMGM components
assert 'vector_results' not in results
assert 'fused_ranking' not in results
```

#### Graceful Degradation

```python
# TCMGM flag is safely ignored without graph storage
config_no_graph = MemoryConfig()  # No graph store
memory = Memory(config=config_no_graph)

# TCMGM is disabled
assert memory._tcmgm_enabled == False

# use_tcmgm=True falls back to standard search
results = memory.search(
    query="test",
    user_id="user123",
    use_tcmgm=True  # Safely ignored
)

# Returns standard results
assert 'results' in results
print("✅ Graceful fallback to standard search")
```

---

## 🎯 Common Use Cases

### Use Case 1: Comprehensive User Profile Search

Find all relevant information about a user across different data sources.

```python
# Add diverse user information
user = "alice_123"

memory.add("Alice loves mountain biking and rock climbing", user_id=user)
memory.add("Alice works as a software engineer at TechCorp", user_id=user)
memory.add("Alice recently completed a marathon", user_id=user)
memory.add("Alice is learning Spanish in her free time", user_id=user)

# Comprehensive profile search
profile = memory.search(
    query="Tell me about Alice",
    user_id=user,
    use_tcmgm=True,
    include_causal=True,
    limit=20
)

# Get holistic view
print("Profile Information:")
print(f"  Semantic matches: {len(profile['vector_results'])}")
print(f"  Entity relationships: {len(profile['graph_results'])}")
print(f"  Timeline events: {len(profile['timeline_results'])}")
print(f"  Causal connections: {len(profile['causal_chains'])}")

# Best consolidated results
print("\nTop Profile Facts:")
for result in profile['fused_ranking'][:10]:
    print(f"  • {result['content']}")
```

### Use Case 2: Recent Activity Analysis

Analyze user activity over a specific time period.

```python
# Analyze last week's activity
last_week = {
    "start": (datetime.utcnow() - timedelta(days=7)).isoformat(),
    "end": datetime.utcnow().isoformat()
}

activity = memory.search(
    query="user activity and events",
    user_id="user123",
    use_tcmgm=True,
    time_window=last_week,
    include_causal=True,
    limit=50
)

# Analyze activity patterns
timeline_events = activity['timeline_results']
print(f"Activity Summary (Last 7 Days):")
print(f"  Total events: {len(timeline_events)}")

# Group by day
from collections import defaultdict
events_by_day = defaultdict(list)

for event in timeline_events:
    timestamp = event.get('timestamp', '')
    if timestamp:
        day = timestamp.split('T')[0]
        events_by_day[day].append(event['content'])

for day, events in sorted(events_by_day.items()):
    print(f"\n  {day}: {len(events)} events")
    for event in events[:3]:
        print(f"    - {event[:60]}...")
```

### Use Case 3: Context-Aware Question Answering

Use fused retrieval for better question answering with full context.

```python
# Add knowledge about a topic
topic_user = "learner_456"

memory.add(
    "Machine learning is a subset of artificial intelligence",
    user_id=topic_user
)
memory.add(
    "Neural networks are inspired by biological neurons",
    user_id=topic_user
)
memory.add(
    "Deep learning uses multiple layers of neural networks",
    user_id=topic_user
)
memory.add(
    "I started learning ML in 2024 and built my first model",
    user_id=topic_user
)

# Answer question with full context
answer_data = memory.search(
    query="Explain machine learning and my experience with it",
    user_id=topic_user,
    use_tcmgm=True,
    include_causal=True,
    limit=15
)

# Combine all sources for comprehensive answer
context = []

# Add semantic matches
for result in answer_data['vector_results'][:5]:
    context.append(result['content'])

# Add timeline events
for event in answer_data['timeline_results']:
    context.append(f"[{event.get('timestamp', 'Past')}] {event['content']}")

# Generate answer using combined context
print("Answer Context:")
for i, ctx in enumerate(context, 1):
    print(f"{i}. {ctx}")
```

### Use Case 4: Project History Tracking

Track project progress with timeline and causal relationships.

```python
# Add project events
project_user = "dev_789"

memory.add(
    "Started new e-commerce project using React and Node.js",
    user_id=project_user
)
memory.add(
    "Implemented user authentication system",
    user_id=project_user
)
memory.add(
    "Authentication enabled secure payment processing",
    user_id=project_user
)
memory.add(
    "Deployed to production after successful testing",
    user_id=project_user
)

# Get project history with causal connections
project_history = memory.search(
    query="e-commerce project development",
    user_id=project_user,
    use_tcmgm=True,
    include_causal=True,
    limit=20
)

# Display timeline
print("Project Timeline:")
for event in project_history['timeline_results']:
    print(f"  {event.get('timestamp', 'N/A')}: {event['content']}")

# Display causal chains
print("\nProject Causal Flow:")
for chain in project_history['causal_chains']:
    print(f"\nSeed: {chain['seed_event']['name']}")
    if chain['forward_chain']:
        print("Effects:")
        for event in chain['forward_chain']:
            print(f"  → {event['name']}")
```

### Use Case 5: Customer Support Context

Get comprehensive customer history for better support.

```python
# Add customer interactions
customer = "customer_abc"

memory.add("Customer reported login issues on 2025-01-10", user_id=customer)
memory.add("Support team reset password and verified email", user_id=customer)
memory.add("Customer successfully logged in after reset", user_id=customer)
memory.add("Customer purchased premium subscription", user_id=customer)
memory.add("Customer enjoys the analytics dashboard feature", user_id=customer)

# Get comprehensive customer context
customer_context = memory.search(
    query="customer history and current status",
    user_id=customer,
    use_tcmgm=True,
    include_causal=True,
    limit=30
)

# Support agent view
print("Customer Support Context:")
print(f"\n📊 Data Sources:")
print(f"  Vector matches: {len(customer_context['vector_results'])}")
print(f"  Graph entities: {len(customer_context['graph_results'])}")
print(f"  Timeline events: {len(customer_context['timeline_results'])}")
print(f"  Causal chains: {len(customer_context['causal_chains'])}")

print(f"\n🎯 Top Relevant Information:")
for i, result in enumerate(customer_context['fused_ranking'][:5], 1):
    print(f"{i}. [{result['source']}] {result['content']}")
```

### Use Case 6: Learning Path Recommendation

Track learning progress and recommend next steps.

```python
# Add learning activities
student = "student_xyz"

memory.add("Completed Python basics course", user_id=student)
memory.add("Built first web scraper project", user_id=student)
memory.add("Learning web scraping enabled data analysis skills", user_id=student)
memory.add("Started pandas and numpy libraries", user_id=student)
memory.add("Interested in machine learning", user_id=student)

# Get learning context for recommendations
learning_context = memory.search(
    query="programming skills and learning interests",
    user_id=student,
    use_tcmgm=True,
    include_causal=True,
    limit=25
)

# Analyze learning path
print("Learning Progress Analysis:")

# Current skills (from vector search)
print("\n📚 Current Skills:")
for result in learning_context['vector_results'][:5]:
    print(f"  • {result['content']}")

# Learning timeline
print("\n📅 Learning Timeline:")
for event in learning_context['timeline_results']:
    print(f"  {event.get('timestamp', 'N/A')}: {event['content']}")

# Skill dependencies (from causal chains)
print("\n🔗 Skill Dependencies:")
for chain in learning_context['causal_chains']:
    if chain['forward_chain']:
        print(f"  {chain['seed_event']['name']} enabled:")
        for effect in chain['forward_chain'][:3]:
            print(f"    → {effect['name']}")
```

---

## 🎓 Advanced Features

### Custom Weighted Scoring

While the orchestrator uses default weights, you can analyze and adjust based on your use case.

```python
# Get fused results
results = memory.search(
    query="test query",
    user_id="user123",
    use_tcmgm=True
)

# Analyze score distribution
print("Score Analysis:")
for result in results['fused_ranking'][:10]:
    print(f"Source: {result['source']:10s} | "
          f"Score: {result['score']:.3f} | "
          f"Weight: {result['weight']:.2f} | "
          f"Final: {result['final_score']:.3f}")

# Custom post-processing (adjust weights if needed)
def apply_custom_weights(results, custom_weights):
    """Re-rank results with custom weights."""
    reranked = []

    for result in results:
        source = result['source']
        weight = custom_weights.get(source, 0.33)
        new_final_score = result['score'] * weight

        reranked.append({
            **result,
            'weight': weight,
            'final_score': new_final_score
        })

    return sorted(reranked, key=lambda x: x['final_score'], reverse=True)

# Example: Prioritize timeline over graph
custom_weights = {
    'vector': 0.4,
    'graph': 0.2,
    'timeline': 0.4
}

reranked = apply_custom_weights(results['fused_ranking'], custom_weights)
```

### Selective Source Querying

While you can't disable individual sources (yet), you can filter results post-search.

```python
# Get all results
results = memory.search(
    query="search query",
    user_id="user123",
    use_tcmgm=True
)

# Use only vector and timeline (exclude graph)
filtered_results = []

for result in results['fused_ranking']:
    if result['source'] in ['vector', 'timeline']:
        filtered_results.append(result)

print(f"Filtered to {len(filtered_results)} results")
```

### Result Deduplication Analysis

The orchestrator automatically deduplicates results by content.

```python
# Search might return duplicates from different sources
results = memory.search(
    query="test",
    user_id="user123",
    use_tcmgm=True
)

# Check for duplicates
all_results = (
    results['vector_results'] +
    results['graph_results'] +
    results['timeline_results']
)

contents = [r['content'] for r in all_results]
unique_contents = set(contents)

print(f"Total results: {len(all_results)}")
print(f"Unique results: {len(unique_contents)}")
print(f"Fused ranking: {len(results['fused_ranking'])}")
print(f"Duplicates removed: {len(all_results) - len(results['fused_ranking'])}")
```

### Combining with Standard Search

Compare TCMGM vs standard search for specific queries.

```python
query = "programming languages"
user = "user123"

# Standard search
standard = memory.search(
    query=query,
    user_id=user,
    use_tcmgm=False,
    limit=10
)

# TCMGM search
tcmgm = memory.search(
    query=query,
    user_id=user,
    use_tcmgm=True,
    limit=10
)

# Compare results
print("Comparison:")
print(f"Standard results: {len(standard['results'])}")
print(f"TCMGM fused results: {len(tcmgm['fused_ranking'])}")
print(f"TCMGM total sources: {len(tcmgm['vector_results']) + len(tcmgm['graph_results']) + len(tcmgm['timeline_results'])}")
```

---

## 🔍 Best Practices

### 1. Choose the Right Search Mode

**Use Standard Search** (`use_tcmgm=False`) when:
- You only need semantic similarity
- Speed is critical
- You don't have graph storage enabled
- Query is simple and straightforward

```python
# Fast semantic search
results = memory.search(
    query="simple query",
    user_id="user123",
    use_tcmgm=False
)
```

**Use TCMGM Search** (`use_tcmgm=True`) when:
- You need comprehensive results
- Context matters (relationships, timeline)
- Query is complex or multi-faceted
- Understanding "why" and "when" matters

```python
# Comprehensive search
results = memory.search(
    query="complex contextual query",
    user_id="user123",
    use_tcmgm=True,
    include_causal=True
)
```

### 2. Use Time Windows Strategically

```python
# ✅ GOOD: Specific time window for recent queries
results = memory.search(
    query="recent events",
    user_id="user123",
    use_tcmgm=True,
    time_window={
        "start": (datetime.utcnow() - timedelta(days=7)).isoformat(),
        "end": datetime.utcnow().isoformat()
    }
)

# ❌ AVOID: Time window for historical queries
results = memory.search(
    query="all time favorite movies",  # Historical query
    user_id="user123",
    use_tcmgm=True,
    time_window=last_week  # Wrong! Filters out old data
)
```

### 3. Toggle Causal Exploration Based on Need

```python
# ✅ Enable for relationship-heavy queries
results = memory.search(
    query="what caused the project delay",
    user_id="user123",
    use_tcmgm=True,
    include_causal=True  # Needed for causality
)

# ✅ Disable for simple lookups
results = memory.search(
    query="user's email address",
    user_id="user123",
    use_tcmgm=True,
    include_causal=False  # Faster, not needed
)
```

### 4. Handle Empty Results Gracefully

```python
results = memory.search(
    query="query",
    user_id="user123",
    use_tcmgm=True
)

# Check each source
if not results['fused_ranking']:
    print("No results found")

    # Diagnose which sources returned nothing
    if not results['vector_results']:
        print("  No semantic matches")
    if not results['graph_results']:
        print("  No entity matches")
    if not results['timeline_results']:
        print("  No timeline events")
else:
    print(f"Found {len(results['fused_ranking'])} results")
```

### 5. Use Appropriate Limits

```python
# ✅ Reasonable limits for performance
results = memory.search(
    query="query",
    user_id="user123",
    use_tcmgm=True,
    limit=20  # Good balance
)

# ⚠️ Very large limits can slow down search
results = memory.search(
    query="query",
    user_id="user123",
    use_tcmgm=True,
    limit=1000  # Potentially slow
)
```

### 6. Verify TCMGM Initialization

```python
# Always check if TCMGM is enabled before using advanced features
if not memory._tcmgm_enabled:
    print("⚠️ TCMGM not available (requires graph storage)")
    print("Falling back to standard search")
    results = memory.search(query, user_id=user, use_tcmgm=False)
else:
    print("✅ TCMGM available")
    results = memory.search(query, user_id=user, use_tcmgm=True)
```

---

## 📊 Performance Tips

### Optimize Query Performance

```python
# ✅ FAST: Disable unnecessary features
results = memory.search(
    query="simple query",
    user_id="user123",
    use_tcmgm=True,
    include_causal=False,  # Skip causal exploration
    limit=10  # Lower limit
)

# ❌ SLOW: All features enabled
results = memory.search(
    query="simple query",
    user_id="user123",
    use_tcmgm=True,
    include_causal=True,  # +100-300ms
    limit=100  # Large limit
)
```

### Batch Multiple Queries

```python
# Process multiple queries efficiently
queries = [
    "user interests",
    "recent activities",
    "programming skills"
]

results_batch = []

for query in queries:
    result = memory.search(
        query=query,
        user_id="user123",
        use_tcmgm=True,
        include_causal=False,  # Faster
        limit=10
    )
    results_batch.append(result)

print(f"Processed {len(results_batch)} queries")
```

### Cache Frequently Accessed Results

```python
from functools import lru_cache
from datetime import datetime

# Cache search results for repeated queries
@lru_cache(maxsize=100)
def cached_search(query, user_id, cache_key):
    """Cache search results for 5 minutes."""
    return memory.search(
        query=query,
        user_id=user_id,
        use_tcmgm=True
    )

# Generate cache key that changes every 5 minutes
cache_key = datetime.utcnow().replace(second=0, microsecond=0)
cache_key = cache_key - timedelta(minutes=cache_key.minute % 5)

results = cached_search("query", "user123", cache_key.isoformat())
```

### Monitor Query Performance

```python
import time

# Measure query performance
start = time.time()

results = memory.search(
    query="test query",
    user_id="user123",
    use_tcmgm=True,
    include_causal=True
)

elapsed = time.time() - start

print(f"Query took {elapsed*1000:.0f}ms")
print(f"  Vector: {len(results['vector_results'])} results")
print(f"  Graph: {len(results['graph_results'])} results")
print(f"  Timeline: {len(results['timeline_results'])} results")
print(f"  Causal: {len(results['causal_chains'])} chains")

# Performance benchmarks:
# Standard search: ~200-500ms
# TCMGM (no causal): ~500-800ms
# TCMGM (with causal): ~800-1200ms
```

---

## 🐛 Troubleshooting

### Issue: TCMGM Not Initializing

**Problem:** `memory._tcmgm_enabled` is `False`

**Solutions:**

```python
# Check 1: Verify graph storage is configured
config = MemoryConfig(
    graph_store=GraphStoreConfig(  # Must be configured!
        provider="neo4j",
        config=Neo4jConfig(
            url="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )
    )
)
memory = Memory(config=config)

# Check 2: Verify Neo4j connection
if not memory.graph:
    print("❌ Neo4j connection failed")
    print("Check: URI, username, password, Neo4j running")
else:
    print("✅ Neo4j connected")

# Check 3: Verify TCMGM components
if memory._tcmgm_enabled:
    print("✅ TCMGM enabled")
    print(f"  Timeline builder: {memory._timeline_builder is not None}")
    print(f"  Orchestrator: {memory._retrieval_orchestrator is not None}")
else:
    print("❌ TCMGM disabled")
```

### Issue: Empty TCMGM Results

**Problem:** All result sources return empty arrays

**Solutions:**

```python
# Check 1: Verify memories exist
results = memory.search(
    query="test",
    user_id="user123",
    use_tcmgm=False  # Standard search first
)

if not results['results']:
    print("❌ No memories found for user")
    print("Add memories first:")
    memory.add("Test memory", user_id="user123")

# Check 2: Verify query is relevant
results = memory.search(
    query="completely unrelated query",
    user_id="user123",
    use_tcmgm=True
)

# Try broader query
results = memory.search(
    query="user information",  # Broader query
    user_id="user123",
    use_tcmgm=True
)

# Check 3: Check each source individually
print(f"Vector: {len(results['vector_results'])}")
print(f"Graph: {len(results['graph_results'])}")
print(f"Timeline: {len(results['timeline_results'])}")
```

### Issue: Time Window Returns No Results

**Problem:** Timeline results are empty despite time window

**Solutions:**

```python
# Check 1: Verify timeline events exist
from outhad_contextkit.memory.temporal.timeline_queries import TimelineQueries

queries = TimelineQueries(memory._timeline_builder)
all_events = queries.get_recent_events("user123", limit=100)

print(f"Total timeline events: {len(all_events)}")

if not all_events:
    print("❌ No timeline events found")
    print("Build timeline first by adding conversations")

# Check 2: Verify time window includes events
import statistics

if all_events:
    timestamps = [e['timestamp'] for e in all_events]
    earliest = min(timestamps)
    latest = max(timestamps)

    print(f"Timeline span: {earliest} to {latest}")
    print("Adjust your time window to include this range")

# Check 3: Use wider time window
wide_window = {
    "start": (datetime.utcnow() - timedelta(days=365)).isoformat(),
    "end": (datetime.utcnow() + timedelta(days=1)).isoformat()
}

results = memory.search(
    query="events",
    user_id="user123",
    use_tcmgm=True,
    time_window=wide_window
)
```

### Issue: Causal Chains Not Found

**Problem:** `causal_chains` array is empty

**Solutions:**

```python
# Check 1: Verify causal exploration is enabled
results = memory.search(
    query="test",
    user_id="user123",
    use_tcmgm=True,
    include_causal=True  # Must be True!
)

# Check 2: Add memories with causal relationships
memory.add(
    "Started new project which led to learning new skills",
    user_id="user123"
)

# Check 3: Verify causal links exist in graph
# (Requires Neo4j access)
if memory.graph:
    # Query causal relationships
    from outhad_contextkit.memory.temporal import get_causal_chain

    chains = get_causal_chain(
        graph=memory.graph.graph,
        start_event_id="some_event_id",
        filters={"user_id": "user123"},
        direction="forward"
    )
    print(f"Found {len(chains)} causal chains")
```

### Issue: Slow Query Performance

**Problem:** Queries taking >2 seconds

**Solutions:**

```python
# Solution 1: Disable causal exploration
results = memory.search(
    query="query",
    user_id="user123",
    use_tcmgm=True,
    include_causal=False  # Saves 100-300ms
)

# Solution 2: Reduce limit
results = memory.search(
    query="query",
    user_id="user123",
    use_tcmgm=True,
    limit=10  # Lower limit
)

# Solution 3: Use standard search for simple queries
results = memory.search(
    query="simple lookup",
    user_id="user123",
    use_tcmgm=False  # Much faster
)

# Solution 4: Optimize Neo4j
# - Add indexes on user_id, timestamp fields
# - Monitor Neo4j query performance
# - Consider query caching
```

### Issue: Inconsistent Results

**Problem:** Same query returns different results

**Solutions:**

```python
# This is expected! TCMGM is dynamic
# Results change as:
# 1. New memories are added
# 2. New timeline events are created
# 3. New causal relationships are discovered

# To debug inconsistency:
def debug_search(query, user_id):
    """Debug search results."""
    results = memory.search(
        query=query,
        user_id=user_id,
        use_tcmgm=True
    )

    print(f"Query: {query}")
    print(f"Vector: {len(results['vector_results'])} results")
    print(f"Graph: {len(results['graph_results'])} results")
    print(f"Timeline: {len(results['timeline_results'])} results")
    print(f"Fused: {len(results['fused_ranking'])} results")

    print("\nTop 3 results:")
    for i, r in enumerate(results['fused_ranking'][:3], 1):
        print(f"{i}. [{r['source']}] {r['content'][:60]}...")

    return results

# Run multiple times to check consistency
result1 = debug_search("test query", "user123")
result2 = debug_search("test query", "user123")

# Compare
print(f"\nConsistency check:")
print(f"Run 1: {len(result1['fused_ranking'])} results")
print(f"Run 2: {len(result2['fused_ranking'])} results")
```

---

## 📚 API Reference

### Memory.search() Extended

```python
def search(
    self,
    query: str,
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: int = 100,
    filters: Optional[Dict[str, Any]] = None,

    # TCMGM Parameters (New in )
    use_tcmgm: bool = False,
    time_window: Optional[Dict[str, str]] = None,
    include_causal: bool = True,
) -> Union[Dict, SearchResult]:
    """
    Search memory with optional TCMGM fused retrieval.

    Args:
        query: Search query string
        user_id: Filter by user (required unless agent_id or run_id)
        agent_id: Filter by agent
        run_id: Filter by run
        limit: Maximum results to return
        filters: Additional metadata filters

        use_tcmgm: Enable TCMGM fused retrieval (requires graph storage)
        time_window: Time range filter with 'start' and 'end' ISO timestamps
        include_causal: Include causal chain exploration (only with TCMGM)

    Returns:
        If use_tcmgm=False:
            SearchResult with 'results' key

        If use_tcmgm=True:
            Dict with keys:
                - vector_results: List[Dict]
                - graph_results: List[Dict]
                - timeline_results: List[Dict]
                - causal_chains: List[Dict]
                - multimodal_results: List[Dict]
                - fused_ranking: List[Dict]

    Examples:
        # Standard search
        results = memory.search("query", user_id="user123")

        # TCMGM fused search
        results = memory.search(
            "query",
            user_id="user123",
            use_tcmgm=True,
            time_window={
                "start": "2025-01-01T00:00:00",
                "end": "2025-01-31T23:59:59"
            },
            include_causal=True
        )
    """
```

### RetrievalOrchestrator

```python
class RetrievalOrchestrator:
    def __init__(
        self,
        vector_store,
        graph_store,
        timeline_builder: TimelineBuilder,
        embedding_model
    ):
        """
        Initialize retrieval orchestrator.

        Args:
            vector_store: Vector storage instance
            graph_store: Graph storage instance
            timeline_builder: Timeline builder instance
            embedding_model: Embedding model for vector search
        """

    def fused_search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        time_window: Optional[Dict] = None,
        include_causal: bool = True,
        include_multimodal: bool = True,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Perform fused retrieval across all sources.

        Args:
            query: Search query string
            user_id: User identifier
            top_k: Number of results per source
            time_window: Optional time range filter
            include_causal: Enable causal chain exploration
            include_multimodal: Enable multimodal search
            filters: Additional metadata filters

        Returns:
            Dict with keys:
                - vector_results: Semantic similarity results
                - graph_results: Entity relationship results
                - timeline_results: Temporal event results
                - causal_chains: Causal relationship chains
                - multimodal_results: Cross-modal results
                - fused_ranking: Combined and weighted results

        Example:
            results = orchestrator.fused_search(
                query="user activities",
                user_id="user123",
                top_k=20,
                include_causal=True
            )
        """
```

### Result Format

```python
# Standard Search Result
{
    "results": [
        {
            "id": "mem_123",
            "memory": "User likes Python",
            "score": 0.92,
            "metadata": {...}
        }
    ]
}

# TCMGM Fused Search Result
{
    "vector_results": [
        {
            "id": "mem_123",
            "content": "User likes Python",
            "score": 0.92,
            "metadata": {...},
            "source": "vector"
        }
    ],
    "graph_results": [
        {
            "id": "entity_456",
            "content": "Python",
            "score": 0.85,
            "metadata": {...},
            "source": "graph"
        }
    ],
    "timeline_results": [
        {
            "id": "event_789",
            "content": "Started learning Python",
            "timestamp": "2025-01-01T10:00:00Z",
            "score": 0.80,
            "source": "timeline"
        }
    ],
    "causal_chains": [
        {
            "seed_event": {...},
            "forward_chain": [...],
            "backward_chain": [...]
        }
    ],
    "multimodal_results": [],
    "fused_ranking": [
        {
            "id": "mem_123",
            "content": "User likes Python",
            "source": "vector",
            "score": 0.92,
            "weight": 0.4,
            "final_score": 0.368
        }
    ]
}
```

---

## 🔗 Integration with Other TCMGM 

### Temporal Attributes

TCMGM uses temporal attributes for timeline filtering:

```python
# Timeline results include temporal metadata
for event in results['timeline_results']:
    print(f"Event: {event['content']}")
    print(f"Timestamp: {event['timestamp']}")
    print(f"Confidence: {event.get('confidence', 'N/A')}")
    print(f"Modality: {event.get('modality', 'text')}")
```

### Causal Relationships

Causal chains are automatically explored in TCMGM:

```python
# Access causal chains from search results
for chain in results['causal_chains']:
    print(f"Seed: {chain['seed_event']['name']}")

    # Forward causality (effects)
    for effect in chain['forward_chain']:
        print(f"  → {effect['name']}")

    # Backward causality (causes)
    for cause in chain['backward_chain']:
        print(f"  ← {cause['name']}")
```

### Multimodal Support

Cross-modal search framework is ready for multimodal content:

```python
# Multimodal results (framework in place)
results = memory.search(
    query="images of vacation",
    user_id="user123",
    use_tcmgm=True,
    include_multimodal=True  # Ready for future implementation
)

# Access multimodal results
print(f"Multimodal results: {len(results['multimodal_results'])}")
```

### Timeline Helper

Timeline queries are integrated into fused search:

```python
# Timeline results are automatically included
results = memory.search(
    query="recent events",
    user_id="user123",
    use_tcmgm=True,
    time_window={
        "start": (datetime.utcnow() - timedelta(days=7)).isoformat(),
        "end": datetime.utcnow().isoformat()
    }
)

# Timeline events from 
for event in results['timeline_results']:
    print(event['content'])
```

---

## 🎉 Summary

 Retrieval Orchestrator provides:

✅ **Multi-Source Fusion** - Combines vector, graph, timeline, and causal data
✅ **Intelligent Ranking** - Weighted scoring ensures best results surface first
✅ **Flexible Filtering** - Time windows and causal exploration toggle
✅ **Backward Compatible** - Existing code continues to work unchanged
✅ **Production Ready** - Robust error handling and graceful degradation
✅ **Comprehensive API** - Easy to use with sensible defaults

### Quick Reference

```python
from outhad_contextkit import Memory
from datetime import datetime, timedelta

# Initialize
memory = Memory(config_with_graph_enabled)

# Add memories
memory.add("User data", user_id="user123")

# Standard search (fast)
results = memory.search(
    query="query",
    user_id="user123",
    use_tcmgm=False
)

# TCMGM fused search (comprehensive)
results = memory.search(
    query="query",
    user_id="user123",
    use_tcmgm=True,
    time_window={
        "start": (datetime.utcnow() - timedelta(days=7)).isoformat(),
        "end": datetime.utcnow().isoformat()
    },
    include_causal=True,
    limit=10
)

# Access results
print(f"Vector: {len(results['vector_results'])}")
print(f"Graph: {len(results['graph_results'])}")
print(f"Timeline: {len(results['timeline_results'])}")
print(f"Causal: {len(results['causal_chains'])}")
print(f"Fused: {len(results['fused_ranking'])}")
```

---
