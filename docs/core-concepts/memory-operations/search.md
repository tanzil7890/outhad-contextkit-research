# Memory Operations: Search Memory - Python Guide

## 🎯 Overview

The **search** operation retrieves relevant memories from Outhad_ContextKit using natural language queries and semantic similarity. Like a smart librarian, it finds exactly what you need from everything you've stored, ranking results by relevance and supporting powerful filtering.

**Why it matters:**
- Retrieves the right facts without rebuilding prompts from scratch
- Supports natural language queries (ask questions, get answers)
- Filters by user, session, metadata, and more
- Ranks results by semantic similarity
- Essential for context-aware AI responses

---

## 🚀 Quick Start

### 1. Import and Initialize

```python
from outhad_contextkit import Memory

# Initialize memory
memory = Memory()
```

### 2. Basic Search Examples

```python
# Add some memories first
memory.add("Alice loves Italian food", user_id="alice")
memory.add("Alice is allergic to peanuts", user_id="alice")
memory.add("Alice prefers morning meetings", user_id="alice")

# Simple search with natural language
results = memory.search(
    query="What food does Alice like?",
    user_id="alice"
)

# Print results
for result in results['results']:
    print(f"Memory: {result['memory']}")
    print(f"Score: {result.get('score', 'N/A')}")
    print()

# Output:
# Memory: Alice loves Italian food
# Score: 0.92
```

---

## 📖 Core Features

### Feature 1: Natural Language Queries

Outhad_ContextKit understands intent, not just keywords:

```python
# All of these work and return relevant results
queries = [
    "What are Alice's dietary restrictions?",
    "food allergies alice",
    "what can't alice eat",
    "dietary preferences and restrictions"
]

for query in queries:
    results = memory.search(query, user_id="alice", limit=3)
    print(f"Query: {query}")
    print(f"Found: {results['results'][0]['memory']}")
    print()

# All return: "Alice is allergic to peanuts"
```

**Search understands:**
- Questions ("What does Alice like?")
- Keywords ("alice food preferences")
- Phrases ("dietary restrictions")
- Synonyms ("allergies" = "sensitivities" = "intolerances")

---

### Feature 2: Scoped Search

#### Search by User

```python
# Always scope by user to prevent cross-contamination
results = memory.search(
    "food preferences",
    user_id="alice"  # Only Alice's memories
)
```

#### Search by Session

```python
# Search within a specific session
results = memory.search(
    "current bug status",
    user_id="dev_team",
    session_id="debug_session_001"  # Only this session
)
```

#### Search by Agent/Run

```python
# Search across all agents in a run
results = memory.search(
    "security vulnerabilities found",
    run_id="audit_2025_01"  # All agents in this audit
)

# Search specific agent's memories
results = memory.search(
    "research findings",
    agent_id="research_agent",
    run_id="audit_2025_01"
)
```

---

### Feature 3: Filtered Search

#### Filter by Metadata

```python
# Add memories with metadata
memory.add(
    "Alice prefers morning meetings",
    user_id="alice",
    metadata={"category": "scheduling", "priority": "high"}
)

memory.add(
    "Alice available Monday-Friday",
    user_id="alice",
    metadata={"category": "availability", "priority": "medium"}
)

# Search with metadata filter
results = memory.search(
    "scheduling",
    user_id="alice",
    filters={"category": "scheduling"}  # Only scheduling memories
)
```

#### Complex Filters

```python
# Multiple filter conditions
results = memory.search(
    "important preferences",
    user_id="alice",
    filters={
        "priority": "high",
        "category": "scheduling"
    }
)
```

---

### Feature 4: Limit Results

Control how many results to retrieve:

```python
# Get top 3 most relevant memories
results = memory.search(
    "user preferences",
    user_id="alice",
    limit=3  # Return top 3 results
)

print(f"Retrieved {len(results['results'])} memories")

# Get top 10 for more context
results = memory.search(
    "everything about alice",
    user_id="alice",
    limit=10
)
```

**Recommendation:**
- Use `limit=3-5` for focused context
- Use `limit=10-20` for broad context
- Default limit depends on your configuration

---

### Feature 5: Analyzing Search Results

```python
# Search and analyze results
results = memory.search("food preferences", user_id="alice", limit=5)

# Check result structure
print(f"Total results: {len(results['results'])}")

for idx, result in enumerate(results['results'], 1):
    print(f"\n--- Result {idx} ---")
    print(f"Memory ID: {result['id']}")
    print(f"Memory: {result['memory']}")
    print(f"Score: {result.get('score', 'N/A')}")
    print(f"Created: {result.get('created_at', 'N/A')}")
    print(f"Metadata: {result.get('metadata', {})}")
```

---

## 🎯 Common Use Cases

### Use Case 1: Context-Aware Chatbot

```python
from openai import OpenAI
from outhad_contextkit import Memory

openai_client = OpenAI()
memory = Memory()

def chat_with_context(user_message: str, user_id: str) -> str:
    """Chat with memory-augmented responses."""

    # 1. Search relevant memories
    memories = memory.search(
        query=user_message,
        user_id=user_id,
        limit=3
    )

    # 2. Build context string
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

    # 4. Store conversation
    memory.add([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response.choices[0].message.content}
    ], user_id=user_id)

    return response.choices[0].message.content

# Usage
print(chat_with_context("I love pizza", user_id="alice"))
# Stores: "Alice loves pizza"

# Later...
print(chat_with_context("What food do I like?", user_id="alice"))
# Uses memory context to answer: "You love pizza!"
```

---

### Use Case 2: Customer Support Context

```python
def handle_support_request(issue: str, user_id: str):
    """Handle support with customer history."""

    # Search customer history
    history = memory.search(
        query=issue,
        user_id=user_id,
        limit=5
    )

    # Check for similar past issues
    past_issues = [
        m for m in history['results']
        if 'issue' in m['memory'].lower()
    ]

    # Build context
    context = {
        "is_repeat_customer": len(history['results']) > 0,
        "past_issues_count": len(past_issues),
        "relevant_history": [m['memory'] for m in history['results'][:3]]
    }

    return context

# Usage
context = handle_support_request(
    "My account is locked",
    user_id="customer_789"
)

if context["is_repeat_customer"]:
    print(f"Returning customer with {context['past_issues_count']} past issues")
    print("Relevant history:")
    for history_item in context["relevant_history"]:
        print(f"  - {history_item}")
```

---

### Use Case 3: Personalized Recommendations

```python
def get_recommendations(user_id: str, category: str = "food"):
    """Get personalized recommendations based on preferences."""

    # Search user preferences
    preferences = memory.search(
        query=f"{category} preferences likes dislikes",
        user_id=user_id,
        filters={"category": f"{category}_preferences"},
        limit=10
    )

    # Extract preferences
    likes = []
    dislikes = []

    for pref in preferences['results']:
        memory_text = pref['memory'].lower()
        if 'like' in memory_text or 'love' in memory_text or 'prefer' in memory_text:
            likes.append(pref['memory'])
        elif 'dislike' in memory_text or 'avoid' in memory_text or 'allergic' in memory_text:
            dislikes.append(pref['memory'])

    return {
        "likes": likes,
        "dislikes": dislikes,
        "recommendation_context": preferences['results']
    }

# Usage
recs = get_recommendations(user_id="alice", category="food")
print("User likes:", recs["likes"])
print("User dislikes:", recs["dislikes"])
```

---

### Use Case 4: Multi-Agent Knowledge Sharing

```python
def search_agent_knowledge(query: str, run_id: str, agent_type: str = None):
    """Search knowledge across agents in a run."""

    # Search across all agents in run
    if agent_type:
        results = memory.search(
            query=query,
            agent_id=agent_type,
            run_id=run_id,
            limit=10
        )
        print(f"Searching {agent_type} agent knowledge...")
    else:
        results = memory.search(
            query=query,
            run_id=run_id,
            limit=10
        )
        print(f"Searching all agents in run {run_id}...")

    return results['results']

# Research agent stores findings
memory.add(
    "Found SQL injection vulnerability in login endpoint",
    agent_id="security_scanner",
    run_id="audit_2025_01"
)

# Development agent searches for security issues
findings = search_agent_knowledge(
    query="security vulnerabilities",
    run_id="audit_2025_01",
    agent_type="security_scanner"
)

print(f"Found {len(findings)} security findings")
```

---

### Use Case 5: Session Context Retrieval

```python
def get_session_context(user_id: str, session_id: str):
    """Get all context for current session."""

    # Search within session
    session_memories = memory.search(
        query="*",  # Get all session memories
        user_id=user_id,
        session_id=session_id,
        limit=50
    )

    # Organize by recency
    sorted_memories = sorted(
        session_memories['results'],
        key=lambda x: x.get('created_at', ''),
        reverse=True
    )

    return {
        "session_id": session_id,
        "memory_count": len(sorted_memories),
        "recent_memories": sorted_memories[:5],
        "all_memories": sorted_memories
    }

# Usage
context = get_session_context(
    user_id="dev_team",
    session_id="debug_session_001"
)
print(f"Session has {context['memory_count']} memories")
```

---

## 🔍 Best Practices

### 1. Always Scope with user_id

```python
# ✅ Good: Scoped to user
results = memory.search("preferences", user_id="alice")

# ❌ Bad: No user scope (returns all users' memories)
results = memory.search("preferences")
```

### 2. Use Natural Language Queries

```python
# ✅ Good: Natural question
results = memory.search(
    "What are Alice's dietary restrictions?",
    user_id="alice"
)

# ✅ Also good: Keywords
results = memory.search(
    "dietary restrictions allergies",
    user_id="alice"
)

# ✅ Also good: Phrases
results = memory.search(
    "food preferences and restrictions",
    user_id="alice"
)
```

### 3. Use Appropriate Limits

```python
# ✅ Good: Focused context (3-5 results)
results = memory.search("recent preferences", user_id="alice", limit=3)

# ✅ Good: Broad context (10-20 results)
results = memory.search("all user information", user_id="alice", limit=15)

# ❌ Unnecessary: Too many results
results = memory.search("preferences", user_id="alice", limit=100)
# Adds noise and slows down LLM processing
```

### 4. Combine Filters for Precision

```python
# ✅ Good: Multiple filters
results = memory.search(
    "scheduling preferences",
    user_id="alice",
    filters={
        "category": "scheduling",
        "priority": "high"
    },
    limit=5
)
```

### 5. Check for Empty Results

```python
# ✅ Good: Handle empty results
results = memory.search("query", user_id="alice")

if not results['results']:
    print("No memories found")
    # Fall back to default behavior
else:
    print(f"Found {len(results['results'])} memories")
    for result in results['results']:
        print(f"  - {result['memory']}")
```

### 6. Use Metadata Filters for Organization

```python
# Add memories with categories
memory.add(
    "Alice likes pizza",
    user_id="alice",
    metadata={"category": "food", "subcategory": "preferences"}
)

# Search by category
food_memories = memory.search(
    "preferences",
    user_id="alice",
    filters={"category": "food"}
)
```

---

## 🐛 Troubleshooting

### Issue: No Results Returned

```python
# Check if memories exist
all_memories = memory.get_all(user_id="alice")
print(f"Total memories for user: {len(all_memories['results'])}")

# Try broader query
results = memory.search("*", user_id="alice", limit=10)
print(f"Broad search found: {len(results['results'])} memories")

# Verify user_id matches
results = memory.search(
    "preferences",
    user_id="alice"  # Make sure this matches the user_id used in add()
)
```

### Issue: Irrelevant Results

```python
# ❌ Problem: Query too generic
results = memory.search("preferences", user_id="alice")
# Returns all preferences (food, scheduling, notification, etc.)

# ✅ Solution: More specific query
results = memory.search(
    "What food does Alice prefer for breakfast?",
    user_id="alice"
)

# ✅ Solution: Use metadata filters
results = memory.search(
    "preferences",
    user_id="alice",
    filters={"category": "food", "meal": "breakfast"}
)
```

### Issue: Wrong User's Memories Returned

```python
# ❌ Problem: user_id mismatch
memory.add("Alice's preference", user_id="alice")
results = memory.search("preference", user_id="bob")  # Wrong user
# Returns nothing

# ✅ Solution: Verify user_id consistency
user_id = "alice"
memory.add("Preference data", user_id=user_id)
results = memory.search("preference", user_id=user_id)  # Same user
```

### Issue: Search Returns Old/Stale Data

```python
# Check when memories were created
results = memory.search("preferences", user_id="alice", limit=10)

for result in results['results']:
    print(f"Memory: {result['memory']}")
    print(f"Created: {result.get('created_at', 'N/A')}")
    print()

# Filter by metadata timestamp if available
results = memory.search(
    "preferences",
    user_id="alice",
    filters={"updated_at": "2025-01-14"}
)
```

### Issue: Session Memories Mixing with User Memories

```python
# ❌ Problem: Not specifying session_id in search
results = memory.search("debug info", user_id="alice")
# Returns both session and permanent memories

# ✅ Solution: Specify session_id for session-only search
results = memory.search(
    "debug info",
    user_id="alice",
    session_id="debug_session_001"  # Only this session
)
```

---

## 📊 Performance Tips

### 1. Limit Results Appropriately

```python
# ✅ Efficient: Get only what you need
results = memory.search("preferences", user_id="alice", limit=3)

# ❌ Inefficient: Get too many results
results = memory.search("preferences", user_id="alice", limit=100)
# Wastes memory and processing time
```

### 2. Use Specific Queries

```python
# ✅ Efficient: Specific query
results = memory.search("dietary restrictions", user_id="alice")

# ❌ Inefficient: Generic query
results = memory.search("preferences", user_id="alice")
# Returns too many unrelated results
```

### 3. Cache Frequently Used Searches

```python
from functools import lru_cache
from datetime import datetime, timedelta

# Cache search results for 5 minutes
@lru_cache(maxsize=128)
def cached_search(query: str, user_id: str, limit: int = 5):
    """Cached search for frequently accessed data."""
    results = memory.search(query, user_id=user_id, limit=limit)
    return results

# Usage
results = cached_search("preferences", user_id="alice")
```

### 4. Use Metadata Filters to Reduce Search Space

```python
# ✅ Efficient: Filter by category first
results = memory.search(
    "preferences",
    user_id="alice",
    filters={"category": "food"}  # Narrows search space
)

# ❌ Less efficient: Search everything then filter in code
all_results = memory.search("preferences", user_id="alice", limit=100)
food_results = [r for r in all_results['results'] if 'food' in r['memory'].lower()]
```

---

## 📚 API Reference

### Memory.search()

```python
def search(
    query: str,
    user_id: str = None,
    session_id: str = None,
    agent_id: str = None,
    run_id: str = None,
    limit: int = 5,
    filters: Dict = None
) -> Dict
```

**Parameters:**
- **query** (str): Natural language query or keywords
- **user_id** (str, optional): Filter by user (recommended)
- **session_id** (str, optional): Filter by session
- **agent_id** (str, optional): Filter by agent
- **run_id** (str, optional): Filter by run
- **limit** (int, default=5): Maximum number of results
- **filters** (Dict, optional): Metadata filters

**Returns:**
- Dict with `results` list containing:
  - `id`: Memory ID
  - `memory`: Memory content
  - `score`: Similarity score (0-1)
  - `metadata`: Associated metadata
  - `created_at`: Creation timestamp

**Example:**
```python
results = memory.search(
    query="What are Alice's food preferences?",
    user_id="alice",
    filters={"category": "food"},
    limit=3
)

for result in results['results']:
    print(f"{result['memory']} (score: {result['score']:.2f})")
```

---

## 🔗 Related Guides

- **[Add Memory Guide](./add.md)** - Store memories
- **[Update Memory Guide](./update.md)** - Modify existing memories
- **[Delete Memory Guide](./delete.md)** - Remove memories
- **[Memory Types Guide](../memory-types.md)** - Understand memory layers
- **[Python User Guide](../PYTHON_USER_GUIDE.md)** - Complete Python guide

---

## 📝 Quick Reference

```python
# Basic search
results = memory.search("query", user_id="alice")

# With limit
results = memory.search("query", user_id="alice", limit=3)

# With metadata filter
results = memory.search(
    "query",
    user_id="alice",
    filters={"category": "food"}
)

# Session-scoped
results = memory.search(
    "query",
    user_id="alice",
    session_id="session_1"
)

# Multi-agent
results = memory.search(
    "query",
    agent_id="agent_1",
    run_id="run_1"
)

# Access results
for result in results['results']:
    print(f"Memory: {result['memory']}")
    print(f"Score: {result.get('score', 'N/A')}")
```

---

**Need help?** Check the [Troubleshooting Guide](../../TROUBLESHOOTING.md) or open an issue on [GitHub](https://github.com/outhad/outhad_contextkit/issues).
