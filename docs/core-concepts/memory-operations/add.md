# Memory Operations: Add Memory - Python Guide

## 🎯 Overview

The **add** operation stores conversations, facts, and preferences in Outhad_ContextKit for later retrieval. It's the foundation of building persistent memory for your AI agents - capturing important details from conversations and structuring them for efficient search.

**Why it matters:**
- Preserves user preferences, goals, and feedback across sessions
- Powers personalization and decision-making in future conversations
- Automatically extracts structured facts from unstructured conversations
- Supports conflict resolution to avoid duplicate or contradictory memories

---

## 🚀 Quick Start

### 1. Import and Initialize

```python
from outhad_contextkit import Memory

# Initialize memory
memory = Memory()
```

### 2. Basic Add Examples

```python
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

# Verify it was stored
results = memory.search("travel plans", user_id="alice")
print(results['results'][0]['memory'])
# Output: "Alice is planning a trip to Tokyo next month"
```

---

## 📖 Core Features

### Feature 1: Inferred vs Raw Storage

#### Inferred Mode (Default - Recommended)

LLM automatically extracts structured facts from conversations:

```python
# Inferred extraction (default)
messages = [
    {"role": "user", "content": "I prefer sci-fi movies over thrillers"},
    {"role": "assistant", "content": "Got it! I'll suggest sci-fi in the future"}
]

result = memory.add(
    messages,
    user_id="alice",
    infer=True  # Default behavior
)

# What gets stored: "Alice prefers sci-fi movies"
# (Extracted fact, not raw conversation)
```

**Benefits:**
- ✅ Removes conversational fluff
- ✅ Extracts key facts and preferences
- ✅ Handles conflict resolution automatically
- ✅ Better search relevance

#### Raw Mode (Exact Storage)

Store messages exactly as provided:

```python
# Raw storage (no extraction)
messages = [
    {"role": "user", "content": "Meeting at 3pm tomorrow"},
    {"role": "assistant", "content": "Reminder set!"}
]

result = memory.add(
    messages,
    user_id="alice",
    infer=False  # Store as-is
)

# What gets stored: Exact messages without modification
```

**⚠️ Warning**: Mixing `infer=True` and `infer=False` for the same content creates duplicates. Choose one approach and stick with it.

---

### Feature 2: Add with Metadata

Enrich memories with custom metadata for better filtering:

```python
# Add with metadata
memory.add(
    "Alice prefers morning meetings before 10 AM",
    user_id="alice",
    metadata={
        "category": "scheduling_preferences",
        "priority": "high",
        "updated_at": "2025-01-14",
        "source": "onboarding_call"
    }
)

# Later, search with metadata filters
results = memory.search(
    "scheduling preferences",
    user_id="alice",
    filters={"priority": "high"}
)
```

**Common Metadata Uses:**
- **Category**: Group related memories ("food_preferences", "work_schedule")
- **Priority**: Mark importance ("high", "medium", "low")
- **Source**: Track origin ("user_input", "agent_inference", "api_import")
- **Timestamp**: Track when information was captured
- **Tags**: Add searchable keywords

---

### Feature 3: Session-Scoped Memory

Add temporary context that expires with the session:

```python
# Short-term session memory
memory.add(
    "Current debugging: API timeout in /auth endpoint",
    user_id="dev_team",
    session_id="debug_session_001"
)

# Add more session context
memory.add(
    "Error occurs only with OAuth2 provider",
    user_id="dev_team",
    session_id="debug_session_001"
)

# Search within session
results = memory.search(
    "authentication issues",
    user_id="dev_team",
    session_id="debug_session_001"
)

# Clear session when done
memory.delete_all(user_id="dev_team", session_id="debug_session_001")
```

**Use Cases for Session Memory:**
- Multi-step onboarding flows
- Debugging sessions
- Temporary task context
- Conversation threads

---

### Feature 4: Multi-Agent Memory

Share knowledge between agents using `agent_id` and `run_id`:

```python
# Research agent stores findings
memory.add(
    "Found 3 security vulnerabilities in authentication module",
    agent_id="research_agent",
    run_id="audit_2025_01"
)

# Development agent retrieves findings
findings = memory.search(
    query="security issues",
    run_id="audit_2025_01"  # All agents in this run
)

# Development agent adds fixes
memory.add(
    "Fixed SQL injection in login endpoint (CVE-2025-1234)",
    agent_id="dev_agent",
    run_id="audit_2025_01"
)

# QA agent can access all audit memories
verification = memory.search(
    query="security fixes",
    run_id="audit_2025_01"
)
```

---

### Feature 5: Batch Add

Add multiple memories efficiently:

```python
# Add multiple messages at once
conversation_history = [
    {"role": "user", "content": "I'm allergic to peanuts"},
    {"role": "assistant", "content": "I'll remember that"},
    {"role": "user", "content": "I also avoid shellfish"},
    {"role": "assistant", "content": "Noted - no peanuts or shellfish"}
]

result = memory.add(
    conversation_history,
    user_id="alice",
    metadata={"category": "dietary_restrictions"}
)

# Check how many memories were created
print(f"Created {len(result['results'])} memories")
```

---

## 🎯 Common Use Cases

### Use Case 1: Chatbot with Persistent Memory

```python
from openai import OpenAI
from outhad_contextkit import Memory

openai_client = OpenAI()
memory = Memory()

def chat_with_memory(user_message: str, user_id: str) -> str:
    """Chat function that stores every conversation."""

    # 1. Retrieve relevant memories
    memories = memory.search(
        query=user_message,
        user_id=user_id,
        limit=3
    )
    context = "\n".join([m['memory'] for m in memories['results']])

    # 2. Generate response with context
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"User context:\n{context}"},
            {"role": "user", "content": user_message}
        ]
    )

    # 3. Store new conversation
    memory.add([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response.choices[0].message.content}
    ], user_id=user_id)

    return response.choices[0].message.content

# Usage
print(chat_with_memory("I love pizza", user_id="alice"))
# Later in a different session...
print(chat_with_memory("What food do I like?", user_id="alice"))
# Output: "You mentioned you love pizza!"
```

---

### Use Case 2: User Onboarding

```python
def store_onboarding_data(user_data: dict, user_id: str):
    """Store user preferences during onboarding."""

    # Store each preference with metadata
    preferences = [
        f"Name: {user_data['name']}",
        f"Email: {user_data['email']}",
        f"Preferred language: {user_data['language']}",
        f"Timezone: {user_data['timezone']}",
        f"Notification preference: {user_data['notifications']}"
    ]

    for pref in preferences:
        memory.add(
            pref,
            user_id=user_id,
            metadata={
                "category": "onboarding",
                "source": "signup_form",
                "verified": True
            }
        )

    print(f"✅ Stored {len(preferences)} preferences for {user_id}")

# Usage
store_onboarding_data(
    user_data={
        "name": "Alice Smith",
        "email": "alice@example.com",
        "language": "English",
        "timezone": "America/New_York",
        "notifications": "email_only"
    },
    user_id="alice"
)
```

---

### Use Case 3: Fitness Tracker with Memory

```python
from outhad_contextkit import Memory

memory = Memory()

# Store fitness conversation history
fitness_messages = [
    {"role": "user", "content": "I'm 26 years old, 5'10\", and weigh 72kg"},
    {"role": "assistant", "content": "Got it - 26, 5'10\", 72kg"},
    {"role": "user", "content": "I follow push-pull-legs, train 5x/week"},
    {"role": "assistant", "content": "Noted - PPL split, 5x/week"},
    {"role": "user", "content": "I have mild lactose intolerance"},
    {"role": "assistant", "content": "Understood - avoiding dairy"}
]

memory.add(
    fitness_messages,
    user_id="anish",
    metadata={
        "category": "fitness_profile",
        "source": "initial_consultation"
    }
)

# Later, query with context
results = memory.search(
    "What are my dietary restrictions?",
    user_id="anish"
)
print(results['results'][0]['memory'])
# Output: "Anish has mild lactose intolerance"
```

---

### Use Case 4: Customer Support Context

```python
def log_support_interaction(ticket_data: dict, user_id: str):
    """Log customer support interactions for context."""

    interaction = f"""
    Support Ticket #{ticket_data['ticket_id']}
    Issue: {ticket_data['issue']}
    Resolution: {ticket_data['resolution']}
    Status: {ticket_data['status']}
    """

    memory.add(
        interaction,
        user_id=user_id,
        metadata={
            "category": "support_history",
            "ticket_id": ticket_data['ticket_id'],
            "priority": ticket_data['priority'],
            "resolved": ticket_data['status'] == "closed"
        }
    )

# Usage
log_support_interaction(
    ticket_data={
        "ticket_id": "SUP-12345",
        "issue": "Password reset not working",
        "resolution": "Reset via email link",
        "status": "closed",
        "priority": "high"
    },
    user_id="customer_789"
)

# Check customer history
history = memory.search(
    "past issues",
    user_id="customer_789"
)
```

---

## 🔍 Best Practices

### 1. Always Use user_id

```python
# ✅ Good: Scoped to user
memory.add("User preference", user_id="alice")

# ❌ Bad: No user scope (will fail or mix users)
memory.add("User preference")  # Don't do this
```

### 2. Use Infer=True for Conversations

```python
# ✅ Good: Let LLM extract facts
memory.add([
    {"role": "user", "content": "I prefer sci-fi over horror"},
    {"role": "assistant", "content": "Got it!"}
], user_id="alice", infer=True)
# Stores: "Alice prefers sci-fi movies"

# ❌ Bad: Store raw conversation unnecessarily
memory.add("User: I like sci-fi. Bot: Ok", user_id="alice", infer=False)
```

### 3. Use Infer=False for Structured Data

```python
# ✅ Good: Store structured data as-is
memory.add(
    "Order #12345: Pizza Margherita, €15.99, delivered 2025-01-14",
    user_id="alice",
    infer=False,  # Don't extract, keep exact format
    metadata={"type": "order", "order_id": "12345"}
)
```

### 4. Add Metadata for Better Organization

```python
# ✅ Good: Rich metadata
memory.add(
    "Alice prefers morning meetings",
    user_id="alice",
    metadata={
        "category": "scheduling",
        "priority": "high",
        "source": "preferences_survey",
        "updated_at": "2025-01-14"
    }
)

# ❌ OK but not optimal: No metadata
memory.add("Alice prefers morning meetings", user_id="alice")
```

### 5. Use Session IDs for Temporary Context

```python
# ✅ Good: Session-scoped temporary memory
memory.add(
    "Debugging login timeout",
    user_id="dev_team",
    session_id="debug_20250114"
)

# Clear when done
memory.delete_all(user_id="dev_team", session_id="debug_20250114")
```

### 6. Verify Storage After Adding

```python
# Add memory
result = memory.add("Important preference", user_id="alice")

# Verify it was stored
if result and 'results' in result:
    memory_id = result['results'][0]['id']
    print(f"✅ Memory stored with ID: {memory_id}")

    # Verify by searching
    search_results = memory.search("important", user_id="alice")
    if search_results['results']:
        print("✅ Memory is searchable")
```

### 7. Handle Privacy Carefully

```python
# ✅ Good: Use PPMF for automatic PII encryption
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.memory.privacy.config import PPMFConfig

config = MemoryConfig(
    ppmf=PPMFConfig(enabled=True, auto_detect_pii=True)
)
memory = Memory(config=config)

memory.add("My email is alice@example.com", user_id="alice")
# Automatically encrypted

# ❌ Bad: Store sensitive data without protection
memory.add("My password is 12345", user_id="alice")  # NEVER DO THIS
```

---

## 🐛 Troubleshooting

### Issue: Memory Not Being Stored

```python
# Check if add was successful
result = memory.add("Test memory", user_id="alice")

if not result or 'results' not in result:
    print("❌ Add failed")
    print(f"Response: {result}")
else:
    print(f"✅ Added {len(result['results'])} memories")

# Verify by searching
search_results = memory.search("test", user_id="alice")
print(f"Found {len(search_results['results'])} results")
```

### Issue: LLM Extraction Not Working

```python
# If infer=True isn't working, check API key
import os
print(f"OpenAI API Key set: {bool(os.getenv('OPENAI_API_KEY'))}")

# Try with infer=False to bypass LLM
result = memory.add(
    "Direct storage test",
    user_id="alice",
    infer=False  # Skip LLM processing
)
```

### Issue: Duplicate Memories Created

```python
# Problem: Mixing infer modes creates duplicates
memory.add("I like pizza", user_id="alice", infer=True)   # Stores: "Alice likes pizza"
memory.add("I like pizza", user_id="alice", infer=False)  # Stores: "I like pizza"
# Result: 2 memories for same fact

# Solution: Choose one mode and stick with it
memory.add("I like pizza", user_id="alice", infer=True)   # Always use infer=True
memory.add("I like pasta", user_id="alice", infer=True)   # Consistent mode
```

### Issue: Messages Not in Correct Format

```python
# ✅ Correct format for conversation
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]
memory.add(messages, user_id="alice")

# ❌ Wrong format
messages = [
    {"user": "Hello"},  # Wrong key (should be "role")
    {"text": "Hi"}      # Wrong key (should be "content")
]
# This will fail or be misinterpreted
```

### Issue: Session Memories Mixing with User Memories

```python
# Problem: Not specifying session_id
memory.add("Temporary debug info", user_id="alice")
# This becomes permanent user memory

# Solution: Use session_id for temporary context
memory.add(
    "Temporary debug info",
    user_id="alice",
    session_id="debug_session_1"  # Scoped to session
)
```

---

## 📊 Performance Tips

### 1. Batch Add for Multiple Memories

```python
# ✅ Efficient: Add multiple messages at once
messages = [
    {"role": "user", "content": f"Message {i}"}
    for i in range(10)
]
memory.add(messages, user_id="alice")

# ❌ Inefficient: Add one at a time
for i in range(10):
    memory.add(f"Message {i}", user_id="alice")
```

### 2. Use infer=False for Large Volumes

```python
# If adding many memories rapidly, skip LLM extraction
for data in large_dataset:
    memory.add(
        data,
        user_id="bulk_import",
        infer=False,  # Skip LLM for speed
        metadata={"source": "bulk_import"}
    )
```

### 3. Add Metadata Upfront

```python
# ✅ Good: Add metadata during creation
memory.add(
    "Preference data",
    user_id="alice",
    metadata={"category": "prefs", "priority": "high"}
)

# ❌ Bad: Update metadata later (requires extra operation)
result = memory.add("Preference data", user_id="alice")
memory_id = result['results'][0]['id']
memory.update(memory_id, data="Preference data", metadata={"category": "prefs"})
```

---

## 📚 API Reference

### Memory.add()

```python
def add(
    messages: Union[str, List[Dict], List[str]],
    user_id: str,
    session_id: str = None,
    agent_id: str = None,
    run_id: str = None,
    metadata: Dict = None,
    infer: bool = True
) -> Dict
```

**Parameters:**
- **messages** (str | List[Dict] | List[str]): Content to store
  - String: Single message
  - List[Dict]: Conversation with role/content format
  - List[str]: Multiple messages
- **user_id** (str): User identifier (required)
- **session_id** (str, optional): Session identifier for temporary context
- **agent_id** (str, optional): Agent identifier for multi-agent systems
- **run_id** (str, optional): Run identifier for grouping agent executions
- **metadata** (Dict, optional): Custom key-value pairs for filtering
- **infer** (bool, default=True): Extract facts with LLM vs store raw

**Returns:**
- Dict with `results` list containing memory IDs and content

**Example:**
```python
result = memory.add(
    "Alice loves Italian food",
    user_id="alice",
    metadata={"category": "food_preferences"}
)
print(result['results'][0]['id'])  # Memory ID
```

---

## 🔗 Related Guides

- **[Search Memory Guide](./search.md)** - Retrieve stored memories
- **[Update Memory Guide](./update.md)** - Modify existing memories
- **[Delete Memory Guide](./delete.md)** - Remove memories
- **[Memory Types Guide](../memory-types.md)** - Understand memory layers
- **[PPMF Privacy Guide](../../ppmf/README.md)** - Secure PII handling

---

## 📝 Quick Reference

```python
# Basic add
memory.add("Content", user_id="alice")

# Add conversation
memory.add([
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"}
], user_id="alice")

# Add with metadata
memory.add("Content", user_id="alice", metadata={"category": "prefs"})

# Session-scoped
memory.add("Temp context", user_id="alice", session_id="session_1")

# Multi-agent
memory.add("Data", agent_id="agent_1", run_id="run_1")

# Raw storage (no LLM extraction)
memory.add("Exact data", user_id="alice", infer=False)
```

---

**Need help?** Check the [Troubleshooting Guide](../../TROUBLESHOOTING.md) or open an issue on [GitHub](https://github.com/outhad/outhad_contextkit/issues).
