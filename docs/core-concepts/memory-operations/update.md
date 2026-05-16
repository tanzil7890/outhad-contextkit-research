# Memory Operations: Update Memory - Python Guide

## 🎯 Overview

The **update** operation modifies existing memories in Outhad_ContextKit, allowing you to correct outdated information, enrich metadata, or adjust memory content as user preferences change. This keeps your memory store accurate and up-to-date without needing to delete and re-add memories.

**Why it matters:**
- Corrects outdated or incorrect information immediately
- Adds new metadata without losing the original memory ID
- Maintains memory history and relationships
- More efficient than delete + re-add
- Preserves references in conversations and logs

---

## 🚀 Quick Start

### 1. Import and Initialize

```python
from outhad_contextkit import Memory

# Initialize memory
memory = Memory()
```

### 2. Basic Update Example

```python
# Add a memory
result = memory.add(
    "Alice's email: alice@oldcompany.com",
    user_id="alice"
)
memory_id = result['results'][0]['id']
print(f"Created memory: {memory_id}")

# Update the memory
memory.update(
    memory_id=memory_id,
    data="Alice's email: alice@newcompany.com"
)

# Verify update
updated = memory.get(memory_id)
print(f"Updated memory: {updated['memory']}")
# Output: "Alice's email: alice@newcompany.com"
```

---

## 📖 Core Features

### Feature 1: Update by Memory ID

The primary way to update memories:

```python
# Step 1: Find the memory ID (via search or get)
results = memory.search("Alice's email", user_id="alice")
memory_id = results['results'][0]['id']

# Step 2: Update the memory
memory.update(
    memory_id=memory_id,
    data="Alice's new email: alice@example.com"
)

# Step 3: Verify the update
updated = memory.get(memory_id)
print(f"✅ Updated: {updated['memory']}")
```

---

### Feature 2: Update with Metadata

Update both content and metadata together:

```python
# Find memory
results = memory.search("scheduling preference", user_id="alice")
memory_id = results['results'][0]['id']

# Update content and metadata
memory.update(
    memory_id=memory_id,
    data="Alice prefers morning meetings before 10 AM (updated preference)",
    metadata={
        "category": "scheduling",
        "priority": "high",
        "updated_at": "2025-01-15",
        "verified": True
    }
)

# Verify metadata
updated = memory.get(memory_id)
print(f"Memory: {updated['memory']}")
print(f"Metadata: {updated.get('metadata', {})}")
```

---

### Feature 3: Update Workflow

Complete workflow for updating memories:

```python
def update_user_preference(query: str, new_value: str, user_id: str):
    """Find and update a user preference."""

    # 1. Search for existing memory
    results = memory.search(query, user_id=user_id, limit=1)

    if not results['results']:
        print(f"❌ No memory found for: {query}")
        return False

    # 2. Get memory ID
    memory_id = results['results'][0]['id']
    old_value = results['results'][0]['memory']

    # 3. Update the memory
    memory.update(
        memory_id=memory_id,
        data=new_value,
        metadata={
            "updated_at": "2025-01-15",
            "previous_value": old_value
        }
    )

    print(f"✅ Updated memory {memory_id}")
    print(f"   Old: {old_value}")
    print(f"   New: {new_value}")
    return True

# Usage
update_user_preference(
    query="email address",
    new_value="Alice's email: alice@newcompany.com",
    user_id="alice"
)
```

---

### Feature 4: Partial Updates

Update only metadata without changing content:

```python
# Find memory
results = memory.search("preference", user_id="alice")
memory_id = results['results'][0]['id']
current_data = results['results'][0]['memory']

# Update only metadata (keep same content)
memory.update(
    memory_id=memory_id,
    data=current_data,  # Same content
    metadata={
        "priority": "high",  # New metadata
        "verified": True
    }
)
```

---

### Feature 5: Bulk Update Pattern

Update multiple memories efficiently:

```python
def bulk_update_category(old_category: str, new_category: str, user_id: str):
    """Update category for multiple memories."""

    # 1. Get all memories with old category
    all_memories = memory.get_all(user_id=user_id)

    updates_count = 0

    # 2. Update each memory
    for mem in all_memories['results']:
        metadata = mem.get('metadata', {})
        if metadata.get('category') == old_category:
            memory.update(
                memory_id=mem['id'],
                data=mem['memory'],
                metadata={**metadata, 'category': new_category}
            )
            updates_count += 1

    print(f"✅ Updated {updates_count} memories")
    return updates_count

# Usage
bulk_update_category(
    old_category="food_prefs",
    new_category="dietary_preferences",
    user_id="alice"
)
```

---

## 🎯 Common Use Cases

### Use Case 1: User Profile Updates

```python
def update_user_profile_field(field: str, new_value: str, user_id: str):
    """Update a specific user profile field."""

    # Search for the field
    results = memory.search(field, user_id=user_id, limit=1)

    if not results['results']:
        print(f"Field '{field}' not found, adding new memory")
        memory.add(
            f"{field}: {new_value}",
            user_id=user_id,
            metadata={"category": "profile", "field": field}
        )
        return

    # Update existing field
    memory_id = results['results'][0]['id']
    memory.update(
        memory_id=memory_id,
        data=f"{field}: {new_value}",
        metadata={
            "category": "profile",
            "field": field,
            "updated_at": "2025-01-15"
        }
    )

    print(f"✅ Updated {field} to: {new_value}")

# Usage
update_user_profile_field("phone number", "+1-555-0123", user_id="alice")
update_user_profile_field("timezone", "America/New_York", user_id="alice")
```

---

### Use Case 2: Preference Correction

```python
def correct_preference(old_pref: str, new_pref: str, user_id: str):
    """Correct a user's stated preference."""

    # Find the old preference
    results = memory.search(old_pref, user_id=user_id, limit=1)

    if not results['results']:
        print(f"Preference not found: {old_pref}")
        return

    memory_id = results['results'][0]['id']

    # Update with correction metadata
    memory.update(
        memory_id=memory_id,
        data=new_pref,
        metadata={
            "corrected": True,
            "correction_date": "2025-01-15",
            "original": old_pref
        }
    )

    print(f"✅ Corrected preference:")
    print(f"   From: {old_pref}")
    print(f"   To:   {new_pref}")

# Usage
correct_preference(
    old_pref="Alice loves horror movies",
    new_pref="Alice prefers sci-fi movies over horror",
    user_id="alice"
)
```

---

### Use Case 3: Enriching Memories with Context

```python
def enrich_memory_with_context(query: str, additional_context: str, user_id: str):
    """Add context to an existing memory."""

    # Find the memory
    results = memory.search(query, user_id=user_id, limit=1)

    if not results['results']:
        print("Memory not found")
        return

    memory_id = results['results'][0]['id']
    original_memory = results['results'][0]['memory']

    # Enrich with additional context
    enriched_memory = f"{original_memory}. {additional_context}"

    memory.update(
        memory_id=memory_id,
        data=enriched_memory,
        metadata={
            "enriched": True,
            "enrichment_date": "2025-01-15"
        }
    )

    print(f"✅ Enriched memory: {enriched_memory}")

# Usage
enrich_memory_with_context(
    query="dietary restriction",
    additional_context="Severity: mild, can tolerate small amounts",
    user_id="alice"
)
```

---

### Use Case 4: Version Tracking

```python
import json
from datetime import datetime

def update_with_version_history(memory_id: str, new_data: str, user_id: str):
    """Update memory while tracking version history."""

    # Get current memory
    current = memory.get(memory_id)
    old_data = current['memory']
    old_metadata = current.get('metadata', {})

    # Build version history
    version_history = old_metadata.get('version_history', [])
    version_history.append({
        "version": len(version_history) + 1,
        "data": old_data,
        "timestamp": datetime.utcnow().isoformat()
    })

    # Update with history
    memory.update(
        memory_id=memory_id,
        data=new_data,
        metadata={
            **old_metadata,
            "version_history": version_history,
            "current_version": len(version_history) + 1,
            "last_updated": datetime.utcnow().isoformat()
        }
    )

    print(f"✅ Updated to version {len(version_history) + 1}")

# Usage
results = memory.search("preference", user_id="alice")
memory_id = results['results'][0]['id']

update_with_version_history(
    memory_id=memory_id,
    new_data="Updated preference data",
    user_id="alice"
)
```

---

### Use Case 5: Scheduled Updates

```python
from datetime import datetime, timedelta

def schedule_memory_expiration(memory_id: str, days_until_expiry: int):
    """Mark a memory for future expiration."""

    # Get current memory
    current = memory.get(memory_id)

    # Calculate expiry date
    expiry_date = (datetime.utcnow() + timedelta(days=days_until_expiry)).isoformat()

    # Update with expiration metadata
    memory.update(
        memory_id=memory_id,
        data=current['memory'],
        metadata={
            **current.get('metadata', {}),
            "expires_at": expiry_date,
            "temporary": True
        }
    )

    print(f"✅ Memory will expire on {expiry_date}")

# Usage
results = memory.search("temporary context", user_id="alice")
memory_id = results['results'][0]['id']

schedule_memory_expiration(memory_id, days_until_expiry=7)
```

---

## 🔍 Best Practices

### 1. Always Verify Before Updating

```python
# ✅ Good: Verify memory exists
results = memory.search("preference", user_id="alice")

if results['results']:
    memory_id = results['results'][0]['id']
    memory.update(memory_id, data="New value")
else:
    print("Memory not found, creating new one")
    memory.add("New value", user_id="alice")

# ❌ Bad: Update without verification
memory.update("random_id", data="New value")  # May fail
```

### 2. Update Both Content and Metadata

```python
# ✅ Good: Update both for consistency
memory.update(
    memory_id=memory_id,
    data="Updated content",
    metadata={
        "updated_at": "2025-01-15",
        "verified": True
    }
)

# ❌ OK but incomplete: Content only
memory.update(memory_id, data="Updated content")
# Metadata remains outdated
```

### 3. Preserve Important Metadata

```python
# ✅ Good: Preserve existing metadata
current = memory.get(memory_id)
existing_metadata = current.get('metadata', {})

memory.update(
    memory_id=memory_id,
    data="New content",
    metadata={
        **existing_metadata,  # Preserve existing
        "updated_at": "2025-01-15"  # Add new
    }
)

# ❌ Bad: Overwrite all metadata
memory.update(
    memory_id=memory_id,
    data="New content",
    metadata={"updated_at": "2025-01-15"}  # Lost all other metadata
)
```

### 4. Use Descriptive Update Comments

```python
# ✅ Good: Track why update happened
memory.update(
    memory_id=memory_id,
    data="Updated preference",
    metadata={
        "updated_at": "2025-01-15",
        "update_reason": "User corrected their preference",
        "updated_by": "user_input"
    }
)
```

### 5. Consider Version History for Critical Data

```python
# ✅ Good: Keep version history for important data
current = memory.get(memory_id)
version_history = current.get('metadata', {}).get('versions', [])
version_history.append({
    "data": current['memory'],
    "timestamp": "2025-01-15T10:00:00"
})

memory.update(
    memory_id=memory_id,
    data="New version",
    metadata={"versions": version_history}
)
```

---

## 🐛 Troubleshooting

### Issue: Memory ID Not Found

```python
# Problem: Invalid memory ID
try:
    memory.update(memory_id="invalid_id", data="New data")
except Exception as e:
    print(f"Update failed: {e}")

# Solution: Verify memory exists first
memory_id = "mem_123"
try:
    current = memory.get(memory_id)
    memory.update(memory_id, data="New data")
    print("✅ Update successful")
except Exception as e:
    print(f"❌ Memory not found: {e}")
```

### Issue: Update Doesn't Seem to Work

```python
# Verify update by reading back
memory_id = "mem_123"

print("Before update:")
before = memory.get(memory_id)
print(f"  Data: {before['memory']}")

# Update
memory.update(memory_id, data="New data")

print("\nAfter update:")
after = memory.get(memory_id)
print(f"  Data: {after['memory']}")

# Check if changed
if before['memory'] != after['memory']:
    print("✅ Update successful")
else:
    print("❌ Update failed")
```

### Issue: Metadata Gets Overwritten

```python
# ❌ Problem: Metadata overwritten
memory.update(
    memory_id=memory_id,
    data="New data",
    metadata={"new_field": "value"}  # Loses all existing metadata
)

# ✅ Solution: Merge with existing metadata
current = memory.get(memory_id)
existing_metadata = current.get('metadata', {})

memory.update(
    memory_id=memory_id,
    data="New data",
    metadata={
        **existing_metadata,  # Keep existing
        "new_field": "value"  # Add new
    }
)
```

### Issue: Can't Find Memory to Update

```python
# Problem: Can't find the right memory
results = memory.search("vague query", user_id="alice", limit=10)
print(f"Found {len(results['results'])} potential matches")

# Solution: Use more specific search + metadata
results = memory.search(
    "specific preference",
    user_id="alice",
    filters={"category": "preferences"}
)

if results['results']:
    memory_id = results['results'][0]['id']
    print(f"Found target memory: {memory_id}")
```

---

## 📊 Performance Tips

### 1. Batch Updates Efficiently

```python
# ✅ Efficient: Collect IDs first, then update
memories_to_update = memory.get_all(user_id="alice")
ids_to_update = [
    m['id'] for m in memories_to_update['results']
    if m.get('metadata', {}).get('category') == 'old_category'
]

for memory_id in ids_to_update:
    memory.update(
        memory_id=memory_id,
        data=updated_content,
        metadata={"category": "new_category"}
    )

# ❌ Inefficient: Search for each update
for item in items:
    results = memory.search(item, user_id="alice")
    if results['results']:
        memory.update(results['results'][0]['id'], data=item)
```

### 2. Update Only When Necessary

```python
# ✅ Efficient: Check if update needed
current = memory.get(memory_id)

if current['memory'] != new_value:
    memory.update(memory_id, data=new_value)
    print("✅ Updated")
else:
    print("ℹ️  No update needed")

# ❌ Inefficient: Always update
memory.update(memory_id, data=new_value)
```

### 3. Use Metadata to Track Updates

```python
# ✅ Efficient: Track last update time
current = memory.get(memory_id)
last_updated = current.get('metadata', {}).get('updated_at')

if should_update(last_updated):
    memory.update(
        memory_id=memory_id,
        data=new_value,
        metadata={"updated_at": "2025-01-15"}
    )
```

---

## 📚 API Reference

### Memory.update()

```python
def update(
    memory_id: str,
    data: str,
    metadata: Dict = None
) -> Dict
```

**Parameters:**
- **memory_id** (str): ID of the memory to update
- **data** (str): New content for the memory
- **metadata** (Dict, optional): New metadata (overwrites existing unless merged)

**Returns:**
- Dict with updated memory information

**Example:**
```python
# Update with new content and metadata
memory.update(
    memory_id="mem_abc123",
    data="Updated preference: Alice prefers tea over coffee",
    metadata={
        "category": "beverages",
        "updated_at": "2025-01-15",
        "verified": True
    }
)
```

**Notes:**
- Memory ID must exist (use `get()` or `search()` first)
- Metadata is replaced, not merged (preserve existing metadata manually)
- Update is immediate and permanent

---

## 🔗 Related Guides

- **[Add Memory Guide](./add.md)** - Store new memories
- **[Search Memory Guide](./search.md)** - Find memories to update
- **[Delete Memory Guide](./delete.md)** - Remove memories
- **[Memory Types Guide](../memory-types.md)** - Understand memory layers
- **[Python User Guide](../PYTHON_USER_GUIDE.md)** - Complete Python guide

---

## 📝 Quick Reference

```python
# Find memory to update
results = memory.search("query", user_id="alice")
memory_id = results['results'][0]['id']

# Basic update
memory.update(memory_id, data="New content")

# Update with metadata
memory.update(
    memory_id=memory_id,
    data="New content",
    metadata={"category": "updated"}
)

# Preserve existing metadata
current = memory.get(memory_id)
memory.update(
    memory_id=memory_id,
    data="New content",
    metadata={
        **current.get('metadata', {}),
        "new_field": "value"
    }
)

# Verify update
updated = memory.get(memory_id)
print(updated['memory'])
```

---

## 💡 When to Update vs Delete+Re-add

**Use Update When:**
- ✅ Correcting a typo or minor error
- ✅ Adding metadata to existing memory
- ✅ Enriching memory with more context
- ✅ Preserving memory ID is important
- ✅ Tracking version history

**Use Delete + Re-add When:**
- ✅ Complete change in meaning
- ✅ Different memory structure needed
- ✅ Want LLM to re-extract facts (infer=True)
- ✅ Memory ID doesn't matter

**Example:**
```python
# Update: Minor correction
memory.update(memory_id, data="Alice prefers Italian food (especially pasta)")

# Delete + Re-add: Complete change
memory.delete(memory_id)
memory.add("Alice is now vegan and avoids all animal products", user_id="alice")
```

---

**Need help?** Check the [Troubleshooting Guide](../../TROUBLESHOOTING.md) or open an issue on [GitHub](https://github.com/outhad/outhad_contextkit/issues).
