# Memory Operations: Delete Memory - Python Guide

## 🎯 Overview

The **delete** operation removes memories from Outhad_ContextKit, whether individually, in bulk, or by filters. Essential for privacy compliance (GDPR/CCPA), data cleanup, session management, and removing incorrect or outdated information.

**Why it matters:**
- Satisfies "right to be forgotten" compliance requirements
- Keeps knowledge bases accurate by removing stale data
- Manages memory growth and storage costs
- Cleans up temporary session data
- Removes incorrect or sensitive information

---

## 🚀 Quick Start

### 1. Import and Initialize

```python
from outhad_contextkit import Memory

# Initialize memory
memory = Memory()
```

### 2. Basic Delete Examples

```python
# Delete single memory by ID
results = memory.search("old preference", user_id="alice")
memory_id = results['results'][0]['id']
memory.delete(memory_id=memory_id)
print(f"✅ Deleted memory: {memory_id}")

# Delete all memories for a user (GDPR compliance)
memory.delete_all(user_id="alice")
print("✅ Deleted all memories for user alice")

# Delete session memories
memory.delete_all(user_id="alice", session_id="temp_session_001")
print("✅ Deleted all session memories")
```

---

## 📖 Core Features

### Feature 1: Delete by Memory ID

Delete a specific memory when you know its ID:

```python
# Find memory to delete
results = memory.search("outdated preference", user_id="alice")

if results['results']:
    memory_id = results['results'][0]['id']
    old_memory = results['results'][0]['memory']

    # Delete it
    memory.delete(memory_id=memory_id)

    print(f"✅ Deleted: {old_memory}")

    # Verify deletion
    try:
        deleted = memory.get(memory_id)
        print("⚠️  Memory still exists")
    except:
        print("✅ Memory successfully deleted")
```

---

### Feature 2: Delete All Memories for User

Complete user data removal (GDPR "right to be forgotten"):

```python
def forget_user(user_id: str):
    """Complete user data deletion for GDPR compliance."""

    # Get count before deletion
    before = memory.get_all(user_id=user_id)
    count = len(before['results'])

    # Delete all user memories
    memory.delete_all(user_id=user_id)

    print(f"✅ Deleted {count} memories for user {user_id}")
    print("📜 GDPR compliance: Right to be forgotten executed")

    # Verify deletion
    after = memory.get_all(user_id=user_id)
    if len(after['results']) == 0:
        print("✅ Verification: All user data removed")
    else:
        print(f"⚠️  Warning: {len(after['results'])} memories remain")

# Usage
forget_user(user_id="alice")
```

---

### Feature 3: Delete by Session

Remove temporary session data after task completion:

```python
def cleanup_session(user_id: str, session_id: str):
    """Clean up temporary session memories."""

    # Get session memories count
    session_memories = memory.search(
        query="*",
        user_id=user_id,
        session_id=session_id,
        limit=100
    )
    count = len(session_memories['results'])

    # Delete session
    memory.delete_all(user_id=user_id, session_id=session_id)

    print(f"✅ Cleaned up session {session_id}")
    print(f"   Deleted {count} temporary memories")

# Usage
cleanup_session(
    user_id="dev_team",
    session_id="debug_session_001"
)
```

---

### Feature 4: Delete by Agent/Run

Remove agent-specific or run-specific memories:

```python
# Delete specific agent's memories
def cleanup_agent_memories(agent_id: str, run_id: str):
    """Remove memories for a specific agent in a run."""

    memory.delete_all(agent_id=agent_id, run_id=run_id)
    print(f"✅ Deleted memories for {agent_id} in run {run_id}")

# Delete entire run (all agents)
def cleanup_run(run_id: str):
    """Remove all memories from a run."""

    # Note: This deletes memories from ALL agents in the run
    memory.delete_all(run_id=run_id)
    print(f"✅ Deleted all memories from run {run_id}")

# Usage
cleanup_agent_memories(agent_id="research_agent", run_id="audit_2025_01")
cleanup_run(run_id="audit_2025_01")
```

---

### Feature 5: Selective Delete with Search

Delete memories matching specific criteria:

```python
def delete_by_criteria(query: str, user_id: str, filters: dict = None):
    """Delete memories matching search criteria."""

    # Find memories to delete
    results = memory.search(
        query=query,
        user_id=user_id,
        filters=filters,
        limit=100
    )

    deleted_count = 0

    # Delete each matching memory
    for result in results['results']:
        memory_id = result['id']
        memory.delete(memory_id=memory_id)
        deleted_count += 1
        print(f"  Deleted: {result['memory'][:50]}...")

    print(f"✅ Deleted {deleted_count} memories")
    return deleted_count

# Usage examples

# Delete all food preferences
delete_by_criteria(
    query="food",
    user_id="alice",
    filters={"category": "food_preferences"}
)

# Delete all low-priority memories
delete_by_criteria(
    query="*",
    user_id="alice",
    filters={"priority": "low"}
)

# Delete all temporary memories
delete_by_criteria(
    query="*",
    user_id="alice",
    filters={"temporary": True}
)
```

---

## 🎯 Common Use Cases

### Use Case 1: GDPR Compliance - Right to be Forgotten

```python
def process_gdpr_request(user_id: str, reason: str = "user_request"):
    """Process GDPR right to be forgotten request."""

    print(f"🔒 Processing GDPR deletion for user: {user_id}")
    print(f"   Reason: {reason}")

    # 1. Get count of memories
    all_memories = memory.get_all(user_id=user_id)
    memory_count = len(all_memories['results'])

    print(f"   Found {memory_count} memories to delete")

    # 2. Delete all user data
    memory.delete_all(user_id=user_id)

    # 3. Verify deletion
    verification = memory.get_all(user_id=user_id)

    if len(verification['results']) == 0:
        print(f"✅ GDPR Request Complete")
        print(f"   Deleted {memory_count} memories")
        print(f"   User data fully removed")
        return True
    else:
        print(f"⚠️  GDPR Request Incomplete")
        print(f"   {len(verification['results'])} memories remain")
        return False

# Usage
process_gdpr_request(
    user_id="alice",
    reason="User requested account deletion"
)
```

---

### Use Case 2: Session Cleanup After Task Completion

```python
def complete_task_and_cleanup(user_id: str, session_id: str, task_result: dict):
    """Complete a task and clean up temporary session data."""

    print(f"📋 Completing task for session {session_id}")

    # 1. Store final result (permanent memory)
    memory.add(
        f"Task completed: {task_result['summary']}",
        user_id=user_id,
        metadata={
            "category": "task_history",
            "session_id": session_id,
            "completed_at": "2025-01-15",
            "result": task_result['status']
        }
    )

    # 2. Clean up temporary session memories
    memory.delete_all(user_id=user_id, session_id=session_id)

    print(f"✅ Task complete and session cleaned up")

# Usage
complete_task_and_cleanup(
    user_id="dev_team",
    session_id="debug_session_001",
    task_result={
        "summary": "Fixed authentication timeout issue",
        "status": "success"
    }
)
```

---

### Use Case 3: Removing Incorrect or Outdated Information

```python
def remove_incorrect_information(query: str, user_id: str, reason: str):
    """Find and remove incorrect memories."""

    print(f"🔍 Searching for incorrect information: {query}")

    # Find memories to remove
    results = memory.search(query, user_id=user_id, limit=10)

    if not results['results']:
        print("No matching memories found")
        return

    print(f"Found {len(results['results'])} potentially incorrect memories")

    # Review and delete
    deleted_count = 0
    for result in results['results']:
        print(f"\n  Memory: {result['memory']}")
        print(f"  Reason: {reason}")

        # Delete
        memory.delete(memory_id=result['id'])
        deleted_count += 1

    print(f"\n✅ Removed {deleted_count} incorrect memories")

# Usage
remove_incorrect_information(
    query="Alice loves horror movies",
    user_id="alice",
    reason="User corrected preference - actually prefers sci-fi"
)
```

---

### Use Case 4: Temporary Data Expiration

```python
from datetime import datetime, timedelta

def cleanup_expired_memories(user_id: str):
    """Delete memories that have passed their expiration date."""

    # Get all memories
    all_memories = memory.get_all(user_id=user_id)

    now = datetime.utcnow()
    deleted_count = 0

    for mem in all_memories['results']:
        metadata = mem.get('metadata', {})
        expires_at = metadata.get('expires_at')

        if expires_at:
            expiry_date = datetime.fromisoformat(expires_at)

            # Check if expired
            if now > expiry_date:
                print(f"  Expired: {mem['memory'][:50]}...")
                memory.delete(memory_id=mem['id'])
                deleted_count += 1

    print(f"✅ Deleted {deleted_count} expired memories")
    return deleted_count

# Usage
cleanup_expired_memories(user_id="alice")
```

---

### Use Case 5: Privacy-Safe Development Testing

```python
def create_test_user_data(user_id: str):
    """Create test data for development."""

    memories = [
        "Test preference 1",
        "Test preference 2",
        "Test preference 3"
    ]

    for mem in memories:
        memory.add(mem, user_id=user_id, metadata={"test": True})

    print(f"✅ Created {len(memories)} test memories")

def cleanup_test_data(user_id: str):
    """Remove all test data after development."""

    # Get test memories
    all_memories = memory.get_all(user_id=user_id)
    test_memories = [
        m for m in all_memories['results']
        if m.get('metadata', {}).get('test') == True
    ]

    # Delete each test memory
    for mem in test_memories:
        memory.delete(memory_id=mem['id'])

    print(f"✅ Cleaned up {len(test_memories)} test memories")

# Usage
create_test_user_data(user_id="test_user")
# ... run tests ...
cleanup_test_data(user_id="test_user")
```

---

## 🔍 Best Practices

### 1. Always Confirm Before Bulk Deletion

```python
# ✅ Good: Confirm before deleting all user data
def safe_delete_user(user_id: str):
    # Get count
    all_memories = memory.get_all(user_id=user_id)
    count = len(all_memories['results'])

    print(f"⚠️  About to delete {count} memories for {user_id}")
    confirm = input("Type 'DELETE' to confirm: ")

    if confirm == "DELETE":
        memory.delete_all(user_id=user_id)
        print(f"✅ Deleted {count} memories")
    else:
        print("❌ Deletion cancelled")

# ❌ Dangerous: No confirmation
memory.delete_all(user_id="alice")  # Irreversible!
```

### 2. Log Deletions for Audit Trail

```python
# ✅ Good: Log all deletions
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_with_audit(memory_id: str, reason: str):
    # Get memory before deletion
    mem = memory.get(memory_id)

    # Delete
    memory.delete(memory_id=memory_id)

    # Log
    logger.info(f"DELETED: {memory_id} | Content: {mem['memory'][:50]} | Reason: {reason}")

# Usage
delete_with_audit(
    memory_id="mem_123",
    reason="GDPR request #45678"
)
```

### 3. Verify Deletion After Operation

```python
# ✅ Good: Verify deletion succeeded
memory_id = "mem_123"

memory.delete(memory_id=memory_id)

# Verify
try:
    deleted = memory.get(memory_id)
    print("⚠️  Delete failed - memory still exists")
except:
    print("✅ Delete confirmed")
```

### 4. Use Specific Filters, Not delete_all() Without Filters

```python
# ✅ Good: Always provide filters
memory.delete_all(user_id="alice")  # Scoped to user
memory.delete_all(user_id="alice", session_id="session_1")  # More specific

# ❌ Dangerous: No filters (will fail with error for safety)
memory.delete_all()  # ERROR: Requires at least one filter
```

### 5. Clean Up Sessions Regularly

```python
# ✅ Good: Regular session cleanup
def cleanup_old_sessions(user_id: str, active_session_ids: list):
    """Remove old session data, keep active ones."""

    all_memories = memory.get_all(user_id=user_id)

    for mem in all_memories['results']:
        metadata = mem.get('metadata', {})
        session_id = metadata.get('session_id')

        # Delete if it's a session memory and not active
        if session_id and session_id not in active_session_ids:
            memory.delete(memory_id=mem['id'])

# Usage
cleanup_old_sessions(
    user_id="alice",
    active_session_ids=["session_current"]
)
```

---

## 🐛 Troubleshooting

### Issue: Delete Doesn't Seem to Work

```python
# Verify memory exists before deletion
memory_id = "mem_123"

print("Before deletion:")
try:
    before = memory.get(memory_id)
    print(f"  Memory exists: {before['memory']}")
except:
    print("  Memory doesn't exist")

# Delete
memory.delete(memory_id=memory_id)

print("\nAfter deletion:")
try:
    after = memory.get(memory_id)
    print(f"  ⚠️  Memory still exists: {after['memory']}")
except:
    print("  ✅ Memory successfully deleted")
```

### Issue: delete_all() Fails

```python
# Problem: No filters provided
try:
    memory.delete_all()  # ERROR
except Exception as e:
    print(f"❌ Error: {e}")
    print("Delete_all requires at least one filter")

# Solution: Always provide filter
memory.delete_all(user_id="alice")  # ✅ Works
```

### Issue: Can't Find Memory to Delete

```python
# Problem: Memory ID not found
memory_id = "mem_nonexistent"

try:
    memory.delete(memory_id=memory_id)
except Exception as e:
    print(f"❌ Delete failed: {e}")

# Solution: Search first, then delete
results = memory.search("query", user_id="alice")

if results['results']:
    memory_id = results['results'][0]['id']
    memory.delete(memory_id=memory_id)
    print("✅ Memory deleted")
else:
    print("Memory not found")
```

### Issue: Partial Deletion in Bulk Operation

```python
# Problem: Some deletions succeed, some fail
def safe_bulk_delete(memory_ids: list):
    """Delete multiple memories with error handling."""

    success_count = 0
    error_count = 0
    errors = []

    for memory_id in memory_ids:
        try:
            memory.delete(memory_id=memory_id)
            success_count += 1
        except Exception as e:
            error_count += 1
            errors.append({"id": memory_id, "error": str(e)})

    print(f"✅ Deleted: {success_count}")
    print(f"❌ Failed: {error_count}")

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  {err['id']}: {err['error']}")

# Usage
memory_ids = ["mem_1", "mem_2", "mem_invalid", "mem_3"]
safe_bulk_delete(memory_ids)
```

---

## ⚠️ Important Warnings

### 1. Deletion is Permanent

```python
# ⚠️  WARNING: No undo operation exists
memory.delete(memory_id="mem_123")
# This memory is GONE FOREVER - cannot be recovered

# Best practice: Export before deletion if needed
mem = memory.get("mem_123")
backup = mem['memory']  # Save if you might need it
memory.delete("mem_123")
```

### 2. delete_all() is Powerful and Dangerous

```python
# ⚠️  DANGER: Deletes ALL user memories permanently
memory.delete_all(user_id="alice")
# All of Alice's data is GONE

# Use with extreme caution
# Always confirm with user first
# Log for audit trail
```

### 3. No Built-in Trash/Recycle Bin

```python
# ⚠️  No trash bin - deletion is immediate and permanent
memory.delete(memory_id="mem_123")

# If you need recovery, implement your own:
def delete_with_backup(memory_id: str):
    # Backup before deletion
    mem = memory.get(memory_id)
    save_to_backup_storage(mem)  # Your backup system

    # Then delete
    memory.delete(memory_id=memory_id)
```

---

## 📊 Performance Tips

### 1. Batch Delete with IDs

```python
# ✅ Efficient: Delete multiple in loop
memory_ids = ["mem_1", "mem_2", "mem_3"]
for memory_id in memory_ids:
    memory.delete(memory_id=memory_id)

# Note: If deleting many memories with same filter, use delete_all()
memory.delete_all(user_id="alice", filters={"category": "temporary"})
```

### 2. Use delete_all() for Large Deletions

```python
# ✅ Efficient: Use delete_all for many memories
memory.delete_all(user_id="alice", session_id="old_session")

# ❌ Inefficient: Delete one by one
results = memory.search("*", user_id="alice", session_id="old_session", limit=1000)
for result in results['results']:
    memory.delete(memory_id=result['id'])  # Slow!
```

### 3. Schedule Regular Cleanup

```python
# ✅ Good: Automated cleanup
def scheduled_cleanup():
    """Run regular memory cleanup."""

    # Clean up expired memories
    cleanup_expired_memories(user_id="alice")

    # Clean up old sessions
    cleanup_old_sessions(user_id="alice", active_session_ids=["current"])

    # Clean up low-priority memories older than 30 days
    # (Implement your logic here)

# Run as cron job or scheduled task
```

---

## 📚 API Reference

### Memory.delete()

```python
def delete(memory_id: str) -> None
```

**Parameters:**
- **memory_id** (str): ID of the memory to delete

**Returns:**
- None

**Raises:**
- Exception if memory_id doesn't exist

**Example:**
```python
results = memory.search("old preference", user_id="alice")
memory_id = results['results'][0]['id']
memory.delete(memory_id=memory_id)
```

---

### Memory.delete_all()

```python
def delete_all(
    user_id: str = None,
    session_id: str = None,
    agent_id: str = None,
    run_id: str = None
) -> None
```

**Parameters:**
- **user_id** (str, optional): Delete all memories for user
- **session_id** (str, optional): Delete all memories for session
- **agent_id** (str, optional): Delete all memories for agent
- **run_id** (str, optional): Delete all memories for run

**Returns:**
- None

**Requires:**
- At least one parameter must be provided

**Example:**
```python
# Delete all user memories
memory.delete_all(user_id="alice")

# Delete session memories
memory.delete_all(user_id="alice", session_id="session_1")

# Delete agent memories
memory.delete_all(agent_id="research_agent", run_id="audit_2025")
```

---

## 🔗 Related Guides

- **[Add Memory Guide](./add.md)** - Store memories
- **[Search Memory Guide](./search.md)** - Find memories to delete
- **[Update Memory Guide](./update.md)** - Modify instead of deleting
- **[Memory Types Guide](../memory-types.md)** - Understand memory layers
- **[Python User Guide](../PYTHON_USER_GUIDE.md)** - Complete Python guide

---

## 📝 Quick Reference

```python
# Delete by ID
memory.delete(memory_id="mem_123")

# Delete all for user (GDPR)
memory.delete_all(user_id="alice")

# Delete session
memory.delete_all(user_id="alice", session_id="session_1")

# Delete agent/run
memory.delete_all(agent_id="agent_1", run_id="run_1")

# Delete with search
results = memory.search("old data", user_id="alice")
for result in results['results']:
    memory.delete(memory_id=result['id'])

# Verify deletion
try:
    memory.get("mem_123")
    print("Still exists")
except:
    print("Deleted successfully")
```

---

## 💡 Delete vs Update

**Use Delete When:**
- ✅ GDPR/CCPA compliance (right to be forgotten)
- ✅ Cleaning up temporary session data
- ✅ Removing completely incorrect information
- ✅ User requests data removal
- ✅ Expired/outdated data that's no longer relevant

**Use Update When:**
- ✅ Correcting a typo or minor error
- ✅ Enriching existing memory with more context
- ✅ Changing a preference (e.g., email address)
- ✅ Adding metadata without losing the memory
- ✅ Preserving memory ID and relationships

---

**Need help?** Check the [Troubleshooting Guide](../../TROUBLESHOOTING.md) or open an issue on [GitHub](https://github.com/outhad/outhad_contextkit/issues).
