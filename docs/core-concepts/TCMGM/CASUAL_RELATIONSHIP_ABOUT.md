# Causal Relationships 

## 🎯 Summary

TCMGM (Temporal-Causal Multimodal Graph Memory) has been successfully implemented! This adds causal relationship extraction and analysis capabilities, enabling the system to track cause-effect chains between events.

---

## ✅ What We have
### 1. Causal Relationship Extractor



Have `CausalExtractor` class with dual extraction methods:

#### LLM-Based Extraction
- Uses LLM to identify cause-effect relationships
- Extracts causal type (caused_by, leads_to, enables, prevents, correlates_with)
- Provides confidence scores and evidence
- Returns structured `CausalLink` objects

#### Rule-Based Extraction
- Keyword-based heuristics for causal relationships
- Temporal proximity analysis
- Fallback when LLM is unavailable
- Automatic deduplication

### 2. Causal Link Storage


 `add_causal_link()` method:
- Stores causal relationships as CAUSAL edges in Neo4j
- Supports both CausalLink objects and dictionaries
- Includes metadata: type, confidence, evidence, timestamp
- Agent and user filtering support

### 3. Causal Chain Queries



three powerful query functions:

#### `get_causal_chain()`
- Traces causal chains forward (effects) or backward (causes)
- Configurable max depth
- Returns full chain with events and causal links

#### `find_root_causes()`
- Finds root causes (events with no incoming causal links)
- Minimum confidence threshold filtering
- Returns causes with distance metrics

#### `get_causal_subgraph()`
- Builds causal subgraph connecting multiple events
- Useful for visualizing relationships
- Returns nodes and relationships

### 4. Module Integration

export all causal components by this:
- `CausalExtractor`
- `get_causal_chain`
- `find_root_causes`
- `get_causal_subgraph`


### 6. Integration Testing with Real LLM

**File**: `test_causal_with_api.py`

comprehensive integration test:
- ✅ LLM-based causal extraction with real OpenAI API
- ✅ Tested on realistic event flows (login, error handling)
- ✅ Extracted 6 causal links
- ✅ Average confidence: 0.88
- ✅ Multiple causal types detected (leads_to, enables, caused_by)

---



## 🎯 Key Features Have

✓ **Dual Extraction Methods** - LLM and rule-based approaches  
✓ **Causal Types** - 5 types (caused_by, leads_to, enables, prevents, correlates_with)  
✓ **Neo4j Storage** - CAUSAL relationship edges with metadata  
✓ **Chain Queries** - Forward and backward causal traversal  
✓ **Root Cause Analysis** - Find origin events  
✓ **Subgraph Building** - Connect multiple events  
✓ **Confidence Scoring** - 0.0-1.0 validated scores  
✓ **Evidence Tracking** - Store reasoning for causal links  


---


## 🚀 Usage Examples

### Extract Causal Relationships

```python
from outhad_contextkit.memory.temporal import CausalExtractor
from outhad_contextkit.utils.factory import LlmFactory

# Initialize LLM
llm = LlmFactory.create("openai", {"model": "gpt-4o-mini"})

# Create extractor
extractor = CausalExtractor(llm=llm)

# Define events
events = [
    {"id": "e1", "content": "User logged in", "timestamp": "2025-01-14T10:00:00"},
    {"id": "e2", "content": "Dashboard loaded", "timestamp": "2025-01-14T10:00:05"}
]

# Extract causal links with LLM
links = extractor.extract_causal_links(events, use_llm=True)

for link in links:
    print(f"{link.cause_id} → {link.effect_id}")
    print(f"Type: {link.causal_type}, Confidence: {link.confidence}")
    print(f"Evidence: {link.evidence}")
```

### Store Causal Links

```python
from outhad_contextkit.memory.graph_memory import MemoryGraph
from datetime import datetime

# Add causal link to graph
memory_graph.add_causal_link(
    causal_link=links[0],
    filters={"user_id": "user_123"}
)
```

### Query Causal Chains

```python
from outhad_contextkit.memory.temporal import get_causal_chain

# Get forward chain (effects)
effects = get_causal_chain(
    graph=memory_graph.graph,
    start_event_id="event_login",
    filters={"user_id": "user_123"},
    max_depth=5,
    direction="forward"
)

# Get backward chain (causes)
causes = get_causal_chain(
    graph=memory_graph.graph,
    start_event_id="event_error",
    filters={"user_id": "user_123"},
    max_depth=5,
    direction="backward"
)
```

### Find Root Causes

```python
from outhad_contextkit.memory.temporal import find_root_causes

# Find what caused an event
root_causes = find_root_causes(
    graph=memory_graph.graph,
    effect_event_id="event_crash",
    filters={"user_id": "user_123"},
    min_confidence=0.7
)

for cause in root_causes:
    print(f"Root cause: {cause['root']}")
    print(f"Distance: {cause['distance']} steps")
```

---



## 🎓 Causal Types

 supports 5 causal relationship types:

| Type | Value | Description | Example |
|------|-------|-------------|---------|
| **CAUSED_BY** | `caused_by` | Direct causation | "Error caused by invalid input" |
| **LEADS_TO** | `leads_to` | Forward causation | "Login led to dashboard load" |
| **ENABLES** | `enables` | Enabling condition | "Auth enables data access" |
| **PREVENTS** | `prevents` | Preventive relationship | "Firewall prevented attack" |
| **CORRELATES_WITH** | `correlates_with` | Correlation | "Traffic correlates with errors" |

---

## 📈 LLM Integration Test Results

### Test Case 1: Login Flow
**Events**: 4 events (login page → credentials → auth → dashboard)  
**Links Extracted**: 3 causal links  
**Types**: leads_to (2x), enables (1x)  
**Confidence**: 0.90 average

### Test Case 2: Error Handling
**Events**: 4 events (connection fail → error msg → retry → restored)  
**Links Extracted**: 3 causal links  
**Types**: leads_to, caused_by, enables  
**Confidence**: 0.85 average

### Overall Statistics
- **Total Links**: 6
- **Average Confidence**: 0.88
- **Unique Causal Types**: 3
- **All Confidence Scores**: Valid (0.80-0.90 range)

---
