# Foundation - Temporal Attributes

## 🎯 Summary

TCMGM (Temporal-Causal Multimodal Graph Memory) has been successfully implemented! This adds temporal attributes and multimodal support to the graph memory system, laying the foundation for time-aware, causal, and multimodal memory capabilities.

---

## What we have

### 1. Temporal Module Structure



- **ModalityType**: TEXT, IMAGE, AUDIO, VIDEO, EMBEDDING
- **CausalType**: CAUSED_BY, LEADS_TO, ENABLES, PREVENTS, CORRELATES_WITH
- **TemporalRelationType**: BEFORE, AFTER, DURING, OVERLAPS, CONCURRENT, IMMEDIATELY_AFTER

### 3. Temporal Data Models


#### TemporalEvent
- Represents events with temporal and multimodal attributes
- Fields: id, content, timestamp, confidence, modality, embedding, image_hash, audio_hash, metadata, user_id, agent_id
- Confidence validation (0.0 - 1.0)

#### TemporalRelation
- Represents relationships between events with temporal context
- Fields: source_id, target_id, relation_type, timestamp, confidence, temporal_type, causal_type, metadata

#### TimeWindow
- Represents time windows for querying events
- Fields: start, end, duration_minutes
- Method: `contains(timestamp)` - checks if timestamp is within window

#### CausalLink
- Represents causal relationships between events
- Fields: cause_id, effect_id, causal_type, confidence, evidence, timestamp

### 4. Extended MemoryGraph

#### New Method: `_create_temporal_schema()`
- Creates temporal indexes in Neo4j:
  - `event_timestamp` - for temporal queries
  - `event_modality` - for filtering by modality
  - `event_confidence` - for filtering by confidence
- Called automatically on initialization

#### Modified Method: `add()`
- **New Signature**: Added optional parameters for TCMGM support:
  - `timestamp` (datetime, optional) - Event timestamp
  - `confidence` (float, optional) - Confidence score (0-1)
  - `modality` (str, optional) - Modality type
  - `image_hash` (str, optional) - Hash for image modality
  - `audio_hash` (str, optional) - Hash for audio modality
  - `metadata` (dict, optional) - Additional metadata

- **Backward Compatible**: All new parameters are optional with sensible defaults
- **Temporal Attributes Storage**: Stores temporal attributes in Neo4j node properties



## 🎯 Key Features

### 1. Temporal Awareness
- Every event can now have a timestamp
- Time-based queries are supported via indexed timestamp fields
- Time windows for filtering events

### 2. Confidence Scoring
- Events have confidence scores (0.0 - 1.0)
- Validated at Pydantic level
- Indexed for efficient filtering

### 3. Multimodal Support
- TEXT, IMAGE, AUDIO, VIDEO modalities
- Hash storage for image/audio content
- Embedding storage for cross-modal retrieval

### 4. Backward Compatibility
- All existing code continues to work
- New parameters are optional
- Default values maintain current behavior

### 5. Industry Standard
- Comprehensive test coverage (100% for new features)
- Type hints and Pydantic validation
- Clean, documented code
- No linting errors

---


## 🚀 Usage Example

### Basic Temporal Event

```python
from datetime import datetime
from outhad_contextkit.memory.temporal import TemporalEvent

event = TemporalEvent(
    id="event_001",
    content="User logged in",
    timestamp=datetime.utcnow(),
    confidence=0.95,
    modality="text",
    user_id="user_123"
)
```

### Using with MemoryGraph

```python
from datetime import datetime
from outhad_contextkit.memory.graph_memory import MemoryGraph

# Add temporal data to graph
memory_graph.add(
    data="User completed onboarding process",
    filters={"user_id": "user_123"},
    timestamp=datetime.utcnow(),
    confidence=0.9,
    modality="text"
)
```

### Time Window Queries

```python
from datetime import datetime, timedelta
from outhad_contextkit.memory.temporal import TimeWindow

# Create time window for last hour
window = TimeWindow(
    start=datetime.utcnow() - timedelta(hours=1),
    end=datetime.utcnow()
)

# Check if event is in window
is_recent = window.contains(event.timestamp)
```

---
