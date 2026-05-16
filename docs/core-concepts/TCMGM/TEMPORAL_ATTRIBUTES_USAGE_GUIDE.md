#  TCMGM: Quick Usage Guide

## 🚀 Getting Started with Temporal Attributes

### 1. Import Temporal Types

```python
from datetime import datetime
from outhad_contextkit.memory.temporal import (
    TemporalEvent,
    TemporalRelation,
    TimeWindow,
    CausalLink,
    ModalityType,
    CausalType,
    TemporalRelationType
)
```

### 2. Create Temporal Events

#### Basic Text Event
```python
event = TemporalEvent(
    id="event_001",
    content="User logged in to the system",
    timestamp=datetime.utcnow(),
    confidence=0.95,
    modality="text",
    user_id="user_123"
)
```

#### Image Event
```python
image_event = TemporalEvent(
    id="event_002",
    content="User uploaded profile picture",
    timestamp=datetime.utcnow(),
    confidence=0.9,
    modality="image",
    image_hash="a1b2c3d4e5f6",
    metadata={"image_size": "1024x768", "format": "jpg"}
)
```

#### Audio Event
```python
audio_event = TemporalEvent(
    id="event_003",
    content="Voice command recorded",
    timestamp=datetime.utcnow(),
    confidence=0.85,
    modality="audio",
    audio_hash="x9y8z7w6v5u4",
    metadata={"duration_seconds": 5, "format": "mp3"}
)
```

### 3. Use with MemoryGraph

#### Add Temporal Data to Graph (Backward Compatible)

```python
from outhad_contextkit.memory.graph_memory import MemoryGraph
from outhad_contextkit.configs.base import MemoryConfig

# Initialize
config = MemoryConfig()
memory_graph = MemoryGraph(config)

# Add with temporal attributes (NEW!)
memory_graph.add(
    data="User completed onboarding process",
    filters={"user_id": "user_123"},
    timestamp=datetime.utcnow(),      # Optional: defaults to now
    confidence=0.9,                    # Optional: defaults to 1.0
    modality="text",                   # Optional: defaults to "text"
    metadata={"step": 5}               # Optional: extra metadata
)

# Still works without temporal params (backward compatible)
memory_graph.add(
    data="User viewed dashboard",
    filters={"user_id": "user_123"}
)
```

#### Add Image Event to Graph
```python
memory_graph.add(
    data="User uploaded document scan",
    filters={"user_id": "user_123"},
    timestamp=datetime.utcnow(),
    confidence=0.88,
    modality="image",
    image_hash="abc123def456",
    metadata={"filename": "document.jpg", "size_kb": 2048}
)
```

### 4. Time Windows for Queries

#### Last Hour Events
```python
from datetime import timedelta

now = datetime.utcnow()
last_hour = TimeWindow(
    start=now - timedelta(hours=1),
    end=now
)

# Check if event is recent
if last_hour.contains(event.timestamp):
    print("Event is from the last hour!")
```

#### Open-Ended Time Windows
```python
# All events after a certain date
since_launch = TimeWindow(
    start=datetime(2025, 1, 1)  # No end date
)

# All events before a certain date
before_deadline = TimeWindow(
    end=datetime(2025, 12, 31)  # No start date
)

# All events (unbounded)
all_time = TimeWindow()
```

### 5. Temporal Relations

```python
relation = TemporalRelation(
    source_id="event_001",
    target_id="event_002",
    relation_type="precedes",
    timestamp=datetime.utcnow(),
    confidence=0.95,
    temporal_type=TemporalRelationType.BEFORE.value,
    causal_type=CausalType.LEADS_TO.value,
    metadata={"verified": True}
)
```

### 6. Causal Links

```python
causal_link = CausalLink(
    cause_id="event_login",
    effect_id="event_dashboard",
    causal_type=CausalType.CAUSED_BY.value,
    confidence=0.92,
    evidence="Login event directly triggered dashboard load",
    timestamp=datetime.utcnow()
)
```

## 🎯 Common Use Cases

### Use Case 1: Timeline of User Actions

```python
from datetime import datetime, timedelta

# Record sequence of events
events = []

# Event 1: Login
events.append(memory_graph.add(
    data="User logged in",
    filters={"user_id": "user_123"},
    timestamp=datetime.utcnow(),
    confidence=1.0
))

# Event 2: View product (2 minutes later)
events.append(memory_graph.add(
    data="User viewed product page",
    filters={"user_id": "user_123"},
    timestamp=datetime.utcnow() + timedelta(minutes=2),
    confidence=0.95
))

# Event 3: Add to cart (5 minutes later)
events.append(memory_graph.add(
    data="User added item to cart",
    filters={"user_id": "user_123"},
    timestamp=datetime.utcnow() + timedelta(minutes=5),
    confidence=0.98
))
```

### Use Case 2: Multimodal Conversation

```python
# Text message
memory_graph.add(
    data="User: Show me images of cats",
    filters={"user_id": "user_456"},
    timestamp=datetime.utcnow(),
    confidence=1.0,
    modality="text"
)

# Image response
memory_graph.add(
    data="Assistant: Showed cat images",
    filters={"user_id": "user_456"},
    timestamp=datetime.utcnow() + timedelta(seconds=1),
    confidence=0.9,
    modality="image",
    image_hash="cat_images_hash_123",
    metadata={"image_count": 5}
)

# Follow-up text
memory_graph.add(
    data="User: I like the third one",
    filters={"user_id": "user_456"},
    timestamp=datetime.utcnow() + timedelta(seconds=10),
    confidence=1.0,
    modality="text"
)
```

### Use Case 3: Confidence-Based Filtering

```python
# High confidence events
high_conf_event = TemporalEvent(
    id="verified_001",
    content="Payment confirmed by bank",
    timestamp=datetime.utcnow(),
    confidence=1.0,  # Verified
    modality="text"
)

# Medium confidence events
medium_conf_event = TemporalEvent(
    id="detected_001",
    content="User sentiment detected as positive",
    timestamp=datetime.utcnow(),
    confidence=0.75,  # ML prediction
    modality="text"
)

# Lower confidence events
low_conf_event = TemporalEvent(
    id="inferred_001",
    content="User may be interested in premium features",
    timestamp=datetime.utcnow(),
    confidence=0.6,  # Weak inference
    modality="text"
)
```

## 📝 Best Practices

### 1. Use Appropriate Confidence Scores
- **1.0**: Verified facts (user confirmed, system verified)
- **0.9-0.95**: High confidence (direct observation)
- **0.7-0.85**: Medium confidence (ML predictions)
- **0.5-0.65**: Low confidence (inferences, weak signals)

### 2. Always Provide Timestamps
```python
# Good: Explicit timestamp
memory_graph.add(
    data="Event occurred",
    filters={"user_id": "123"},
    timestamp=datetime.utcnow()  # Explicit
)

# Also fine: Let it default
memory_graph.add(
    data="Event occurred",
    filters={"user_id": "123"}
    # timestamp defaults to now
)
```

### 3. Use Metadata for Context
```python
memory_graph.add(
    data="User uploaded file",
    filters={"user_id": "123"},
    timestamp=datetime.utcnow(),
    confidence=0.95,
    modality="image",
    image_hash="file_hash_789",
    metadata={
        "filename": "contract.pdf",
        "size_bytes": 2048000,
        "mime_type": "application/pdf",
        "source": "mobile_app"
    }
)
```

### 4. Leverage Time Windows for Queries
```python
# Session-based queries
session_window = TimeWindow(
    start=session_start_time,
    end=session_end_time
)

# Recent activity (last 24 hours)
recent = TimeWindow(
    start=datetime.utcnow() - timedelta(days=1)
)

# Historical analysis (all time before date)
historical = TimeWindow(
    end=datetime(2025, 1, 1)
)
```

## 🔍 Validation Examples

### Valid Temporal Events
```python
# All of these are valid
valid_events = [
    TemporalEvent(
        id="1", content="test", timestamp=datetime.utcnow(),
        confidence=0.0  # Valid: lower bound
    ),
    TemporalEvent(
        id="2", content="test", timestamp=datetime.utcnow(),
        confidence=0.5  # Valid: middle
    ),
    TemporalEvent(
        id="3", content="test", timestamp=datetime.utcnow(),
        confidence=1.0  # Valid: upper bound
    )
]
```

### Invalid Temporal Events
```python
# These will raise ValidationError
try:
    invalid_event = TemporalEvent(
        id="bad", content="test", timestamp=datetime.utcnow(),
        confidence=1.5  # Invalid: > 1.0
    )
except ValueError as e:
    print(f"Validation error: {e}")

try:
    invalid_event = TemporalEvent(
        id="bad", content="test", timestamp=datetime.utcnow(),
        confidence=-0.1  # Invalid: < 0.0
    )
except ValueError as e:
    print(f"Validation error: {e}")
```
