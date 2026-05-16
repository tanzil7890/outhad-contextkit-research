#  TCMGM: Causal Relationships - Usage Guide

## 🎯 Overview

 **causal relationship extraction and analysis** to TCMGM. This enables the system to understand and track cause-effect relationships between events, answer "why" questions, and perform root cause analysis.

---

## 🚀 Quick Start

### 1. Import Components

```python
from outhad_contextkit.memory.temporal import (
    CausalExtractor,
    CausalLink,
    CausalType,
    get_causal_chain,
    find_root_causes,
    get_causal_subgraph
)
from outhad_contextkit.utils.factory import LlmFactory
from outhad_contextkit.llms.configs import LlmConfig
from datetime import datetime
```

### 2. Initialize Causal Extractor

```python
# Initialize LLM
llm_config = LlmConfig(
    provider="openai",
    config={
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 2000
    }
)
llm = LlmFactory.create("openai", llm_config.config)

# Create extractor
extractor = CausalExtractor(llm=llm)
```

---

## 📖 Core Features

### Feature 1: Extract Causal Relationships

#### LLM-Based Extraction (Recommended)

```python
# Define events
events = [
    {
        "id": "event_001",
        "timestamp": "2025-01-14T10:00:00",
        "content": "User opened the login page"
    },
    {
        "id": "event_002",
        "timestamp": "2025-01-14T10:00:05",
        "content": "User entered username and password"
    },
    {
        "id": "event_003",
        "timestamp": "2025-01-14T10:00:10",
        "content": "Authentication succeeded"
    },
    {
        "id": "event_004",
        "timestamp": "2025-01-14T10:00:12",
        "content": "Dashboard was loaded"
    }
]

# Extract causal links using LLM
causal_links = extractor.extract_causal_links(events, use_llm=True)

# Print results
for link in causal_links:
    print(f"{link.cause_id} → {link.effect_id}")
    print(f"  Type: {link.causal_type}")
    print(f"  Confidence: {link.confidence:.2f}")
    print(f"  Evidence: {link.evidence}")
    print()
```

**Output Example:**
```
event_001 → event_002
  Type: leads_to
  Confidence: 0.90
  Evidence: User opening the login page is a prerequisite for entering credentials.

event_002 → event_003
  Type: leads_to
  Confidence: 0.90
  Evidence: Entering credentials is necessary for authentication to succeed.

event_003 → event_004
  Type: enables
  Confidence: 0.90
  Evidence: Successful authentication enables the loading of the dashboard.
```

#### Rule-Based Extraction (Fallback)

```python
# Extract using keyword-based rules
causal_links = extractor.extract_causal_links(events, use_llm=False)

# Works well when events contain causal keywords
events_with_keywords = [
    {"id": "e1", "content": "System overheated", "timestamp": "..."},
    {"id": "e2", "content": "Overheating caused system shutdown", "timestamp": "..."}
]

links = extractor.extract_causal_links(events_with_keywords, use_llm=False)
# Finds: e1 → e2 (caused_by relationship)
```

**Causal Keywords Detected:**
- `caused_by`: "because", "due to", "caused by", "resulted from"
- `leads_to`: "led to", "resulted in", "caused", "triggered"
- `enables`: "enabled", "allowed", "made possible", "facilitated"
- `prevents`: "prevented", "stopped", "blocked", "inhibited"

---

### Feature 2: Store Causal Links in Graph

```python
from outhad_contextkit.memory.graph_memory import MemoryGraph
from outhad_contextkit.configs.base import MemoryConfig

# Initialize memory graph
config = MemoryConfig()
memory_graph = MemoryGraph(config)

# Store each causal link
for link in causal_links:
    memory_graph.add_causal_link(
        causal_link=link,
        filters={"user_id": "user_123"}
    )

print(f"✅ Stored {len(causal_links)} causal links in Neo4j")
```

#### Store with Additional Metadata

```python
# Can also pass as dictionary
memory_graph.add_causal_link(
    causal_link={
        "cause_id": "event_login",
        "effect_id": "event_dashboard",
        "causal_type": "enables",
        "confidence": 0.95,
        "evidence": "Login enables dashboard access",
        "timestamp": datetime.utcnow()
    },
    filters={"user_id": "user_123", "agent_id": "agent_456"}
)
```

---

### Feature 3: Query Causal Chains

#### Forward Chain (Effects)

Find what happened **because** of an event:

```python
from outhad_contextkit.memory.temporal import get_causal_chain

# Get all effects of user login
effects = get_causal_chain(
    graph=memory_graph.graph,
    start_event_id="event_login",
    filters={"user_id": "user_123"},
    max_depth=5,
    direction="forward",
    node_label=memory_graph.node_label
)

print(f"Found {len(effects)} causal chains:")
for i, chain in enumerate(effects, 1):
    print(f"\nChain {i} (depth: {chain['depth']}):")
    for event in chain['events']:
        print(f"  - {event['name']}")
```

#### Backward Chain (Causes)

Find what **led to** an event:

```python
# Get all causes of a system crash
causes = get_causal_chain(
    graph=memory_graph.graph,
    start_event_id="event_crash",
    filters={"user_id": "user_123"},
    max_depth=5,
    direction="backward",
    node_label=memory_graph.node_label
)

print(f"Found {len(causes)} causal chains leading to crash:")
for chain in causes:
    print(f"\nCausal chain (depth: {chain['depth']}):")
    for event in chain['events']:
        print(f"  → {event['name']}")
```

---

### Feature 4: Find Root Causes

Identify the **origin events** that started a chain:

```python
from outhad_contextkit.memory.temporal import find_root_causes

# Find what ultimately caused an error
root_causes = find_root_causes(
    graph=memory_graph.graph,
    effect_event_id="event_error",
    filters={"user_id": "user_123"},
    min_confidence=0.7,
    node_label=memory_graph.node_label
)

print("Root causes found:")
for cause in root_causes:
    print(f"  - {cause['root']['name']}")
    print(f"    Distance: {cause['distance']} steps away")
```

**Use Cases:**
- Debugging: "What originally caused this error?"
- Analysis: "What was the root cause of the outage?"
- Attribution: "Which event started this chain reaction?"

---

### Feature 5: Build Causal Subgraph

Connect multiple events and show their relationships:

```python
from outhad_contextkit.memory.temporal import get_causal_subgraph

# Get causal connections between specific events
event_ids = ["event_login", "event_auth", "event_dashboard", "event_error"]

subgraph = get_causal_subgraph(
    graph=memory_graph.graph,
    event_ids=event_ids,
    filters={"user_id": "user_123"},
    max_depth=3,
    node_label=memory_graph.node_label
)

print(f"Subgraph has {len(subgraph['nodes'])} nodes")
print(f"Subgraph has {len(subgraph['relationships'])} causal links")
```

**Use Cases:**
- Visualization: Show how events are causally connected
- Analysis: Understand complex event relationships
- Debugging: Trace paths between events

---

## 🎯 Common Use Cases

### Use Case 1: User Journey Analysis

Understand the causal flow of user actions:

```python
# Track user journey
journey_events = [
    {"id": "visit", "content": "User visited homepage", "timestamp": "..."},
    {"id": "search", "content": "User searched for product", "timestamp": "..."},
    {"id": "view", "content": "User viewed product page", "timestamp": "..."},
    {"id": "cart", "content": "User added to cart", "timestamp": "..."},
    {"id": "checkout", "content": "User completed checkout", "timestamp": "..."}
]

# Extract causal relationships
links = extractor.extract_causal_links(journey_events, use_llm=True)

# Store in graph
for link in links:
    memory_graph.add_causal_link(link, filters={"user_id": "user_456"})

# Analyze: What led to checkout?
causes = get_causal_chain(
    graph=memory_graph.graph,
    start_event_id="checkout",
    filters={"user_id": "user_456"},
    direction="backward",
    node_label=memory_graph.node_label
)
```

### Use Case 2: Error Root Cause Analysis

Debug system issues by finding root causes:

```python
# System error scenario
error_events = [
    {"id": "high_load", "content": "High server load detected", "timestamp": "..."},
    {"id": "slow_db", "content": "Database queries slowed down", "timestamp": "..."},
    {"id": "timeout", "content": "Request timeout occurred", "timestamp": "..."},
    {"id": "error", "content": "500 error returned to user", "timestamp": "..."}
]

# Extract and store
links = extractor.extract_causal_links(error_events, use_llm=True)
for link in links:
    memory_graph.add_causal_link(link, filters={"user_id": "system"})

# Find root cause
roots = find_root_causes(
    graph=memory_graph.graph,
    effect_event_id="error",
    filters={"user_id": "system"},
    min_confidence=0.7,
    node_label=memory_graph.node_label
)

print("Root cause of error:")
for root in roots:
    print(f"  {root['root']['name']} ({root['distance']} steps away)")
```

### Use Case 3: Feature Adoption Flow

Understand what enables users to adopt features:

```python
# Feature adoption scenario
adoption_events = [
    {"id": "signup", "content": "User signed up", "timestamp": "..."},
    {"id": "onboard", "content": "User completed onboarding", "timestamp": "..."},
    {"id": "tutorial", "content": "User watched tutorial", "timestamp": "..."},
    {"id": "feature", "content": "User used advanced feature", "timestamp": "..."}
]

# Extract enablement relationships
links = extractor.extract_causal_links(adoption_events, use_llm=True)

# Find "enables" relationships
enablers = [link for link in links if link.causal_type == CausalType.ENABLES.value]
print(f"Found {len(enablers)} enabling factors:")
for link in enablers:
    print(f"  {link.cause_id} enables {link.effect_id}")
```

### Use Case 4: Prevent Negative Outcomes

Identify what prevents bad outcomes:

```python
# Security scenario
security_events = [
    {"id": "attempt", "content": "Unauthorized access attempted", "timestamp": "..."},
    {"id": "auth", "content": "2FA authentication enabled", "timestamp": "..."},
    {"id": "blocked", "content": "Access blocked successfully", "timestamp": "..."}
]

links = extractor.extract_causal_links(security_events, use_llm=True)

# Find prevention relationships
preventions = [link for link in links if link.causal_type == CausalType.PREVENTS.value]
print(f"Security measures that prevented attacks:")
for link in preventions:
    print(f"  {link.cause_id} prevented {link.effect_id}")
    print(f"  Confidence: {link.confidence:.2f}")
```

---

## 🎓 Advanced Features

### Custom Causal Links

Create causal links manually:

```python
from datetime import datetime

# Create custom causal link
custom_link = CausalLink(
    cause_id="event_payment",
    effect_id="event_confirmation",
    causal_type=CausalType.LEADS_TO.value,
    confidence=1.0,  # Certain relationship
    evidence="Payment directly triggers confirmation email",
    timestamp=datetime.utcnow()
)

# Store it
memory_graph.add_causal_link(custom_link, filters={"user_id": "user_789"})
```

### Filter by Confidence

Query only high-confidence relationships:

```python
# Find root causes with high confidence only
high_conf_roots = find_root_causes(
    graph=memory_graph.graph,
    effect_event_id="event_success",
    filters={"user_id": "user_123"},
    min_confidence=0.9,  # Only 90%+ confidence
    node_label=memory_graph.node_label
)
```

### Multi-Event Analysis

Analyze relationships between multiple events:

```python
# Get subgraph of critical events
critical_events = [
    "event_auth_fail",
    "event_retry",
    "event_auth_success",
    "event_access_granted"
]

subgraph = get_causal_subgraph(
    graph=memory_graph.graph,
    event_ids=critical_events,
    filters={"user_id": "user_123"},
    max_depth=2,
    node_label=memory_graph.node_label
)

# Analyze the subgraph
print(f"Nodes: {len(subgraph['nodes'])}")
print(f"Causal relationships: {len(subgraph['relationships'])}")
```

---

## 🔍 Best Practices

### 1. Use LLM for Complex Scenarios

```python
# LLM understands context better
complex_events = [
    {"id": "e1", "content": "User expressed frustration in support chat"},
    {"id": "e2", "content": "System had been slow for 3 days"},
    {"id": "e3", "content": "User canceled subscription"}
]

# LLM can infer: slow system → frustration → cancellation
links = extractor.extract_causal_links(complex_events, use_llm=True)
```

### 2. Use Rules for Simple Patterns

```python
# Rule-based works when events have causal keywords
simple_events = [
    {"id": "e1", "content": "Button clicked"},
    {"id": "e2", "content": "Click caused modal to open"}
]

# Rules will quickly find: e1 → e2 (caused_by)
links = extractor.extract_causal_links(simple_events, use_llm=False)
```

### 3. Combine with Temporal Attributes

```python
# Events with timestamps from 
from outhad_contextkit.memory.temporal import TemporalEvent

temporal_events = [
    TemporalEvent(
        id="e1",
        content="User logged in",
        timestamp=datetime(2025, 1, 14, 10, 0, 0),
        confidence=1.0,
        modality="text"
    ).dict()
]

# Extract causal relationships
links = extractor.extract_causal_links(temporal_events, use_llm=True)
```

### 4. Validate Confidence Scores

```python
# Check confidence before storing
for link in causal_links:
    if link.confidence >= 0.8:
        memory_graph.add_causal_link(link, filters={"user_id": "user_123"})
        print(f"✅ Stored high-confidence link: {link.cause_id} → {link.effect_id}")
    else:
        print(f"⚠️  Skipped low-confidence link: {link.cause_id} → {link.effect_id}")
```

---

## 📊 Performance Tips

### Batch Processing

```python
# Process multiple event sets efficiently
all_links = []

for event_set in event_sets:
    links = extractor.extract_causal_links(event_set, use_llm=True)
    all_links.extend(links)

# Store in bulk
for link in all_links:
    memory_graph.add_causal_link(link, filters={"user_id": "user_123"})

print(f"Processed {len(all_links)} causal links")
```

### Limit Chain Depth

```python
# Limit depth for performance
chains = get_causal_chain(
    graph=memory_graph.graph,
    start_event_id="event_start",
    filters={"user_id": "user_123"},
    max_depth=3,  # Don't go deeper than 3 steps
    direction="forward",
    node_label=memory_graph.node_label
)
```

---

## 🐛 Troubleshooting

### Issue: LLM Returns Empty Results

```python
# Check if LLM is initialized
if not extractor.llm:
    print("❌ LLM not initialized, using rule-based fallback")
    links = extractor.extract_causal_links(events, use_llm=False)
```

### Issue: No Causal Links Found

```python
# Verify events have causal content
events = [
    {"id": "e1", "content": "User logged in"},  # Generic
    {"id": "e2", "content": "Dashboard loaded"}  # Generic
]

# Add more context
events_with_context = [
    {"id": "e1", "content": "User successfully logged in with credentials"},
    {"id": "e2", "content": "Login enabled dashboard to load"}
]

links = extractor.extract_causal_links(events_with_context, use_llm=True)
```

### Issue: Low Confidence Scores

```python
# Add more evidence in event content
events_with_evidence = [
    {
        "id": "e1",
        "content": "Server CPU reached 100% utilization",
        "timestamp": "2025-01-14T10:00:00"
    },
    {
        "id": "e2",
        "content": "Response times increased to 5 seconds due to high CPU",
        "timestamp": "2025-01-14T10:00:30"
    }
]

links = extractor.extract_causal_links(events_with_evidence, use_llm=True)
# Should get higher confidence with explicit causality
```

---

## 📚 API Reference

### CausalExtractor

```python
class CausalExtractor:
    def __init__(self, llm=None):
        """Initialize with optional LLM."""
        
    def extract_causal_links(
        self,
        events: List[Dict],
        use_llm: bool = True
    ) -> List[CausalLink]:
        """Extract causal links from events."""
```

### Functions

```python
get_causal_chain(
    graph,
    start_event_id: str,
    filters: dict,
    max_depth: int = 5,
    direction: str = "forward",
    node_label: str = ""
) -> List[Dict]

find_root_causes(
    graph,
    effect_event_id: str,
    filters: dict,
    min_confidence: float = 0.7,
    node_label: str = ""
) -> List[Dict]

get_causal_subgraph(
    graph,
    event_ids: List[str],
    filters: dict,
    max_depth: int = 3,
    node_label: str = ""
) -> Dict
```

---
