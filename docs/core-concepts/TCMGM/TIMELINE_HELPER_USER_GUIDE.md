#  TCMGM: Timeline Helper - Usage Guide

## 🎯 Overview

 adds **comprehensive timeline functionality** to TCMGM. This enables the system to extract temporal events from conversations, track event sequences over time, perform time-based queries, and generate natural language timeline summaries.

**Key Capabilities:**
- Extract events from conversation transcripts automatically
- Store events with temporal metadata in Neo4j
- Query timelines with flexible time windows
- Answer natural language questions like "What happened after X?"
- Generate timeline summaries and statistics
- Filter events by modality (text, image, audio)

---

## 🚀 Quick Start

### 1. Import Components

```python
from outhad_contextkit.memory.timeline import TimelineBuilder
from outhad_contextkit.memory.temporal.timeline_queries import TimelineQueries
from outhad_contextkit.memory.temporal.types import TimeWindow
from outhad_contextkit.utils.factory import LlmFactory
from outhad_contextkit.llms.configs import LlmConfig
from outhad_contextkit.memory.graph_memory import MemoryGraph
from datetime import datetime, timedelta
```

### 2. Initialize Timeline Builder

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

# Initialize graph storage
from outhad_contextkit.configs.base import MemoryConfig
config = MemoryConfig()
memory_graph = MemoryGraph(config)

# Create timeline builder
timeline_builder = TimelineBuilder(
    llm=llm,
    graph=memory_graph.graph
)
```

### 3. Build Your First Timeline

```python
# Conversation transcript
transcript = """
User: I went to the grocery store this morning.
Assistant: What did you buy?
User: I bought milk, bread, and eggs. Then I went home and made breakfast.
Assistant: That sounds nice!
User: Yes, after breakfast I started working on my project.
"""

# Build timeline from transcript
result = timeline_builder.build_from_transcript(
    transcript=transcript,
    user_id="user_123",
    session_start=datetime.utcnow()
)

print(f"✅ Extracted {len(result['events'])} events")
print(f"✅ Found {len(result['causal_links'])} causal relationships")
```

---

## 📖 Core Features

### Feature 1: Extract Events from Transcripts

The Timeline Builder automatically extracts discrete events from conversational text using LLM-based fact extraction.

#### Basic Transcript Processing

```python
# Define your conversation
transcript = """
User: I logged into the system at 9am.
Assistant: Welcome back!
User: I checked my emails and found 15 unread messages.
Assistant: Would you like to read them?
User: Yes, I read the important ones and archived the rest.
"""

# Extract events with automatic temporal attribution
result = timeline_builder.build_from_transcript(
    transcript=transcript,
    user_id="user_456",
    session_start=datetime(2025, 1, 14, 9, 0, 0)
)

# Print extracted events
for event in result['events']:
    print(f"Event: {event['content']}")
    print(f"Time: {event['timestamp']}")
    print(f"Confidence: {event['confidence']:.2f}")
    print()
```

**Output Example:**
```
Event: User logged into the system
Time: 2025-01-14 09:00:00
Confidence: 0.90

Event: User checked emails
Time: 2025-01-14 09:02:00
Confidence: 0.85

Event: User found 15 unread messages
Time: 2025-01-14 09:04:00
Confidence: 0.90

Event: User read important emails
Time: 2025-01-14 09:06:00
Confidence: 0.85

Event: User archived remaining messages
Time: 2025-01-14 09:08:00
Confidence: 0.85
```

#### Multimodal Event Extraction

```python
# Transcript with different modalities
multimodal_transcript = """
User: [Uploads profile photo]
Assistant: Great photo! Uploaded successfully.
User: Thanks! Now let me record a voice message.
User: [Records audio message]
Assistant: Audio message saved.
User: Perfect, now I'll type my bio.
"""

result = timeline_builder.build_from_transcript(
    transcript=multimodal_transcript,
    user_id="user_789",
    session_start=datetime.utcnow()
)

# Events are tagged with modality
for event in result['events']:
    print(f"{event['modality']}: {event['content']}")
```

**Output:**
```
image: User uploaded profile photo
audio: User recorded voice message
text: User typed bio
```

---

### Feature 2: Retrieve Timeline Events

#### Get Complete Timeline

```python
# Retrieve all events for a user
events = timeline_builder.get_timeline(
    user_id="user_123",
    time_window=None  # No filter, get all events
)

print(f"Found {len(events)} total events")
for event in events:
    print(f"{event['timestamp']}: {event['name']}")
```

#### Get Timeline with Time Window

```python
from outhad_contextkit.memory.temporal.types import TimeWindow

# Get events from last 7 days
now = datetime.utcnow()
last_week = TimeWindow(
    start=now - timedelta(days=7),
    end=now
)

recent_events = timeline_builder.get_timeline(
    user_id="user_123",
    time_window=last_week
)

print(f"Events in last 7 days: {len(recent_events)}")
```

#### Get Events in Date Range

```python
# Get events between specific dates
start_date = datetime(2025, 1, 1, 0, 0, 0)
end_date = datetime(2025, 1, 31, 23, 59, 59)

january_events = timeline_builder.get_events_between(
    user_id="user_123",
    start_time=start_date,
    end_time=end_date
)

print(f"Events in January 2025: {len(january_events)}")
```

---

### Feature 3: Natural Language Timeline Queries

The `TimelineQueries` helper provides intuitive methods for querying timelines.

#### Initialize Query Helper

```python
from outhad_contextkit.memory.temporal.timeline_queries import TimelineQueries

# Create query helper
queries = TimelineQueries(timeline_builder)
```

#### "What Happened After?" Queries

```python
# Find what happened after a specific event
events_after = queries.what_happened_after(
    user_id="user_123",
    reference_event="user logged in"
)

print("Events after login:")
for event in events_after:
    print(f"  → {event['name']}")
```

#### "What Happened Before?" Queries

```python
# Find what led up to an event
events_before = queries.what_happened_before(
    user_id="user_123",
    reference_event="user completed checkout"
)

print("Events before checkout:")
for event in events_before:
    print(f"  → {event['name']}")
```

#### Recent Events

```python
# Get most recent events
recent = queries.get_recent_events(
    user_id="user_123",
    limit=10
)

print("Last 10 events:")
for event in recent:
    print(f"{event['timestamp']}: {event['name']}")
```

#### Events on Specific Date

```python
# Get all events on a specific date
today = datetime.utcnow()
todays_events = queries.events_on_date(
    user_id="user_123",
    date=today
)

print(f"Events today: {len(todays_events)}")
```

#### Events This Week

```python
# Get all events from current week (Monday to Sunday)
this_week = queries.events_this_week(user_id="user_123")

print(f"Events this week: {len(this_week)}")
```

#### Events in Last N Days

```python
# Get events from last N days
last_30_days = queries.events_last_n_days(
    user_id="user_123",
    n_days=30
)

print(f"Events in last 30 days: {len(last_30_days)}")
```

#### Find Event by Description

```python
# Search for specific event
found_events = queries.find_event_by_description(
    user_id="user_123",
    description="login"
)

print("Found login events:")
for event in found_events:
    print(f"  {event['timestamp']}: {event['name']}")
```

#### Filter by Modality

```python
# Get only image events
image_events = queries.find_events_by_modality(
    user_id="user_123",
    modality="image"
)

print(f"Found {len(image_events)} image events")

# Get only audio events
audio_events = queries.find_events_by_modality(
    user_id="user_123",
    modality="audio"
)

print(f"Found {len(audio_events)} audio events")
```

---

### Feature 4: Timeline Summarization

Generate natural language summaries of timeline periods using LLM.

#### Summarize Entire Timeline

```python
# Get natural language summary
summary = timeline_builder.summarize_timeline(
    user_id="user_123",
    time_window=None  # All time
)

print(summary)
```

**Example Output:**
```
User had 45 events over 30 days, including login activities,
profile updates, 15 messages sent, 3 images uploaded, and
checkout completion.
```

#### Summarize Specific Period

```python
# Summarize last week
last_week_window = TimeWindow(
    start=datetime.utcnow() - timedelta(days=7),
    end=datetime.utcnow()
)

weekly_summary = timeline_builder.summarize_timeline(
    user_id="user_123",
    time_window=last_week_window
)

print(weekly_summary)
```

#### Summarize with Query Helper

```python
# Get summary for specific time period
summary = queries.get_timeline_summary_for_period(
    user_id="user_123",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 1, 31)
)

print(f"January Summary:\n{summary}")
```

---

### Feature 5: Timeline Statistics

Get aggregate metrics about timeline composition.

#### Get Full Statistics

```python
# Get comprehensive statistics
stats = timeline_builder.get_timeline_stats(user_id="user_123")

print(f"Total Events: {stats['total_events']}")
print(f"Time Span: {stats['time_span']}")
print(f"Average Confidence: {stats['avg_confidence']:.2f}")
print(f"\nEvents by Modality:")
for modality, count in stats['modalities'].items():
    print(f"  {modality}: {count}")
```

**Example Output:**
```
Total Events: 125
Time Span: 30 days
Average Confidence: 0.87

Events by Modality:
  text: 100
  image: 15
  audio: 10
```

#### Monitor Timeline Growth

```python
# Track timeline metrics over time
import time

for day in range(7):
    stats = timeline_builder.get_timeline_stats(user_id="user_123")
    print(f"Day {day + 1}: {stats['total_events']} events")
    time.sleep(86400)  # Wait 1 day
```

---

### Feature 6: Advanced Time Windows

TimeWindow provides flexible temporal filtering.

#### Unbounded Windows

```python
from outhad_contextkit.memory.temporal.types import TimeWindow

# Get all past events (no start time)
all_past = TimeWindow(end=datetime.utcnow())

# Get all future events (no end time)
all_future = TimeWindow(start=datetime.utcnow())

# Get all events ever (no bounds)
everything = TimeWindow()
```

#### Custom Time Ranges

```python
# Business hours only (9am to 5pm)
business_start = datetime(2025, 1, 14, 9, 0, 0)
business_end = datetime(2025, 1, 14, 17, 0, 0)

business_hours = TimeWindow(
    start=business_start,
    end=business_end
)

events = timeline_builder.get_timeline(
    user_id="user_123",
    time_window=business_hours
)
```

#### Check Time Window Containment

```python
# Check if a timestamp falls within a window
window = TimeWindow(
    start=datetime(2025, 1, 1),
    end=datetime(2025, 1, 31)
)

test_time = datetime(2025, 1, 15)
if window.contains(test_time):
    print("✅ Time is within window")
else:
    print("❌ Time is outside window")
```

---

## 🎯 Common Use Cases

### Use Case 1: User Activity Tracking

Track and analyze user activity patterns over time.

```python
# Build timeline from user sessions
session_transcript = """
User: Logged in to dashboard
User: Viewed analytics report
User: Exported data to CSV
User: Updated profile settings
User: Logged out
"""

result = timeline_builder.build_from_transcript(
    transcript=session_transcript,
    user_id="user_activity_123",
    session_start=datetime.utcnow()
)

# Analyze activity patterns
stats = timeline_builder.get_timeline_stats("user_activity_123")
print(f"Total activities: {stats['total_events']}")

# Get recent activity
queries = TimelineQueries(timeline_builder)
recent = queries.get_recent_events("user_activity_123", limit=5)

print("\nRecent activities:")
for event in recent:
    print(f"  {event['timestamp'].strftime('%H:%M')}: {event['name']}")
```

### Use Case 2: Customer Journey Mapping

Map the complete customer journey from acquisition to conversion.

```python
# Customer journey transcript
journey = """
User: Visited homepage from Google search
User: Browsed product catalog
User: Added item to wishlist
User: Returned next day and viewed wishlist
User: Added item to cart
User: Applied discount code
User: Completed checkout
User: Received order confirmation email
"""

result = timeline_builder.build_from_transcript(
    transcript=journey,
    user_id="customer_journey_456",
    session_start=datetime(2025, 1, 10, 10, 0, 0)
)

# Analyze journey
queries = TimelineQueries(timeline_builder)

# What happened after adding to cart?
after_cart = queries.what_happened_after(
    user_id="customer_journey_456",
    reference_event="added item to cart"
)

print("Conversion funnel after cart:")
for event in after_cart:
    print(f"  → {event['name']}")

# Get journey summary
summary = timeline_builder.summarize_timeline("customer_journey_456")
print(f"\nJourney Summary:\n{summary}")
```

### Use Case 3: Project Progress Tracking

Track project milestones and tasks over time.

```python
# Project timeline
project_transcript = """
Developer: Created new project repository
Developer: Set up CI/CD pipeline
Developer: Implemented authentication module
Developer: Wrote unit tests for auth
Developer: Fixed bug in login flow
Developer: Deployed to staging environment
Developer: Conducted QA testing
Developer: Deployed to production
"""

result = timeline_builder.build_from_transcript(
    transcript=project_transcript,
    user_id="project_timeline_789",
    session_start=datetime(2025, 1, 1, 0, 0, 0)
)

# Get project statistics
stats = timeline_builder.get_timeline_stats("project_timeline_789")
print(f"Project milestones: {stats['total_events']}")
print(f"Project duration: {stats['time_span']}")

# Get events by week
queries = TimelineQueries(timeline_builder)
this_week = queries.events_this_week("project_timeline_789")
print(f"\nThis week's progress: {len(this_week)} tasks")
```

### Use Case 4: Support Ticket Timeline

Track support ticket lifecycle and resolution steps.

```python
# Support ticket timeline
ticket_transcript = """
User: Reported login issue
Support: Acknowledged ticket, assigned to engineer
Engineer: Investigated error logs
Engineer: Identified database connection issue
Engineer: Applied database patch
Engineer: Tested login functionality
Support: Marked ticket as resolved
User: Confirmed issue is fixed
"""

result = timeline_builder.build_from_transcript(
    transcript=ticket_transcript,
    user_id="ticket_456",
    session_start=datetime.utcnow()
)

# Analyze ticket resolution time
queries = TimelineQueries(timeline_builder)
events = queries.events_on_date("ticket_456", datetime.utcnow())

if len(events) >= 2:
    first_event = events[0]
    last_event = events[-1]
    resolution_time = last_event['timestamp'] - first_event['timestamp']
    print(f"Resolution time: {resolution_time}")
```

### Use Case 5: Learning Progress Tracking

Track student learning activities and progress.

```python
# Learning timeline
learning_transcript = """
Student: Started Module 1: Python Basics
Student: Completed video lecture on variables
Student: Took quiz on data types, scored 85%
Student: Started Module 2: Functions
Student: Completed coding exercise on functions
Student: Submitted final project
"""

result = timeline_builder.build_from_transcript(
    transcript=learning_transcript,
    user_id="student_123",
    session_start=datetime(2025, 1, 1, 0, 0, 0)
)

# Track learning progress
queries = TimelineQueries(timeline_builder)

# Get events from last 30 days
progress = queries.events_last_n_days("student_123", n_days=30)
print(f"Learning activities in last month: {len(progress)}")

# Get summary of learning journey
summary = timeline_builder.summarize_timeline("student_123")
print(f"\nLearning Summary:\n{summary}")
```

### Use Case 6: Multimodal Content Timeline

Track mixed content creation (text, images, audio).

```python
# Multimodal content timeline
content_transcript = """
Creator: Wrote blog post draft
Creator: [Uploaded 3 images for blog]
Creator: Recorded audio introduction
Creator: [Uploaded cover image]
Creator: Published blog post
Creator: Shared on social media
"""

result = timeline_builder.build_from_transcript(
    transcript=content_transcript,
    user_id="creator_789",
    session_start=datetime.utcnow()
)

# Analyze content by modality
queries = TimelineQueries(timeline_builder)

text_events = queries.find_events_by_modality("creator_789", "text")
image_events = queries.find_events_by_modality("creator_789", "image")
audio_events = queries.find_events_by_modality("creator_789", "audio")

print(f"Content created:")
print(f"  Text: {len(text_events)}")
print(f"  Images: {len(image_events)}")
print(f"  Audio: {len(audio_events)}")
```

---

## 🎓 Advanced Features

### Custom Event Timestamps

Override automatic timestamping with custom times.

```python
# Manually specify event times
from outhad_contextkit.memory.temporal import TemporalEvent

events = [
    TemporalEvent(
        id="event_1",
        content="User registered account",
        timestamp=datetime(2025, 1, 1, 10, 0, 0),
        confidence=1.0,
        modality="text"
    ),
    TemporalEvent(
        id="event_2",
        content="User completed onboarding",
        timestamp=datetime(2025, 1, 1, 10, 15, 0),
        confidence=1.0,
        modality="text"
    )
]

# Store events directly
for event in events:
    timeline_builder.graph.add_temporal_event(
        event=event.dict(),
        filters={"user_id": "custom_user_123"}
    )
```

### Combine Timeline with Causal Analysis

Integrate timeline queries with causal relationship extraction.

```python
from outhad_contextkit.memory.temporal import CausalExtractor

# Build timeline
result = timeline_builder.build_from_transcript(
    transcript=transcript,
    user_id="user_123",
    session_start=datetime.utcnow()
)

# Causal links are automatically extracted!
print(f"Found {len(result['causal_links'])} causal relationships")

for link in result['causal_links']:
    print(f"{link['cause_id']} → {link['effect_id']}")
    print(f"  Type: {link['causal_type']}")
    print(f"  Confidence: {link['confidence']:.2f}")
```

### Search Timeline with Text Query

Search for events containing specific keywords.

```python
# Text-based search
search_results = queries.search_timeline(
    user_id="user_123",
    query="login"
)

print(f"Found {len(search_results)} events matching 'login'")
for event in search_results:
    print(f"  {event['name']}")
```

### Filter Events by Confidence Score

Query only high-confidence events.

```python
# Get all events
all_events = timeline_builder.get_timeline(user_id="user_123")

# Filter by confidence
high_confidence_events = [
    event for event in all_events
    if event.get('confidence', 0) >= 0.9
]

print(f"High-confidence events: {len(high_confidence_events)}")
```

### Timeline Comparison Across Users

Compare timeline patterns between different users.

```python
# Get statistics for multiple users
user_ids = ["user_123", "user_456", "user_789"]

for user_id in user_ids:
    stats = timeline_builder.get_timeline_stats(user_id)
    print(f"\n{user_id}:")
    print(f"  Events: {stats['total_events']}")
    print(f"  Time span: {stats['time_span']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
```

---

## 🔍 Best Practices

### 1. Provide Session Start Times

Always provide accurate session start times for better temporal attribution.

```python
# ✅ GOOD: Provide actual session start time
result = timeline_builder.build_from_transcript(
    transcript=transcript,
    user_id="user_123",
    session_start=datetime(2025, 1, 14, 10, 30, 0)  # Actual time
)

# ❌ BAD: Using current time when transcript is historical
result = timeline_builder.build_from_transcript(
    transcript=historical_transcript,
    user_id="user_123",
    session_start=datetime.utcnow()  # Wrong!
)
```

### 2. Use Time Windows for Large Timelines

Limit queries to relevant time periods for better performance.

```python
# ✅ GOOD: Query specific time range
last_month = TimeWindow(
    start=datetime.utcnow() - timedelta(days=30),
    end=datetime.utcnow()
)
events = timeline_builder.get_timeline("user_123", time_window=last_month)

# ❌ AVOID: Querying entire timeline when unnecessary
all_events = timeline_builder.get_timeline("user_123")  # May be slow
```

### 3. Combine with Other TCMGM 

Leverage timeline with causal relationships and multimodal support.

```python
# Build timeline (automatically extracts causal relationships)
result = timeline_builder.build_from_transcript(
    transcript=transcript,
    user_id="user_123",
    session_start=datetime.utcnow()
)

# Timeline events have temporal attributes 
# Causal links are extracted 
# Multimodal events are supported 

# Now query with full context
queries = TimelineQueries(timeline_builder)
recent = queries.get_recent_events("user_123", limit=10)
```

### 4. Handle Empty Timelines Gracefully

Check for empty results before processing.

```python
events = timeline_builder.get_timeline(user_id="new_user")

if not events:
    print("No timeline events found for this user")
else:
    print(f"Found {len(events)} events")
    for event in events:
        print(f"  {event['name']}")
```

### 5. Use Natural Language Queries for Readability

Prefer TimelineQueries methods for clearer code.

```python
# ✅ GOOD: Clear intent
queries = TimelineQueries(timeline_builder)
recent = queries.events_last_n_days("user_123", 7)

# ❌ LESS CLEAR: Manual time calculation
window = TimeWindow(
    start=datetime.utcnow() - timedelta(days=7),
    end=datetime.utcnow()
)
recent = timeline_builder.get_timeline("user_123", time_window=window)
```

### 6. Monitor Timeline Growth

Track timeline metrics to detect unusual patterns.

```python
# Get baseline statistics
stats = timeline_builder.get_timeline_stats("user_123")
baseline_count = stats['total_events']

# After some time...
new_stats = timeline_builder.get_timeline_stats("user_123")
new_count = new_stats['total_events']

growth = new_count - baseline_count
print(f"Timeline grew by {growth} events")

if growth > 100:
    print("⚠️ Unusual activity detected")
```

---

## 📊 Performance Tips

### Batch Transcript Processing

Process multiple transcripts efficiently.

```python
# Process multiple sessions in batch
transcripts = [
    (transcript1, "user_123", datetime(2025, 1, 1, 10, 0, 0)),
    (transcript2, "user_456", datetime(2025, 1, 2, 11, 0, 0)),
    (transcript3, "user_789", datetime(2025, 1, 3, 12, 0, 0))
]

total_events = 0
for transcript, user_id, start_time in transcripts:
    result = timeline_builder.build_from_transcript(
        transcript=transcript,
        user_id=user_id,
        session_start=start_time
    )
    total_events += len(result['events'])
    print(f"✅ Processed {user_id}: {len(result['events'])} events")

print(f"\nTotal events extracted: {total_events}")
```

### Limit Query Results

Use limits to retrieve only what you need.

```python
# Get only most recent 10 events
recent = queries.get_recent_events("user_123", limit=10)

# Much faster than retrieving all and then slicing
# all_events = timeline_builder.get_timeline("user_123")
# recent = all_events[-10:]  # Slower
```

### Cache Timeline Statistics

Cache statistics for frequently accessed timelines.

```python
from functools import lru_cache
from datetime import datetime, timedelta

# Cache stats for 5 minutes
@lru_cache(maxsize=100)
def get_cached_stats(user_id, cache_key):
    return timeline_builder.get_timeline_stats(user_id)

# Generate cache key that changes every 5 minutes
cache_key = datetime.utcnow().replace(second=0, microsecond=0)
cache_key = cache_key - timedelta(minutes=cache_key.minute % 5)

stats = get_cached_stats("user_123", cache_key.isoformat())
```

### Use Specific Time Windows

Narrow time windows improve query performance.

```python
# ✅ FAST: Specific 1-week window
last_week = TimeWindow(
    start=datetime.utcnow() - timedelta(days=7),
    end=datetime.utcnow()
)
events = timeline_builder.get_timeline("user_123", time_window=last_week)

# ❌ SLOWER: Query entire timeline
all_events = timeline_builder.get_timeline("user_123")
```

---

## 🐛 Troubleshooting

### Issue: No Events Extracted from Transcript

**Problem:** `build_from_transcript()` returns empty events list.

**Solutions:**

```python
# ✅ Check transcript content
transcript = """
User: I went to the store.
User: I bought groceries.
"""

result = timeline_builder.build_from_transcript(
    transcript=transcript,
    user_id="user_123",
    session_start=datetime.utcnow()
)

if not result['events']:
    print("❌ No events extracted")
    print("Check:")
    print("1. Is LLM initialized correctly?")
    print("2. Does transcript contain extractable events?")
    print("3. Is transcript in correct format?")
```

**Fix:**
```python
# Ensure clear, discrete events in transcript
better_transcript = """
User: I logged into the system.
User: I viewed my dashboard.
User: I created a new project.
User: I invited team members.
"""

result = timeline_builder.build_from_transcript(
    transcript=better_transcript,
    user_id="user_123",
    session_start=datetime.utcnow()
)
```

### Issue: Timeline Query Returns Empty Results

**Problem:** Queries like `events_last_n_days()` return no results.

**Solutions:**

```python
# Check if timeline exists
all_events = timeline_builder.get_timeline("user_123")

if not all_events:
    print("❌ No timeline exists for this user")
    print("Build timeline first:")

    # Build timeline
    result = timeline_builder.build_from_transcript(
        transcript=transcript,
        user_id="user_123",
        session_start=datetime.utcnow()
    )
    print(f"✅ Created timeline with {len(result['events'])} events")
```

### Issue: Events Have Low Confidence Scores

**Problem:** Extracted events have confidence scores below 0.7.

**Solutions:**

```python
# Add more context to transcript
low_confidence_transcript = """
User: Did something.
User: Then did another thing.
"""

# ✅ Better: Provide specific, clear events
high_confidence_transcript = """
User: Logged into account using email and password.
User: Viewed account settings page.
User: Updated profile information with new phone number.
User: Saved changes successfully.
"""

result = timeline_builder.build_from_transcript(
    transcript=high_confidence_transcript,
    user_id="user_123",
    session_start=datetime.utcnow()
)

# Check confidence scores
for event in result['events']:
    if event['confidence'] < 0.7:
        print(f"⚠️ Low confidence: {event['content']} ({event['confidence']:.2f})")
```

### Issue: Timestamp Estimation Inaccurate

**Problem:** Auto-generated timestamps don't match real event times.

**Solution:**

```python
# Current: 2-minute intervals (approximate)
# If you need precise timestamps, extract them from transcript

# Option 1: Include timestamps in transcript
timestamped_transcript = """
[10:00] User: Logged in
[10:05] User: Checked emails
[10:15] User: Sent message
[10:30] User: Logged out
"""

# Option 2: Use custom TemporalEvent with exact timestamps
from outhad_contextkit.memory.temporal import TemporalEvent

precise_events = [
    TemporalEvent(
        id="evt_1",
        content="User logged in",
        timestamp=datetime(2025, 1, 14, 10, 0, 0),  # Exact time
        confidence=1.0,
        modality="text"
    )
]
```

### Issue: Neo4j Connection Errors

**Problem:** Timeline queries fail with Neo4j connection errors.

**Solutions:**

```python
# Check Neo4j connection
try:
    # Test query
    events = timeline_builder.get_timeline("user_123")
    print("✅ Neo4j connection working")
except Exception as e:
    print(f"❌ Neo4j error: {e}")
    print("\nTroubleshooting steps:")
    print("1. Ensure Neo4j is running")
    print("2. Check connection settings in MemoryConfig")
    print("3. Verify authentication credentials")
    print("4. Test connection manually:")
    print("   from neo4j import GraphDatabase")
    print("   driver = GraphDatabase.driver(uri, auth=(user, password))")
```

### Issue: Slow Timeline Queries

**Problem:** Queries take too long to execute.

**Solutions:**

```python
# ❌ SLOW: Getting entire timeline
all_events = timeline_builder.get_timeline("user_123")

# ✅ FAST: Use time windows
recent_window = TimeWindow(
    start=datetime.utcnow() - timedelta(days=7),
    end=datetime.utcnow()
)
recent_events = timeline_builder.get_timeline(
    user_id="user_123",
    time_window=recent_window
)

# ✅ FAST: Use limits
recent = queries.get_recent_events("user_123", limit=20)

# ✅ FAST: Query specific dates
today = queries.events_on_date("user_123", datetime.utcnow())
```

---

## 📚 API Reference

### TimelineBuilder

```python
class TimelineBuilder:
    def __init__(self, llm, graph):
        """Initialize timeline builder with LLM and graph storage."""

    def build_from_transcript(
        self,
        transcript: str,
        user_id: str,
        session_start: datetime,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> Dict:
        """
        Extract events from transcript and store in graph.

        Returns:
            {
                'events': List[Dict],  # Extracted temporal events
                'causal_links': List[Dict]  # Causal relationships
            }
        """

    def get_timeline(
        self,
        user_id: str,
        time_window: Optional[TimeWindow] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve timeline events for a user.

        Args:
            user_id: User identifier
            time_window: Optional time range filter
            agent_id: Optional agent filter
            run_id: Optional run filter

        Returns:
            List of event dictionaries
        """

    def get_events_between(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """Get events between two timestamps."""

    def get_events_after(
        self,
        user_id: str,
        reference_event_id: str
    ) -> List[Dict]:
        """Get events that occurred after a reference event."""

    def get_events_before(
        self,
        user_id: str,
        reference_event_id: str
    ) -> List[Dict]:
        """Get events that occurred before a reference event."""

    def summarize_timeline(
        self,
        user_id: str,
        time_window: Optional[TimeWindow] = None
    ) -> str:
        """Generate natural language summary of timeline."""

    def get_timeline_stats(
        self,
        user_id: str
    ) -> Dict:
        """
        Get timeline statistics.

        Returns:
            {
                'total_events': int,
                'modalities': Dict[str, int],
                'time_span': str,
                'avg_confidence': float
            }
        """
```

### TimelineQueries

```python
class TimelineQueries:
    def __init__(self, timeline_builder: TimelineBuilder):
        """Initialize with timeline builder."""

    def what_happened_after(
        self,
        user_id: str,
        reference_event: str
    ) -> List[Dict]:
        """Find events after a reference event."""

    def what_happened_before(
        self,
        user_id: str,
        reference_event: str
    ) -> List[Dict]:
        """Find events before a reference event."""

    def events_on_date(
        self,
        user_id: str,
        date: datetime
    ) -> List[Dict]:
        """Get events on a specific date."""

    def events_this_week(
        self,
        user_id: str
    ) -> List[Dict]:
        """Get events from current week (Monday-Sunday)."""

    def events_last_n_days(
        self,
        user_id: str,
        n_days: int
    ) -> List[Dict]:
        """Get events from last N days."""

    def events_between_dates(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Get events between two dates."""

    def find_event_by_description(
        self,
        user_id: str,
        description: str
    ) -> List[Dict]:
        """Find events matching text description."""

    def find_events_by_modality(
        self,
        user_id: str,
        modality: str
    ) -> List[Dict]:
        """Filter events by modality (text, image, audio)."""

    def get_recent_events(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get N most recent events."""

    def search_timeline(
        self,
        user_id: str,
        query: str
    ) -> List[Dict]:
        """Text-based search in timeline."""

    def get_timeline_summary_for_period(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """Get natural language summary for time period."""
```

### TimeWindow

```python
class TimeWindow:
    def __init__(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ):
        """
        Create time window for filtering.

        Args:
            start: Start time (None = unbounded past)
            end: End time (None = unbounded future)
        """

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within window."""

    def is_unbounded(self) -> bool:
        """Check if window has no bounds."""
```

---

## 🔗 Integration with Other TCMGM 

### Temporal Attributes

Timeline events use `TemporalEvent` from 

```python
from outhad_contextkit.memory.temporal import TemporalEvent

# Events have temporal attributes
event = TemporalEvent(
    id="evt_1",
    content="User logged in",
    timestamp=datetime.utcnow(),
    confidence=0.95,
    modality="text"
)
```

### Causal Relationships

Timeline automatically extracts causal relationships:

```python
# Build timeline
result = timeline_builder.build_from_transcript(
    transcript=transcript,
    user_id="user_123",
    session_start=datetime.utcnow()
)

# Causal links extracted automatically
print(f"Events: {len(result['events'])}")
print(f"Causal links: {len(result['causal_links'])}")

# Access causal relationships
for link in result['causal_links']:
    print(f"{link['cause_id']} → {link['effect_id']}")
```

### Multimodal Support

Timeline supports multimodal events:

```python
# Filter timeline by modality
queries = TimelineQueries(timeline_builder)

text_events = queries.find_events_by_modality("user_123", "text")
image_events = queries.find_events_by_modality("user_123", "image")
audio_events = queries.find_events_by_modality("user_123", "audio")

print(f"Text: {len(text_events)}")
print(f"Images: {len(image_events)}")
print(f"Audio: {len(audio_events)}")
```

### Retrieval Orchestrator (Future)

Timeline queries will integrate with fused retrieval:

```python
# Future: Timeline + Vector + Graph search
from outhad_contextkit.memory import Memory

memory = Memory()

# Search will include timeline context
results = memory.search(
    query="What did I do last week?",
    user_id="user_123",
    include_timeline=True  # Future feature
)
```

---

## 🎉 Summary

Timeline Helper provides:

✅ **Automatic event extraction** from conversations
✅ **Flexible time-based queries** with TimeWindow
✅ **Natural language queries** like "What happened after X?"
✅ **Timeline summarization** with LLM
✅ **Statistics and analytics** for timeline monitoring
✅ **Multimodal support** for text, image, and audio events
✅ **Causal relationship integration** from 
✅ **Production-ready** with robust error handling

### Quick Reference

```python
# Initialize
from outhad_contextkit.memory.timeline import TimelineBuilder
from outhad_contextkit.memory.temporal.timeline_queries import TimelineQueries

timeline_builder = TimelineBuilder(llm, graph)
queries = TimelineQueries(timeline_builder)

# Build timeline
result = timeline_builder.build_from_transcript(
    transcript=transcript,
    user_id="user_123",
    session_start=datetime.utcnow()
)

# Query timeline
recent = queries.get_recent_events("user_123", limit=10)
today = queries.events_on_date("user_123", datetime.utcnow())
after = queries.what_happened_after("user_123", "login")

# Get insights
summary = timeline_builder.summarize_timeline("user_123")
stats = timeline_builder.get_timeline_stats("user_123")
```

