#  TCMGM: Multimodal Support - Usage Guide

## 🎯 Overview

THis is **multimodal support** to TCMGM. This enables the system to handle content beyond text—including images, audio, and video—with cross-modal retrieval capabilities.

---

## 🚀 Quick Start

### 1. Import Components

```python
from outhad_contextkit.memory.temporal import (
    MultimodalContent,
    MultimodalEmbedder,
    create_text_content,
    create_image_content,
    create_audio_content,
    cross_modal_search,
    find_image_mentions_in_text,
    find_audio_mentions_in_text,
    compute_multimodal_relevance,
    group_by_modality
)
from PIL import Image
```

### 2. Create Multimodal Content

```python
# Text content
text_content = create_text_content(
    "Beautiful sunset over the ocean",
    metadata={"source": "user_message"}
)

# Image content
img = Image.open("photo.jpg")
image_content = create_image_content(
    img,
    metadata={"source": "camera", "location": "beach"}
)

# Audio content (bytes)
audio_bytes = open("recording.wav", "rb").read()
audio_content = create_audio_content(
    audio_bytes,
    metadata={"duration_sec": 30}
)
```

---

## 📖 Core Features

### Feature 1: Content Hashing for Deduplication

Every piece of content gets a unique SHA256 hash for deduplication:

```python
from outhad_contextkit.memory.temporal import MultimodalContent

content = MultimodalContent("Hello world", "text")
hash1 = content.compute_hash()

# Same content = same hash
content2 = MultimodalContent("Hello world", "text")
hash2 = content2.compute_hash()

assert hash1 == hash2  # ✅ Deduplication works!
```

**Use Cases**:
- Prevent storing duplicate images
- Check if content already exists
- Reference content by hash

---

### Feature 2: Multimodal Embeddings

Generate embeddings for any modality:

```python
from outhad_contextkit.memory.temporal import MultimodalEmbedder
from outhad_contextkit.utils.factory import EmbeddingFactory

# Initialize embedder
embedding_model = EmbeddingFactory.create("openai")
embedder = MultimodalEmbedder(embedding_model)

# Embed text
text_emb = embedder.embed_text("sunset over ocean")
print(f"Text embedding: dim={len(text_emb)}")

# Embed image (placeholder - integrate CLIP for production)
img = Image.open("photo.jpg")
image_emb = embedder.embed_image(img)
print(f"Image embedding: dim={len(image_emb)}")

# Embed audio (placeholder - integrate Wav2Vec2 for production)
audio_bytes = b"audio data"
audio_emb = embedder.embed_audio(audio_bytes)
print(f"Audio embedding: dim={len(audio_emb)}")
```

**Note**: Image and audio embeddings currently use placeholders. For production:
- **Images**: Integrate CLIP, ViT, or ResNet
- **Audio**: Integrate Wav2Vec2 or Whisper embeddings

---

### Feature 3: Cross-Modal Search

Find content across modalities using semantic similarity:

```python
from outhad_contextkit.memory.temporal import cross_modal_search

# Example: Find images using text query
query_embedding = embedder.embed_text("beautiful sunset")

candidates = [
    {
        "id": "img1",
        "content": "sunset_photo.jpg",
        "modality": "image",
        "embedding": image_embedding1
    },
    {
        "id": "img2",
        "content": "car_photo.jpg",
        "modality": "image",
        "embedding": image_embedding2
    },
    {
        "id": "text1",
        "content": "Amazing sunset colors",
        "modality": "text",
        "embedding": text_embedding1
    }
]

# Search across all modalities
results = cross_modal_search(
    query_embedding=query_embedding,
    candidate_embeddings=candidates,
    top_k=3
)

for result in results:
    print(f"{result['content']} ({result['modality']})")
    print(f"  Similarity: {result['similarity']:.4f}")
```

**Filter by Modality**:
```python
# Only search images
image_results = cross_modal_search(
    query_embedding=query_embedding,
    candidate_embeddings=candidates,
    modality_filter="image",  # Only return images
    top_k=5
)
```

**Set Minimum Similarity**:
```python
# Only return highly similar results
results = cross_modal_search(
    query_embedding=query_embedding,
    candidate_embeddings=candidates,
    min_similarity=0.8,  # At least 80% similar
    top_k=10
)
```

---

### Feature 4: Add Multimodal Events to Graph

Store multimodal content in the memory graph:

```python
from outhad_contextkit.memory.graph_memory import MemoryGraph
from outhad_contextkit.configs.base import MemoryConfig
from PIL import Image
from datetime import datetime

# Initialize memory graph
config = MemoryConfig()
memory_graph = MemoryGraph(config)

# Add image event
img = Image.open("sunset.jpg")
memory_graph.add_multimodal_event(
    content=img,
    modality="image",
    filters={"user_id": "user_123"},
    timestamp=datetime.utcnow(),
    confidence=1.0,
    metadata={"location": "beach", "time": "evening"}
)

print("✅ Image added to memory graph")
```

**Add Text Event**:
```python
memory_graph.add_multimodal_event(
    content="Beautiful sunset over the ocean",
    modality="text",
    filters={"user_id": "user_123"},
    metadata={"source": "chat"}
)
```

**Add Audio Event**:
```python
audio_bytes = open("recording.wav", "rb").read()
memory_graph.add_multimodal_event(
    content=audio_bytes,
    modality="audio",
    filters={"user_id": "user_123"},
    metadata={"duration_sec": 30}
)
```

---

### Feature 5: Search Multimodal Content in Graph

Search for content across modalities stored in the graph:

```python
# Find images using text query
results = memory_graph.search_multimodal(
    query_content="sunset over ocean",
    query_modality="text",
    filters={"user_id": "user_123"},
    modality_filter="image",  # Only return images
    top_k=5
)

for result in results:
    print(f"Found: {result['content']}")
    print(f"Modality: {result['modality']}")
    print(f"Similarity: {result['similarity']:.4f}")
```

**Search All Modalities**:
```python
# Search across text, images, audio, etc.
results = memory_graph.search_multimodal(
    query_content="vacation memories",
    query_modality="text",
    filters={"user_id": "user_123"},
    top_k=10
)

# Group results by modality
from outhad_contextkit.memory.temporal import group_by_modality

grouped = group_by_modality(results)
print(f"Found {len(grouped['text'])} text items")
print(f"Found {len(grouped['image'])} images")
print(f"Found {len(grouped['audio'])} audio clips")
```

---

### Feature 6: Find Image Mentions in Text

Detect when text references an image:

```python
from outhad_contextkit.memory.temporal import find_image_mentions_in_text
from datetime import datetime

# Compute image hash
image_hash = image_content.compute_hash()

# Text events from conversation
text_events = [
    {
        "content": "Look at this beautiful photo I took",
        "timestamp": datetime.now().isoformat()
    },
    {
        "content": "The image shows amazing colors",
        "timestamp": datetime.now().isoformat()
    },
    {
        "content": "Just some random text",
        "timestamp": datetime.now().isoformat()
    }
]

# Find mentions
mentions = find_image_mentions_in_text(
    image_hash=image_hash,
    text_events=text_events
)

for mention in mentions:
    print(f"Found mention: {mention['event']['content']}")
    print(f"Confidence: {mention['confidence']:.2f}")
    print(f"Keywords: {', '.join(mention['matched_keywords'])}")
```

**Keywords Detected**:
- "image", "picture", "photo", "screenshot"
- "look at", "see this", "showed", "attached"
- "visual", "diagram", "chart", "illustration"

---

### Feature 7: Find Audio Mentions in Text

Detect when text references audio:

```python
from outhad_contextkit.memory.temporal import find_audio_mentions_in_text

audio_hash = audio_content.compute_hash()

text_events = [
    {"content": "Listen to this recording", "timestamp": "..."},
    {"content": "The voice said hello", "timestamp": "..."},
    {"content": "Sound of waves crashing", "timestamp": "..."}
]

mentions = find_audio_mentions_in_text(
    audio_hash=audio_hash,
    text_events=text_events
)

for mention in mentions:
    print(f"Found: {mention['event']['content']}")
    print(f"Confidence: {mention['confidence']:.2f}")
```

**Keywords Detected**:
- "audio", "sound", "voice", "recording"
- "listen", "heard", "said", "speaking"
- "music", "clip", "transcript"

---

### Feature 8: Multimodal Relevance Scoring

Compute relevance scores with modality-aware penalties/bonuses:

```python
from outhad_contextkit.memory.temporal import compute_multimodal_relevance

base_score = 0.8

# Same modality gets bonus
same_modal_score = compute_multimodal_relevance(
    query_modality="text",
    result_modality="text",
    similarity_score=base_score
)
print(f"Text→Text: {base_score} → {same_modal_score:.2f}")
# Result: 0.92 (10% bonus + 5% session bonus)

# Cross-modal gets penalty
cross_modal_score = compute_multimodal_relevance(
    query_modality="text",
    result_modality="audio",
    similarity_score=base_score,
    same_session=False
)
print(f"Text→Audio: {base_score} → {cross_modal_score:.2f}")
# Result: 0.72 (10% penalty)
```

**Modality Factors**:
- Same modality: 1.1x bonus
- Text ↔ Image: 0.95x (common)
- Text ↔ Audio: 0.90x (less common)
- Image ↔ Audio: 0.85x (distant)
- Same session: 1.05x bonus

---

### Feature 9: Base64 Encoding

Convert content to base64 for serialization:

```python
# Text to base64
text_b64 = text_content.to_base64()
print(f"Text base64: {text_b64}")

# Image to base64
image_b64 = image_content.to_base64()
print(f"Image base64: {len(image_b64)} chars")

# Audio to base64
audio_b64 = audio_content.to_base64()
print(f"Audio base64: {len(audio_b64)} chars")
```

**Use Cases**:
- Serialize for storage in JSON
- Transfer over network
- Store in non-binary databases

---

## 🎯 Common Use Cases

### Use Case 1: Visual Memory System

Build a memory system that remembers images:

```python
from outhad_contextkit.memory.graph_memory import MemoryGraph
from PIL import Image

memory_graph = MemoryGraph(config)

# User shows an image
img = Image.open("vacation_photo.jpg")
memory_graph.add_multimodal_event(
    content=img,
    modality="image",
    filters={"user_id": "user_123"},
    metadata={"location": "Paris", "date": "2025-01-14"}
)

# User adds context
memory_graph.add_multimodal_event(
    content="This was taken at the Eiffel Tower",
    modality="text",
    filters={"user_id": "user_123"}
)

# Later: User asks "Show me photos from Paris"
results = memory_graph.search_multimodal(
    query_content="Paris vacation photos",
    query_modality="text",
    filters={"user_id": "user_123"},
    modality_filter="image"
)

for result in results:
    print(f"Found photo: {result['metadata']['location']}")
```

### Use Case 2: Voice Note Memory

Remember audio recordings with context:

```python
# User records voice note
audio_bytes = record_audio()
audio_hash = hashlib.sha256(audio_bytes).hexdigest()

memory_graph.add_multimodal_event(
    content=audio_bytes,
    modality="audio",
    filters={"user_id": "user_123"},
    metadata={"topic": "meeting_notes", "duration": 120}
)

# User references it in text
memory_graph.add_multimodal_event(
    content=f"The recording {audio_hash[:8]} has important action items",
    modality="text",
    filters={"user_id": "user_123"}
)

# Later: Find all text referencing this audio
text_events = memory_graph.search(...)  # Get text events
mentions = find_audio_mentions_in_text(audio_hash, text_events)

print(f"Found {len(mentions)} text references to this audio")
```

### Use Case 3: Image Deduplication

Prevent storing duplicate images:

```python
def add_image_if_new(img, filters):
    """Add image only if it doesn't already exist."""
    # Compute hash
    content = create_image_content(img)
    img_hash = content.compute_hash()
    
    # Check if already exists
    existing = memory_graph.search_by_hash(img_hash)
    
    if existing:
        print("Image already stored!")
        return existing
    else:
        # Add new image
        memory_graph.add_multimodal_event(
            content=img,
            modality="image",
            filters=filters
        )
        print("New image added!")
```

### Use Case 4: Cross-Modal Question Answering

Answer questions using multimodal context:

```python
# User asks: "What was in the photo I showed you?"
query = "photo I showed you"
query_emb = embedder.embed_text(query)

# Search for images
image_results = memory_graph.search_multimodal(
    query_content=query,
    query_modality="text",
    filters={"user_id": "user_123"},
    modality_filter="image",
    top_k=1
)

# Find text near that image
if image_results:
    img_hash = image_results[0]['metadata']['content_hash']
    text_events = memory_graph.search(...)  # Get recent text
    
    mentions = find_image_mentions_in_text(img_hash, text_events)
    
    for mention in mentions:
        print(f"Context: {mention['event']['content']}")
```

---

## 🔍 Best Practices

### 1. Use Content Hashing for Deduplication

```python
# Always check hash before storing
content = create_image_content(img)
content_hash = content.compute_hash()

# Check if exists
existing = search_by_hash(content_hash)
if not existing:
    store(content)
```

### 2. Add Rich Metadata

```python
# Good: Rich metadata
memory_graph.add_multimodal_event(
    content=img,
    modality="image",
    filters={"user_id": "user_123"},
    metadata={
        "location": "beach",
        "time": "sunset",
        "camera": "iPhone 14",
        "resolution": "1920x1080"
    }
)
```

### 3. Use Modality Filters for Efficient Search

```python
# More efficient: Filter by modality
results = memory_graph.search_multimodal(
    query_content="sunset",
    query_modality="text",
    filters={"user_id": "user_123"},
    modality_filter="image"  # Only search images
)
```

### 4. Group Results by Modality

```python
# Search all modalities
all_results = memory_graph.search_multimodal(...)

# Group for presentation
grouped = group_by_modality(all_results)

print(f"Text: {len(grouped.get('text', []))} results")
print(f"Images: {len(grouped.get('image', []))} results")
print(f"Audio: {len(grouped.get('audio', []))} results")
```

---

## 📊 Performance Tips

### Batch Processing

```python
# Process multiple images at once
images = load_images()
for img in images:
    memory_graph.add_multimodal_event(
        content=img,
        modality="image",
        filters=filters
    )
```

### Use Minimum Similarity Threshold

```python
# Skip low-relevance results
results = cross_modal_search(
    query_embedding=query_emb,
    candidate_embeddings=candidates,
    min_similarity=0.7,  # Only 70%+ similar
    top_k=10
)
```

---

## 🐛 Troubleshooting

### Issue: Image Embedding is All Zeros

```python
# Problem: Placeholder embedder
embedder = MultimodalEmbedder(embedding_model)
embedding = embedder.embed_image(img)
# Returns: [0.0, 0.0, ...]

# Solution: Integrate CLIP for production
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image_with_clip(img):
    inputs = processor(images=img, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs[0].tolist()
```

### Issue: No Cross-Modal Results

```python
# Problem: Embeddings from different spaces
text_emb = text_embedder.embed("sunset")
image_emb = image_embedder.embed(img)
# Incompatible embedding spaces!

# Solution: Use unified embedding model (e.g., CLIP)
from transformers import CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
text_emb = model.get_text_features(...)
image_emb = model.get_image_features(...)
# Now comparable!
```

---
