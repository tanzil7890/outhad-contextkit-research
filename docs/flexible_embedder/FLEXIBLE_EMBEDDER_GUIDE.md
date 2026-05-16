# Flexible Embedder System - User Guide


---

## Overview

Includes a **flexible embedder system** that allows you to easily switch between different models for each modality (text, images, audio) without breaking your code.

### Why This Matters

- ✅ **Switch audio embedders**: Choose between Whisper (speech) or CLAP (all audio)
- ✅ **No code changes**: Just change configuration, everything still works
- ✅ **Add new models**: Extend with custom embedders easily
- ✅ **Mix and match**: Different models for different modalities

---

## Quick Start

### Option 1: Default Setup (Speech-focused)

```python
from outhad_contextkit.memory.temporal import create_default_embedder

# Uses: CLIP for text/images, Whisper for audio
embedder = create_default_embedder()

# Use normally
text_emb = embedder.embed_text("dog in snow")
image_emb = embedder.embed_image(my_image)
audio_emb = embedder.embed_audio(audio_bytes)  # Speech only
```

### Option 2: CLAP Setup (All audio types)

```python
from outhad_contextkit.memory.temporal import create_clap_embedder

# Uses: CLIP for text/images, CLAP for audio
embedder = create_clap_embedder()

# Now audio works for environmental sounds too!
audio_emb = embedder.embed_audio(birds_chirping_wav)  # ✅ Works!
audio_emb = embedder.embed_audio(thunder_wav)  # ✅ Works!
audio_emb = embedder.embed_audio(speech_wav)  # ✅ Still works!
```

### Option 3: Hybrid Setup (Maximum flexibility)

```python
from outhad_contextkit.memory.temporal import create_hybrid_embedder

# Uses: CLIP for text/images, CLAP+Whisper for audio
# Strategy: Try CLAP first, fallback to Whisper if unavailable
embedder = create_hybrid_embedder()

# Works with everything, robust to missing dependencies
audio_emb = embedder.embed_audio(any_audio_file)
```

---

## Advanced Configuration

### Custom Configuration

```python
from outhad_contextkit.memory.temporal import create_embedder

# Fully customizable
embedder = create_embedder(
    text_embedder="clip",     # or "openai"
    image_embedder="clip",    # currently only option
    audio_embedder="clap",    # "whisper", "clap", or "hybrid"
    openai_api_key="sk-..."   # optional, reads from env
)
```

### Available Options

#### Text Embedders
- `"clip"` - CLIP text encoder (recommended for cross-modal search)
- `"openai"` - OpenAI text-embedding-ada-002 (pure text search)

#### Image Embedders
- `"clip"` - CLIP image encoder (currently only option)

#### Audio Embedders
- `"whisper"` - Whisper transcription + text embedding
  - ✅ Best for: Speech (conversations, lectures, meetings)
  - ❌ Poor for: Environmental sounds (birds, thunder, etc.)
  
- `"clap"` - CLAP audio-language model
  - ✅ Best for: All audio (speech + environmental sounds)
  - ✅ Handles: Birds, thunder, music, effects, and speech
  
- `"hybrid"` - Intelligent combination
  - ✅ Tries CLAP first (handles everything)
  - ✅ Falls back to Whisper if CLAP unavailable
  - ✅ Most robust option

---

## Switching Embedders

### Example: Switch from Whisper to CLAP

**Before** (Speech only):
```python
embedder = create_embedder(audio_embedder="whisper")

# Works for speech
audio_emb = embedder.embed_audio(meeting_recording)  # ✅

# Doesn't work well for environmental sounds
audio_emb = embedder.embed_audio(birds_chirping)  # ❌ Poor results
```

**After** (All audio types):
```python
# Just change one parameter!
embedder = create_embedder(audio_embedder="clap")

# Works for speech
audio_emb = embedder.embed_audio(meeting_recording)  # ✅

# Now works for environmental sounds too!
audio_emb = embedder.embed_audio(birds_chirping)  # ✅ Great results!
```

**Everything else stays the same!** No other code changes needed.

---

## Installation

### Default (Whisper only)

```bash
pip install outhad-contextkitai[multimodal]
```

### With CLAP Support

```bash
# Install multimodal dependencies including CLAP
pip install outhad-contextkitai[multimodal]

# CLAP will be included in the multimodal extras
```

### Manual CLAP Installation

```bash
pip install laion-clap
```

---

## Use Cases

### Use Case 1: Podcast/Interview Search (Speech-focused)

```python
from outhad_contextkit.memory.temporal import create_default_embedder

# Use default (Whisper) for speech-heavy content
embedder = create_default_embedder()

# Index podcast episodes
for episode_audio in podcast_episodes:
    embedding = embedder.embed_audio(episode_audio)
    # Store embedding...

# Search with text queries
results = search_audio("discussion about AI ethics")
```

### Use Case 2: Music/Sound Effect Library

```python
from outhad_contextkit.memory.temporal import create_clap_embedder

# Use CLAP for environmental sounds and music
embedder = create_clap_embedder()

# Index sound effects
for sound_file in sound_library:
    embedding = embedder.embed_audio(sound_file)
    # Store embedding...

# Search with text queries
results = search_audio("birds chirping in forest")  # ✅ Works!
results = search_audio("thunder storm")  # ✅ Works!
results = search_audio("upbeat jazz music")  # ✅ Works!
```

### Use Case 3: Mixed Content (Speech + Sounds)

```python
from outhad_contextkit.memory.temporal import create_hybrid_embedder

# Use hybrid for maximum flexibility
embedder = create_hybrid_embedder()

# Works for everything
speech_emb = embedder.embed_audio(interview_clip)  # ✅
sound_emb = embedder.embed_audio(thunder_sound)  # ✅
music_emb = embedder.embed_audio(background_music)  # ✅
```

---

## Integration with MemoryGraph

### Example: Flexible Multimodal Memory

```python
from outhad_contextkit.memory.graph_memory import MemoryGraph
from outhad_contextkit.memory.temporal import create_clap_embedder

# Create memory graph with CLAP for audio
embedder = create_clap_embedder()

memory_graph = MemoryGraph(
    config=your_config,
    embedding_model=embedder  # Use flexible embedder
)

# Now add multimodal events
memory_graph.add_multimodal_event(
    content=thunder_audio,
    modality="audio",
    filters={"user_id": "user_123"}
)

# Search across modalities
results = memory_graph.search_multimodal(
    query_content="thunder storm",
    query_modality="text",
    filters={"user_id": "user_123"},
    modality_filter="audio"
)
```

---

## Creating Custom Embedders

### Extend with Your Own Model

```python
from outhad_contextkit.memory.temporal.base_embedders import BaseAudioEmbedder

class MyCustomAudioEmbedder(BaseAudioEmbedder):
    """Custom audio embedder using your own model."""
    
    def __init__(self):
        self.model = load_your_model()
    
    def embed_audio(self, audio_bytes: bytes) -> List[float]:
        # Your custom logic here
        features = extract_features(audio_bytes)
        embedding = self.model.encode(features)
        return embedding.tolist()
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "model": "my-custom-model",
            "available": True,
            "dimensions": 512,
            "type": "custom-audio",
            "best_for": "specialized audio tasks"
        }

# Use your custom embedder
from outhad_contextkit.memory.temporal.embedder_factory import FlexibleMultimodalEmbedder
from outhad_contextkit.memory.temporal.text_embedders import CLIPTextEmbedder
from outhad_contextkit.memory.temporal.image_embedders import CLIPImageEmbedder

embedder = FlexibleMultimodalEmbedder.__init__(
    text_embedder=CLIPTextEmbedder(),
    image_embedder=CLIPImageEmbedder(),
    audio_embedder=MyCustomAudioEmbedder()  # Your custom embedder!
)
```

---

## Comparison Table

| Feature | Whisper | CLAP | Hybrid |
|---------|---------|------|--------|
| **Speech** | ✅ Excellent | ✅ Good | ✅ Excellent |
| **Environmental Sounds** | ❌ Poor | ✅ Excellent | ✅ Excellent |
| **Music** | ❌ Poor | ✅ Good | ✅ Good |
| **Installation** | Simple | Requires laion-clap | Requires laion-clap |
| **Dependencies** | openai | laion-clap | Both |
| **Robustness** | Moderate | Moderate | High (fallback) |
| **Use Case** | Speech-only | All audio | Production (mixed) |

---





**New code** (flexible system):
```python
# Option 1: Keep old behavior (backward compatible)
from outhad_contextkit.memory.temporal import create_production_embedder
embedder = create_production_embedder()  # Still works!

# Option 2: Use new flexible system
from outhad_contextkit.memory.temporal import create_default_embedder
embedder = create_default_embedder()  # Same behavior

# Option 3: Switch to CLAP
from outhad_contextkit.memory.temporal import create_clap_embedder
embedder = create_clap_embedder()  # Better audio!
```


---

## Best Practices

### 1. Choose the Right Embedder for Your Use Case

- **Speech-heavy**: Use `create_default_embedder()` or `audio_embedder="whisper"`
- **Environmental sounds**: Use `create_clap_embedder()` or `audio_embedder="clap"`
- **Mixed content**: Use `create_hybrid_embedder()` or `audio_embedder="hybrid"`

### 2. Test Before Switching

```python
# Test with your actual data
embedder_whisper = create_embedder(audio_embedder="whisper")
embedder_clap = create_embedder(audio_embedder="clap")

# Compare results
results_whisper = search_with_embedder(embedder_whisper, query)
results_clap = search_with_embedder(embedder_clap, query)

# Pick the one that works better for your use case
```

### 3. Use Hybrid for Production

```python
# Most robust option - works even if CLAP isn't installed
embedder = create_hybrid_embedder()
```

### 4. Check Capabilities

```python
embedder = create_clap_embedder()

# Check what's available
capabilities = embedder.get_capabilities()
print(capabilities)

# Output:
# {
#   "text": {"model": "openai/clip-vit-base-patch32", "available": True, ...},
#   "image": {"model": "openai/clip-vit-base-patch32", "available": True, ...},
#   "audio": {"model": "LAION-CLAP", "available": True, ...}
# }
```

---

## Troubleshooting

### CLAP Not Working?

```python
# Check if CLAP is installed
try:
    import laion_clap
    print("✅ CLAP installed")
except ImportError:
    print("❌ CLAP not installed")
    print("Install with: pip install laion-clap")
```

### Whisper Not Working?

```python
# Check OpenAI API key
import os
print("API key:", os.getenv("OPENAI_API_KEY"))

# Or pass explicitly
embedder = create_embedder(
    audio_embedder="whisper",
    openai_api_key="sk-..."
)
```

### Embeddings All Similar?

```python
# If audio embeddings are all similar (e.g., 0.80-0.91),
# you're probably using Whisper for environmental sounds

# Solution: Switch to CLAP!
embedder = create_clap_embedder()
```

---

## Summary

### Key Benefits

✅ **Flexibility**: Switch models without code changes  
✅ **No Breaking Changes**: Old code continues to work  
✅ **Easy Configuration**: Simple API for common cases  
✅ **Extensible**: Add your own custom embedders  
✅ **Production-Ready**: Robust fallback mechanisms  

### Quick Reference

```python
# Speech only
from outhad_contextkit.memory.temporal import create_default_embedder
embedder = create_default_embedder()

# All audio
from outhad_contextkit.memory.temporal import create_clap_embedder
embedder = create_clap_embedder()

# Maximum flexibility
from outhad_contextkit.memory.temporal import create_hybrid_embedder
embedder = create_hybrid_embedder()

# Custom
from outhad_contextkit.memory.temporal import create_embedder
embedder = create_embedder(
    text_embedder="clip",
    image_embedder="clip",
    audio_embedder="clap"
)
```

---



