# Installing Multimodal Dependencies for Production



## 📦 Installation Steps

### Step 1: Install Core Dependencies

```bash
# Install multimodal dependencies
pip install "outhad_contextkitai[multimodal]"

# Or install manually:
pip install transformers torch torchvision openai pillow numpy
```

### Step 2: Fix macOS SSL Certificate Issue (if needed)

If you see SSL certificate errors when downloading CLIP:

```bash
# Run this command to install certificates
/Applications/Python\ 3.11/Install\ Certificates.command

# Or manually install certifi
pip install --upgrade certifi
```

### Step 3: Verify Installation

```bash
python test_production_multimodal.py
```

---

## 🔑 API Key Setup

Make sure your `.env` file contains:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 📊 What Gets Installed

| Package | Purpose | Size |
|---------|---------|------|
| `transformers` | CLIP model for images | ~100MB |
| `torch` | PyTorch for model inference | ~200MB |
| `torchvision` | Image preprocessing | ~50MB |
| `openai` | Whisper API client | ~5MB |

**Total**: ~355MB

---

## 🎯 Quick Test

After installation:

```python
from outhad_contextkit.memory.temporal import create_production_embedder
from PIL import Image

# Initialize
embedder = create_production_embedder()

# Check what's available
caps = embedder.get_capabilities()
print(f"CLIP available: {caps['image']}")
print(f"Whisper available: {caps['audio']}")

# Test image embedding
img = Image.open("photo.jpg")
embedding = embedder.embed_image(img)
print(f"Image embedding: {len(embedding)} dimensions")
```

---

## 🐛 Troubleshooting

### Issue: SSL Certificate Error

**Solution**:
```bash
# macOS:
/Applications/Python\ 3.11/Install\ Certificates.command

# Or set environment variable:
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
```

### Issue: CUDA Out of Memory

**Solution**: CLIP will automatically fall back to CPU:
```python
# It detects and uses CPU automatically
embedder = create_production_embedder()
# Will use CPU if GPU not available
```

### Issue: Whisper API Error

**Solution**: Make sure your API key is valid:
```bash
# Test your API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

---

## 📝 Production Usage

```python
from outhad_contextkit.memory.graph_memory import MemoryGraph
from outhad_contextkit.memory.temporal import create_production_embedder
from outhad_contextkit.configs.base import MemoryConfig
from PIL import Image

# Initialize with production embedder
config = MemoryConfig()
memory_graph = MemoryGraph(config)

# The graph will automatically use production embeddings
img = Image.open("photo.jpg")
memory_graph.add_multimodal_event(
    content=img,
    modality="image",
    filters={"user_id": "user_123"}
)

# Search across modalities
results = memory_graph.search_multimodal(
    query_content="beautiful sunset",
    query_modality="text",
    filters={"user_id": "user_123"},
    modality_filter="image"
)
```

---

## ✅ Verification Checklist

After installation, verify:

- [ ] CLIP model loads successfully
- [ ] Whisper API connects
- [ ] Image embeddings are 768-dim
- [ ] Audio transcription works
- [ ] Cross-modal search returns results
- [ ] No SSL certificate errors

Run: `python test_production_multimodal.py`

---

## 📚 Additional Resources

- **CLIP Model**: https://huggingface.co/openai/clip-vit-base-patch32
- **Whisper API**: https://platform.openai.com/docs/guides/speech-to-text
- **PyTorch**: https://pytorch.org/get-started/locally/

---



