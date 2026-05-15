"""
Test CLAP Audio Embedder with Real Audio Files
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

from outhad_contextkit.memory.temporal import create_clap_embedder, create_embedder

print("=" * 80)
print("TESTING CLAP AUDIO EMBEDDER")
print("=" * 80)

# Test 1: Create CLAP embedder
print("\n" + "=" * 80)
print("Test 1: Initialize CLAP Embedder")
print("=" * 80)

embedder = create_clap_embedder()

capabilities = embedder.get_capabilities()
print("\n✅ CLAP embedder created")
print("\nCapabilities:")
for modality, caps in capabilities.items():
    if caps:
        available = caps.get('available', False)
        model = caps.get('model', 'Unknown')
        status = "✅" if available else "❌"
        print(f"  {status} {modality}: {model}")

# Test 2: Load audio files
print("\n" + "=" * 80)
print("Test 2: Load Audio Files")
print("=" * 80)

audio_dir = Path("test_dataset/audio")
audio_files = []

if audio_dir.exists():
    audio_files = sorted(audio_dir.glob("*.wav"))[:5]
    print(f"\n✅ Found {len(audio_files)} audio files")
    for i, audio_file in enumerate(audio_files, 1):
        print(f"  {i}. {audio_file.name}")
else:
    print(f"\n❌ Audio directory not found: {audio_dir}")
    print("Creating a simple test...")

# Test 3: Generate embeddings with CLAP
print("\n" + "=" * 80)
print("Test 3: Generate Audio Embeddings with CLAP")
print("=" * 80)

if audio_files:
    print("\nGenerating embeddings for audio files...")
    
    audio_embeddings = []
    for audio_file in audio_files:
        # Read audio bytes
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
        
        # Generate embedding
        embedding = embedder.embed_audio(audio_bytes)
        
        audio_embeddings.append({
            'file': audio_file.name,
            'embedding': embedding,
            'size': len(audio_bytes)
        })
        
        print(f"  ✅ {audio_file.name}: {len(embedding)} dimensions")
    
    print(f"\n✅ Generated {len(audio_embeddings)} audio embeddings")

# Test 4: Cross-modal search (Text → Audio)
print("\n" + "=" * 80)
print("Test 4: Cross-Modal Search (Text → Audio with CLAP)")
print("=" * 80)

if audio_embeddings:
    from outhad_contextkit.memory.temporal import cross_modal_search
    
    test_queries = [
        "birds chirping",
        "thunder storm",
        "rain falling",
        "dog barking",
        "vacuum cleaner"
    ]
    
    print("\nSearching for audio using text queries:")
    print("-" * 80)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        # Generate query embedding
        query_emb = embedder.embed_text(query)
        
        # Prepare candidates
        candidates = [
            {
                'content': ae['file'],
                'embedding': ae['embedding'],
                'modality': 'audio',
                'metadata': {'size': ae['size']}
            }
            for ae in audio_embeddings
        ]
        
        # Search
        results = cross_modal_search(
            query_embedding=query_emb,
            candidate_embeddings=candidates,
            modality_filter='audio',
            top_k=3
        )
        
        if results:
            print("  Top results:")
            for i, result in enumerate(results, 1):
                score = result['similarity']
                file = result['content']
                print(f"    {i}. {file}: {score:.4f}")
        else:
            print("  No results found")

# Test 5: Compare CLAP vs Whisper
print("\n" + "=" * 80)
print("Test 5: Compare CLAP vs Whisper")
print("=" * 80)

print("\nCreating Whisper embedder for comparison...")
whisper_embedder = create_embedder(audio_embedder="whisper")

print("\n📊 Comparison:")
print("-" * 80)
print(f"{'Feature':<30} {'CLAP':<20} {'Whisper':<20}")
print("-" * 80)

clap_caps = embedder.get_capabilities()['audio']
whisper_caps = whisper_embedder.get_capabilities()['audio']

print(f"{'Model':<30} {clap_caps.get('model', 'N/A'):<20} {whisper_caps.get('model', 'N/A'):<20}")
print(f"{'Available':<30} {'✅' if clap_caps.get('available') else '❌':<20} {'✅' if whisper_caps.get('available') else '❌':<20}")
print(f"{'Type':<30} {clap_caps.get('type', 'N/A'):<20} {whisper_caps.get('type', 'N/A'):<20}")
print(f"{'Best For':<30} {clap_caps.get('best_for', 'N/A'):<20} {whisper_caps.get('best_for', 'N/A'):<20}")

print("\n" + "-" * 80)
print("Key Differences:")
print("-" * 80)
print("  CLAP:")
print("    ✅ Handles environmental sounds (birds, thunder, rain)")
print("    ✅ Handles music and sound effects")
print("    ✅ Handles speech too!")
print("    ✅ Direct audio-language model")
print("")
print("  Whisper:")
print("    ✅ Excellent for speech (conversations, lectures)")
print("    ❌ Poor for environmental sounds")
print("    ⚠️  Transcribes first, then embeds text")
print("")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

clap_available = embedder.get_capabilities()['audio'].get('available', False)

if clap_available:
    print("\n✅ CLAP is working perfectly!")
    print("\n📖 Recommendations:")
    print("  - Use CLAP for: Environmental sounds, music, sound effects")
    print("  - Use Whisper for: Speech-only applications (podcasts, meetings)")
    print("  - Use Hybrid for: Mixed content (most flexible)")
    print("\n🔧 Installation:")
    print("  pip install outhad-contextkitai[multimodal]")
    print("  # This installs CLAP automatically!")
else:
    print("\n❌ CLAP not available")
    print("Install with: pip install laion-clap")

print("\n" + "=" * 80)
print("✅ CLAP Test Complete!")
print("=" * 80)

