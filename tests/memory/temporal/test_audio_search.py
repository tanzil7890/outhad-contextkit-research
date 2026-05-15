"""
Test audio cross-modal search with environmental sounds.

Tests queries like:
- clapping
- birds chirping
- thunder storm
- rain
- dog barking

Uses the new flexible embedder system.
NOTE: For environmental sounds, use create_clap_embedder() for better results!
      Whisper (default) is better for speech content.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from outhad_contextkit.memory.temporal import (
    create_default_embedder,  # New flexible embedder system (uses Whisper)
    # create_clap_embedder,   # For better environmental sound results, use CLAP!
    cross_modal_search
)


def load_audio_files(audio_dir="test_dataset/audio"):
    """Load all .wav audio files."""
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    audio_data = []
    for audio_file in sorted(audio_files):
        audio_path = os.path.join(audio_dir, audio_file)
        
        # Read audio bytes
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        
        # Parse ESC-50 filename format: audio_1-100032-A-0.wav
        # The number after "A-" is the class ID
        parts = audio_file.split('-')
        if len(parts) >= 4 and parts[2] == 'A':
            class_id = int(parts[3].replace('.wav', ''))
        else:
            class_id = -1
        
        # ESC-50 class mappings (subset we care about)
        esc50_classes = {
            0: "dog barking",
            14: "birds chirping",
            19: "thunderstorm",
            22: "rain",
            36: "vacuum cleaner",
            37: "clock alarm",
            38: "clock tick",
        }
        
        expected_sound = esc50_classes.get(class_id, "unknown")
        
        audio_data.append({
            "filename": audio_file,
            "path": audio_path,
            "bytes": audio_bytes,
            "class_id": class_id,
            "expected_sound": expected_sound,
            "size": len(audio_bytes)
        })
    
    return audio_data


def main():
    print("=" * 80)
    print("Testing Audio Cross-Modal Search (Phase 3)")
    print("=" * 80)
    
    # Load audio files
    audio_data = load_audio_files()
    
    if not audio_data:
        print("\n❌ No audio files found in test_dataset/audio/")
        print("   Please ensure .wav files exist")
        return
    
    print(f"\n📁 Found {len(audio_data)} audio files:")
    for audio in audio_data:
        print(f"  - {audio['filename']}")
        print(f"    Expected: {audio['expected_sound']}")
        print(f"    Size: {audio['size']:,} bytes")
    
    # ========================================================================
    # Initialize Embedder (New Flexible System)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Initializing Embedder (Flexible System)")
    print("=" * 80)
    print("Using: Whisper (speech-focused)")
    print("Note: For environmental sounds, use create_clap_embedder() instead!")
    
    embedder = create_default_embedder()  # Uses Whisper for audio
    caps = embedder.get_capabilities()
    
    print(f"\n✅ Capabilities:")
    print(f"  ├─ CLIP (images): {caps['image']}")
    print(f"  ├─ Whisper (audio): {caps['audio']}")
    print(f"  └─ Text embeddings: {caps['text']}")
    
    if not caps['audio']:
        print("\n⚠️  WARNING: Whisper not available!")
        print("   Set OPENAI_API_KEY to test audio transcription")
        print("   Continuing with limited testing...")
    
    # ========================================================================
    # Generate Audio Embeddings
    # ========================================================================
    print("\n" + "=" * 80)
    print("Generating Audio Embeddings")
    print("=" * 80)
    
    print(f"\n🎵 Processing {len(audio_data)} audio files...")
    print("   Note: Whisper transcribes speech. Environmental sounds may")
    print("   produce empty/unexpected transcriptions.\n")
    
    for audio in audio_data:
        try:
            print(f"  Processing: {audio['filename']}")
            print(f"    Expected: {audio['expected_sound']}")
            
            # Generate embedding
            embedding = embedder.embed_audio(audio['bytes'])
            audio['embedding'] = embedding
            
            print(f"    Embedding: {len(embedding)} dimensions ✅")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            audio['embedding'] = [0.0] * 768  # Fallback
    
    # ========================================================================
    # Test Environmental Sound Queries
    # ========================================================================
    print("\n" + "=" * 80)
    print("Testing Environmental Sound Queries")
    print("=" * 80)
    
    # Test queries for environmental sounds
    test_queries = [
        "clapping",
        "applause",
        "birds chirping",
        "bird sounds",
        "thunder storm",
        "thunderstorm",
        "rain",
        "raining",
        "dog barking",
        "dog",
        "vacuum cleaner",
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"🔍 Query: \"{query}\"")
        print(f"{'=' * 80}")
        
        # Generate query embedding using CLIP text encoder
        query_embedding = embedder.embed_text(query)
        
        # Search for matching audio
        candidates = [
            {
                "id": audio["filename"],
                "content": audio["filename"],
                "expected": audio["expected_sound"],
                "embedding": audio["embedding"],
                "modality": "audio",
                "metadata": {
                    "path": audio["path"],
                    "expected_sound": audio["expected_sound"]
                }
            }
            for audio in audio_data
        ]
        
        results = cross_modal_search(
            query_embedding=query_embedding,
            candidate_embeddings=candidates,
            modality_filter="audio",
            top_k=3
        )
        
        if results:
            print(f"\n  📊 Top 3 matches:")
            for i, result in enumerate(results, 1):
                similarity = result['similarity']
                expected = result['metadata']['expected_sound']
                
                # Color code
                if similarity > 0.3:
                    marker = "🟢 HIGH"
                elif similarity > 0.2:
                    marker = "🟡 MEDIUM"
                else:
                    marker = "🔴 LOW"
                
                print(f"    {i}. {result['content']}")
                print(f"       Expected: {expected}")
                print(f"       Similarity: {similarity:.4f} {marker}")
            
            # Show best match
            top_match = results[0]
            print(f"\n  ⭐ Best match: {top_match['content']}")
            print(f"     Expected sound: {top_match['metadata']['expected_sound']}")
            print(f"     Similarity: {top_match['similarity']:.4f}")
            
            # Check if it matches what we expect
            query_lower = query.lower()
            expected_lower = top_match['metadata']['expected_sound'].lower()
            
            # Simple keyword matching for validation
            is_match = False
            if "bird" in query_lower and "bird" in expected_lower:
                is_match = True
            elif "thunder" in query_lower and "thunder" in expected_lower:
                is_match = True
            elif "rain" in query_lower and "rain" in expected_lower:
                is_match = True
            elif "dog" in query_lower and "dog" in expected_lower:
                is_match = True
            elif "clap" in query_lower and "clap" in expected_lower:
                is_match = True
            elif "vacuum" in query_lower and "vacuum" in expected_lower:
                is_match = True
            
            if is_match:
                print(f"     ✅ CORRECT MATCH!")
            else:
                print(f"     ⚠️  May not be the expected sound")
        else:
            print("  ❌ No results returned")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Tests Completed:")
    print(f"  ├─ Audio files tested: {len(audio_data)}")
    print(f"  ├─ Queries tested: {len(test_queries)}")
    print(f"  ├─ Whisper available: {caps['audio']}")
    print(f"  └─ CLIP text encoder: {caps['text']}")
    
    print(f"\n💡 Important Notes:")
    print(f"  1. Whisper is designed for SPEECH transcription")
    print(f"  2. Environmental sounds (clapping, thunder, birds) may not")
    print(f"     transcribe well or at all")
    print(f"  3. For production environmental sound search, consider:")
    print(f"     - Audio classification models (VGGish, PANN)")
    print(f"     - Audio-specific embeddings (AudioCLIP, CLAP)")
    print(f"     - Pre-computed sound labels + text search")
    
    print(f"\n📁 Audio files:")
    for audio in audio_data:
        print(f"  - {audio['filename']} ({audio['expected_sound']})")
    
    print(f"\n🔊 Listen to audio:")
    print(f"  open test_dataset/audio/")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

