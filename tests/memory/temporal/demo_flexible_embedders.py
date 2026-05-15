"""
Demo: Flexible Embedder System

Shows how easy it is to switch between different embedders
without breaking your code.
"""
import os
from dotenv import load_dotenv
from PIL import Image

# Load environment
load_dotenv()

from outhad_contextkit.memory.temporal import (
    create_default_embedder,
    create_clap_embedder,
    create_hybrid_embedder,
    create_embedder,
)


def main():
    print("=" * 80)
    print("FLEXIBLE EMBEDDER SYSTEM DEMO")
    print("=" * 80)
    
    # ========================================================================
    # Example 1: Default Setup (Speech-focused)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 1: Default Setup (Whisper for audio)")
    print("=" * 80)
    
    embedder = create_default_embedder()
    
    print("\n✅ Created default embedder")
    print("Configuration:")
    print("  - Text: CLIP")
    print("  - Image: CLIP")
    print("  - Audio: Whisper (speech-focused)")
    
    # Test capabilities
    capabilities = embedder.get_capabilities()
    print("\nCapabilities:")
    for modality, caps in capabilities.items():
        if caps:
            print(f"  {modality}: {caps.get('model')} - {caps.get('available')}")
    
    # Generate embeddings
    print("\nGenerating embeddings:")
    text_emb = embedder.embed_text("dog in snow")
    print(f"  Text: {len(text_emb)} dimensions")
    
    # ========================================================================
    # Example 2: CLAP Setup (All audio types)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 2: CLAP Setup (All audio types)")
    print("=" * 80)
    
    embedder = create_clap_embedder()
    
    print("\n✅ Created CLAP embedder")
    print("Configuration:")
    print("  - Text: CLIP")
    print("  - Image: CLIP")
    print("  - Audio: CLAP (all audio types!)")
    
    capabilities = embedder.get_capabilities()
    print("\nCapabilities:")
    for modality, caps in capabilities.items():
        if caps:
            print(f"  {modality}: {caps.get('model')} - {caps.get('available')}")
    
    print("\nThis embedder can handle:")
    print("  ✅ Speech (conversations, lectures)")
    print("  ✅ Environmental sounds (birds, thunder, rain)")
    print("  ✅ Music and sound effects")
    
    # ========================================================================
    # Example 3: Hybrid Setup (Maximum flexibility)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 3: Hybrid Setup (CLAP + Whisper fallback)")
    print("=" * 80)
    
    embedder = create_hybrid_embedder()
    
    print("\n✅ Created hybrid embedder")
    print("Configuration:")
    print("  - Text: CLIP")
    print("  - Image: CLIP")
    print("  - Audio: CLAP (with Whisper fallback)")
    
    print("\nStrategy:")
    print("  1. Try CLAP first (handles everything)")
    print("  2. Fallback to Whisper if CLAP unavailable")
    print("  3. Most robust option for production")
    
    # ========================================================================
    # Example 4: Custom Configuration
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 4: Custom Configuration")
    print("=" * 80)
    
    print("\nYou can mix and match any combination:")
    
    # Option 1: OpenAI text embeddings + CLIP images + Whisper audio
    print("\n  Option A: OpenAI text + CLIP images + Whisper audio")
    embedder_a = create_embedder(
        text_embedder="openai",  # Different text embedder!
        image_embedder="clip",
        audio_embedder="whisper"
    )
    print("    ✅ Created")
    
    # Option 2: CLIP text + CLIP images + CLAP audio
    print("\n  Option B: CLIP text + CLIP images + CLAP audio")
    embedder_b = create_embedder(
        text_embedder="clip",
        image_embedder="clip",
        audio_embedder="clap"  # Different audio embedder!
    )
    print("    ✅ Created")
    
    # ========================================================================
    # Example 5: Switching Embedders (No Code Changes!)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 5: Switching Embedders")
    print("=" * 80)
    
    print("\nSame code, different configuration:")
    
    # Function that uses embedder
    def process_audio(embedder, audio_type):
        print(f"\n  Processing {audio_type} audio...")
        # Your actual processing code here - doesn't change!
        # embedding = embedder.embed_audio(audio_bytes)
        # ...
        caps = embedder.get_capabilities()
        audio_caps = caps.get('audio', {})
        
        # Handle both regular and hybrid embedders
        if 'model' in audio_caps:
            print(f"    Using: {audio_caps['model']}")
        elif 'primary' in audio_caps:
            print(f"    Using: Hybrid ({audio_caps['primary']['model']} + fallback)")
        else:
            print(f"    Using: {audio_caps}")
    
    # Switch embedders without changing process_audio()!
    print("\nUsing Whisper:")
    embedder_whisper = create_embedder(audio_embedder="whisper")
    process_audio(embedder_whisper, "speech")
    
    print("\nUsing CLAP:")
    embedder_clap = create_embedder(audio_embedder="clap")
    process_audio(embedder_clap, "environmental sound")
    
    print("\nUsing Hybrid:")
    embedder_hybrid = create_embedder(audio_embedder="hybrid")
    process_audio(embedder_hybrid, "mixed content")
    
    print("\n✅ Same function, different embedders!")
    print("   No code changes needed!")
    
    # ========================================================================
    # Example 6: Testing with Real Images
    # ========================================================================
    print("\n" + "=" * 80)
    print("Example 6: Testing with Real Data")
    print("=" * 80)
    
    # Check if test images exist
    images_dir = "test_dataset/Images"
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')][:1]
        
        if image_files:
            print(f"\n✅ Found test images in: {images_dir}")
            
            # Test with image
            img_path = os.path.join(images_dir, image_files[0])
            img = Image.open(img_path)
            
            print(f"\nProcessing: {image_files[0]}")
            
            # Use any embedder - they all work the same!
            embedder = create_default_embedder()
            
            image_emb = embedder.embed_image(img)
            text_emb = embedder.embed_text("dog in snow")
            
            print(f"  Image embedding: {len(image_emb)} dimensions")
            print(f"  Text embedding: {len(text_emb)} dimensions")
            print(f"  ✅ Cross-modal search ready!")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n✅ Key Benefits:")
    print("  1. Easy to switch embedders (just change parameter)")
    print("  2. No code changes needed")
    print("  3. Backward compatible (old code still works)")
    print("  4. Can mix and match models")
    print("  5. Extensible (add your own embedders)")
    
    print("\n📖 Quick Reference:")
    print("  - Speech only:       create_default_embedder()")
    print("  - All audio:         create_clap_embedder()")
    print("  - Max flexibility:   create_hybrid_embedder()")
    print("  - Custom:            create_embedder(...)")
    
    print("\n🎯 When to Use What:")
    print("  - Podcasts/Interviews:           Whisper (default)")
    print("  - Music/Sound effects:           CLAP")
    print("  - Environmental sounds:          CLAP")
    print("  - Mixed content:                 Hybrid")
    print("  - Production (safety):           Hybrid")
    
    print("\n" + "=" * 80)
    print("✅ Demo Complete!")
    print("=" * 80)
    print("\nSee FLEXIBLE_EMBEDDER_GUIDE.md for detailed documentation")


if __name__ == "__main__":
    main()


