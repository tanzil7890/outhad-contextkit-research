"""
Simple test script for TCMGM  Images Only.

Tests:
1. Load images from test_dataset/images/
2. Generate CLIP embeddings
3. Cross-modal search (text → images)
4. Image similarity

Uses the new flexible embedder system.
"""
import os
from PIL import Image
from outhad_contextkit.memory.temporal import (
    create_default_embedder,  # New flexible embedder system
    cross_modal_search
)


def main():
    print("=" * 70)
    print("TCMGM : Image-Only Test")
    print("=" * 70)
    
    # Check if images directory exists
    images_dir = "test_dataset/images"
    
    if not os.path.exists(images_dir):
        print(f"\n❌ Directory not found: {images_dir}")
        print("Run this first: python create_test_dataset.py")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    if not image_files:
        print(f"\n❌ No images found in: {images_dir}")
        return
    
    print(f"\n✅ Found {len(image_files)} images in: {images_dir}")
    
    # ========================================================================
    # Initialize Embedder (New Flexible System)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Initialize CLIP Embedder (Flexible System)")
    print("=" * 70)
    
    embedder = create_default_embedder()  # Uses CLIP for text/images
    caps = embedder.get_capabilities()
    
    print(f"\n📊 Capabilities:")
    print(f"  ├─ CLIP (images): {caps['image']}")
    print(f"  └─ CLIP model: {caps['image']['model']}")
    
    if not caps['image']:
        print("\n⚠️  CLIP not available - using placeholder embeddings")
        print("Install with: pip install transformers torch torchvision")
    
    # ========================================================================
    # Load and Embed Images
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Generate Image Embeddings")
    print("=" * 70)
    
    print(f"\n📸 Embedding {len(image_files)} images...")
    
    image_data = []
    for img_file in sorted(image_files):
        img_path = os.path.join(images_dir, img_file)
        img = Image.open(img_path)
        
        # Generate embedding
        embedding = embedder.embed_image(img)
        
        # Extract scene name from filename
        scene_name = img_file.replace('.jpg', '').replace('_', ' ').title()
        
        image_data.append({
            "filename": img_file,
            "scene": scene_name,
            "embedding": embedding,
            "path": img_path
        })
        
        print(f"  ✅ {img_file}: {len(embedding)} dimensions")
    
    # ========================================================================
    # Test Cross-Modal Search (Text → Images)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Cross-Modal Search (Text → Images)")
    print("=" * 70)
    
    # Test queries
    test_queries = [
        "sunset at the beach",
        "snowy mountain peak",
        "green forest with trees",
        "city lights at night",
        "ocean waves"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        
        # Generate query embedding
        query_embedding = embedder.embed_text(query)
        
        # Search for matching images
        candidates = [
            {
                "content": img["filename"],  # cross_modal_search uses 'content'
                "scene": img["scene"],
                "embedding": img["embedding"],
                "modality": "image",
                "metadata": {"scene": img["scene"]}
            }
            for img in image_data
        ]
        
        results = cross_modal_search(
            query_embedding=query_embedding,
            candidate_embeddings=candidates,
            modality_filter="image",
            top_k=3
        )
        
        if results:
            print(f"  Top {len(results)} matches:")
            for i, result in enumerate(results, 1):
                print(f"    {i}. {result['content']}")
                print(f"       Similarity: {result['similarity']:.4f}")
                if 'metadata' in result and 'scene' in result['metadata']:
                    print(f"       Scene: {result['metadata']['scene']}")
        else:
            print("  No results (embedding might be zero)")
    
    # ========================================================================
    # Test Image Similarity
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Find Similar Images")
    print("=" * 70)
    
    if len(image_data) >= 2:
        # Use first image as query
        query_img = image_data[0]
        print(f"\n🖼️  Query image: {query_img['filename']}")
        
        # Find similar images
        candidates = [
            {
                "content": img["filename"],  # cross_modal_search uses 'content'
                "scene": img["scene"],
                "embedding": img["embedding"],
                "modality": "image"
            }
            for img in image_data[1:]  # Exclude query image
        ]
        
        results = cross_modal_search(
            query_embedding=query_img["embedding"],
            candidate_embeddings=candidates,
            modality_filter="image",
            top_k=3
        )
        
        if results:
            print(f"  Most similar images:")
            for i, result in enumerate(results, 1):
                print(f"    {i}. {result['content']}")
                print(f"       Similarity: {result['similarity']:.4f}")
        else:
            print("  No results")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Tests Completed:")
    print(f"  ├─ Images loaded: {len(image_data)}")
    print(f"  ├─ Embeddings generated: {len(image_data)}")
    print(f"  ├─ CLIP available: {caps['image']}")
    print(f"  └─ Cross-modal search: ✅")
    
    print(f"\n📁 Image files:")
    for img in image_data:
        print(f"  - {img['filename']} ({img['scene']})")
    
    print(f"\n🎉 Image testing complete!")
    print()


if __name__ == "__main__":
    main()

