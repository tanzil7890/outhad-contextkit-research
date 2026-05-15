"""
Test CLIP cross-modal search with kayak and pool queries.

This tests if the system can find relevant images when searching for:
- kayak
- pool
- water sports
- swimming

Uses the new flexible embedder system.
"""
import os
from PIL import Image
from outhad_contextkit.memory.temporal import (
    create_default_embedder,  # New flexible embedder system
    cross_modal_search
)


def main():
    print("=" * 80)
    print("Testing Cross-Modal Search: Kayak & Pool Queries")
    print("=" * 80)
    
    # Check directory (capital I)
    images_dir = "test_dataset/Images"
    
    if not os.path.exists(images_dir):
        # Try lowercase
        images_dir = "test_dataset/images"
        if not os.path.exists(images_dir):
            print(f"\n❌ Directory not found: {images_dir}")
            return
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    print(f"\n📁 Found {len(image_files)} images in: {images_dir}")
    print("Images:")
    for img_file in sorted(image_files):
        img_path = os.path.join(images_dir, img_file)
        img = Image.open(img_path)
        print(f"  - {img_file} ({img.size[0]}x{img.size[1]})")
    
    # ========================================================================
    # Initialize Embedder (New Flexible System)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Initializing CLIP Embedder (Flexible System)")
    print("=" * 80)
    
    embedder = create_default_embedder()  # Uses CLIP for text/images
    caps = embedder.get_capabilities()
    
    print(f"\n✅ CLIP available: {caps['image']}")
    print(f"✅ Model: {caps['image']['model']}")
    
    if not caps['image']:
        print("\n⚠️  CLIP not available - results may not be meaningful")
    
    # ========================================================================
    # Generate Image Embeddings
    # ========================================================================
    print("\n" + "=" * 80)
    print("Generating Image Embeddings")
    print("=" * 80)
    
    print(f"\n📸 Embedding {len(image_files)} images with CLIP...")
    
    image_data = []
    for img_file in sorted(image_files):
        img_path = os.path.join(images_dir, img_file)
        img = Image.open(img_path)
        
        # Generate embedding
        embedding = embedder.embed_image(img)
        
        image_data.append({
            "filename": img_file,
            "embedding": embedding,
            "path": img_path,
            "size": img.size
        })
        
        print(f"  ✅ {img_file}: {len(embedding)} dimensions")
    
    # ========================================================================
    # Test Kayak and Pool Queries
    # ========================================================================
    print("\n" + "=" * 80)
    print("Testing Water-Related Queries")
    print("=" * 80)
    
    # Specific queries for kayak and pool
    test_queries = [
        "kayak",
        "kayaking",
        "person in a kayak on water",
        "pool",
        "swimming pool",
        "people swimming in a pool",
        "water sports",
        "boat on water",
        "people in water",
        "outdoor water activities"
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"🔍 Query: \"{query}\"")
        print(f"{'=' * 80}")
        
        # Generate query embedding
        query_embedding = embedder.embed_text(query)
        
        # Search for matching images
        candidates = [
            {
                "id": img["filename"],
                "content": img["filename"],
                "embedding": img["embedding"],
                "modality": "image",
                "metadata": {
                    "path": img["path"],
                    "size": img["size"]
                }
            }
            for img in image_data
        ]
        
        results = cross_modal_search(
            query_embedding=query_embedding,
            candidate_embeddings=candidates,
            modality_filter="image",
            top_k=len(image_data)  # Show all results ranked
        )
        
        if results:
            print(f"\n  📊 All images ranked by similarity:")
            for i, result in enumerate(results, 1):
                similarity = result['similarity']
                
                # Color code based on similarity
                if similarity > 0.3:
                    marker = "🟢 HIGH"
                elif similarity > 0.2:
                    marker = "🟡 MEDIUM"
                else:
                    marker = "🔴 LOW"
                
                print(f"    {i}. {result['content']}")
                print(f"       Similarity: {similarity:.4f} {marker}")
            
            # Highlight top match
            top_match = results[0]
            print(f"\n  ⭐ Best match: {top_match['content']}")
            print(f"     Similarity: {top_match['similarity']:.4f}")
            
            if top_match['similarity'] < 0.2:
                print(f"     ⚠️  Low confidence - image may not match query")
        else:
            print("  ❌ No results returned")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Tests Completed:")
    print(f"  ├─ Images tested: {len(image_data)}")
    print(f"  ├─ Queries tested: {len(test_queries)}")
    print(f"  ├─ CLIP working: {caps['image']}")
    print(f"  └─ Cross-modal search: ✅")
    
    print(f"\n💡 Interpretation:")
    print(f"  - Similarity > 0.30: Image likely matches query")
    print(f"  - Similarity 0.20-0.30: Possible match")
    print(f"  - Similarity < 0.20: Unlikely match")
    
    print(f"\n📸 To view images:")
    print(f"  open {images_dir}/")
    
    print()


if __name__ == "__main__":
    main()

