"""
Test CLIP cross-modal search with dog-related queries.

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
    print("Testing Cross-Modal Search: Dog Running in Field")
    print("=" * 80)
    
    images_dir = "test_dataset/Images"
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    print(f"\n📁 Found {len(image_files)} images")
    
    # ========================================================================
    # Initialize Embedder (New Flexible System)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Initializing CLIP Embedder (Flexible System)")
    print("=" * 80)
    
    embedder = create_default_embedder()  # Uses CLIP for text/images
    caps = embedder.get_capabilities()
    
    print(f"\n✅ CLIP available: {caps['image']}")
    
    # ========================================================================
    # Generate Image Embeddings
    # ========================================================================
    print("\n" + "=" * 80)
    print("Generating Image Embeddings")
    print("=" * 80)
    
    print(f"\n📸 Embedding {len(image_files)} images...")
    
    image_data = []
    for img_file in sorted(image_files):
        img_path = os.path.join(images_dir, img_file)
        img = Image.open(img_path)
        
        embedding = embedder.embed_image(img)
        
        image_data.append({
            "filename": img_file,
            "embedding": embedding,
            "path": img_path,
            "size": img.size
        })
        
        print(f"  ✅ {img_file}")
    
    # ========================================================================
    # Test Dog-Related Queries
    # ========================================================================
    print("\n" + "=" * 80)
    print("Testing Dog-Related Queries")
    print("=" * 80)
    
    # Specific dog queries
    test_queries = [
        "dog running in the field",
        "dog running",
        "dog in field",
        "dog outdoors",
        "running dog",
        "animal running",
        "pet in grass",
        "dog playing outside",
        "dog in snow",
        "two dogs playing",
        "kayak"
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"🔍 Query: \"{query}\"")
        print(f"{'=' * 80}")
        
        # Generate query embedding using CLIP text encoder
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
            top_k=len(image_data)
        )
        
        if results:
            print(f"\n  📊 Top 3 matches:")
            for i, result in enumerate(results[:3], 1):
                similarity = result['similarity']
                
                # Color code
                if similarity > 0.3:
                    marker = "🟢 HIGH"
                elif similarity > 0.2:
                    marker = "🟡 MEDIUM"
                else:
                    marker = "🔴 LOW"
                
                print(f"    {i}. {result['content']}")
                print(f"       Similarity: {similarity:.4f} {marker}")
            
            # Show best match
            top_match = results[0]
            print(f"\n  ⭐ Best match: {top_match['content']}")
            print(f"     Similarity: {top_match['similarity']:.4f}")
            
            if top_match['similarity'] > 0.25:
                print(f"     💡 This image likely contains: dog-related content")
                print(f"     📂 View: open {top_match['metadata']['path']}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Search completed")
    print(f"  ├─ Images tested: {len(image_data)}")
    print(f"  ├─ Queries tested: {len(test_queries)}")
    print(f"  └─ CLIP working: {caps['image']}")
    
    print(f"\n💡 Note: Similarity scores depend on actual image content")
    print(f"   - If no dog images exist, all scores will be low")
    print(f"   - High scores (>0.25) suggest dog-related content")
    
    print(f"\n📸 To view all images:")
    print(f"  open {images_dir}/")
    
    print()


if __name__ == "__main__":
    main()

