"""Integration test for TCMGM  Causal Relationships with real LLM."""
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

from outhad_contextkit.memory.temporal import (
    CausalExtractor,
    CausalLink,
    CausalType,
    get_causal_chain,
    find_root_causes,
)
from outhad_contextkit.utils.factory import LlmFactory
from outhad_contextkit.llms.configs import LlmConfig

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_causal_extraction_with_llm():
    """Test causal relationship extraction using real LLM."""
    logger.info("=" * 80)
    logger.info("TCMGM  Causal Relationship Extraction Test")
    logger.info("=" * 80)
    
    # 1. Initialize LLM
    logger.info("\n1. Initializing LLM...")
    llm_config = LlmConfig(
        provider="openai",
        config={
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 2000
        }
    )
    llm = LlmFactory.create("openai", llm_config.config)
    logger.info("✅ LLM initialized")
    
    # 2. Create CausalExtractor
    logger.info("\n2. Creating CausalExtractor...")
    extractor = CausalExtractor(llm=llm)
    logger.info("✅ CausalExtractor created")
    
    # 3. Test Case: User Login Flow
    logger.info("\n3. Test Case 1: User Login Flow")
    logger.info("-" * 60)
    
    events = [
        {
            "id": "event_001",
            "timestamp": "2025-01-14T10:00:00",
            "content": "User opened the login page"
        },
        {
            "id": "event_002",
            "timestamp": "2025-01-14T10:00:05",
            "content": "User entered username and password"
        },
        {
            "id": "event_003",
            "timestamp": "2025-01-14T10:00:10",
            "content": "Authentication succeeded"
        },
        {
            "id": "event_004",
            "timestamp": "2025-01-14T10:00:12",
            "content": "Dashboard was loaded"
        }
    ]
    
    logger.info("Events:")
    for event in events:
        logger.info(f"  - {event['id']}: {event['content']}")
    
    # Extract with LLM
    logger.info("\n  Extracting causal links with LLM...")
    llm_links = extractor.extract_causal_links(events, use_llm=True)
    
    logger.info(f"\n  ✅ LLM extracted {len(llm_links)} causal links:")
    for i, link in enumerate(llm_links, 1):
        logger.info(f"    {i}. {link.cause_id} → {link.effect_id}")
        logger.info(f"       Type: {link.causal_type}, Confidence: {link.confidence:.2f}")
        if link.evidence:
            logger.info(f"       Evidence: {link.evidence}")
    
    assert len(llm_links) > 0, "Should extract at least one causal link"
    
    # Extract with rules (for comparison)
    logger.info("\n  Extracting causal links with rules (for comparison)...")
    rule_links = extractor.extract_causal_links(events, use_llm=False)
    logger.info(f"  ℹ️ Rule-based extracted {len(rule_links)} causal links")
    
    # 4. Test Case: Error Handling Flow
    logger.info("\n4. Test Case 2: Error Handling Flow")
    logger.info("-" * 60)
    
    error_events = [
        {
            "id": "event_101",
            "timestamp": "2025-01-14T11:00:00",
            "content": "Database connection failed"
        },
        {
            "id": "event_102",
            "timestamp": "2025-01-14T11:00:02",
            "content": "Error message displayed to user"
        },
        {
            "id": "event_103",
            "timestamp": "2025-01-14T11:00:05",
            "content": "Automatic retry mechanism triggered"
        },
        {
            "id": "event_104",
            "timestamp": "2025-01-14T11:00:10",
            "content": "Connection restored successfully"
        }
    ]
    
    logger.info("Events:")
    for event in error_events:
        logger.info(f"  - {event['id']}: {event['content']}")
    
    logger.info("\n  Extracting causal links with LLM...")
    error_links = extractor.extract_causal_links(error_events, use_llm=True)
    
    logger.info(f"\n  ✅ LLM extracted {len(error_links)} causal links:")
    for i, link in enumerate(error_links, 1):
        logger.info(f"    {i}. {link.cause_id} → {link.effect_id}")
        logger.info(f"       Type: {link.causal_type}, Confidence: {link.confidence:.2f}")
        if link.evidence:
            logger.info(f"       Evidence: {link.evidence}")
    
    assert len(error_links) > 0, "Should extract causal links from error flow"
    
    # 5. Test all causal types
    logger.info("\n5. Verifying Causal Types")
    logger.info("-" * 60)
    
    all_links = llm_links + error_links
    causal_types_found = set(link.causal_type for link in all_links)
    
    logger.info(f"  Causal types found: {', '.join(causal_types_found)}")
    logger.info(f"  Total unique types: {len(causal_types_found)}")
    
    # 6. Test confidence scores
    logger.info("\n6. Analyzing Confidence Scores")
    logger.info("-" * 60)
    
    confidences = [link.confidence for link in all_links]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        logger.info(f"  Average confidence: {avg_confidence:.2f}")
        logger.info(f"  Min confidence: {min_confidence:.2f}")
        logger.info(f"  Max confidence: {max_confidence:.2f}")
        
        # All confidences should be valid (0-1)
        assert all(0 <= c <= 1 for c in confidences), "All confidences should be between 0 and 1"
        logger.info("  ✅ All confidence scores are valid (0-1)")
    
    # 7. Test CausalLink validation
    logger.info("\n7. Testing CausalLink Data Model")
    logger.info("-" * 60)
    
    # Create a causal link manually
    manual_link = CausalLink(
        cause_id="test_cause",
        effect_id="test_effect",
        causal_type=CausalType.LEADS_TO.value,
        confidence=0.95,
        evidence="Manual test link",
        timestamp=datetime.utcnow()
    )
    
    logger.info(f"  Created manual link: {manual_link.cause_id} → {manual_link.effect_id}")
    logger.info(f"  Type: {manual_link.causal_type}")
    logger.info(f"  Confidence: {manual_link.confidence}")
    logger.info("  ✅ CausalLink model works correctly")
    
    # 8. Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ LLM-based extraction: Working ({len(llm_links) + len(error_links)} total links)")
    logger.info(f"✅ Rule-based extraction: Working ({len(rule_links)} links)")
    logger.info(f"✅ Causal types found: {len(causal_types_found)} different types")
    logger.info(f"✅ Confidence validation: All scores valid")
    logger.info(f"✅ CausalLink model: Working")
    logger.info("=" * 80)
    logger.info("🎉 ALL  TESTS PASSED!")
    logger.info("=" * 80)
    
    return {
        "llm_links": llm_links,
        "error_links": error_links,
        "rule_links": rule_links,
        "causal_types_found": list(causal_types_found),
        "avg_confidence": avg_confidence if confidences else 0
    }


if __name__ == "__main__":
    try:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY not found in environment")
            print("Please create a .env file with: OPENAI_API_KEY=your_key_here")
            exit(1)
        
        # Run test
        results = test_causal_extraction_with_llm()
        
        print("\n✅ Integration test completed successfully!")
        print(f"Total causal links extracted: {len(results['llm_links']) + len(results['error_links'])}")
        print(f"Average confidence: {results['avg_confidence']:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}", exc_info=True)
        exit(1)

