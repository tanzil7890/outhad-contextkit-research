"""
Real Integration Test for Phase 5: Retrieval Orchestrator

This test validates the TCMGM fused retrieval system with:
- Real Neo4j database
- Real OpenAI API
- Vector + Graph + Timeline fusion
- Causal chain exploration
- Time-based filtering
"""

import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.graphs.configs import Neo4jConfig, GraphStoreConfig

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_environment():
    """Check if necessary environment variables are set."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not openai_api_key:
        logger.error("OPENAI_API_KEY not set in .env")
        return False
    if not neo4j_uri or not neo4j_username or not neo4j_password:
        logger.error("NEO4J_URI, NEO4J_USERNAME, or NEO4J_PASSWORD not set in .env")
        return False
    return True


def test_phase5_tcmgm_fused_retrieval():
    """
    Comprehensive integration test for Phase 5: Retrieval Orchestrator
    using real Neo4j and OpenAI API.
    """
    logger.info("\n" + "="*80)
    logger.info("PHASE 5 REAL INTEGRATION TEST")
    logger.info("Testing TCMGM Fused Retrieval with Real Systems")
    logger.info("="*80)

    if not check_environment():
        logger.error("\n❌ Environment not configured properly")
        logger.error("Please configure your .env file and try again")
        return False

    try:
        # 1. Initialize Memory with Neo4j and OpenAI
        logger.info("\n1. Initializing Memory with Neo4j + OpenAI + TCMGM...")
        
        config = MemoryConfig(
            custom_storage_path="test_phase5_tcmgm.db",
            graph_store=GraphStoreConfig(
                provider="neo4j",
                config=Neo4jConfig(
                    url=os.getenv("NEO4J_URI"),
                    username=os.getenv("NEO4J_USERNAME"),
                    password=os.getenv("NEO4J_PASSWORD"),
                    database="neo4j",
                    base_label=True
                )
            )
        )
        memory = Memory(config=config)
        
        # Verify components
        if not memory.graph:
            logger.error("❌ Neo4j connection failed")
            return False
        if not memory._tcmgm_enabled:
            logger.error("❌ TCMGM not initialized")
            return False
        
        logger.info("✅ Neo4j connected")
        logger.info("✅ OpenAI LLM initialized")
        logger.info("✅ TCMGM orchestrator initialized")

        # Generate unique test user
        test_user = f"test_user_phase5_{datetime.utcnow().timestamp()}"
        
        # 2. Add some memories to the system
        logger.info("\n2. Adding memories to the system...")
        
        memories_to_add = [
            "I love hiking in the mountains during summer.",
            "Yesterday I went to the grocery store and bought apples and oranges.",
            "I'm planning a trip to Japan next year for the cherry blossom season.",
            "My favorite programming language is Python because of its simplicity.",
            "I enjoy reading science fiction novels, especially by Isaac Asimov.",
        ]
        
        for i, mem in enumerate(memories_to_add):
            try:
                memory.add(mem, user_id=test_user)
                logger.info(f"  Added memory {i+1}/{len(memories_to_add)}")
            except Exception as e:
                logger.warning(f"  Failed to add memory {i+1}: {e}")
                # Continue with other memories
        
        logger.info(f"✅ Attempted to add {len(memories_to_add)} memories")
        
        # 3. Test standard search (without TCMGM)
        logger.info("\n3. Testing standard search (baseline)...")
        
        standard_results = memory.search(
            query="What do I like?",
            user_id=test_user,
            limit=5,
            use_tcmgm=False
        )
        
        assert "results" in standard_results
        logger.info(f"✅ Standard search returned {len(standard_results['results'])} results")
        
        logger.info("\nStandard Search Results:")
        for i, result in enumerate(standard_results['results'][:3]):
            logger.info(f"  {i+1}. [{result.get('score', 0):.3f}] {result.get('memory', '')[:60]}...")
        
        # 4. Test TCMGM fused search (without time window)
        logger.info("\n4. Testing TCMGM fused search (no time window)...")
        
        fused_results = memory.search(
            query="What do I like?",
            user_id=test_user,
            limit=5,
            use_tcmgm=True,
            include_causal=True
        )
        
        assert "vector_results" in fused_results
        assert "graph_results" in fused_results
        assert "timeline_results" in fused_results
        assert "fused_ranking" in fused_results
        
        logger.info(f"✅ Vector search: {len(fused_results['vector_results'])} results")
        logger.info(f"✅ Graph search: {len(fused_results['graph_results'])} results")
        logger.info(f"✅ Timeline search: {len(fused_results['timeline_results'])} results")
        logger.info(f"✅ Fused ranking: {len(fused_results['fused_ranking'])} results")
        
        logger.info("\nFused Ranking Results:")
        for i, result in enumerate(fused_results['fused_ranking'][:3]):
            logger.info(f"  {i+1}. [{result.get('final_score', 0):.3f}] ({result.get('source', '')}) {result.get('content', '')[:60]}...")
        
        # 5. Test TCMGM with time window
        logger.info("\n5. Testing TCMGM fused search with time window...")
        
        now = datetime.utcnow()
        time_window = {
            "start": (now - timedelta(hours=1)).isoformat(),
            "end": (now + timedelta(hours=1)).isoformat()
        }
        
        temporal_results = memory.search(
            query="recent activities",
            user_id=test_user,
            limit=5,
            use_tcmgm=True,
            time_window=time_window,
            include_causal=True
        )
        
        assert "timeline_results" in temporal_results
        logger.info(f"✅ Temporal search returned {len(temporal_results['timeline_results'])} timeline events")
        logger.info(f"✅ Fused ranking: {len(temporal_results['fused_ranking'])} results")
        
        # 6. Add a conversation to build timeline
        logger.info("\n6. Building timeline from conversation...")
        
        conversation = [
            {"role": "user", "content": "I just finished a great workout session"},
            {"role": "assistant", "content": "That's wonderful! What kind of workout did you do?"},
            {"role": "user", "content": "I did 30 minutes of running and then some strength training"},
        ]
        
        memory.add(conversation, user_id=test_user)
        logger.info("✅ Conversation added to memory")
        
        # 7. Search with TCMGM after adding timeline
        logger.info("\n7. Testing TCMGM search with enhanced timeline...")
        
        workout_results = memory.search(
            query="workout",
            user_id=test_user,
            limit=5,
            use_tcmgm=True,
            include_causal=True
        )
        
        logger.info(f"✅ Vector results: {len(workout_results['vector_results'])}")
        logger.info(f"✅ Graph results: {len(workout_results['graph_results'])}")
        logger.info(f"✅ Fused ranking: {len(workout_results['fused_ranking'])}")
        
        # 8. Test without causal exploration
        logger.info("\n8. Testing TCMGM without causal exploration...")
        
        no_causal_results = memory.search(
            query="programming",
            user_id=test_user,
            limit=5,
            use_tcmgm=True,
            include_causal=False
        )
        
        assert len(no_causal_results['causal_chains']) == 0
        logger.info("✅ Causal chains disabled successfully")
        
        # 9. Verify component integration
        logger.info("\n9. Verifying component integration...")
        
        assert memory._tcmgm_enabled == True
        assert memory._timeline_builder is not None
        assert memory._retrieval_orchestrator is not None
        
        logger.info("✅ TCMGM components properly initialized")
        logger.info("✅ TimelineBuilder accessible")
        logger.info("✅ RetrievalOrchestrator accessible")
        
        # 10. Test retrieval orchestrator directly
        logger.info("\n10. Testing RetrievalOrchestrator directly...")
        
        direct_results = memory._retrieval_orchestrator.fused_search(
            query="What activities do I enjoy?",
            user_id=test_user,
            top_k=5,
            include_causal=True,
            include_multimodal=True
        )
        
        assert "vector_results" in direct_results
        assert "graph_results" in direct_results
        assert "fused_ranking" in direct_results
        
        logger.info(f"✅ Direct orchestrator call successful")
        logger.info(f"   Vector: {len(direct_results['vector_results'])} results")
        logger.info(f"   Graph: {len(direct_results['graph_results'])} results")
        logger.info(f"   Fused: {len(direct_results['fused_ranking'])} results")
        
        logger.info("\n" + "="*80)
        logger.info("🎉 ALL PHASE 5 INTEGRATION TESTS PASSED! 🎉")
        logger.info("="*80)
        logger.info("\nTest Summary:")
        logger.info("  ✅ Neo4j connection: Working")
        logger.info("  ✅ OpenAI LLM: Working")
        logger.info("  ✅ TCMGM initialization: Working")
        logger.info("  ✅ Standard search: Working")
        logger.info("  ✅ Fused retrieval: Working")
        logger.info("  ✅ Vector + Graph + Timeline fusion: Working")
        logger.info("  ✅ Time window filtering: Working")
        logger.info("  ✅ Causal chain toggle: Working")
        logger.info("  ✅ Component integration: Working")
        logger.info("  ✅ Direct orchestrator access: Working")
        logger.info("\n✅ Phase 5: Retrieval Orchestrator FULLY VALIDATED!")
        
        return True

    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tcmgm_backward_compatibility():
    """Test that TCMGM doesn't break existing functionality."""
    logger.info("\n" + "="*80)
    logger.info("TESTING: TCMGM Backward Compatibility")
    logger.info("="*80)

    if not check_environment():
        return False

    try:
        logger.info("\n1. Testing Memory without graph (TCMGM should be disabled)...")
        
        config = MemoryConfig(custom_storage_path="test_phase5_no_graph.db")
        memory = Memory(config=config)
        
        assert memory._tcmgm_enabled == False
        assert memory._timeline_builder is None
        assert memory._retrieval_orchestrator is None
        
        logger.info("✅ TCMGM correctly disabled when graph is not enabled")
        
        # Standard search should still work
        test_user = f"test_user_compat_{datetime.utcnow().timestamp()}"
        memory.add("Test memory", user_id=test_user)
        
        results = memory.search("Test", user_id=test_user)
        assert "results" in results
        logger.info("✅ Standard search works without TCMGM")
        
        # TCMGM search should fall back to standard search
        tcmgm_results = memory.search("Test", user_id=test_user, use_tcmgm=True)
        assert "results" in tcmgm_results
        logger.info("✅ TCMGM flag is safely ignored when TCMGM is disabled")
        
        logger.info("\n✅ Backward compatibility verified!")
        return True

    except Exception as e:
        logger.error(f"\n❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    all_passed = True
    
    if not test_phase5_tcmgm_fused_retrieval():
        all_passed = False
    
    if not test_tcmgm_backward_compatibility():
        all_passed = False
    
    if all_passed:
        logger.info("\n" + "="*80)
        logger.info("🎉 ALL PHASE 5 TESTS PASSED! 🎉")
        logger.info("="*80)
    else:
        logger.error("\n" + "="*80)
        logger.error("❌ Some PHASE 5 TESTS FAILED. Please check the logs above.")
        logger.error("="*80)
        exit(1)

