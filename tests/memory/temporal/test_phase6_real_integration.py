"""Real integration tests for Phase 6: LoCoMo Evaluation."""
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.graphs.configs import Neo4jConfig, GraphStoreConfig
from evaluation.locomo_adapter import LoCoMoAdapter

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
        logger.warning("Neo4j not configured. TCMGM features will not be available.")
        return "partial"
    return True


def test_phase6_locomo_baseline():
    """Test LoCoMo evaluation with baseline (no TCMGM)."""
    logger.info("\n" + "="*80)
    logger.info("PHASE 6 REAL INTEGRATION TEST: Baseline Evaluation")
    logger.info("="*80)

    if not check_environment():
        logger.error("\n❌ Environment not configured properly")
        return False

    test_user = f"test_user_locomo_{datetime.utcnow().timestamp()}"
    
    try:
        # 1. Initialize Memory (without graph)
        logger.info("\n1. Initializing Memory (Baseline - No TCMGM)...")
        config = MemoryConfig(custom_storage_path="test_phase6_baseline.db")
        memory = Memory(config=config)
        
        assert memory.graph is None
        logger.info("✅ Memory initialized (baseline mode)")

        # 2. Create LoCoMo adapter
        logger.info("\n2. Creating LoCoMo adapter...")
        adapter = LoCoMoAdapter(memory)
        
        assert adapter.memory is not None
        assert adapter._tcmgm_available is False
        logger.info("✅ LoCoMo adapter created (TCMGM not available)")

        # 3. Ingest persona
        logger.info("\n3. Ingesting persona...")
        persona_data = {
            "events": [
                {"content": "Alice graduated from MIT in 2020", "type": "education"},
                {"content": "Alice started working at Tech Corp in January 2021", "type": "employment"},
                {"content": "Alice enjoys hiking and photography", "type": "hobby"}
            ]
        }
        
        adapter.ingest_persona(persona_data, user_id=test_user)
        logger.info("✅ Persona ingested")

        # 4. Ingest conversation
        logger.info("\n4. Ingesting conversation...")
        conversation = [
            {"role": "user", "content": "I just finished the project proposal"},
            {"role": "assistant", "content": "Great! What's next?"},
            {"role": "user", "content": "I need to present it to the team tomorrow"}
        ]
        
        result = adapter.ingest_conversation(conversation, user_id=test_user, session_start=datetime.utcnow())
        
        assert "events" in result
        assert "causal_links" in result
        logger.info("✅ Conversation ingested")

        # 5. Test QA evaluation
        logger.info("\n5. Testing QA evaluation...")
        questions = [
            {"question": "Where did Alice work after graduating?", "answer": "Tech Corp"},
            {"question": "What did Alice study?", "answer": "MIT"},
            {"question": "What does Alice enjoy doing?", "answer": "hiking"}
        ]
        
        qa_results = adapter.evaluate_qa(questions, user_id=test_user, use_tcmgm=False)
        
        assert "accuracy" in qa_results
        assert "total_questions" in qa_results
        assert qa_results["total_questions"] == 3
        logger.info(f"✅ QA evaluated: {qa_results['accuracy']:.2%} accuracy ({qa_results['correct_answers']}/{qa_results['total_questions']})")

        # 6. Test summarization evaluation
        logger.info("\n6. Testing summarization evaluation...")
        summaries = [
            {
                "time_window": None,
                "reference_summary": "Alice graduated from MIT and works at Tech Corp. She enjoys hiking and photography."
            }
        ]
        
        summ_results = adapter.evaluate_summarization(summaries, user_id=test_user, use_tcmgm=False)
        
        assert "avg_fact_score" in summ_results
        assert "total_tasks" in summ_results
        assert summ_results["total_tasks"] == 1
        logger.info(f"✅ Summarization evaluated: {summ_results['avg_fact_score']:.3f} FactScore")

        # 7. Verify helper methods
        logger.info("\n7. Verifying helper methods...")
        
        # Test answer evaluation
        assert adapter._evaluate_answer("The answer is Paris", "Paris") is True
        assert adapter._evaluate_answer("London", "Paris") is False
        logger.info("✅ Answer evaluation working")
        
        # Test fact score
        score = adapter._compute_fact_score("Event A. Event B.", "Event A. Event C.")
        assert 0.0 <= score <= 1.0
        logger.info(f"✅ FactScore computation working (score: {score:.3f})")

        logger.info("\n" + "="*80)
        logger.info("🎉 PHASE 6 BASELINE EVALUATION PASSED! 🎉")
        logger.info("="*80)
        logger.info("\nTest Summary:")
        logger.info("  ✅ Memory initialization: Working")
        logger.info("  ✅ LoCoMo adapter: Working")
        logger.info("  ✅ Persona ingestion: Working")
        logger.info("  ✅ Conversation ingestion: Working")
        logger.info(f"  ✅ QA evaluation: {qa_results['accuracy']:.2%} accuracy")
        logger.info(f"  ✅ Summarization: {summ_results['avg_fact_score']:.3f} FactScore")
        logger.info("  ✅ Helper methods: Working")
        logger.info("\n✅ Phase 6 Baseline: FULLY VALIDATED!")
        return True

    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase6_locomo_tcmgm():
    """Test LoCoMo evaluation with TCMGM."""
    logger.info("\n" + "="*80)
    logger.info("PHASE 6 REAL INTEGRATION TEST: TCMGM Evaluation")
    logger.info("="*80)

    env_status = check_environment()
    if env_status is False:
        logger.error("\n❌ Environment not configured properly")
        return False
    elif env_status == "partial":
        logger.warning("\n⚠️  Neo4j not configured. Skipping TCMGM test.")
        return True  # Skip but don't fail

    test_user = f"test_user_locomo_tcmgm_{datetime.utcnow().timestamp()}"
    
    try:
        # 1. Initialize Memory with Neo4j
        logger.info("\n1. Initializing Memory with Neo4j + TCMGM...")
        config = MemoryConfig(
            custom_storage_path="test_phase6_tcmgm.db",
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
        
        assert memory.graph is not None
        assert memory._tcmgm_enabled is True
        logger.info("✅ Memory initialized with TCMGM")

        # 2. Create LoCoMo adapter
        logger.info("\n2. Creating LoCoMo adapter...")
        adapter = LoCoMoAdapter(memory)
        
        assert adapter._tcmgm_available is True
        logger.info("✅ LoCoMo adapter created (TCMGM available)")

        # 3. Ingest persona
        logger.info("\n3. Ingesting persona...")
        persona_data = {
            "events": [
                {"content": "Bob graduated from Stanford in 2019", "type": "education"},
                {"content": "Bob works as a Data Scientist at AI Labs", "type": "employment"}
            ]
        }
        
        adapter.ingest_persona(persona_data, user_id=test_user)
        logger.info("✅ Persona ingested")

        # 4. Ingest conversation with timeline
        logger.info("\n4. Ingesting conversation (with timeline building)...")
        conversation = [
            {"role": "user", "content": "I completed the data analysis project"},
            {"role": "assistant", "content": "Excellent work! What did you find?"},
            {"role": "user", "content": "The model accuracy improved by 15 percent"}
        ]
        
        result = adapter.ingest_conversation(conversation, user_id=test_user, session_start=datetime.utcnow())
        
        assert "events" in result
        assert "causal_links" in result
        logger.info(f"✅ Conversation ingested (timeline: {len(result['events'])} events)")

        # 5. Test QA evaluation with TCMGM
        logger.info("\n5. Testing QA evaluation with TCMGM...")
        questions = [
            {"question": "Where did Bob graduate from?", "answer": "Stanford"},
            {"question": "What is Bob's job title?", "answer": "Data Scientist"},
        ]
        
        qa_results = adapter.evaluate_qa(questions, user_id=test_user, use_tcmgm=True)
        
        assert "accuracy" in qa_results
        logger.info(f"✅ QA with TCMGM: {qa_results['accuracy']:.2%} accuracy ({qa_results['correct_answers']}/{qa_results['total_questions']})")

        # 6. Test summarization with TCMGM
        logger.info("\n6. Testing summarization with TCMGM...")
        from datetime import timedelta
        now = datetime.utcnow()
        summaries = [
            {
                "time_window": {
                    "start": (now - timedelta(hours=1)).isoformat(),
                    "end": now.isoformat()
                },
                "reference_summary": "Bob completed a data analysis project with 15% model accuracy improvement."
            }
        ]
        
        summ_results = adapter.evaluate_summarization(summaries, user_id=test_user, use_tcmgm=True)
        
        assert "avg_fact_score" in summ_results
        logger.info(f"✅ Summarization with TCMGM: {summ_results['avg_fact_score']:.3f} FactScore")

        logger.info("\n" + "="*80)
        logger.info("🎉 PHASE 6 TCMGM EVALUATION PASSED! 🎉")
        logger.info("="*80)
        logger.info("\nTest Summary:")
        logger.info("  ✅ Neo4j connection: Working")
        logger.info("  ✅ TCMGM initialization: Working")
        logger.info("  ✅ LoCoMo adapter with TCMGM: Working")
        logger.info("  ✅ Timeline building from conversation: Working")
        logger.info(f"  ✅ QA with TCMGM: {qa_results['accuracy']:.2%} accuracy")
        logger.info(f"  ✅ Summarization with TCMGM: {summ_results['avg_fact_score']:.3f} FactScore")
        logger.info("\n✅ Phase 6 TCMGM: FULLY VALIDATED!")
        return True

    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    all_passed = True
    
    # Test baseline
    if not test_phase6_locomo_baseline():
        all_passed = False
    
    # Test TCMGM
    if not test_phase6_locomo_tcmgm():
        all_passed = False
    
    if all_passed:
        logger.info("\n" + "="*80)
        logger.info("🎉 ALL PHASE 6 TESTS PASSED! 🎉")
        logger.info("="*80)
    else:
        logger.error("\n" + "="*80)
        logger.error("❌ Some PHASE 6 TESTS FAILED")
        logger.error("="*80)
        exit(1)

