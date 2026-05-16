"""Tests for  LoCoMo Evaluation."""
import os
import pytest
import logging
from datetime import datetime
from pathlib import Path

from outhad_contextkit import Memory
from outhad_contextkit.configs.base import MemoryConfig
from outhad_contextkit.graphs.configs import Neo4jConfig, GraphStoreConfig
from evaluation.locomo_adapter import LoCoMoAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestLoCoMoAdapter:
    """Test LoCoMo adapter."""
    
    @pytest.fixture
    def memory_instance(self):
        """Create basic memory instance."""
        config = MemoryConfig(custom_storage_path="test_locomo_adapter.db")
        return Memory(config=config)
    
    @pytest.fixture
    def adapter(self, memory_instance):
        """Create LoCoMo adapter."""
        return LoCoMoAdapter(memory_instance)
    
    def test_adapter_creation(self, adapter):
        """Test creating LoCoMo adapter."""
        assert adapter is not None
        assert adapter.memory is not None
        assert hasattr(adapter, '_tcmgm_available')
    
    def test_ingest_persona(self, adapter):
        """Test persona ingestion."""
        persona_data = {
            "events": [
                {"content": "Alice graduated from MIT in 2020", "type": "education"},
                {"content": "Alice works at Tech Corp", "type": "employment"}
            ]
        }
        
        adapter.ingest_persona(persona_data, user_id="test_user")
        
        # Verify memories were added
        results = adapter.memory.search("Alice", user_id="test_user", limit=5)
        assert len(results.get("results", [])) > 0
    
    def test_ingest_conversation(self, adapter):
        """Test conversation ingestion."""
        conversation = [
            {"role": "user", "content": "I finished the project"},
            {"role": "assistant", "content": "Great job!"},
            {"role": "user", "content": "Now I need to present it"}
        ]
        
        result = adapter.ingest_conversation(
            conversation,
            user_id="test_user",
            session_start=datetime.utcnow()
        )
        
        assert isinstance(result, dict)
        assert "events" in result
        assert "causal_links" in result
    
    def test_evaluate_qa_basic(self, adapter):
        """Test basic QA evaluation."""
        # Add some test data
        adapter.memory.add("The capital of France is Paris", user_id="test_user")
        
        questions = [
            {"question": "What is the capital of France?", "answer": "Paris"}
        ]
        
        results = adapter.evaluate_qa(questions, user_id="test_user", use_tcmgm=False)
        
        assert "accuracy" in results
        assert "results" in results
        assert len(results["results"]) == 1
    
    def test_evaluate_summarization_basic(self, adapter):
        """Test basic summarization evaluation."""
        # Add some test data
        adapter.memory.add("Event 1: Started project", user_id="test_user")
        adapter.memory.add("Event 2: Completed project", user_id="test_user")
        
        summaries = [
            {
                "time_window": None,
                "reference_summary": "A project was started and completed."
            }
        ]
        
        results = adapter.evaluate_summarization(summaries, user_id="test_user", use_tcmgm=False)
        
        assert "avg_fact_score" in results
        assert "results" in results
        assert len(results["results"]) == 1
    
    def test_extract_context_from_standard(self, adapter):
        """Test context extraction from standard search results."""
        search_results = {
            "results": [
                {"memory": "First memory"},
                {"memory": "Second memory"},
                {"memory": "Third memory"}
            ]
        }
        
        context = adapter._extract_context_from_standard(search_results)
        
        assert "First memory" in context
        assert "Second memory" in context
        assert "Third memory" in context
    
    def test_extract_context_from_fused(self, adapter):
        """Test context extraction from fused TCMGM results."""
        fused_results = {
            "fused_ranking": [
                {"content": "First result"},
                {"content": {"memory": "Second result"}},
                {"content": "Third result"}
            ]
        }
        
        context = adapter._extract_context_from_fused(fused_results)
        
        assert "First result" in context
        assert "Second result" in context
        assert "Third result" in context
    
    def test_evaluate_answer(self, adapter):
        """Test answer evaluation."""
        # Correct answer
        assert adapter._evaluate_answer("The capital is Paris", "Paris") is True
        
        # Incorrect answer
        assert adapter._evaluate_answer("The capital is London", "Paris") is False
        
        # Partial match
        assert adapter._evaluate_answer("Paris is the capital of France", "Paris") is True
    
    def test_compute_fact_score(self, adapter):
        """Test FactScore computation."""
        # Identical summaries
        score1 = adapter._compute_fact_score(
            "Event A happened. Event B followed.",
            "Event A happened. Event B followed."
        )
        assert score1 > 0.5
        
        # Different summaries
        score2 = adapter._compute_fact_score(
            "Event X occurred.",
            "Event Y occurred."
        )
        assert score2 == 0.0
        
        # Partial overlap
        score3 = adapter._compute_fact_score(
            "Event A happened. Event B happened.",
            "Event A happened. Event C happened."
        )
        assert 0.0 < score3 < 1.0


@pytest.mark.skip(reason="Requires Neo4j for full TCMGM testing")
class TestLoCoMoWithTCMGM:
    """Test LoCoMo with TCMGM enabled (requires Neo4j)."""
    
    def test_tcmgm_evaluation(self):
        """Placeholder for TCMGM evaluation test."""
        pass


def test_locomo_dataset_format():
    """Test that sample dataset has correct format."""
    import json
    
    dataset_path = Path(__file__).parent.parent.parent.parent / "evaluation" / "locomo_dataset_sample.json"
    
    if not dataset_path.exists():
        pytest.skip("Sample dataset not found")
    
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    # Check required fields
    assert "personas" in dataset
    assert "conversations" in dataset
    assert "qa_questions" in dataset
    assert "summarization_tasks" in dataset
    
    # Check personas format
    for persona in dataset["personas"]:
        assert "user_id" in persona
        assert "events" in persona
    
    # Check conversations format
    for conv in dataset["conversations"]:
        assert "user_id" in conv
        assert "start_time" in conv
        assert "messages" in conv
        for msg in conv["messages"]:
            assert "role" in msg
            assert "content" in msg
    
    # Check QA questions format
    for q in dataset["qa_questions"]:
        assert "question" in q
        assert "answer" in q
    
    # Check summarization tasks format
    for task in dataset["summarization_tasks"]:
        assert "time_window" in task or "reference_summary" in task


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

