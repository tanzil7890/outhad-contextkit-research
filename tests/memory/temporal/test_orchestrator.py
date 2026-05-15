"""Tests for retrieval orchestrator."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from outhad_contextkit.memory.temporal.orchestrator import RetrievalOrchestrator
from outhad_contextkit.memory.temporal.types import TimeWindow


class TestRetrievalOrchestrator:
    """Test fused retrieval orchestrator."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock = Mock()
        mock.search.return_value = []
        return mock
    
    @pytest.fixture
    def mock_graph_store(self):
        """Create mock graph store."""
        mock = Mock()
        mock.search.return_value = []
        mock.graph = Mock()  # Inner graph object
        return mock
    
    @pytest.fixture
    def mock_timeline_builder(self):
        """Create mock timeline builder."""
        mock = Mock()
        mock.get_timeline.return_value = []
        return mock
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create mock embedding model."""
        mock = Mock()
        mock.embed.return_value = [0.1] * 768  # Mock embedding vector
        return mock
    
    @pytest.fixture
    def orchestrator(self, mock_vector_store, mock_graph_store, mock_timeline_builder, mock_embedding_model):
        """Create orchestrator with mocks."""
        return RetrievalOrchestrator(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            timeline_builder=mock_timeline_builder,
            embedding_model=mock_embedding_model
        )
    
    def test_orchestrator_creation(self, orchestrator):
        """Test creating orchestrator."""
        assert orchestrator is not None
        assert orchestrator.vector_store is not None
        assert orchestrator.graph_store is not None
        assert orchestrator.timeline is not None
    
    def test_fused_search_basic(self, orchestrator):
        """Test basic fused search."""
        results = orchestrator.fused_search(
            query="test query",
            user_id="user123",
            top_k=10
        )
        
        assert results is not None
        assert "vector_results" in results
        assert "graph_results" in results
        assert "timeline_results" in results
        assert "causal_chains" in results
        assert "multimodal_results" in results
        assert "fused_ranking" in results
    
    def test_fused_search_with_time_window(self, orchestrator):
        """Test fused search with time window."""
        now = datetime.utcnow()
        time_window = TimeWindow(
            start=now - timedelta(hours=1),
            end=now + timedelta(hours=1)
        )
        
        results = orchestrator.fused_search(
            query="test query",
            user_id="user123",
            time_window=time_window,
            top_k=10
        )
        
        assert results is not None
        # Timeline results should be included when time_window is provided
        assert "timeline_results" in results
    
    def test_vector_search_empty(self, orchestrator, mock_vector_store):
        """Test vector search with no results."""
        mock_vector_store.search.return_value = []
        
        results = orchestrator._vector_search(
            query="test query",
            user_id="user123",
            time_window=None,
            top_k=10
        )
        
        assert results == []
        mock_vector_store.search.assert_called_once()
    
    def test_graph_search_empty(self, orchestrator, mock_graph_store):
        """Test graph search with no results."""
        mock_graph_store.search.return_value = []
        
        results = orchestrator._graph_search(
            query="test query",
            user_id="user123",
            time_window=None
        )
        
        assert results == []
        mock_graph_store.search.assert_called_once()
    
    def test_timeline_search_empty(self, orchestrator, mock_timeline_builder):
        """Test timeline search with no results."""
        mock_timeline_builder.get_timeline.return_value = []
        
        time_window = TimeWindow(
            start=datetime.utcnow() - timedelta(hours=1),
            end=datetime.utcnow() + timedelta(hours=1)
        )
        
        results = orchestrator._timeline_search(
            user_id="user123",
            time_window=time_window
        )
        
        assert results == []
        mock_timeline_builder.get_timeline.assert_called_once()
    
    def test_result_fusion_logic(self, orchestrator):
        """Test result fusion and ranking logic."""
        all_results = {
            "vector_results": [
                {"id": "v1", "content": "Vector result 1", "score": 0.9}
            ],
            "graph_results": [
                {"id": "g1", "content": "Graph result 1"}
            ],
            "timeline_results": [
                {"id": "t1", "content": "Timeline result 1", "timestamp": datetime.utcnow().isoformat()}
            ],
            "causal_chains": [],
            "multimodal_results": []
        }
        
        fused = orchestrator._fuse_results(all_results, "test query")
        
        assert len(fused) == 3  # One from each source
        assert all("final_score" in item for item in fused)
        assert all("source" in item for item in fused)
        
        # Check sorting (highest final_score first)
        for i in range(len(fused) - 1):
            assert fused[i]["final_score"] >= fused[i + 1]["final_score"]
    
    def test_fused_search_without_causal(self, orchestrator):
        """Test fused search without causal chains."""
        results = orchestrator.fused_search(
            query="test query",
            user_id="user123",
            include_causal=False,
            top_k=10
        )
        
        assert results is not None
        # Causal chains should be empty when include_causal=False
        assert len(results["causal_chains"]) == 0
    
    def test_fused_search_without_multimodal(self, orchestrator):
        """Test fused search without multimodal."""
        results = orchestrator.fused_search(
            query="test query",
            user_id="user123",
            include_multimodal=False,
            top_k=10
        )
        
        assert results is not None
        # Multimodal results should be empty when include_multimodal=False
        assert len(results["multimodal_results"]) == 0
    
    def test_deduplication_in_fusion(self, orchestrator):
        """Test that duplicate content is removed in fusion."""
        all_results = {
            "vector_results": [
                {"id": "v1", "content": "Same content", "score": 0.9}
            ],
            "graph_results": [
                {"id": "g1", "content": "Same content"}
            ],
            "timeline_results": [
                {"id": "t1", "content": "Unique content", "timestamp": datetime.utcnow().isoformat()}
            ],
            "causal_chains": [],
            "multimodal_results": []
        }
        
        fused = orchestrator._fuse_results(all_results, "test query")
        
        # Should have only 2 results after deduplication ("Same content" and "Unique content")
        assert len(fused) == 2
        contents = [item["content"] for item in fused]
        assert "Same content" in contents
        assert "Unique content" in contents
        # "Same content" should only appear once
        assert contents.count("Same content") == 1
    
    @pytest.mark.skip(reason="Requires Neo4j and full setup")
    def test_fused_search_real_integration(self):
        """Test fused search with real components."""
        # This would require full Neo4j + OpenAI setup
        pass
    
    @pytest.mark.skip(reason="Requires Neo4j and full setup")
    def test_causal_chain_exploration_real(self):
        """Test causal chain exploration with real graph."""
        # This would require full Neo4j setup
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

