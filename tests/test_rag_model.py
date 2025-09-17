"""
Tests for RAG model
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from core.rag_model import RAGModel

class TestRAGModel:
    """Test cases for RAGModel"""
    
    @pytest.fixture
    def rag_config(self):
        """Create test RAG config"""
        return {
            "search_top_k": 5,
            "score_threshold": 0.7,
            "max_query_length": 200
        }
    
    @pytest.fixture
    async def rag_model(self, rag_config):
        """Create test RAG model instance"""
        # Mock the dependencies
        vectorstore = Mock()
        vectorstore.search = AsyncMock(return_value=[])
        
        embedder = Mock()
        embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
        
        rag_model = RAGModel(rag_config, vectorstore, embedder)
        await rag_model.initialize()
        yield rag_model
        await rag_model.cleanup()
    
    @pytest.mark.asyncio
    async def test_rag_initialization(self, rag_config):
        """Test RAG model initialization"""
        vectorstore = Mock()
        embedder = Mock()
        
        rag_model = RAGModel(rag_config, vectorstore, embedder)
        assert rag_model.config == rag_config
        assert rag_model.vectorstore == vectorstore
        assert rag_model.embedder == embedder
    
    @pytest.mark.asyncio
    async def test_search_documents(self, rag_model):
        """Test document search functionality"""
        # Mock vectorstore search results
        mock_results = [
            {
                "content": "iPhone 15 Pro with advanced camera system",
                "score": 0.9,
                "metadata": {"type": "product", "category": "smartphones"}
            },
            {
                "content": "MacBook Air M2 with powerful performance",
                "score": 0.8,
                "metadata": {"type": "product", "category": "laptops"}
            }
        ]
        
        rag_model.vectorstore.search = AsyncMock(return_value=mock_results)
        rag_model.embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
        
        results = await rag_model.search("best smartphone")
        
        assert len(results) == 2
        assert results[0]["content"] == "iPhone 15 Pro with advanced camera system"
        assert results[0]["score"] == 0.9
        assert results[1]["content"] == "MacBook Air M2 with powerful performance"
        assert results[1]["score"] == 0.8
    
    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, rag_model):
        """Test search with score threshold filtering"""
        # Mock vectorstore search results with low scores
        mock_results = [
            {
                "content": "iPhone 15 Pro",
                "score": 0.9,
                "metadata": {"type": "product"}
            },
            {
                "content": "Unrelated content",
                "score": 0.5,  # Below threshold
                "metadata": {"type": "other"}
            }
        ]
        
        rag_model.vectorstore.search = AsyncMock(return_value=mock_results)
        rag_model.embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
        
        results = await rag_model.search("iPhone")
        
        # Should filter out low score results
        assert len(results) == 1
        assert results[0]["content"] == "iPhone 15 Pro"
        assert results[0]["score"] == 0.9
    
    @pytest.mark.asyncio
    async def test_search_empty_query(self, rag_model):
        """Test search with empty query"""
        results = await rag_model.search("")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_query_too_long(self, rag_model):
        """Test search with query too long"""
        long_query = "a" * 300  # Longer than max_query_length
        results = await rag_model.search(long_query)
        assert results == []
    
    @pytest.mark.asyncio
    async def test_add_documents(self, rag_model):
        """Test adding documents to RAG system"""
        documents = [
            {
                "id": "doc1",
                "content": "Test document 1",
                "metadata": {"type": "test"}
            },
            {
                "id": "doc2", 
                "content": "Test document 2",
                "metadata": {"type": "test"}
            }
        ]
        
        rag_model.embedder.embed_texts = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768])
        rag_model.vectorstore.add_documents = AsyncMock()
        
        await rag_model.add_documents(documents)
        
        # Verify embedder was called
        rag_model.embedder.embed_texts.assert_called_once()
        
        # Verify vectorstore was called
        rag_model.vectorstore.add_documents.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup(self, rag_model):
        """Test RAG model cleanup"""
        await rag_model.cleanup()
        # Should not raise any exceptions
        assert True
