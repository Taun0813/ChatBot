"""
Tests for the Agno router
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from core.router import AgnoRouter, RouterConfig

class TestAgnoRouter:
    """Test cases for AgnoRouter"""
    
    @pytest.fixture
    def router_config(self):
        """Create test router config"""
        return RouterConfig(
            rag_config={},
            interaction_config={},
            api_config={}
        )
    
    @pytest.fixture
    async def router(self, router_config):
        """Create test router instance"""
        router = AgnoRouter(router_config)
        await router.initialize()
        yield router
        await router.cleanup()
    
    @pytest.mark.asyncio
    async def test_router_initialization(self, router_config):
        """Test router initialization"""
        router = AgnoRouter(router_config)
        assert router is not None
        assert router.config == router_config
    
    @pytest.mark.asyncio
    async def test_router_initialize(self, router):
        """Test router initialization process"""
        # Router should be initialized in fixture
        assert router.rag_model is not None
        assert router.interaction_model is not None
        assert router.api_model is not None
    
    @pytest.mark.asyncio
    async def test_process_request_search_intent(self, router):
        """Test processing search intent request"""
        # Mock the models
        router.rag_model.search = AsyncMock(return_value=[
            {"content": "Test product", "score": 0.9, "metadata": {}}
        ])
        router.interaction_model.generate = AsyncMock(return_value={
            "response": "Here are some products that match your search",
            "confidence": 0.8
        })
        
        response = await router.process_request(
            message="I'm looking for a laptop",
            user_id="test_user",
            intent="search"
        )
        
        assert response["intent"] == "search"
        assert "response" in response
        assert response["confidence"] > 0
    
    @pytest.mark.asyncio
    async def test_process_request_chat_intent(self, router):
        """Test processing chat intent request"""
        # Mock the interaction model
        router.interaction_model.generate = AsyncMock(return_value={
            "response": "Hello! How can I help you today?",
            "confidence": 0.9
        })
        
        response = await router.process_request(
            message="Hello",
            user_id="test_user",
            intent="chat"
        )
        
        assert response["intent"] == "chat"
        assert "Hello" in response["response"]
        assert response["confidence"] > 0
    
    @pytest.mark.asyncio
    async def test_process_request_api_intent(self, router):
        """Test processing API call intent request"""
        # Mock the API model
        router.api_model.process_api_request = AsyncMock(return_value={
            "response": "Order status: Shipped",
            "confidence": 0.95,
            "metadata": {"order_id": "12345"}
        })
        
        response = await router.process_request(
            message="What's the status of my order?",
            user_id="test_user",
            intent="api_call"
        )
        
        assert response["intent"] == "api_call"
        assert "Order status" in response["response"]
        assert response["confidence"] > 0
    
    @pytest.mark.asyncio
    async def test_process_request_auto_intent(self, router):
        """Test processing request with automatic intent detection"""
        # Mock intent detection
        router.detect_intent = AsyncMock(return_value="search")
        router.rag_model.search = AsyncMock(return_value=[])
        router.interaction_model.generate = AsyncMock(return_value={
            "response": "I couldn't find any products matching your search",
            "confidence": 0.7
        })
        
        response = await router.process_request(
            message="Show me some phones",
            user_id="test_user"
        )
        
        assert response["intent"] == "search"
        assert "response" in response
    
    @pytest.mark.asyncio
    async def test_cleanup(self, router):
        """Test router cleanup"""
        await router.cleanup()
        # Should not raise any exceptions
        assert True
