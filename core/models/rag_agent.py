"""
RAG Agent for document retrieval and question answering
"""
import logging
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentConfig, AgentResponse
from ..rag_model import RAGModel

logger = logging.getLogger(__name__)

class RAGAgent(BaseAgent):
    """RAG-specific agent for document retrieval and Q&A"""
    
    def __init__(self, config: AgentConfig, rag_model: RAGModel):
        super().__init__(config)
        self.rag_model = rag_model
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def initialize(self) -> None:
        """Initialize RAG agent"""
        try:
            await self.rag_model.initialize()
            self.is_initialized = True
            self.logger.info(f"RAG Agent {self.config.name} initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG Agent: {e}")
            raise
    
    async def process(self, request: Dict[str, Any]) -> AgentResponse:
        """Process RAG request"""
        try:
            query = request.get("query", "")
            user_id = request.get("user_id")
            context = request.get("context", {})
            
            if not query.strip():
                return AgentResponse(
                    content="Please provide a search query",
                    confidence=0.0,
                    agent_name=self.config.name,
                    processing_time=0.0,
                    metadata={"error": "empty_query"}
                )
            
            # Search for relevant documents
            search_results = await self.rag_model.search(query)
            
            if not search_results:
                return AgentResponse(
                    content="I couldn't find any relevant information for your query.",
                    confidence=0.3,
                    agent_name=self.config.name,
                    processing_time=0.0,
                    metadata={"search_results": []}
                )
            
            # Format response based on search results
            response_content = self._format_search_results(search_results, query)
            confidence = self._calculate_confidence(search_results)
            
            return AgentResponse(
                content=response_content,
                confidence=confidence,
                agent_name=self.config.name,
                processing_time=0.0,  # Will be set by base class
                metadata={
                    "search_results": search_results,
                    "query": query,
                    "user_id": user_id,
                    "result_count": len(search_results)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing RAG request: {e}")
            return AgentResponse(
                content="I encountered an error while searching for information.",
                confidence=0.0,
                agent_name=self.config.name,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    def _format_search_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results into readable response"""
        if not results:
            return "I couldn't find any relevant information for your query."
        
        response = f"Based on your query '{query}', I found the following information:\n\n"
        
        for i, result in enumerate(results[:5], 1):  # Limit to top 5 results
            content = result.get("content", "")
            score = result.get("score", 0.0)
            metadata = result.get("metadata", {})
            
            # Truncate content if too long
            if len(content) > 200:
                content = content[:200] + "..."
            
            response += f"{i}. {content}\n"
            
            # Add metadata if available
            if metadata.get("price"):
                response += f"   Price: {metadata['price']:,} VNĐ\n"
            if metadata.get("brand"):
                response += f"   Brand: {metadata['brand']}\n"
            
            response += f"   Relevance: {score:.2f}\n\n"
        
        return response.strip()
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence based on search results"""
        if not results:
            return 0.0
        
        # Use the highest score as base confidence
        max_score = max(result.get("score", 0.0) for result in results)
        
        # Adjust based on number of results
        result_count_factor = min(len(results) / 3, 1.0)  # More results = higher confidence
        
        # Combine factors
        confidence = (max_score * 0.7) + (result_count_factor * 0.3)
        
        return min(confidence, 1.0)
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to RAG system"""
        try:
            await self.rag_model.add_documents(documents)
            self.logger.info(f"Added {len(documents)} documents to RAG system")
            return True
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup RAG agent resources"""
        try:
            await self.rag_model.cleanup()
            self.logger.info(f"RAG Agent {self.config.name} cleaned up")
        except Exception as e:
            self.logger.error(f"Error during RAG agent cleanup: {e}")
