"""
Conversation Agent for general chat and interaction
"""
import logging
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentConfig, AgentResponse
from ..interaction_model import InteractionModel

logger = logging.getLogger(__name__)

class ConversationAgent(BaseAgent):
    """Conversation-specific agent for general chat"""
    
    def __init__(self, config: AgentConfig, interaction_model: InteractionModel):
        super().__init__(config)
        self.interaction_model = interaction_model
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def initialize(self) -> None:
        """Initialize conversation agent"""
        try:
            await self.interaction_model.initialize()
            self.is_initialized = True
            self.logger.info(f"Conversation Agent {self.config.name} initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Conversation Agent: {e}")
            raise
    
    async def process(self, request: Dict[str, Any]) -> AgentResponse:
        """Process conversation request"""
        try:
            message = request.get("message", "")
            user_id = request.get("user_id")
            context = request.get("context", {})
            conversation_history = request.get("conversation_history", [])
            
            if not message.strip():
                return AgentResponse(
                    content="Please provide a message to continue our conversation.",
                    confidence=0.0,
                    agent_name=self.config.name,
                    processing_time=0.0,
                    metadata={"error": "empty_message"}
                )
            
            # Generate response using interaction model
            response = await self.interaction_model.generate(
                message, user_id, context, conversation_history
            )
            
            # Extract response content and confidence
            content = response.get("response", "I'm sorry, I couldn't generate a response.")
            confidence = response.get("confidence", 0.5)
            
            return AgentResponse(
                content=content,
                confidence=confidence,
                agent_name=self.config.name,
                processing_time=0.0,  # Will be set by base class
                metadata={
                    "user_id": user_id,
                    "message": message,
                    "conversation_history_length": len(conversation_history),
                    "model_metadata": response.get("metadata", {})
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing conversation request: {e}")
            return AgentResponse(
                content="I encountered an error while processing your message. Please try again.",
                confidence=0.0,
                agent_name=self.config.name,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    async def generate_with_context(
        self, 
        message: str, 
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> AgentResponse:
        """Generate response with full context"""
        request = {
            "message": message,
            "user_id": user_id,
            "context": context or {},
            "conversation_history": conversation_history or []
        }
        
        return await self.execute(request)
    
    async def start_conversation(self, user_id: str, initial_message: str = None) -> AgentResponse:
        """Start a new conversation"""
        if initial_message:
            return await self.generate_with_context(initial_message, user_id)
        else:
            return AgentResponse(
                content="Xin chào! Tôi là trợ lý AI của bạn. Tôi có thể giúp gì cho bạn hôm nay?",
                confidence=1.0,
                agent_name=self.config.name,
                processing_time=0.0,
                metadata={"conversation_started": True, "user_id": user_id}
            )
    
    async def continue_conversation(
        self, 
        user_id: str, 
        message: str,
        conversation_history: List[Dict[str, Any]]
    ) -> AgentResponse:
        """Continue an existing conversation"""
        return await self.generate_with_context(
            message, user_id, conversation_history=conversation_history
        )
    
    async def cleanup(self) -> None:
        """Cleanup conversation agent resources"""
        try:
            await self.interaction_model.cleanup()
            self.logger.info(f"Conversation Agent {self.config.name} cleaned up")
        except Exception as e:
            self.logger.error(f"Error during conversation agent cleanup: {e}")
