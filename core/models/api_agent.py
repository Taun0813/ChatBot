"""
API Agent for external service integration
"""
import logging
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentConfig, AgentResponse
from ..api_model import APIModel

logger = logging.getLogger(__name__)

class APIAgent(BaseAgent):
    """API-specific agent for external service integration"""
    
    def __init__(self, config: AgentConfig, api_model: APIModel):
        super().__init__(config)
        self.api_model = api_model
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def initialize(self) -> None:
        """Initialize API agent"""
        try:
            await self.api_model.initialize()
            self.is_initialized = True
            self.logger.info(f"API Agent {self.config.name} initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize API Agent: {e}")
            raise
    
    async def process(self, request: Dict[str, Any]) -> AgentResponse:
        """Process API request"""
        try:
            message = request.get("message", "")
            user_id = request.get("user_id")
            context = request.get("context", {})
            api_type = request.get("api_type", "auto")
            
            if not message.strip():
                return AgentResponse(
                    content="Please provide a request for API processing.",
                    confidence=0.0,
                    agent_name=self.config.name,
                    processing_time=0.0,
                    metadata={"error": "empty_message"}
                )
            
            # Process API request using API model
            response = await self.api_model.process_api_request(
                message, user_id, context, api_type
            )
            
            # Extract response content and confidence
            content = response.get("response", "I couldn't process your API request.")
            confidence = response.get("confidence", 0.5)
            
            return AgentResponse(
                content=content,
                confidence=confidence,
                agent_name=self.config.name,
                processing_time=0.0,  # Will be set by base class
                metadata={
                    "user_id": user_id,
                    "api_type": api_type,
                    "api_metadata": response.get("metadata", {}),
                    "api_response": response
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing API request: {e}")
            return AgentResponse(
                content="I encountered an error while processing your API request. Please try again.",
                confidence=0.0,
                agent_name=self.config.name,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    async def process_order_request(
        self, 
        message: str, 
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process order-related API request"""
        request = {
            "message": message,
            "user_id": user_id,
            "context": context or {},
            "api_type": "order"
        }
        return await self.execute(request)
    
    async def process_payment_request(
        self, 
        message: str, 
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process payment-related API request"""
        request = {
            "message": message,
            "user_id": user_id,
            "context": context or {},
            "api_type": "payment"
        }
        return await self.execute(request)
    
    async def process_warranty_request(
        self, 
        message: str, 
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process warranty-related API request"""
        request = {
            "message": message,
            "user_id": user_id,
            "context": context or {},
            "api_type": "warranty"
        }
        return await self.execute(request)
    
    async def get_available_services(self) -> List[str]:
        """Get list of available API services"""
        try:
            return await self.api_model.get_available_services()
        except Exception as e:
            self.logger.error(f"Error getting available services: {e}")
            return []
    
    async def test_service_connection(self, service_name: str) -> bool:
        """Test connection to a specific service"""
        try:
            return await self.api_model.test_service_connection(service_name)
        except Exception as e:
            self.logger.error(f"Error testing service connection: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup API agent resources"""
        try:
            await self.api_model.cleanup()
            self.logger.info(f"API Agent {self.config.name} cleaned up")
        except Exception as e:
            self.logger.error(f"Error during API agent cleanup: {e}")
