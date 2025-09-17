"""
Agent Orchestrator - Coordinates multiple AI agents
"""
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
import time

from .base_agent import BaseAgent, AgentConfig, AgentResponse
from .rag_agent import RAGAgent
from .conversation_agent import ConversationAgent
from .api_agent import APIAgent

logger = logging.getLogger(__name__)

@dataclass
class OrchestratorConfig:
    """Configuration for agent orchestrator"""
    enable_parallel_processing: bool = True
    max_concurrent_agents: int = 3
    default_timeout: int = 30
    enable_fallback: bool = True
    fallback_agent: str = "conversation"

class AgentOrchestrator:
    """Orchestrates multiple AI agents for complex tasks"""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents: Dict[str, BaseAgent] = {}
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize orchestrator and all agents"""
        try:
            self.logger.info("Initializing Agent Orchestrator...")
            self.is_initialized = True
            self.logger.info("Agent Orchestrator initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator"""
        self.agents[name] = agent
        self.logger.info(f"Registered agent: {name}")
    
    async def process_request(
        self, 
        request: Dict[str, Any],
        agent_preference: Optional[str] = None
    ) -> AgentResponse:
        """Process request using appropriate agent(s)"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Orchestrator not initialized")
            
            # Determine which agent(s) to use
            agents_to_use = self._select_agents(request, agent_preference)
            
            if not agents_to_use:
                return self._create_error_response("No suitable agents available")
            
            # Process with selected agents
            if len(agents_to_use) == 1:
                return await self._process_single_agent(agents_to_use[0], request)
            else:
                return await self._process_multiple_agents(agents_to_use, request)
                
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return self._create_error_response(str(e))
    
    def _select_agents(self, request: Dict[str, Any], preference: Optional[str] = None) -> List[str]:
        """Select appropriate agents for the request"""
        message = request.get("message", "").lower()
        intent = request.get("intent", "auto")
        
        # If preference is specified and agent exists, use it
        if preference and preference in self.agents:
            return [preference]
        
        # Auto-select based on intent or message content
        selected_agents = []
        
        if intent == "search" or any(keyword in message for keyword in ["tìm", "search", "mua", "buy", "sản phẩm"]):
            if "rag" in self.agents:
                selected_agents.append("rag")
        
        if intent == "api_call" or any(keyword in message for keyword in ["đơn hàng", "order", "thanh toán", "payment"]):
            if "api" in self.agents:
                selected_agents.append("api")
        
        # Always include conversation agent as fallback
        if "conversation" in self.agents:
            if not selected_agents or intent == "chat":
                selected_agents.append("conversation")
        
        return selected_agents
    
    async def _process_single_agent(self, agent_name: str, request: Dict[str, Any]) -> AgentResponse:
        """Process request with a single agent"""
        agent = self.agents[agent_name]
        return await agent.execute(request)
    
    async def _process_multiple_agents(
        self, 
        agent_names: List[str], 
        request: Dict[str, Any]
    ) -> AgentResponse:
        """Process request with multiple agents"""
        if self.config.enable_parallel_processing:
            return await self._process_parallel(agent_names, request)
        else:
            return await self._process_sequential(agent_names, request)
    
    async def _process_parallel(
        self, 
        agent_names: List[str], 
        request: Dict[str, Any]
    ) -> AgentResponse:
        """Process request with multiple agents in parallel"""
        try:
            # Limit concurrent agents
            semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)
            
            async def process_with_semaphore(agent_name: str):
                async with semaphore:
                    agent = self.agents[agent_name]
                    return await agent.execute(request)
            
            # Execute agents in parallel
            tasks = [process_with_semaphore(name) for name in agent_names]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine responses
            return self._combine_responses(responses, agent_names)
            
        except Exception as e:
            self.logger.error(f"Error in parallel processing: {e}")
            return self._create_error_response(str(e))
    
    async def _process_sequential(
        self, 
        agent_names: List[str], 
        request: Dict[str, Any]
    ) -> AgentResponse:
        """Process request with multiple agents sequentially"""
        try:
            responses = []
            
            for agent_name in agent_names:
                agent = self.agents[agent_name]
                response = await agent.execute(request)
                responses.append(response)
                
                # If we get a good response, we can stop
                if response.confidence > 0.8:
                    break
            
            return self._combine_responses(responses, agent_names)
            
        except Exception as e:
            self.logger.error(f"Error in sequential processing: {e}")
            return self._create_error_response(str(e))
    
    def _combine_responses(
        self, 
        responses: List[AgentResponse], 
        agent_names: List[str]
    ) -> AgentResponse:
        """Combine multiple agent responses into one"""
        # Filter out exceptions and failed responses
        valid_responses = [
            r for r in responses 
            if isinstance(r, AgentResponse) and r.error is None
        ]
        
        if not valid_responses:
            return self._create_error_response("All agents failed to process the request")
        
        # Select the best response (highest confidence)
        best_response = max(valid_responses, key=lambda r: r.confidence)
        
        # If we have multiple good responses, combine them
        if len(valid_responses) > 1:
            combined_content = self._merge_response_content(valid_responses)
            best_response.content = combined_content
        
        # Update metadata to include all agent info
        best_response.metadata["agents_used"] = agent_names
        best_response.metadata["total_agents"] = len(agent_names)
        best_response.metadata["successful_agents"] = len(valid_responses)
        
        return best_response
    
    def _merge_response_content(self, responses: List[AgentResponse]) -> str:
        """Merge content from multiple responses"""
        if len(responses) == 1:
            return responses[0].content
        
        # Combine responses with clear separation
        combined = "Tôi đã tìm thấy thông tin từ nhiều nguồn:\n\n"
        
        for i, response in enumerate(responses, 1):
            if response.content:
                combined += f"**Nguồn {i}:** {response.content}\n\n"
        
        return combined.strip()
    
    def _create_error_response(self, error_message: str) -> AgentResponse:
        """Create error response"""
        return AgentResponse(
            content=f"Xin lỗi, tôi gặp lỗi: {error_message}",
            confidence=0.0,
            agent_name="orchestrator",
            processing_time=0.0,
            metadata={"error": error_message}
        )
    
    async def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """Get status of a specific agent"""
        if agent_name not in self.agents:
            return {"error": f"Agent {agent_name} not found"}
        
        agent = self.agents[agent_name]
        return agent.get_health_status()
    
    async def get_all_agents_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {}
        for name, agent in self.agents.items():
            status[name] = agent.get_health_status()
        return status
    
    async def enable_agent(self, agent_name: str) -> bool:
        """Enable a specific agent"""
        if agent_name in self.agents:
            self.agents[agent_name].enable()
            return True
        return False
    
    async def disable_agent(self, agent_name: str) -> bool:
        """Disable a specific agent"""
        if agent_name in self.agents:
            self.agents[agent_name].disable()
            return True
        return False
    
    async def cleanup(self) -> None:
        """Cleanup orchestrator and all agents"""
        try:
            self.logger.info("Cleaning up Agent Orchestrator...")
            
            # Cleanup all agents
            for name, agent in self.agents.items():
                try:
                    await agent.cleanup()
                    self.logger.info(f"Cleaned up agent: {name}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up agent {name}: {e}")
            
            self.agents.clear()
            self.is_initialized = False
            self.logger.info("Agent Orchestrator cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during orchestrator cleanup: {e}")
