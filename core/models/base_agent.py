"""
Base Agent class for all AI agents
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
import time

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Base configuration for agents"""
    name: str
    enabled: bool = True
    timeout: int = 30
    retry_attempts: int = 3
    cache_enabled: bool = True
    cache_ttl: int = 3600

@dataclass
class AgentResponse:
    """Standard response format for all agents"""
    content: str
    confidence: float
    agent_name: str
    processing_time: float
    metadata: Dict[str, Any]
    error: Optional[str] = None

class BaseAgent(ABC):
    """Abstract base class for all AI agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.is_initialized = False
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0
        }
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent"""
        pass
    
    @abstractmethod
    async def process(self, request: Dict[str, Any]) -> AgentResponse:
        """Process a request and return response"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        pass
    
    async def execute(self, request: Dict[str, Any]) -> AgentResponse:
        """Execute agent with error handling and metrics"""
        if not self.is_initialized:
            raise RuntimeError(f"Agent {self.config.name} not initialized")
        
        if not self.config.enabled:
            return AgentResponse(
                content="Agent is disabled",
                confidence=0.0,
                agent_name=self.config.name,
                processing_time=0.0,
                metadata={},
                error="Agent disabled"
            )
        
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Execute with timeout
            response = await asyncio.wait_for(
                self.process(request),
                timeout=self.config.timeout
            )
            
            # Update metrics
            processing_time = time.time() - start_time
            response.processing_time = processing_time
            self.metrics["successful_requests"] += 1
            self._update_average_response_time(processing_time)
            
            self.logger.info(
                f"Agent {self.config.name} processed request in {processing_time:.2f}s"
            )
            
            return response
            
        except asyncio.TimeoutError:
            self.metrics["failed_requests"] += 1
            self.logger.error(f"Agent {self.config.name} timed out after {self.config.timeout}s")
            return AgentResponse(
                content="Request timed out",
                confidence=0.0,
                agent_name=self.config.name,
                processing_time=time.time() - start_time,
                metadata={},
                error="Timeout"
            )
            
        except Exception as e:
            self.metrics["failed_requests"] += 1
            self.logger.error(f"Agent {self.config.name} failed: {e}")
            return AgentResponse(
                content="Agent processing failed",
                confidence=0.0,
                agent_name=self.config.name,
                processing_time=time.time() - start_time,
                metadata={},
                error=str(e)
            )
    
    def _update_average_response_time(self, response_time: float) -> None:
        """Update average response time metric"""
        total_requests = self.metrics["successful_requests"]
        if total_requests == 1:
            self.metrics["average_response_time"] = response_time
        else:
            current_avg = self.metrics["average_response_time"]
            self.metrics["average_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_requests"] / max(self.metrics["total_requests"], 1)
            ),
            "agent_name": self.config.name,
            "enabled": self.config.enabled,
            "initialized": self.is_initialized
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        return {
            "agent_name": self.config.name,
            "status": "healthy" if self.is_initialized and self.config.enabled else "unhealthy",
            "initialized": self.is_initialized,
            "enabled": self.config.enabled,
            "metrics": self.get_metrics()
        }
    
    def enable(self) -> None:
        """Enable the agent"""
        self.config.enabled = True
        self.logger.info(f"Agent {self.config.name} enabled")
    
    def disable(self) -> None:
        """Disable the agent"""
        self.config.enabled = False
        self.logger.info(f"Agent {self.config.name} disabled")
    
    def reset_metrics(self) -> None:
        """Reset agent metrics"""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0
        }
        self.logger.info(f"Agent {self.config.name} metrics reset")
