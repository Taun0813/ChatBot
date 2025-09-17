"""
Base Model Classes
Abstract base classes for all AI models in the system
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import time

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Base configuration for all models"""
    name: str
    enabled: bool = True
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class ModelResponse:
    """Standard response format for all models"""
    content: str
    metadata: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class BaseModel(ABC):
    """Abstract base class for all AI models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.is_initialized = False
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up model resources"""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from prompt"""
        pass
    
    async def generate_stream(self, prompt: str, **kwargs):
        """Generate streaming response (optional)"""
        raise NotImplementedError("Streaming not implemented for this model")
    
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """Generate responses for multiple prompts"""
        results = []
        for prompt in prompts:
            try:
                response = await self.generate(prompt, **kwargs)
                results.append(response)
            except Exception as e:
                self.logger.error(f"Error processing prompt: {e}")
                results.append(ModelResponse(
                    content="",
                    metadata={},
                    success=False,
                    error=str(e)
                ))
        return results
    
    def is_ready(self) -> bool:
        """Check if model is ready for use"""
        return self.is_initialized and self.config.enabled

class VectorStoreModel(BaseModel):
    """Base class for vector store models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.dimension: int = 768
        self.document_store: Dict[str, Dict[str, Any]] = {}
        
    @abstractmethod
    async def add_documents(
        self, 
        documents: List[str], 
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to vector store"""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from vector store"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        pass

class EmbeddingModel(BaseModel):
    """Base class for embedding models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.dimension: int = 768
        
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        return await self.embed_text(query)
