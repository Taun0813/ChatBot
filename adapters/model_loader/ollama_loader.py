"""
Ollama Model Loader
Ollama local model integration
"""

import asyncio
import logging
from typing import Optional
import httpx
from .base_loader import BaseModelLoader

logger = logging.getLogger(__name__)

class OllamaLoader(BaseModelLoader):
    """Ollama local model loader"""
    
    def __init__(
        self,
        model_name: str = "llama2",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        base_url: str = "http://localhost:11434"
    ):
        super().__init__(model_name, max_tokens, temperature, top_p)
        self.base_url = base_url
        self.client = None
        
    async def initialize(self) -> bool:
        """Initialize Ollama client"""
        try:
            # Initialize HTTP client
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0
            )
            
            # Test connection
            response = await self.client.get("/api/tags")
            if response.status_code == 200:
                logger.info(f"Ollama model {self.model_name} initialized successfully")
                return True
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Ollama model: {e}")
            return False
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """Generate response using Ollama"""
        try:
            if not self.client:
                raise ValueError("Ollama client not initialized")
            
            # Use provided parameters or defaults
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            top_p = top_p or self.top_p
            
            # Prepare request data
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
            
            # Generate response
            response = await self.client.post("/api/generate", json=data)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"Failed to generate response with Ollama: {e}")
            return "Xin lỗi, tôi gặp lỗi khi tạo phản hồi. Vui lòng thử lại sau."
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            logger.info("Cleaning up Ollama loader...")
            if self.client:
                await self.client.aclose()
            logger.info("Ollama loader cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Ollama cleanup: {e}")