"""
Groq Model Loader
Groq API integration
"""

import asyncio
import logging
from typing import Optional
from groq import Groq
from .base_loader import BaseModelLoader

logger = logging.getLogger(__name__)

class GroqLoader(BaseModelLoader):
    """Groq model loader"""
    
    def __init__(
        self,
        model_name: str = "llama3-8b-8192",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        api_key: Optional[str] = None
    ):
        super().__init__(model_name, max_tokens, temperature, top_p)
        self.api_key = api_key
        self.client = None
        
    async def initialize(self) -> bool:
        """Initialize Groq client"""
        try:
            if not self.api_key:
                logger.error("Groq API key not provided")
                return False
            
            # Initialize Groq client
            self.client = Groq(api_key=self.api_key)
            
            logger.info(f"Groq model {self.model_name} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq model: {e}")
            return False
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """Generate response using Groq"""
        try:
            if not self.client:
                raise ValueError("Groq client not initialized")
            
            # Use provided parameters or defaults
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            top_p = top_p or self.top_p
            
            # Generate response
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate response with Groq: {e}")
            return "Xin lỗi, tôi gặp lỗi khi tạo phản hồi. Vui lòng thử lại sau."
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            logger.info("Cleaning up Groq loader...")
            # No explicit cleanup needed for Groq
            logger.info("Groq loader cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Groq cleanup: {e}")