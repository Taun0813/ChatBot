"""
OpenAI Model Loader
Handles OpenAI API integration
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import openai
from .base_loader import BaseModelLoader

logger = logging.getLogger(__name__)

class OpenAILoader(BaseModelLoader):
    """OpenAI model loader implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        self.client = None
    
    async def initialize(self) -> bool:
        """Initialize OpenAI client"""
        try:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            logger.info(f"OpenAI loader initialized with model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI loader: {e}")
            return False
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate response using OpenAI API
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional parameters
        
        Returns:
            Generated response
        """
        try:
            if not self.client:
                raise ValueError("OpenAI client not initialized")
            
            # Use provided parameters or fallback to config
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            top_p = top_p or self.top_p
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            # Extract response
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                return "No response generated"
                
        except Exception as e:
            logger.error(f"Failed to generate response with OpenAI: {e}")
            return f"Error generating response: {str(e)}"
    
    async def cleanup(self):
        """Cleanup OpenAI client"""
        try:
            if self.client:
                # OpenAI client doesn't need explicit cleanup
                self.client = None
            logger.info("OpenAI loader cleanup completed")
        except Exception as e:
            logger.error(f"Error during OpenAI loader cleanup: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "backend": "openai",
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
