"""
Gemini Model Loader
Google Gemini API integration
"""

import asyncio
import logging
from typing import Optional
import google.generativeai as genai
from .base_loader import BaseModelLoader

logger = logging.getLogger(__name__)

class GeminiLoader(BaseModelLoader):
    """Google Gemini model loader"""
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        api_key: Optional[str] = None
    ):
        super().__init__(model_name, max_tokens, temperature, top_p)
        self.api_key = api_key
        self.model = None
        
    async def initialize(self) -> bool:
        """Initialize Gemini model"""
        try:
            if not self.api_key:
                logger.error("Gemini API key not provided")
                return False
            
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            
            # Initialize model
            self.model = genai.GenerativeModel(self.model_name)
            
            logger.info(f"Gemini model {self.model_name} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            return False
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """Generate response using Gemini"""
        try:
            if not self.model:
                raise ValueError("Model not initialized")
            
            # Use provided parameters or defaults
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            top_p = top_p or self.top_p
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Failed to generate response with Gemini: {e}")
            return "Xin lỗi, tôi gặp lỗi khi tạo phản hồi. Vui lòng thử lại sau."
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            logger.info("Cleaning up Gemini loader...")
            # No explicit cleanup needed for Gemini
            logger.info("Gemini loader cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Gemini cleanup: {e}")