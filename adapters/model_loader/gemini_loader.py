"""
Gemini Model Loader
Google Gemini API integration
"""

import asyncio
import logging
from typing import Optional
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from .base_loader import BaseModelLoader

logger = logging.getLogger(__name__)

class GeminiLoader(BaseModelLoader):
    """Google Gemini model loader"""
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
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
            
            logger.info("Gemini model %s initialized successfully", self.model_name)
            return True
            
        except Exception as e:
            logger.error("Failed to initialize Gemini model: %s", e)
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
            
            # Gemini có thể trả response bị chặn (safety) hoặc không có text
            if not response or not response.candidates:
                logger.warning("Gemini returned empty or blocked response (no candidates)")
                return "Xin lỗi, tôi không thể tạo phản hồi cho nội dung này. Bạn thử hỏi khác nhé."
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                reason = getattr(candidate, "finish_reason", None) or "unknown"
                logger.warning("Gemini blocked or empty content: finish_reason=%s", reason)
                return "Xin lỗi, tôi không thể tạo phản hồi cho nội dung này. Bạn thử hỏi khác nhé."
            text = response.text
            if not (text and text.strip()):
                return "Xin lỗi, tôi không thể tạo phản hồi cho nội dung này. Bạn thử hỏi khác nhé."
            return text
            
        except google_exceptions.ResourceExhausted as e:
            logger.warning("Gemini quota exceeded (429): %s", e)
            return "Hiện đã hết lượt gọi API (quota) cho hôm nay. Bạn vui lòng thử lại sau hoặc đợi vài phút."
        except Exception as e:
            logger.exception("Failed to generate response with Gemini: %s", e)
            # In ra console để dễ debug khi test Postman
            print(f"[Gemini ERROR] {type(e).__name__}: {e}", flush=True)
            return "Xin lỗi, tôi gặp lỗi khi tạo phản hồi. Vui lòng thử lại sau."
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            logger.info("Cleaning up Gemini loader...")
            # No explicit cleanup needed for Gemini
            logger.info("Gemini loader cleanup completed")
            
        except Exception as e:
            logger.error("Error during Gemini cleanup: %s", e)