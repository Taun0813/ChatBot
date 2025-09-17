"""
Multi-Model Loader
Handles multiple model backends with fallback and load balancing
"""

import asyncio
import logging
import random
from typing import Dict, Any, Optional, List
from .base_loader import BaseModelLoader
from .gemini_loader import GeminiLoader
from .groq_loader import GroqLoader
from .ollama_loader import OllamaLoader
from .openai_loader import OpenAILoader

logger = logging.getLogger(__name__)

class MultiModelLoader(BaseModelLoader):
    """
    Multi-model loader with fallback and load balancing
    
    Features:
    - Multiple model backends
    - Automatic fallback on failure
    - Load balancing strategies
    - Cost optimization
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.models = []
        self.current_model_index = 0
        self.load_balancing_strategy = config.get("load_balancing", "round_robin")  # round_robin, random, cost_optimized
        self.fallback_enabled = config.get("fallback_enabled", True)
        self.cost_optimization = config.get("cost_optimization", True)
        
        # Model configurations
        self.model_configs = config.get("models", [])
        if not self.model_configs:
            raise ValueError("No model configurations provided")
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all configured models"""
        try:
            for model_config in self.model_configs:
                backend = model_config.get("backend")
                if not backend:
                    logger.warning(f"Model config missing backend: {model_config}")
                    continue
                
                # Create model loader
                if backend == "gemini":
                    model = GeminiLoader(model_config)
                elif backend == "groq":
                    model = GroqLoader(model_config)
                elif backend == "ollama":
                    model = OllamaLoader(model_config)
                elif backend == "openai":
                    model = OpenAILoader(model_config)
                else:
                    logger.warning(f"Unsupported backend: {backend}")
                    continue
                
                self.models.append({
                    "loader": model,
                    "config": model_config,
                    "priority": model_config.get("priority", 1),
                    "cost_per_token": model_config.get("cost_per_token", 0.001),
                    "max_tokens": model_config.get("max_tokens", 2048),
                    "is_available": True,
                    "success_count": 0,
                    "failure_count": 0,
                    "total_tokens": 0
                })
            
            logger.info(f"Initialized {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def initialize(self) -> bool:
        """Initialize all models"""
        try:
            success_count = 0
            
            for model_info in self.models:
                try:
                    success = await model_info["loader"].initialize()
                    model_info["is_available"] = success
                    if success:
                        success_count += 1
                        logger.info(f"Model {model_info['config'].get('backend')} initialized successfully")
                    else:
                        logger.warning(f"Model {model_info['config'].get('backend')} failed to initialize")
                except Exception as e:
                    logger.error(f"Failed to initialize model {model_info['config'].get('backend')}: {e}")
                    model_info["is_available"] = False
            
            if success_count == 0:
                logger.error("No models available")
                return False
            
            logger.info(f"Multi-model loader initialized with {success_count}/{len(self.models)} models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-model loader: {e}")
            return False
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        preferred_backend: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response using best available model
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            preferred_backend: Preferred backend to use
            **kwargs: Additional parameters
        
        Returns:
            Generated response
        """
        try:
            # Select model
            selected_model = await self._select_model(preferred_backend, max_tokens)
            if not selected_model:
                return "No available models"
            
            # Generate response
            response = await selected_model["loader"].generate_response(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            # Update statistics
            selected_model["success_count"] += 1
            selected_model["total_tokens"] += len(prompt.split()) + len(response.split())
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            
            # Try fallback if enabled
            if self.fallback_enabled:
                return await self._fallback_generate_response(
                    prompt, max_tokens, temperature, top_p, **kwargs
                )
            
            return f"Error generating response: {str(e)}"
    
    async def _select_model(
        self,
        preferred_backend: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Select best model based on strategy"""
        try:
            # Filter available models
            available_models = [m for m in self.models if m["is_available"]]
            if not available_models:
                return None
            
            # Filter by preferred backend
            if preferred_backend:
                preferred_models = [m for m in available_models if m["config"].get("backend") == preferred_backend]
                if preferred_models:
                    available_models = preferred_models
            
            # Filter by max_tokens if specified
            if max_tokens:
                available_models = [m for m in available_models if m["max_tokens"] >= max_tokens]
            
            if not available_models:
                return None
            
            # Select model based on strategy
            if self.load_balancing_strategy == "round_robin":
                return self._round_robin_select(available_models)
            elif self.load_balancing_strategy == "random":
                return self._random_select(available_models)
            elif self.load_balancing_strategy == "cost_optimized":
                return self._cost_optimized_select(available_models)
            else:
                return available_models[0]
                
        except Exception as e:
            logger.error(f"Failed to select model: {e}")
            return available_models[0] if available_models else None
    
    def _round_robin_select(self, available_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Round-robin model selection"""
        selected = available_models[self.current_model_index % len(available_models)]
        self.current_model_index = (self.current_model_index + 1) % len(available_models)
        return selected
    
    def _random_select(self, available_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Random model selection"""
        return random.choice(available_models)
    
    def _cost_optimized_select(self, available_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cost-optimized model selection"""
        if not self.cost_optimization:
            return available_models[0]
        
        # Sort by cost per token
        sorted_models = sorted(available_models, key=lambda x: x["cost_per_token"])
        return sorted_models[0]
    
    async def _fallback_generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """Fallback response generation"""
        try:
            # Try other available models
            for model_info in self.models:
                if not model_info["is_available"]:
                    continue
                
                try:
                    response = await model_info["loader"].generate_response(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        **kwargs
                    )
                    
                    # Update statistics
                    model_info["success_count"] += 1
                    model_info["total_tokens"] += len(prompt.split()) + len(response.split())
                    
                    return response
                    
                except Exception as e:
                    logger.warning(f"Fallback model {model_info['config'].get('backend')} failed: {e}")
                    model_info["failure_count"] += 1
                    continue
            
            return "All models failed to generate response"
            
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return f"Fallback error: {str(e)}"
    
    async def cleanup(self):
        """Cleanup all models"""
        try:
            for model_info in self.models:
                try:
                    await model_info["loader"].cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up model {model_info['config'].get('backend')}: {e}")
            
            logger.info("Multi-model loader cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during multi-model loader cleanup: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get multi-model information"""
        available_models = [m for m in self.models if m["is_available"]]
        
        return {
            "backend": "multi",
            "total_models": len(self.models),
            "available_models": len(available_models),
            "load_balancing": self.load_balancing_strategy,
            "fallback_enabled": self.fallback_enabled,
            "cost_optimization": self.cost_optimization,
            "models": [
                {
                    "backend": m["config"].get("backend"),
                    "priority": m["priority"],
                    "is_available": m["is_available"],
                    "success_count": m["success_count"],
                    "failure_count": m["failure_count"],
                    "total_tokens": m["total_tokens"]
                }
                for m in self.models
            ]
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_requests = sum(m["success_count"] + m["failure_count"] for m in self.models)
        total_tokens = sum(m["total_tokens"] for m in self.models)
        
        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "average_tokens_per_request": total_tokens / total_requests if total_requests > 0 else 0,
            "model_stats": [
                {
                    "backend": m["config"].get("backend"),
                    "success_rate": m["success_count"] / (m["success_count"] + m["failure_count"]) if (m["success_count"] + m["failure_count"]) > 0 else 0,
                    "total_tokens": m["total_tokens"],
                    "is_available": m["is_available"]
                }
                for m in self.models
            ]
        }
