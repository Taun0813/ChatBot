"""
Model Loader Factory
Factory for creating different model loaders
"""

from .base_loader import BaseModelLoader
from .gemini_loader import GeminiLoader
from .groq_loader import GroqLoader
from .ollama_loader import OllamaLoader

def create_model_loader_adapter(
    backend: str,
    model_name: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs
) -> BaseModelLoader:
    """
    Create model loader based on backend
    
    Args:
        backend: Backend type (gemini, groq, ollama)
        model_name: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        **kwargs: Additional parameters
    
    Returns:
        Model loader instance
    """
    return ModelLoaderFactory.create_loader(
        backend=backend,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        **kwargs
    )

class ModelLoaderFactory:
    """Factory for creating model loaders"""
    
    @staticmethod
    def create_loader(
        backend: str,
        model_name: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> BaseModelLoader:
        """
        Create model loader based on backend
        
        Args:
            backend: Backend type (gemini, groq, ollama)
            model_name: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional parameters
        
        Returns:
            Model loader instance
        """
        if backend == "gemini":
            return GeminiLoader(
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
        elif backend == "groq":
            return GroqLoader(
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
        elif backend == "ollama":
            return OllamaLoader(
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

__all__ = ["ModelLoaderFactory", "create_model_loader_adapter", "BaseModelLoader", "GeminiLoader", "GroqLoader", "OllamaLoader"]