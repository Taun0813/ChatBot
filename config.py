# config.py - Configuration Settings
import logging
from typing import Any, Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # Environment
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = Field("0.0.0.0", alias="API_HOST")
    port: int = Field(8000, alias="API_PORT")

    # API Keys
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None

    # Pinecone Configuration
    pinecone_environment: str = "us-west1-gcp-free"
    pinecone_index_name: str = "product-search"
    pinecone_dimension: int = 1024
    pinecone_metric: str = "cosine"
    pinecone_namespace: str = "default"

    # Backends
    model_loader_backend: str = "gemini"
    vectorstore_backend: str = "pinecone"

    # Model Configuration
    model_name: str = "gemini-1.5-flash"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

    # Personalization
    enable_personalization: bool = True
    enable_recommendations: bool = True
    enable_rl_learning: bool = True

    # External services
    order_service_url: str = "http://localhost:8081/api/orders"
    payment_service_url: str = "http://localhost:8082/api/payments"
    warranty_service_url: str = "http://localhost:8083/api/warranties"
    product_service_url: str = "http://localhost:8084/api/products"

    order_service_api_key: Optional[str] = None
    payment_service_api_key: Optional[str] = None
    warranty_service_api_key: Optional[str] = None
    product_service_api_key: Optional[str] = None

    # API timeout
    api_timeout: int = 30

    # Multi-model + Cache
    enable_multi_model: bool = True
    model_switching_strategy: str = "round_robin"
    fallback_enabled: bool = True
    cost_optimization: bool = True

    # Cache
    enable_caching: bool = True
    memory_cache_size: int = 1000
    memory_cache_ttl: int = 300
    redis_cache_host: str = "localhost"
    redis_cache_port: int = 6379
    redis_cache_db: int = 0
    redis_cache_ttl: int = 3600
    redis_cache_prefix: str = "ai_agent:"

    # Multi-model configs
    models: list[dict[str, Any]] = [
        {
            "backend": "gemini",
            "priority": 1,
            "cost_per_token": 0.001,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        {
            "backend": "groq",
            "priority": 2,
            "cost_per_token": 0.0005,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    ]

    # Hybrid Orchestrator Configuration
    enable_hybrid_orchestrator: bool = True
    hybrid_fusion_weights: dict[str, float] = {
        "rule_based": 0.4,
        "ml_based": 0.6
    }
    adaptive_weights: bool = True
    min_samples_for_adaptation: int = 10
    hybrid_fallback_to_rule: bool = True

    model_config = ConfigDict(
        extra="allow",
        env_file=".env",
        case_sensitive=False,
        protected_namespaces=(),
    )


settings = Settings()

def get_settings() -> Settings:
    return settings
