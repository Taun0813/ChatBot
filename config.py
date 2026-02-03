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
    # Railway và nhiều platform khác sử dụng PORT (uppercase)
    port: int = Field(8000, alias="PORT")  # Railway uses PORT, fallback to API_PORT if needed

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
    model_loader_backend: str = Field("gemini", alias="MODEL_LOADER_BACKEND")
    vectorstore_backend: str = Field("pinecone", alias="VECTORSTORE_BACKEND")

    # Phase 1 - RAG & API (bật qua env khi đã có Pinecone/Spring Boot)
    rag_enabled: bool = Field(False, alias="RAG_ENABLED")
    enable_api_calls: bool = Field(False, alias="ENABLE_API_CALLS")

    # Model Configuration
    model_name: str = Field("gemini-2.5-flash", alias="MODEL_NAME")
    max_tokens: int = Field(2048, alias="MAX_TOKENS")
    temperature: float = Field(0.7, alias="TEMPERATURE")
    top_p: float = Field(0.9, alias="TOP_P")

    # Personalization (tắt mặc định để tránh lỗi DB khi chưa setup)
    enable_personalization: bool = Field(False, alias="ENABLE_PERSONALIZATION")
    enable_recommendations: bool = Field(False, alias="ENABLE_RECOMMENDATIONS")
    enable_rl_learning: bool = Field(False, alias="ENABLE_RL_LEARNING")

    # External services - Spring Boot Microservices (via API Gateway)
    api_gateway_url: str = Field("http://localhost:8181", alias="API_GATEWAY_URL")
    order_service_url: str = Field("http://localhost:8181/api/orders", alias="ORDER_SERVICE_URL")
    payment_service_url: str = Field("http://localhost:8181/api/payments", alias="PAYMENT_SERVICE_URL")
    warranty_service_url: str = Field("http://localhost:8181/api/warranties", alias="WARRANTY_SERVICE_URL")
    product_service_url: str = Field("http://localhost:8181/api/products", alias="PRODUCT_SERVICE_URL")
    cart_service_url: str = Field("http://localhost:8181/api/carts", alias="CART_SERVICE_URL")
    user_service_url: str = Field("http://localhost:8181/api/users", alias="USER_SERVICE_URL")

    order_service_api_key: Optional[str] = None
    payment_service_api_key: Optional[str] = None
    warranty_service_api_key: Optional[str] = None
    product_service_api_key: Optional[str] = None
    
    # JWT Token for Spring Boot services authentication
    jwt_token: Optional[str] = Field(None, alias="JWT_TOKEN")

    # API timeout
    api_timeout: int = Field(30, alias="API_TIMEOUT")

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
