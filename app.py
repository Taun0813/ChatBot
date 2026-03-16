"""
AI Agent FastAPI Application
Entry point for the AI Agent system with /chat endpoint
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
import logging
import time
import json
from contextlib import asynccontextmanager

# Import sẽ được thực hiện trong runtime để tránh circular import
from config import get_settings
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Global router instance
router_instance = None


def _parse_cors_settings() -> tuple[list[str], bool]:
    """Parse CORS origins from settings and derive safe credential policy."""
    settings = get_settings()
    raw_origins = getattr(settings, "cors_allowed_origins", "")

    if isinstance(raw_origins, str) and raw_origins.strip():
        origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    elif isinstance(raw_origins, list):
        origins = [str(origin).strip() for origin in raw_origins if str(origin).strip()]
    else:
        origins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]

    # Wildcard origin cannot be combined with credentials in browsers.
    allow_credentials = "*" not in origins
    return origins, allow_credentials

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global router_instance
    
    # Startup
    logger.info("Starting AI Agent application...")
    settings = get_settings()
    
    try:
        # Initialize the Agno router
        from core.router import AgnoRouter, RouterConfig
        
        # Create config with personalization and hybrid orchestrator
        config = RouterConfig(
            rag_config={
                "name": "rag",
                "enabled": settings.rag_enabled,
                "pinecone_config": {
                    "api_key": settings.pinecone_api_key,
                    "environment": settings.pinecone_environment,
                    "index_name": settings.pinecone_index_name,
                    "dimension": settings.pinecone_dimension,
                    "metric": settings.pinecone_metric
                },
                "model_loader_config": {
                    "name": "model_loader",
                    "enabled": True,
                    "backend": settings.model_loader_backend,
                    "model_name": settings.model_name,
                    "max_tokens": settings.max_tokens,
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                    "api_key": settings.gemini_api_key if settings.model_loader_backend == "gemini" else settings.groq_api_key if settings.model_loader_backend == "groq" else settings.openai_api_key if settings.model_loader_backend == "openai" else settings.anthropic_api_key
                }
            },
            interaction_config={},
            api_config={
                "enable_api_calls": settings.enable_api_calls,
                "order_service_url": settings.order_service_url,
                "payment_service_url": settings.payment_service_url,
                "warranty_service_url": settings.warranty_service_url,
                "product_service_url": settings.product_service_url,
                "jwt_token": settings.jwt_token,
                "order_service_api_key": settings.order_service_api_key,
                "payment_service_api_key": settings.payment_service_api_key,
                "warranty_service_api_key": settings.warranty_service_api_key,
                "product_service_api_key": settings.product_service_api_key,
                "api_timeout": settings.api_timeout
            },
            personalization_config={
                "enable_personalization": settings.enable_personalization,
                "enable_recommendations": settings.enable_recommendations,
                "enable_rl_learning": settings.enable_rl_learning,
                "db_path": "data/profiles/profiles.db",
                "json_backup": True,
                "profiles_dir": "./data/profiles",
                "models_dir": "./data/models"
            },
            hybrid_config={
                "enable_hybrid": settings.enable_hybrid_orchestrator,
                "fusion_weights": settings.hybrid_fusion_weights,
                "adaptive_weights": settings.adaptive_weights,
                "min_samples_for_adaptation": settings.min_samples_for_adaptation,
                "fallback_to_rule": settings.hybrid_fallback_to_rule
            }
        )
        
        router_instance = AgnoRouter(config)
        await router_instance.initialize()
        logger.info("Agno router initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to initialize application: %s", e)
        raise
    finally:
        # Shutdown
        logger.info("Shutting down AI Agent application...")
        if router_instance:
            await router_instance.cleanup()

# Create FastAPI app with enhanced OpenAPI documentation
app = FastAPI(
    title="AI Agent API",
    description="""
    Intelligent AI Agent System for E-commerce with Hybrid Orchestrator
    
    ## Features
    
    * **Hybrid Orchestrator**: Combines rule-based + ML-based routing (85-95% accuracy)
    * **RAG System**: Semantic search with Pinecone vector database
    * **Smart Conversation**: Natural interaction with context-aware routing
    * **API Integration**: Connect with microservices (orders, payments, warranty)
    * **Personalization**: Learn from user behavior and provide relevant recommendations
    * **Multi-model**: Support multiple LLMs (Gemini, Groq, Ollama, OpenAI, Claude)
    * **Caching**: Smart caching system with Redis and Memory cache
    * **Monitoring**: Real-time performance monitoring with detailed dashboard
    * **Training**: Fine-tune models for e-commerce domain
    
    ## Dataset
    
    The system uses **Mobiles Dataset (2025).csv** with 900+ mobile phone products from major brands.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "AI Agent Support",
        "email": "support@ai-agent.com",
    },
    license_info={
        "name": "MIT License",
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.ai-agent.com",
            "description": "Production server"
        }
    ]
)

# Add CORS middleware
cors_origins, cors_allow_credentials = _parse_cors_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models with enhanced documentation
class ChatRequest(BaseModel):
    """Request model for chat/ask endpoint"""
    message: str = Field(..., description="User message to process", example="Tôi muốn tìm điện thoại OnePlus dưới 50 triệu")
    user_id: Optional[str] = Field(None, description="Unique user identifier", example="user123")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation context", example="session001")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the request")
    intent: Optional[str] = Field(None, description="Pre-specified intent: search, chat, or api_call", example="search")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Tôi muốn tìm điện thoại OnePlus dưới 50 triệu",
                "user_id": "user123",
                "session_id": "session001",
                "intent": "search"
            }
        }

class ChatResponse(BaseModel):
    """Response model for chat/ask endpoint"""
    user_id: Optional[str] = Field(None, description="User identifier")
    response: str = Field(..., description="AI agent response message")
    intent: str = Field(..., description="Detected intent: search, chat, or api_call")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)", ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the response")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "response": "Tôi tìm thấy một số điện thoại OnePlus phù hợp với ngân sách của bạn...",
                "intent": "search",
                "confidence": 0.95,
                "session_id": "session001",
                "metadata": {
                    "model_info": {
                        "backend": "gemini",
                        "model_name": "gemini-2.5-flash"
                    }
                }
            }
        }

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Health status: healthy or unhealthy")
    message: str = Field(..., description="Status message")
    version: str = Field(..., description="API version")

# Dependency to get router
async def get_router():
    if router_instance is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    return router_instance

@app.get("/health", tags=["Monitoring"], response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint
    
    Returns the health status of the AI Agent system including:
    - Overall system status
    - Router initialization status
    - API version
    - Timestamp
    """
    try:
        # Simple health check without complex monitoring
        router_status = "initialized" if router_instance is not None else "not_initialized"
        
        return {
            "status": "healthy" if router_instance is not None else "unhealthy",
            "message": "AI Agent is running",
            "version": "1.0.0",
            "router_status": router_status,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}",
            "version": "1.0.0",
            "timestamp": time.time()
        }

@app.post("/ask", response_model=ChatResponse, tags=["Chat"])
async def ask(
    request: ChatRequest,
    http_request: Request,
    router = Depends(get_router)
):
    """
    Main ask endpoint that processes user messages through the Hybrid Orchestrator
    
    This endpoint is the primary interface for interacting with the AI Agent system.
    It routes user messages through the hybrid orchestrator which combines rule-based
    and ML-based routing to determine the best response.
    
    **Process Flow:**
    1. User message is received
    2. Hybrid Orchestrator analyzes the message
    3. Intent is detected (search, chat, or api_call)
    4. Appropriate agent processes the request
    5. Response is generated and returned
    
    **Example Use Cases:**
    - Product search: "Tìm điện thoại Samsung dưới 20 triệu"
    - General chat: "Xin chào, bạn có thể giúp tôi không?"
    - Order inquiry: "Đơn hàng #1234 của tôi ở đâu?"
    """
    try:
        logger.info("Received ask request: %s...", request.message[:100])
        
        # Build context and propagate auth token from incoming Authorization header
        request_context = dict(request.context or {})
        auth_header = http_request.headers.get("authorization")
        if auth_header:
            request_context.setdefault("jwt_token", auth_header)

        # Process the request through the Agno router
        response = await router.process_request(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            context=request_context,
            intent=request.intent
        )
        
        logger.info("Generated response: %s...", response["response"][:100])
        
        # Add model info to response metadata
        from config import get_settings
        settings = get_settings()
        
        if "metadata" not in response:
            response["metadata"] = {}
        
        response["metadata"]["model_info"] = {
            "backend": settings.model_loader_backend,
            "model_name": settings.model_name,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "top_p": settings.top_p
        }
        
        # Collect conversation for training (async)
        # Note: Training pipeline is optional and may not be available
        try:
            # Try to import training pipeline if available
            try:
                from training.training_pipeline import get_training_pipeline
                pipeline = get_training_pipeline()
                
                conversation = {
                    "user_message": request.message,
                    "assistant_response": response["response"],
                    "intent": response.get("intent", "unknown"),
                    "confidence": response.get("confidence", 0.0),
                    "user_id": request.user_id,
                    "session_id": request.session_id or "unknown",
                    "timestamp": time.time(),
                    "metadata": response.get("metadata", {})
                }
                
                # Collect conversation in background
                pipeline.collect_conversation(conversation)
            except ImportError:
                # Training pipeline not available, skip collection
                pass
            
        except Exception as e:
            logger.warning("Failed to collect conversation for training: %s", e)
        
        return ChatResponse(
            user_id=request.user_id,
            response=response["response"],
            intent=response["intent"],
            confidence=response["confidence"],
            metadata=response.get("metadata"),
            session_id=response.get("session_id")
        )
        
    except Exception as e:
        logger.error("Error processing ask request: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat", response_model=ChatResponse, tags=["Chat"], deprecated=True)
async def chat(
    request: ChatRequest,
    http_request: Request,
    router = Depends(get_router)
):
    """
    Legacy chat endpoint - redirects to /ask
    
    **Deprecated**: Please use `/ask` endpoint instead.
    This endpoint is kept for backward compatibility.
    """
    return await ask(request, http_request, router)

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics(router = Depends(get_router)):
    """
    Get Hybrid Orchestrator metrics
    
    Returns detailed performance metrics including:
    - Total requests processed
    - Rule-based vs ML-based vs Hybrid request counts
    - Average response times
    - Request distribution percentages
    """
    try:
        metrics = router.get_metrics()
        return {
            "status": "success",
            "metrics": metrics,
            "orchestrator_type": "hybrid" if router.enable_hybrid else "rule_based"
        }
    except Exception as e:
        logger.error("Error getting metrics: %s", e)
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

@app.get("/dashboard", tags=["Monitoring"])
async def get_dashboard(router = Depends(get_router)):
    """
    Get comprehensive monitoring dashboard
    
    Returns a complete overview of system performance including:
    - System health metrics (uptime, memory, CPU)
    - Performance metrics (response times, success rates)
    - Query breakdown by type (RAG, conversation, API)
    - Router performance statistics
    - Request tracing information
    """
    try:
        from monitoring.metrics import MetricsCollector, MetricsConfig
        from monitoring.health_check import HealthChecker, HealthCheckConfig
        from monitoring.tracing import tracer
        
        # Get metrics
        metrics_config = MetricsConfig()
        metrics_collector = MetricsCollector(metrics_config)
        metrics_collector.update_system_metrics()
        metrics_summary = metrics_collector.get_metrics_summary()
        
        # Get health status
        health_config = HealthCheckConfig()
        health_checker = HealthChecker(health_config)
        health_checker.register_check("application", health_checker.check_application_health, router)
        await health_checker.run_all_checks()
        health_summary = health_checker.get_health_summary()
        
        # Get trace statistics
        trace_stats = tracer.get_trace_stats()
        
        # Get router metrics
        router_metrics = router.get_metrics()
        
        return {
            "status": "success",
            "timestamp": time.time(),
            "dashboard": {
                "system_health": {
                    "overall_status": health_summary["overall_status"],
                    "health_score": health_summary["health_score"],
                    "uptime": metrics_summary["uptime"],
                    "memory_usage_mb": metrics_summary["memory_usage"],
                    "cpu_usage_percent": metrics_summary["cpu_usage"]
                },
                "performance_metrics": {
                    "total_requests": metrics_summary["total_requests"],
                    "success_rate": metrics_summary["success_rate"],
                    "error_rate": metrics_summary["error_rate"],
                    "average_response_time": metrics_summary["average_response_time"],
                    "avg_rag_time": metrics_summary["avg_rag_time"],
                    "avg_conversation_time": metrics_summary["avg_conversation_time"],
                    "avg_api_time": metrics_summary["avg_api_time"]
                },
                "query_breakdown": {
                    "total_queries": metrics_summary["total_queries"],
                    "rag_queries": metrics_summary["rag_queries"],
                    "conversation_queries": metrics_summary["conversation_queries"],
                    "api_queries": metrics_summary["api_queries"],
                    "rag_error_rate": metrics_summary["rag_error_rate"],
                    "conversation_error_rate": metrics_summary["conversation_error_rate"],
                    "api_error_rate": metrics_summary["api_error_rate"]
                },
                "router_performance": {
                    "rule_based_requests": metrics_summary["rule_based_requests"],
                    "ml_based_requests": metrics_summary["ml_based_requests"],
                    "hybrid_requests": metrics_summary["hybrid_requests"],
                    "rule_based_percentage": metrics_summary["rule_based_percentage"],
                    "ml_based_percentage": metrics_summary["ml_based_percentage"],
                    "hybrid_percentage": metrics_summary["hybrid_percentage"]
                },
                "tracing": {
                    "active_traces": trace_stats["active_traces"],
                    "completed_traces": trace_stats["completed_traces"],
                    "average_duration": trace_stats["average_duration"],
                    "max_duration": trace_stats["max_duration"],
                    "min_duration": trace_stats["min_duration"]
                },
                "router_metrics": router_metrics
            }
        }
    except Exception as e:
        logger.error("Error getting dashboard: %s", e)
        raise HTTPException(status_code=500, detail=f"Error getting dashboard: {str(e)}")

@app.get("/traces", tags=["Monitoring"])
async def get_traces(limit: int = 100):
    """
    Get recent request traces
    
    Returns the most recent request traces for debugging and monitoring.
    
    **Parameters:**
    - limit: Maximum number of traces to return (default: 100, max: 1000)
    """
    try:
        from monitoring.tracing import tracer
        traces = tracer.get_completed_traces(limit)
        return {
            "status": "success",
            "traces": [trace.to_dict() for trace in traces],
            "count": len(traces)
        }
    except Exception as e:
        logger.error("Error getting traces: %s", e)
        raise HTTPException(status_code=500, detail=f"Error getting traces: {str(e)}")

# ===========================================
# TRAINING & FINE-TUNING ENDPOINTS
# ===========================================

@app.post("/training/start", tags=["Training"])
async def start_training(
    data_source: str = "dataset",
    auto_mode: bool = False
):
    """
    Start training pipeline
    
    **Note**: Training pipeline module is optional and may not be available.
    This endpoint requires the training_pipeline module to be implemented.
    
    **Parameters:**
    - data_source: Source of training data ("dataset" or "conversations")
    - auto_mode: Enable automatic retraining mode
    """
    try:
        try:
            from training.training_pipeline import get_training_pipeline
            
            # Get training pipeline
            pipeline = get_training_pipeline({
                "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "conversation_buffer_size": 100,
                "auto_retrain_threshold": 1000,
                "auto_retrain_enabled": True,
                "retrain_interval": 86400
            })
            
            # Start training pipeline
            result = await pipeline.start_training_pipeline(data_source, auto_mode)
            
            return {
                "status": "success",
                "message": "Training pipeline started",
                "result": result
            }
        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Training pipeline module not available. Please implement training/training_pipeline.py"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error starting training: %s", e)
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")

@app.get("/training/status", tags=["Training"])
async def get_training_status():
    """Get current training status"""
    try:
        try:
            from training.training_pipeline import get_training_pipeline
            
            pipeline = get_training_pipeline()
            status = pipeline.get_training_status()
            
            return {
                "status": "success",
                "training_status": status
            }
        except ImportError:
            return {
                "status": "not_available",
                "message": "Training pipeline module not available"
            }
        
    except Exception as e:
        logger.error("Error getting training status: %s", e)
        raise HTTPException(status_code=500, detail=f"Error getting training status: {str(e)}")

@app.get("/training/history", tags=["Training"])
async def get_training_history():
    """Get training history"""
    try:
        try:
            from training.training_pipeline import get_training_pipeline
            
            pipeline = get_training_pipeline()
            history = pipeline.get_training_history()
            
            return {
                "status": "success",
                "training_history": history
            }
        except ImportError:
            return {
                "status": "not_available",
                "message": "Training pipeline module not available",
                "training_history": []
            }
        
    except Exception as e:
        logger.error("Error getting training history: %s", e)
        raise HTTPException(status_code=500, detail=f"Error getting training history: {str(e)}")

@app.post("/training/collect", tags=["Training"])
async def collect_conversation(conversation: Dict[str, Any]):
    """Collect conversation data for training"""
    try:
        try:
            from training.training_pipeline import get_training_pipeline
            
            pipeline = get_training_pipeline()
            pipeline.collect_conversation(conversation)
            
            return {
                "status": "success",
                "message": "Conversation collected for training",
                "buffer_size": len(pipeline.conversation_buffer)
            }
        except ImportError:
            return {
                "status": "not_available",
                "message": "Training pipeline module not available"
            }
        
    except Exception as e:
        logger.error("Error collecting conversation: %s", e)
        raise HTTPException(status_code=500, detail=f"Error collecting conversation: {str(e)}")

@app.post("/training/auto-retrain", tags=["Training"])
async def toggle_auto_retrain(enabled: bool = True):
    """Enable/disable auto-retrain"""
    try:
        try:
            from training.training_pipeline import get_training_pipeline
            
            pipeline = get_training_pipeline()
            pipeline.enable_auto_retrain(enabled)
            
            return {
                "status": "success",
                "message": f"Auto-retrain {'enabled' if enabled else 'disabled'}",
                "auto_retrain_enabled": enabled
            }
        except ImportError:
            return {
                "status": "not_available",
                "message": "Training pipeline module not available"
            }
        
    except Exception as e:
        logger.error("Error toggling auto-retrain: %s", e)
        raise HTTPException(status_code=500, detail=f"Error toggling auto-retrain: {str(e)}")

@app.post("/training/prepare-data", tags=["Training"])
async def prepare_training_data():
    """Prepare training data from conversations"""
    try:
        from training.prepare_data import DataPreparator
        
        preparator = DataPreparator()
        
        # Load existing dataset
        conversations = preparator.load_dataset("training/dataset/dataset.json")
        
        # Generate synthetic data if needed
        if len(conversations) < 100:
            conversations = preparator.generate_synthetic_data(conversations, multiplier=5)
        
        # Prepare training data
        training_data = preparator.prepare_conversation_data(conversations)
        train_data, val_data, test_data = preparator.create_training_splits(training_data)
        
        # Save prepared data
        preparator.save_training_data(train_data, "train_conversations.json")
        preparator.save_training_data(val_data, "val_conversations.json")
        preparator.save_training_data(test_data, "test_conversations.json")
        
        return {
            "status": "success",
            "message": "Training data prepared successfully",
            "data_stats": {
                "total_conversations": len(conversations),
                "training_samples": len(train_data),
                "validation_samples": len(val_data),
                "test_samples": len(test_data)
            }
        }
        
    except Exception as e:
        logger.error("Error preparing training data: %s", e)
        raise HTTPException(status_code=500, detail=f"Error preparing training data: {str(e)}")

@app.post("/training/evaluate", tags=["Training"])
async def evaluate_model():
    """Evaluate current model"""
    try:
        from training.evaluate import CloudModelEvaluator

        evaluator = CloudModelEvaluator()
        
        # Load test data
        with open("training/dataset/test_conversations.json", 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        # Initialize configured cloud model backend
        initialized = await evaluator.initialize_model()
        if not initialized:
            raise HTTPException(status_code=503, detail="Model backend initialization failed")

        # Evaluate model
        results = await evaluator.evaluate_model(test_data)
        
        # Save results
        evaluator.save_evaluation_results(results, "training/evaluation_results.json")
        
        return {
            "status": "success",
            "message": "Model evaluation completed",
            "evaluation_results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error evaluating model: %s", e)
        raise HTTPException(status_code=500, detail=f"Error evaluating model: {str(e)}")

@app.get("/", tags=["Information"])
async def root():
    """
    Root endpoint with API information
    
    Returns basic information about the API including available endpoints and features.
    """
    return {
        "message": "AI Agent API - Hybrid Orchestrator",
        "version": "1.0.0",
        "endpoints": {
            "ask": "/ask",
            "chat": "/chat (legacy)",
            "health": "/health",
            "metrics": "/metrics",
            "dashboard": "/dashboard",
            "traces": "/traces",
            "training": {
                "start": "/training/start",
                "status": "/training/status",
                "history": "/training/history",
                "collect": "/training/collect",
                "auto-retrain": "/training/auto-retrain",
                "prepare-data": "/training/prepare-data",
                "evaluate": "/training/evaluate"
            },
            "docs": "/docs"
        },
        "features": [
            "Hybrid routing (Rule-based + ML-based)",
            "Product search with RAG",
            "Order tracking",
            "Natural conversation",
            "Personalization",
            "Multi-model support",
            "Caching layer"
        ]
    }

@app.get("/test", tags=["Testing"], include_in_schema=False)
async def test_endpoint():
    """Simple test endpoint (excluded from OpenAPI schema)"""
    return {
        "status": "ok",
        "message": "Test endpoint working",
        "timestamp": time.time()
    }

@app.get("/model-info", tags=["Information"])
async def get_model_info():
    """
    Get current model information
    
    Returns information about the currently configured AI model including:
    - Backend (Gemini, Groq, etc.)
    - Model name
    - Configuration parameters (max_tokens, temperature, top_p)
    - Feature flags (personalization, recommendations, RL learning)
    """
    try:
        from config import get_settings
        settings = get_settings()
        
        return {
            "status": "success",
            "model_info": {
                "backend": settings.model_loader_backend,
                "model_name": settings.model_name,
                "max_tokens": settings.max_tokens,
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "api_timeout": settings.api_timeout
            },
            "features": {
                "personalization": settings.enable_personalization,
                "recommendations": settings.enable_recommendations,
                "rl_learning": settings.enable_rl_learning
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error("Error getting model info: %s", e)
        return {
            "status": "error",
            "message": f"Error getting model info: {str(e)}",
            "timestamp": time.time()
        }

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )