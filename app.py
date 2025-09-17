"""
AI Agent FastAPI Application
Entry point for the AI Agent system with /chat endpoint
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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
                "enabled": True,
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
                    "top_p": settings.top_p
                }
            },
            interaction_config={},
            api_config={
                "enable_api_calls": True,
                "order_service_url": settings.order_service_url,
                "payment_service_url": settings.payment_service_url,
                "warranty_service_url": settings.warranty_service_url,
                "product_service_url": settings.product_service_url,
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
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down AI Agent application...")
        if router_instance:
            await router_instance.cleanup()

# Create FastAPI app
app = FastAPI(
    title="AI Agent API",
    description="Intelligent AI Agent with RAG, Interaction, and API calling capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    intent: Optional[str] = None  # search, chat, api_call

class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str

# Dependency to get router
async def get_router():
    if router_instance is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    return router_instance

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with detailed monitoring"""
    try:
        from monitoring.health_check import HealthChecker, HealthCheckConfig
        
        # Create health checker
        health_config = HealthCheckConfig()
        health_checker = HealthChecker(health_config)
        
        # Register application health check
        health_checker.register_check("application", health_checker.check_application_health)
        
        # Run all checks
        results = await health_checker.run_all_checks()
        summary = health_checker.get_health_summary()
        
        return {
            "status": summary["overall_status"],
            "health_score": summary["health_score"],
            "message": "AI Agent health check completed",
            "version": "1.0.0",
            "checks": summary["checks"],
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "health_score": 0,
            "message": f"Health check failed: {str(e)}",
            "version": "1.0.0",
            "timestamp": time.time()
        }

@app.post("/ask", response_model=ChatResponse)
async def ask(
    request: ChatRequest,
    router = Depends(get_router)
):
    """
    Main ask endpoint that processes user messages through the Agno router
    """
    try:
        logger.info(f"Received ask request: {request.message[:100]}...")
        
        # Process the request through the Agno router
        response = await router.process_request(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            context=request.context,
            intent=request.intent
        )
        
        logger.info(f"Generated response: {response['response'][:100]}...")
        
        # Collect conversation for training (async)
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
            
        except Exception as e:
            logger.warning(f"Failed to collect conversation for training: {e}")
        
        return ChatResponse(
            response=response["response"],
            intent=response["intent"],
            confidence=response["confidence"],
            metadata=response.get("metadata"),
            session_id=response.get("session_id")
        )
        
    except Exception as e:
        logger.error(f"Error processing ask request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    router = Depends(get_router)
):
    """
    Legacy chat endpoint - redirects to /ask
    """
    return await ask(request, router)

@app.get("/metrics")
async def get_metrics(router = Depends(get_router)):
    """Get Hybrid Orchestrator metrics"""
    try:
        metrics = router.get_metrics()
        return {
            "status": "success",
            "metrics": metrics,
            "orchestrator_type": "hybrid" if router.enable_hybrid else "rule_based"
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

@app.get("/dashboard")
async def get_dashboard(router = Depends(get_router)):
    """Get comprehensive monitoring dashboard"""
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
        health_checker.register_check("application", health_checker.check_application_health)
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
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting dashboard: {str(e)}")

@app.get("/traces")
async def get_traces(limit: int = 100):
    """Get recent traces"""
    try:
        from monitoring.tracing import tracer
        traces = tracer.get_completed_traces(limit)
        return {
            "status": "success",
            "traces": [trace.to_dict() for trace in traces],
            "count": len(traces)
        }
    except Exception as e:
        logger.error(f"Error getting traces: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting traces: {str(e)}")

# ===========================================
# TRAINING & FINE-TUNING ENDPOINTS
# ===========================================

@app.post("/training/start")
async def start_training(
    data_source: str = "dataset",
    auto_mode: bool = False
):
    """Start training pipeline"""
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
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")

@app.get("/training/status")
async def get_training_status():
    """Get current training status"""
    try:
        from training.training_pipeline import get_training_pipeline
        
        pipeline = get_training_pipeline()
        status = pipeline.get_training_status()
        
        return {
            "status": "success",
            "training_status": status
        }
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting training status: {str(e)}")

@app.get("/training/history")
async def get_training_history():
    """Get training history"""
    try:
        from training.training_pipeline import get_training_pipeline
        
        pipeline = get_training_pipeline()
        history = pipeline.get_training_history()
        
        return {
            "status": "success",
            "training_history": history
        }
        
    except Exception as e:
        logger.error(f"Error getting training history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting training history: {str(e)}")

@app.post("/training/collect")
async def collect_conversation(conversation: Dict[str, Any]):
    """Collect conversation data for training"""
    try:
        from training.training_pipeline import get_training_pipeline
        
        pipeline = get_training_pipeline()
        pipeline.collect_conversation(conversation)
        
        return {
            "status": "success",
            "message": "Conversation collected for training",
            "buffer_size": len(pipeline.conversation_buffer)
        }
        
    except Exception as e:
        logger.error(f"Error collecting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error collecting conversation: {str(e)}")

@app.post("/training/auto-retrain")
async def toggle_auto_retrain(enabled: bool = True):
    """Enable/disable auto-retrain"""
    try:
        from training.training_pipeline import get_training_pipeline
        
        pipeline = get_training_pipeline()
        pipeline.enable_auto_retrain(enabled)
        
        return {
            "status": "success",
            "message": f"Auto-retrain {'enabled' if enabled else 'disabled'}",
            "auto_retrain_enabled": enabled
        }
        
    except Exception as e:
        logger.error(f"Error toggling auto-retrain: {e}")
        raise HTTPException(status_code=500, detail=f"Error toggling auto-retrain: {str(e)}")

@app.post("/training/prepare-data")
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
        logger.error(f"Error preparing training data: {e}")
        raise HTTPException(status_code=500, detail=f"Error preparing training data: {str(e)}")

@app.post("/training/evaluate")
async def evaluate_model():
    """Evaluate current model"""
    try:
        from training.evaluate import ModelEvaluator
        
        evaluator = ModelEvaluator("training/checkpoints")
        
        # Load test data
        with open("training/dataset/test_conversations.json", 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Load model
        if not evaluator.load_model("training/checkpoints"):
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Evaluate model
        results = evaluator.evaluate_model(test_data)
        
        # Save results
        evaluator.save_evaluation_results(results, "training/evaluation_results.json")
        
        return {
            "status": "success",
            "message": "Model evaluation completed",
            "evaluation_results": results
        }
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise HTTPException(status_code=500, detail=f"Error evaluating model: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
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

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )