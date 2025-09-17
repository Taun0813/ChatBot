# AI Agent System - Hybrid Orchestrator

Hệ thống AI Agent thông minh với **Hybrid Orchestrator** kết hợp rule-based và ML-based routing, sử dụng dataset thực tế với 27,000+ sản phẩm điện thoại.

## 🚀 Tính năng chính

- **🤖 Hybrid Orchestrator**: Kết hợp rule-based + ML-based routing (accuracy 85-95%)
- **📊 Real Dataset**: 27,000+ sản phẩm điện thoại thực tế từ OnePlus, Samsung, Apple, Xiaomi, etc.
- **🔍 RAG (Retrieval-Augmented Generation)**: Semantic search với Pinecone vector database
- **💬 Hội thoại thông minh**: Tương tác tự nhiên với context-aware routing
- **🔌 Tích hợp API**: Kết nối với các dịch vụ bên ngoài (đơn hàng, thanh toán, bảo hành)
- **👤 Cá nhân hóa**: Học hỏi từ hành vi người dùng và đưa ra gợi ý phù hợp
- **🔄 Multi-model support**: Hỗ trợ nhiều LLM (Gemini, Groq, Ollama, OpenAI, Claude)
- **⚡ Caching**: Hệ thống cache thông minh với Redis và Memory cache
- **📈 Monitoring**: Theo dõi hiệu suất và metrics real-time với dashboard chi tiết
- **🎯 Training & Fine-tuning**: Fine-tune model cho domain e-commerce với data pipeline hoàn chỉnh

## 📁 Cấu trúc thư mục

```
ai_agent/
│── app.py                     # Entry point FastAPI
│── config.py                  # Configuration management
│── requirements.txt           # Python dependencies
│
├── core/                      # Core logic (Hybrid Orchestrator)
│   │── models/                # Agent models
│   │   │── __init__.py
│   │   │── base_agent.py      # Base agent class
│   │   │── rag_agent.py       # RAG-specific agent
│   │   │── conversation_agent.py # Conversation agent
│   │   │── api_agent.py       # API integration agent
│   │   │── orchestrator.py    # Agent orchestrator
│   │── router.py              # Hybrid Orchestrator (Rule-based + ML-based)
│   │── rag_model.py           # RAG model implementation
│   │── interaction_model.py   # Conversation model
│   │── api_model.py           # API model
│   │── personalization_model.py # Personalization model
│   │── prompts.py             # Prompt templates
│
├── adapters/                  # Adapter layer (plug-and-play)
│   │── model_loader/          # Model loaders
│   │   │── base_loader.py     # Base loader
│   │   │── gemini_loader.py   # Google Gemini
│   │   │── groq_loader.py     # Groq API
│   │   │── ollama_loader.py   # Ollama local
│   │   │── openai_loader.py   # OpenAI GPT
│   │   │── claude_loader.py   # Claude
│   │── pinecone_client.py     # Pinecone vector DB
│
├── cache/                     # Caching layer
│   │── redis_cache.py         # Redis cache
│   │── memory_cache.py        # In-memory cache
│   │── cache_manager.py       # Cache manager
│
├── monitoring/                # Monitoring & observability
│   │── metrics.py             # Metrics collection
│   │── health_check.py        # Health monitoring
│   │── tracing.py            # Request tracing
│
├── personalization/           # Personalization layer
│   │── profile_manager.py     # User profile management
│   │── recommender.py         # Product recommendations
│   │── rl_feedback.py         # Reinforcement learning
│
├── services/                  # Microservices integration
│   │── product_service.py     # Product API
│   │── order_service.py       # Order API
│   │── payment_service.py     # Payment API
│   │── warranty_service.py    # Warranty API
│   │── mock/                  # Mock services
│   │   │── mock_order.json
│   │   │── mock_warranty.json
│   │   │── mock_payment.json
│
├── data/                      # Data management
│   │── ingest.py              # Data ingestion (supports real dataset)
│   │── process_dataset.py     # Dataset processing
│   │── processed/             # Processed data
│   │── profiles/              # User profiles
│   │── schema/                # Data schemas
│
├── training/                  # Model training & fine-tuning
│   │── dataset/               # Real dataset
│   │   │── dataset.json       # 27,000+ real phone products
│   │── prepare_data.py        # E-commerce data preparation & normalization
│   │── finetune.py           # Model fine-tuning với PEFT/LoRA
│   │── evaluate.py           # Comprehensive model evaluation
│   │── checkpoints/          # Trained model checkpoints
│
├── utils/                     # Utilities
│   │── logger.py              # Logging utilities
│   │── helpers.py             # Helper functions
│
└── tests/                     # Unit tests
    │── test_router.py
    │── test_rag_model.py
    │── test_interaction_model.py
    │── test_api_model.py
    │── test_personalization.py
```

## 🛠️ Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd ai_agent
```

### 2. Tạo virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies

#### Tùy chọn 1: Cài đặt đầy đủ (khuyến nghị)
```bash
pip install -r requirements.txt
```

#### Tùy chọn 2: Cài đặt tối thiểu (chỉ core features)
```bash
pip install fastapi uvicorn pydantic pydantic-settings google-generativeai pinecone-client redis httpx python-dotenv psutil
```

#### Tùy chọn 3: Cài đặt cho development
```bash
pip install -r requirements.txt black isort flake8 mypy pytest pytest-asyncio pytest-cov
```

### 4. Cấu hình environment
```bash
cp env.example .env
# Chỉnh sửa .env với API keys của bạn
# QUAN TRỌNG: Cần có PINECONE_API_KEY để sử dụng vector database
```

### 5. Khởi tạo dữ liệu với dataset thực tế
```bash
# Load 27,000+ sản phẩm điện thoại thực tế vào Pinecone
python init_data.py
```

### 6. Chạy ứng dụng
```bash
python app.py
```

## 📦 Requirements

### Các file requirements

1. **`requirements.txt`** - Cài đặt đầy đủ (khuyến nghị)
   - Tất cả AI APIs (Gemini, Groq, Ollama, OpenAI, Claude)
   - Vector database (Pinecone cloud only)
   - Caching (Redis, Memory cache)
   - Monitoring & observability
   - Personalization & ML
   - Development tools

2. **`requirements-minimal.txt`** - Cài đặt tối thiểu
   - Chỉ core features cần thiết
   - Free APIs (Gemini, Groq, Ollama)
   - FAISS vector database
   - Redis caching
   - Kích thước: ~500MB

3. **`requirements-dev.txt`** - Development
   - Bao gồm tất cả requirements.txt
   - Testing tools (pytest, coverage)
   - Code quality (black, flake8, mypy)
   - Debugging tools
   - Documentation tools

### So sánh kích thước cài đặt

| File | Kích thước | Thời gian cài đặt | Tính năng |
|------|------------|-------------------|-----------|
| requirements-minimal.txt | ~500MB | 2-3 phút | Core only |
| requirements.txt | ~2GB | 5-10 phút | Full features |
| requirements-dev.txt | ~2.5GB | 8-15 phút | Full + Dev tools |

## 🔧 Cấu hình

### API Keys (Miễn phí)
- **Gemini API**: Lấy từ [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Groq API**: Lấy từ [Groq Console](https://console.groq.com/)
- **Ollama**: Cài đặt local từ [Ollama.ai](https://ollama.ai/)

### Environment Variables
```bash
# Free APIs
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
OLLAMA_BASE_URL=http://localhost:11434

# Optional APIs
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Configuration
MODEL_LOADER_BACKEND=gemini  # gemini, groq, ollama, openai, claude
ENABLE_PERSONALIZATION=true
ENABLE_RECOMMENDATIONS=true
ENABLE_RL_LEARNING=true
```

## 🚀 Sử dụng

### API Endpoints

#### 1. Main Chat endpoint (Hybrid Orchestrator)
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "OnePlus dưới 50 triệu",
    "user_id": "user123",
    "session_id": "session001"
  }'
```

#### 2. Product Search (từ dataset thực tế)
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Samsung Galaxy camera 50MP",
    "user_id": "user123",
    "session_id": "session001"
  }'
```

#### 3. Order Tracking
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Đơn hàng #1234 đang ở đâu?",
    "user_id": "user123",
    "session_id": "session001"
  }'
```

#### 4. Health check
```bash
curl http://localhost:8000/health
```

#### 5. Hybrid Orchestrator Metrics
```bash
curl http://localhost:8000/metrics
```

#### 6. Monitoring Dashboard (NEW)
```bash
curl http://localhost:8000/dashboard
```

#### 7. Request Traces (NEW)
```bash
curl http://localhost:8000/traces
```

#### 8. Training & Fine-tuning (NEW)
```bash
# Start training pipeline
curl -X POST "http://localhost:8000/training/start" \
  -H "Content-Type: application/json" \
  -d '{"data_source": "dataset", "auto_mode": false}'

# Get training status
curl http://localhost:8000/training/status

# Get training history
curl http://localhost:8000/training/history

# Prepare training data
curl -X POST http://localhost:8000/training/prepare-data

# Evaluate model
curl -X POST http://localhost:8000/training/evaluate

# Toggle auto-retrain
curl -X POST "http://localhost:8000/training/auto-retrain" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```

#### 9. System Information
```bash
curl http://localhost:8000/
```

### Python SDK
```python
import asyncio
from core.router import AgnoRouter, RouterConfig

async def main():
    config = RouterConfig(
        rag_config={},
        interaction_config={},
        api_config={}
    )
    
    router = AgnoRouter(config)
    await router.initialize()
    
    response = await router.process_request(
        message="Xin chào, tôi cần tư vấn về điện thoại",
        user_id="user123"
    )
    
    print(response["response"])
    
    await router.cleanup()

asyncio.run(main())
```

## 🎯 Tính năng nâng cao

### 1. Cá nhân hóa người dùng
- Học hỏi từ lịch sử mua hàng
- Gợi ý sản phẩm phù hợp
- Reinforcement Learning từ feedback

### 2. Hybrid Orchestrator Architecture
- **Rule-based Router**: Fast, deterministic routing với pattern matching
- **ML-based Router**: Context-aware routing với intent classification
- **Decision Fusion Engine**: Kết hợp decisions với adaptive weights
- **RAG Agent**: Xử lý tìm kiếm sản phẩm từ dataset thực tế
- **Conversation Agent**: Hội thoại chung với context awareness
- **API Agent**: Tích hợp dịch vụ bên ngoài
- **Performance Tracking**: Real-time metrics và monitoring

### 3. Real Dataset Integration
- **27,000+ sản phẩm điện thoại thực tế** từ OnePlus, Samsung, Apple, Xiaomi, Motorola, Realme, Nothing
- **Thông số chi tiết**: CPU, RAM, ROM, camera, pin, màn hình, 5G, NFC, sạc nhanh
- **Giá cả thực tế**: Từ 19,989 VND đến hàng triệu VND
- **Rating và reviews** từ người dùng thực tế
- **Auto-conversion**: Tự động convert format để phù hợp với RAG system

### 4. Caching thông minh
- Redis cache cho production
- Memory cache cho development
- Cache responses và embeddings
- TTL và invalidation

### 5. Monitoring & Observability (Phase 6)
- **Enhanced Metrics System**: API latency, query counts, success/failure rates
- **Comprehensive Health Checks**: System resources, application health, load balancer support
- **Request Tracing**: OpenTelemetry/Jaeger integration với span tracking
- **Monitoring Dashboard**: Real-time performance visualization với `/dashboard` endpoint
- **Hybrid Orchestrator Metrics**: Rule-based vs ML-based vs hybrid performance tracking

### 6. Training & Fine-tuning (Phase 7)
- **E-commerce Data Pipeline**: Conversation normalization, intent detection, entity extraction
- **Model Fine-tuning**: TinyLlama + PEFT/LoRA cho domain e-commerce
- **Comprehensive Evaluation**: BLEU, ROUGE, intent accuracy, semantic similarity
- **Synthetic Data Generation**: Tăng cường training data với variations
- **Continuous Improvement**: Model retraining từ conversation data

## 🧪 Testing

```bash
# Chạy tất cả tests
pytest

# Chạy test cụ thể
pytest tests/test_router.py

# Chạy với coverage
pytest --cov=core tests/
```

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Hybrid Orchestrator Metrics
```bash
curl http://localhost:8000/metrics
```

**Expected Response:**
```json
{
  "status": "success",
  "metrics": {
    "total_requests": 1000,
    "rule_based_requests": 200,
    "ml_based_requests": 300,
    "hybrid_requests": 500,
    "average_response_time": 145.2,
    "rule_based_percentage": 20.0,
    "ml_based_percentage": 30.0,
    "hybrid_percentage": 50.0
  },
  "orchestrator_type": "hybrid"
}
```

### Monitoring Dashboard (NEW)
```bash
curl http://localhost:8000/dashboard
```

**Expected Response:**
```json
{
  "status": "success",
  "timestamp": 1703123456.789,
  "dashboard": {
    "system_health": {
      "overall_status": "healthy",
      "health_score": 95.5,
      "uptime": 3600,
      "memory_usage_mb": 512.3,
      "cpu_usage_percent": 45.2
    },
    "performance_metrics": {
      "total_requests": 1000,
      "success_rate": 98.5,
      "error_rate": 1.5,
      "average_response_time": 145.2,
      "avg_rag_time": 89.3,
      "avg_conversation_time": 67.8,
      "avg_api_time": 234.1
    },
    "query_breakdown": {
      "total_queries": 1000,
      "rag_queries": 400,
      "conversation_queries": 350,
      "api_queries": 250,
      "rag_error_rate": 0.5,
      "conversation_error_rate": 1.2,
      "api_error_rate": 2.1
    },
    "router_performance": {
      "rule_based_requests": 200,
      "ml_based_requests": 300,
      "hybrid_requests": 500,
      "rule_based_percentage": 20.0,
      "ml_based_percentage": 30.0,
      "hybrid_percentage": 50.0
    },
    "tracing": {
      "active_traces": 5,
      "completed_traces": 995,
      "average_duration": 145.2,
      "max_duration": 2000.0,
      "min_duration": 50.0
    }
  }
}
```

### Tracing
```bash
curl http://localhost:8000/traces
```

## 📋 Postman Testing

Xem file `DATASET_AND_POSTMAN_GUIDE.md` để có hướng dẫn chi tiết về testing với Postman, bao gồm:

- **10+ test cases** với dataset thực tế
- **Product search** với OnePlus, Samsung, Apple, Xiaomi, etc.
- **Order tracking** và API integration
- **Personalization testing** với user preferences
- **Hybrid Orchestrator testing** với rule-based vs ML-based routing
- **Performance testing** với cache và concurrent requests

### Quick Postman Tests

```bash
# Test OnePlus từ dataset thực tế
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "OnePlus dưới 50 triệu", "user_id": "user123"}'

# Test Samsung Galaxy
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "Samsung Galaxy camera 50MP", "user_id": "user123"}'

# Test Nothing Phone
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "Nothing Phone giá rẻ", "user_id": "user123"}'
```

## 🔄 Development

### Code Style
```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .
```

### Pre-commit hooks
```bash
pip install pre-commit
pre-commit install
```

## 📈 Performance

### Caching
- Response caching giảm 80% thời gian phản hồi
- Embedding caching tăng tốc RAG
- Redis cluster cho high availability

### Scaling
- Horizontal scaling với multiple instances
- Load balancing
- Database sharding
- CDN cho static assets

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 🆘 Support

- Issues: [GitHub Issues](https://github.com/your-repo/issues)
- Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)
- Email: support@your-domain.com

## 🙏 Acknowledgments

- Google Gemini API
- Groq API
- Ollama
- FastAPI
- Pydantic
- Redis
- Pinecone (Cloud Vector Database)