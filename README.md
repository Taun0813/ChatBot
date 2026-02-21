# AI Agent System - Hybrid Orchestrator

Intelligent AI Agent system for e-commerce with **Hybrid Orchestrator** combining rule-based and ML-based routing. Hỗ trợ 900+ điện thoại và đa danh mục (Laptop, Tablet, Phụ kiện).

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-5.0.1+-orange.svg)](https://pinecone.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Mục lục (Table of Contents)

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Luồng hoạt động của hệ thống](#luồng-hoạt-động-của-hệ-thống-system-workflow)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage & API Endpoints](#usage)
- [Advanced Features](#advanced-features)
- [FAQ](#faq)
- [Documentation](#documentation)

## Key Features

- **Hybrid Orchestrator**: Combines rule-based + ML-based routing (85-95% accuracy)
- **Multi-category Dataset**: Điện thoại, Laptop, Tablet, Phụ kiện (CSV + JSON)
- **RAG System**: Semantic search with Pinecone (bật qua `RAG_ENABLED=true`)
- **Smart Conversation**: Natural interaction, fallback khi RAG tắt
- **API Integration**: Spring Boot microservices (orders, payments, warranty) qua `ENABLE_API_CALLS`
- **Personalization**: User behavior, recommendations (tùy chọn)
- **Multi-model**: Support multiple LLMs (Gemini 0.8.3+, Groq 0.9.0+, Ollama 0.4.2+, OpenAI 1.58.1+, Claude 0.40.0+)
- **Caching**: Smart caching system with Redis 5.2.1+ and Memory cache
- **Monitoring**: Real-time performance monitoring with detailed dashboard
- **Training**: Fine-tune models for e-commerce domain with complete data pipeline
- **Production Ready**: FastAPI 0.115.6+, PyTorch 2.5.1+, modern async/await patterns

## System Architecture

### Hybrid Orchestrator Architecture
```mermaid
graph TB
    A[Client Request] --> B[FastAPI App]
    B --> C[AgnoRouter - Hybrid Orchestrator]
    
    C --> D[Rule-based Router]
    C --> E[ML-based Router]
    
    D --> F[Pattern Matching]
    E --> G[Intent Classification]
    
    F --> H[Decision Fusion Engine]
    G --> H
    
    H --> I{Intent Decision}
    
    I -->|search| J[RAG Agent]
    I -->|chat| K[Conversation Agent]
    I -->|api| L[API Agent]
    
    J --> M[Pinecone Vector Search]
    M --> N[Product Results]
    N --> O[Personalization]
    O --> P[Natural Language Response]
    
    K --> Q[LLM Model]
    Q --> R[Context-aware Response]
    
    L --> S[External APIs]
    S --> T[API Response]
    
    P --> U[Cache Manager]
    R --> U
    T --> U
    
    U --> V[Response to Client]
```

## Luồng hoạt động của hệ thống (System Workflow)

### 1. Tổng quan luồng request

Mỗi request từ client đi qua các bước sau:

```
Client (POST /ask hoặc /chat)
    → FastAPI app (app.py)
    → Chuẩn hóa user_id (nếu null/trống → "anonymous")
    → AgnoRouter.process_request(message, user_id, session_id, context, intent)
    → [Nếu intent đã cho sẵn] _process_with_intent(message, intent, ...)
    → [Nếu chưa có intent] Hybrid: _get_rule_decision + _get_ml_decision → fuse_decisions → intent
    → _process_with_intent(message, intent_final, ...)
        → intent = "search"  → _handle_search_request
        → intent = "order"  → _handle_order_request
        → intent = "api"    → _handle_api_request
        → intent = "chat"   → _handle_chat_request
    → response["user_id"] = user_id, response["session_id"] = session_id
    → ChatResponse(user_id=user_id, response=..., intent=..., confidence=..., metadata=..., session_id=...)
    → JSON trả về client
```

- **user_id**: Bắt buộc trong response. Nếu client không gửi `user_id` hoặc gửi `null`, hệ thống dùng `"anonymous"` (trong `app.py` và `core/router.py`).

### 2. Luồng Search (RAG – tìm kiếm sản phẩm)

Khi intent = **search**:

1. **RAG tắt** (`RAG_ENABLED=false`): gọi `_handle_search_fallback` → LLM trả lời chung, không search thật.
2. **RAG bật**:
   - **Cache**: Kiểm tra cache theo `{ type: "search", query, user_id }`. Nếu có → trả về kết quả đã cache (TTL 30 phút).
   - **RAG search**: `rag_model.search_products(query=message, user_id=user_id, top_k=5)`:
     - **Trích metadata từ câu** (`_extract_metadata_from_query`):
       - Giá: "dưới X triệu", "từ X đến Y triệu", "khoảng X triệu" → `price_range` (VND).
       - Thương hiệu: từ khóa "iphone"/"apple" → **Apple**, "samsung" → Samsung, ... (ánh xạ đúng tên trong dataset/Pinecone; ví dụ iPhone → Apple).
       - Spec: "pin trâu", "pin khỏe", "camera tốt", "RAM X GB", ... → `specs`.
     - **Embedding**: Tạo query embedding (Pinecone managed model, `input_type="passage"`).
     - **Pinecone**: `search_products(query_vector, top_k, price_range, brand, category)` — filter metadata (brand, price, category) + vector similarity.
     - **Xử lý kết quả**: `_process_search_results` (format, similarity_score, relevance_score).
     - **Lọc theo spec**: Nếu có `specs` (pin, camera, ...) → `_filter_by_specs`.
     - **Relaxed search**: Nếu không còn sản phẩm nào mà vẫn có filter (giá/thương hiệu/spec) → gọi lại Pinecone **không** filter (chỉ vector) → lấy thêm kết quả → lọc spec cơ bản.
   - **Cá nhân hóa** (nếu bật): `personalization_model.record_user_interaction` (search, query) rồi `get_personalized_recommendations` để sắp xếp/giới hạn kết quả.
   - **Tạo câu trả lời**: `interaction_model.generate_search_response(query, search_results, user_id, context)` → văn bản tự nhiên từ danh sách sản phẩm.
   - **Cache**: Lưu kết quả vào cache (nếu có CacheManager).
   - **Trả về**: `{ response, intent: "search", confidence: 0.9, metadata: { search_results, results_count, model_used: "rag", personalized, cached } }`.

Dữ liệu sản phẩm trong Pinecone đến từ **init_data.py** (CSV/JSON → transform → upsert). File `data/processed/products_export.json` được tạo khi export từ init_data; **ProductService** (nếu dùng) đọc file này; **luồng search RAG không đọc file JSON** mà chỉ dùng Pinecone.

### 3. Luồng Chat (hội thoại chung)

- Intent = **chat** → `_handle_chat_request(message, user_id, context)`.
- Gọi `interaction_model.generate_response(message, user_id, context)` (LLM, có thể dùng context/session).
- Trả về response dạng hội thoại, không có `search_results`.

### 4. Luồng Order / API

- **Order**: Intent = **order** → `_handle_order_request` → gọi API model (tra cứu đơn hàng, v.v.) nếu `ENABLE_API_CALLS=true`.
- **API**: Intent = **api** → `_handle_api_request` → tích hợp microservices (order, payment, warranty, product) theo config.

### 5. Hybrid Orchestrator (khi enable_hybrid = true)

- **Rule-based**: Pattern matching trên message → intent (search, order, api, chat) + confidence.
- **ML-based**: Feature extraction (product/order/price mentions, session, …) → intent classification (LLM/Model) → intent + confidence.
- **Fusion**: Kết hợp rule + ML (fusion_weights, adaptive) → intent cuối cùng → xử lý theo intent như trên.

### 6. Tóm tắt thành phần chính

| Thành phần | Vai trò |
|------------|--------|
| **app.py** | Entry FastAPI: lifespan khởi tạo/dọn AgnoRouter, endpoint `/ask`, `/chat`; chuẩn hóa `user_id` → `"anonymous"` nếu null/trống. |
| **core/router.py** | AgnoRouter: process_request, hybrid routing (rule + ML), fusion, gọi _handle_search_request / _handle_chat_request / _handle_order_request / _handle_api_request; cache get/set cho search. |
| **core/rag_model.py** | RAG: search_products (trích metadata query, embedding, Pinecone search, filter_by_specs, relaxed search), _extract_metadata_from_query (brand map iPhone→Apple, giá, spec), _filter_by_specs. |
| **adapters/pinecone_client.py** | Pinecone: search_products với filter metadata (brand, price, category) + vector similarity. |
| **core/interaction_model.py** | Tạo câu trả lời: generate_search_response (từ search_results), generate_response (chat). |
| **core/personalization_model.py** | Cá nhân hóa (khi bật): record_user_interaction, get_personalized_recommendations; profile từ JSON/SQLite. |
| **cache/** (optional) | CacheManager: cache kết quả search (TTL 30 phút) để giảm gọi RAG/Pinecone. |
| **init_data.py** | Load CSV/JSON → transform → upsert Pinecone, export `data/processed/products_export.json`. |
| **services/product_service.py** | Đọc `data/processed/products_export.json` (dùng khi có Product API riêng); luồng RAG search không đọc file này. |

---

## Directory Structure

```
ai_agent/
├── app.py                        # FastAPI entry point
├── config.py                     # Configuration management
├── requirements.txt              # Python dependencies
├── env.example                   # Environment variables template
├── init_data.py                  # Data initialization (CSV + JSON)
├── dockerfile                    # Docker build
├── docker-compose.yml            # Docker Compose (AI Agent + Redis)
├── railway.json                  # Railway deployment config
├── DEPLOYMENT.md                 # Hướng dẫn deploy chi tiết
├── ECOMMERCE_AI_AGENT_ROADMAP.md # Roadmap E-commerce
│
├── core/                         # Core logic (Hybrid Orchestrator)
│   ├── models/                   # Agent models
│   │   ├── base_agent.py         # Base agent class
│   │   ├── rag_agent.py          # RAG-specific agent
│   │   ├── conversation_agent.py # Conversation agent
│   │   ├── api_agent.py          # API integration agent
│   │   └── orchestrator.py       # Agent orchestrator
│   ├── router.py                 # Hybrid Orchestrator
│   ├── rag_model.py              # RAG model implementation
│   ├── interaction_model.py      # Conversation model
│   ├── api_model.py              # API model
│   ├── personalization_model.py  # Personalization model
│   └── prompts.py                # Prompt templates
│
├── adapters/                     # Adapter layer
│   ├── model_loader/             # Model loaders
│   │   ├── base_loader.py        # Base loader
│   │   ├── gemini_loader.py      # Google Gemini
│   │   ├── groq_loader.py        # Groq API
│   │   ├── ollama_loader.py      # Ollama local
│   │   └── openai_loader.py      # OpenAI GPT
│   └── pinecone_client.py        # Pinecone vector DB
│
├── cache/                        # Caching layer (optional - router dùng khi có CacheManager)
│   ├── redis_cache.py            # Redis cache
│   ├── memory_cache.py           # In-memory cache
│   └── cache_manager.py          # Cache manager
│
├── monitoring/                   # Monitoring & observability
│   ├── metrics.py                # Metrics collection
│   ├── health_check.py           # Health monitoring
│   └── tracing.py                # Request tracing
│
├── personalization/              # Personalization layer
│   ├── profile_manager.py        # User profile management
│   ├── recommender.py            # Product recommendations
│   └── rl_feedback.py            # Reinforcement learning
│
├── services/                     # Microservices integration
│   ├── product_service.py        # Product API
│   ├── order_service.py          # Order API
│   ├── payment_service.py        # Payment API
│   ├── warranty_service.py       # Warranty API
│   └── mock/                     # Mock services
│       ├── mock_order.json
│       ├── mock_warranty.json
│       └── mock_payment.json
│
├── data/                         # Data management
│   ├── ingest.py                 # Data ingestion
│   ├── process_dataset.py        # Dataset processing
│   ├── processed/                # Processed data
│   │   ├── products_export.json  # Export từ init_data (ProductService đọc; RAG dùng Pinecone)
│   │   ├── sample_products_extra.json  # Mẫu Laptop, Tai nghe, Sạc
│   │   ├── knowledge_base.json   # Knowledge base
│   │   └── training_data.json    # Training data
│   ├── profiles/                # User profiles (JSON/SQLite khi bật personalization)
│   └── schema/                   # Product schemas (đa danh mục)
│
├── training/                     # Model training & fine-tuning
│   ├── dataset/                  # Training dataset
│   │   └── dataset.json          # Training conversations
│   ├── prepare_data.py           # Data preparation
│   ├── finetune.py               # Model fine-tuning
│   ├── evaluate.py               # Model evaluation
│   └── training_pipeline.py      # Training pipeline
│
└── utils/                        # Utilities
    ├── logger.py                 # Logging utilities
    └── helpers.py                # Helper functions
```

## Quick Start

```bash
git clone <repository-url>
cd ai-agent
pip install -r requirements.txt
cp env.example .env   # Điền GEMINI_API_KEY
python app.py         # http://localhost:8000
curl http://localhost:8000/health
```

---

## Installation

### 1. Clone repository
```bash
git clone <repository-url>
cd ai_agent
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

#### Option 1: Full installation (recommended)
```bash
pip install -r requirements.txt
```

#### Option 2: Minimal installation (core features only)
```bash
pip install fastapi==0.115.6 uvicorn[standard]==0.32.1 pydantic==2.10.4 pydantic-settings==2.7.0 google-generativeai==0.8.3 pinecone-client==5.0.1 redis[hiredis]==5.2.1 httpx==0.28.1 python-dotenv==1.0.1 psutil==6.1.0
```

#### Option 3: Development installation
```bash
pip install -r requirements.txt black==24.10.0 isort==5.13.2 flake8==7.1.1 mypy==1.13.0 pytest==8.3.4 pytest-asyncio==0.24.0 pytest-cov==6.0.0
```

#### Option 4: Production installation
```bash
pip install fastapi==0.115.6 uvicorn[standard]==0.32.1 gunicorn==23.0.0 redis[hiredis]==5.2.1 pinecone-client==5.0.1 google-generativeai==0.8.3
```

### 4. Configure environment
```bash
cp env.example .env
# Chỉ cần GEMINI_API_KEY hoặc GROQ_API_KEY để chạy
# PINECONE_API_KEY chỉ cần khi RAG_ENABLED=true
```

### 5. Run application
```bash
python app.py
# App chạy được ngay (RAG tắt mặc định, dùng conversation fallback cho search)
```

### 6. (Optional) Bật RAG - Load sản phẩm lên Pinecone
```bash
# Trong .env: RAG_ENABLED=true, điền PINECONE_API_KEY
# Load điện thoại từ CSV
python init_data.py

# Hoặc load sản phẩm Laptop/Tablet/Phụ kiện từ JSON
python init_data.py data/processed/sample_products_extra.json
```

### 7. (Optional) Docker
```bash
docker-compose up -d
# Hoặc: docker build -f dockerfile -t ai-agent:v1 .
# Chi tiết: xem DEPLOYMENT.md
```

## Training & Fine-tuning (Optional)

### Prepare training data
```bash
python training/prepare_data.py
```

### Fine-tune model
```bash
python training/finetune.py
```

### Evaluate model
```bash
python training/evaluate.py
```

**Note**: Training is only necessary when you want to improve the model. The system works normally without training.

## Requirements

### Requirements files

1. **`requirements.txt`** - Full installation (recommended)
   - All AI APIs (Gemini 0.8.3+, Groq 0.9.0+, Ollama 0.4.2+, OpenAI 1.58.1+, Claude 0.40.0+)
   - Vector database (Pinecone 5.0.1+ cloud only)
   - Caching (Redis 5.2.1+, Memory cache)
   - Monitoring & observability (Prometheus, OpenTelemetry)
   - Personalization & ML (PyTorch 2.5.1+, Transformers 4.47.1+)
   - Development tools (Black 24.10.0+, pytest 8.3.4+)
   - Production server (Gunicorn 23.0.0+)

### Installation size comparison

| Installation Type | Size | Installation Time | Features |
|-------------------|------|-------------------|----------|
| Minimal | ~800MB | 3-5 minutes | Core APIs only |
| Full | ~3GB | 8-15 minutes | All features |
| Development | ~3.5GB | 10-20 minutes | Full + Dev tools |
| Production | ~1.2GB | 5-8 minutes | Production optimized |

### Version Compatibility

- **Python**: 3.10+ (recommended: 3.11+)
- **FastAPI**: 0.115.6+ (latest stable)
- **Pydantic**: 2.10.4+ (v2 only)
- **PyTorch**: 2.5.1+ (CUDA 12.1+ supported)
- **Transformers**: 4.47.1+ (latest)
- **Pinecone**: 5.0.1+ (latest API)

## Configuration

### API Keys (Free)
- **Gemini API**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey) (v0.8.3+)
- **Groq API**: Get from [Groq Console](https://console.groq.com/) (v0.9.0+)
- **Ollama**: Install locally from [Ollama.ai](https://ollama.ai/) (v0.4.2+)

### Environment Variables
```bash
# API Keys (chọn 1 trong các key miễn phí)
GEMINI_API_KEY=your_gemini_api_key   # Khuyến nghị
GROQ_API_KEY=your_groq_api_key
OLLAMA_BASE_URL=http://localhost:11434

# Optional Paid APIs
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
COHERE_API_KEY=your_cohere_api_key

# Model
MODEL_LOADER_BACKEND=gemini
MODEL_NAME=gemini-2.5-flash

# Phase 1 - RAG & API (E-commerce)
RAG_ENABLED=false                    # Bật khi đã có Pinecone + init_data
ENABLE_API_CALLS=false               # Bật khi đã có Spring Boot backend

# Pinecone (chỉ cần khi RAG_ENABLED=true)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=product-search
PINECONE_DIMENSION=1024

# Spring Boot Services (khi ENABLE_API_CALLS=true)
ORDER_SERVICE_URL=http://localhost:8181/api/orders
PRODUCT_SERVICE_URL=http://localhost:8181/api/products
PAYMENT_SERVICE_URL=http://localhost:8181/api/payments

# Personalization (tùy chọn)
ENABLE_PERSONALIZATION=false
ENABLE_RECOMMENDATIONS=false
```

**Lưu ý**: `user_id` trong request là tùy chọn; nếu không gửi hoặc null, response vẫn trả về `user_id` với giá trị `"anonymous"`.

## Usage

### API Endpoints

#### 1. Main Chat endpoint (Hybrid Orchestrator)
- **Body**: `message` (bắt buộc), `user_id` (tùy chọn, mặc định `"anonymous"`), `session_id`, `context`, `intent` (tùy chọn).
- **Response**: Luôn có `user_id` (string). Nếu không gửi `user_id` thì server trả về `"anonymous"`.

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "OnePlus under 50 million",
    "user_id": "user123",
    "session_id": "session001"
  }'
```

Gửi không có `user_id` (vẫn hợp lệ):
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"message": "Điện thoại iPhone pin trâu"}'
# response.user_id sẽ là "anonymous"
```

#### 2. Product Search (from real dataset)
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
    "message": "Where is order #1234?",
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
        rag_config={"enabled": False, "pinecone_config": {}, "model_loader_config": {}},
        interaction_config={},
        api_config={"enable_api_calls": False},
        personalization_config={"enable_personalization": False},
        hybrid_config={"enable_hybrid": True}
    )
    router = AgnoRouter(config)
    await router.initialize()
    response = await router.process_request(
        message="Hello, I need advice about phones",
        user_id="user123"
    )
    print(response["response"])
    await router.cleanup()

asyncio.run(main())
```

## Advanced Features

### 1. User Personalization
- Learn from purchase history
- Suggest relevant products
- Reinforcement Learning from feedback

### 2. Hybrid Orchestrator Architecture
- **Rule-based Router**: Fast, deterministic routing with pattern matching
- **ML-based Router**: Context-aware routing with intent classification
- **Decision Fusion Engine**: Combine decisions with adaptive weights
- **RAG Agent**: Process product search from real dataset
- **Conversation Agent**: General conversation with context awareness
- **API Agent**: External service integration
- **Performance Tracking**: Real-time metrics and monitoring

### 3. Multi-category Dataset & RAG Search
- **Điện thoại**: 900+ sản phẩm (`Mobiles Dataset (2025).csv`) - Apple, Samsung, OnePlus, Xiaomi, etc.
- **Laptop, Tablet, Phụ kiện**: Hỗ trợ JSON (`data/processed/sample_products_extra.json`)
- **Schema**: `data/schema/product_schema.py` - Điện thoại, Laptop, Tablet, Tai nghe, Sạc dự phòng, ...
- **Init**: `python init_data.py [file.csv|file.json]` - Tự động detect format
- **RAG brand mapping**: Trong dataset/Pinecone, iPhone lưu với brand **Apple**. Hệ thống tự map "iphone"/"apple" → Apple khi trích metadata từ query (trong `core/rag_model.py`).
- **Spec từ khóa**: "pin trâu", "pin khỏe", "camera tốt", "RAM X GB", ... được trích và dùng để lọc/ưu tiên kết quả (và relaxed search khi filter quá chặt).

### 4. Smart Caching
- Redis cache for production (v5.2.1+)
- Memory cache for development
- Cache responses and embeddings
- TTL and invalidation

### 5. Monitoring & Observability
- **Enhanced Metrics System**: API latency, query counts, success/failure rates
- **Comprehensive Health Checks**: System resources, application health, load balancer support
- **Request Tracing**: OpenTelemetry 1.28.0+ integration with span tracking
- **Monitoring Dashboard**: Real-time performance visualization with `/dashboard` endpoint
- **Hybrid Orchestrator Metrics**: Rule-based vs ML-based vs hybrid performance tracking

### 6. Training & Fine-tuning
- **E-commerce Data Pipeline**: Conversation normalization, intent detection, entity extraction
- **Model Fine-tuning**: PyTorch 2.5.1+ + PEFT 0.15.0+ for e-commerce domain
- **Comprehensive Evaluation**: BLEU, ROUGE, intent accuracy, semantic similarity
- **Synthetic Data Generation**: Enhance training data with variations
- **Continuous Improvement**: Model retraining from conversation data

### 7. Phase 1 E-commerce (2025)
- **RAG_ENABLED / ENABLE_API_CALLS**: Cấu hình qua env, chạy được ngay không cần Pinecone
- **Spring Boot Integration**: URLs qua config, mock fallback khi API tắt
- **Multi-category**: Laptop, Tablet, Phụ kiện qua JSON
- **Docker**: dockerfile + docker-compose, deploy Railway

## Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_router.py

# Run with coverage
pytest --cov=core tests/
```

## Monitoring

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

## Testing with Postman

### Test Cases with Real Dataset

#### 1. **Product Search Tests**
```bash
# Test OnePlus from real dataset
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "OnePlus under 50 million", "user_id": "user123"}'

# Test Samsung Galaxy
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "Samsung Galaxy camera 50MP", "user_id": "user123"}'

# Test Nothing Phone
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "Nothing Phone cheap", "user_id": "user123"}'

# Test Apple iPhone
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "iPhone 15 Pro Max 256GB", "user_id": "user123"}'
```

#### 2. **Conversation Tests**
```bash
# Test general conversation
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, can you help me?", "user_id": "user123"}'

# Test product consultation
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "I need a phone with good camera", "user_id": "user123"}'
```

#### 3. **API Integration Tests**
```bash
# Test order tracking
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is order #1234?", "user_id": "user123"}'

# Test payment
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to pay for my order", "user_id": "user123"}'
```

#### 4. **Performance Tests**
```bash
# Test health check
curl http://localhost:8000/health

# Test metrics
curl http://localhost:8000/metrics

# Test dashboard
curl http://localhost:8000/dashboard
```

### Postman Collection

Create Postman collection with the following requests:

1. **Environment Variables**:
   - `base_url`: `http://localhost:8000`
   - `user_id`: `user123`
   - `session_id`: `session001`

2. **Request Templates**:
   ```json
   {
     "message": "{{message}}",
     "user_id": "{{user_id}}",
     "session_id": "{{session_id}}",
     "context": {}
   }
   ```

3. **Test Scripts** (in Postman Tests tab):
   ```javascript
   pm.test("Status code is 200", function () {
       pm.response.to.have.status(200);
   });
   
   pm.test("Response has required fields", function () {
       const jsonData = pm.response.json();
       pm.expect(jsonData).to.have.property('response');
       pm.expect(jsonData).to.have.property('intent');
       pm.expect(jsonData).to.have.property('confidence');
   });
   ```

## Development

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

## Performance

### Caching
- Response caching reduces 80% response time
- Embedding caching speeds up RAG
- Redis cluster for high availability

### Scaling
- Horizontal scaling with multiple instances
- Load balancing
- Database sharding
- CDN for static assets

## FAQ

### Q: Request không gửi user_id có lỗi không?
A: Không. Nếu không gửi `user_id` hoặc gửi `null`, hệ thống dùng `"anonymous"` và response vẫn trả về đúng format (user_id là string).

### Q: Tìm "iPhone pin trâu" nhưng ra Vivo/Oppo?
A: Đảm bảo đã bật RAG và index Pinecone từ CSV. Hệ thống map "iphone" → brand **Apple** (dataset dùng Company Name = Apple). Nếu vẫn sai, kiểm tra đã chạy `init_data.py` với CSV và Pinecone có metadata `brand: "Apple"` cho iPhone.

### Q: How to change LLM model?
A: Update environment variable `MODEL_LOADER_BACKEND` in `.env` file:
```bash
MODEL_LOADER_BACKEND=gemini  # or groq, ollama, openai, claude, cohere
```

### Q: How to add new product dataset?
A: **Điện thoại (CSV)**: Dùng `Mobiles Dataset (2025).csv` format, chạy `python init_data.py`

**Laptop/Tablet/Phụ kiện (JSON)**:
```bash
python init_data.py data/processed/sample_products_extra.json
```
JSON format: `{"products": [{"id","name","brand","category","price","description",...}]}`

### Q: How to enable/disable RAG or API calls?
A: Trong `.env`:
```bash
RAG_ENABLED=true          # Cần PINECONE_API_KEY + đã chạy init_data.py
ENABLE_API_CALLS=true     # Cần Spring Boot backend
```

### Q: How to enable/disable personalization?
A: Trong `.env` (tắt mặc định):
```bash
ENABLE_PERSONALIZATION=true
ENABLE_RECOMMENDATIONS=true
ENABLE_RL_LEARNING=true
```

### Q: How to monitor performance?
A: Use these endpoints:
- `/health` - Health check
- `/metrics` - Detailed metrics
- `/dashboard` - Overview dashboard
- `/traces` - Request tracing

### Q: How to scale the system?
A: Use load balancer and multiple instances with Redis cluster.

### Q: What Python version is required?
A: Python 3.10+ is required, but Python 3.11+ is recommended for best performance.

### Q: How to update dependencies?
A: Run `pip install -r requirements.txt --upgrade` to update all packages to latest versions.

### Q: How to run in production?
A: **Docker** (khuyến nghị):
```bash
docker-compose up -d
# Hoặc docker build -f dockerfile -t ai-agent:v1 .
```

**Gunicorn**:
```bash
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
Chi tiết deploy: xem `DEPLOYMENT.md`

## Roadmap

### Phase 1: Core Features ✅
- [x] Hybrid Orchestrator
- [x] RAG System với Pinecone 5.0.1+
- [x] Multi-model support (Gemini, Groq, Ollama, OpenAI, Claude, Cohere)
- [x] Basic caching (Redis 5.2.1+)

### Phase 2: Advanced Features ✅
- [x] Personalization system
- [x] API integration
- [x] Monitoring & observability (OpenTelemetry 1.28.0+)
- [x] Training pipeline (PyTorch 2.5.1+)

### Phase 3: Production Ready ✅
- [x] Updated dependencies (FastAPI 0.115.6+, PyTorch 2.5.1+)
- [x] Production server (Gunicorn 23.0.0+)
- [x] Docker containerization (dockerfile + docker-compose)
- [x] Railway deployment (railway.json)
- [ ] Kubernetes deployment
- [ ] Rate limiting

### Phase 4: Enterprise Features 📋
- [ ] Multi-tenant support
- [ ] Advanced analytics
- [ ] A/B testing
- [ ] Custom model training
- [ ] Auto-scaling

## Contributing

We welcome all contributions! Please:

1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Create Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Write test cases for new code
- Update documentation
- Use conventional commits

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Documentation

| File | Nội dung |
|------|----------|
| [DEPLOYMENT.md](DEPLOYMENT.md) | Docker, DockerHub, Railway deploy, troubleshooting |
| [ECOMMERCE_AI_AGENT_ROADMAP.md](ECOMMERCE_AI_AGENT_ROADMAP.md) | Roadmap E-commerce, gợi ý Phase 2-4 |
| [INTEGRATION_PLAN.md](INTEGRATION_PLAN.md) | Tích hợp Spring Boot microservices |
| [FRONTEND_INTEGRATION.md](FRONTEND_INTEGRATION.md) | Tích hợp Frontend React/Vue |

## Support & Contact

- **Email**: support@ai-agent.com
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

## Acknowledgments

- [Google Gemini API](https://ai.google.dev/) - LLM capabilities (v0.8.3+)
- [Groq API](https://groq.com/) - Fast inference (v0.9.0+)
- [Ollama](https://ollama.ai/) - Local LLM hosting (v0.4.2+)
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework (v0.115.6+)
- [Pinecone](https://www.pinecone.io/) - Vector database (v5.0.1+)
- [Redis](https://redis.io/) - Caching layer (v5.2.1+)
- [Pydantic](https://pydantic.dev/) - Data validation (v2.10.4+)
- [PyTorch](https://pytorch.org/) - Deep learning framework (v2.5.1+)
- [Transformers](https://huggingface.co/transformers/) - NLP models (v4.47.1+)

---

<div align="center">

**If this project is helpful, please give us a star!**

Made with ❤️ by Taun

</div>