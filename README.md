# AI Agent System - Hybrid Orchestrator

Intelligent AI Agent system for e-commerce with **Hybrid Orchestrator** combining rule-based and ML-based routing. Hỗ trợ 900+ điện thoại và đa danh mục (Laptop, Tablet, Phụ kiện).

> 📌 Bản tài liệu tổng hợp mới nhất: `README_NEW.md`
>
> 📌 Bộ câu hỏi test chức năng: `docs/TEST_QUESTIONS.md`

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-5.0.1+-orange.svg)](https://pinecone.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

**Mục lục nhanh:** [System Architecture](#system-architecture) · [Workflows](#workflows) · [Directory Structure](#directory-structure) · [Quick Start](#quick-start) · [Configuration](#configuration) · [API Endpoints](#usage) · [Testing](#testing) · [Monitoring](#monitoring)

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

---

## Workflows

Các luồng xử lý chính của hệ thống, từ khởi động ứng dụng đến xử lý từng loại request.

### Workflow 1: Khởi động ứng dụng (Startup)

Luồng chạy khi `python app.py` hoặc uvicorn start.

```mermaid
sequenceDiagram
    participant U as Uvicorn
    participant App as FastAPI App
    participant Config as config.get_settings()
    participant Router as AgnoRouter
    participant Cache as CacheManager
    participant LLM as ModelLoader
    participant PC as Pinecone (nếu RAG)
    participant RAG as RAGModel
    participant Int as InteractionModel
    participant API as APIModel

    U->>App: lifespan startup
    App->>Config: get_settings()
    App->>Router: AgnoRouter(config)
    App->>Router: initialize()

    Router->>Cache: _initialize_cache_manager()
    Cache->>Cache: MemoryCache + RedisCache init

    Router->>LLM: _initialize_model_loader()
    LLM->>LLM: create_loader(backend, model_name, api_key)

    alt RAG_ENABLED=true
        Router->>PC: _initialize_pinecone()
        Router->>RAG: _initialize_rag_model()
    else RAG_ENABLED=false
        Router->>Router: rag_model = None
    end

    Router->>Int: _initialize_interaction_model()
    Router->>API: _initialize_api_model()

    opt ENABLE_PERSONALIZATION=true
        Router->>Router: _initialize_personalization_model()
    end

    opt enable_hybrid=true
        Router->>Router: ml_router.initialize()
    end

    Router-->>App: initialized
    App-->>U: yield (ready)
```

**Các bước:**

1. **Lifespan** (`app.py`): FastAPI gọi `lifespan` → đọc `config` từ `.env`.
2. **AgnoRouter(config)**: Tạo router với `rag_config`, `api_config`, `personalization_config`, `hybrid_config`.
3. **initialize()**:
   - Cache: khởi tạo MemoryCache + RedisCache (Redis fail → chỉ dùng memory).
   - Model loader: tạo loader theo `MODEL_LOADER_BACKEND` (gemini/groq/openai/...) và `api_key`.
   - Nếu `RAG_ENABLED=true`: init Pinecone client → init RAGModel (embedding qua Pinecone Inference).
   - InteractionModel: dùng chung model loader cho chat và format search response.
   - APIModel: cấu hình URL Spring Boot, `enable_api_calls`, timeout.
   - Nếu `ENABLE_PERSONALIZATION=true`: ProfileManager + Recommender + PersonalizationModel.
   - Nếu `enable_hybrid=true`: MLRouter (SimpleIntentClassifier, ContextAnalyzer, ConfidenceScorer).
4. Router sẵn sàng; lifespan yield → app nhận request.

---

### Workflow 2: Xử lý request chính (/ask)

Luồng từ khi client gọi `POST /ask` đến khi trả response.

```mermaid
flowchart TB
    A[POST /ask] --> B[get_router]
    B --> C[router.process_request]
    C --> D{intent có sẵn?}
    D -->|Có| E[_process_with_intent]
    D -->|Không| F{enable_hybrid?}
    F -->|Có| G[_process_hybrid_request]
    F -->|Không| H[_process_rule_based_request]
    G --> I[Rule + ML song song]
    I --> J[DecisionFusionEngine.fuse_decisions]
    J --> E
    H --> K[_route_request rules]
    K --> E
    E --> L{intent?}
    L -->|search| M[_handle_search_request]
    L -->|order| N[_handle_order_request]
    L -->|api| O[_handle_api_request]
    L -->|chat| P[_handle_chat_request]
    M --> Q[Response]
    N --> Q
    O --> Q
    P --> Q
    Q --> R[ChatResponse + metadata]
    R --> S[Optional: training collect_conversation]
    S --> T[Return 200]
```

**Các bước:**

1. **Request**: Body gồm `message`, `user_id`, `session_id`, `context`, `intent` (tùy chọn).
2. **Intent có sẵn**: Nếu client gửi `intent` → gọi trực tiếp `_process_with_intent(message, intent, ...)`.
3. **Hybrid routing** (khi bật):
   - Chạy song song: `_get_rule_decision(message)` và `_get_ml_decision(message)`.
   - Rule: so khớp regex theo thứ tự priority → trả handler (search/order/api/chat).
   - ML: ContextAnalyzer → SimpleIntentClassifier → map intent (product_search→search, order_inquiry→order, ...).
   - Fusion: gộp confidence theo weight (rule_based/ml_based), chọn intent (ưu tiên ML nếu confidence > 0.8, else rule hoặc hybrid).
4. **Rule-only**: Chỉ `_route_request(message)` → handler.
5. **Dispatch theo intent**:
   - **search** → `_handle_search_request`
   - **order** → `_handle_order_request`
   - **api** → `_handle_api_request`
   - **chat** → `_handle_chat_request`
6. **Response**: Trả `response`, `intent`, `confidence`, `metadata`; app thêm `model_info`, có thể gọi `training_pipeline.collect_conversation` (async).

---

### Workflow 3: Tìm kiếm sản phẩm (Search – RAG vs Fallback)

Luồng khi intent = **search**.

```mermaid
flowchart TB
    A[_handle_search_request] --> B{rag_model có?}
    B -->|Không| C[_handle_search_fallback]
    B -->|Có| D[Cache key: type=search, query, user_id]
    D --> E{cache hit?}
    E -->|Có| F[Return cached result]
    E -->|Không| G[rag_model.search_products]
    G --> H[Pinecone: embed query + search_products]
    H --> I[_process_search_results]
    I --> J{personalization_model + user_id?}
    J -->|Có| K[record_user_interaction + get_personalized_recommendations]
    J -->|Không| L[search_results]
    K --> L
    L --> M[interaction_model.generate_search_response]
    M --> N[LLM: prompt + products → response]
    N --> O[Build result + metadata]
    O --> P[Cache set TTL 30min]
    P --> Q[Return result]
    C --> R[interaction_model.generate_response]
    R --> S[LLM: chat only, no RAG]
    S --> T[Return rag_disabled hint]
```

**Các bước:**

1. **RAG tắt**: Gọi `_handle_search_fallback` → `InteractionModel.generate_response(message)` (chỉ LLM), metadata có `rag_disabled: true`.
2. **RAG bật**:
   - Tạo cache key từ query + user_id. Nếu cache hit → trả luôn.
   - `RAGModel.search_products`: extract metadata từ query (giá, brand, category, specs) → embed query (Pinecone Inference `llama-text-embed-v2`) → `pinecone_client.search_products` (vector + filter) → `_process_search_results` (format, relevance score, parse specs string→dict). Có thể relaxed search nếu filter chặt không ra kết quả.
   - Nếu bật personalization và có user_id: `record_user_interaction`, `get_personalized_recommendations` (re-rank).
   - `InteractionModel.generate_search_response`: dùng `PromptTemplates.get_contextual_prompt` (query + products) → LLM trả lời tự nhiên.
   - Lưu cache (TTL 1800s), trả result.

---

### Workflow 4: Đơn hàng & API (Order)

Luồng khi intent = **order** hoặc **api**.

```mermaid
sequenceDiagram
    participant R as Router
    participant A as APIModel
    participant S as Spring Boot

    R->>R: _is_authenticated(user_id, context)
    alt Không có user_id / auth
        R-->>Client: auth_required response
    end
    R->>A: handle_order_request(message, user_id, context)
    A->>A: _extract_order_id(message)
    alt Không có order_id
        A-->>R: "Cần số đơn hàng..."
    end
    alt ENABLE_API_CALLS=false
        A-->>R: "Tính năng gọi API đang tắt"
    end
    A->>S: GET /orders/{order_id}
    S-->>A: order JSON hoặc error
    A->>A: _transform_order_response
    A-->>R: response text
    R-->>Client: intent=order, response, metadata
```

**Các bước:**

1. **Auth**: `_is_authenticated(user_id, context)` — cần user_id hoặc context có `is_authenticated`/`jwt_token`. Không đủ → trả "Bạn cần đăng nhập...".
2. **Order**: `APIModel.handle_order_request`: trích order id từ message (regex #\d+ hoặc \d{4,}) → nếu không có id trả "Cần số đơn hàng...". Nếu `enable_api_calls=false` trả message tắt API.
3. **Gọi service**: HTTP GET `order_service_url/{order_id}`, header Authorization nếu có api_key. Transform response → text trả về.
4. **API chung**: intent=api → `handle_general_request` (tương tự, tùy message có thể gọi order/payment/warranty/product).

---

### Workflow 5: Hội thoại chung (Chat)

Luồng khi intent = **chat**.

```mermaid
flowchart LR
    A[_handle_chat_request] --> B[interaction_model.generate_response]
    B --> C[_create_system_prompt]
    C --> D[Prompt: trợ lý bán hàng, tiếng Việt, không bịa]
    D --> E[conversation_prompt = system + Người dùng: message + Trợ lý AI:]
    E --> F[model_loader.generate_response]
    F --> G[LLM API]
    G --> H[response text]
    H --> I[Return intent=chat, confidence=0.8]
```

**Các bước:**

1. `InteractionModel.generate_response(message, user_id, context)`.
2. Tạo system prompt (trợ lý bán hàng, tiếng Việt, tư vấn chính xác, hỗ trợ đơn hàng/bảo hành).
3. Ghép prompt: system + "Người dùng: {message}\nTrợ lý AI:".
4. Gọi `model_loader.generate_response(prompt, max_tokens=512, temperature=0.7)`.
5. Trả response; nếu lỗi → fallback "Xin lỗi, tôi gặp lỗi...".

---

### Workflow 6: Khởi tạo dữ liệu (init_data.py)

Luồng load sản phẩm từ CSV/JSON lên Pinecone (chạy tay khi cần).

```mermaid
flowchart TB
    A[python init_data.py path] --> B[DataInitializer]
    B --> C[initialize: Pinecone + ModelLoader + RAGModel]
    C --> D[load_dataset path]
    D --> E{format?}
    E -->|.json| F[generic_json: list hoặc .products/.items]
    E -->|.csv Mobiles| G[mobile_csv: pandas, transform_product_data]
    E -->|.csv khác| H[generic_csv]
    F --> I[transform_product_data_generic]
    G --> J[transform_product_data CSV]
    I --> K[ingest_products batch]
    J --> K
    K --> L[rag_model.upsert_product từng sản phẩm]
    L --> M[Pinecone: embed qua Inference + upsert_vectors]
    M --> N[Log success/failed]
```

**Các bước:**

1. **DataInitializer**: Khởi tạo Pinecone client, ModelLoader, RAGModel (giống app nhưng độc lập).
2. **load_dataset(path)**: Auto detect format: `.json` → generic_json; CSV có "Mobiles" → mobile_csv; còn lại → generic_csv.
3. **Transform**:
   - **mobile_csv**: `transform_product_data` — map cột Company Name, Model Name, Launched Price (USA), RAM, Screen Size, Battery, Camera... → product schema (id, name, brand, price VND, description, specifications).
   - **generic_json**: `transform_product_data_generic` — map name, brand, category, price, specifications, ...
4. **ingest_products**: Chia batch (mặc định 50); mỗi sản phẩm → `rag_model.upsert_product` (tạo text → embed qua Pinecone Inference → upsert vector + metadata vào namespace "default").
5. Có thể dùng `export_products_to_json` để chỉ transform và xuất JSON không đẩy Pinecone.

---

### Workflow 7: Training & Fine-tuning (tùy chọn)

Thu thập hội thoại và chạy pipeline training (module có thể chưa có đầy đủ).

```mermaid
flowchart TB
    subgraph Mỗi /ask
        A[ask endpoint] --> B[response xong]
        B --> C[get_training_pipeline]
        C --> D[collect_conversation]
        D --> E[buffer: user_message, assistant_response, intent, ...]
    end
    subgraph Training thủ công
        F[POST /training/start] --> G[training_pipeline.start_training_pipeline]
        G --> H[data_source: dataset | conversations]
        H --> I[prepare_data / load buffer]
        I --> J[finetune / evaluate]
    end
    subgraph Trạng thái
        K[GET /training/status]
        L[GET /training/history]
    end
```

**Các bước:**

1. **Thu thập**: Sau mỗi `/ask`, nếu import được `training_pipeline` → `get_training_pipeline().collect_conversation(conversation)` (user_message, assistant_response, intent, confidence, user_id, session_id, timestamp).
2. **Start**: `POST /training/start` (data_source, auto_mode) → gọi `start_training_pipeline`; có thể dùng dataset cố định hoặc buffer từ conversations.
3. **Status/History**: `GET /training/status`, `GET /training/history` — trả trạng thái và lịch sử training (nếu module có implement).

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
├── docs/                         # Tài liệu & test
│   ├── OPTIMIZATION_AND_ROADMAP.md # Đề xuất tối ưu và phát triển
│   ├── TEST_QUESTIONS.md         # Bộ câu hỏi test hệ thống
│   └── test_questions.json      # Test cases E2E/regression
├── scripts/                      # Scripts tiện ích
│   └── run_test_questions.py    # Chạy bộ câu hỏi test (/ask)
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
├── cache/                        # Caching layer
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
│   │   └── sample_products_extra.json  # Mẫu Laptop, Tai nghe, Sạc
│   ├── profiles/                 # User profiles
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

## Usage

### API Endpoints

#### 1. Main Chat endpoint (Hybrid Orchestrator)
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "OnePlus under 50 million",
    "user_id": "user123",
    "session_id": "session001"
  }'
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

### 3. Multi-category Dataset
- **Điện thoại**: 900+ sản phẩm (`Mobiles Dataset (2025).csv`) - Apple, Samsung, OnePlus, Xiaomi, etc.
- **Laptop, Tablet, Phụ kiện**: Hỗ trợ JSON (`data/processed/sample_products_extra.json`)
- **Schema**: `data/schema/product_schema.py` - Điện thoại, Laptop, Tablet, Tai nghe, Sạc dự phòng, ...
- **Init**: `python init_data.py [file.csv|file.json]` - Tự động detect format

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

**Bộ câu hỏi test (E2E):** Danh sách câu hỏi và test cases để kiểm tra routing (search/order/chat/api) — xem [docs/TEST_QUESTIONS.md](docs/TEST_QUESTIONS.md) và [docs/test_questions.json](docs/test_questions.json). Chạy script (cần server đang chạy):

```bash
python scripts/run_test_questions.py
python scripts/run_test_questions.py --url http://localhost:8000 --json docs/test_questions.json
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