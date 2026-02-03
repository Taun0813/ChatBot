# 🛒 AI Agent for E-commerce - Roadmap & Gợi ý

Tài liệu gợi ý các bước cần làm để hoàn thiện AI Agent cho E-commerce.

---

## ✅ Đã hoàn thành

- [x] **Phase 1**: RAG/API config qua env, kết nối Spring Boot, mở rộng schema
- [x] Hybrid Orchestrator (Rule-based + ML-based routing)
- [x] RAG với Pinecone (tìm kiếm sản phẩm semantic)
- [x] Multi-model (Gemini, Groq, OpenAI, Ollama)
- [x] API integration (Order, Payment, Warranty, Product services)
- [x] Dataset Mobiles (900+ sản phẩm điện thoại)
- [x] Docker + Docker Compose
- [x] Deploy Railway
- [x] Monitoring (health, metrics, dashboard, traces)

---

## 📋 Các phần cần chỉnh sửa / cập nhật

### 1. Config & Environment

| Vấn đề | Gợi ý |
|--------|-------|
| **env.example vs config.py** | Thống nhất tên biến: `PINECONE_INDEX` (env.example) vs `pinecone_index_name` (config.py) - config.py dùng `PINECONE_INDEX_NAME` |
| **RAG disabled mặc định** | Trong `app.py`, RAG đang `enabled: False` để tránh lỗi Pinecone. Khi có Pinecone key ổn định, set qua env `RAG_ENABLED=true` |
| **API calls disabled** | `enable_api_calls: False` trong app.py - cần bật khi đã có Spring Boot backend |

### 2. Dockerfile

| Vấn đề | Gợi ý |
|--------|-------|
| **Tên file** | Một số CI/CD yêu cầu `Dockerfile` (chữ hoa). Có thể tạo symlink hoặc rename |
| **Init data** | Container không chạy `init_data.py` khi start. Cân nhắc entrypoint script chạy init nếu chưa có data |

### 3. Data & RAG

| Vấn đề | Gợi ý |
|--------|-------|
| **Pinecone index** | Cần tạo index trước (dimension=1024 với embedding model hiện tại) |
| **Dataset** | `Mobiles Dataset (2025).csv` - có thể thêm category khác (laptop, phụ kiện) |
| **Embedding model** | Kiểm tra dimension của model đang dùng khớp với Pinecone index |

---

## 🚀 Gợi ý để làm AI Agent E-commerce hoàn chỉnh

### Phase 1: Core E-commerce (Ưu tiên cao)

1. **Bật RAG khi đã có Pinecone**
   - Cấu hình `RAG_ENABLED=true` qua env
   - Chạy `python init_data.py` để ingest sản phẩm lên Pinecone

2. **Kết nối Spring Boot / Backend thật**
   - Cập nhật `ORDER_SERVICE_URL`, `PRODUCT_SERVICE_URL`, v.v. trong .env
   - Bật `enable_api_calls: True` trong router config
   - Test flow: tìm sản phẩm → đặt hàng → thanh toán

3. **Mở rộng dataset**
   - Thêm sản phẩm (laptop, tablet, phụ kiện)
   - Cập nhật schema trong `data/schema/`

### Phase 2: Trải nghiệm người dùng

4. **Personalization**
   - Bật `ENABLE_PERSONALIZATION=true` khi đã có user profiles
   - Tích hợp `profile_manager` với User Service

5. **Recommendations**
   - Bật `ENABLE_RECOMMENDATIONS=true`
   - Fine-tune `recommender.py` dựa trên behavior

6. **Session & Context**
   - Lưu conversation history theo `session_id`
   - Hỗ trợ "tiếp tục cuộc trò chuyện trước"

### Phase 3: Production & Scale

7. **Rate limiting**
   - Thêm rate limit cho `/ask` (ví dụ: 60 req/phút/user)
   - Tránh abuse API

8. **Caching Redis**
   - Dùng Redis trong Docker Compose (`--profile full`)
   - Set `CACHE_TYPE=redis`, `REDIS_HOST=redis`

9. **Health check nâng cao**
   - Kiểm tra Pinecone connection
   - Kiểm tra external APIs (Order, Product...)

10. **Logging & Observability**
    - Structured logging (JSON format)
    - Export metrics Prometheus nếu cần
    - Alert khi error rate cao

### Phase 4: Nâng cao

11. **A/B Testing**
    - So sánh model A vs B
    - Track conversion, satisfaction

12. **Fine-tune model**
    - Dùng data từ conversations thật
    - Chạy `training/prepare_data.py` → `finetune.py`

13. **Multi-language**
    - Hỗ trợ tiếng Anh, tiếng Việt
    - Detect language và phản hồi phù hợp

14. **Cart & Checkout flow**
    - "Thêm vào giỏ" qua chat
    - "Thanh toán" - tích hợp Payment Service

---

## 📁 Cấu trúc file quan trọng

```
├── config.py           # Cấu hình - thêm RAG_ENABLED, API_ENABLED
├── app.py              # Lifespan - đọc config từ env thay vì hardcode
├── dockerfile          # Build Docker image
├── docker-compose.yml  # Chạy local với Redis
├── railway.json        # Cấu hình Railway
├── env.example         # Template biến môi trường
├── DEPLOYMENT.md       # Hướng dẫn deploy
└── data/
    ├── ingest.py       # Load data lên Pinecone
    └── processed/      # products.json, knowledge_base.json
```

---

## 🔗 Tham khảo

- **INTEGRATION_PLAN.md** - Chi tiết tích hợp với Spring Boot
- **FRONTEND_INTEGRATION.md** - Tích hợp Frontend React/Vue
- **DEPLOYMENT.md** - Docker, Railway, troubleshooting

---

## Quick checklist trước khi production

- [ ] Có GEMINI_API_KEY hoặc GROQ_API_KEY hợp lệ
- [ ] Pinecone index đã tạo và đã chạy init_data.py
- [ ] External service URLs đã cấu hình (nếu dùng)
- [ ] CORS đã cấu hình đúng domain frontend (thay `allow_origins=["*"]` nếu cần)
- [ ] Health check `/health` trả về 200
- [ ] Test `/ask` với câu hỏi tìm sản phẩm
- [ ] Docker image build thành công
- [ ] Environment variables không chứa secret trong code
