# Đề xuất tối ưu và phát triển hệ thống AI Agent

## 1. Tối ưu hiệu năng

### 1.1 Router & Intent
- **Rule engine**: Biên soạn regex một lần khi khởi tạo (đã có); bổ sung rule cho Laptop, Tablet, Phụ kiện để tránh nhầm với "chat".
- **ML Router**: Thay SimpleIntentClassifier bằng model nhẹ (e.g. distilled transformer hoặc sklearn pipeline) đã fine-tune trên dataset tiếng Việt; cache embedding/features cho cùng message.
- **Fusion**: Cho phép cấu hình fusion_weights qua env; A/B test rule vs ML vs hybrid để đo accuracy/latency.

### 1.2 RAG & Vector
- **Embedding**: Giữ Pinecone Inference; thêm batch embed cho nhiều query/upsert để giảm round-trip.
- **Search**: Thử hybrid search (vector + keyword/metadata) nếu Pinecone hỗ trợ; top_k động theo độ dài query; re-ranking bằng cross-encoder nhẹ (optional).
- **Cache**: Đã có cache search; mở rộng cache cho embedding (hash query → embedding) với TTL ngắn để trùng query.

### 1.3 LLM & API
- **Model loader**: Connection pooling / keep-alive cho HTTP client; timeout và retry có backoff cho Gemini/Groq/OpenAI.
- **Streaming**: Hỗ trợ SSE/stream cho `/ask` để trả response từng chunk, cải thiện perceived latency.
- **Spring Boot**: Circuit breaker (e.g. tenacity) cho order/payment/warranty; fallback message khi service lỗi.

### 1.4 Infra
- **Async**: Rà soát toàn bộ I/O (Pinecone, LLM, Redis, HTTP) dùng async; tránh blocking trong event loop.
- **Connection**: Singleton HTTP client (httpx.AsyncClient) dùng chung; Redis connection pool.
- **Graceful shutdown**: Đợi request đang xử lý xong, đóng connection đúng thứ tự (app → router → cache → pinecone).

---

## 2. Độ tin cậy & ổn định

### 2.1 Retry & Resilience
- Retry có backoff (exponential) cho: Pinecone, LLM API, Spring Boot; số lần và max delay cấu hình qua env.
- Health check sâu: kiểm tra Pinecone (list_indexes hoặc query thử), Redis ping, LLM (generate 1 token), Spring Boot (GET /actuator/health nếu có).

### 2.2 Input & Security
- Giới hạn độ dài `message` (e.g. 2000 ký tự); trim và chuẩn hóa khoảng trắng.
- Rate limit theo `user_id` hoặc IP (e.g. 60 req/phút) để tránh lạm dụng.
- Không log full message có thể chứa thông tin nhạy cảm; log length và hash.

### 2.3 Observability
- Structured logging (JSON) với trace_id, user_id, intent, latency, cache_hit.
- Metrics: latency p50/p95/p99 theo intent (search/order/chat/api), lỗi theo endpoint, số request rule vs ML vs hybrid.
- Alert khi error_rate > ngưỡng hoặc latency p95 > ngưỡng.

---

## 3. Phát triển tính năng

### 3.1 RAG & Search
- **Đa ngôn ngữ**: Chuẩn hóa query (normalize, synonym) trước khi embed; hỗ trợ tìm bằng tiếng Anh (brand/model).
- **Filter nâng cao**: Facet theo brand, giá, category từ query tự nhiên; lưu filter vào context để follow-up.
- **Không có kết quả**: Trả gợi ý mở rộng (bỏ bớt filter, đổi từ khóa) thay vì chỉ “không tìm thấy”.

### 3.2 Conversation
- **Session context**: Lưu N tin nhắn gần nhất (user + assistant) vào cache/Redis theo session_id; đưa vào prompt để có follow-up.
- **Clarification**: Khi confidence thấp, trả câu hỏi làm rõ (ví dụ: “Bạn muốn xem đơn hàng hay tìm sản phẩm?”) thay vì đoán.

### 3.3 Order & API
- **JWT từ client**: Nhận JWT trong header/context; gửi sang Spring Boot thay vì chỉ API key server-side.
- **Webhook**: Cho phép đăng ký webhook khi đơn hàng thay đổi trạng thái (nếu backend hỗ trợ).

### 3.4 Personalization
- **Cold start**: Gợi ý phổ biến hoặc theo category khi user mới.
- **Feedback**: Ghi nhận click/mua từ UI; cập nhật profile và re-rank cho lần sau.
- **Consent**: Chỉ lưu profile khi user đồng ý (flag trong context/profile).

### 3.5 Training & Evaluation
- **Dataset**: Thu thập (user_message, intent_expected, response) từ production (có consent); gán nhãn bán tự động (rule + review).
- **Eval pipeline**: Script đánh giá intent accuracy, RAG relevance (MRR/NDCG), satisfaction (thumbs up/down) trên bộ test.
- **Fine-tune**: Chu kỳ fine-tune intent model và (nếu cần) LLM nhỏ cho domain; deploy qua canary.

---

## 4. Chất lượng code & vận hành

### 4.1 Testing
- **Unit**: Router rule match, intent mapping, cache key generation, parse specs trong RAG.
- **Integration**: Mock Pinecone/LLM/Spring Boot; test flow search/order/chat end-to-end.
- **E2E**: Bộ câu hỏi chuẩn (xem `docs/TEST_QUESTIONS.md`) chạy định kỳ; so sánh intent và so sánh response (diff hoặc embedding similarity).

### 4.2 Config & Deploy
- Tách config theo môi trường (dev/staging/prod); secret từ vault/env, không hardcode.
- Docker build multi-stage; health check trong container; rollout với readiness probe.

### 4.3 Documentation
- README: bảng config env, ví dụ request/response, mô tả luồng Hybrid.
- Runbook: Cách xử lý lỗi thường gặp (Pinecone timeout, LLM 429, Spring Boot down).
- API: Giữ OpenAPI/Swagger cập nhật; thêm example cho từng endpoint.

---

## 5. Thứ tự ưu tiên gợi ý

| Ưu tiên | Hạng mục | Lý do |
|--------|----------|--------|
| P0 | Rate limit + input validation | Bảo vệ hệ thống và dữ liệu |
| P0 | Health check sâu + retry/backoff | Ổn định production |
| P1 | Session context cho conversation | Trải nghiệm hội thoại tốt hơn |
| P1 | Streaming response cho /ask | Cảm giác nhanh hơn |
| P1 | Bộ test E2E với câu hỏi chuẩn | Regression và chất lượng |
| P2 | Circuit breaker cho API | Giảm cascade failure |
| P2 | Structured logging + metrics | Debug và monitoring |
| P2 | RAG: hybrid search / re-rank | Độ chính xác tìm kiếm |
| P3 | Fine-tune intent model | Độ chính xác routing |
| P3 | Personalization cold start + feedback | Tăng conversion |

---

*Tài liệu này nên được cập nhật khi triển khai xong từng hạng mục hoặc khi yêu cầu sản phẩm thay đổi.*
