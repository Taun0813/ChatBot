# Hệ thống AI Agent - Hybrid Orchestrator

Một hệ thống AI Agent thông minh cho thương mại điện tử với **Hybrid Orchestrator** kết hợp định tuyến dựa trên quy tắc và dựa trên ML. Hệ thống hỗ trợ hơn 900 điện thoại di động và nhiều danh mục sản phẩm khác như Máy tính xách tay, Máy tính bảng và Phụ kiện.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-5.0.1+-orange.svg)](https://pinecone.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Tính năng

- **Hybrid Orchestrator**: Kết hợp định tuyến dựa trên quy tắc và dựa trên ML để đạt độ chính xác cao (85-95%).
- **Bộ dữ liệu đa danh mục**: Hỗ trợ điện thoại di động, máy tính xách tay, máy tính bảng và phụ kiện (CSV + JSON).
- **Hệ thống RAG**: Thực hiện tìm kiếm ngữ nghĩa bằng Pinecone (có thể bật/tắt bằng `RAG_ENABLED=true`).
- **Trò chuyện thông minh**: Cung cấp tương tác tự nhiên, với cơ chế dự phòng khi RAG bị tắt.
- **Tích hợp API**: Kết nối với các microservice Spring Boot cho order/payment/warranty; đã có intent cho shipping/cart/checkout/refund/return.
- **Cá nhân hóa**: Thích ứng với hành vi của người dùng và cung cấp các đề xuất (tùy chọn).
- **Hỗ trợ đa mô hình**: Tương thích với nhiều LLM khác nhau, bao gồm Gemini, Groq, Ollama và OpenAI.
- **Bộ nhớ đệm**: Có hệ thống bộ nhớ đệm thông minh với các tùy chọn Redis và trong bộ nhớ.
- **Giám sát**: Cung cấp giám sát hiệu suất thời gian thực thông qua một bảng điều khiển chi tiết.
- **Đào tạo**: Bao gồm một quy trình dữ liệu hoàn chỉnh để tinh chỉnh các mô hình cho lĩnh vực thương mại điện tử.
- **Sẵn sàng cho sản xuất**: Được xây dựng với FastAPI, PyTorch và các mẫu async/await hiện đại.

## Các khái niệm cốt lõi

### Hybrid Orchestrator
Hybrid Orchestrator là bộ não của hệ thống, quyết định cách xử lý các yêu cầu của người dùng. Nó sử dụng hai chiến lược chính:

- **Định tuyến dựa trên quy tắc**: Nhanh chóng và có thể dự đoán được, phương pháp này sử dụng các biểu thức chính quy để khớp các truy vấn của người dùng với các ý định cụ thể (ví dụ: "tìm" cho `search`, "#1234" cho `order`). Nó lý tưởng cho các yêu cầu đơn giản, rõ ràng.
- **Định tuyến dựa trên ML**: Linh hoạt và nhận biết ngữ cảnh hơn, phương pháp này sử dụng một mô hình phân loại ý định để hiểu mục tiêu của người dùng. Nó tốt hơn cho các truy vấn phức tạp hoặc không rõ ràng.
- **Kết hợp quyết định**: Hệ thống kết hợp đầu ra của cả hai bộ định tuyến, sử dụng cơ chế trọng số để đưa ra quyết định cuối cùng. Cách tiếp cận kết hợp này cung cấp cả tốc độ của các hệ thống dựa trên quy tắc và sự thông minh của các mô hình ML.

### RAG (Retrieval-Augmented Generation)
Khi người dùng đặt câu hỏi về một sản phẩm, hệ thống sử dụng RAG để cung cấp câu trả lời chính xác. Đây là cách nó hoạt động:
1. **Truy xuất**: Truy vấn của người dùng được chuyển đổi thành một vector embedding và được sử dụng để tìm kiếm một cơ sở dữ liệu vector Pinecone chứa tất cả thông tin sản phẩm.
2. **Bổ sung**: Thông tin sản phẩm phù hợp nhất được truy xuất và thêm vào truy vấn ban đầu của người dùng làm ngữ cảnh.
3. **Tạo**: Văn bản kết hợp (truy vấn ban đầu + ngữ cảnh được truy xuất) được gửi đến một mô hình ngôn ngữ lớn (LLM), mô hình này sẽ tạo ra một câu trả lời tự nhiên, giống như con người dựa trên thông tin được cung cấp.
Quá trình này đảm bảo rằng các câu trả lời được dựa trên dữ liệu sản phẩm thực tế, giảm nguy cơ mô hình "ảo giác" ra các chi tiết không chính xác.

### Cá nhân hóa
Hệ thống có thể điều chỉnh các câu trả lời và đề xuất của mình cho từng người dùng. Khi được bật, nó sẽ theo dõi các tương tác của người dùng (như tìm kiếm và mua hàng) để xây dựng hồ sơ người dùng. Hồ sơ này sau đó được sử dụng để:
- Sắp xếp lại kết quả tìm kiếm để hiển thị các sản phẩm mà người dùng có nhiều khả năng quan tâm hơn.
- Cung cấp các đề xuất sản phẩm được cá nhân hóa.

### Tích hợp dịch vụ
AI Agent được thiết kế để hoạt động như một phần của một hệ sinh thái microservice lớn hơn. Nó có thể giao tiếp với các dịch vụ backend khác (ví dụ: được viết bằng Spring Boot) để xử lý các tác vụ như:
- Lấy trạng thái đơn hàng.
- Xử lý thanh toán.
- Kiểm tra thông tin bảo hành.
Điều này được xử lý bởi `APIModel`, thực hiện các yêu cầu HTTP đến các dịch vụ khác.

## Kiến trúc hệ thống

```mermaid
graph TB
    A[Yêu cầu của khách hàng] --> B[Ứng dụng FastAPI]
    B --> C[AgnoRouter - Hybrid Orchestrator]
    
    C --> D[Bộ định tuyến dựa trên quy tắc]
    C --> E[Bộ định tuyến dựa trên ML]
    
    D --> F[Đối sánh mẫu]
    E --> G[Phân loại ý định]
    
    F --> H[Công cụ kết hợp quyết định]
    G --> H
    
    H --> I{Quyết định ý định}
    
    I -->|search| J[Tác nhân RAG]
    I -->|chat| K[Tác nhân hội thoại]
    I -->|order/shipping/payment/warranty/cart| L[Tác nhân API]
    I -->|checkout/refund/return| W[Flow giao dịch có hướng dẫn]
    
    J --> M[Tìm kiếm Vector Pinecone]
    M --> N[Kết quả sản phẩm]
    N --> O[Cá nhân hóa]
    O --> P[Phản hồi ngôn ngữ tự nhiên]
    
    K --> Q[Mô hình LLM]
    Q --> R[Phản hồi nhận biết ngữ cảnh]
    
    L --> S[API bên ngoài]
    S --> T[Phản hồi API]
    
    P --> U[Trình quản lý bộ nhớ đệm]
    R --> U
    T --> U
    W --> U
    
    U --> V[Phản hồi cho khách hàng]
```

## Bắt đầu

### Điều kiện tiên quyết
- Python 3.10+
- Docker (tùy chọn, để chạy với Docker)
- Một khóa API từ một nhà cung cấp LLM (ví dụ: Google AI Studio cho Gemini)

### Cài đặt
1. **Sao chép kho lưu trữ**:
    ```bash
    git clone <repository-url>
    cd ai-agent
    ```
2. **Tạo một môi trường ảo**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows, sử dụng `venv\Scripts\activate`
    ```
3. **Cài đặt các phụ thuộc**:
    ```bash
    pip install -r requirements.txt
    ```

### Cấu hình (`.env`)
1. **Tạo một tệp `.env`** bằng cách sao chép tệp ví dụ:
    ```bash
    cp env.example .env
    ```
2. **Chỉnh sửa tệp `.env`** để thêm cấu hình của bạn. Tối thiểu, bạn cần cung cấp một khóa API cho LLM bạn muốn sử dụng.

    **Cấu hình cần thiết**:
    ```env
    # Backend để sử dụng cho mô hình ngôn ngữ. Các tùy chọn: "gemini", "groq", "openai", v.v.
    MODEL_LOADER_BACKEND=gemini
    
    # Mô hình cụ thể để sử dụng.
    MODEL_NAME=gemini-1.5-flash
    
    # Khóa API của bạn cho backend đã chọn.
    GEMINI_API_KEY=your_gemini_api_key
    ```
    **Bật RAG và các cuộc gọi API**:
    ```env
    # Đặt thành true để bật hệ thống RAG để tìm kiếm sản phẩm.
    # Yêu cầu phải đặt PINECONE_API_KEY và PINECONE_INDEX_NAME.
    RAG_ENABLED=true
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_INDEX_NAME=your-pinecone-index
    
    # Đặt thành true để cho phép tác nhân gọi các microservice bên ngoài.
    # Yêu cầu phải đặt các URL dịch vụ (ví dụ: ORDER_SERVICE_URL).
    ENABLE_API_CALLS=true
    ORDER_SERVICE_URL=http://localhost:8181/api/orders
    ```

### Chạy ứng dụng

**Cục bộ**:
```bash
python app.py
```
Ứng dụng sẽ có sẵn tại `http://localhost:8000`.

**Với Docker**:
Đây là cách được khuyến nghị để chạy ứng dụng trong một môi trường giống như sản xuất.
```bash
docker-compose up -d
```
Lệnh này sẽ xây dựng hình ảnh Docker và khởi động các container `ai-agent` và `redis`.

## Các điểm cuối API

### `POST /ask`
Đây là điểm cuối chính để tương tác với AI agent.

**Nội dung yêu cầu**:
```json
{
  "message": "OnePlus dưới 50 triệu",
  "user_id": "user123",
  "session_id": "session001",
  "intent": "search"
}
```
- `message` (string, bắt buộc): Tin nhắn của người dùng.
- `user_id` (string, tùy chọn): Một mã định danh duy nhất cho người dùng, được sử dụng để cá nhân hóa.
- `session_id` (string, tùy chọn): Một mã định danh cho phiên trò chuyện hiện tại.
- `intent` (string, tùy chọn): Có thể được sử dụng để gợi ý/buộc intent cụ thể (`search`, `chat`, `order`, `shipping`, `payment`, `warranty`, `cart`, `checkout`, `refund`, `return`, `api_call`).

**Nội dung phản hồi**:
```json
{
  "user_id": "user123",
  "response": "Tôi đã tìm thấy một số điện thoại OnePlus phù hợp với ngân sách của bạn...",
  "intent": "search",
  "confidence": 0.95,
  "session_id": "session001",
  "metadata": {
    "flow": "search",
    "grounding": {
      "grounded": true,
      "evidence_count": 3,
      "evidence_ids": ["mobile_oneplus_12_256gb_black"],
      "citations": [...],
      "grounding_policy": "retrieval_only"
    },
    "model_info": { "backend": "gemini", "model_name": "gemini-1.5-flash" },
    "search_results": [...],
    "action_required": "backend_integration"
  }
}
```

## Trạng thái hiện tại (04/2026)

Hệ thống đã đạt mức **beta tốt** cho trợ lý ecommerce, đặc biệt mạnh ở search + grounding + orchestration. Tuy nhiên chưa đạt mức "production-complete" cho toàn bộ transactional flow.

**Đã ổn định:**
- Hybrid router (rule + ML + fusion) chạy ổn và có metrics.
- Luồng tìm kiếm sản phẩm bằng RAG đã gắn grounding metadata (citations/evidence IDs).
- Intent coverage đã mở rộng cho nghiệp vụ ecommerce cốt lõi.
- OpenAPI đã phản ánh intent mới và metadata chính.

**Chưa hoàn tất:**
- `checkout`, `refund`, `return` hiện là handler có hướng dẫn nghiệp vụ vì backend chưa cung cấp service chuyên dụng.
- Chưa có bộ test tự động đủ sâu cho routing/grounding/regression.
- Một số endpoint training phụ thuộc module pipeline tùy chọn (có thể không tồn tại trong mọi môi trường).

## Ma trận intent và trạng thái thực thi

| Intent | Nguồn xử lý chính | Trạng thái | Metadata đặc trưng |
|---|---|---|---|
| `search` | RAG + Interaction model | Hoàn chỉnh | `grounding`, `search_results`, `results_count`, `product_links` |
| `chat` | Interaction model | Hoàn chỉnh | `model_used=interaction` |
| `order` | API model (order service) | Hoàn chỉnh (phụ thuộc backend) | `model_used=api` |
| `shipping` | API model (order service) | Hoàn chỉnh mức tracking cơ bản | `flow=shipping` |
| `payment` | API model (payment service) | Hoàn chỉnh (phụ thuộc backend) | `flow=payment` |
| `warranty` | API model (warranty service) | Hoàn chỉnh (phụ thuộc backend) | `flow=warranty` |
| `cart` | API model (cart service) | Hoàn chỉnh (phụ thuộc backend) | `flow=cart`, `model_used=api` |
| `checkout` | Router guided flow | Chưa full backend | `flow=checkout`, `action_required=backend_integration` |
| `refund` | Router guided flow + parse order id | Chưa full backend | `flow=refund`, `order_id`, `action_required` |
| `return` | Router guided flow + parse order id | Chưa full backend | `flow=return`, `order_id`, `action_required` |
| `api` / `api_call` | API model general | Hoàn chỉnh mức generic | `model_used=api` |

## Luồng Agent chi tiết

### 1) Luồng Search Agent (RAG-first)
1. Nhận message + context từ `/ask`.
2. Router phân loại intent qua Rule Router, ML Router, sau đó fusion.
3. Nếu `search`: gọi `rag_model.search_products`.
4. Kết quả được re-rank theo personalization (nếu bật).
5. Interaction model tạo câu trả lời dựa trên top products.
6. Router build `grounding` metadata gồm `citations`, `evidence_ids`, `grounded`.
7. Cache kết quả truy vấn để giảm latency cho lần hỏi lại.

### 2) Luồng API Agent (order/payment/warranty/shipping)
1. Router xác thực trạng thái auth (`user_id` hoặc token trong context).
2. Gọi `APIModel` đến Spring Boot service tương ứng.
3. Chuẩn hóa lỗi backend (401, service unavailable) thành thông điệp thân thiện.
4. Trả metadata `flow` + `model_used=api` để frontend phân loại hiển thị.

### 3) Luồng Guided Transaction Agent (checkout/refund/return)
1. Router nhận intent transaction.
2. Nếu thiếu dữ liệu quan trọng (ví dụ `order_id`) thì phản hồi yêu cầu bổ sung.
3. Nếu đủ dữ liệu thì phản hồi theo flow nghiệp vụ hiện có.
4. Metadata trả về `action_required` để FE/BE orchestration layer xử lý bước tiếp theo.

## Hybrid Orchestrator: cơ chế ra quyết định

### Rule-based router
- Dùng regex/keyword rules có priority để bắt intent rõ ràng nhanh và ổn định.
- Phù hợp các pattern có cấu trúc: mã đơn hàng, yêu cầu trạng thái, cụm từ nghiệp vụ cụ thể.

### ML-based router
- Dùng bộ phân loại intent heuristic + context features.
- Ưu thế ở truy vấn mơ hồ, ngôn ngữ tự nhiên, thiếu từ khóa trực diện.

### Fusion layer
- Kết hợp confidence của rule và ML theo trọng số cấu hình.
- Trả metadata orchestrator để theo dõi router nào được chọn và độ tin cậy.

## Contract metadata cho frontend/backend

Các key metadata nên được FE/BE coi là contract:
- `model_info`: backend/model/version chạy response.
- `flow`: luồng nghiệp vụ đang xử lý (`search`, `payment`, `checkout`, ...).
- `grounding`: dữ liệu chứng cứ truy xuất thực tế cho câu trả lời search.
- `action_required`: tín hiệu còn bước bắt buộc bên ngoài router (`order_id`, `backend_integration`, `clarification`).
- `auth_required`: yêu cầu đăng nhập trước khi xử lý tác vụ giao dịch.

## Gợi ý tối ưu tiếp theo (ưu tiên cao)

1. Tích hợp backend thật cho `checkout`, `refund`, `return`.
2. Bổ sung test tự động cho:
   - Intent classification (rule/ML/fusion)
   - Grounding integrity (có/không có citations)
   - API failure contract (401/5xx)
3. Xây dashboard chất lượng routing theo intent (precision/recall theo tuần).
4. Thêm e2e test matrix cho các hội thoại đa bước (search -> cart -> checkout).

## Test Matrix (đã bổ sung)

Các test được thêm trong thư mục `tests/`:

- `tests/test_intent_routing_matrix.py`
  - Kiểm tra độ chính xác routing rule-based theo từng intent ecommerce.
  - Kiểm tra mapping alias intent về canonical intent.

- `tests/test_grounding_integrity.py`
  - Kiểm tra tính hợp lệ của `grounding` metadata.
  - Đảm bảo search grounding có evidence hợp lệ (`product_id`, `name`, `price_vnd`, `source`).

- `tests/test_api_error_contract.py`
  - Kiểm tra contract lỗi cho API integration với các mã: `401`, `403`, `404`, `500`.
  - Xác nhận message phân biệt đúng nhánh `401` (auth) và các lỗi còn lại.

- `tests/test_e2e_multistep_flow.py`
  - Kiểm tra luồng đa bước: `search -> cart -> checkout -> payment -> shipping -> refund -> return`.
  - Kiểm tra replay với `Idempotency-Key` cho transaction flow.

## Guardrails bảo mật và quan sát (đã bổ sung)

Guardrail được thêm tại tầng API trong `app.py`:

- Rate limit cho transaction-like requests tại `/ask`.
  - Cửa sổ mặc định: 60 giây.
  - Giới hạn mặc định: 20 request.

- Idempotency cho transaction-like requests.
  - Bắt buộc header `Idempotency-Key`.
  - Replay cùng key trả lại kết quả đã lưu, tránh xử lý trùng.

- Alerting theo chất lượng vận hành.
  - Theo dõi rolling intent error-rate.
  - Theo dõi tỷ lệ API timeout theo ngưỡng latency.

- Endpoint quan sát mới:
  - `GET /guardrails/stats`
  - `GET /guardrails/alerts`

### Các điểm cuối giám sát
- **`GET /health`**: Kiểm tra tình trạng của ứng dụng.
- **`GET /metrics`**: Cung cấp các chỉ số hiệu suất cho Hybrid Orchestrator.
- **`GET /dashboard`**: Trả về một bảng điều khiển toàn diện về hiệu suất hệ thống.
- **`GET /traces`**: Hiển thị các dấu vết yêu cầu gần đây để gỡ lỗi.

### Các điểm cuối đào tạo
- **`POST /training/start`**: Bắt đầu quy trình đào tạo mô hình.
- **`GET /training/status`**: Lấy trạng thái hiện tại của quá trình đào tạo.

## Cấu trúc dự án
```
ai_agent/
├── app.py # Điểm vào ứng dụng FastAPI
├── config.py # Cấu hình ứng dụng
├── requirements.txt # Các phụ thuộc Python
├── env.example # Mẫu biến môi trường
├── init_data.py # Tập lệnh để tải dữ liệu vào Pinecone
├── docker-compose.yml # Cấu hình Docker Compose
│
├── core/ # Lõi logic ứng dụng
│ ├── router.py # Hybrid Orchestrator
│ ├── rag_model.py # Triển khai RAG
│ ├── interaction_model.py # Xử lý cuộc trò chuyện chung
│ └── api_model.py # Xử lý giao tiếp với các API bên ngoài
│
├── adapters/ # Bộ điều hợp cho các dịch vụ bên ngoài
│ ├── model_loader/ # Bộ tải cho các LLM khác nhau
│ └── pinecone_client.py # Máy khách Pinecone
│
├── data/ # Tệp dữ liệu và lược đồ
│ ├── Mobiles Dataset (2025).csv # Dữ liệu sản phẩm mẫu
│ └── schema/ # Lược đồ dữ liệu
│
├── personalization/ # Lõi logic cá nhân hóa
│ ├── profile_manager.py # Quản lý hồ sơ người dùng
│ └── recommender.py # Tạo đề xuất
│
├── services/ # Máy khách cho các microservice bên ngoài
│
├── training/ # Đào tạo và tinh chỉnh mô hình
│
└── utils/ # Các hàm tiện ích
```

## Cách đóng góp
Chúng tôi hoan nghênh các đóng góp cho dự án! Đây là cách bạn có thể giúp đỡ:
1. **Fork kho lưu trữ**.
2. **Tạo một nhánh tính năng mới**: `git checkout -b feature/your-feature-name`
3. **Thực hiện các thay đổi của bạn** và cam kết chúng: `git commit -m 'Thêm một số tính năng tuyệt vời'`
4. **Đẩy vào nhánh**: `git push origin feature/your-feature-name`
5. **Mở một yêu cầu kéo**.

Vui lòng đảm bảo viết các bài kiểm tra cho bất kỳ chức năng mới nào và tuân theo kiểu mã hiện có.

## Giấy phép
Dự án này được cấp phép theo Giấy phép MIT. Xem tệp `LICENSE` để biết thêm chi tiết.
