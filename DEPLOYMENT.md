# Hướng dẫn Deploy và Quản lý Source Code

Tài liệu hướng dẫn chi tiết cách build, push Docker image và xử lý các vấn đề thường gặp khi clone source code sang máy mới.

## 1. Chuẩn bị Môi trường

Đảm bảo máy của bạn đã cài đặt:
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Git](https://git-scm.com/)

## 2. Docker Deployment

### Bước 1: Build Docker Image
Mở terminal tại thư mục gốc của dự án và chạy lệnh:

```bash
# Build image với tag là 'ai-agent:v1'
docker build -t ai-agent:v1 .
```

*Lưu ý: Quá trình build lần đầu có thể mất vài phút để tải các thư viện.*

### Bước 2: Chạy thử Image
```bash
docker run -d -p 8000:8000 --env-file .env ai-agent:v1
```
Kiểm tra tại: `http://localhost:8000/health`

### Bước 3: Push lên DockerHub
Để chia sẻ image cho các máy khác hoặc deploy lên server, bạn cần push lên DockerHub.

1. **Đăng nhập DockerHub**:
   ```bash
   docker login
   ```
2. **Tag image đúng chuẩn**:
   Cú pháp: `docker tag <tên-image-local> <username-dockerhub>/<tên-repo>:<tag>`
   ```bash
   # Ví dụ: username là 'taun0813'
   docker tag ai-agent:v1 taun0813/ai-agent:latest
   ```
3. **Push image**:
   ```bash
   docker push taun0813/ai-agent:latest
   ```

### Bước 4: Pull và Run trên máy khác
Trên máy khác (server hoặc máy cá nhân khác):
```bash
# Pull image về
docker pull taun0813/ai-agent:latest

# Run container (nhớ copy file .env sang máy mới)
docker run -d -p 8000:8000 --env-file .env taun0813/ai-agent:latest
```

### Bước 5: Docker Compose (Phát triển Local)

Sử dụng Docker Compose để chạy AI Agent + Redis (tùy chọn) trong môi trường development:

```bash
# Tạo file .env từ env.example trước
cp env.example .env
# Điền GEMINI_API_KEY hoặc GROQ_API_KEY vào .env

# Chạy AI Agent
docker-compose up -d

# Chạy kèm Redis (khi dùng cache Redis)
docker-compose --profile full up -d

# Xem logs
docker-compose logs -f ai-agent

# Dừng
docker-compose down
```

Kiểm tra: `http://localhost:8000/health`

## 3. Quản lý Git & Fix lỗi Clone

Khi clone repo này sang máy khác, lỗi thường gặp nhất là thiếu thư viện hoặc xung đột phiên bản Python.

### Các bước chuẩn khi sang máy mới

1. **Clone repo**:
   ```bash
   git clone <link-repo>
   cd ai-agent
   ```

2. **Tạo môi trường ảo (Virtual Env)**:
   *Bắt buộc* để tránh lỗi xung đột.
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Cài đặt thư viện**:
   File `requirements.txt` đã được cập nhật để ổn định hơn (đã thêm pandas, numpy).
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**:
   Copy `env.example` thành `.env` và điền key:
   ```bash
   cp env.example .env
   # Điền API Key vào .env
   ```

### Xử lý sự cố thường gặp (Troubleshooting)

**Lỗi: `ModuleNotFoundError: No module named 'pandas'`**
- Nguyên nhân: Chưa cài đủ thư viện.
- Khắc phục: Chạy lại `pip install -r requirements.txt`. Đảm bảo file requirements có dòng `pandas` (đã được thêm ở bản cập nhật mới nhất).

**Lỗi: `ERROR: Could not build wheels for...`**
- Nguyên nhân: Thiếu C++ build tools (thường gặp trên Windows).
- Khắc phục: Cài [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) hoặc dùng Docker để tránh lỗi hệ điều hành.

**Lỗi: Kết nối Pinecone thất bại**
- Nguyên nhân: API Key sai hoặc mạng chặn.
- Khắc phục: Kiểm tra biến `PINECONE_API_KEY` trong file `.env`.

## 4. Kiểm tra hệ thống sau khi deploy

Sau khi chạy, hãy kiểm tra:
1. **Health Check**: `GET /health` -> Trả về `{"status": "healthy"}`
2. **Load Data**: `GET /training/status` (nếu pipeline đang chạy)
3. **Test Chat**: Gửi request POST tới `/ask` để test model.

---
**Note**: Dataset mới `Mobiles Dataset (2025).csv` đã được tích hợp. Khi hệ thống khởi động lại, nếu chạy lệnh `init_data.py`, nó sẽ tự động load data mới này (giá đã được chuyển sang VND).

---

## 5. Deploy lên Railway (Cho Frontend Testing)

Railway là một platform đơn giản để deploy ứng dụng Python/FastAPI. Hướng dẫn chi tiết:

### 5.1. Chuẩn bị

1. **Đăng ký tài khoản Railway**:
   - Truy cập: https://railway.app/
   - Đăng ký bằng GitHub/GitLab account (miễn phí)

2. **Đảm bảo code đã push lên GitHub/GitLab**:
   ```bash
   git add .
   git commit -m "Prepare for Railway deployment"
   git push origin main
   ```

### 5.2. Tạo Project trên Railway

1. **Tạo New Project**:
   - Đăng nhập Railway dashboard
   - Click "New Project"
   - Chọn "Deploy from GitHub repo"
   - Chọn repository của bạn
   - Railway sẽ tự động detect Dockerfile

2. **Cấu hình Build**:
   - Railway sẽ tự động sử dụng `dockerfile` trong repo
   - File `railway.json` đã được tạo để cấu hình Railway

### 5.3. Cấu hình Environment Variables

**QUAN TRỌNG**: Bạn cần set các biến môi trường sau trong Railway dashboard:

#### Bước 1: Vào Variables Tab
- Trong Railway project, click vào service của bạn
- Chọn tab "Variables"

#### Bước 2: Thêm các biến bắt buộc

**API Keys (Bắt buộc ít nhất 1 trong các key sau):**
```
GEMINI_API_KEY=your_gemini_api_key_here
# HOẶC
GROQ_API_KEY=your_groq_api_key_here
# HOẶC
OPENAI_API_KEY=your_openai_api_key_here
```

**Model Configuration:**
```
MODEL_LOADER_BACKEND=gemini
MODEL_NAME=gemini-2.5-flash
MAX_TOKENS=2048
TEMPERATURE=0.7
TOP_P=0.9
```

**Pinecone Configuration (Nếu dùng RAG):**
```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-west1-gcp-free
PINECONE_INDEX_NAME=product-search
PINECONE_DIMENSION=1024
PINECONE_METRIC=cosine
```

**Server Configuration:**
```
API_HOST=0.0.0.0
API_PORT=8000
# Railway sẽ tự động set PORT, nhưng bạn có thể override
```

**Optional - External Services (Nếu có Spring Boot backend):**
```
ORDER_SERVICE_URL=https://your-springboot-api.com/api/orders
PAYMENT_SERVICE_URL=https://your-springboot-api.com/api/payments
WARRANTY_SERVICE_URL=https://your-springboot-api.com/api/warranties
PRODUCT_SERVICE_URL=https://your-springboot-api.com/api/products
```

**Optional - Personalization:**
```
ENABLE_PERSONALIZATION=false
ENABLE_RECOMMENDATIONS=false
ENABLE_RL_LEARNING=false
```

**Optional - Hybrid Orchestrator:**
```
ENABLE_HYBRID_ORCHESTRATOR=true
```

#### Bước 3: Copy từ env.example
Bạn có thể copy tất cả các biến từ file `env.example` và điền giá trị thực tế vào Railway Variables.

### 5.4. Deploy và Kiểm tra

1. **Railway sẽ tự động deploy**:
   - Sau khi set variables, Railway sẽ tự động build và deploy
   - Xem logs trong tab "Deployments" để theo dõi quá trình

2. **Kiểm tra Health Check**:
   - Railway sẽ cung cấp URL công khai (ví dụ: `https://your-app.up.railway.app`)
   - Test endpoint: `https://your-app.up.railway.app/health`
   - Response mong đợi: `{"status": "healthy", ...}`

3. **Test API từ Frontend**:
   ```javascript
   // Trong frontend code
   const AI_AGENT_URL = 'https://your-app.up.railway.app';
   
   // Test health check
   fetch(`${AI_AGENT_URL}/health`)
     .then(res => res.json())
     .then(data => console.log(data));
   
   // Test chat endpoint
   fetch(`${AI_AGENT_URL}/ask`, {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({
       message: "Xin chào",
       user_id: "test_user"
     })
   })
     .then(res => res.json())
     .then(data => console.log(data));
   ```

### 5.5. Cấu hình Custom Domain (Optional)

1. Vào tab "Settings" trong Railway service
2. Click "Generate Domain" để có domain tùy chỉnh
3. Hoặc thêm custom domain của bạn

### 5.6. Monitoring và Logs

- **View Logs**: Tab "Deployments" > Click vào deployment > Xem logs
- **Metrics**: Railway cung cấp metrics về CPU, Memory, Network
- **Health Checks**: Railway tự động monitor health endpoint

### 5.7. Troubleshooting Railway Deployment

**Lỗi: Build failed**
- Kiểm tra logs trong Railway dashboard
- Đảm bảo `requirements.txt` có đầy đủ dependencies
- Kiểm tra Dockerfile syntax

**Lỗi: Application crashed**
- Kiểm tra logs để xem lỗi cụ thể
- Đảm bảo đã set đầy đủ environment variables
- Kiểm tra API keys có hợp lệ không

**Lỗi: Health check failed**
- Kiểm tra `/health` endpoint có hoạt động không
- Xem logs để tìm lỗi khởi tạo router
- Kiểm tra các dependencies như Pinecone connection

**Lỗi: PORT not found**
- Railway tự động set PORT, nhưng nếu có lỗi, thêm vào Variables:
  ```
  PORT=8000
  ```

**Lỗi: CORS khi gọi từ Frontend**
- FastAPI đã config CORS với `allow_origins=["*"]`
- Nếu vẫn lỗi, kiểm tra frontend URL có đúng không

### 5.8. Cập nhật Code

Mỗi khi push code mới lên GitHub:
1. Railway sẽ tự động detect changes
2. Tự động build và deploy version mới
3. Có thể xem progress trong tab "Deployments"

### 5.9. Frontend Integration

Sau khi deploy thành công, cập nhật frontend:

```typescript
// .env trong frontend project
REACT_APP_AI_AGENT_URL=https://your-app.up.railway.app

// Hoặc trong code
const AI_AGENT_URL = process.env.REACT_APP_AI_AGENT_URL || 'https://your-app.up.railway.app';
```

### 5.10. Cost và Limits

- **Free Tier**: Railway cung cấp $5 credit miễn phí mỗi tháng
- **Usage**: Monitor trong tab "Usage" để theo dõi chi phí
- **Recommendation**: Tắt service khi không dùng để tiết kiệm credits

---

## 6. Quick Reference - Railway Deployment Checklist

- [ ] Code đã push lên GitHub/GitLab
- [ ] Đã tạo Railway project và connect repo
- [ ] Đã set GEMINI_API_KEY hoặc GROQ_API_KEY
- [ ] Đã set MODEL_LOADER_BACKEND và MODEL_NAME
- [ ] Đã set PINECONE_API_KEY (nếu dùng RAG)
- [ ] Đã test `/health` endpoint
- [ ] Đã test `/ask` endpoint từ frontend
- [ ] Đã cập nhật frontend URL trong code
- [ ] Đã test CORS từ frontend

---

**Lưu ý quan trọng**:
- Railway sẽ tự động set PORT environment variable
- Không cần set PORT trong Variables (trừ khi có vấn đề)
- Đảm bảo API keys hợp lệ trước khi deploy
- Monitor logs để debug nếu có lỗi
