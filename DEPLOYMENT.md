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
