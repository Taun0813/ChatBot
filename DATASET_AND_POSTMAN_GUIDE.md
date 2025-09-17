# 📊 Dataset & Postman Testing Guide

## 🎯 Tổng Quan

Hệ thống AI Agent đã sẵn sàng sử dụng dataset của bạn và có đầy đủ các endpoint để test trên Postman với Hybrid Orchestrator.

---

## 📁 Dataset Integration

### ✅ **Dataset Hiện Có**

Hệ thống đã được cập nhật để sử dụng dataset thực tế của bạn:

```
training/dataset/
└── dataset.json              # Dataset thực tế với 27,000+ sản phẩm điện thoại

data/
├── processed/
│   ├── products.json          # Sản phẩm đã convert từ dataset
│   ├── conversations.json     # Dữ liệu hội thoại
│   ├── knowledge_base.json    # Knowledge base
│   └── training_data.json     # Training data
├── profiles/
│   ├── user_001.json         # Profile user mẫu
│   ├── user_002.json         # Profile user mẫu
│   └── profiles.db           # SQLite database
└── ingest.py                 # Data ingestion module (đã cập nhật)
```

**Dataset Features:**
- ✅ **27,000+ sản phẩm điện thoại** từ các thương hiệu: OnePlus, Samsung, Apple, Xiaomi, Motorola, Realme, Nothing, etc.
- ✅ **Thông số chi tiết**: CPU, RAM, ROM, camera, pin, màn hình, 5G, NFC, etc.
- ✅ **Giá cả thực tế** từ 19,989 VND đến hàng triệu VND
- ✅ **Rating và reviews** từ người dùng thực tế

### 🔄 **Sử Dụng Dataset Của Bạn**

#### 1. **Thay Thế Dataset Sản Phẩm**

Tạo file `data/your_products.json`:

```json
[
  {
    "id": "your_product_1",
    "name": "Tên sản phẩm của bạn",
    "brand": "Thương hiệu",
    "price": 10000000,
    "description": "Mô tả chi tiết sản phẩm",
    "category": "Danh mục",
    "rating": 4.5,
    "reviews_count": 100,
    "availability": "In Stock",
    "specifications": {
      "màn hình": "6.1 inch",
      "chip": "A17 Pro",
      "camera": "48MP",
      "pin": "Lên đến 20 giờ",
      "ram": "8GB",
      "rom": "128GB"
    }
  }
]
```

#### 2. **Cập Nhật Data Ingestion**

Sửa `data/ingest.py`:

```python
def _generate_mock_products(self) -> List[Dict[str, Any]]:
    """Load your custom product data"""
    try:
        with open('data/your_products.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback to mock data
        return self._generate_default_mock_products()
```

#### 3. **Ingest Dataset**

```bash
# Chạy script để ingest dataset của bạn
python init_data.py
```

---

## 🚀 Postman Testing Guide

### 📋 **Setup Postman**

1. **Base URL**: `http://localhost:8000`
2. **Headers**: 
   - `Content-Type: application/json`
   - `Accept: application/json`

### 🔧 **1. Health Check & System Info**

#### **GET /health**
```http
GET http://localhost:8000/health
```

**Expected Response:**
```json
{
    "status": "healthy",
    "message": "AI Agent is running",
    "version": "1.0.0"
}
```

#### **GET /**
```http
GET http://localhost:8000/
```

**Expected Response:**
```json
{
    "message": "AI Agent API - Hybrid Orchestrator",
    "version": "1.0.0",
    "endpoints": {
        "ask": "/ask",
        "chat": "/chat (legacy)",
        "health": "/health",
        "metrics": "/metrics",
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
```

#### **GET /metrics**
```http
GET http://localhost:8000/metrics
```

**Expected Response:**
```json
{
    "status": "success",
    "metrics": {
        "total_requests": 0,
        "rule_based_requests": 0,
        "ml_based_requests": 0,
        "hybrid_requests": 0,
        "average_response_time": 0.0,
        "rule_based_percentage": 0.0,
        "ml_based_percentage": 0.0,
        "hybrid_percentage": 0.0
    },
    "orchestrator_type": "hybrid"
}
```

---

### 🛍️ **2. Product Search (RAG + Hybrid Orchestrator)**

#### **Test 1: Tìm kiếm OnePlus (từ dataset thực tế)**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "OnePlus dưới 50 triệu",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 2: Tìm kiếm Samsung Galaxy (từ dataset)**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "Samsung Galaxy camera 50MP",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 3: Tìm kiếm theo giá thực tế**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "điện thoại từ 20-40 triệu",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 4: Tìm kiếm theo thông số kỹ thuật**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "điện thoại Snapdragon 8GB RAM 256GB",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 5: Tìm kiếm theo tính năng**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "điện thoại có 5G và NFC",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 6: Tìm kiếm Xiaomi (từ dataset)**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "Xiaomi Redmi pin 5000mAh",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 7: Tìm kiếm theo màn hình**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "điện thoại màn hình 6.7 inch 120Hz",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 8: Tìm kiếm Apple iPhone (từ dataset)**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "iPhone 14 dưới 30 triệu",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 9: Tìm kiếm Nothing Phone (từ dataset)**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "Nothing Phone giá rẻ",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 10: Tìm kiếm theo sạc nhanh**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "điện thoại sạc nhanh 100W",
    "user_id": "user123",
    "session_id": "session001"
}
```

**Expected Response Pattern:**
```json
{
    "response": "Tôi đã tìm thấy một số sản phẩm phù hợp với yêu cầu của bạn...",
    "intent": "search",
    "confidence": 0.9,
    "metadata": {
        "search_results": [...],
        "results_count": 3,
        "model_used": "rag",
        "personalized": true,
        "cached": false,
        "orchestrator": {
            "type": "hybrid",
            "selected_router": "ml_based",
            "fusion_weights": {
                "rule_based": 0.4,
                "ml_based": 0.6
            },
            "rule_confidence": 0.8,
            "ml_confidence": 0.9,
            "final_confidence": 0.86,
            "processing_time": 0.15
        }
    },
    "session_id": "session001",
    "user_id": "user123"
}
```

---

### 📦 **3. Order Tracking (API Integration)**

#### **Test 1: Kiểm tra đơn hàng**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "Đơn hàng #1234 đang ở đâu?",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 2: Hủy đơn hàng**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "Tôi muốn hủy đơn hàng #5678",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 3: Đổi trả hàng**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "Tôi muốn đổi trả sản phẩm iPhone 15",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 4: Thanh toán**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "Tôi muốn thanh toán đơn hàng #1234",
    "user_id": "user123",
    "session_id": "session001"
}
```

**Expected Response Pattern:**
```json
{
    "response": "Tôi đã kiểm tra đơn hàng #1234 của bạn. Trạng thái hiện tại là...",
    "intent": "order",
    "confidence": 0.8,
    "metadata": {
        "model_used": "api",
        "order_id": "1234",
        "order_status": "shipped"
    },
    "session_id": "session001",
    "user_id": "user123"
}
```

---

### 💬 **4. General Conversation (Hybrid Orchestrator)**

#### **Test 1: Chào hỏi**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "Xin chào, bạn có thể giúp gì cho tôi?",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 2: Hỏi về dịch vụ**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "Bạn có những dịch vụ gì?",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 3: Hỏi về chính sách**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "Chính sách đổi trả như thế nào?",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 4: Hỏi về bảo hành**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "Sản phẩm này có bảo hành bao lâu?",
    "user_id": "user123",
    "session_id": "session001"
}
```

**Expected Response Pattern:**
```json
{
    "response": "Xin chào! Tôi có thể giúp bạn tìm kiếm sản phẩm, kiểm tra đơn hàng, và trả lời các câu hỏi...",
    "intent": "chat",
    "confidence": 0.8,
    "metadata": {
        "model_used": "interaction",
        "orchestrator": {
            "type": "hybrid",
            "selected_router": "rule_based",
            "fusion_weights": {
                "rule_based": 0.4,
                "ml_based": 0.6
            },
            "rule_confidence": 0.9,
            "ml_confidence": 0.7,
            "final_confidence": 0.78,
            "processing_time": 0.12
        }
    },
    "session_id": "session001",
    "user_id": "user123"
}
```

---

### 👤 **5. Personalization Testing**

#### **Test 1: User với preference iPhone**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "điện thoại tốt",
    "user_id": "user_iphone_lover",
    "session_id": "session002"
}
```

#### **Test 2: User với preference Samsung**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "điện thoại tốt",
    "user_id": "user_samsung_lover",
    "session_id": "session003"
}
```

#### **Test 3: User mới (chưa có preference)**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "điện thoại tốt",
    "user_id": "new_user",
    "session_id": "session004"
}
```

**Expected Response Pattern:**
```json
{
    "response": "Dựa trên sở thích của bạn, tôi recommend iPhone 15 Pro...",
    "intent": "search",
    "confidence": 0.9,
    "metadata": {
        "search_results": [...],
        "results_count": 3,
        "model_used": "rag",
        "personalized": true,
        "cached": false,
        "user_preferences": {
            "preferred_brands": ["Apple"],
            "price_range": "high",
            "previous_searches": ["iPhone", "Apple"]
        }
    },
    "session_id": "session002",
    "user_id": "user_iphone_lover"
}
```

---

### 🔄 **6. Hybrid Orchestrator Testing**

#### **Test 1: Force Rule-based Routing**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "iPhone",
    "user_id": "user123",
    "session_id": "session001",
    "intent": "search"
}
```

#### **Test 2: Force ML-based Routing**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "Tôi cần một chiếc điện thoại để chụp ảnh đẹp",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 3: Ambiguous Query (Hybrid Decision)**
```http
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "Tôi muốn mua điện thoại nhưng không biết chọn gì",
    "user_id": "user123",
    "session_id": "session001"
}
```

---

### ⚡ **7. Performance Testing**

#### **Test 1: Cache Hit Test**
```http
# Lần 1: Cache miss
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "iPhone dưới 30 triệu",
    "user_id": "user123",
    "session_id": "session001"
}

# Lần 2: Cache hit (nhanh hơn)
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "iPhone dưới 30 triệu",
    "user_id": "user123",
    "session_id": "session001"
}
```

#### **Test 2: Concurrent Requests**
```http
# Gửi nhiều request cùng lúc để test performance
POST http://localhost:8000/ask
Content-Type: application/json

{
    "message": "Samsung Galaxy",
    "user_id": "user123",
    "session_id": "session001"
}
```

---

### 📊 **8. Monitoring & Debugging**

#### **Check Metrics After Testing**
```http
GET http://localhost:8000/metrics
```

**Expected Response:**
```json
{
    "status": "success",
    "metrics": {
        "total_requests": 25,
        "rule_based_requests": 8,
        "ml_based_requests": 10,
        "hybrid_requests": 7,
        "average_response_time": 145.2,
        "rule_based_percentage": 32.0,
        "ml_based_percentage": 40.0,
        "hybrid_percentage": 28.0
    },
    "orchestrator_type": "hybrid"
}
```

---

## 🎯 **Test Scenarios Tổng Hợp**

### **Scenario 1: New User Journey**
1. Health check
2. Chào hỏi
3. Tìm kiếm sản phẩm
4. Hỏi về đơn hàng
5. Check metrics

### **Scenario 2: Returning User Journey**
1. Tìm kiếm với user_id cũ
2. Kiểm tra personalization
3. Hỏi về đơn hàng cũ
4. Tìm kiếm sản phẩm mới

### **Scenario 3: Error Handling**
1. Invalid JSON
2. Empty message
3. Very long message
4. Special characters

### **Scenario 4: Performance Testing**
1. Multiple concurrent requests
2. Cache testing
3. Long conversation session
4. Memory usage monitoring

---

## 🚀 **Quick Start Commands**

```bash
# 1. Khởi động server
python app.py

# 2. Init data với dataset thực tế (chạy 1 lần)
python init_data.py

# 3. Test với dataset thực tế
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "OnePlus dưới 50 triệu", "user_id": "user123"}'

# 4. Test Samsung từ dataset
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "Samsung Galaxy camera 50MP", "user_id": "user123"}'

# 5. Check metrics
curl http://localhost:8000/metrics
```

## 📊 **Dataset Information**

**Dataset của bạn có:**
- ✅ **27,000+ sản phẩm điện thoại** thực tế
- ✅ **Các thương hiệu**: OnePlus, Samsung, Apple, Xiaomi, Motorola, Realme, Nothing, etc.
- ✅ **Thông số chi tiết**: CPU, RAM, ROM, camera, pin, màn hình, 5G, NFC, sạc nhanh
- ✅ **Giá cả thực tế**: Từ 19,989 VND đến hàng triệu VND
- ✅ **Rating**: Từ 0-100 (đã convert thành 0-5 sao)

**Hệ thống sẽ:**
1. Tự động load 100 sản phẩm đầu tiên từ dataset
2. Convert format để phù hợp với RAG system
3. Index vào Pinecone vector database
4. Sử dụng cho semantic search và recommendations

---

## 📝 **Notes**

1. **Dataset**: Hệ thống đã sẵn sàng sử dụng dataset của bạn
2. **Hybrid Orchestrator**: Tự động chọn rule-based hoặc ML-based routing
3. **Personalization**: Học từ user behavior và preferences
4. **Caching**: Tự động cache để tối ưu performance
5. **Monitoring**: Real-time metrics và performance tracking

Hệ thống đã sẵn sàng cho production testing! 🎉
