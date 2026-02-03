# 🚀 Kế hoạch Tích hợp AI Agent với Spring Boot Microservices

## 📋 Tổng quan

Tài liệu này mô tả chi tiết cách merge **AI Agent System (Python/FastAPI)** với **Spring Boot Microservices (Java)** và tích hợp Frontend.

---

## 🏗 Kiến trúc Tổng thể

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                          │
│  (React/Vue/Angular - Port 3000)                               │
│  • Chat Interface                                                │
│  • Product Catalog                                               │
│  • Order Management                                              │
│  • User Dashboard                                                │
└──────────────┬──────────────────────────────┬──────────────────┘
               │                                │
               │                                │
    ┌──────────▼──────────┐        ┌───────────▼──────────┐
    │   API Gateway       │        │   AI Agent API        │
    │   (Spring Boot)     │        │   (FastAPI)           │
    │   Port: 8181        │        │   Port: 8000         │
    └──────────┬──────────┘        └──────────┬───────────┘
               │                                │
               │                                │
    ┌──────────▼────────────────────────────────▼──────────┐
    │         Spring Boot Microservices                     │
    │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
    │  │ User Service │  │Product Service│ │Order Service│ │
    │  │   (8081)     │  │   (8082)      │ │  (8084)     │ │
    │  └──────────────┘  └──────────────┘  └────────────┘ │
    │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
    │  │Payment Service│  │Cart Service  │ │Warranty Svc │ │
    │  │   (8085)      │  │   (8086)      │ │  (8088)     │ │
    │  └──────────────┘  └──────────────┘  └────────────┘ │
    └──────────────────────────────────────────────────────┘
               │                                │
    ┌──────────▼────────────────────────────────▼──────────┐
    │              Infrastructure Layer                      │
    │  • PostgreSQL (5433)                                   │
    │  • Redis (6379)                                         │
    │  • RabbitMQ (5672)                                     │
    │  • Pinecone (Cloud)                                    │
    └─────────────────────────────────────────────────────────┘
```

---

## 🔄 Luồng Tích hợp

### 1. **Chat Flow với AI Agent**

```
User → Frontend → AI Agent (8000) → API Agent → Spring Boot Services
                                              ↓
                                    API Gateway (8181)
                                              ↓
                                    Microservices
```

### 2. **Product Search Flow**

```
User: "Tìm điện thoại iPhone dưới 30 triệu"
  ↓
AI Agent (RAG Agent)
  ↓
Pinecone Vector Search
  ↓
Product Results
  ↓
AI Agent → Product Service (8082) via API Gateway (8181)
  ↓
Return formatted response to Frontend
```

### 3. **Order Flow**

```
User: "Tôi muốn đặt hàng iPhone 16"
  ↓
AI Agent (API Agent)
  ↓
API Gateway (8181) → Order Service (8084)
  ↓
Order Service → Inventory Service → Payment Service
  ↓
Response → AI Agent → Frontend
```

---

## 📝 Các Bước Tích hợp

### Phase 1: Cấu hình API Endpoints trong AI Agent

#### 1.1. Cập nhật `config.py`

```python
# config.py
class Settings(BaseSettings):
    # Spring Boot Services URLs (qua API Gateway)
    order_service_url: str = "http://localhost:8181/api/orders"
    payment_service_url: str = "http://localhost:8181/api/payments"
    warranty_service_url: str = "http://localhost:8181/api/warranties"
    product_service_url: str = "http://localhost:8181/api/products"
    user_service_url: str = "http://localhost:8181/api/users"
    cart_service_url: str = "http://localhost:8181/api/carts"
    
    # API Gateway Authentication
    api_gateway_api_key: Optional[str] = None
    jwt_token: Optional[str] = None  # JWT token từ User Service
    
    # API timeout
    api_timeout: int = 30
```

#### 1.2. Cập nhật `core/api_model.py`

Thêm JWT authentication và error handling tốt hơn:

```python
# core/api_model.py
class APIModel:
    def __init__(self, config: Dict[str, Any]):
        self.jwt_token = config.get("jwt_token")
        self.api_gateway_url = config.get("api_gateway_url", "http://localhost:8181")
        
    async def _make_authenticated_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None
    ):
        """Make authenticated request to Spring Boot services"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.jwt_token}" if self.jwt_token else None
        }
        
        url = f"{self.api_gateway_url}{endpoint}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(method, url, json=data, headers=headers)
            return response.json()
```

### Phase 2: Tạo Service Client trong AI Agent

#### 2.1. Tạo `services/spring_boot_client.py`

```python
# services/spring_boot_client.py
"""
Client để giao tiếp với Spring Boot Microservices qua API Gateway
"""
import httpx
import logging
from typing import Dict, Any, Optional, List
from config import get_settings

logger = logging.getLogger(__name__)

class SpringBootClient:
    """Client để giao tiếp với Spring Boot services"""
    
    def __init__(self):
        self.settings = get_settings()
        self.api_gateway_url = "http://localhost:8181"
        self.jwt_token = self.settings.jwt_token
        self.timeout = self.settings.api_timeout
        
    async def get_products(
        self, 
        search: Optional[str] = None,
        category: Optional[str] = None,
        page: int = 0,
        size: int = 20
    ) -> Dict[str, Any]:
        """Get products from Product Service"""
        try:
            params = {"page": page, "size": size}
            if search:
                params["search"] = search
            if category:
                params["category"] = category
                
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = self._get_headers()
                response = await client.get(
                    f"{self.api_gateway_url}/api/products",
                    params=params,
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error getting products: {e}")
            return {"content": [], "totalElements": 0}
    
    async def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product by ID"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = self._get_headers()
                response = await client.get(
                    f"{self.api_gateway_url}/api/products/{product_id}",
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error getting product {product_id}: {e}")
            return None
    
    async def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create order via Order Service"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = self._get_headers()
                response = await client.post(
                    f"{self.api_gateway_url}/api/orders",
                    json=order_data,
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise
    
    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order by ID"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = self._get_headers()
                response = await client.get(
                    f"{self.api_gateway_url}/api/orders/{order_id}",
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return None
    
    async def add_to_cart(self, cart_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add item to cart"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = self._get_headers()
                response = await client.post(
                    f"{self.api_gateway_url}/api/carts/items",
                    json=cart_data,
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error adding to cart: {e}")
            raise
    
    async def get_cart(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's cart"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = self._get_headers()
                response = await client.get(
                    f"{self.api_gateway_url}/api/carts/user/{user_id}",
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error getting cart: {e}")
            return None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        headers = {"Content-Type": "application/json"}
        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        return headers
```

#### 2.2. Cập nhật `core/api_model.py` để sử dụng SpringBootClient

```python
# core/api_model.py - Thêm vào class APIModel
from services.spring_boot_client import SpringBootClient

class APIModel:
    def __init__(self, config: Dict[str, Any]):
        # ... existing code ...
        self.spring_boot_client = SpringBootClient()
        
    async def process_order_inquiry(self, message: str, user_id: str) -> str:
        """Process order inquiry using Spring Boot Order Service"""
        # Extract order ID from message
        order_id = self._extract_order_id(message)
        
        if order_id:
            order = await self.spring_boot_client.get_order(order_id)
            if order:
                return self._format_order_response(order)
        
        return "Tôi không tìm thấy thông tin đơn hàng. Vui lòng cung cấp mã đơn hàng."
```

### Phase 3: Cập nhật API Agent để sử dụng Spring Boot Services

#### 3.1. Cập nhật `core/models/api_agent.py`

```python
# core/models/api_agent.py
from services.spring_boot_client import SpringBootClient

class APIAgent(BaseAgent):
    async def process(self, request: Dict[str, Any]) -> AgentResponse:
        """Process API requests using Spring Boot services"""
        message = request.get("message", "")
        user_id = request.get("user_id")
        intent = request.get("intent", "api_call")
        
        client = SpringBootClient()
        
        # Route to appropriate service
        if "đơn hàng" in message.lower() or "order" in message.lower():
            # Order inquiry
            order_id = self._extract_order_id(message)
            if order_id:
                order = await client.get_order(order_id)
                response_text = self._format_order(order) if order else "Không tìm thấy đơn hàng"
            else:
                response_text = "Vui lòng cung cấp mã đơn hàng"
                
        elif "giỏ hàng" in message.lower() or "cart" in message.lower():
            # Cart operations
            cart = await client.get_cart(user_id)
            response_text = self._format_cart(cart) if cart else "Giỏ hàng trống"
            
        elif "thanh toán" in message.lower() or "payment" in message.lower():
            # Payment operations
            # Implement payment logic
            response_text = "Tính năng thanh toán đang được phát triển"
            
        else:
            response_text = "Tôi không hiểu yêu cầu của bạn. Vui lòng thử lại."
        
        return AgentResponse(
            content=response_text,
            confidence=0.9,
            agent_name="api_agent",
            metadata={"service": "spring_boot"}
        )
```

### Phase 4: Tạo Frontend Integration Layer

#### 4.1. Frontend API Service (JavaScript/TypeScript)

```typescript
// frontend/src/services/apiService.ts
const API_GATEWAY_URL = 'http://localhost:8181';
const AI_AGENT_URL = 'http://localhost:8000';

class ApiService {
  private getAuthHeaders() {
    const token = localStorage.getItem('jwt_token');
    return {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` })
    };
  }

  // AI Agent Chat
  async sendChatMessage(message: string, userId: string, sessionId?: string) {
    const response = await fetch(`${AI_AGENT_URL}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        user_id: userId,
        session_id: sessionId
      })
    });
    return response.json();
  }

  // Spring Boot Services
  async getProducts(search?: string, category?: string) {
    const params = new URLSearchParams();
    if (search) params.append('search', search);
    if (category) params.append('category', category);
    
    const response = await fetch(
      `${API_GATEWAY_URL}/api/products?${params}`,
      { headers: this.getAuthHeaders() }
    );
    return response.json();
  }

  async getOrder(orderId: string) {
    const response = await fetch(
      `${API_GATEWAY_URL}/api/orders/${orderId}`,
      { headers: this.getAuthHeaders() }
    );
    return response.json();
  }

  async addToCart(productId: string, quantity: number) {
    const response = await fetch(
      `${API_GATEWAY_URL}/api/carts/items`,
      {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify({ productId, quantity })
      }
    );
    return response.json();
  }
}

export default new ApiService();
```

#### 4.2. Frontend Chat Component

```typescript
// frontend/src/components/ChatInterface.tsx
import React, { useState } from 'react';
import ApiService from '../services/apiService';

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const userId = localStorage.getItem('user_id') || 'user123';

  const handleSend = async () => {
    if (!input.trim()) return;

    // Add user message
    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);

    // Send to AI Agent
    const response = await ApiService.sendChatMessage(input, userId);
    
    // Add AI response
    const aiMessage = { role: 'assistant', content: response.response };
    setMessages(prev => [...prev, aiMessage]);

    setInput('');
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
      </div>
      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Nhập tin nhắn..."
        />
        <button onClick={handleSend}>Gửi</button>
      </div>
    </div>
  );
};

export default ChatInterface;
```

### Phase 5: Authentication Integration

#### 5.1. Cập nhật AI Agent để nhận JWT token

```python
# app.py - Thêm authentication endpoint
@app.post("/auth/login")
async def login(credentials: Dict[str, str]):
    """Login và lấy JWT token từ Spring Boot User Service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8181/api/auth/login",
                json=credentials
            )
            response.raise_for_status()
            token_data = response.json()
            
            # Store token in config or return to client
            return token_data
    except Exception as e:
        raise HTTPException(status_code=401, detail="Login failed")
```

#### 5.2. Middleware để validate JWT trong AI Agent (optional)

```python
# app.py
from fastapi import Header, HTTPException

async def verify_token(authorization: str = Header(None)):
    """Verify JWT token from Spring Boot"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    token = authorization.split(" ")[1]
    # Validate token với Spring Boot User Service
    # Hoặc decode và verify locally nếu có shared secret
    return token

@app.post("/ask")
async def ask(
    request: ChatRequest,
    token: str = Depends(verify_token),  # Optional: chỉ cần nếu muốn auth
    router = Depends(get_router)
):
    # ... existing code ...
```

---

## 🚀 Deployment Strategy

### Option 1: Docker Compose (Recommended)

```yaml
# docker-compose.integrated.yml
version: '3.8'

services:
  # Infrastructure
  postgresql:
    image: postgres:15
    ports:
      - "5433:5432"
    environment:
      POSTGRES_DB: ai_agent
      POSTGRES_USER: ai_agent
      POSTGRES_PASSWORD: 123

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"

  # Spring Boot Services
  discovery-service:
    build: ./be_v3/Backend_Chatbot/discovery-service
    ports:
      - "8761:8761"

  api-gateway:
    build: ./be_v3/Backend_Chatbot/api-gateway
    ports:
      - "8181:8181"
    depends_on:
      - discovery-service

  # ... other Spring Boot services ...

  # AI Agent
  ai-agent:
    build: ./AI_Agent
    ports:
      - "8000:8000"
    environment:
      - ORDER_SERVICE_URL=http://api-gateway:8181/api/orders
      - PRODUCT_SERVICE_URL=http://api-gateway:8181/api/products
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - api-gateway

  # Frontend
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - ai-agent
      - api-gateway
```

### Option 2: Kubernetes (Production)

Tạo các Kubernetes manifests cho từng service.

---

## 📊 Monitoring & Observability

### 1. Unified Logging

- Sử dụng **ELK Stack** hoặc **Loki** để aggregate logs từ cả Python và Java services

### 2. Distributed Tracing

- **Jaeger** hoặc **Zipkin** để trace requests qua cả 2 systems

### 3. Metrics

- **Prometheus** + **Grafana** để monitor:
  - AI Agent metrics (FastAPI)
  - Spring Boot Actuator metrics
  - Infrastructure metrics

---

## ✅ Checklist Tích hợp

### Backend Integration
- [ ] Cập nhật `config.py` với Spring Boot service URLs
- [ ] Tạo `SpringBootClient` class
- [ ] Cập nhật `APIModel` để sử dụng Spring Boot services
- [ ] Cập nhật `APIAgent` với các operations mới
- [ ] Test JWT authentication flow
- [ ] Test product search integration
- [ ] Test order operations
- [ ] Test cart operations

### Frontend Integration
- [ ] Tạo API service layer
- [ ] Tạo Chat interface component
- [ ] Tích hợp authentication
- [ ] Tích hợp product catalog
- [ ] Tích hợp order management
- [ ] Error handling và loading states

### Deployment
- [ ] Docker Compose configuration
- [ ] Environment variables setup
- [ ] Network configuration
- [ ] Health checks
- [ ] Load balancing (nếu cần)

### Testing
- [ ] Integration tests cho API calls
- [ ] End-to-end tests
- [ ] Performance testing
- [ ] Security testing

---

## 🔐 Security Considerations

1. **JWT Token Management**
   - Store tokens securely
   - Implement token refresh
   - Validate tokens on both sides

2. **CORS Configuration**
   - Configure CORS cho cả AI Agent và API Gateway
   - Whitelist frontend domain

3. **Rate Limiting**
   - API Gateway đã có rate limiting
   - Thêm rate limiting cho AI Agent nếu cần

4. **API Keys**
   - Secure storage cho API keys
   - Rotate keys regularly

---

## 📚 Next Steps

1. **Immediate**: Implement Phase 1-3 (Backend integration)
2. **Short-term**: Implement Frontend integration (Phase 4)
3. **Medium-term**: Add authentication flow (Phase 5)
4. **Long-term**: Production deployment và monitoring

---

## 🆘 Troubleshooting

### AI Agent không kết nối được Spring Boot services

1. Kiểm tra API Gateway đang chạy: `http://localhost:8181/actuator/health`
2. Kiểm tra network connectivity
3. Kiểm tra JWT token validity
4. Xem logs: `docker-compose logs ai-agent`

### Frontend không kết nối được AI Agent

1. Kiểm tra CORS configuration
2. Kiểm tra AI Agent đang chạy: `http://localhost:8000/health`
3. Kiểm tra network requests trong browser DevTools

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: 📋 Planning Phase
