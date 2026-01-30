# 🎨 Frontend Integration Guide

Hướng dẫn tích hợp Frontend với AI Agent và Spring Boot Microservices.

## 📋 Tổng quan

Frontend sẽ giao tiếp với 2 backend services:
1. **AI Agent API** (Port 8000) - Chat và AI interactions
2. **Spring Boot API Gateway** (Port 8181) - E-commerce operations

---

## 🏗 Kiến trúc Frontend

```
Frontend Application
├── Services Layer
│   ├── aiAgentService.ts    # AI Agent API calls
│   ├── apiService.ts         # Spring Boot API calls
│   └── authService.ts        # Authentication
├── Components
│   ├── ChatInterface.tsx     # AI Chat UI
│   ├── ProductCatalog.tsx    # Product listing
│   ├── OrderManagement.tsx   # Order operations
│   └── Cart.tsx              # Shopping cart
└── State Management
    ├── Auth Context          # User authentication
    └── Cart Context          # Shopping cart state
```

---

## 🔧 Implementation

### 1. API Service Layer

#### `src/services/aiAgentService.ts`

```typescript
const AI_AGENT_URL = process.env.REACT_APP_AI_AGENT_URL || 'http://localhost:8000';

export interface ChatRequest {
  message: string;
  user_id?: string;
  session_id?: string;
  context?: Record<string, any>;
  intent?: 'search' | 'chat' | 'api_call';
}

export interface ChatResponse {
  user_id: string;
  response: string;
  intent: string;
  confidence: number;
  metadata?: Record<string, any>;
  session_id?: string;
}

class AIAgentService {
  private baseURL = AI_AGENT_URL;

  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${this.baseURL}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`AI Agent error: ${response.statusText}`);
    }

    return response.json();
  }

  async getHealth(): Promise<any> {
    const response = await fetch(`${this.baseURL}/health`);
    return response.json();
  }

  async getMetrics(): Promise<any> {
    const response = await fetch(`${this.baseURL}/metrics`);
    return response.json();
  }
}

export default new AIAgentService();
```

#### `src/services/apiService.ts`

```typescript
const API_GATEWAY_URL = process.env.REACT_APP_API_GATEWAY_URL || 'http://localhost:8181';

class ApiService {
  private baseURL = API_GATEWAY_URL;

  private getAuthHeaders(): HeadersInit {
    const token = localStorage.getItem('jwt_token');
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    return headers;
  }

  // Authentication
  async login(username: string, password: string) {
    const response = await fetch(`${this.baseURL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
      throw new Error('Login failed');
    }

    const data = await response.json();
    if (data.accessToken) {
      localStorage.setItem('jwt_token', data.accessToken);
      localStorage.setItem('refresh_token', data.refreshToken || '');
    }

    return data;
  }

  async logout() {
    localStorage.removeItem('jwt_token');
    localStorage.removeItem('refresh_token');
  }

  // Products
  async getProducts(params?: {
    search?: string;
    category?: string;
    page?: number;
    size?: number;
  }) {
    const queryParams = new URLSearchParams();
    if (params?.search) queryParams.append('search', params.search);
    if (params?.category) queryParams.append('category', params.category);
    if (params?.page) queryParams.append('page', params.page.toString());
    if (params?.size) queryParams.append('size', params.size.toString());

    const response = await fetch(
      `${this.baseURL}/api/products?${queryParams}`,
      { headers: this.getAuthHeaders() }
    );

    return response.json();
  }

  async getProduct(id: string) {
    const response = await fetch(`${this.baseURL}/api/products/${id}`, {
      headers: this.getAuthHeaders(),
    });
    return response.json();
  }

  // Orders
  async getOrder(id: string) {
    const response = await fetch(`${this.baseURL}/api/orders/${id}`, {
      headers: this.getAuthHeaders(),
    });
    return response.json();
  }

  async getUserOrders(userId: string) {
    const response = await fetch(`${this.baseURL}/api/orders/user/${userId}`, {
      headers: this.getAuthHeaders(),
    });
    return response.json();
  }

  async createOrder(orderData: any) {
    const response = await fetch(`${this.baseURL}/api/orders`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: JSON.stringify(orderData),
    });
    return response.json();
  }

  // Cart
  async getCart(userId: string) {
    const response = await fetch(`${this.baseURL}/api/carts/user/${userId}`, {
      headers: this.getAuthHeaders(),
    });
    return response.json();
  }

  async addToCart(userId: string, productId: string, quantity: number = 1) {
    const response = await fetch(`${this.baseURL}/api/carts/items`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: JSON.stringify({
        userId,
        productId,
        quantity,
      }),
    });
    return response.json();
  }

  async removeFromCart(userId: string, itemId: string) {
    const response = await fetch(`${this.baseURL}/api/carts/items/${itemId}`, {
      method: 'DELETE',
      headers: this.getAuthHeaders(),
    });
    return response.json();
  }

  async clearCart(userId: string) {
    const response = await fetch(`${this.baseURL}/api/carts/user/${userId}`, {
      method: 'DELETE',
      headers: this.getAuthHeaders(),
    });
    return response.json();
  }
}

export default new ApiService();
```

### 2. React Components

#### `src/components/ChatInterface.tsx`

```typescript
import React, { useState, useRef, useEffect } from 'react';
import AIAgentService from '../services/aiAgentService';
import './ChatInterface.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  intent?: string;
  confidence?: number;
}

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(`session_${Date.now()}`);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const userId = localStorage.getItem('user_id') || 'user123';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await AIAgentService.sendMessage({
        message: input,
        user_id: userId,
        session_id: sessionId,
      });

      const aiMessage: Message = {
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        intent: response.intent,
        confidence: response.confidence,
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>AI Assistant</h2>
      </div>
      <div className="messages-container">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-content">{msg.content}</div>
            {msg.intent && (
              <div className="message-meta">
                Intent: {msg.intent} | Confidence: {(msg.confidence || 0).toFixed(2)}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="message assistant">
            <div className="message-content">Đang suy nghĩ...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Nhập tin nhắn..."
          disabled={loading}
        />
        <button onClick={handleSend} disabled={loading || !input.trim()}>
          Gửi
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;
```

#### `src/components/ProductCatalog.tsx`

```typescript
import React, { useState, useEffect } from 'react';
import ApiService from '../services/apiService';
import AIAgentService from '../services/aiAgentService';
import './ProductCatalog.css';

interface Product {
  id: string;
  name: string;
  price: number;
  description: string;
  imageUrl?: string;
}

const ProductCatalog: React.FC = () => {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    loadProducts();
  }, []);

  const loadProducts = async () => {
    setLoading(true);
    try {
      const data = await ApiService.getProducts({ size: 20 });
      setProducts(data.content || []);
    } catch (error) {
      console.error('Error loading products:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAISearch = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      // Use AI Agent to search products
      const response = await AIAgentService.sendMessage({
        message: searchQuery,
        intent: 'search',
      });

      // Parse AI response and extract product IDs if possible
      // Then fetch products from Spring Boot service
      loadProducts();
    } catch (error) {
      console.error('Error in AI search:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="product-catalog">
      <div className="search-bar">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Tìm kiếm sản phẩm..."
        />
        <button onClick={handleAISearch}>Tìm với AI</button>
        <button onClick={loadProducts}>Tìm thông thường</button>
      </div>

      {loading ? (
        <div>Đang tải...</div>
      ) : (
        <div className="products-grid">
          {products.map((product) => (
            <div key={product.id} className="product-card">
              <img src={product.imageUrl || '/placeholder.png'} alt={product.name} />
              <h3>{product.name}</h3>
              <p>{product.description}</p>
              <div className="product-price">{product.price.toLocaleString()} VNĐ</div>
              <button>Thêm vào giỏ</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ProductCatalog;
```

### 3. Environment Configuration

#### `.env`

```env
REACT_APP_AI_AGENT_URL=http://localhost:8000
REACT_APP_API_GATEWAY_URL=http://localhost:8181
REACT_APP_ENV=development
```

---

## 🚀 Quick Start

### 1. Setup Frontend Project

```bash
# Create React app
npx create-react-app frontend --template typescript
cd frontend

# Install dependencies
npm install axios
```

### 2. Copy Services và Components

Copy các files từ examples trên vào project.

### 3. Start Development

```bash
# Start Frontend
npm start

# Frontend sẽ chạy tại http://localhost:3000
```

---

## 📱 Usage Examples

### Chat với AI Agent

```typescript
import AIAgentService from './services/aiAgentService';

const response = await AIAgentService.sendMessage({
  message: 'Tìm điện thoại iPhone dưới 30 triệu',
  user_id: 'user123',
});
console.log(response.response);
```

### Lấy sản phẩm từ Spring Boot

```typescript
import ApiService from './services/apiService';

const products = await ApiService.getProducts({
  search: 'iPhone',
  page: 0,
  size: 20,
});
```

### Tạo đơn hàng

```typescript
const order = await ApiService.createOrder({
  userId: 'user123',
  items: [
    { productId: 'prod1', quantity: 1 },
  ],
  shippingAddress: '123 Main St',
});
```

---

## 🔐 Authentication Flow

1. User login qua Spring Boot User Service
2. Lưu JWT token vào localStorage
3. Include token trong headers cho mọi API calls
4. Refresh token khi hết hạn

---

## 🎨 UI/UX Recommendations

1. **Chat Interface**
   - Real-time message updates
   - Typing indicators
   - Message timestamps
   - Intent badges (search, chat, api_call)

2. **Product Catalog**
   - Infinite scroll
   - Filters và sorting
   - AI-powered search suggestions
   - Product comparison

3. **Order Management**
   - Order status tracking
   - Order history
   - Invoice download

---

## 🐛 Error Handling

```typescript
try {
  const response = await ApiService.getProducts();
} catch (error) {
  if (error.response?.status === 401) {
    // Redirect to login
    window.location.href = '/login';
  } else if (error.response?.status === 404) {
    // Show not found message
  } else {
    // Show generic error
  }
}
```

---

**Last Updated**: January 2026  
**Version**: 1.0.0
