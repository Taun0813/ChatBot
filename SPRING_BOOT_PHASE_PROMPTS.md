# Spring Boot AI Agent Backend - Phase by Phase Prompts

## 🚀 PHASE 1: PROJECT SETUP & INFRASTRUCTURE

```markdown
# PHASE 1: Spring Boot Backend Setup - AI Agent E-commerce System

## Yêu cầu Phase 1:
Tạo cấu trúc dự án Spring Boot microservices cho AI Agent e-commerce system với:

### 1. Project Structure
```
ai-agent-spring-backend/
├── pom.xml                          # Parent POM
├── .gitignore
├── .env.example
├── docker-compose.yml
├── README.md
├── ai-agent-discovery/              # Eureka Server
│   ├── pom.xml
│   ├── src/main/java/com/aiagent/discovery/
│   │   └── DiscoveryApplication.java
│   └── src/main/resources/
│       └── application.yml
├── ai-agent-gateway/                # API Gateway
│   ├── pom.xml
│   ├── src/main/java/com/aiagent/gateway/
│   │   ├── GatewayApplication.java
│   │   ├── config/
│   │   │   ├── GatewayConfig.java
│   │   │   └── SecurityConfig.java
│   │   └── filters/
│   │       ├── AuthFilter.java
│   │       └── RateLimitFilter.java
│   └── src/main/resources/
│       └── application.yml
├── ai-agent-core/                   # Core AI Service
│   ├── pom.xml
│   ├── src/main/java/com/aiagent/core/
│   │   ├── CoreApplication.java
│   │   ├── config/
│   │   │   ├── DatabaseConfig.java
│   │   │   ├── RedisConfig.java
│   │   │   └── AIConfig.java
│   │   ├── controller/
│   │   │   ├── ChatController.java
│   │   │   └── HealthController.java
│   │   ├── service/
│   │   │   ├── AgnoRouterService.java
│   │   │   ├── RAGAgentService.java
│   │   │   ├── ConversationAgentService.java
│   │   │   └── APIAgentService.java
│   │   ├── model/
│   │   │   ├── entity/
│   │   │   │   ├── User.java
│   │   │   │   ├── Product.java
│   │   │   │   ├── Order.java
│   │   │   │   └── Conversation.java
│   │   │   ├── dto/
│   │   │   │   ├── ChatRequest.java
│   │   │   │   ├── ChatResponse.java
│   │   │   │   └── ProductSearchRequest.java
│   │   │   └── enums/
│   │   │       ├── IntentType.java
│   │   │       └── OrderStatus.java
│   │   ├── repository/
│   │   │   ├── UserRepository.java
│   │   │   ├── ProductRepository.java
│   │   │   ├── OrderRepository.java
│   │   │   └── ConversationRepository.java
│   │   ├── util/
│   │   │   ├── CacheUtil.java
│   │   │   ├── JsonUtil.java
│   │   │   └── ValidationUtil.java
│   │   └── exception/
│   │       ├── GlobalExceptionHandler.java
│   │       ├── BusinessException.java
│   │       └── ValidationException.java
│   └── src/main/resources/
│       ├── application.yml
│       ├── application-dev.yml
│       ├── application-prod.yml
│       └── logback-spring.xml
├── shared-lib/                      # Shared Libraries
│   ├── pom.xml
│   └── src/main/java/com/aiagent/shared/
│       ├── dto/
│       ├── util/
│       ├── exception/
│       └── config/
├── infrastructure/
│   ├── docker/
│   │   ├── Dockerfile.discovery
│   │   ├── Dockerfile.gateway
│   │   ├── Dockerfile.core
│   │   ├── docker-compose.yml
│   │   └── docker-compose.dev.yml
│   └── monitoring/
│       ├── prometheus.yml
│       └── grafana/
└── scripts/
    ├── setup.sh
    ├── build.sh
    └── test.sh
```

### 2. Tech Stack Requirements
- **Spring Boot**: 3.2.0+
- **Java**: 17+
- **Maven**: 3.8+
- **Database**: PostgreSQL 15+
- **Cache**: Redis 7+
- **Message Queue**: RabbitMQ 3.12+
- **Service Discovery**: Eureka Server
- **API Gateway**: Spring Cloud Gateway
- **Monitoring**: Micrometer + Prometheus + Grafana
- **Container**: Docker + Docker Compose

### 3. Dependencies (Parent POM)
```xml
<properties>
    <spring-boot.version>3.2.0</spring-boot.version>
    <spring-cloud.version>2023.0.0</spring-cloud.version>
    <java.version>17</java.version>
    <maven.compiler.source>17</maven.compiler.source>
    <maven.compiler.target>17</maven.compiler.target>
</properties>

<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-dependencies</artifactId>
            <version>${spring-boot.version}</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-dependencies</artifactId>
            <version>${spring-cloud.version}</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>
```

### 4. Docker Configuration
- Multi-stage Dockerfile cho mỗi service
- Docker Compose cho development
- Health checks cho tất cả services
- Volume mapping cho development

### 5. Git Workflow
- Branch: `main`, `develop`, `feature/*`, `hotfix/*`
- Commit convention: `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `test:`, `chore:`
- Pre-commit hooks với checkstyle, spotbugs

### 6. Testing Setup
- Unit tests với JUnit 5 + Mockito
- Integration tests với TestContainers
- Test coverage với JaCoCo

## Deliverables Phase 1:
1. ✅ Project structure hoàn chỉnh
2. ✅ Parent POM với dependency management
3. ✅ Basic Spring Boot applications (Discovery, Gateway, Core)
4. ✅ Docker configuration
5. ✅ Git repository setup
6. ✅ Basic CI/CD pipeline
7. ✅ Health check endpoints
8. ✅ Basic logging configuration

## Testing Phase 1:
```bash
# Build all modules
mvn clean compile

# Run tests
mvn test

# Build Docker images
docker-compose build

# Start services
docker-compose up -d

# Health checks
curl http://localhost:8761/actuator/health  # Discovery
curl http://localhost:8080/actuator/health  # Gateway
curl http://localhost:8081/actuator/health  # Core
```

## Git Commands Phase 1:
```bash
# Initialize repository
git init
git add .
git commit -m "feat: initial project setup with Spring Boot microservices"

# Create develop branch
git checkout -b develop
git push -u origin develop

# Create feature branch
git checkout -b feature/phase1-setup
# ... make changes ...
git add .
git commit -m "feat: add Docker configuration and basic services"
git push origin feature/phase1-setup
```

Hãy bắt đầu với Phase 1 này trước!
```

---

## 🤖 PHASE 2: CORE AI SERVICES IMPLEMENTATION

```markdown
# PHASE 2: Core AI Services Implementation

## Yêu cầu Phase 2:
Implement core AI services với Hybrid Orchestrator pattern:

### 1. AgnoRouter Service (Hybrid Orchestrator)
- Rule-based routing với regex patterns
- ML-based routing với intent classification
- Decision fusion engine
- Fallback mechanism

### 2. RAG Agent Service
- Pinecone vector database integration
- Product search với semantic similarity
- Multi-LLM support (Gemini, Groq, OpenAI)
- Response generation với context

### 3. Conversation Agent Service
- Context management với Redis
- Conversation history
- Intent classification
- Natural language processing

### 4. API Agent Service
- External API integration
- Order management
- Payment processing
- Inventory management

## Deliverables Phase 2:
1. ✅ AgnoRouterService implementation
2. ✅ RAGAgentService với Pinecone
3. ✅ ConversationAgentService
4. ✅ APIAgentService
5. ✅ Database entities và repositories
6. ✅ DTOs và validation
7. ✅ Exception handling
8. ✅ Unit tests cho services

## Testing Phase 2:
```bash
# Run specific tests
mvn test -Dtest=AgnoRouterServiceTest
mvn test -Dtest=RAGAgentServiceTest

# Integration tests
mvn test -Dtest=*IntegrationTest

# Test with Docker
docker-compose up -d
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tôi muốn mua điện thoại Samsung", "userId": "user123"}'
```

## Git Commands Phase 2:
```bash
# Create feature branch
git checkout develop
git checkout -b feature/phase2-core-ai-services

# ... implement Phase 2 ...

# Commit changes
git add .
git commit -m "feat: phase2 - core AI services implementation"
git push origin feature/phase2-core-ai-services

# Create PR: feature/phase2-core-ai-services -> develop
```
```

---

## 🗄️ PHASE 3: DATABASE & CACHING

```markdown
# PHASE 3: Database & Caching Implementation

## Yêu cầu Phase 3:
Setup database schema và caching layer:

### 1. Database Schema
- User management
- Product catalog (27,000+ products)
- Order management
- Conversation history
- Analytics data

### 2. Caching Strategy
- Redis cho session management
- Cache cho AI responses
- Cache cho product data
- Cache invalidation strategy

### 3. Data Migration
- Flyway migrations
- Seed data cho products
- User profiles
- Test data

## Deliverables Phase 3:
1. ✅ Database schema với JPA entities
2. ✅ Repository layer với custom queries
3. ✅ Redis caching implementation
4. ✅ Data migration scripts
5. ✅ Seed data cho 27,000+ products
6. ✅ Cache configuration
7. ✅ Performance optimization

## Testing Phase 3:
```bash
# Database tests
mvn test -Dtest=*RepositoryTest
mvn test -Dtest=*CacheTest

# Check database
docker-compose exec postgres psql -U ai_agent -d ai_agent -c "SELECT COUNT(*) FROM products;"

# Check Redis
docker-compose exec redis redis-cli ping
```

## Git Commands Phase 3:
```bash
# Create feature branch
git checkout develop
git checkout -b feature/phase3-database-caching

# ... implement Phase 3 ...

# Commit changes
git add .
git commit -m "feat: phase3 - database and caching implementation"
git push origin feature/phase3-database-caching
```
```

---

## 🔐 PHASE 4: API GATEWAY & SECURITY

```markdown
# PHASE 4: API Gateway & Security

## Yêu cầu Phase 4:
Implement API Gateway và security layer:

### 1. API Gateway
- Request routing
- Load balancing
- Rate limiting
- Circuit breaker
- Request/Response transformation

### 2. Security
- JWT authentication
- OAuth2 integration
- Role-based authorization
- API key management
- CORS configuration

### 3. Monitoring
- Request tracing
- Metrics collection
- Health checks
- Alerting

## Deliverables Phase 4:
1. ✅ API Gateway configuration
2. ✅ Security implementation
3. ✅ Authentication/Authorization
4. ✅ Rate limiting
5. ✅ Monitoring setup
6. ✅ API documentation

## Testing Phase 4:
```bash
# Security tests
mvn test -Dtest=*SecurityTest
mvn test -Dtest=*GatewayTest

# Test authentication
curl -H "Authorization: Bearer <token>" http://localhost:8080/api/v1/chat

# Test rate limiting
for i in {1..10}; do curl http://localhost:8080/api/v1/chat; done
```

## Git Commands Phase 4:
```bash
# Create feature branch
git checkout develop
git checkout -b feature/phase4-gateway-security

# ... implement Phase 4 ...

# Commit changes
git add .
git commit -m "feat: phase4 - API gateway and security"
git push origin feature/phase4-gateway-security
```
```

---

## 🔗 PHASE 5: MICROSERVICES & INTEGRATION

```markdown
# PHASE 5: Microservices & Integration

## Yêu cầu Phase 5:
Implement additional microservices:

### 1. Product Service
- Product catalog management
- Search functionality
- Recommendations
- Inventory management

### 2. Order Service
- Order creation/management
- Order tracking
- Payment integration
- Shipping management

### 3. User Service
- User management
- Profile management
- Authentication
- Preferences

### 4. Integration
- Service-to-service communication
- Event-driven architecture
- Message queuing
- Data synchronization

## Deliverables Phase 5:
1. ✅ Product Service
2. ✅ Order Service
3. ✅ User Service
4. ✅ Service integration
5. ✅ Event handling
6. ✅ End-to-end testing

## Testing Phase 5:
```bash
# End-to-end tests
mvn test -Dtest=*E2ETest

# Load testing
./scripts/load-test.sh

# Integration tests
mvn test -Dtest=*IntegrationTest
```

## Git Commands Phase 5:
```bash
# Create feature branch
git checkout develop
git checkout -b feature/phase5-microservices

# ... implement Phase 5 ...

# Commit changes
git add .
git commit -m "feat: phase5 - microservices and integration"
git push origin feature/phase5-microservices

# Merge to main
git checkout develop
git checkout main
git merge develop
git tag v1.0.0
git push origin main --tags
```
```

---

## 🐳 DOCKER CONFIGURATION

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ai_agent
      POSTGRES_USER: ai_agent
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ai_agent"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  rabbitmq:
    image: rabbitmq:3-management
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: password
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  eureka:
    build: ./ai-agent-discovery
    ports:
      - "8761:8761"
    environment:
      - SPRING_PROFILES_ACTIVE=docker
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8761/actuator/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  gateway:
    build: ./ai-agent-gateway
    ports:
      - "8080:8080"
    depends_on:
      eureka:
        condition: service_healthy
    environment:
      - SPRING_PROFILES_ACTIVE=docker
      - EUREKA_CLIENT_SERVICE_URL_DEFAULTZONE=http://eureka:8761/eureka
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/actuator/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  core:
    build: ./ai-agent-core
    ports:
      - "8081:8081"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      rabbitmq:
        condition: service_healthy
      eureka:
        condition: service_healthy
    environment:
      - SPRING_PROFILES_ACTIVE=docker
      - DATABASE_URL=jdbc:postgresql://postgres:5432/ai_agent
      - REDIS_HOST=redis
      - EUREKA_CLIENT_SERVICE_URL_DEFAULTZONE=http://eureka:8761/eureka
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/actuator/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:
  redis_data:
  rabbitmq_data:
```

---

## 🧪 TESTING COMMANDS

```bash
# Phase 1: Setup Testing
mvn clean compile
mvn test -Dtest=*SetupTest
docker-compose build
docker-compose up -d
./scripts/health-check.sh

# Phase 2: Core AI Services Testing
mvn test -Dtest=*ServiceTest
mvn test -Dtest=*IntegrationTest
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Test message", "userId": "test-user"}'

# Phase 3: Database Testing
mvn test -Dtest=*RepositoryTest
mvn test -Dtest=*CacheTest
docker-compose exec postgres psql -U ai_agent -d ai_agent -c "SELECT COUNT(*) FROM products;"

# Phase 4: Security Testing
mvn test -Dtest=*SecurityTest
mvn test -Dtest=*GatewayTest
curl -H "Authorization: Bearer <token>" http://localhost:8080/api/v1/chat

# Phase 5: End-to-End Testing
mvn test -Dtest=*E2ETest
./scripts/load-test.sh
```

---

## 🔄 GIT WORKFLOW

```bash
# 1. Initialize Repository
git init
git remote add origin <your-repo-url>

# 2. Create Main Branches
git checkout -b main
git checkout -b develop

# 3. Phase 1: Setup
git checkout -b feature/phase1-setup
# ... implement Phase 1 ...
git add .
git commit -m "feat: phase1 - project setup and infrastructure"
git push origin feature/phase1-setup
# Create PR: feature/phase1-setup -> develop

# 4. Phase 2: Core AI Services
git checkout develop
git checkout -b feature/phase2-core-ai-services
# ... implement Phase 2 ...
git add .
git commit -m "feat: phase2 - core AI services implementation"
git push origin feature/phase2-core-ai-services
# Create PR: feature/phase2-core-ai-services -> develop

# 5. Phase 3: Database & Caching
git checkout develop
git checkout -b feature/phase3-database-caching
# ... implement Phase 3 ...
git add .
git commit -m "feat: phase3 - database and caching implementation"
git push origin feature/phase3-database-caching

# 6. Phase 4: API Gateway & Security
git checkout develop
git checkout -b feature/phase4-gateway-security
# ... implement Phase 4 ...
git add .
git commit -m "feat: phase4 - API gateway and security"
git push origin feature/phase4-gateway-security

# 7. Phase 5: Microservices
git checkout develop
git checkout -b feature/phase5-microservices
# ... implement Phase 5 ...
git add .
git commit -m "feat: phase5 - microservices and integration"
git push origin feature/phase5-microservices

# 8. Merge to Main
git checkout develop
git checkout main
git merge develop
git tag v1.0.0
git push origin main --tags
```

---

## 📋 CI/CD PIPELINE

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up JDK 17
      uses: actions/setup-java@v3
      with:
        java-version: '17'
        distribution: 'temurin'
    
    - name: Cache Maven dependencies
      uses: actions/cache@v3
      with:
        path: ~/.m2
        key: ${{ runner.os }}-m2-${{ hashFiles('**/pom.xml') }}
    
    - name: Run tests
      run: mvn clean test
    
    - name: Generate test report
      uses: dorny/test-reporter@v1
      if: success() || failure()
      with:
        name: Maven Tests
        path: target/surefire-reports/*.xml
        reporter: java-junit

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up JDK 17
      uses: actions/setup-java@v3
      with:
        java-version: '17'
        distribution: 'temurin'
    
    - name: Build with Maven
      run: mvn clean package -DskipTests
    
    - name: Build Docker images
      run: docker-compose build
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker-compose push

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to production..."
        # Add deployment commands here
```

---

## 🎯 QUICK START GUIDE

### 1. Bắt đầu với Phase 1:
```bash
# Tạo repository mới
mkdir ai-agent-spring-backend
cd ai-agent-spring-backend
git init
git remote add origin <your-repo-url>

# Copy prompt Phase 1 vào Cursor Chat
# Implement theo hướng dẫn
# Test và commit
git add .
git commit -m "feat: phase1 - project setup and infrastructure"
git push origin main
```

### 2. Workflow cho mỗi Phase:
1. **Tạo feature branch** từ develop
2. **Copy prompt tương ứng** vào Cursor Chat
3. **Implement** theo hướng dẫn
4. **Test** với commands được cung cấp
5. **Commit** với conventional commits
6. **Push** và tạo Pull Request
7. **Merge** vào develop sau khi review

### 3. Tools được đề xuất:
- **IDE**: IntelliJ IDEA hoặc VS Code
- **Database**: PostgreSQL + Redis
- **Message Queue**: RabbitMQ
- **Monitoring**: Prometheus + Grafana
- **Container**: Docker + Docker Compose
- **CI/CD**: GitHub Actions
- **Testing**: JUnit 5 + TestContainers
- **Code Quality**: SpotBugs + PMD + Checkstyle

### 4. Lợi ích của approach này:
- ✅ **Incremental development** - phát triển từng bước
- ✅ **Testable** - test được từng phase
- ✅ **Maintainable** - dễ maintain và debug
- ✅ **Scalable** - có thể scale theo nhu cầu
- ✅ **Production-ready** - sẵn sàng cho production

---

## 📝 NOTES

- Mỗi phase có thể mất 1-2 tuần để hoàn thành
- Test thường xuyên để đảm bảo chất lượng
- Commit thường xuyên với messages rõ ràng
- Review code trước khi merge
- Backup code trước khi thay đổi lớn

Chúc bạn thành công với dự án Spring Boot AI Agent! 🚀
