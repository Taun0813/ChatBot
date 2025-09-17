"""
Agno Router - Hybrid Orchestrator for AI Agent system
Handles request routing using rule-based + ML-based logic
"""

import asyncio
import logging
import re
import time
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

class RouterType(Enum):
    RULE_BASED = "rule_based"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"

@dataclass
class RoutingDecision:
    """Kết quả routing từ một router"""
    intent: str
    confidence: float
    router_type: RouterType
    metadata: Dict[str, Any]
    processing_time: float

@dataclass
class FinalDecision:
    """Kết quả cuối cùng sau khi fusion"""
    intent: str
    confidence: float
    selected_router: RouterType
    fusion_weights: Dict[str, float]
    metadata: Dict[str, Any]
    processing_time: float

class Rule:
    """Rule definition for routing"""
    def __init__(self, name: str, pattern: str, handler: str, priority: int = 0):
        self.name = name
        self.pattern = pattern
        self.handler = handler
        self.priority = priority
        self.compiled_pattern = re.compile(pattern, re.IGNORECASE)
    
    def matches(self, message: str) -> bool:
        """Check if message matches this rule"""
        return bool(self.compiled_pattern.search(message))

@dataclass
class RouterConfig:
    """Configuration for Agno Router"""
    rag_config: Dict[str, Any]
    interaction_config: Dict[str, Any]
    api_config: Dict[str, Any]
    personalization_config: Dict[str, Any]
    hybrid_config: Dict[str, Any] = None

class MLRouter:
    """ML-based router sử dụng neural networks"""
    
    def __init__(self):
        self.intent_classifier = None
        self.context_analyzer = None
        self.confidence_scorer = None
        self.is_initialized = False
        
        # Intent mapping
        self.intent_mapping = {
            "product_search": "search",
            "order_inquiry": "order", 
            "payment_question": "order",
            "general_chat": "chat",
            "api_call": "api"
        }
        
        # Context features
        self.context_features = [
            "user_history_length",
            "session_duration", 
            "previous_intent",
            "query_complexity",
            "has_product_mentions",
            "has_order_mentions",
            "has_price_mentions"
        ]
    
    async def initialize(self):
        """Initialize ML components"""
        try:
            # Initialize intent classifier (simplified version)
            self.intent_classifier = SimpleIntentClassifier()
            await self.intent_classifier.initialize()
            
            # Initialize context analyzer
            self.context_analyzer = ContextAnalyzer()
            await self.context_analyzer.initialize()
            
            # Initialize confidence scorer
            self.confidence_scorer = ConfidenceScorer()
            await self.confidence_scorer.initialize()
            
            self.is_initialized = True
            logger.info("ML Router initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML Router: {e}")
            raise
    
    async def route(self, message: str, context: Dict[str, Any]) -> RoutingDecision:
        """Route message using ML-based approach"""
        if not self.is_initialized:
            raise RuntimeError("ML Router not initialized")
        
        start_time = time.time()
        
        try:
            # Extract context features
            context_features = await self.context_analyzer.analyze(message, context)
            
            # Classify intent
            intent_probs = await self.intent_classifier.predict(message, context_features)
            predicted_intent = max(intent_probs, key=intent_probs.get)
            confidence = intent_probs[predicted_intent]
            
            # Map to router intent
            router_intent = self.intent_mapping.get(predicted_intent, "chat")
            
            processing_time = time.time() - start_time
            
            return RoutingDecision(
                intent=router_intent,
                confidence=confidence,
                router_type=RouterType.ML_BASED,
                metadata={
                    "intent_probs": intent_probs,
                    "context_features": context_features,
                    "original_intent": predicted_intent
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in ML routing: {e}")
            processing_time = time.time() - start_time
            return RoutingDecision(
                intent="chat",
                confidence=0.0,
                router_type=RouterType.ML_BASED,
                metadata={"error": str(e)},
                processing_time=processing_time
            )

class SimpleIntentClassifier:
    """Simplified intent classifier using keyword matching + heuristics"""
    
    def __init__(self):
        self.intent_keywords = {
            "product_search": [
                "tìm", "mua", "điện thoại", "iphone", "samsung", "xiaomi", 
                "oppo", "vivo", "realme", "oneplus", "huawei", "nokia",
                "motorola", "lg", "sony", "sản phẩm", "máy", "phone"
            ],
            "order_inquiry": [
                "đơn hàng", "order", "giao hàng", "vận chuyển", "trạng thái",
                "hủy", "đổi", "trả", "theo dõi", "tracking"
            ],
            "payment_question": [
                "thanh toán", "payment", "invoice", "hóa đơn", "tiền",
                "giá", "cost", "price", "vnd", "triệu", "nghìn"
            ],
            "api_call": [
                "api", "service", "dịch vụ", "tích hợp", "kết nối",
                "webhook", "endpoint"
            ],
            "general_chat": [
                "xin chào", "hello", "hi", "cảm ơn", "thanks", "help",
                "giúp", "hỗ trợ", "support"
            ]
        }
        
        self.intent_weights = {
            "product_search": 1.0,
            "order_inquiry": 0.9,
            "payment_question": 0.8,
            "api_call": 0.7,
            "general_chat": 0.5
        }
    
    async def initialize(self):
        """Initialize classifier"""
        logger.info("Simple Intent Classifier initialized")
    
    async def predict(self, message: str, context_features: Dict[str, Any]) -> Dict[str, float]:
        """Predict intent probabilities"""
        message_lower = message.lower()
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in message_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            # Normalize by keyword count and apply weights
            if keywords:
                normalized_score = score / len(keywords) * self.intent_weights[intent]
                intent_scores[intent] = normalized_score
            else:
                intent_scores[intent] = 0.0
        
        # Apply context features
        if context_features.get("has_product_mentions", False):
            intent_scores["product_search"] *= 1.2
        
        if context_features.get("has_order_mentions", False):
            intent_scores["order_inquiry"] *= 1.2
        
        if context_features.get("has_price_mentions", False):
            intent_scores["payment_question"] *= 1.1
        
        # Normalize to probabilities
        total_score = sum(intent_scores.values())
        if total_score > 0:
            intent_probs = {intent: score / total_score for intent, score in intent_scores.items()}
        else:
            # Default to general chat
            intent_probs = {intent: 0.0 for intent in intent_scores.keys()}
            intent_probs["general_chat"] = 1.0
        
        return intent_probs

class ContextAnalyzer:
    """Analyze context features for routing decisions"""
    
    def __init__(self):
        self.feature_extractors = {
            "user_history_length": self._extract_user_history_length,
            "session_duration": self._extract_session_duration,
            "previous_intent": self._extract_previous_intent,
            "query_complexity": self._extract_query_complexity,
            "has_product_mentions": self._extract_product_mentions,
            "has_order_mentions": self._extract_order_mentions,
            "has_price_mentions": self._extract_price_mentions
        }
    
    async def initialize(self):
        """Initialize context analyzer"""
        logger.info("Context Analyzer initialized")
    
    async def analyze(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context features"""
        features = {}
        
        for feature_name, extractor in self.feature_extractors.items():
            try:
                features[feature_name] = await extractor(message, context)
            except Exception as e:
                logger.warning(f"Error extracting feature {feature_name}: {e}")
                features[feature_name] = 0.0
        
        return features
    
    async def _extract_user_history_length(self, message: str, context: Dict[str, Any]) -> int:
        """Extract user history length"""
        user_id = context.get("user_id")
        if not user_id:
            return 0
        
        # This would typically query a database
        # For now, return a mock value
        return context.get("user_history_length", 0)
    
    async def _extract_session_duration(self, message: str, context: Dict[str, Any]) -> float:
        """Extract session duration in minutes"""
        session_start = context.get("session_start_time")
        if not session_start:
            return 0.0
        
        return (time.time() - session_start) / 60.0
    
    async def _extract_previous_intent(self, message: str, context: Dict[str, Any]) -> str:
        """Extract previous intent"""
        return context.get("previous_intent", "unknown")
    
    async def _extract_query_complexity(self, message: str, context: Dict[str, Any]) -> float:
        """Extract query complexity score"""
        # Simple complexity based on length and special characters
        length_score = min(len(message) / 100.0, 1.0)
        special_chars = sum(1 for c in message if not c.isalnum() and not c.isspace())
        special_score = min(special_chars / 10.0, 1.0)
        
        return (length_score + special_score) / 2.0
    
    async def _extract_product_mentions(self, message: str, context: Dict[str, Any]) -> bool:
        """Check if message mentions products"""
        product_keywords = [
            "điện thoại", "iphone", "samsung", "xiaomi", "oppo", "vivo",
            "realme", "oneplus", "huawei", "nokia", "motorola", "lg", "sony"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in product_keywords)
    
    async def _extract_order_mentions(self, message: str, context: Dict[str, Any]) -> bool:
        """Check if message mentions orders"""
        order_keywords = [
            "đơn hàng", "order", "giao hàng", "vận chuyển", "trạng thái",
            "hủy", "đổi", "trả", "theo dõi", "tracking"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in order_keywords)
    
    async def _extract_price_mentions(self, message: str, context: Dict[str, Any]) -> bool:
        """Check if message mentions prices"""
        price_keywords = [
            "giá", "price", "vnd", "triệu", "nghìn", "dưới", "trên",
            "khoảng", "từ", "đến", "thanh toán", "payment"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in price_keywords)

class ConfidenceScorer:
    """Score confidence for routing decisions"""
    
    def __init__(self):
        self.confidence_factors = {
            "keyword_match_strength": 0.4,
            "context_consistency": 0.3,
            "intent_probability": 0.3
        }
    
    async def initialize(self):
        """Initialize confidence scorer"""
        logger.info("Confidence Scorer initialized")
    
    async def score(self, intent: str, context_features: Dict[str, Any]) -> float:
        """Calculate confidence score"""
        try:
            # Keyword match strength (simplified)
            keyword_strength = context_features.get("query_complexity", 0.5)
            
            # Context consistency
            context_consistency = self._calculate_context_consistency(context_features)
            
            # Intent probability (from classifier)
            intent_probability = context_features.get("intent_probability", 0.5)
            
            # Weighted combination
            confidence = (
                keyword_strength * self.confidence_factors["keyword_match_strength"] +
                context_consistency * self.confidence_factors["context_consistency"] +
                intent_probability * self.confidence_factors["intent_probability"]
            )
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_context_consistency(self, context_features: Dict[str, Any]) -> float:
        """Calculate context consistency score"""
        # Check if features are consistent with each other
        consistency_score = 0.5  # Default
        
        # Example: If user has long history and mentions products, higher consistency
        if (context_features.get("user_history_length", 0) > 5 and 
            context_features.get("has_product_mentions", False)):
            consistency_score = 0.8
        
        # If session is long and previous intent was similar, higher consistency
        if (context_features.get("session_duration", 0) > 10 and
            context_features.get("previous_intent") != "unknown"):
            consistency_score = 0.7
        
        return consistency_score

class DecisionFusionEngine:
    """Fuse decisions from multiple routers"""
    
    def __init__(self):
        self.fusion_weights = {
            "rule_based": 0.4,
            "ml_based": 0.6
        }
        self.performance_history = defaultdict(list)
        self.adaptive_weights = True
        self.min_samples_for_adaptation = 10
    
    async def fuse_decisions(
        self, 
        rule_decision: RoutingDecision, 
        ml_decision: RoutingDecision,
        context: Dict[str, Any]
    ) -> FinalDecision:
        """Fuse decisions from rule-based and ML-based routers"""
        
        start_time = time.time()
        
        try:
            # Calculate fusion weights
            if self.adaptive_weights:
                weights = await self._calculate_adaptive_weights(rule_decision, ml_decision)
            else:
                weights = self.fusion_weights.copy()
            
            # Weighted fusion of confidences
            fused_confidence = (
                rule_decision.confidence * weights["rule_based"] +
                ml_decision.confidence * weights["ml_based"]
            )
            
            # Select best intent
            if ml_decision.confidence > 0.8 and ml_decision.confidence > rule_decision.confidence:
                selected_intent = ml_decision.intent
                selected_router = RouterType.ML_BASED
            elif rule_decision.confidence > 0.7 and rule_decision.confidence > ml_decision.confidence:
                selected_intent = rule_decision.intent
                selected_router = RouterType.RULE_BASED
            else:
                # Use fused decision
                if fused_confidence > 0.6:
                    selected_intent = ml_decision.intent if ml_decision.confidence > rule_decision.confidence else rule_decision.intent
                    selected_router = RouterType.HYBRID
                else:
                    selected_intent = "chat"
                    selected_router = RouterType.RULE_BASED
            
            processing_time = time.time() - start_time
            
            return FinalDecision(
                intent=selected_intent,
                confidence=fused_confidence,
                selected_router=selected_router,
                fusion_weights=weights,
                metadata={
                    "rule_decision": {
                        "intent": rule_decision.intent,
                        "confidence": rule_decision.confidence,
                        "processing_time": rule_decision.processing_time
                    },
                    "ml_decision": {
                        "intent": ml_decision.intent,
                        "confidence": ml_decision.confidence,
                        "processing_time": ml_decision.processing_time
                    },
                    "fusion_method": "weighted_average"
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in decision fusion: {e}")
            processing_time = time.time() - start_time
            
            # Fallback to rule-based decision
            return FinalDecision(
                intent=rule_decision.intent,
                confidence=rule_decision.confidence,
                selected_router=RouterType.RULE_BASED,
                fusion_weights=self.fusion_weights,
                metadata={"error": str(e)},
                processing_time=processing_time
            )
    
    async def _calculate_adaptive_weights(
        self, 
        rule_decision: RoutingDecision, 
        ml_decision: RoutingDecision
    ) -> Dict[str, float]:
        """Calculate adaptive weights based on historical performance"""
        
        # Get recent performance for each router
        rule_performance = self.performance_history.get("rule_based", [])
        ml_performance = self.performance_history.get("ml_based", [])
        
        if len(rule_performance) < self.min_samples_for_adaptation or len(ml_performance) < self.min_samples_for_adaptation:
            return self.fusion_weights.copy()
        
        # Calculate average performance
        rule_avg_performance = np.mean(rule_performance[-10:])  # Last 10 samples
        ml_avg_performance = np.mean(ml_performance[-10:])
        
        # Normalize to weights
        total_performance = rule_avg_performance + ml_avg_performance
        if total_performance > 0:
            weights = {
                "rule_based": rule_avg_performance / total_performance,
                "ml_based": ml_avg_performance / total_performance
            }
        else:
            weights = self.fusion_weights.copy()
        
        return weights
    
    def update_performance(self, router_type: str, performance_score: float):
        """Update performance history for a router"""
        self.performance_history[router_type].append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history[router_type]) > 100:
            self.performance_history[router_type] = self.performance_history[router_type][-50:]

class AgnoRouter:
    """
    Agno Router - Hybrid Orchestrator for AI Agent system
    
    Combines rule-based and ML-based routing:
    - Rule-based: Fast, deterministic routing
    - ML-based: Context-aware, adaptive routing
    - Hybrid: Best of both worlds with decision fusion
    """
    
    def __init__(self, config: RouterConfig):
        self.config = config
        self.rag_model = None
        self.interaction_model = None
        self.api_model = None
        self.personalization_model = None
        self.pinecone_client = None
        self.cache_manager = None
        self.model_loader = None
        
        # Hybrid Orchestrator components
        self.ml_router = MLRouter()
        self.fusion_engine = DecisionFusionEngine()
        self.enable_hybrid = config.hybrid_config.get("enable_hybrid", True) if config.hybrid_config else True
        
        # Performance tracking
        self.metrics = {
            "total_requests": 0,
            "rule_based_requests": 0,
            "ml_based_requests": 0,
            "hybrid_requests": 0,
            "average_response_time": 0.0,
            "accuracy_scores": []
        }
        
        # Initialize routing rules
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[Rule]:
        """Initialize routing rules"""
        return [
            # Product search rules (highest priority)
            Rule("product_search", r"(tìm|mua|mua|điện thoại|iphone|samsung|xiaomi|oppo|vivo|realme|oneplus|huawei|nokia|motorola|lg|sony)", "search", priority=10),
            Rule("product_search_price", r"(dưới|trên|khoảng|từ|đến|giá|vnd|triệu|nghìn)", "search", priority=9),
            Rule("product_search_specs", r"(pin|camera|ram|rom|màn hình|chơi game|chụp ảnh|battery|storage|display)", "search", priority=8),
            Rule("product_comparison", r"(so sánh|so sanh|compare|đối chiếu)", "search", priority=7),
            
            # Order-related rules
            Rule("order_status", r"(đơn hàng|order|giao hàng|vận chuyển|trạng thái|hủy|đổi|trả)", "order", priority=7),
            Rule("order_number", r"#\d+|\d{4,}", "order", priority=6),
            Rule("payment", r"(thanh toán|payment|invoice|hóa đơn)", "order", priority=5),
            
            # API/Service rules
            Rule("api_call", r"(api|service|dịch vụ|tích hợp|kết nối)", "api", priority=4),
            
            # Default chat rule (lowest priority)
            Rule("general_chat", r".*", "chat", priority=1)
        ]
    
    async def initialize(self):
        """Initialize all models and services"""
        try:
            logger.info("Initializing Agno Router...")
            
            # Initialize Pinecone client
            await self._initialize_pinecone()
            
            # Initialize cache manager
            await self._initialize_cache_manager()
            
            # Initialize model loader
            await self._initialize_model_loader()
            
            # Initialize RAG model
            await self._initialize_rag_model()
            
            # Initialize interaction model
            await self._initialize_interaction_model()
            
            # Initialize API model
            await self._initialize_api_model()
            
            # Initialize personalization model
            await self._initialize_personalization_model()
            
            # Initialize ML router for hybrid mode
            if self.enable_hybrid:
                await self.ml_router.initialize()
                logger.info("ML Router initialized for hybrid mode")
            
            logger.info("Agno Router initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Agno Router: {e}")
            raise
    
    async def _initialize_pinecone(self):
        """Initialize Pinecone vector database client"""
        try:
            from adapters.pinecone_client import PineconeClient
            
            pinecone_config = self.config.rag_config.get("pinecone_config", {})
            self.pinecone_client = PineconeClient(
                api_key=pinecone_config.get("api_key"),
                environment=pinecone_config.get("environment"),
                index_name=pinecone_config.get("index_name"),
                dimension=pinecone_config.get("dimension"),
                metric=pinecone_config.get("metric")
            )
            
            await self.pinecone_client.initialize()
            logger.info("Pinecone client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    async def _initialize_cache_manager(self):
        """Initialize cache manager"""
        try:
            from cache.cache_manager import CacheManager
            
            # Cache configuration
            cache_config = {
                "memory_cache_config": {
                    "max_size": 1000,
                    "default_ttl": 300,  # 5 minutes
                    "cleanup_interval": 60
                },
                "redis_cache_config": {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0,
                    "default_ttl": 3600,  # 1 hour
                    "key_prefix": "ai_agent:",
                    "fallback_to_memory": True
                }
            }
            
            self.cache_manager = CacheManager(**cache_config)
            await self.cache_manager.initialize()
            
            logger.info("Cache manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            # Don't raise - cache is optional
            self.cache_manager = None
    
    async def _initialize_model_loader(self):
        """Initialize model loader"""
        try:
            from adapters.model_loader import ModelLoaderFactory
            
            model_config = self.config.rag_config.get("model_loader_config", {})
            self.model_loader = ModelLoaderFactory.create_loader(
                backend=model_config.get("backend", "gemini"),
                model_name=model_config.get("model_name", "gemini-1.5-flash"),
                max_tokens=model_config.get("max_tokens", 2048),
                temperature=model_config.get("temperature", 0.7),
                top_p=model_config.get("top_p", 0.9)
            )
            
            logger.info(f"Model loader initialized: {model_config.get('backend')}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model loader: {e}")
            raise
    
    async def _initialize_rag_model(self):
        """Initialize RAG model"""
        try:
            from core.rag_model import RAGModel
            
            self.rag_model = RAGModel(
                pinecone_client=self.pinecone_client,
                model_loader=self.model_loader
            )
            
            logger.info("RAG model initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG model: {e}")
            raise
    
    async def _initialize_interaction_model(self):
        """Initialize interaction model"""
        try:
            from core.interaction_model import InteractionModel
            
            self.interaction_model = InteractionModel(
                model_loader=self.model_loader
            )
            
            logger.info("Interaction model initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize interaction model: {e}")
            raise
    
    async def _initialize_api_model(self):
        """Initialize API model"""
        try:
            from core.api_model import APIModel
            
            self.api_model = APIModel()
            
            logger.info("API model initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize API model: {e}")
            raise
    
    async def _initialize_personalization_model(self):
        """Initialize personalization model with Profile Manager and Recommender"""
        try:
            from core.personalization_model import PersonalizationModel
            from personalization.profile_manager import ProfileManager
            from personalization.recommender import Recommender
            from cache.cache_manager import CacheManager
            
            personalization_config = self.config.personalization_config
            
            # Initialize Profile Manager
            self.profile_manager = ProfileManager(
                db_path=personalization_config.get("db_path", "data/profiles/profiles.db"),
                json_backup=personalization_config.get("json_backup", True)
            )
            
            # Initialize Recommender
            self.recommender = Recommender(profile_manager=self.profile_manager)
            
            # Initialize Personalization Model with components
            self.personalization_model = PersonalizationModel(
                enable_personalization=personalization_config.get("enable_personalization", True),
                enable_recommendations=personalization_config.get("enable_recommendations", True),
                enable_rl_learning=personalization_config.get("enable_rl_learning", True),
                profiles_dir=personalization_config.get("profiles_dir", "./data/profiles"),
                models_dir=personalization_config.get("models_dir", "./data/models"),
                profile_manager=self.profile_manager,
                recommender=self.recommender
            )
            
            # Initialize the model
            await self.personalization_model.initialize()
            
            logger.info("Personalization model with Profile Manager and Recommender initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize personalization model: {e}")
            raise
    
    async def process_request(
        self, 
        message: str, 
        user_id: Optional[str] = None, 
        session_id: Optional[str] = None, 
        context: Optional[Dict[str, Any]] = None, 
        intent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process incoming request using hybrid routing (rule-based + ML-based)
        
        Args:
            message: User input message
            user_id: User identifier for personalization
            session_id: Session identifier
            context: Additional context
            intent: Pre-determined intent (optional)
        
        Returns:
            Response dictionary with response, intent, confidence, metadata
        """
        try:
            logger.info(f"Processing request: {message[:100]}...")
            start_time = time.time()
            self.metrics["total_requests"] += 1
            
            # If intent is pre-determined, use it directly
            if intent:
                response = await self._process_with_intent(message, intent, user_id, session_id, context)
                response["session_id"] = session_id
                response["user_id"] = user_id
                return response
            
            # Prepare context
            if context is None:
                context = {}
            
            context.update({
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": time.time()
            })
            
            # Use hybrid routing if enabled
            if self.enable_hybrid:
                response = await self._process_hybrid_request(message, user_id, session_id, context)
            else:
                # Fallback to rule-based routing
                response = await self._process_rule_based_request(message, user_id, session_id, context)
            
            # Add session information
            response["session_id"] = session_id
            response["user_id"] = user_id
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(response, processing_time)
            
            logger.info(f"Generated response: {response['response'][:100]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "response": "Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại sau.",
                "intent": "error",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
    
    async def _process_hybrid_request(
        self, 
        message: str, 
        user_id: Optional[str], 
        session_id: Optional[str], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process request using hybrid routing (rule-based + ML-based)"""
        try:
            # Get decisions from both routers in parallel
            rule_task = asyncio.create_task(self._get_rule_decision(message, context))
            ml_task = asyncio.create_task(self._get_ml_decision(message, context))
            
            rule_decision, ml_decision = await asyncio.gather(rule_task, ml_task)
            
            # Fuse decisions
            final_decision = await self.fusion_engine.fuse_decisions(
                rule_decision, ml_decision, context
            )
            
            # Process with selected intent
            response = await self._process_with_intent(
                message, 
                final_decision.intent, 
                user_id, 
                session_id, 
                context
            )
            
            # Add orchestrator metadata
            response["metadata"]["orchestrator"] = {
                "type": "hybrid",
                "selected_router": final_decision.selected_router.value,
                "fusion_weights": final_decision.fusion_weights,
                "rule_confidence": rule_decision.confidence,
                "ml_confidence": ml_decision.confidence,
                "final_confidence": final_decision.confidence,
                "processing_time": final_decision.processing_time
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in hybrid processing: {e}")
            # Fallback to rule-based
            return await self._process_rule_based_request(message, user_id, session_id, context)
    
    async def _process_rule_based_request(
        self, 
        message: str, 
        user_id: Optional[str], 
        session_id: Optional[str], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process request using rule-based routing only"""
        try:
            # Determine handler using rule-based routing
            handler = self._route_request(message)
            
            logger.info(f"Rule-based routing to handler: {handler}")
            
            # Process with determined intent
            return await self._process_with_intent(message, handler, user_id, session_id, context)
            
        except Exception as e:
            logger.error(f"Error in rule-based processing: {e}")
            return {
                "response": "Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại sau.",
                "intent": "error",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
    
    async def _get_rule_decision(self, message: str, context: Dict[str, Any]) -> RoutingDecision:
        """Get decision from rule-based router"""
        start_time = time.time()
        
        try:
            # Use existing rule router logic
            handler = self._route_request(message)
            
            # Calculate confidence based on rule matching
            confidence = 0.8  # Default confidence for rule-based
            
            processing_time = time.time() - start_time
            
            return RoutingDecision(
                intent=handler,
                confidence=confidence,
                router_type=RouterType.RULE_BASED,
                metadata={"handler": handler},
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in rule-based routing: {e}")
            processing_time = time.time() - start_time
            
            return RoutingDecision(
                intent="chat",
                confidence=0.0,
                router_type=RouterType.RULE_BASED,
                metadata={"error": str(e)},
                processing_time=processing_time
            )
    
    async def _get_ml_decision(self, message: str, context: Dict[str, Any]) -> RoutingDecision:
        """Get decision from ML-based router"""
        return await self.ml_router.route(message, context)
    
    async def _process_with_intent(
        self, 
        message: str, 
        intent: str, 
        user_id: Optional[str], 
        session_id: Optional[str], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process request with determined intent using existing router logic"""
        
        # Use existing router's request handling logic
        if intent == "search":
            return await self._handle_search_request(message, user_id, context)
        elif intent == "order":
            return await self._handle_order_request(message, user_id, context)
        elif intent == "api":
            return await self._handle_api_request(message, user_id, context)
        else:  # chat
            return await self._handle_chat_request(message, user_id, context)
    
    def _route_request(self, message: str) -> str:
        """
        Route request using rule-based matching
        
        Args:
            message: User message
        
        Returns:
            Handler name (search, order, api, chat)
        """
        try:
            # Sort rules by priority (highest first)
            sorted_rules = sorted(self.rules, key=lambda x: x.priority, reverse=True)
            
            # Find first matching rule
            for rule in sorted_rules:
                if rule.matches(message):
                    logger.info(f"Matched rule: {rule.name} -> {rule.handler}")
                    return rule.handler
            
            # Default to chat if no rules match
            logger.info("No rules matched, defaulting to chat")
            return "chat"
            
        except Exception as e:
            logger.error(f"Error in rule-based routing: {e}")
        return "chat"
    
    def _update_metrics(self, response: Dict[str, Any], processing_time: float):
        """Update performance metrics"""
        orchestrator_info = response.get("metadata", {}).get("orchestrator", {})
        router_type = orchestrator_info.get("selected_router", "rule_based")
        
        if router_type == "rule_based":
            self.metrics["rule_based_requests"] += 1
        elif router_type == "ml_based":
            self.metrics["ml_based_requests"] += 1
        elif router_type == "hybrid":
            self.metrics["hybrid_requests"] += 1
        
        # Update average response time
        total_time = self.metrics["average_response_time"] * (self.metrics["total_requests"] - 1)
        total_time += processing_time
        self.metrics["average_response_time"] = total_time / self.metrics["total_requests"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            **self.metrics,
            "rule_based_percentage": (
                self.metrics["rule_based_requests"] / max(self.metrics["total_requests"], 1) * 100
            ),
            "ml_based_percentage": (
                self.metrics["ml_based_requests"] / max(self.metrics["total_requests"], 1) * 100
            ),
            "hybrid_percentage": (
                self.metrics["hybrid_requests"] / max(self.metrics["total_requests"], 1) * 100
            )
        }
    
    
    async def _handle_search_request(
        self, 
        message: str, 
        user_id: Optional[str], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle product search requests using RAG with personalization and caching"""
        try:
            # Check cache first
            cache_key = {
                "type": "search",
                "query": message,
                "user_id": user_id
            }
            
            cached_result = None
            if self.cache_manager:
                cached_result = await self.cache_manager.get(
                    cache_key, 
                    data_type="query",
                    context={"is_search": True}
                )
            
            if cached_result:
                logger.info("Cache hit for search query")
                return cached_result
            
            # Use RAG model to search for products
            search_results = await self.rag_model.search_products(
                query=message,
                user_id=user_id,
                top_k=5
            )
            
            # Record user query for personalization
            if self.personalization_model and user_id:
                await self.personalization_model.record_user_interaction(
                    user_id=user_id,
                    interaction_type="search",
                    query=message
                )
            
            # Get personalized recommendations
            if self.personalization_model and user_id and search_results:
                personalized_results = await self.personalization_model.get_personalized_recommendations(
                    user_id=user_id,
                    query=message,
                    search_results=search_results,
                    max_recommendations=5
                )
                search_results = personalized_results
            
            # Use interaction model to generate natural response
            response = await self.interaction_model.generate_search_response(
                query=message,
                search_results=search_results,
                user_id=user_id,
                context=context
            )
            
            # Prepare result
            result = {
                "response": response,
                "intent": "search",
                "confidence": 0.9,
                "metadata": {
                    "search_results": search_results,
                    "results_count": len(search_results),
                    "model_used": "rag",
                    "personalized": self.personalization_model is not None and user_id is not None,
                    "cached": False
                }
            }
            
            # Cache the result
            if self.cache_manager:
                await self.cache_manager.set(
                    cache_key,
                    result,
                    data_type="query",
                    ttl=1800,  # 30 minutes
                    context={"is_search": True}
                )
                logger.info("Search result cached")
            
            return result
                
        except Exception as e:
            logger.error(f"Error in search request: {e}")
            return {
                "response": "Xin lỗi, tôi không thể tìm kiếm sản phẩm lúc này. Vui lòng thử lại sau.",
                "intent": "search",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
    
    async def _handle_order_request(
        self, 
        message: str, 
        user_id: Optional[str], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle order-related requests using API model"""
        try:
            # Use API model to handle order requests
            response = await self.api_model.handle_order_request(
                message=message,
                user_id=user_id,
                context=context
            )
            
            return {
                "response": response,
                "intent": "order",
                "confidence": 0.8,
                "metadata": {
                    "model_used": "api"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in order request: {e}")
            return {
                "response": "Xin lỗi, tôi không thể xử lý yêu cầu đơn hàng lúc này. Vui lòng thử lại sau.",
                "intent": "order",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
    
    async def _handle_api_request(
        self, 
        message: str, 
        user_id: Optional[str], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle API-related requests"""
        try:
            # Use API model to handle general API requests
            response = await self.api_model.handle_general_request(
                message=message,
                user_id=user_id,
                context=context
            )
            
            return {
                "response": response,
                "intent": "api",
                "confidence": 0.7,
                "metadata": {
                    "model_used": "api"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in API request: {e}")
            return {
                "response": "Xin lỗi, tôi không thể xử lý yêu cầu API lúc này. Vui lòng thử lại sau.",
                "intent": "api",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
    
    async def _handle_chat_request(
        self, 
        message: str, 
        user_id: Optional[str], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle general chat requests using interaction model"""
        try:
            # Use interaction model for general conversation
            response = await self.interaction_model.generate_response(
                message=message,
                user_id=user_id,
                context=context
            )
            
            return {
                "response": response,
                "intent": "chat",
                "confidence": 0.8,
                "metadata": {
                    "model_used": "interaction"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in chat request: {e}")
            return {
                "response": "Xin lỗi, tôi không thể trả lời lúc này. Vui lòng thử lại sau.",
                "intent": "chat",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("Cleaning up Agno Router...")
            
            if self.pinecone_client:
                await self.pinecone_client.cleanup()
            
            if self.model_loader:
                await self.model_loader.cleanup()
            
            logger.info("Agno Router cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")