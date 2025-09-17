"""
Interaction Model - Natural conversation handling
Handles general chat and response generation with advanced LLM integration
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from core.prompts import PromptTemplates

logger = logging.getLogger(__name__)

class InteractionModel:
    """
    Interaction Model for natural conversation
    
    Features:
    - Natural language response generation
    - Context-aware conversations
    - User personalization
    - Search result formatting
    """
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        
    async def generate_response(
        self,
        message: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate natural response for general conversation
        
        Args:
            message: User message
            user_id: User identifier
            context: Additional context
        
        Returns:
            Generated response
        """
        try:
            logger.info(f"Generating response for: {message[:50]}...")
            
            # Create system prompt
            system_prompt = self._create_system_prompt(user_id, context)
            
            # Create conversation prompt
            conversation_prompt = f"""
{system_prompt}

Người dùng: {message}

Trợ lý AI:"""
            
            # Generate response using model loader
            response = await self.model_loader.generate_response(
                prompt=conversation_prompt,
                max_tokens=512,
                temperature=0.7
            )
            
            logger.info(f"Generated response: {response[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại sau."
    
    async def generate_search_response(
        self, 
        query: str,
        search_results: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate response for product search results using advanced prompts
        
        Args:
            query: Original search query
            search_results: List of search results
            user_id: User identifier
            context: Additional context
        
        Returns:
            Formatted search response
        """
        try:
            logger.info(f"Generating search response for query: {query}")
            
            if not search_results:
                return PromptTemplates.get_no_results_prompt(query)
            
            # Create contextual prompt
            prompt = PromptTemplates.get_contextual_prompt(
                query=query,
                context=context or {},
                products=search_results
            )
            
            # Generate response using LLM
            response = await self.model_loader.generate_response(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.7
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate search response: {e}")
            return self._generate_fallback_search_response(query, search_results)
    
    def _create_system_prompt(
        self, 
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create system prompt for conversation"""
        return """Bạn là một trợ lý bán hàng chuyên nghiệp, thân thiện và hiểu biết sâu về công nghệ điện thoại. 

Hãy luôn:
- Trả lời một cách tự nhiên, thân thiện bằng tiếng Việt
- Tư vấn sản phẩm dựa trên nhu cầu thực tế của khách hàng  
- Cung cấp thông tin chính xác và hữu ích
- Không bịa đặt thông tin về sản phẩm
- Hỏi thêm để hiểu rõ nhu cầu khi cần thiết
- Giữ giọng điệu chuyên nghiệp nhưng gần gũi
- Luôn sẵn sàng hỗ trợ về đơn hàng, bảo hành và thanh toán

Nếu không chắc chắn về thông tin, hãy nói rõ và đề xuất cách tìm hiểu thêm."""
    
    def _create_search_prompt(
        self, 
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for search response generation"""
        
        # Format search results
        products_text = self._format_search_results(search_results)
        
        return f"""Bạn là trợ lý bán hàng chuyên nghiệp. Dựa trên yêu cầu tìm kiếm và kết quả tìm được, hãy tạo một phản hồi tự nhiên và hữu ích.

Yêu cầu tìm kiếm: "{query}"

Kết quả tìm được:
{products_text}

Hãy tạo một phản hồi:
1. Xác nhận hiểu yêu cầu của khách hàng
2. Giới thiệu các sản phẩm phù hợp nhất (tối đa 3 sản phẩm)
3. So sánh ưu nhược điểm của từng sản phẩm
4. Đưa ra lời khuyên dựa trên nhu cầu
5. Hỏi thêm thông tin nếu cần thiết

Trả lời bằng tiếng Việt, tự nhiên và thân thiện."""
    
    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results for prompt"""
        formatted_results = []
        
        for i, product in enumerate(search_results[:5], 1):
            name = product.get("name", "Unknown")
            brand = product.get("brand", "Unknown")
            price = product.get("price", 0)
            rating = product.get("rating", 0)
            description = product.get("description", "")
            specs = product.get("specifications", {})
            
            # Format specifications
            specs_text = ""
            if specs:
                spec_items = []
                for key, value in specs.items():
                    spec_items.append(f"{key}: {value}")
                specs_text = f" | {', '.join(spec_items)}"
            
            formatted_result = f"""
{i}. {name} ({brand})
   - Giá: {price:,} VNĐ
   - Đánh giá: ⭐ {rating}/5
   - Mô tả: {description}
   - Thông số: {specs_text}
   - Điểm phù hợp: {product.get('similarity_score', 0):.2f}
"""
            formatted_results.append(formatted_result)
        
        return "\n".join(formatted_results)
    
    def _generate_no_results_response(self, query: str) -> str:
        """Generate response when no results found"""
        return f"""Xin lỗi, tôi không tìm thấy sản phẩm nào phù hợp với yêu cầu "{query}" của bạn.

Để tôi có thể hỗ trợ tốt hơn, bạn có thể:
- Mở rộng phạm vi tìm kiếm (ví dụ: thay đổi giá, thương hiệu)
- Cung cấp thêm thông tin về nhu cầu cụ thể
- Cho tôi biết bạn quan tâm đến dòng sản phẩm nào

Bạn có muốn tôi gợi ý một số sản phẩm phổ biến không?"""
    
    async def generate_personalized_response(
        self,
        query: str,
        user_preferences: Dict[str, Any],
        products: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> str:
        """Generate personalized response based on user preferences"""
        try:
            logger.info(f"Generating personalized response for user: {user_id}")
            
            # Create personalized prompt
            prompt = PromptTemplates.get_product_recommendation_prompt(
                user_preferences=user_preferences,
                products=products
            )
            
            # Generate response
            response = await self.model_loader.generate_response(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.7
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate personalized response: {e}")
            return self._generate_fallback_search_response(query, products)
    
    async def generate_comparison_response(
        self,
        products: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> str:
        """Generate product comparison response"""
        try:
            logger.info(f"Generating comparison response for {len(products)} products")
            
            # Create comparison prompt
            prompt = PromptTemplates.get_comparison_prompt(products)
            
            # Generate response
            response = await self.model_loader.generate_response(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.7
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate comparison response: {e}")
            return "Xin lỗi, tôi không thể so sánh sản phẩm lúc này. Vui lòng thử lại sau."
    
    async def generate_order_response(
        self, 
        order_info: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """Generate order status response"""
        try:
            logger.info(f"Generating order response for order: {order_info.get('order_id')}")
            
            # Create order status prompt
            prompt = PromptTemplates.get_order_status_prompt(order_info)
            
            # Generate response
            response = await self.model_loader.generate_response(
                prompt=prompt,
                max_tokens=512,
                temperature=0.7
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate order response: {e}")
            return "Xin lỗi, tôi không thể tra cứu thông tin đơn hàng lúc này. Vui lòng thử lại sau."
    
    def _generate_fallback_search_response(
        self, 
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Generate fallback response when model fails"""
        if not search_results:
            return PromptTemplates.get_no_results_prompt(query)
        
        # Simple text-based response
        response_parts = [f"Dựa trên yêu cầu '{query}', tôi tìm thấy {len(search_results)} sản phẩm phù hợp:"]
        
        for i, product in enumerate(search_results[:3], 1):
            name = product.get("name", "Unknown")
            brand = product.get("brand", "Unknown")
            price = product.get("price", 0)
            rating = product.get("rating", 0)
            
            response_parts.append(
                f"{i}. {name} ({brand}) - {price:,} VNĐ - ⭐ {rating}/5"
            )
        
        if len(search_results) > 3:
            response_parts.append(f"... và {len(search_results) - 3} sản phẩm khác")
        
        response_parts.append("\nBạn có muốn tôi cung cấp thêm thông tin chi tiết về sản phẩm nào không?")
        
        return "\n".join(response_parts)