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
            logger.info("Generating response for: %s...", message[:50])
            
            # Create system prompt
            system_prompt = self._create_system_prompt(user_id, context)
            history_block = self._format_history_for_prompt(context)
            
            # Create conversation prompt
            conversation_prompt = f"""
{system_prompt}

{history_block}

Người dùng: {message}

Trợ lý AI:"""
            
            # Generate response using model loader
            response = await self.model_loader.generate_response(
                prompt=conversation_prompt,
                max_tokens=512,
                temperature=0.7
            )
            
            logger.info("Generated response: %s...", response[:50])
            return response
            
        except Exception as e:
            logger.error("Failed to generate response: %s", e)
            return "Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại sau."
    
    async def generate_search_response(
        self, 
        query: str,
        search_results: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_products_in_prompt: int = 3,
    ) -> str:
        """
        Generate response for product search - chỉ đưa tối đa max_products_in_prompt vào prompt để giảm token và latency.
        """
        try:
            logger.info("Generating search response for query: %s", query)
            
            if not search_results:
                return PromptTemplates.get_no_results_prompt(query)
            
            # Chỉ đưa top N sản phẩm vào prompt để giảm thời gian LLM
            products_for_prompt = search_results[:max_products_in_prompt]
            prompt = PromptTemplates.get_grounded_search_prompt(
                query=query,
                context=context or {},
                products=products_for_prompt,
            )
            
            # max_tokens vừa đủ để trả lời ngắn gọn, giảm latency
            response = await self.model_loader.generate_response(
                prompt=prompt,
                max_tokens=768,
                temperature=0.5,
            )

            text = (response or "").strip()
            if not text:
                return self._generate_fallback_search_response(query, search_results)

            return self._ensure_grounded_response(
                response_text=text,
                products=products_for_prompt,
            )
            
        except Exception as e:
            logger.error("Failed to generate search response: %s", e)
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

    def _format_history_for_prompt(self, context: Optional[Dict[str, Any]]) -> str:
        """Format short recent conversation history to improve multi-turn memory."""
        context = context or {}
        history = context.get("conversation_history") or []
        if not isinstance(history, list) or not history:
            return ""

        lines = ["Ngữ cảnh hội thoại gần đây:"]
        for turn in history[-4:]:
            if not isinstance(turn, dict):
                continue
            user_text = str(turn.get("user", "")).strip()
            assistant_text = str(turn.get("assistant", "")).strip()
            if user_text:
                lines.append(f"- Người dùng: {user_text}")
            if assistant_text:
                lines.append(f"- Trợ lý: {assistant_text}")

        return "\n".join(lines)
    
    def _create_search_prompt(
        self, 
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for search response generation"""
        
        # Format search results (kèm spec liên quan query: pin, camera, ram...)
        products_text = self._format_search_results(search_results, query=query)
        
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
    
    def _format_search_results(
        self,
        search_results: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> str:
        """Format search results - nếu có query thì thêm spec liên quan (pin, camera...) để so sánh."""
        return PromptTemplates._format_products(
            search_results[:3], max_items=3, query=query
        )
    
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
            logger.info("Generating personalized response for user: %s", user_id)
            
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
            logger.error("Failed to generate personalized response: %s", e)
            return self._generate_fallback_search_response(query, products)
    
    async def generate_comparison_response(
        self,
        products: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> str:
        """Generate product comparison response"""
        try:
            logger.info("Generating comparison response for %s products", len(products))
            
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
            logger.error("Failed to generate comparison response: %s", e)
            return "Xin lỗi, tôi không thể so sánh sản phẩm lúc này. Vui lòng thử lại sau."
    
    async def generate_order_response(
        self, 
        order_info: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """Generate order status response"""
        try:
            logger.info("Generating order response for order: %s", order_info.get("order_id"))
            
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
            logger.error("Failed to generate order response: %s", e)
            return "Xin lỗi, tôi không thể tra cứu thông tin đơn hàng lúc này. Vui lòng thử lại sau."
    
    def _generate_fallback_search_response(
        self, 
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Template response - giá VNĐ và kèm spec liên quan (pin, camera...) để user so sánh."""
        if not search_results:
            return PromptTemplates.get_no_results_prompt(query)
        from core.prompts import get_spec_keys_for_query, format_product_line_with_specs
        spec_keys = get_spec_keys_for_query(query)
        response_parts = [f"Dựa trên yêu cầu '{query}', tôi tìm thấy {len(search_results)} sản phẩm phù hợp:"]
        for i, product in enumerate(search_results[:5], 1):
            price_vnd = PromptTemplates._price_for_display(product.get("price", 0))
            rating = float(product.get("rating", 0))
            if spec_keys:
                line = format_product_line_with_specs(
                    product, spec_keys, price_vnd, rating, index=i
                )
            else:
                name = product.get("name", "Unknown")
                brand = product.get("brand", "Unknown")
                line = f"{i}. {name} ({brand}) - {price_vnd:,} VNĐ - ⭐ {rating}/5"
            response_parts.append(line)
        if len(search_results) > 5:
            response_parts.append(f"... và {len(search_results) - 5} sản phẩm khác.")
        response_parts.append(PromptTemplates.build_grounding_reference_block(search_results, max_items=3))
        response_parts.append("Bạn có muốn tôi cung cấp thêm thông tin chi tiết về sản phẩm nào không?")
        return "\n".join(response_parts)

    def _ensure_grounded_response(self, response_text: str, products: List[Dict[str, Any]]) -> str:
        """Ensure final answer includes references to retrieved products."""
        if not products:
            return response_text

        lowered_text = response_text.lower()
        has_reference = False
        for product in products:
            product_id = str(product.get("backend_id") or product.get("id") or "").strip().lower()
            product_name = str(product.get("name") or "").strip().lower()
            if (product_id and product_id in lowered_text) or (product_name and product_name in lowered_text):
                has_reference = True
                break

        final_text = response_text
        if not has_reference:
            final_text = self._generate_fallback_search_response(
                query="sản phẩm phù hợp",
                search_results=products,
            )

        if "nguồn đối chiếu" not in final_text.lower():
            final_text = f"{final_text}\n\n{PromptTemplates.build_grounding_reference_block(products, max_items=3)}"

        return final_text