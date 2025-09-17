"""
Prompt Templates for AI Agent System
Templates for RAG answers, product recommendations, and conversation
"""

from typing import List, Dict, Any, Optional

class PromptTemplates:
    """Prompt templates for different use cases"""
    
    @staticmethod
    def get_system_prompt() -> str:
        """System prompt for general conversation"""
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

    @staticmethod
    def get_product_search_prompt(query: str, products: List[Dict[str, Any]]) -> str:
        """Prompt for product search results"""
        
        # Format products
        products_text = PromptTemplates._format_products(products)
        
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

    @staticmethod
    def get_product_recommendation_prompt(
        user_preferences: Dict[str, Any], 
        products: List[Dict[str, Any]]
    ) -> str:
        """Prompt for personalized product recommendations"""
        
        products_text = PromptTemplates._format_products(products)
        
        # Format user preferences
        preferences_text = ""
        if user_preferences.get("brands"):
            brands = ", ".join(user_preferences["brands"].keys())
            preferences_text += f"Thương hiệu yêu thích: {brands}\n"
        
        if user_preferences.get("price_range"):
            price_range = user_preferences["price_range"]
            preferences_text += f"Khoảng giá: {price_range['min']:,} - {price_range['max']:,} VNĐ\n"
        
        if user_preferences.get("categories"):
            categories = ", ".join(user_preferences["categories"].keys())
            preferences_text += f"Danh mục quan tâm: {categories}\n"
        
        return f"""Bạn là trợ lý bán hàng thông minh. Dựa trên sở thích của khách hàng và sản phẩm có sẵn, hãy đưa ra gợi ý cá nhân hóa.

Sở thích khách hàng:
{preferences_text}

Sản phẩm có sẵn:
{products_text}

Hãy tạo một phản hồi:
1. Thể hiện sự hiểu biết về sở thích khách hàng
2. Gợi ý 2-3 sản phẩm phù hợp nhất
3. Giải thích tại sao những sản phẩm này phù hợp
4. Đưa ra lời khuyên dựa trên lịch sử mua hàng
5. Hỏi về nhu cầu cụ thể nếu cần

Trả lời bằng tiếng Việt, thân thiện và cá nhân hóa."""

    @staticmethod
    def get_order_status_prompt(order_info: Dict[str, Any]) -> str:
        """Prompt for order status responses"""
        
        order_details = f"""
Số đơn hàng: {order_info.get('order_id', 'N/A')}
Trạng thái: {order_info.get('status', 'N/A')}
Tổng tiền: {order_info.get('total_amount', 0):,} VNĐ
"""
        
        products_text = ""
        for product in order_info.get('products', []):
            products_text += f"- {product.get('name', 'N/A')} x{product.get('quantity', 1)}\n"
        
        shipping_info = f"""
Địa chỉ giao hàng:
- Tên: {order_info.get('shipping_address', {}).get('name', 'N/A')}
- Địa chỉ: {order_info.get('shipping_address', {}).get('address', 'N/A')}
- SĐT: {order_info.get('shipping_address', {}).get('phone', 'N/A')}
"""
        
        return f"""Bạn là trợ lý hỗ trợ đơn hàng. Dựa trên thông tin đơn hàng, hãy tạo phản hồi thông tin và hữu ích.

Thông tin đơn hàng:
{order_details}

Sản phẩm:
{products_text}

{shipping_info}

Dự kiến giao hàng: {order_info.get('estimated_delivery', 'N/A')}

Hãy tạo phản hồi:
1. Xác nhận thông tin đơn hàng
2. Giải thích trạng thái hiện tại
3. Cung cấp thông tin giao hàng
4. Hướng dẫn các bước tiếp theo nếu cần
5. Hỏi xem khách có cần hỗ trợ gì thêm

Trả lời bằng tiếng Việt, chuyên nghiệp và hữu ích."""

    @staticmethod
    def get_comparison_prompt(products: List[Dict[str, Any]]) -> str:
        """Prompt for product comparison"""
        
        if len(products) < 2:
            return "Cần ít nhất 2 sản phẩm để so sánh."
        
        products_text = PromptTemplates._format_products(products)
        
        return f"""Bạn là chuyên gia tư vấn sản phẩm. Hãy so sánh các sản phẩm sau và đưa ra lời khuyên.

Sản phẩm cần so sánh:
{products_text}

Hãy tạo một bảng so sánh chi tiết:
1. So sánh giá cả
2. So sánh thông số kỹ thuật
3. So sánh ưu điểm của từng sản phẩm
4. So sánh nhược điểm
5. Đưa ra lời khuyên dựa trên từng nhu cầu sử dụng
6. Kết luận sản phẩm nào phù hợp nhất cho từng đối tượng

Trả lời bằng tiếng Việt, chi tiết và khách quan."""

    @staticmethod
    def get_no_results_prompt(query: str) -> str:
        """Prompt when no products found"""
        return f"""Tôi hiểu bạn đang tìm kiếm: "{query}"

Tuy nhiên, tôi không tìm thấy sản phẩm nào phù hợp hoàn toàn với yêu cầu của bạn. Điều này có thể do:

1. **Ngân sách**: Sản phẩm bạn quan tâm có thể vượt quá ngân sách
2. **Thương hiệu**: Thương hiệu cụ thể có thể không có sản phẩm phù hợp
3. **Thông số kỹ thuật**: Yêu cầu kỹ thuật có thể quá cụ thể

**Gợi ý của tôi:**
- Mở rộng phạm vi tìm kiếm (ví dụ: tăng ngân sách, thay đổi thương hiệu)
- Cung cấp thêm thông tin về nhu cầu sử dụng
- Cho tôi biết bạn quan tâm đến dòng sản phẩm nào

Bạn có muốn tôi gợi ý một số sản phẩm phổ biến trong khoảng giá tương tự không?"""

    @staticmethod
    def get_fallback_prompt(message: str) -> str:
        """Fallback prompt for general conversation"""
        return f"""Tôi hiểu bạn đang nói về: "{message}"

Tôi có thể hỗ trợ bạn với:
- Tìm kiếm sản phẩm điện thoại
- So sánh sản phẩm
- Tra cứu đơn hàng
- Tư vấn kỹ thuật
- Hỗ trợ mua hàng

Bạn cần hỗ trợ gì cụ thể?"""

    @staticmethod
    def _format_products(products: List[Dict[str, Any]]) -> str:
        """Format products for prompt"""
        if not products:
            return "Không có sản phẩm nào."
        
        formatted_products = []
        for i, product in enumerate(products, 1):
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
            
            formatted_product = f"""
{i}. {name} ({brand})
   - Giá: {price:,} VNĐ
   - Đánh giá: ⭐ {rating}/5
   - Mô tả: {description}
   - Thông số: {specs_text}
   - Điểm phù hợp: {product.get('similarity_score', 0):.2f}
"""
            formatted_products.append(formatted_product)
        
        return "\n".join(formatted_products)

    @staticmethod
    def get_contextual_prompt(
        query: str, 
        context: Dict[str, Any], 
        products: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generate contextual prompt based on query and context"""
        
        # Determine prompt type based on context
        if products and len(products) > 0:
            return PromptTemplates.get_product_search_prompt(query, products)
        elif context.get("order_info"):
            return PromptTemplates.get_order_status_prompt(context["order_info"])
        elif context.get("user_preferences") and products:
            return PromptTemplates.get_product_recommendation_prompt(
                context["user_preferences"], products
            )
        else:
            return PromptTemplates.get_fallback_prompt(query)
