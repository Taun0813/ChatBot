"""
API Model - Spring Boot Services Integration
Handles API calls to Spring Boot microservices
"""

import asyncio
import logging
import httpx
import re
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class APIModel:
    """
    API Model for Spring Boot Services Integration
    
    Features:
    - Order service integration (Spring Boot)
    - Payment service integration (Spring Boot)
    - Warranty service integration (Spring Boot)
    - Product service integration (Spring Boot)
    - Spring Boot service integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Spring Boot service URLs
        self.services = {
            "order": self.config.get("order_service_url", "http://localhost:8181/api/orders"),
            "payment": self.config.get("payment_service_url", "http://localhost:8181/api/payments"),
            "warranty": self.config.get("warranty_service_url", "http://localhost:8181/api/warranties"),
            "product": self.config.get("product_service_url", "http://localhost:8181/api/products"),
            "cart": self.config.get("cart_service_url", "http://localhost:8181/api/carts"),
        }
        
        # API Keys for Spring Boot services
        self.api_keys = {
            "order": self.config.get("order_service_api_key"),
            "payment": self.config.get("payment_service_api_key"),
            "warranty": self.config.get("warranty_service_api_key"),
            "product": self.config.get("product_service_api_key"),
            "cart": self.config.get("cart_service_api_key") or self.config.get("order_service_api_key"),
        }
        
        # Timeout settings
        self.timeout = self.config.get("api_timeout", 30)

        # JWT token fallback (used when frontend does not pass token in context)
        self.jwt_token = self.config.get("jwt_token")
        
        # When False: skip real API calls (no mock fallback)
        self.enable_api_calls = self.config.get("enable_api_calls", True)
        
        # HTTP client
        self.client = None
    
    async def initialize(self):
        """Initialize HTTP client"""
        try:
            self.client = httpx.AsyncClient(timeout=self.timeout)
            logger.info("API Model initialized with Spring Boot services")
        except Exception as e:
            logger.error("Failed to initialize API Model: %s", e)
            raise
    
    async def cleanup(self):
        """Cleanup HTTP client"""
        try:
            if self.client:
                await self.client.aclose()
            logger.info("API Model cleanup completed")
        except Exception as e:
            logger.error("Error during API Model cleanup: %s", e)
    
    async def _call_spring_boot_service(
        self,
        service_name: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call Spring Boot service"""
        try:
            if not self.client:
                logger.warning("HTTP client not initialized, auto-initializing API model client")
                await self.initialize()
                if not self.client:
                    raise ValueError("HTTP client not initialized")
            
            url = f"{self.services[service_name]}/{endpoint.lstrip('/')}"
            headers = {"Content-Type": "application/json"}

            token_from_context = self._extract_jwt_from_context(context)
            auth_token = token_from_context or self.jwt_token

            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
            
            # Add API key only when Authorization is not set by JWT
            if "Authorization" not in headers and self.api_keys.get(service_name):
                headers["Authorization"] = f"Bearer {self.api_keys[service_name]}"
            
            # Make request
            if method.upper() == "GET":
                response = await self.client.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                response = await self.client.post(url, headers=headers, json=data)
            elif method.upper() == "PUT":
                response = await self.client.put(url, headers=headers, json=data)
            elif method.upper() == "DELETE":
                response = await self.client.delete(url, headers=headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error calling %s: %s", service_name, e.response.status_code)
            return {
                "error": f"Service error: {e.response.status_code}",
                "status_code": e.response.status_code
            }
        except Exception as e:
            logger.error("Error calling %s: %s", service_name, e)
            return {"error": str(e)}

    def _extract_jwt_from_context(self, context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Extract JWT token from context payload from frontend."""
        if not context or not isinstance(context, dict):
            return None

        token = (
            context.get("jwt_token")
            or context.get("access_token")
            or context.get("token")
            or context.get("authorization")
        )

        if isinstance(token, str) and token.strip():
            token = token.strip()
            if token.lower().startswith("bearer "):
                return token[7:].strip()
            return token
        return None
    
    async def handle_order_request(
        self, 
        message: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Handle order-related requests using Spring Boot service
        
        Args:
            message: User message about orders
            user_id: User identifier
            context: Additional context
        
        Returns:
            Response about order status
        """
        try:
            logger.info("Handling order request: %s", message)
            
            # Extract order ID from message
            order_id = self._extract_order_id(message)
            
            if not order_id:
                return "Tôi cần số đơn hàng để tra cứu thông tin. Bạn có thể cung cấp số đơn hàng không?"
            
            if not self.enable_api_calls:
                return "Tính năng gọi API hiện đang tắt nên không thể tra cứu đơn hàng."

            order_info = await self._call_spring_boot_service(
                service_name="order",
                endpoint=f"/{order_id}",
                method="GET",
                context=context
            )
            if "error" in order_info:
                logger.warning("Spring Boot service error: %s", order_info["error"])
                if order_info.get("status_code") == 401:
                    return "Phiên đăng nhập không hợp lệ hoặc đã hết hạn khi tra cứu đơn hàng. Vui lòng đăng nhập lại và gửi kèm token cho API /ask."
                return "Xin lỗi, hiện không thể tra cứu thông tin đơn hàng. Vui lòng thử lại sau."

            order_info = self._transform_order_response(order_info)
            
            if not order_info:
                return f"Không tìm thấy đơn hàng với số {order_id}. Vui lòng kiểm tra lại số đơn hàng."
            
            # Generate response
            response = self._format_order_response(order_info)
            
            return response
            
        except Exception as e:
            logger.error("Failed to handle order request: %s", e)
            return "Xin lỗi, tôi không thể tra cứu thông tin đơn hàng lúc này. Vui lòng thử lại sau."
    
    async def handle_payment_request(
        self, 
        message: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle payment-related requests using Spring Boot service"""
        try:
            logger.info("Handling payment request: %s", message)
            
            # Extract order ID or payment ID
            order_id = self._extract_order_id(message)
            
            if not order_id:
                return "Tôi cần số đơn hàng để tra cứu thông tin thanh toán. Bạn có thể cung cấp số đơn hàng không?"
            
            if not self.enable_api_calls:
                return "Tính năng gọi API hiện đang tắt nên không thể tra cứu thanh toán."

            payment_info = await self._call_spring_boot_service(
                service_name="payment",
                endpoint=f"/order/{order_id}",
                method="GET",
                context=context
            )
            if "error" in payment_info:
                logger.warning("Spring Boot payment error: %s", payment_info["error"])
                if payment_info.get("status_code") == 401:
                    return "Phiên đăng nhập không hợp lệ hoặc đã hết hạn khi tra cứu thanh toán. Vui lòng đăng nhập lại và gửi kèm token cho API /ask."
                return "Xin lỗi, hiện không thể tra cứu thông tin thanh toán. Vui lòng thử lại sau."

            payment_info = self._transform_payment_response(payment_info)
            
            if not payment_info:
                return f"Không tìm thấy thông tin thanh toán cho đơn hàng {order_id}."
            
            return self._format_payment_response(payment_info)
            
        except Exception as e:
            logger.error("Failed to handle payment request: %s", e)
            return "Xin lỗi, tôi không thể tra cứu thông tin thanh toán lúc này."
    
    async def handle_warranty_request(
        self, 
        message: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle warranty-related requests using Spring Boot service"""
        try:
            logger.info("Handling warranty request: %s", message)
            
            # Extract product ID or order ID
            product_id = self._extract_product_id(message)
            order_id = self._extract_order_id(message)
            
            if not product_id and not order_id:
                return "Tôi cần số sản phẩm hoặc đơn hàng để tra cứu thông tin bảo hành."
            
            if not self.enable_api_calls:
                return "Tính năng gọi API hiện đang tắt nên không thể tra cứu bảo hành."

            if product_id:
                warranty_info = await self._call_spring_boot_service(
                    service_name="warranty",
                    endpoint=f"/product/{product_id}",
                    method="GET",
                    context=context
                )
            else:
                warranty_info = await self._call_spring_boot_service(
                    service_name="warranty",
                    endpoint=f"/order/{order_id}",
                    method="GET",
                    context=context
                )

            if "error" in warranty_info:
                logger.warning("Spring Boot warranty error: %s", warranty_info["error"])
                if warranty_info.get("status_code") == 401:
                    return "Phiên đăng nhập không hợp lệ hoặc đã hết hạn khi tra cứu bảo hành. Vui lòng đăng nhập lại và gửi kèm token cho API /ask."
                return "Xin lỗi, hiện không thể tra cứu thông tin bảo hành. Vui lòng thử lại sau."

            warranty_info = self._transform_warranty_response(warranty_info)
            
            if not warranty_info:
                return f"Không tìm thấy thông tin bảo hành."
            
            return self._format_warranty_response(warranty_info)
            
        except Exception as e:
            logger.error("Failed to handle warranty request: %s", e)
            return "Xin lỗi, tôi không thể tra cứu thông tin bảo hành lúc này."
    
    def _transform_order_response(self, spring_boot_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Spring Boot order response to our format"""
        items = spring_boot_response.get("items", []) or []
        normalized_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized_items.append({
                "id": item.get("id"),
                "product_id": item.get("productId"),
                "name": item.get("name") or item.get("productName") or "Unknown",
                "quantity": item.get("quantity", 1),
                "price": item.get("price") if item.get("price") is not None else item.get("unitPrice", 0),
                "subtotal": item.get("subtotal", 0)
            })

        return {
            "order_id": spring_boot_response.get("id"),
            "order_number": spring_boot_response.get("orderNumber"),
            "user_id": spring_boot_response.get("userId"),
            "status": spring_boot_response.get("status"),
            "total_amount": spring_boot_response.get("totalAmount", 0),
            "payment_method": spring_boot_response.get("paymentMethod"),
            "notes": spring_boot_response.get("notes"),
            "products": normalized_items,
            "shipping_address": spring_boot_response.get("shippingAddress"),
            "shipping_city": spring_boot_response.get("shippingCity"),
            "shipping_postal_code": spring_boot_response.get("shippingPostalCode"),
            "estimated_delivery": spring_boot_response.get("estimatedDelivery"),
            "created_at": spring_boot_response.get("createdAt"),
            "updated_at": spring_boot_response.get("updatedAt")
        }
    
    def _transform_payment_response(self, spring_boot_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Spring Boot payment response to our format"""
        return {
            "payment_id": spring_boot_response.get("id"),
            "order_id": spring_boot_response.get("orderId"),
            "amount": spring_boot_response.get("amount", 0),
            "status": spring_boot_response.get("status"),
            "method": spring_boot_response.get("paymentMethod"),
            "transaction_id": spring_boot_response.get("transactionId"),
            "created_at": spring_boot_response.get("createdAt")
        }
    
    def _transform_warranty_response(self, spring_boot_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Spring Boot warranty response to our format"""
        return {
            "warranty_id": spring_boot_response.get("id"),
            "product_id": spring_boot_response.get("productId"),
            "order_id": spring_boot_response.get("orderId"),
            "status": spring_boot_response.get("status"),
            "start_date": spring_boot_response.get("startDate"),
            "end_date": spring_boot_response.get("endDate"),
            "terms": spring_boot_response.get("terms", ""),
            "created_at": spring_boot_response.get("createdAt")
        }
    
    def _extract_product_id(self, message: str) -> Optional[str]:
        """Extract product ID from message"""
        import re
        patterns = [
            r'sản phẩm\s+(\d+)',
            r'product\s+(\d+)',
            r'item\s+(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                return match.group(1)
        
        return None
    
    def _format_payment_response(self, payment_info: Dict[str, Any]) -> str:
        """Format payment information into response"""
        payment_id = payment_info.get("payment_id", "Unknown")
        order_id = payment_info.get("order_id", "Unknown")
        amount = payment_info.get("amount", 0)
        status = payment_info.get("status", "Unknown")
        method = payment_info.get("method", "Unknown")
        
        return f"""💳 **Thông tin thanh toán**
        
**Mã thanh toán**: {payment_id}
**Đơn hàng**: #{order_id}
**Số tiền**: {amount:,} VNĐ
**Trạng thái**: {status}
**Phương thức**: {method}

Bạn cần hỗ trợ gì thêm về thanh toán?"""
    
    def _format_warranty_response(self, warranty_info: Dict[str, Any]) -> str:
        """Format warranty information into response"""
        warranty_id = warranty_info.get("warranty_id", "Unknown")
        product_id = warranty_info.get("product_id", "Unknown")
        status = warranty_info.get("status", "Unknown")
        start_date = warranty_info.get("start_date", "Unknown")
        terms = warranty_info.get("terms", "Không có thông tin")
        end_date = warranty_info.get("end_date", "Unknown")
        
        return f"""🛡️ **Thông tin bảo hành**
        
**Mã bảo hành**: {warranty_id}
**Sản phẩm**: {product_id}
**Trạng thái**: {status}
**Ngày bắt đầu**: {start_date}
**Ngày kết thúc**: {end_date}
**Điều khoản**: {terms}

Bạn cần hỗ trợ gì thêm về bảo hành?"""
    
    async def handle_general_request(
        self, 
        message: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Handle general API requests
        
        Args:
            message: User message
            user_id: User identifier
            context: Additional context
        
        Returns:
            Response about API services
        """
        try:
            logger.info("Handling general API request: %s", message)
            
            # For Phase 1, return basic response
            return """Tôi có thể hỗ trợ bạn với các dịch vụ sau:
            
1. **Tìm kiếm sản phẩm**: Tìm điện thoại theo nhu cầu và ngân sách
2. **Tra cứu đơn hàng**: Kiểm tra trạng thái đơn hàng bằng số đơn
3. **Hỗ trợ kỹ thuật**: Tư vấn về sản phẩm và tính năng
4. **Thông tin bảo hành**: Hướng dẫn về chính sách bảo hành

Bạn cần hỗ trợ gì cụ thể?"""
            
        except Exception as e:
            logger.error("Failed to handle general API request: %s", e)
            return "Xin lỗi, tôi không thể xử lý yêu cầu lúc này. Vui lòng thử lại sau."

    async def handle_cart_request(
        self,
        message: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Handle cart operations using cart service endpoints."""
        try:
            logger.info("Handling cart request: %s", message)

            if not self.enable_api_calls:
                return "Tính năng gọi API hiện đang tắt nên chưa thể thao tác giỏ hàng."

            context = context or {}
            action = self._detect_cart_action(message, context)

            if action == "get_admin":
                admin_user_id = (
                    context.get("target_user_id")
                    or context.get("user_id")
                    or self._extract_user_id(message)
                )
                if not admin_user_id:
                    return "Vui lòng cung cấp userId để xem giỏ hàng theo quyền ADMIN."

                cart_payload = await self._call_spring_boot_service(
                    service_name="cart",
                    endpoint=f"/{admin_user_id}",
                    method="GET",
                    context=context,
                )
                return self._format_cart_response(cart_payload, action="get_admin")

            if action == "clear":
                clear_payload = await self._call_spring_boot_service(
                    service_name="cart",
                    endpoint="/clear",
                    method="DELETE",
                    context=context,
                )
                return self._format_cart_response(clear_payload, action="clear")

            if action == "remove_item":
                item_id = context.get("item_id") or self._extract_item_id(message)
                if not item_id:
                    return "Vui lòng cung cấp itemId để xóa sản phẩm khỏi giỏ hàng."

                remove_payload = await self._call_spring_boot_service(
                    service_name="cart",
                    endpoint=f"/items/{item_id}",
                    method="DELETE",
                    context=context,
                )
                return self._format_cart_response(remove_payload, action="remove_item")

            if action == "update_item":
                item_id = context.get("item_id") or self._extract_item_id(message)
                quantity = context.get("quantity") or self._extract_quantity(message)
                if not item_id or quantity <= 0:
                    return "Vui lòng cung cấp itemId và quantity hợp lệ để cập nhật giỏ hàng."

                update_payload = await self._call_spring_boot_service(
                    service_name="cart",
                    endpoint=f"/items/{item_id}",
                    method="PUT",
                    data={"quantity": int(quantity)},
                    context=context,
                )
                return self._format_cart_response(update_payload, action="update_item")

            if action == "add_item":
                product_id = context.get("product_id") or self._extract_product_id_for_cart(message)
                quantity = context.get("quantity") or self._extract_quantity(message) or 1
                if not product_id:
                    return "Vui lòng cung cấp productId để thêm sản phẩm vào giỏ hàng."

                add_payload = await self._call_spring_boot_service(
                    service_name="cart",
                    endpoint="/items",
                    method="POST",
                    data={
                        "productId": str(product_id),
                        "quantity": int(quantity),
                    },
                    context=context,
                )
                return self._format_cart_response(add_payload, action="add_item")

            # default action: get current cart
            cart_payload = await self._call_spring_boot_service(
                service_name="cart",
                endpoint="/me",
                method="GET",
                context=context,
            )
            return self._format_cart_response(cart_payload, action="get_me")

        except Exception as e:
            logger.error("Failed to handle cart request: %s", e)
            return "Xin lỗi, tôi không thể xử lý giỏ hàng lúc này. Vui lòng thử lại sau."
    
    def _extract_order_id(self, message: str) -> Optional[str]:
        """Extract order ID from message"""
        # Look for patterns like #1234, order 1234, đơn hàng 1234
        patterns = [
            r'#(\d+)',
            r'order\s+(\d+)',
            r'đơn\s+hàng\s+(\d+)',
            r'số\s+(\d+)',
            r'(\d{4,})'  # Any 4+ digit number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                return match.group(1)
        
        return None

    def _extract_item_id(self, message: str) -> Optional[str]:
        """Extract cart item ID from user message."""
        text = (message or "").strip().lower()
        patterns = [
            r"item\s*id\s*[:#-]?\s*([a-z0-9_-]+)",
            r"item\s*[:#-]?\s*([a-z0-9_-]+)",
            r"cart\s*item\s*[:#-]?\s*([a-z0-9_-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _extract_product_id_for_cart(self, message: str) -> Optional[str]:
        """Extract product ID for add-to-cart flow."""
        text = (message or "").strip().lower()
        patterns = [
            r"product\s*id\s*[:#-]?\s*([a-z0-9_-]+)",
            r"sản\s*phẩm\s*[:#-]?\s*([a-z0-9_-]+)",
            r"mã\s*sản\s*phẩm\s*[:#-]?\s*([a-z0-9_-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _extract_quantity(self, message: str) -> int:
        """Extract quantity from message text."""
        text = (message or "").strip().lower()
        patterns = [
            r"số\s*lượng\s*[:=]?\s*(\d+)",
            r"quantity\s*[:=]?\s*(\d+)",
            r"x\s*(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    value = int(match.group(1))
                    return max(0, value)
                except (TypeError, ValueError):
                    continue
        return 0

    def _extract_user_id(self, message: str) -> Optional[str]:
        """Extract user id from message for admin cart lookup."""
        text = (message or "").strip()
        patterns = [
            r"user\s*id\s*[:#-]?\s*([a-zA-Z0-9_-]+)",
            r"user\s*[:#-]?\s*([a-zA-Z0-9_-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _detect_cart_action(self, message: str, context: Dict[str, Any]) -> str:
        """Infer cart operation from message and context hints."""
        hinted = str(context.get("cart_action", "")).strip().lower()
        if hinted in {"get_me", "add_item", "update_item", "remove_item", "clear", "get_admin"}:
            return hinted

        text = (message or "").strip().lower()

        if any(token in text for token in ["admin", "quản trị", "xem giỏ user", "giỏ của user"]):
            return "get_admin"
        if any(token in text for token in ["clear", "xóa hết", "xoa het", "làm trống", "lam trong"]):
            return "clear"
        if any(token in text for token in ["cập nhật", "cap nhat", "update", "đổi số lượng", "doi so luong"]):
            return "update_item"
        if any(token in text for token in ["xóa", "xoa", "remove", "delete"]):
            return "remove_item"
        if any(token in text for token in ["thêm", "them", "add"]):
            return "add_item"
        return "get_me"

    def _format_cart_response(self, payload: Dict[str, Any], action: str) -> str:
        """Format cart service payload into user-friendly text."""
        if payload is None:
            return "Không nhận được dữ liệu giỏ hàng từ backend."

        if "error" in payload:
            status_code = payload.get("status_code")
            if status_code == 401:
                return "Phiên đăng nhập không hợp lệ hoặc đã hết hạn khi thao tác giỏ hàng. Vui lòng đăng nhập lại."
            if status_code == 403:
                return "Bạn không có quyền thực hiện thao tác giỏ hàng này."
            if status_code == 404:
                return "Không tìm thấy giỏ hàng hoặc item tương ứng."
            return "Xin lỗi, hiện không thể thao tác giỏ hàng. Vui lòng thử lại sau."

        action_messages = {
            "add_item": "Đã thêm sản phẩm vào giỏ hàng thành công.",
            "update_item": "Đã cập nhật số lượng sản phẩm trong giỏ hàng.",
            "remove_item": "Đã xóa sản phẩm khỏi giỏ hàng.",
            "clear": "Đã xóa toàn bộ giỏ hàng.",
            "get_admin": "Thông tin giỏ hàng của user:",
            "get_me": "Thông tin giỏ hàng hiện tại của bạn:",
        }

        items = payload.get("items") or payload.get("cartItems") or []
        total = payload.get("totalAmount")
        total_items = payload.get("totalItems")
        cart_id = payload.get("id") or payload.get("cartId")

        lines = [action_messages.get(action, "Thao tác giỏ hàng thành công.")]
        if cart_id is not None:
            lines.append(f"Mã giỏ hàng: {cart_id}")
        if total_items is not None:
            lines.append(f"Tổng số lượng: {total_items}")
        if total is not None:
            try:
                lines.append(f"Tạm tính: {float(total):,.0f} VNĐ")
            except Exception:
                lines.append(f"Tạm tính: {total}")

        if isinstance(items, list) and items:
            lines.append("Sản phẩm trong giỏ:")
            for idx, item in enumerate(items[:5], 1):
                name = item.get("productName") or item.get("name") or item.get("product_id") or "Sản phẩm"
                quantity = item.get("quantity", 1)
                price = item.get("price") if item.get("price") is not None else item.get("unitPrice")
                if price is not None:
                    try:
                        lines.append(f"{idx}. {name} x{quantity} - {float(price):,.0f} VNĐ")
                    except Exception:
                        lines.append(f"{idx}. {name} x{quantity} - {price}")
                else:
                    lines.append(f"{idx}. {name} x{quantity}")

        return "\n".join(lines)
    
    
    def _format_order_response(self, order_info: Dict[str, Any]) -> str:
        """Format order information into response"""
        try:
            def _fmt_money(value: Any) -> str:
                try:
                    return f"{float(value):,.0f} VNĐ"
                except Exception:
                    return f"{value} VNĐ"

            def _fmt_time(value: Any) -> str:
                if not value:
                    return "Không có"
                raw = str(value).strip()
                normalized = raw.replace("T", " ").replace("Z", "")
                return normalized

            status_map = {
                "PENDING": "Chờ xử lý",
                "PROCESSING": "Đang xử lý",
                "SHIPPED": "Đang giao",
                "DELIVERED": "Đã giao",
                "CANCELLED": "Đã hủy",
                "FAILED": "Thất bại",
                "PAID": "Đã thanh toán",
                "UNPAID": "Chưa thanh toán",
            }

            order_id = order_info.get("order_id", "Unknown")
            order_number = order_info.get("order_number")
            raw_status = str(order_info.get("status", "Unknown"))
            status = status_map.get(raw_status.upper(), raw_status)
            products = order_info.get("products", [])
            total_amount = order_info.get("total_amount", 0)
            shipping_address = order_info.get("shipping_address")
            shipping_city = order_info.get("shipping_city")
            shipping_postal_code = order_info.get("shipping_postal_code")
            payment_method = order_info.get("payment_method", "Unknown")
            created_at = order_info.get("created_at", "Unknown")
            updated_at = order_info.get("updated_at", "Unknown")

            if isinstance(shipping_address, dict):
                address_lines = [
                    shipping_address.get("name"),
                    shipping_address.get("address"),
                    shipping_address.get("phone")
                ]
            else:
                address_lines = [shipping_address, shipping_city, shipping_postal_code]
            address_lines = [str(line) for line in address_lines if line]
            
            response_parts = [
                "📦 **Chi tiết đơn hàng**",
                f"**Mã đơn hàng**: {order_number or f'#{order_id}'}",
                f"**Trạng thái**: {status}",
                f"**Tổng tiền**: {_fmt_money(total_amount)}",
                f"**Phương thức thanh toán**: {payment_method}",
                "",
                "🛒 **Sản phẩm trong đơn**"
            ]
            
            for idx, product in enumerate(products, 1):
                name = product.get("name", "Unknown")
                quantity = int(product.get("quantity", 1) or 1)
                price = product.get("price", 0)
                subtotal = product.get("subtotal")
                line = f"{idx}. {name} ×{quantity} — {_fmt_money(price)}"
                if subtotal is not None:
                    line += f" (tạm tính: {_fmt_money(subtotal)})"
                response_parts.append(line)

            if not products:
                response_parts.append("- Không có dữ liệu sản phẩm")
            
            response_parts.extend([
                "",
                "📍 **Địa chỉ giao hàng**"
            ])

            if address_lines:
                for line in address_lines:
                    response_parts.append(f"- {line}")
            else:
                response_parts.append("- Chưa có thông tin địa chỉ")

            response_parts.extend([
                "",
                "🕒 **Mốc thời gian**",
                f"- Tạo lúc: {_fmt_time(created_at)}",
                f"- Cập nhật: {_fmt_time(updated_at)}"
            ])
            
            return "\n".join(response_parts)
                        
        except Exception as e:
            logger.error("Failed to format order response: %s", e)
            return f"Đơn hàng #{order_info.get('order_id', 'Unknown')} - Trạng thái: {order_info.get('status', 'Unknown')}"
    
    