"""
API Model - Spring Boot Services Integration
Handles API calls to Spring Boot microservices
"""

import asyncio
import logging
import httpx
from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class APIModel:
    """
    API Model for Spring Boot Services Integration
    
    Features:
    - Order service integration (Spring Boot)
    - Payment service integration (Spring Boot)
    - Warranty service integration (Spring Boot)
    - Product service integration (Spring Boot)
    - Mock service support for testing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.mock_data_dir = "services/mock"
        
        # Spring Boot service URLs
        self.services = {
            "order": self.config.get("order_service_url", "http://localhost:8081/api/orders"),
            "payment": self.config.get("payment_service_url", "http://localhost:8082/api/payments"),
            "warranty": self.config.get("warranty_service_url", "http://localhost:8083/api/warranties"),
            "product": self.config.get("product_service_url", "http://localhost:8084/api/products")
        }
        
        # API Keys for Spring Boot services
        self.api_keys = {
            "order": self.config.get("order_service_api_key"),
            "payment": self.config.get("payment_service_api_key"),
            "warranty": self.config.get("warranty_service_api_key"),
            "product": self.config.get("product_service_api_key")
        }
        
        # Timeout settings
        self.timeout = self.config.get("api_timeout", 30)
        
        # HTTP client
        self.client = None
    
    async def initialize(self):
        """Initialize HTTP client"""
        try:
            self.client = httpx.AsyncClient(timeout=self.timeout)
            logger.info("API Model initialized with Spring Boot services")
        except Exception as e:
            logger.error(f"Failed to initialize API Model: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup HTTP client"""
        try:
            if self.client:
                await self.client.aclose()
            logger.info("API Model cleanup completed")
        except Exception as e:
            logger.error(f"Error during API Model cleanup: {e}")
    
    async def _call_spring_boot_service(
        self,
        service_name: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call Spring Boot service"""
        try:
            if not self.client:
                raise ValueError("HTTP client not initialized")
            
            url = f"{self.services[service_name]}/{endpoint.lstrip('/')}"
            headers = {"Content-Type": "application/json"}
            
            # Add API key if available
            if self.api_keys.get(service_name):
                headers["Authorization"] = f"Bearer {self.api_keys[service_name]}"
            
            # Make request
            if method.upper() == "GET":
                response = await self.client.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                response = await self.client.post(url, headers=headers, json=data)
            elif method.upper() == "PUT":
                response = await self.client.put(url, headers=headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling {service_name}: {e.response.status_code}")
            return {"error": f"Service error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Error calling {service_name}: {e}")
            return {"error": str(e)}
    
    async def _fallback_to_mock(self, service_name: str, endpoint: str) -> Dict[str, Any]:
        """Fallback to mock data when Spring Boot service is unavailable"""
        try:
            mock_file = os.path.join(self.mock_data_dir, f"mock_{service_name}.json")
            
            if not os.path.exists(mock_file):
                return {"error": f"Mock data not found for {service_name}"}
            
            with open(mock_file, 'r', encoding='utf-8') as f:
                mock_data = json.load(f)
            
            # Simple endpoint matching
            if "orders" in endpoint:
                return mock_data.get("orders", [])
            elif "payments" in endpoint:
                return mock_data.get("payments", [])
            elif "warranties" in endpoint:
                return mock_data.get("warranties", [])
            else:
                return mock_data
            
        except Exception as e:
            logger.error(f"Error loading mock data for {service_name}: {e}")
            return {"error": str(e)}
        
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
            logger.info(f"Handling order request: {message}")
            
            # Extract order ID from message
            order_id = self._extract_order_id(message)
            
            if not order_id:
                return "Tôi cần số đơn hàng để tra cứu thông tin. Bạn có thể cung cấp số đơn hàng không?"
            
            # Try Spring Boot service first
            try:
                order_info = await self._call_spring_boot_service(
                    service_name="order",
                    endpoint=f"/{order_id}",
                    method="GET"
                )
                
                if "error" in order_info:
                    # Fallback to mock data
                    logger.warning(f"Spring Boot service error, using mock data: {order_info['error']}")
                    order_info = await self._get_order_info(order_id)
                else:
                    # Transform Spring Boot response to our format
                    order_info = self._transform_order_response(order_info)
                    
            except Exception as e:
                logger.warning(f"Spring Boot service unavailable, using mock data: {e}")
                order_info = await self._get_order_info(order_id)
            
            if not order_info:
                return f"Không tìm thấy đơn hàng với số {order_id}. Vui lòng kiểm tra lại số đơn hàng."
            
            # Generate response
            response = self._format_order_response(order_info)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to handle order request: {e}")
            return "Xin lỗi, tôi không thể tra cứu thông tin đơn hàng lúc này. Vui lòng thử lại sau."
    
    async def handle_payment_request(
        self, 
        message: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle payment-related requests using Spring Boot service"""
        try:
            logger.info(f"Handling payment request: {message}")
            
            # Extract order ID or payment ID
            order_id = self._extract_order_id(message)
            
            if not order_id:
                return "Tôi cần số đơn hàng để tra cứu thông tin thanh toán. Bạn có thể cung cấp số đơn hàng không?"
            
            # Try Spring Boot service
            try:
                payment_info = await self._call_spring_boot_service(
                    service_name="payment",
                    endpoint=f"/order/{order_id}",
                    method="GET"
                )
                
                if "error" in payment_info:
                    payment_info = await self._get_payment_info(order_id)
                else:
                    payment_info = self._transform_payment_response(payment_info)
                    
            except Exception as e:
                logger.warning(f"Spring Boot payment service unavailable: {e}")
                payment_info = await self._get_payment_info(order_id)
            
            if not payment_info:
                return f"Không tìm thấy thông tin thanh toán cho đơn hàng {order_id}."
            
            return self._format_payment_response(payment_info)
            
        except Exception as e:
            logger.error(f"Failed to handle payment request: {e}")
            return "Xin lỗi, tôi không thể tra cứu thông tin thanh toán lúc này."
    
    async def handle_warranty_request(
        self, 
        message: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Handle warranty-related requests using Spring Boot service"""
        try:
            logger.info(f"Handling warranty request: {message}")
            
            # Extract product ID or order ID
            product_id = self._extract_product_id(message)
            order_id = self._extract_order_id(message)
            
            if not product_id and not order_id:
                return "Tôi cần số sản phẩm hoặc đơn hàng để tra cứu thông tin bảo hành."
            
            # Try Spring Boot service
            try:
                if product_id:
                    warranty_info = await self._call_spring_boot_service(
                        service_name="warranty",
                        endpoint=f"/product/{product_id}",
                        method="GET"
                    )
                else:
                    warranty_info = await self._call_spring_boot_service(
                        service_name="warranty",
                        endpoint=f"/order/{order_id}",
                        method="GET"
                    )
                
                if "error" in warranty_info:
                    warranty_info = await self._get_warranty_info(product_id or order_id)
                else:
                    warranty_info = self._transform_warranty_response(warranty_info)
                    
            except Exception as e:
                logger.warning(f"Spring Boot warranty service unavailable: {e}")
                warranty_info = await self._get_warranty_info(product_id or order_id)
            
            if not warranty_info:
                return f"Không tìm thấy thông tin bảo hành."
            
            return self._format_warranty_response(warranty_info)
            
        except Exception as e:
            logger.error(f"Failed to handle warranty request: {e}")
            return "Xin lỗi, tôi không thể tra cứu thông tin bảo hành lúc này."
    
    def _transform_order_response(self, spring_boot_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Spring Boot order response to our format"""
        return {
            "order_id": spring_boot_response.get("id"),
            "status": spring_boot_response.get("status"),
            "total_amount": spring_boot_response.get("totalAmount", 0),
            "products": spring_boot_response.get("items", []),
            "shipping_address": spring_boot_response.get("shippingAddress", {}),
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
        end_date = warranty_info.get("end_date", "Unknown")
        terms = warranty_info.get("terms", "Không có thông tin")
        
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
            logger.info(f"Handling general API request: {message}")
            
            # For Phase 1, return basic response
            return """Tôi có thể hỗ trợ bạn với các dịch vụ sau:
            
1. **Tìm kiếm sản phẩm**: Tìm điện thoại theo nhu cầu và ngân sách
2. **Tra cứu đơn hàng**: Kiểm tra trạng thái đơn hàng bằng số đơn
3. **Hỗ trợ kỹ thuật**: Tư vấn về sản phẩm và tính năng
4. **Thông tin bảo hành**: Hướng dẫn về chính sách bảo hành

Bạn cần hỗ trợ gì cụ thể?"""
            
        except Exception as e:
            logger.error(f"Failed to handle general API request: {e}")
            return "Xin lỗi, tôi không thể xử lý yêu cầu lúc này. Vui lòng thử lại sau."
    
    def _extract_order_id(self, message: str) -> Optional[str]:
        """Extract order ID from message"""
        import re
        
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
    
    async def _get_order_info(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order information (mock implementation)"""
        try:
            # Load mock order data
            mock_file = os.path.join(self.mock_data_dir, "mock_order.json")
            
            if not os.path.exists(mock_file):
                logger.warning("Mock order file not found")
                return None
            
            with open(mock_file, 'r', encoding='utf-8') as f:
                mock_orders = json.load(f)
            
            # Find order by ID
            for order in mock_orders:
                if order.get("order_id") == order_id:
                    return order
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get order info: {e}")
            return None
    
    def _format_order_response(self, order_info: Dict[str, Any]) -> str:
        """Format order information into response"""
        try:
            order_id = order_info.get("order_id", "Unknown")
            status = order_info.get("status", "Unknown")
            products = order_info.get("products", [])
            total_amount = order_info.get("total_amount", 0)
            shipping_address = order_info.get("shipping_address", {})
            estimated_delivery = order_info.get("estimated_delivery", "Unknown")
            
            response_parts = [
                f"📦 **Thông tin đơn hàng #{order_id}**",
                f"",
                f"**Trạng thái**: {status}",
                f"**Tổng tiền**: {total_amount:,} VNĐ",
                f"",
                f"**Sản phẩm**:"
            ]
            
            for product in products:
                name = product.get("name", "Unknown")
                quantity = product.get("quantity", 1)
                price = product.get("price", 0)
                response_parts.append(f"- {name} x{quantity} - {price:,} VNĐ")
            
            response_parts.extend([
                f"",
                f"**Địa chỉ giao hàng**:",
                f"- {shipping_address.get('name', 'Unknown')}",
                f"- {shipping_address.get('address', 'Unknown')}",
                f"- {shipping_address.get('phone', 'Unknown')}",
                f"",
                f"**Dự kiến giao hàng**: {estimated_delivery}"
            ])
            
            return "\n".join(response_parts)
                        
        except Exception as e:
            logger.error(f"Failed to format order response: {e}")
            return f"Đơn hàng #{order_info.get('order_id', 'Unknown')} - Trạng thái: {order_info.get('status', 'Unknown')}"
    
    async def get_payment_info(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get payment information for order"""
        try:
            mock_file = os.path.join(self.mock_data_dir, "mock_payment.json")
            
            if not os.path.exists(mock_file):
                return None
            
            with open(mock_file, 'r', encoding='utf-8') as f:
                mock_payments = json.load(f)
            
            for payment in mock_payments:
                if payment.get("order_id") == order_id:
                    return payment
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get payment info: {e}")
            return None
    
    async def get_warranty_info(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get warranty information for product"""
        try:
            mock_file = os.path.join(self.mock_data_dir, "mock_warranty.json")
            
            if not os.path.exists(mock_file):
                return None
            
            with open(mock_file, 'r', encoding='utf-8') as f:
                mock_warranties = json.load(f)
            
            for warranty in mock_warranties:
                if warranty.get("product_id") == product_id:
                    return warranty
            
            return None
                        
        except Exception as e:
            logger.error(f"Failed to get warranty info: {e}")
            return None