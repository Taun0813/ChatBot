"""
Spring Boot Microservices Client
Client để giao tiếp với Spring Boot services qua API Gateway
"""
import httpx
import logging
from typing import Dict, Any, Optional, List
from config import get_settings

logger = logging.getLogger(__name__)

class SpringBootClient:
    """Client để giao tiếp với Spring Boot Microservices"""
    
    def __init__(self):
        self.settings = get_settings()
        self.api_gateway_url = self.settings.api_gateway_url
        self.jwt_token = self.settings.jwt_token if hasattr(self.settings, 'jwt_token') else None
        self.timeout = self.settings.api_timeout
        self.client = None
        
    async def initialize(self):
        """Initialize HTTP client"""
        try:
            self.client = httpx.AsyncClient(timeout=self.timeout)
            logger.info("SpringBootClient initialized")
        except Exception as e:
            logger.error("Failed to initialize SpringBootClient: %s", e)
            raise
    
    async def cleanup(self):
        """Cleanup HTTP client"""
        if self.client:
            await self.client.aclose()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        headers = {"Content-Type": "application/json"}
        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        return headers
    
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
                
            if not self.client:
                await self.initialize()
                
            response = await self.client.get(
                f"{self.api_gateway_url}/api/products",
                params=params,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error getting products: %s", e.response.status_code)
            return {"content": [], "totalElements": 0}
        except Exception as e:
            logger.error("Error getting products: %s", e)
            return {"content": [], "totalElements": 0}
    
    async def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product by ID"""
        try:
            if not self.client:
                await self.initialize()
                
            response = await self.client.get(
                f"{self.api_gateway_url}/api/products/{product_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error getting product %s: %s", product_id, e.response.status_code)
            return None
        except Exception as e:
            logger.error("Error getting product %s: %s", product_id, e)
            return None
    
    async def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create order via Order Service"""
        try:
            if not self.client:
                await self.initialize()
                
            response = await self.client.post(
                f"{self.api_gateway_url}/api/orders",
                json=order_data,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error creating order: %s", e.response.status_code)
            raise
        except Exception as e:
            logger.error("Error creating order: %s", e)
            raise
    
    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order by ID"""
        try:
            if not self.client:
                await self.initialize()
                
            response = await self.client.get(
                f"{self.api_gateway_url}/api/orders/{order_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("Order %s not found", order_id)
            else:
                logger.error("HTTP error getting order %s: %s", order_id, e.response.status_code)
            return None
        except Exception as e:
            logger.error("Error getting order %s: %s", order_id, e)
            return None
    
    async def get_user_orders(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all orders for a user"""
        try:
            if not self.client:
                await self.initialize()
                
            response = await self.client.get(
                f"{self.api_gateway_url}/api/orders/user/{user_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()
            return data.get("content", []) if isinstance(data, dict) else data
        except Exception as e:
            logger.error("Error getting user orders: %s", e)
            return []
    
    async def add_to_cart(self, user_id: str, product_id: str, quantity: int = 1) -> Dict[str, Any]:
        """Add item to cart"""
        try:
            if not self.client:
                await self.initialize()
                
            cart_data = {
                "userId": user_id,
                "productId": product_id,
                "quantity": quantity
            }
            
            response = await self.client.post(
                f"{self.api_gateway_url}/api/carts/items",
                json=cart_data,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error adding to cart: %s", e.response.status_code)
            raise
        except Exception as e:
            logger.error("Error adding to cart: %s", e)
            raise
    
    async def get_cart(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's cart"""
        try:
            if not self.client:
                await self.initialize()
                
            response = await self.client.get(
                f"{self.api_gateway_url}/api/carts/user/{user_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("Cart for user %s not found", user_id)
            else:
                logger.error("HTTP error getting cart: %s", e.response.status_code)
            return None
        except Exception as e:
            logger.error("Error getting cart: %s", e)
            return None
    
    async def clear_cart(self, user_id: str) -> bool:
        """Clear user's cart"""
        try:
            if not self.client:
                await self.initialize()
                
            response = await self.client.delete(
                f"{self.api_gateway_url}/api/carts/user/{user_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error("Error clearing cart: %s", e)
            return False
    
    async def process_payment(self, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process payment via Payment Service"""
        try:
            if not self.client:
                await self.initialize()
                
            response = await self.client.post(
                f"{self.api_gateway_url}/api/payments",
                json=payment_data,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error processing payment: %s", e.response.status_code)
            raise
        except Exception as e:
            logger.error("Error processing payment: %s", e)
            raise
    
    async def get_warranty(self, warranty_id: str) -> Optional[Dict[str, Any]]:
        """Get warranty information"""
        try:
            if not self.client:
                await self.initialize()
                
            response = await self.client.get(
                f"{self.api_gateway_url}/api/warranties/{warranty_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Error getting warranty %s: %s", warranty_id, e)
            return None
    
    async def login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Login và lấy JWT token"""
        try:
            if not self.client:
                await self.initialize()
                
            response = await self.client.post(
                f"{self.api_gateway_url}/api/auth/login",
                json={"username": username, "password": password},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            token_data = response.json()
            
            # Store token
            if "accessToken" in token_data:
                self.jwt_token = token_data["accessToken"]
            
            return token_data
        except Exception as e:
            logger.error("Error during login: %s", e)
            return None
    
    def set_token(self, token: str):
        """Set JWT token manually"""
        self.jwt_token = token
