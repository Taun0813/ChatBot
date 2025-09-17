"""
Order Service
Handles order-related operations
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

@dataclass
class OrderItem:
    product_id: str
    quantity: int
    price: float
    name: str

@dataclass
class Order:
    id: str
    customer_id: str
    items: List[OrderItem]
    total_amount: float
    status: OrderStatus
    shipping_address: str
    created_at: datetime
    updated_at: datetime
    notes: Optional[str] = None

class OrderService:
    """Service for managing orders"""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample order data"""
        sample_orders = [
            {
                "id": "order-001",
                "customer_id": "customer-001",
                "items": [
                    OrderItem("laptop-001", 1, 2499.99, "MacBook Pro 16-inch"),
                    OrderItem("headphones-001", 1, 399.99, "Sony WH-1000XM5")
                ],
                "total_amount": 2899.98,
                "status": OrderStatus.DELIVERED,
                "shipping_address": "123 Main St, City, State 12345",
                "created_at": datetime(2024, 1, 15, 10, 30),
                "updated_at": datetime(2024, 1, 20, 14, 45),
                "notes": "Gift wrapping requested"
            },
            {
                "id": "order-002",
                "customer_id": "customer-002",
                "items": [
                    OrderItem("phone-001", 2, 999.99, "iPhone 15 Pro")
                ],
                "total_amount": 1999.98,
                "status": OrderStatus.SHIPPED,
                "shipping_address": "456 Oak Ave, City, State 67890",
                "created_at": datetime(2024, 1, 18, 15, 20),
                "updated_at": datetime(2024, 1, 22, 9, 15)
            }
        ]
        
        for order_data in sample_orders:
            order = Order(**order_data)
            self.orders[order.id] = order
    
    async def create_order(
        self,
        customer_id: str,
        items: List[Dict[str, Any]],
        shipping_address: str,
        notes: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a new order"""
        try:
            order_id = f"order-{uuid.uuid4().hex[:8]}"
            
            # Convert items to OrderItem objects
            order_items = []
            total_amount = 0.0
            
            for item in items:
                order_item = OrderItem(
                    product_id=item["product_id"],
                    quantity=item["quantity"],
                    price=item["price"],
                    name=item.get("name", f"Product {item['product_id']}")
                )
                order_items.append(order_item)
                total_amount += order_item.price * order_item.quantity
            
            # Create order
            order = Order(
                id=order_id,
                customer_id=customer_id,
                items=order_items,
                total_amount=total_amount,
                status=OrderStatus.PENDING,
                shipping_address=shipping_address,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                notes=notes
            )
            
            self.orders[order_id] = order
            
            logger.info(f"Created order {order_id} for customer {customer_id}")
            
            return {
                "id": order.id,
                "customer_id": order.customer_id,
                "total_amount": order.total_amount,
                "status": order.status.value,
                "created_at": order.created_at.isoformat(),
                "items": [
                    {
                        "product_id": item.product_id,
                        "name": item.name,
                        "quantity": item.quantity,
                        "price": item.price
                    }
                    for item in order.items
                ]
            }
            
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status"""
        try:
            order = self.orders.get(order_id)
            if not order:
                return None
            
            return {
                "id": order.id,
                "status": order.status.value,
                "created_at": order.created_at.isoformat(),
                "updated_at": order.updated_at.isoformat(),
                "total_amount": order.total_amount,
                "shipping_address": order.shipping_address
            }
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None
    
    async def update_order_status(self, order_id: str, status: str) -> bool:
        """Update order status"""
        try:
            order = self.orders.get(order_id)
            if not order:
                return False
            
            # Validate status
            try:
                new_status = OrderStatus(status)
            except ValueError:
                logger.error(f"Invalid order status: {status}")
                return False
            
            order.status = new_status
            order.updated_at = datetime.now()
            
            logger.info(f"Updated order {order_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
            return False
    
    async def get_customer_orders(self, customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get orders for a customer"""
        try:
            customer_orders = []
            
            for order in self.orders.values():
                if order.customer_id == customer_id:
                    customer_orders.append({
                        "id": order.id,
                        "status": order.status.value,
                        "total_amount": order.total_amount,
                        "created_at": order.created_at.isoformat(),
                        "updated_at": order.updated_at.isoformat(),
                        "items_count": len(order.items)
                    })
            
            # Sort by creation date (newest first)
            customer_orders.sort(key=lambda x: x["created_at"], reverse=True)
            
            return customer_orders[:limit]
            
        except Exception as e:
            logger.error(f"Error getting customer orders: {e}")
            return []
    
    async def cancel_order(self, order_id: str, reason: Optional[str] = None) -> bool:
        """Cancel an order"""
        try:
            order = self.orders.get(order_id)
            if not order:
                return False
            
            # Only allow cancellation if order is not shipped or delivered
            if order.status in [OrderStatus.SHIPPED, OrderStatus.DELIVERED]:
                logger.warning(f"Cannot cancel order {order_id} with status {order.status.value}")
                return False
            
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            if reason:
                order.notes = f"Cancelled: {reason}"
            
            logger.info(f"Cancelled order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def get_order_details(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed order information"""
        try:
            order = self.orders.get(order_id)
            if not order:
                return None
            
            return {
                "id": order.id,
                "customer_id": order.customer_id,
                "status": order.status.value,
                "total_amount": order.total_amount,
                "shipping_address": order.shipping_address,
                "created_at": order.created_at.isoformat(),
                "updated_at": order.updated_at.isoformat(),
                "notes": order.notes,
                "items": [
                    {
                        "product_id": item.product_id,
                        "name": item.name,
                        "quantity": item.quantity,
                        "price": item.price,
                        "subtotal": item.price * item.quantity
                    }
                    for item in order.items
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting order details: {e}")
            return None
    
    async def get_order_stats(self) -> Dict[str, Any]:
        """Get order statistics"""
        try:
            total_orders = len(self.orders)
            
            status_counts = {}
            for order in self.orders.values():
                status = order.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            total_revenue = sum(order.total_amount for order in self.orders.values())
            avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
            
            return {
                "total_orders": total_orders,
                "status_distribution": status_counts,
                "total_revenue": round(total_revenue, 2),
                "average_order_value": round(avg_order_value, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting order stats: {e}")
            return {}

# Global service instance
order_service = OrderService()
