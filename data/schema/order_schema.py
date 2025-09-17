"""
Order Schema Definition
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

@dataclass
class OrderItemSchema:
    product_id: str
    quantity: int
    price: float
    name: str

@dataclass
class OrderSchema:
    id: str
    customer_id: str
    items: List[OrderItemSchema]
    total_amount: float
    status: OrderStatus
    shipping_address: str
    created_at: datetime
    updated_at: datetime
    notes: Optional[str] = None
