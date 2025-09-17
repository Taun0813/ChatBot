"""
Product Schema Definition
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class ProductSchema:
    id: str
    name: str
    description: str
    category: str
    price: float
    brand: str
    features: List[str]
    specifications: Dict[str, Any]
    images: List[str]
    rating: float = 0.0
    reviews_count: int = 0
    stock: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
