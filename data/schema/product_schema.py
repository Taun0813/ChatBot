"""
Product Schema Definition - E-commerce
Hỗ trợ: Điện thoại, Laptop, Tablet, Phụ kiện
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

# Danh mục sản phẩm hỗ trợ
PRODUCT_CATEGORIES = [
    "Điện thoại",
    "Laptop",
    "Tablet",
    "Phụ kiện",
    "Đồng hồ thông minh",
    "Tai nghe",
    "Sạc dự phòng",
    "Khác",
]


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


def normalize_category(category: str) -> str:
    """Chuẩn hóa category về danh mục hỗ trợ"""
    cat_lower = (category or "").strip().lower()
    mapping = {
        "phone": "Điện thoại", "mobile": "Điện thoại", "smartphone": "Điện thoại",
        "laptop": "Laptop", "notebook": "Laptop", "macbook": "Laptop",
        "tablet": "Tablet", "ipad": "Tablet",
        "accessory": "Phụ kiện", "phụ kiện": "Phụ kiện", "phu kien": "Phụ kiện",
        "watch": "Đồng hồ thông minh", "smartwatch": "Đồng hồ thông minh",
        "headphone": "Tai nghe", "tai nghe": "Tai nghe", "earphone": "Tai nghe",
        "power bank": "Sạc dự phòng", "sạc dự phòng": "Sạc dự phòng",
    }
    return mapping.get(cat_lower, "Khác")
