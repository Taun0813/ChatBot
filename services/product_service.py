"""
Product Service
Handles product-related operations
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class Product:
    """Product data model"""
    id: str
    name: str
    description: str
    category: str
    price: float
    stock: int
    features: List[str]
    specifications: Dict[str, Any]
    images: List[str]
    brand: str
    rating: float = 0.0
    reviews_count: int = 0

class ProductService:
    """Service for managing products"""
    
    def __init__(self):
        self.products: Dict[str, Product] = {}
        self.categories: List[str] = []
        loaded = self._load_products_from_dataset()
        if not loaded:
            self._initialize_sample_data()

    def _normalize_category(self, category: str) -> str:
        """Normalize category to supported labels"""
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
        return mapping.get(cat_lower, category or "Khác")

    def _availability_to_stock(self, availability: Optional[str]) -> int:
        """Map availability text to a stock number"""
        if not availability:
            return 0
        availability_lower = availability.strip().lower()
        if "in stock" in availability_lower or "còn hàng" in availability_lower:
            return 10
        if "out of stock" in availability_lower or "hết hàng" in availability_lower:
            return 0
        return 0

    def _load_products_from_dataset(self) -> bool:
        """Load products from processed dataset file"""
        try:
            dataset_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "data", "processed", "products_export.json")
            )
            if not os.path.exists(dataset_path):
                logger.warning("Dataset file not found: %s", dataset_path)
                return False

            with open(dataset_path, "r", encoding="utf-8") as f:
                items = json.load(f)

            if not isinstance(items, list) or not items:
                logger.warning("Dataset is empty or invalid: %s", dataset_path)
                return False

            for item in items:
                if not isinstance(item, dict):
                    continue

                image_url = item.get("image_url") or ""
                images = item.get("images") if isinstance(item.get("images"), list) else []
                if image_url and not images:
                    images = [image_url]

                product_data = {
                    "id": item.get("id") or "",
                    "name": item.get("name") or "",
                    "description": item.get("description") or "",
                    "category": self._normalize_category(item.get("category")),
                    "price": float(item.get("price") or 0),
                    "stock": self._availability_to_stock(item.get("availability")),
                    "features": item.get("features") or [],
                    "specifications": item.get("specifications") or {},
                    "images": images,
                    "brand": item.get("brand") or "",
                    "rating": float(item.get("rating") or 0.0),
                    "reviews_count": int(item.get("reviews_count") or 0),
                }

                if not product_data["id"]:
                    continue

                product = Product(**product_data)
                self.products[product.id] = product

                if product.category not in self.categories:
                    self.categories.append(product.category)

            logger.info("Loaded %d products from dataset", len(self.products))
            return len(self.products) > 0

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False
    
    def _initialize_sample_data(self):
        """Initialize with sample product data"""
        sample_products = [
            {
                "id": "laptop-001",
                "name": "MacBook Pro 16-inch",
                "description": "Powerful laptop for professionals",
                "category": "laptops",
                "price": 2499.99,
                "stock": 10,
                "features": ["M2 Pro chip", "16GB RAM", "512GB SSD", "Retina display"],
                "specifications": {
                    "processor": "M2 Pro",
                    "memory": "16GB",
                    "storage": "512GB SSD",
                    "display": "16-inch Retina",
                    "weight": "2.1 kg"
                },
                "images": ["macbook-pro-1.jpg", "macbook-pro-2.jpg"],
                "brand": "Apple",
                "rating": 4.8,
                "reviews_count": 156
            },
            {
                "id": "phone-001",
                "name": "iPhone 15 Pro",
                "description": "Latest iPhone with advanced features",
                "category": "smartphones",
                "price": 999.99,
                "stock": 25,
                "features": ["A17 Pro chip", "48MP camera", "5G", "Titanium design"],
                "specifications": {
                    "processor": "A17 Pro",
                    "camera": "48MP",
                    "connectivity": "5G",
                    "material": "Titanium",
                    "weight": "187g"
                },
                "images": ["iphone-15-pro-1.jpg", "iphone-15-pro-2.jpg"],
                "brand": "Apple",
                "rating": 4.7,
                "reviews_count": 89
            },
            {
                "id": "headphones-001",
                "name": "Sony WH-1000XM5",
                "description": "Premium noise-canceling headphones",
                "category": "audio",
                "price": 399.99,
                "stock": 15,
                "features": ["Noise canceling", "30-hour battery", "Hi-Res Audio", "Quick charge"],
                "specifications": {
                    "battery": "30 hours",
                    "noise_canceling": True,
                    "connectivity": "Bluetooth 5.2",
                    "weight": "250g"
                },
                "images": ["sony-wh1000xm5-1.jpg", "sony-wh1000xm5-2.jpg"],
                "brand": "Sony",
                "rating": 4.6,
                "reviews_count": 234
            }
        ]
        
        for product_data in sample_products:
            product = Product(**product_data)
            self.products[product.id] = product
        
            if product.category not in self.categories:
                self.categories.append(product.category)
    
    async def search_products(
        self,
        query: str,
        category: Optional[str] = None,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for products"""
        try:
            results = []
            query_lower = query.lower()
            
            for product in self.products.values():
                # Filter by category
                if category and product.category != category:
                    continue
                
                # Filter by price range
                if price_min is not None and product.price < price_min:
                    continue
                if price_max is not None and product.price > price_max:
                    continue
                
                # Search in name, description, and features
                searchable_text = f"{product.name} {product.description} {' '.join(product.features)}".lower()
                
                if query_lower in searchable_text:
                    results.append({
                        "id": product.id,
                        "name": product.name,
                        "description": product.description,
                        "category": product.category,
                        "price": product.price,
                        "stock": product.stock,
                        "brand": product.brand,
                        "rating": product.rating,
                        "reviews_count": product.reviews_count,
                        "features": product.features,
                        "images": product.images
                    })
            
            # Sort by relevance (simple implementation)
            results.sort(key=lambda x: x["rating"], reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return []
    
    async def get_product_details(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a product"""
        try:
            product = self.products.get(product_id)
            if not product:
                return None
            
            return {
                "id": product.id,
                "name": product.name,
                "description": product.description,
                "category": product.category,
                "price": product.price,
                "stock": product.stock,
                "brand": product.brand,
                "rating": product.rating,
                "reviews_count": product.reviews_count,
                "features": product.features,
                "specifications": product.specifications,
                "images": product.images
            }
            
        except Exception as e:
            logger.error(f"Error getting product details: {e}")
            return None
    
    async def get_products_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get products by category"""
        try:
            results = []
            
            for product in self.products.values():
                if product.category == category:
                    results.append({
                        "id": product.id,
                        "name": product.name,
                        "description": product.description,
                        "price": product.price,
                        "stock": product.stock,
                        "brand": product.brand,
                        "rating": product.rating,
                        "images": product.images
                    })
            
            return results[:limit]
                
        except Exception as e:
            logger.error(f"Error getting products by category: {e}")
            return []
    
    async def get_categories(self) -> List[str]:
        """Get all available categories"""
        return self.categories.copy()
    
    async def update_stock(self, product_id: str, quantity: int) -> bool:
        """Update product stock"""
        try:
            product = self.products.get(product_id)
            if not product:
                return False
            
            product.stock = max(0, product.stock + quantity)
            return True
            
        except Exception as e:
            logger.error(f"Error updating stock: {e}")
            return False
    
    async def get_product_stats(self) -> Dict[str, Any]:
        """Get product statistics"""
        try:
            total_products = len(self.products)
            total_categories = len(self.categories)
            
            category_counts = {}
            for product in self.products.values():
                category_counts[product.category] = category_counts.get(product.category, 0) + 1
            
            avg_price = sum(p.price for p in self.products.values()) / total_products if total_products > 0 else 0
            
            return {
                "total_products": total_products,
                "total_categories": total_categories,
                "category_distribution": category_counts,
                "average_price": round(avg_price, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting product stats: {e}")
            return {}

# Global service instance
product_service = ProductService()