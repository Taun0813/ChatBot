"""
Product Service
Handles product-related operations
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

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
        self._initialize_sample_data()
    
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