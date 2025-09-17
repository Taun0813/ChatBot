"""
Data Ingestion Module
Handles data ingestion and vector database population
"""

import asyncio
import json
import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DataIngestion:
    """Data ingestion for AI Agent system"""
    
    def __init__(self, pinecone_client, rag_model):
        self.pinecone_client = pinecone_client
        self.rag_model = rag_model
        
    async def ingest_mock_products(self, namespace: str = "default") -> bool:
        """Ingest mock product data into vector database"""
        try:
            logger.info("Starting mock product data ingestion...")
            
            # Generate mock product data
            mock_products = self._generate_mock_products()
            
            # Ingest products into vector database
            success_count = 0
            for product in mock_products:
                success = await self.rag_model.upsert_product(
                    product_id=product["id"],
                    product_data=product,
                    namespace=namespace
                )
                if success:
                    success_count += 1
            
            logger.info(f"Successfully ingested {success_count}/{len(mock_products)} products")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to ingest mock products: {e}")
            return False
    
    def _generate_mock_products(self) -> List[Dict[str, Any]]:
        """Generate mock product data for testing"""
        return [
            {
                "id": "iphone_15_pro_128gb",
                "name": "iPhone 15 Pro 128GB",
                "brand": "Apple",
                "price": 29990000,
                "description": "iPhone 15 Pro với chip A17 Pro mạnh mẽ, camera 48MP, màn hình Super Retina XDR 6.1 inch",
                "category": "Điện thoại",
                "rating": 4.8,
                "reviews_count": 1250,
                "availability": "In Stock",
                "specifications": {
                    "màn hình": "6.1 inch Super Retina XDR",
                    "chip": "A17 Pro",
                    "camera": "48MP chính + 12MP ultra wide + 12MP telephoto",
                    "pin": "Lên đến 23 giờ video",
                    "ram": "8GB",
                    "rom": "128GB"
                }
            },
            {
                "id": "iphone_15_128gb",
                "name": "iPhone 15 128GB",
                "brand": "Apple",
                "price": 22990000,
                "description": "iPhone 15 với chip A16 Bionic, camera 48MP, màn hình Super Retina XDR 6.1 inch",
                "category": "Điện thoại",
                "rating": 4.7,
                "reviews_count": 2100,
                "availability": "In Stock",
                "specifications": {
                    "màn hình": "6.1 inch Super Retina XDR",
                    "chip": "A16 Bionic",
                    "camera": "48MP chính + 12MP ultra wide",
                    "pin": "Lên đến 20 giờ video",
                    "ram": "6GB",
                    "rom": "128GB"
                }
            },
            {
                "id": "samsung_galaxy_s24_ultra_256gb",
                "name": "Samsung Galaxy S24 Ultra 256GB",
                "brand": "Samsung",
                "price": 28990000,
                "description": "Galaxy S24 Ultra với S Pen, camera 200MP, màn hình Dynamic AMOLED 2X 6.8 inch",
                "category": "Điện thoại",
                "rating": 4.6,
                "reviews_count": 890,
                "availability": "In Stock",
                "specifications": {
                    "màn hình": "6.8 inch Dynamic AMOLED 2X",
                    "chip": "Snapdragon 8 Gen 3",
                    "camera": "200MP chính + 50MP periscope + 10MP telephoto + 12MP ultra wide",
                    "pin": "5000mAh",
                    "ram": "12GB",
                    "rom": "256GB"
                }
            },
            {
                "id": "xiaomi_14_pro_256gb",
                "name": "Xiaomi 14 Pro 256GB",
                "brand": "Xiaomi",
                "price": 18990000,
                "description": "Xiaomi 14 Pro với chip Snapdragon 8 Gen 3, camera Leica 50MP, màn hình LTPO AMOLED 6.73 inch",
                "category": "Điện thoại",
                "rating": 4.5,
                "reviews_count": 650,
                "availability": "In Stock",
                "specifications": {
                    "màn hình": "6.73 inch LTPO AMOLED",
                    "chip": "Snapdragon 8 Gen 3",
                    "camera": "50MP Leica chính + 50MP ultra wide + 50MP telephoto",
                    "pin": "4880mAh",
                    "ram": "12GB",
                    "rom": "256GB"
                }
            },
            {
                "id": "oppo_find_x7_ultra_256gb",
                "name": "OPPO Find X7 Ultra 256GB",
                "brand": "OPPO",
                "price": 21990000,
                "description": "OPPO Find X7 Ultra với camera Hasselblad 50MP, chip Snapdragon 8 Gen 3, màn hình LTPO AMOLED 6.82 inch",
                "category": "Điện thoại",
                "rating": 4.4,
                "reviews_count": 420,
                "availability": "In Stock",
                "specifications": {
                    "màn hình": "6.82 inch LTPO AMOLED",
                    "chip": "Snapdragon 8 Gen 3",
                    "camera": "50MP Hasselblad chính + 50MP ultra wide + 50MP periscope + 50MP telephoto",
                    "pin": "5000mAh",
                    "ram": "12GB",
                    "rom": "256GB"
                }
            }
        ]
    
    def _convert_dataset_product(self, raw_product: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Convert dataset product format to our internal format"""
        try:
            # Generate unique ID
            product_id = f"{raw_product['brand_name']}_{raw_product['model'].lower().replace(' ', '_').replace('(', '').replace(')', '')}_{index}"
            
            # Convert price (assuming dataset price is in different currency/format)
            price = int(raw_product['price']) * 1000  # Convert to VND if needed
            
            # Generate description
            description = f"{raw_product['model']} với {raw_product['processor_brand']} {raw_product['processor_speed']}GHz, camera {raw_product['primary_camera_rear']}MP, màn hình {raw_product['screen_size']} inch"
            
            # Build specifications
            specifications = {
                "màn hình": f"{raw_product['screen_size']} inch",
                "chip": f"{raw_product['processor_brand']} {raw_product['processor_speed']}GHz",
                "camera": f"{raw_product['primary_camera_rear']}MP chính + {raw_product['primary_camera_front']}MP selfie",
                "pin": f"{raw_product['battery_capacity']}mAh",
                "ram": f"{raw_product['ram_capacity']}GB",
                "rom": f"{raw_product['internal_memory']}GB",
                "os": raw_product['os'].title(),
                "5g": "Có" if raw_product['has_5g'] == "TRUE" else "Không",
                "nfc": "Có" if raw_product['has_nfc'] == "TRUE" else "Không",
                "sạc nhanh": f"{raw_product['fast_charging']}W" if raw_product['fast_charging_available'] else "Không"
            }
            
            return {
                "id": product_id,
                "name": raw_product['model'],
                "brand": raw_product['brand_name'].title(),
                "price": price,
                "description": description,
                "category": "Điện thoại",
                "rating": raw_product['rating'] / 20,  # Convert 0-100 to 0-5 scale
                "reviews_count": 100,  # Default value
                "availability": "In Stock",
                "specifications": specifications
            }
            
        except Exception as e:
            logger.error(f"Error converting product {index}: {e}")
            # Return a basic product if conversion fails
            return {
                "id": f"product_{index}",
                "name": raw_product.get('model', 'Unknown Model'),
                "brand": raw_product.get('brand_name', 'Unknown').title(),
                "price": int(raw_product.get('price', 0)) * 1000,
                "description": f"Điện thoại {raw_product.get('brand_name', 'Unknown')}",
                "category": "Điện thoại",
                "rating": 4.0,
                "reviews_count": 100,
                "availability": "In Stock",
                "specifications": {}
            }