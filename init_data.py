"""
Data Initialization Script
Initialize the system with data from dataset/dataset.json
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List
from config import get_settings
from adapters.pinecone_client import PineconeClient
from core.rag_model import RAGModel
from adapters.model_loader import ModelLoaderFactory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataInitializer:
    """Initialize system with dataset data"""
    
    def __init__(self):
        self.settings = get_settings()
        self.pinecone_client = None
        self.rag_model = None
        self.model_loader = None
        
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("Initializing data initialization system...")
            
            # Initialize Pinecone client
            await self._initialize_pinecone()
            
            # Initialize model loader
            await self._initialize_model_loader()
            
            # Initialize RAG model
            await self._initialize_rag_model()
            
            logger.info("Data initialization system ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize data system: {e}")
            raise
    
    async def _initialize_pinecone(self):
        """Initialize Pinecone client"""
        try:
            self.pinecone_client = PineconeClient(
                api_key=self.settings.pinecone_api_key,
                environment=self.settings.pinecone_environment,
                index_name=self.settings.pinecone_index_name,
                dimension=self.settings.pinecone_dimension,
                metric=self.settings.pinecone_metric
            )
            
            await self.pinecone_client.initialize()
            logger.info("Pinecone client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    async def _initialize_model_loader(self):
        """Initialize model loader"""
        try:
            self.model_loader = ModelLoaderFactory.create_loader(
                backend=self.settings.model_loader_backend,
                model_name=self.settings.model_name,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p
            )
            
            logger.info(f"Model loader initialized: {self.settings.model_loader_backend}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model loader: {e}")
            raise
    
    async def _initialize_rag_model(self):
        """Initialize RAG model"""
        try:
            self.rag_model = RAGModel(
                pinecone_client=self.pinecone_client,
                model_loader=self.model_loader
            )
            
            await self.rag_model.initialize()
            logger.info("RAG model initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG model: {e}")
            raise
    
    async def load_dataset(self, dataset_path: str = "training/dataset/dataset.json") -> List[Dict[str, Any]]:
        """Load dataset from JSON file"""
        try:
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
            logger.info(f"Loading dataset from {dataset_path}")
            
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            logger.info(f"Loaded {len(dataset)} products from dataset")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def transform_product_data(self, raw_product: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw product data to our format"""
        try:
            # Extract basic information
            brand = raw_product.get("brand_name", "Unknown").title()
            model = raw_product.get("model", "Unknown")
            price = raw_product.get("price", 0)
            rating = raw_product.get("rating", 0) / 100.0  # Convert to 0-1 scale
            
            # Create product ID
            product_id = f"{brand.lower()}_{model.lower().replace(' ', '_').replace('-', '_')}"
            
            # Extract specifications
            specifications = {
                "màn hình": f"{raw_product.get('screen_size', 0)} inch",
                "ram": f"{raw_product.get('ram_capacity', 0)}GB",
                "rom": f"{raw_product.get('internal_memory', 0)}GB",
                "pin": f"{raw_product.get('battery_capacity', 0)}mAh",
                "camera": f"{raw_product.get('primary_camera_rear', 0)}MP",
                "camera trước": f"{raw_product.get('primary_camera_front', 0)}MP",
                "chip": raw_product.get("processor_brand", "Unknown").title(),
                "tốc độ chip": f"{raw_product.get('processor_speed', 0)}GHz",
                "số nhân": raw_product.get("num_cores", 0),
                "hệ điều hành": raw_product.get("os", "Unknown").title(),
                "tần số quét": f"{raw_product.get('refresh_rate', 0)}Hz",
                "5G": "Có" if raw_product.get("has_5g") == "TRUE" else "Không",
                "NFC": "Có" if raw_product.get("has_nfc") == "TRUE" else "Không",
                "sạc nhanh": f"{raw_product.get('fast_charging', 0)}W" if raw_product.get("fast_charging_available") else "Không"
            }
            
            # Create description
            description = f"{brand} {model} - Điện thoại thông minh với màn hình {raw_product.get('screen_size', 0)} inch, camera {raw_product.get('primary_camera_rear', 0)}MP, pin {raw_product.get('battery_capacity', 0)}mAh"
            
            # Determine category
            category = "Điện thoại"
            
            # Create product data
            product_data = {
                "id": product_id,
                "name": f"{brand} {model}",
                "brand": brand,
                "price": price,
                "description": description,
                "category": category,
                "rating": rating,
                "reviews_count": 0,  # Not available in dataset
                "availability": "In Stock",
                "specifications": specifications,
                "image_url": "",  # Not available in dataset
                "features": self._extract_features(raw_product)
            }
            
            return product_data
            
        except Exception as e:
            logger.error(f"Failed to transform product data: {e}")
            return None
    
    def _extract_features(self, raw_product: Dict[str, Any]) -> List[str]:
        """Extract features from raw product data"""
        features = []
        
        # Camera features
        if raw_product.get("primary_camera_rear", 0) >= 50:
            features.append("camera cao cấp")
        elif raw_product.get("primary_camera_rear", 0) >= 20:
            features.append("camera tốt")
        
        # Battery features
        if raw_product.get("battery_capacity", 0) >= 5000:
            features.append("pin khỏe")
        elif raw_product.get("battery_capacity", 0) >= 4000:
            features.append("pin tốt")
        
        # Performance features
        if raw_product.get("ram_capacity", 0) >= 8:
            features.append("ram cao")
        if raw_product.get("internal_memory", 0) >= 256:
            features.append("rom lớn")
        
        # Display features
        if raw_product.get("screen_size", 0) >= 6.5:
            features.append("màn hình lớn")
        if raw_product.get("refresh_rate", 0) >= 90:
            features.append("tần số quét cao")
        
        # Connectivity features
        if raw_product.get("has_5g") == "TRUE":
            features.append("5G")
        if raw_product.get("has_nfc") == "TRUE":
            features.append("NFC")
        
        # Gaming features
        if raw_product.get("ram_capacity", 0) >= 8 and raw_product.get("processor_speed", 0) >= 2.5:
            features.append("chơi game")
        
        return features
    
    async def ingest_products(self, products: List[Dict[str, Any]], batch_size: int = 50) -> bool:
        """Ingest products into Pinecone"""
        try:
            logger.info(f"Starting to ingest {len(products)} products...")
            
            success_count = 0
            failed_count = 0
            
            # Process in batches
            for i in range(0, len(products), batch_size):
                batch = products[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(products) + batch_size - 1)//batch_size}")
                
                for product in batch:
                    try:
                        # Transform product data
                        transformed_product = self.transform_product_data(product)
                        if not transformed_product:
                            failed_count += 1
                            continue
                        
                        # Upsert to Pinecone
                        success = await self.rag_model.upsert_product(
                            product_id=transformed_product["id"],
                            product_data=transformed_product,
                            namespace="default"
                        )
                        
                        if success:
                            success_count += 1
                        else:
                            failed_count += 1
                            
                    except Exception as e:
                        logger.error(f"Failed to ingest product {product.get('model', 'Unknown')}: {e}")
                        failed_count += 1
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            logger.info(f"Ingestion completed: {success_count} success, {failed_count} failed")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to ingest products: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.pinecone_client:
                await self.pinecone_client.cleanup()
            
            if self.model_loader:
                await self.model_loader.cleanup()
            
            logger.info("Data initializer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main initialization function"""
    try:
        # Create initializer
        initializer = DataInitializer()
        
        # Initialize system
        await initializer.initialize()
        
        # Load dataset
        dataset = await initializer.load_dataset()
        
        # Transform and ingest products
        success = await initializer.ingest_products(dataset)
        
        if success:
            logger.info("Data initialization completed successfully!")
        else:
            logger.error("Data initialization failed!")
        
        # Cleanup
        await initializer.cleanup()
        
    except Exception as e:
        logger.error(f"Data initialization failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())