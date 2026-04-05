"""
Data Initialization Script
Initialize the system with data from dataset/dataset.json
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict, Any, List, Optional
from config import get_settings
from adapters.pinecone_client import PineconeClient
from core.rag_model import RAGModel
from adapters.model_loader import ModelLoaderFactory
from data.schema.product_schema import ProductSchema

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
        self.require_backend_id_for_indexing = bool(
            getattr(self.settings, "require_backend_id_for_indexing", True)
        )
        
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
            logger.error("Failed to initialize data system: %s", e)
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
            logger.error("Failed to initialize Pinecone: %s", e)
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
            
            logger.info("Model loader initialized: %s", self.settings.model_loader_backend)
            
        except Exception as e:
            logger.error("Failed to initialize model loader: %s", e)
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
            logger.error("Failed to initialize RAG model: %s", e)
            raise
    
    async def load_dataset(
        self,
        dataset_path: str = "Mobiles Dataset (2025).csv",
        format: str = "auto"
    ) -> List[Dict[str, Any]]:
        try:
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

            logger.info("Loading dataset from %s", dataset_path)

            if format == "auto":
                if dataset_path.endswith(".json"):
                    format = "generic_json"
                elif dataset_path.endswith(".csv") and "Mobiles" in dataset_path:
                    format = "mobile_csv"
                else:
                    format = "generic_csv"

            if format == "generic_json":
                with open(dataset_path, "r", encoding="utf-8-sig", errors="replace") as f:
                    data = json.load(f)
                products = data if isinstance(data, list) else data.get("products", data.get("items", []))
                logger.info("Loaded %s products from JSON", len(products))
                return products

            import pandas as pd

            df = pd.read_csv(
                dataset_path,
                encoding="latin1",
                engine="python"
            )

            dataset = df.to_dict("records")


            # Log first row keys để debug nếu có lỗi (unknown_unknown)
            if dataset and format == "mobile_csv":
                sample_keys = list(dataset[0].keys())[:3]
                logger.info("CSV columns sample: %s", sample_keys)

            logger.info("Loaded %s products (%s)", len(dataset), format)
            return dataset

        except Exception as e:
            logger.error("Failed to load dataset: %s", e)
            raise

    
    def transform_product_data_generic(
        self,
        raw_product: Dict[str, Any],
        default_category: str = "Khác"
    ) -> Optional[Dict[str, Any]]:
        try:
            normalized = ProductSchema.from_raw(
                raw_product,
                default_category=default_category,
                source="init_data",
                source_id=self._extract_backend_id(raw_product) or None,
            )
            return normalized.to_dict()

        except Exception as e:
            logger.warning(
                f"Transform generic product failed: {e} | "
                f"brand={raw_product.get('Company Name')} name={raw_product.get('Model Name')}"
            )
            return None


    
    def _get_csv_value(self, raw: Dict[str, Any], *keys: str, default: str = "Unknown") -> str:
        """Lấy giá trị từ dict, thử nhiều key (hỗ trợ BOM, tên cột khác nhau)"""
        for k in keys:
            val = raw.get(k)
            if val is not None and str(val).strip():
                return str(val).strip()
        return default

    def _parse_live_status(self, raw_value: Any, default: bool = True) -> bool:
        """Parse live/in_website flags from mixed input formats."""
        if raw_value is None:
            return default
        if isinstance(raw_value, bool):
            return raw_value
        normalized = str(raw_value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "live", "active", "in_website"}:
            return True
        if normalized in {"0", "false", "no", "n", "inactive", "out", "not_live"}:
            return False
        return default

    def _extract_backend_id(self, raw_product: Dict[str, Any]) -> str:
        """Extract backend product ID from common source keys."""
        candidates = (
            raw_product.get("backend_id"),
            raw_product.get("id"),
            raw_product.get("ID"),
            raw_product.get("product_id"),
            raw_product.get("productId"),
            raw_product.get("db_id"),
        )
        for value in candidates:
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str:
                return value_str
        return ""

    def transform_product_data(self, raw_product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform raw product data from CSV to our format"""
        try:
            normalized = ProductSchema.from_raw(
                raw_product,
                default_category="Điện thoại",
                source="init_data",
                source_id=self._extract_backend_id(raw_product) or None,
            )
            return normalized.to_dict()
            
        except Exception as e:
            logger.error("Failed to transform product data: %s | Data: %s", e, raw_product)
            return None
    
    def _extract_features(self, specs: Dict[str, Any], price: int) -> List[str]:
        """Extract features based on transformed specs"""
        features = []
        
        # Parse numeric values again for logic
        try:
            # Parse RAM (format: "6GB" or "8GB")
            ram_str = specs.get("ram", "0")
            ram = float(re.sub(r'[^\d.]', '', ram_str)) if ram_str != "Unknown" else 0
            
            # Parse Battery (format: "3600mAh")
            battery_str = specs.get("pin", "0")
            battery = float(re.sub(r'[^\d.]', '', battery_str)) if battery_str != "Unknown" else 0
            
            # Parse Camera (format: "48MP" or "12MP / 4K")
            cam_str = specs.get("camera", "0")
            cam_match = re.search(r'(\d+)', cam_str)
            cam_main = float(cam_match.group(1)) if cam_match else 0
        except Exception as e:
            logger.warning("Error parsing features: %s", e)
            ram, battery, cam_main = 0, 0, 0
            
        # Camera features
        if cam_main >= 50:
            features.append("camera cao cấp")
        elif cam_main >= 20:
            features.append("camera tốt")
        
        # Battery features
        if battery >= 5000:
            features.append("pin khỏe")
        elif battery >= 4000:
            features.append("pin tốt")
        
        # Performance features
        if ram >= 8:
            features.append("ram cao")
            features.append("đa nhiệm tốt")
            
        # Price segments
        if price > 20000000:
            features.append("cao cấp")
            features.append("sang trọng")
        elif price < 5000000:
            features.append("giá rẻ")
            features.append("sinh viên")
        
        # Default features for modern phones
        features.append("5G")
        features.append("sạc nhanh")
        
        return features
    
    async def ingest_products(
        self,
        products: List[Dict[str, Any]],
        batch_size: int = 50,
        use_generic_transform: bool = False,
    ) -> bool:
        """Ingest products into Pinecone. use_generic_transform=True for laptop/tablet/accessories."""
        try:
            logger.info("Starting to ingest %s products (generic=%s)...", len(products), use_generic_transform)
            
            success_count = 0
            failed_count = 0
            
            def _transform(p):
                if use_generic_transform:
                    return self.transform_product_data_generic(p, default_category="Khác")
                return self.transform_product_data(p)
            
            # Process in batches
            for i in range(0, len(products), batch_size):
                batch = products[i:i + batch_size]
                logger.info("Processing batch %s/%s", i // batch_size + 1, (len(products) + batch_size - 1) // batch_size)
                
                for product in batch:
                    try:
                        transformed_product = _transform(product)
                        if not transformed_product:
                            failed_count += 1
                            continue

                        backend_id = str(transformed_product.get("backend_id") or "").strip()
                        if self.require_backend_id_for_indexing and not backend_id:
                            logger.info(
                                "Skipping product without backend_id: %s",
                                transformed_product.get("name", "Unknown"),
                            )
                            failed_count += 1
                            continue

                        if backend_id and str(transformed_product.get("id", "")).strip() != backend_id:
                            transformed_product["id"] = backend_id
                        
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
                        logger.error("Failed to ingest product %s: %s", product.get("Model Name", "Unknown"), e)
                        failed_count += 1
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            logger.info("Ingestion completed: %s success, %s failed", success_count, failed_count)
            return success_count > 0
            
        except Exception as e:
            logger.error("Failed to ingest products: %s", e)
            return False
        
    async def export_products_to_json(
        self,
        products: List[Dict[str, Any]],
        output_path: str = "data/processed/products_export.json",
        use_generic_transform: bool = False,
    ) -> bool:
        """
        Transform products and export to JSON instead of Pinecone
        """
        try:
            logger.info("Exporting %s products to JSON...", len(products))

            exported = []
            failed = 0

            for raw in products:
                product = (
                    self.transform_product_data_generic(raw)
                    if use_generic_transform
                    else self.transform_product_data(raw)
                )

                if product:
                    exported.append(product)
                else:
                    failed += 1

            if not exported:
                logger.error("No valid products to export")
                return False

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(exported, f, ensure_ascii=False, indent=2)

            logger.info(
                f"Export completed: {len(exported)} success, {failed} failed → {output_path}"
            )
            return True

        except Exception as e:
            logger.error("Failed to export products to JSON: %s", e)
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
            logger.error("Error during cleanup: %s", e)

async def main():
    """Main initialization function. Supports: init_data.py [dataset_path] [--generic]"""
    import sys
    dataset_path = "Mobiles Dataset (2025).csv"
    use_generic = False
    export_json = "--export-json" in sys.argv
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        use_generic = "--generic" in sys.argv or dataset_path.endswith(".json")
    try:
        initializer = DataInitializer()
        await initializer.initialize()
        
        format_type = "generic_json" if dataset_path.endswith(".json") else (
            "generic_csv" if use_generic else "auto"
        )
        dataset = await initializer.load_dataset(dataset_path, format=format_type)
        if export_json:
            success = await initializer.export_products_to_json(
                dataset,
                output_path="data/processed/products_export.json",
                use_generic_transform=use_generic or format_type != "mobile_csv"
            )

            if success:
                logger.info("Export JSON completed successfully!")
            else:
                logger.error("Export JSON failed!")

            await initializer.cleanup()
            return

        
        success = await initializer.ingest_products(
            dataset,
            use_generic_transform=use_generic or format_type != "mobile_csv"
        )
        
        if success:
            logger.info("Data initialization completed successfully!")
        else:
            logger.error("Data initialization failed!")
        
        # Cleanup
        await initializer.cleanup()
        
    except Exception as e:
        logger.error("Data initialization failed: %s", e)

if __name__ == "__main__":
    asyncio.run(main())