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
    
    async def load_dataset(
        self,
        dataset_path: str = "Mobiles Dataset (2025).csv",
        format: str = "auto"
    ) -> List[Dict[str, Any]]:
        try:
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

            logger.info(f"Loading dataset from {dataset_path}")

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
                logger.info(f"Loaded {len(products)} products from JSON")
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
                logger.info(f"CSV columns sample: {sample_keys}")

            logger.info(f"Loaded {len(dataset)} products ({format})")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    
    def transform_product_data_generic(
        self,
        raw_product: Dict[str, Any],
        default_category: str = "Khác"
    ) -> Optional[Dict[str, Any]]:
        try:
            from data.schema.product_schema import normalize_category
            import re

            # --- SAFE GET ---
            def s(val):
                return str(val).strip() if val is not None else ""

            # --- MAP ĐÚNG CSV ---
            name = s(
                raw_product.get("name")
                or raw_product.get("model")
                or raw_product.get("Model Name")
                or raw_product.get("title")
            )

            brand = s(
                raw_product.get("brand")
                or raw_product.get("company")
                or raw_product.get("Company Name")
            )

            if not name or not brand:
                raise ValueError("Missing required field: name or brand")

            category = normalize_category(
                s(raw_product.get("category") or raw_product.get("type") or default_category)
            )

            # --- PRICE: handle 'USD 799', '79,999' ---
            raw_price = raw_product.get("price") or raw_product.get("price_vnd") \
                        or raw_product.get("Launched Price (USA)") or 0
            price = int(re.sub(r"[^\d]", "", str(raw_price)) or 0)

            description = s(raw_product.get("description") or raw_product.get("desc"))

            # --- SPECS ---
            specs = raw_product.get("specifications") or raw_product.get("specs") or {}
            if not isinstance(specs, dict):
                specs = {}

            # --- FEATURES ---
            features = raw_product.get("features") or []
            if isinstance(features, str):
                features = [f.strip() for f in features.split(",") if f.strip()]

            # --- PRODUCT ID (KHÔNG cho Unknown) ---
            raw_id = s(raw_product.get("id") or raw_product.get("product_id"))
            product_id = raw_id or f"{brand}_{name}"
            product_id = re.sub(r"[^a-zA-Z0-9_-]", "", product_id.lower().replace(" ", "_"))

            if not description:
                description = f"{name} - {brand} - {category}. Giá {price:,.0f}."

            return {
                "id": product_id,
                "name": name,
                "brand": brand,
                "price": price,
                "description": description,
                "category": category,
                "rating": float(raw_product.get("rating", 4.5)),
                "reviews_count": int(raw_product.get("reviews_count", 0)),
                "availability": str(raw_product.get("availability", "In Stock")),
                "specifications": specs,
                "image_url": s(raw_product.get("image_url") or raw_product.get("image")),
                "features": features,
            }

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

    def transform_product_data(self, raw_product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform raw product data from CSV to our format"""
        try:
            # Extract basic information (hỗ trợ nhiều tên cột do BOM/encoding)
            brand = self._get_csv_value(
                raw_product,
                "Company Name", "\ufeffCompany Name", "company name", "Company",
                default="Unknown"
            )
            full_model_name = self._get_csv_value(
                raw_product,
                "Model Name", "model name", "Model",
                default="Unknown"
            )
            
            # Attempt to extract storage (ROM) from model name (e.g., "iPhone 16 128GB")
            rom_match = re.search(r'(\d+)(GB|TB)', full_model_name, re.IGNORECASE)
            rom_val = rom_match.group(0) if rom_match else "Unknown"
            
            # Clean Model Name (remove storage info for cleaner name if desired, or keep full)
            model = full_model_name
            
            # Price Conversion (USD to VND)
            # Format: "USD 799" -> 799 -> * 25000
            price_str = str(raw_product.get("Launched Price (USA)", "0") or "0")
            try:
                price_inr = float(raw_product.get("price", 0))
                price_vnd = int(price_inr * 300)
            except:
                price_vnd = 0
            
            # Extract specs directly from JSON fields
            ram = float(raw_product.get("ram_capacity", 0))
            rom = str(raw_product.get("internal_memory", "0"))
            screen_size = float(raw_product.get("screen_size", 0))
            battery = float(raw_product.get("battery_capacity", 0))
            
            # Camera
            back_cam = raw_product.get("primary_camera_rear", "0")
            front_cam = raw_product.get("primary_camera_front", "0")
            
            # Chipset
            processor_brand = str(raw_product.get("processor_brand", "Unknown"))
            processor_speed = str(raw_product.get("processor_speed", ""))
            chip = f"{processor_brand} {processor_speed}GHz".strip()
            
            # OS
            os_type = str(raw_product.get("os", "Android")).capitalize()
            
            front_cam_str = str(raw_product.get("Front Camera", "0"))
            front_cam = float(re.search(r'(\d+)', front_cam_str).group(1)) if re.search(r'(\d+)', front_cam_str) else 0

            # Create product ID (skip nếu cả brand và model đều Unknown - có thể lỗi đọc cột)
            if brand == "Unknown" and full_model_name == "Unknown":
                logger.warning(
                    "Row has Unknown brand/model - kiểm tra tên cột CSV. Keys: %s",
                    list(raw_product.keys())[:5]
                )
            product_id = f"{brand.lower()}_{model.lower().replace(' ', '_').replace('-', '_')}"
            product_id = re.sub(r'[^a-zA-Z0-9_]', '', product_id)
            
            # Extract features boolean
            has_5g = str(raw_product.get("has_5g", "FALSE")).upper() == "TRUE"
            has_nfc = str(raw_product.get("has_nfc", "FALSE")).upper() == "TRUE"
            fast_charging_w = raw_product.get("fast_charging", 0)
            
            # Extract specifications
            specifications = {
                "màn hình": f"{screen_size} inch",
                "ram": f"{int(ram)}GB" if ram > 0 else "Unknown",
                "rom": f"{rom}GB",
                "pin": f"{int(battery)}mAh" if battery > 0 else "Unknown",
                "camera": f"{back_cam}MP",
                "camera trước": f"{front_cam}MP",
                "chip": chip,
                "hệ điều hành": os_type,
                "5G": "Có" if has_5g else "Không",
                "NFC": "Có" if has_nfc else "Không",
                "sạc nhanh": f"{fast_charging_w}W" if fast_charging_w else "Không"
            }
            
            # Create description
            description_parts = [
                f"{model} - Điện thoại {brand}",
                f"màn hình {screen_size} inch" if screen_size > 0 else "",
                f"chip {chip}",
                f"camera chính {back_cam}MP",
                f"pin {int(battery)}mAh" if battery > 0 else "",
                f"RAM {int(ram)}GB" if ram > 0 else "",
                f"bộ nhớ trong {rom}GB",
                f"Hệ điều hành {os_type}"
            ]
            description = ". ".join([p for p in description_parts if p]) + f". Giá khoảng {price_vnd:,.0f} VNĐ."
            
            # Create product data
            product_data = {
                "id": product_id,
                "name": model,
                "brand": brand.capitalize(),
                "price": price_vnd,
                "description": description,
                "category": "Điện thoại",
                "rating": float(raw_product.get("rating", 0)) / 10.0 if raw_product.get("rating") else 4.5, # Rating 0-100 -> 0-10 or 0-5? Assuming 100 scale -> 10 or keep as is? Let's check sample. Sample 89 -> maybe 8.9? App likely expects 5 star. Let's do /20 for 5 star scale or keep raw? JSON has 89, 81. Let's assume /20 for 5-star scale.
                "reviews_count": 0,
                "availability": "In Stock",
                "specifications": specifications,
                "image_url": "",
                "features": self._extract_features(specifications, price_vnd, raw_product)
            }
            
            # Adjust rating to 5-star scale
            raw_rating = float(raw_product.get("rating", 0))
            if raw_rating > 10:
                product_data["rating"] = round(raw_rating / 20.0, 1) # 100 -> 5
            
            return product_data
            
        except Exception as e:
            logger.error(f"Failed to transform product data: {e} | Data: {raw_product.get('model', 'Unknown')}")
            return None
    
    def _extract_features(self, specs: Dict[str, Any], price: int, raw: Dict[str, Any]) -> List[str]:
        """Extract features based on transformed specs"""
        features = []
        
        try:
            # Direct access from raw for reliability
            ram = float(raw.get("ram_capacity", 0))
            battery = float(raw.get("battery_capacity", 0))
            cam_main = float(raw.get("primary_camera_rear", 0))
            has_5g = str(raw.get("has_5g", "FALSE")).upper() == "TRUE"
            refresh_rate = float(raw.get("refresh_rate", 60))
            
            # Camera features
            if cam_main >= 100:
                features.append("camera siêu nét")
            elif cam_main >= 50:
                features.append("camera cao cấp")
            
            # Battery features
            if battery >= 5000:
                features.append("pin trâu")
            
            # Screen features
            if refresh_rate >= 120:
                features.append("màn hình 120Hz")
            elif refresh_rate >= 90:
                features.append("màn hình 90Hz")
            
            # Performance features
            if ram >= 12:
                features.append("cấu hình khủng")
            elif ram >= 8:
                features.append("đa nhiệm tốt")
                
            # Connectivity
            if has_5g:
                features.append("hỗ trợ 5G")
                
            # Price segments
            if price > 20000000:
                features.append("flagship")
                features.append("cao cấp")
            elif price < 5000000:
                features.append("giá rẻ")
                features.append("học sinh sinh viên")
            
        except Exception as e:
            logger.warning(f"Error parsing features: {e}")
            
        return features
    
    async def ingest_products(
        self,
        products: List[Dict[str, Any]],
        batch_size: int = 50,
        use_generic_transform: bool = False,
    ) -> bool:
        """Ingest products into Pinecone. use_generic_transform=True for laptop/tablet/accessories."""
        try:
            logger.info(f"Starting to ingest {len(products)} products (generic={use_generic_transform})...")
            
            success_count = 0
            failed_count = 0
            
            def _transform(p):
                if use_generic_transform:
                    return self.transform_product_data_generic(p, default_category="Khác")
                return self.transform_product_data(p)
            
            # Process in batches
            for i in range(0, len(products), batch_size):
                batch = products[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(products) + batch_size - 1)//batch_size}")
                
                for product in batch:
                    try:
                        transformed_product = _transform(product)
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
                        logger.error(f"Failed to ingest product {product.get('Model Name', 'Unknown')}: {e}")
                        failed_count += 1
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            logger.info(f"Ingestion completed: {success_count} success, {failed_count} failed")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to ingest products: {e}")
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
            logger.info(f"Exporting {len(products)} products to JSON...")

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
            logger.error(f"Failed to export products to JSON: {e}")
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
        logger.error(f"Data initialization failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())