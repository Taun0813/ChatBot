"""
Dataset processing pipeline for the ecommerce agent.

Loads raw CSV/JSON data, normalizes it into the canonical product schema,
and writes standardized processed artifacts for RAG, search, and training.
"""

import csv
import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from data.schema.product_schema import (
    ProductSchema,
    derive_features_from_specifications,
    normalize_text,
)

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """Process raw ecommerce datasets into canonical product artifacts."""

    def __init__(
        self,
        dataset_path: str = "./Mobiles Dataset (2025).csv",
        processed_data_dir: str = "./data/processed",
        default_category: str = "Điện thoại",
    ):
        self.dataset_path = dataset_path
        self.processed_data_dir = processed_data_dir
        self.default_category = default_category
        self.logger = logging.getLogger(self.__class__.__name__)

        os.makedirs(self.processed_data_dir, exist_ok=True)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the raw dataset from CSV or JSON."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        _, extension = os.path.splitext(self.dataset_path.lower())
        if extension == ".json":
            return self._load_json_dataset()
        if extension == ".csv":
            return self._load_csv_dataset()

        # Try CSV first, then JSON as a fallback.
        try:
            return self._load_csv_dataset()
        except Exception:
            return self._load_json_dataset()

    def _load_json_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from JSON file."""
        with open(self.dataset_path, "r", encoding="utf-8-sig") as file:
            data = json.load(file)

        if isinstance(data, list):
            dataset = data
        elif isinstance(data, dict):
            dataset = data.get("products") or data.get("items") or data.get("data") or []
        else:
            dataset = []

        self.logger.info("Loaded %s records from JSON dataset", len(dataset))
        return dataset

    def _load_csv_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from CSV file, handling BOM and mixed encodings."""
        for encoding in ("utf-8-sig", "utf-8", "latin1"):
            try:
                with open(self.dataset_path, "r", encoding=encoding, newline="") as file:
                    reader = csv.DictReader(file)
                    dataset = list(reader)
                self.logger.info("Loaded %s records from CSV dataset using %s", len(dataset), encoding)
                return dataset
            except UnicodeDecodeError:
                continue

        raise UnicodeDecodeError("Unable to decode CSV dataset with supported encodings")

    def process_products(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize raw rows into canonical product objects."""
        products: List[Dict[str, Any]] = []
        source_name = os.path.basename(self.dataset_path)

        for index, raw_product in enumerate(dataset, start=1):
            try:
                normalized = ProductSchema.from_raw(
                    raw_product,
                    default_category=self.default_category,
                    source=source_name,
                    source_id=str(index),
                )
                product = normalized.to_dict()

                # Keep a compact search text to support retrieval and downstream indexing.
                product["search_text"] = self._build_search_text(product)
                product["normalized_category"] = product.get("category", "Khác")
                product["spec_summary"] = self._build_spec_summary(product.get("specifications", {}))
                product["features"] = product.get("features") or derive_features_from_specifications(
                    product.get("specifications", {}),
                    price_vnd=int(product.get("price_vnd") or product.get("price") or 0),
                )
                product["backend_id"] = product.get("backend_id") or product["id"]
                product["is_live"] = bool(product.get("is_live", True))

                products.append(product)
            except Exception as error:
                self.logger.warning("Skipping row %s because normalization failed: %s", index, error)
                continue

        self.logger.info("Processed %s normalized products", len(products))
        return products

    def _build_search_text(self, product: Dict[str, Any]) -> str:
        """Create a lightweight search document for RAG/indexing."""
        parts = [
            normalize_text(product.get("name")),
            normalize_text(product.get("brand")),
            normalize_text(product.get("category")),
            normalize_text(product.get("description")),
        ]

        features = product.get("features") or []
        if isinstance(features, list):
            parts.extend(normalize_text(feature) for feature in features)

        specs = product.get("specifications") or {}
        if isinstance(specs, dict):
            for key, value in specs.items():
                key_text = normalize_text(key)
                value_text = normalize_text(value)
                if key_text and value_text:
                    parts.append(f"{key_text}: {value_text}")

        return " ".join(part for part in parts if part).strip()

    def _build_spec_summary(self, specifications: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a compact set of spec highlights for UI and prompts."""
        if not isinstance(specifications, dict):
            return {}

        preferred_keys = [
            "màn hình",
            "ram",
            "rom",
            "pin",
            "camera",
            "camera trước",
            "camera sau",
            "chip",
            "trọng lượng",
            "hệ điều hành",
            "5g",
            "nfc",
            "sạc nhanh",
        ]
        summary: Dict[str, Any] = {}
        for key in preferred_keys:
            value = specifications.get(key)
            if value:
                summary[key] = value
        return summary

    def generate_conversations(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate sample conversations for training and evaluation."""
        if not products:
            return []

        conversations: List[Dict[str, Any]] = []
        templates = [
            {
                "intent": "product_search",
                "user": "Tôi muốn mua {category} trong tầm giá {price_range}",
                "assistant": "Tôi có thể giúp bạn tìm {category} phù hợp với ngân sách {price_range}. Bạn ưu tiên pin, camera hay hiệu năng?",
            },
            {
                "intent": "product_comparison",
                "user": "So sánh {product1} và {product2} cho tôi",
                "assistant": "Tôi sẽ so sánh {product1} và {product2} theo giá, thông số và trải nghiệm sử dụng để bạn dễ quyết định.",
            },
            {
                "intent": "product_details",
                "user": "Cho tôi biết thông tin chi tiết về {product}",
                "assistant": "Đây là thông tin chi tiết về {product}: {description}",
            },
            {
                "intent": "price_inquiry",
                "user": "Giá của {product} là bao nhiêu?",
                "assistant": "Giá của {product} là {price:,} VNĐ. Bạn có muốn tôi gợi ý thêm vài mẫu cùng tầm giá không?",
            },
            {
                "intent": "warranty_inquiry",
                "user": "{product} có bảo hành bao lâu?",
                "assistant": "Tôi sẽ kiểm tra thông tin bảo hành của {product} cho bạn. Nếu cần, tôi cũng có thể gợi ý chính sách đổi trả.",
            },
            {
                "intent": "shipping_inquiry",
                "user": "Mất bao lâu để giao {product}?",
                "assistant": "Tôi sẽ kiểm tra thời gian giao hàng và trạng thái vận chuyển cho {product}.",
            },
        ]

        categories = sorted({normalize_text(product.get("category"), default="Khác") for product in products})
        category_pool = categories or [self.default_category]

        for index in range(200):
            template = random.choice(templates)
            product = random.choice(products)
            product_two = random.choice(products)
            category = random.choice(category_pool)
            price_range = random.choice(["5-10 triệu", "10-20 triệu", "20-30 triệu", "trên 30 triệu"])

            user_message = template["user"].format(
                category=category,
                price_range=price_range,
                product1=product.get("name", "Sản phẩm A"),
                product2=product_two.get("name", "Sản phẩm B"),
                product=product.get("name", "Sản phẩm"),
                price=int(product.get("price") or product.get("price_vnd") or 0),
            )
            assistant_message = template["assistant"].format(
                category=category,
                price_range=price_range,
                product1=product.get("name", "Sản phẩm A"),
                product2=product_two.get("name", "Sản phẩm B"),
                product=product.get("name", "Sản phẩm"),
                price=int(product.get("price") or product.get("price_vnd") or 0),
                description=normalize_text(product.get("description"))[:220] + ("..." if len(normalize_text(product.get("description"))) > 220 else ""),
            )

            conversations.append(
                {
                    "id": f"conv_{index + 1:04d}",
                    "user_id": f"user_{random.randint(1, 50):03d}",
                    "session_id": f"session_{random.randint(1, 20):03d}",
                    "intent": template["intent"],
                    "products_mentioned": [product.get("id"), product_two.get("id")],
                    "messages": [
                        {
                            "role": "user",
                            "content": user_message,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        {
                            "role": "assistant",
                            "content": assistant_message,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    ],
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        self.logger.info("Generated %s sample conversations", len(conversations))
        return conversations

    def generate_knowledge_base(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate knowledge base entries that ground the assistant in the catalog."""
        knowledge_base: List[Dict[str, Any]] = []
        if not products:
            return knowledge_base

        brands: Dict[str, List[Dict[str, Any]]] = {}
        categories: Dict[str, List[Dict[str, Any]]] = {}

        for product in products:
            brand = normalize_text(product.get("brand"), default="Khác")
            category = normalize_text(product.get("category"), default="Khác")
            brands.setdefault(brand, []).append(product)
            categories.setdefault(category, []).append(product)

        for brand, brand_products in sorted(brands.items()):
            top_products = sorted(brand_products, key=lambda item: float(item.get("price") or item.get("price_vnd") or 0), reverse=True)[:5]
            knowledge_base.append(
                {
                    "id": f"kb_brand_{brand.lower().replace(' ', '_')}",
                    "title": f"Thương hiệu {brand}",
                    "category": "brand_info",
                    "source": "dataset_analysis",
                    "tags": [brand.lower(), "brand", "catalog"],
                    "evidence": [item.get("id") for item in top_products if item.get("id")],
                    "content": {
                        "summary": f"{brand} có {len(brand_products)} sản phẩm trong catalog.",
                        "highlight_products": [
                            {
                                "id": item.get("id"),
                                "name": item.get("name"),
                                "price_vnd": int(item.get("price") or item.get("price_vnd") or 0),
                            }
                            for item in top_products
                        ],
                    },
                }
            )

        for category, category_products in sorted(categories.items()):
            top_products = sorted(category_products, key=lambda item: float(item.get("price") or item.get("price_vnd") or 0), reverse=True)[:5]
            knowledge_base.append(
                {
                    "id": f"kb_category_{category.lower().replace(' ', '_')}",
                    "title": f"Danh mục {category}",
                    "category": "category_info",
                    "source": "dataset_analysis",
                    "tags": [category.lower(), "category", "catalog"],
                    "evidence": [item.get("id") for item in top_products if item.get("id")],
                    "content": {
                        "summary": f"Danh mục {category} có {len(category_products)} sản phẩm.",
                        "price_range_vnd": {
                            "min": int(min(float(item.get("price") or item.get("price_vnd") or 0) for item in category_products)),
                            "max": int(max(float(item.get("price") or item.get("price_vnd") or 0) for item in category_products)),
                        },
                        "highlight_products": [
                            {
                                "id": item.get("id"),
                                "name": item.get("name"),
                                "price_vnd": int(item.get("price") or item.get("price_vnd") or 0),
                            }
                            for item in top_products
                        ],
                    },
                }
            )

        return knowledge_base

    def save_processed_data(
        self,
        products: List[Dict[str, Any]],
        conversations: List[Dict[str, Any]],
        knowledge_base: List[Dict[str, Any]],
    ) -> None:
        """Save normalized data to the processed data directory."""
        try:
            products_file = os.path.join(self.processed_data_dir, "products_export.json")
            with open(products_file, "w", encoding="utf-8") as file:
                json.dump(products, file, ensure_ascii=False, indent=2)
            self.logger.info("Saved %s products to %s", len(products), products_file)

            products_legacy_file = os.path.join(self.processed_data_dir, "products.json")
            with open(products_legacy_file, "w", encoding="utf-8") as file:
                json.dump(products, file, ensure_ascii=False, indent=2)

            conversations_file = os.path.join(self.processed_data_dir, "conversations.json")
            with open(conversations_file, "w", encoding="utf-8") as file:
                json.dump(conversations, file, ensure_ascii=False, indent=2)
            self.logger.info("Saved %s conversations to %s", len(conversations), conversations_file)

            knowledge_base_file = os.path.join(self.processed_data_dir, "knowledge_base.json")
            with open(knowledge_base_file, "w", encoding="utf-8") as file:
                json.dump(knowledge_base, file, ensure_ascii=False, indent=2)
            self.logger.info("Saved %s knowledge base entries to %s", len(knowledge_base), knowledge_base_file)

            training_data = self._create_training_data(products, conversations)
            training_file = os.path.join(self.processed_data_dir, "training_data.json")
            with open(training_file, "w", encoding="utf-8") as file:
                json.dump(training_data, file, ensure_ascii=False, indent=2)
            self.logger.info("Saved %s training examples to %s", len(training_data), training_file)

        except Exception as error:
            self.logger.error("Error saving processed data: %s", error)
            raise

    def _create_training_data(
        self,
        products: List[Dict[str, Any]],
        conversations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Create training examples from normalized conversations."""
        training_data: List[Dict[str, Any]] = []

        for conversation in conversations:
            user_message = conversation["messages"][0]["content"]
            assistant_response = conversation["messages"][1]["content"]
            mentioned_ids = conversation.get("products_mentioned") or []

            product_context = []
            for product_id in mentioned_ids:
                matched_product = next((item for item in products if item.get("id") == product_id), None)
                if matched_product:
                    product_context.append(
                        {
                            "id": matched_product.get("id"),
                            "name": matched_product.get("name"),
                            "brand": matched_product.get("brand"),
                            "category": matched_product.get("category"),
                            "price_vnd": int(matched_product.get("price") or matched_product.get("price_vnd") or 0),
                        }
                    )

            training_data.append(
                {
                    "instruction": "Bạn là trợ lý bán hàng thương mại điện tử. Hãy trả lời ngắn gọn, chính xác, bám sát dữ liệu sản phẩm và không bịa thông tin.",
                    "input": user_message,
                    "output": assistant_response,
                    "intent": conversation["intent"],
                    "grounding": {
                        "product_ids": [item.get("id") for item in product_context if item.get("id")],
                        "products": product_context,
                    },
                    "metadata": {
                        "user_id": conversation["user_id"],
                        "session_id": conversation["session_id"],
                        "timestamp": conversation["timestamp"],
                    },
                }
            )

        return training_data

    def process_all(self) -> None:
        """Run the full processing pipeline."""
        self.logger.info("Starting dataset processing...")
        dataset = self.load_dataset()
        products = self.process_products(dataset)
        conversations = self.generate_conversations(products)
        knowledge_base = self.generate_knowledge_base(products)
        self.save_processed_data(products, conversations, knowledge_base)

        print("\n" + "=" * 60)
        print("DATASET PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Dataset rows: {len(dataset)}")
        print(f"Products processed: {len(products)}")
        print(f"Conversations generated: {len(conversations)}")
        print(f"Knowledge base entries: {len(knowledge_base)}")
        print(f"Training examples: {len(conversations)}")
        print("=" * 60)


def main():
    """Command line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Process ecommerce dataset into canonical artifacts")
    parser.add_argument("--dataset_path", default="./Mobiles Dataset (2025).csv", help="Path to the raw dataset file")
    parser.add_argument("--output_dir", default="./data/processed", help="Output directory for processed artifacts")
    parser.add_argument("--default_category", default="Điện thoại", help="Fallback category when the source lacks one")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    processor = DatasetProcessor(
        dataset_path=args.dataset_path,
        processed_data_dir=args.output_dir,
        default_category=args.default_category,
    )
    processor.process_all()


if __name__ == "__main__":
    main()
