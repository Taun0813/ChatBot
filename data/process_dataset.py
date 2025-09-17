"""
Process the mobile phone dataset for AI Agent system
"""
import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class DatasetProcessor:
    """Process mobile phone dataset for AI Agent"""
    
    def __init__(self, dataset_path: str = "./training/dataset/dataset.json"):
        self.dataset_path = dataset_path
        self.processed_data_dir = "./data/processed"
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create processed data directory
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the mobile phone dataset"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            self.logger.info(f"Loaded {len(dataset)} mobile phones from dataset")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def process_products(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert dataset to product format for AI Agent"""
        products = []
        
        for i, phone in enumerate(dataset):
            try:
                # Generate product ID
                product_id = f"PHONE-{i+1:04d}"
                
                # Create product name
                brand = phone.get("brand_name", "Unknown").title()
                model = phone.get("model", "Unknown")
                product_name = f"{brand} {model}"
                
                # Create description
                description = self._create_product_description(phone)
                
                # Create category based on price
                price = phone.get("price", 0)
                category = self._get_price_category(price)
                
                # Create tags
                tags = self._create_tags(phone)
                
                # Create specifications
                specifications = self._create_specifications(phone)
                
                # Create product object
                product = {
                    "id": product_id,
                    "name": product_name,
                    "description": description,
                    "category": category,
                    "brand": brand,
                    "price": price,
                    "rating": phone.get("rating", 0) / 100,  # Convert to 0-1 scale
                    "review_count": random.randint(10, 500),  # Random review count
                    "tags": tags,
                    "specifications": specifications,
                    "availability": "in_stock",
                    "warranty": "12 months",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                
                products.append(product)
                
            except Exception as e:
                self.logger.warning(f"Error processing phone {i}: {e}")
                continue
        
        self.logger.info(f"Processed {len(products)} products")
        return products
    
    def _create_product_description(self, phone: Dict[str, Any]) -> str:
        """Create product description from phone data"""
        brand = phone.get("brand_name", "Unknown").title()
        model = phone.get("model", "Unknown")
        price = phone.get("price", 0)
        rating = phone.get("rating", 0)
        screen_size = phone.get("screen_size", 0)
        ram = phone.get("ram_capacity", 0)
        storage = phone.get("internal_memory", 0)
        battery = phone.get("battery_capacity", 0)
        os_type = phone.get("os", "Unknown").title()
        
        description = f"""
        {brand} {model} là một chiếc điện thoại thông minh cao cấp với hiệu năng mạnh mẽ và thiết kế hiện đại.
        
        Thông số kỹ thuật nổi bật:
        - Màn hình: {screen_size}" với độ phân giải cao
        - RAM: {ram}GB cho hiệu năng mượt mà
        - Bộ nhớ trong: {storage}GB
        - Pin: {battery}mAh với sạc nhanh
        - Hệ điều hành: {os_type}
        - Đánh giá: {rating}/100 điểm
        
        Giá bán: {price:,} VNĐ
        """
        
        return description.strip()
    
    def _get_price_category(self, price: int) -> str:
        """Categorize product by price"""
        if price < 5000000:  # < 5M VNĐ
            return "budget"
        elif price < 15000000:  # < 15M VNĐ
            return "mid_range"
        elif price < 30000000:  # < 30M VNĐ
            return "premium"
        else:
            return "flagship"
    
    def _create_tags(self, phone: Dict[str, Any]) -> List[str]:
        """Create tags from phone features"""
        tags = []
        
        # Brand tag
        brand = phone.get("brand_name", "").lower()
        if brand:
            tags.append(brand)
        
        # OS tag
        os_type = phone.get("os", "").lower()
        if os_type:
            tags.append(os_type)
        
        # Feature tags
        if phone.get("has_5g") == "TRUE":
            tags.append("5g")
        if phone.get("has_nfc") == "TRUE":
            tags.append("nfc")
        if phone.get("has_ir_blaster") == "TRUE":
            tags.append("ir_blaster")
        if phone.get("fast_charging_available") == 1:
            tags.append("fast_charging")
        
        # Camera tags
        rear_cameras = phone.get("num_rear_cameras", 0)
        if rear_cameras > 1:
            tags.append(f"{rear_cameras}_cameras")
        
        # RAM tag
        ram = phone.get("ram_capacity", 0)
        if ram >= 8:
            tags.append("high_ram")
        
        # Storage tag
        storage = phone.get("internal_memory", 0)
        if storage >= 256:
            tags.append("large_storage")
        
        return tags
    
    def _create_specifications(self, phone: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed specifications"""
        return {
            "display": {
                "screen_size": f"{phone.get('screen_size', 0)}\"",
                "resolution": f"{phone.get('resolution_width', 0)}x{phone.get('resolution_height', 0)}",
                "refresh_rate": f"{phone.get('refresh_rate', 0)}Hz"
            },
            "performance": {
                "processor": phone.get("processor_brand", "Unknown").title(),
                "cores": phone.get("num_cores", 0),
                "speed": f"{phone.get('processor_speed', 0)}GHz",
                "ram": f"{phone.get('ram_capacity', 0)}GB",
                "storage": f"{phone.get('internal_memory', 0)}GB"
            },
            "camera": {
                "rear_cameras": phone.get("num_rear_cameras", 0),
                "front_cameras": phone.get("num_front_cameras", 0),
                "rear_primary": f"{phone.get('primary_camera_rear', 0)}MP",
                "front_primary": f"{phone.get('primary_camera_front', 0)}MP"
            },
            "battery": {
                "capacity": f"{phone.get('battery_capacity', 0)}mAh",
                "fast_charging": f"{phone.get('fast_charging', 0)}W" if phone.get("fast_charging_available") else "No"
            },
            "connectivity": {
                "5g": phone.get("has_5g") == "TRUE",
                "nfc": phone.get("has_nfc") == "TRUE",
                "ir_blaster": phone.get("has_ir_blaster") == "TRUE"
            },
            "os": phone.get("os", "Unknown").title()
        }
    
    def generate_conversations(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate sample conversations for training"""
        conversations = []
        
        # Sample conversation templates
        templates = [
            {
                "intent": "product_search",
                "user": "Tôi muốn mua điện thoại trong tầm giá {price_range}",
                "assistant": "Tôi có thể giúp bạn tìm điện thoại phù hợp với ngân sách {price_range}. Bạn có yêu cầu gì đặc biệt về camera, pin, hay hiệu năng không?"
            },
            {
                "intent": "product_comparison",
                "user": "So sánh {product1} và {product2} cho tôi",
                "assistant": "Tôi sẽ so sánh {product1} và {product2} dựa trên các tiêu chí quan trọng như giá, camera, pin, và hiệu năng."
            },
            {
                "intent": "product_details",
                "user": "Cho tôi biết thông tin chi tiết về {product}",
                "assistant": "Đây là thông tin chi tiết về {product}: {description}"
            },
            {
                "intent": "price_inquiry",
                "user": "Giá của {product} là bao nhiêu?",
                "assistant": "Giá của {product} là {price:,} VNĐ. Bạn có muốn tôi tìm các sản phẩm tương tự với giá tốt hơn không?"
            }
        ]
        
        # Generate conversations
        for i in range(100):  # Generate 100 sample conversations
            template = random.choice(templates)
            product = random.choice(products)
            
            # Replace placeholders
            user_message = template["user"].format(
                price_range="5-10 triệu",
                product1=random.choice(products)["name"],
                product2=random.choice(products)["name"],
                product=product["name"],
                price=product["price"]
            )
            
            assistant_message = template["assistant"].format(
                price_range="5-10 triệu",
                product1=random.choice(products)["name"],
                product2=random.choice(products)["name"],
                product=product["name"],
                price=product["price"],
                description=product["description"][:200] + "..."
            )
            
            conversation = {
                "id": f"conv_{i+1:04d}",
                "user_id": f"user_{random.randint(1, 50):03d}",
                "session_id": f"session_{random.randint(1, 20):03d}",
                "intent": template["intent"],
                "messages": [
                    {
                        "role": "user",
                        "content": user_message,
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "role": "assistant", 
                        "content": assistant_message,
                        "timestamp": datetime.now().isoformat()
                    }
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            conversations.append(conversation)
        
        self.logger.info(f"Generated {len(conversations)} sample conversations")
        return conversations
    
    def generate_knowledge_base(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate knowledge base entries"""
        knowledge_base = []
        
        # Brand information
        brands = {}
        for product in products:
            brand = product["brand"]
            if brand not in brands:
                brands[brand] = []
            brands[brand].append(product)
        
        for brand, brand_products in brands.items():
            entry = {
                "id": f"kb_brand_{brand.lower()}",
                "title": f"Thông tin về thương hiệu {brand}",
                "content": f"""
                {brand} là một thương hiệu điện thoại thông minh nổi tiếng với {len(brand_products)} sản phẩm trong danh mục.
                
                Các sản phẩm phổ biến:
                {chr(10).join([f"- {p['name']}: {p['price']:,} VNĐ" for p in brand_products[:5]])}
                
                Đặc điểm nổi bật của {brand}:
                - Chất lượng cao và độ bền tốt
                - Hiệu năng mạnh mẽ
                - Thiết kế hiện đại
                - Giá cả hợp lý
                """,
                "category": "brand_info",
                "tags": [brand.lower(), "brand", "information"],
                "source": "dataset_analysis"
            }
            knowledge_base.append(entry)
        
        # Category information
        categories = {}
        for product in products:
            category = product["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(product)
        
        for category, cat_products in categories.items():
            category_names = {
                "budget": "Tầm giá phổ thông",
                "mid_range": "Tầm giá trung bình", 
                "premium": "Cao cấp",
                "flagship": "Flagship"
            }
            
            entry = {
                "id": f"kb_category_{category}",
                "title": f"Điện thoại {category_names.get(category, category)}",
                "content": f"""
                Điện thoại {category_names.get(category, category)} là những sản phẩm phù hợp với nhu cầu và ngân sách của đại đa số người dùng.
                
                Đặc điểm chung:
                - Giá cả hợp lý
                - Hiệu năng ổn định
                - Đầy đủ tính năng cần thiết
                - Dễ sử dụng
                
                Sản phẩm tiêu biểu:
                {chr(10).join([f"- {p['name']}: {p['price']:,} VNĐ" for p in cat_products[:5]])}
                """,
                "category": "category_info",
                "tags": [category, "category", "information"],
                "source": "dataset_analysis"
            }
            knowledge_base.append(entry)
        
        self.logger.info(f"Generated {len(knowledge_base)} knowledge base entries")
        return knowledge_base
    
    def save_processed_data(
        self, 
        products: List[Dict[str, Any]], 
        conversations: List[Dict[str, Any]], 
        knowledge_base: List[Dict[str, Any]]
    ) -> None:
        """Save processed data to files"""
        try:
            # Save products
            products_file = os.path.join(self.processed_data_dir, "products.json")
            with open(products_file, 'w', encoding='utf-8') as f:
                json.dump(products, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved {len(products)} products to {products_file}")
            
            # Save conversations
            conversations_file = os.path.join(self.processed_data_dir, "conversations.json")
            with open(conversations_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved {len(conversations)} conversations to {conversations_file}")
            
            # Save knowledge base
            kb_file = os.path.join(self.processed_data_dir, "knowledge_base.json")
            with open(kb_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved {len(knowledge_base)} knowledge base entries to {kb_file}")
            
            # Save training data
            training_data = self._create_training_data(products, conversations)
            training_file = os.path.join(self.processed_data_dir, "training_data.json")
            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved {len(training_data)} training examples to {training_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {e}")
            raise
    
    def _create_training_data(
        self, 
        products: List[Dict[str, Any]], 
        conversations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create training data for fine-tuning"""
        training_data = []
        
        for conversation in conversations:
            # Create instruction-following format
            instruction = f"Bạn là trợ lý bán hàng chuyên nghiệp. Hãy trả lời câu hỏi của khách hàng về sản phẩm điện thoại."
            
            # Get user message
            user_message = conversation["messages"][0]["content"]
            
            # Get assistant response
            assistant_response = conversation["messages"][1]["content"]
            
            training_example = {
                "instruction": instruction,
                "input": user_message,
                "output": assistant_response,
                "intent": conversation["intent"],
                "metadata": {
                    "user_id": conversation["user_id"],
                    "session_id": conversation["session_id"],
                    "timestamp": conversation["timestamp"]
                }
            }
            
            training_data.append(training_example)
        
        return training_data
    
    def process_all(self) -> None:
        """Process the entire dataset"""
        try:
            self.logger.info("Starting dataset processing...")
            
            # Load dataset
            dataset = self.load_dataset()
            
            # Process products
            products = self.process_products(dataset)
            
            # Generate conversations
            conversations = self.generate_conversations(products)
            
            # Generate knowledge base
            knowledge_base = self.generate_knowledge_base(products)
            
            # Save processed data
            self.save_processed_data(products, conversations, knowledge_base)
            
            self.logger.info("Dataset processing completed successfully!")
            
            # Print summary
            print("\n" + "="*60)
            print("DATASET PROCESSING COMPLETE")
            print("="*60)
            print(f"Products processed: {len(products)}")
            print(f"Conversations generated: {len(conversations)}")
            print(f"Knowledge base entries: {len(knowledge_base)}")
            print(f"Training examples: {len(conversations)}")
            print("="*60)
            
        except Exception as e:
            self.logger.error(f"Dataset processing failed: {e}")
            raise

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process mobile phone dataset for AI Agent")
    parser.add_argument("--dataset_path", default="./training/dataset/dataset.json",
                       help="Path to the dataset file")
    parser.add_argument("--output_dir", default="./data/processed",
                       help="Output directory for processed data")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process dataset
    processor = DatasetProcessor(args.dataset_path)
    processor.process_all()

if __name__ == "__main__":
    main()
