"""
Data Preparation for Training
Enhanced for e-commerce conversation normalization
"""

import json
import logging
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class DataPreparator:
    def __init__(self):
        self.dataset_path = Path("training/dataset")
        self.dataset_path.mkdir(exist_ok=True)
        
        # E-commerce specific patterns
        self.intent_patterns = {
            "product_search": [
                r"tìm.*điện thoại", r"mua.*laptop", r"có.*sản phẩm", 
                r"giá.*bao nhiêu", r"thông số.*kỹ thuật", r"so sánh.*sản phẩm"
            ],
            "product_recommendation": [
                r"gợi ý.*sản phẩm", r"nên mua.*gì", r"phù hợp.*với", 
                r"tư vấn.*mua", r"lựa chọn.*tốt"
            ],
            "order_inquiry": [
                r"kiểm tra.*đơn hàng", r"trạng thái.*đơn hàng", r"mã đơn hàng",
                r"hủy.*đơn hàng", r"thay đổi.*đơn hàng"
            ],
            "payment_inquiry": [
                r"thanh toán", r"phương thức.*thanh toán", r"hóa đơn",
                r"hoàn tiền", r"chuyển khoản"
            ],
            "warranty_inquiry": [
                r"bảo hành", r"bảo dưỡng", r"sửa chữa", r"thay thế",
                r"chính sách.*bảo hành"
            ],
            "general_chat": [
                r"xin chào", r"cảm ơn", r"tạm biệt", r"hỏi.*chung"
            ]
        }
        
        # E-commerce entities
        self.brand_entities = [
            "iPhone", "Samsung", "Xiaomi", "OnePlus", "Huawei", "Oppo", 
            "Vivo", "Realme", "Nothing", "Motorola", "Apple", "Google"
        ]
        
        self.price_ranges = [
            "dưới 5 triệu", "5-10 triệu", "10-20 triệu", "20-50 triệu", 
            "trên 50 triệu", "giá rẻ", "tầm trung", "cao cấp"
        ]
    
    def prepare_conversation_data(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Prepare conversation data for training with e-commerce normalization"""
        prepared_data = []
        
        for conv in conversations:
            if "user_message" in conv and "assistant_response" in conv:
                # Normalize the conversation
                normalized_conv = self.normalize_conversation(conv)
                
                prepared_data.append({
                    "instruction": normalized_conv["user_message"],
                    "output": normalized_conv["assistant_response"],
                    "input": normalized_conv.get("context", ""),
                    "intent": normalized_conv.get("intent", "general_chat"),
                    "entities": normalized_conv.get("entities", []),
                    "metadata": normalized_conv.get("metadata", {})
                })
        
        return prepared_data
    
    def normalize_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize e-commerce conversation data"""
        user_message = conversation["user_message"]
        assistant_response = conversation["assistant_response"]
        
        # Detect intent
        intent = self.detect_intent(user_message)
        
        # Extract entities
        entities = self.extract_entities(user_message)
        
        # Normalize text
        normalized_user = self.normalize_text(user_message)
        normalized_response = self.normalize_text(assistant_response)
        
        # Create metadata
        metadata = {
            "original_intent": conversation.get("intent", intent),
            "confidence": conversation.get("confidence", 0.8),
            "language": conversation.get("language", "vi"),
            "timestamp": conversation.get("timestamp", datetime.now().isoformat()),
            "session_id": conversation.get("session_id", ""),
            "user_id": conversation.get("user_id", "")
        }
        
        return {
            "user_message": normalized_user,
            "assistant_response": normalized_response,
            "intent": intent,
            "entities": entities,
            "context": conversation.get("context", ""),
            "metadata": metadata
        }
    
    def detect_intent(self, text: str) -> str:
        """Detect intent from text using pattern matching"""
        text_lower = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent
        
        return "general_chat"
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract e-commerce entities from text"""
        entities = []
        text_lower = text.lower()
        
        # Extract brands
        for brand in self.brand_entities:
            if brand.lower() in text_lower:
                entities.append({
                    "type": "brand",
                    "value": brand,
                    "start": text_lower.find(brand.lower()),
                    "end": text_lower.find(brand.lower()) + len(brand)
                })
        
        # Extract price ranges
        for price_range in self.price_ranges:
            if price_range in text_lower:
                entities.append({
                    "type": "price_range",
                    "value": price_range,
                    "start": text_lower.find(price_range),
                    "end": text_lower.find(price_range) + len(price_range)
                })
        
        # Extract numbers (prices, quantities)
        numbers = re.findall(r'\d+', text)
        for i, number in enumerate(numbers):
            entities.append({
                "type": "number",
                "value": number,
                "start": text.find(number),
                "end": text.find(number) + len(number)
            })
        
        return entities
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for training"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize common e-commerce terms
        replacements = {
            r'\bđt\b': 'điện thoại',
            r'\blaptop\b': 'máy tính xách tay',
            r'\bpc\b': 'máy tính để bàn',
            r'\bram\b': 'RAM',
            r'\brom\b': 'ROM',
            r'\bcpu\b': 'CPU',
            r'\bgpu\b': 'GPU'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def save_training_data(self, data: List[Dict[str, str]], filename: str):
        """Save training data to file"""
        file_path = self.dataset_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data)} samples to {file_path}")
    
    def load_sample_data(self) -> List[Dict[str, Any]]:
        """Load sample conversation data"""
        return [
            {
                "user_message": "Xin chào, bạn có thể giúp tôi tìm laptop không?",
                "assistant_response": "Xin chào! Tôi rất vui được giúp bạn tìm laptop phù hợp. Bạn có ngân sách bao nhiêu và cần laptop để làm gì ạ?",
                "intent": "product_search",
                "language": "vi"
            },
            {
                "user_message": "Tôi muốn mua laptop cho lập trình, ngân sách khoảng 20 triệu",
                "assistant_response": "Với ngân sách 20 triệu, tôi có thể gợi ý một số laptop phù hợp cho lập trình như MacBook Air M2, Dell XPS 13, hoặc ThinkPad X1 Carbon. Bạn có ưu tiên thương hiệu nào không?",
                "intent": "product_recommendation",
                "language": "vi"
            },
            {
                "user_message": "Làm thế nào để kiểm tra đơn hàng?",
                "assistant_response": "Để kiểm tra đơn hàng, bạn cần cung cấp mã đơn hàng. Mã đơn hàng thường được gửi qua email khi bạn đặt hàng. Bạn có mã đơn hàng không ạ?",
                "intent": "order_inquiry",
                "language": "vi"
            }
        ]
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load conversation data from dataset file"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert dataset format to conversation format
            conversations = []
            for item in data:
                if isinstance(item, dict):
                    # Check if it's already in conversation format
                    if "user_message" in item and "assistant_response" in item:
                        conversations.append(item)
                    elif "question" in item and "answer" in item:
                        conversations.append({
                            "user_message": item["question"],
                            "assistant_response": item["answer"],
                            "intent": item.get("intent", "general_chat"),
                            "language": item.get("language", "vi")
                        })
                    elif "input" in item and "output" in item:
                        conversations.append({
                            "user_message": item["input"],
                            "assistant_response": item["output"],
                            "intent": item.get("intent", "general_chat"),
                            "language": item.get("language", "vi")
                        })
            
            logger.info(f"Loaded {len(conversations)} conversations from dataset")
            return conversations
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []
    
    def create_training_splits(self, data: List[Dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into train/validation/test sets"""
        import random
        random.shuffle(data)
        
        total_size = len(data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        return train_data, val_data, test_data
    
    def generate_synthetic_data(self, base_conversations: List[Dict[str, Any]], multiplier: int = 3) -> List[Dict[str, Any]]:
        """Generate synthetic training data by varying existing conversations"""
        synthetic_data = []
        
        for conv in base_conversations:
            # Add original
            synthetic_data.append(conv)
            
            # Generate variations
            for i in range(multiplier - 1):
                variation = self.create_variation(conv)
                if variation:
                    synthetic_data.append(variation)
        
        logger.info(f"Generated {len(synthetic_data)} synthetic conversations from {len(base_conversations)} base conversations")
        return synthetic_data
    
    def create_variation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Create a variation of a conversation"""
        user_msg = conversation["user_message"]
        assistant_resp = conversation["assistant_response"]
        
        # Simple variations
        variations = [
            # Add greetings
            lambda x: f"Xin chào, {x.lower()}",
            # Add politeness
            lambda x: f"Bạn có thể giúp tôi {x.lower()} không?",
            # Add urgency
            lambda x: f"Tôi cần {x.lower()} gấp",
            # Add context
            lambda x: f"Tôi đang tìm hiểu về {x.lower()}"
        ]
        
        import random
        variation_func = random.choice(variations)
        new_user_msg = variation_func(user_msg)
        
        return {
            "user_message": new_user_msg,
            "assistant_response": assistant_resp,
            "intent": conversation.get("intent", "general_chat"),
            "language": conversation.get("language", "vi"),
            "metadata": {
                "synthetic": True,
                "base_conversation": conversation.get("metadata", {}),
                "variation_type": "text_variation"
            }
        }

def main():
    preparator = DataPreparator()
    
    # Try to load existing dataset first
    dataset_path = "training/dataset/dataset.json"
    conversations = []
    
    if Path(dataset_path).exists():
        logger.info(f"Loading existing dataset from {dataset_path}")
        conversations = preparator.load_dataset(dataset_path)
    else:
        logger.info("Loading sample data")
        conversations = preparator.load_sample_data()
    
    # Generate synthetic data if we have few conversations
    if len(conversations) < 100:
        logger.info("Generating synthetic data to augment training set")
        conversations = preparator.generate_synthetic_data(conversations, multiplier=5)
    
    # Prepare training data
    training_data = preparator.prepare_conversation_data(conversations)
    
    # Create train/val/test splits
    train_data, val_data, test_data = preparator.create_training_splits(training_data)
    
    # Save training data
    preparator.save_training_data(train_data, "train_conversations.json")
    preparator.save_training_data(val_data, "val_conversations.json")
    preparator.save_training_data(test_data, "test_conversations.json")
    preparator.save_training_data(training_data, "all_conversations.json")
    
    # Save statistics
    stats = {
        "total_conversations": len(conversations),
        "total_training_samples": len(training_data),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "intent_distribution": {},
        "entity_types": set(),
        "timestamp": datetime.now().isoformat()
    }
    
    # Calculate intent distribution
    for item in training_data:
        intent = item.get("intent", "general_chat")
        stats["intent_distribution"][intent] = stats["intent_distribution"].get(intent, 0) + 1
        
        # Collect entity types
        for entity in item.get("entities", []):
            stats["entity_types"].add(entity.get("type", "unknown"))
    
    stats["entity_types"] = list(stats["entity_types"])
    
    preparator.save_training_data([stats], "training_stats.json")
    
    logger.info("Data preparation completed")
    logger.info(f"Total conversations: {stats['total_conversations']}")
    logger.info(f"Training samples: {stats['train_samples']}")
    logger.info(f"Validation samples: {stats['val_samples']}")
    logger.info(f"Test samples: {stats['test_samples']}")
    logger.info(f"Intent distribution: {stats['intent_distribution']}")

if __name__ == "__main__":
    main()
