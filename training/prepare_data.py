"""
Data Preparation for Training
Enhanced for e-commerce conversation normalization
"""

import json
import logging
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
# import pandas as pd  # Not used in current implementation
from datetime import datetime

logger = logging.getLogger(__name__)

class DataPreparator:
    def __init__(self):
        self.dataset_path = Path("training/dataset")
        self.dataset_path.mkdir(exist_ok=True)
        
        # E-commerce specific patterns - Enhanced
        self.intent_patterns = {
            "product_search": [
                r"tìm.*điện thoại", r"mua.*laptop", r"có.*sản phẩm", 
                r"giá.*bao nhiêu", r"thông số.*kỹ thuật", r"so sánh.*sản phẩm",
                r"camera.*tốt", r"hiệu năng.*như thế nào", r"có.*hàng.*không",
                r"đặc điểm.*của", r"tính năng.*gì", r"cấu hình.*ra sao",
                r"pin.*bao lâu", r"màn hình.*như thế nào", r"âm thanh.*có tốt"
            ],
            "product_recommendation": [
                r"gợi ý.*sản phẩm", r"nên mua.*gì", r"phù hợp.*với", 
                r"tư vấn.*mua", r"lựa chọn.*tốt", r"cái nào.*tốt hơn",
                r"so sánh.*và", r"nên chọn.*cái nào", r"ưu điểm.*của",
                r"khuyến nghị.*gì", r"tốt nhất.*dưới", r"phù hợp.*cho.*tuổi",
                r"cho.*công tác", r"cho.*học tập", r"cho.*gaming"
            ],
            "order_inquiry": [
                r"kiểm tra.*đơn hàng", r"trạng thái.*đơn hàng", r"mã đơn hàng",
                r"hủy.*đơn hàng", r"thay đổi.*đơn hàng", r"đơn hàng.*đã giao",
                r"theo dõi.*đơn hàng", r"cập nhật.*đơn hàng", r"thông tin.*đơn hàng",
                r"đơn hàng.*của tôi", r"giao hàng.*khi nào", r"địa chỉ.*giao hàng"
            ],
            "payment_inquiry": [
                r"thanh toán", r"phương thức.*thanh toán", r"hóa đơn",
                r"hoàn tiền", r"chuyển khoản", r"thẻ.*tín dụng",
                r"ví.*điện tử", r"cod", r"thanh toán.*khi.*nhận",
                r"chưa.*cập nhật", r"giao dịch.*thành công", r"tiền.*đã chuyển"
            ],
            "warranty_inquiry": [
                r"bảo hành", r"bảo dưỡng", r"sửa chữa", r"thay thế",
                r"chính sách.*bảo hành", r"bị lỗi", r"không.*hoạt động",
                r"treo.*thường xuyên", r"chậm", r"nóng.*máy", r"pin.*chai",
                r"màn hình.*bị", r"âm thanh.*lỗi", r"camera.*không.*hoạt động"
            ],
            "general_chat": [
                r"xin chào", r"cảm ơn", r"tạm biệt", r"hỏi.*chung",
                r"giúp.*đỡ", r"hỗ trợ", r"tư vấn", r"thông tin",
                r"cửa hàng.*ở đâu", r"giờ.*mở cửa", r"liên hệ.*như thế nào"
            ]
        }
        
        # E-commerce entities - Enhanced
        self.brand_entities = [
            "iPhone", "Samsung", "Xiaomi", "OnePlus", "Huawei", "Oppo", 
            "Vivo", "Realme", "Nothing", "Motorola", "Apple", "Google",
            "ASUS", "Dell", "HP", "Lenovo", "MSI", "Acer", "MacBook",
            "Galaxy", "Pixel", "Redmi", "Note", "Pro", "Ultra", "Plus"
        ]
        
        self.price_ranges = [
            "dưới 5 triệu", "5-10 triệu", "10-20 triệu", "20-50 triệu", 
            "trên 50 triệu", "giá rẻ", "tầm trung", "cao cấp", "15 triệu",
            "20 triệu", "25 triệu", "30 triệu", "35 triệu", "40 triệu"
        ]
        
        self.product_categories = [
            "điện thoại", "laptop", "máy tính", "tablet", "phụ kiện",
            "tai nghe", "sạc", "ốp lưng", "màn hình", "bàn phím", "chuột"
        ]
        
        self.technical_specs = [
            "RAM", "ROM", "CPU", "GPU", "camera", "pin", "màn hình",
            "âm thanh", "wifi", "bluetooth", "5G", "4G", "LTE"
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
        """Extract e-commerce entities from text - Enhanced"""
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
        
        # Extract product categories
        for category in self.product_categories:
            if category in text_lower:
                entities.append({
                    "type": "product_category",
                    "value": category,
                    "start": text_lower.find(category),
                    "end": text_lower.find(category) + len(category)
                })
        
        # Extract technical specifications
        for spec in self.technical_specs:
            if spec.lower() in text_lower:
                entities.append({
                    "type": "technical_spec",
                    "value": spec,
                    "start": text_lower.find(spec.lower()),
                    "end": text_lower.find(spec.lower()) + len(spec)
                })
        
        # Extract order IDs
        order_patterns = [
            r'#ORD\d+', r'ORD\d+', r'đơn hàng\s*#?\d+', r'mã đơn hàng\s*#?\d+'
        ]
        for pattern in order_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "type": "order_id",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Extract ages
        age_pattern = r'(\d+)\s*tuổi'
        age_matches = re.finditer(age_pattern, text_lower)
        for match in age_matches:
            entities.append({
                "type": "age",
                "value": match.group(1),
                "start": match.start(),
                "end": match.end()
            })
        
        # Extract numbers (prices, quantities, storage)
        number_patterns = [
            r'\d+\s*GB', r'\d+\s*TB', r'\d+\s*MP', r'\d+\s*inch',
            r'\d+\s*kg', r'\d+\s*giờ', r'\d+\s*phút'
        ]
        for pattern in number_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "type": "specification",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Extract standalone numbers
        numbers = re.findall(r'\b\d+\b', text)
        for number in numbers:
            if len(number) >= 2:  # Only numbers with 2+ digits
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
        
        logger.info("Saved %d samples to %s", len(data), file_path)
    
    def load_sample_data(self) -> List[Dict[str, Any]]:
        """Load comprehensive e-commerce conversation data"""
        return [
            # Product Search - Điện thoại
            {
                "user_message": "Xin chào, bạn có thể giúp tôi tìm điện thoại iPhone không?",
                "assistant_response": "Xin chào! Tôi rất vui được giúp bạn tìm iPhone phù hợp. Hiện tại chúng tôi có nhiều model iPhone từ iPhone 12 đến iPhone 15 Pro Max. Bạn có ngân sách bao nhiêu và ưu tiên tính năng gì ạ?",
                "intent": "product_search",
                "language": "vi",
                "context": "Khách hàng tìm kiếm điện thoại iPhone",
                "entities": [{"type": "brand", "value": "iPhone"}]
            },
            {
                "user_message": "Tôi muốn mua iPhone 15 Pro, giá bao nhiêu?",
                "assistant_response": "iPhone 15 Pro 128GB có giá 29.990.000 VNĐ, 256GB là 33.990.000 VNĐ, và 512GB là 39.990.000 VNĐ. Bạn muốn phiên bản nào ạ? Tôi có thể kiểm tra tình trạng còn hàng cho bạn.",
                "intent": "product_search",
                "language": "vi",
                "context": "Khách hàng hỏi giá iPhone 15 Pro",
                "entities": [{"type": "brand", "value": "iPhone 15 Pro"}]
            },
            {
                "user_message": "Samsung Galaxy S24 có camera tốt không?",
                "assistant_response": "Samsung Galaxy S24 có camera rất tốt với hệ thống camera 3 ống kính: camera chính 50MP, camera góc siêu rộng 12MP và camera telephoto 10MP. Nó cũng hỗ trợ quay video 8K và chụp ảnh đêm rất đẹp. Bạn có muốn xem thêm thông số kỹ thuật không?",
                "intent": "product_search",
                "language": "vi",
                "context": "Khách hàng hỏi về camera Samsung Galaxy S24",
                "entities": [{"type": "brand", "value": "Samsung Galaxy S24"}]
            },
            
            # Product Search - Laptop
            {
                "user_message": "Tôi cần laptop cho lập trình, ngân sách khoảng 20 triệu",
                "assistant_response": "Với ngân sách 20 triệu, tôi có thể gợi ý một số laptop phù hợp cho lập trình như MacBook Air M2 (22.990.000 VNĐ), Dell XPS 13 (19.990.000 VNĐ), hoặc ThinkPad X1 Carbon (21.990.000 VNĐ). Bạn có ưu tiên thương hiệu nào và cần cấu hình gì đặc biệt không?",
                "intent": "product_recommendation",
                "language": "vi",
                "context": "Khách hàng tìm laptop cho lập trình",
                "entities": [{"type": "price_range", "value": "20 triệu"}]
            },
            {
                "user_message": "Laptop gaming nào tốt nhất dưới 30 triệu?",
                "assistant_response": "Dưới 30 triệu, tôi gợi ý ASUS ROG Strix G15 (28.990.000 VNĐ) với RTX 4060, MSI Katana GF66 (26.990.000 VNĐ) với RTX 4050, hoặc Lenovo Legion 5 (27.990.000 VNĐ) với RTX 4060. Tất cả đều có hiệu năng gaming tốt. Bạn chơi game gì chủ yếu?",
                "intent": "product_recommendation",
                "language": "vi",
                "context": "Khách hàng tìm laptop gaming",
                "entities": [{"type": "price_range", "value": "dưới 30 triệu"}]
            },
            
            # Product Recommendation
            {
                "user_message": "Tôi đang phân vân giữa iPhone 14 và iPhone 15, nên chọn cái nào?",
                "assistant_response": "Để tư vấn chính xác, tôi cần biết thêm về nhu cầu của bạn. iPhone 15 có camera 48MP, chip A17 Pro mạnh hơn, cổng USB-C, và giá cao hơn khoảng 3-4 triệu. iPhone 14 vẫn rất tốt với chip A15, camera 12MP và giá hợp lý hơn. Bạn ưu tiên camera hay hiệu năng?",
                "intent": "product_recommendation",
                "language": "vi",
                "context": "Khách hàng so sánh iPhone 14 và iPhone 15",
                "entities": [{"type": "brand", "value": "iPhone 14"}, {"type": "brand", "value": "iPhone 15"}]
            },
            {
                "user_message": "Tôi cần laptop nhẹ để đi công tác, có gợi ý gì không?",
                "assistant_response": "Cho công tác, tôi gợi ý MacBook Air M2 (1.24kg), Dell XPS 13 (1.27kg), hoặc ASUS ZenBook 13 (1.1kg). Tất cả đều nhẹ, pin lâu và hiệu năng tốt. Bạn có ngân sách bao nhiêu và cần chạy phần mềm gì đặc biệt không?",
                "intent": "product_recommendation",
                "language": "vi",
                "context": "Khách hàng cần laptop nhẹ cho công tác",
                "entities": []
            },
            
            # Order Inquiry
            {
                "user_message": "Làm thế nào để kiểm tra đơn hàng của tôi?",
                "assistant_response": "Để kiểm tra đơn hàng, bạn cần cung cấp mã đơn hàng. Mã đơn hàng thường được gửi qua email khi bạn đặt hàng. Bạn có thể kiểm tra trong email hoặc app. Bạn có mã đơn hàng không ạ?",
                "intent": "order_inquiry",
                "language": "vi",
                "context": "Khách hàng muốn kiểm tra đơn hàng",
                "entities": []
            },
            {
                "user_message": "Đơn hàng #ORD123456 của tôi đã giao chưa?",
                "assistant_response": "Tôi sẽ kiểm tra đơn hàng #ORD123456 cho bạn. Vui lòng đợi một chút... Đơn hàng của bạn đã được giao thành công vào 14:30 hôm nay. Bạn có nhận được sản phẩm chưa ạ?",
                "intent": "order_inquiry",
                "language": "vi",
                "context": "Khách hàng hỏi trạng thái đơn hàng cụ thể",
                "entities": [{"type": "order_id", "value": "ORD123456"}]
            },
            {
                "user_message": "Tôi muốn hủy đơn hàng #ORD789012",
                "assistant_response": "Tôi hiểu bạn muốn hủy đơn hàng #ORD789012. Để hủy đơn hàng, tôi cần xác nhận thông tin với bạn. Đơn hàng đã được đóng gói chưa? Nếu chưa giao, tôi có thể hủy ngay. Bạn có lý do gì muốn hủy không ạ?",
                "intent": "order_inquiry",
                "language": "vi",
                "context": "Khách hàng muốn hủy đơn hàng",
                "entities": [{"type": "order_id", "value": "ORD789012"}]
            },
            
            # Payment Inquiry
            {
                "user_message": "Tôi có thể thanh toán bằng cách nào?",
                "assistant_response": "Chúng tôi hỗ trợ nhiều phương thức thanh toán: thẻ tín dụng/ghi nợ, chuyển khoản ngân hàng, ví điện tử (MoMo, ZaloPay), và thanh toán khi nhận hàng (COD). Bạn muốn sử dụng phương thức nào ạ?",
                "intent": "payment_inquiry",
                "language": "vi",
                "context": "Khách hàng hỏi về phương thức thanh toán",
                "entities": []
            },
            {
                "user_message": "Tôi đã chuyển khoản nhưng chưa thấy cập nhật, phải làm sao?",
                "assistant_response": "Tôi hiểu bạn đã chuyển khoản nhưng chưa thấy cập nhật. Thông thường giao dịch chuyển khoản cần 5-15 phút để xử lý. Bạn có thể gửi tôi ảnh chụp màn hình giao dịch để tôi kiểm tra và xử lý nhanh hơn không ạ?",
                "intent": "payment_inquiry",
                "language": "vi",
                "context": "Khách hàng gặp vấn đề với thanh toán",
                "entities": []
            },
            
            # Warranty Inquiry
            {
                "user_message": "iPhone của tôi bị lỗi camera, có được bảo hành không?",
                "assistant_response": "Tôi rất tiếc về vấn đề camera của iPhone. Để được bảo hành, tôi cần kiểm tra: iPhone còn trong thời hạn bảo hành không, có bị va đập hay vào nước không. Bạn có thể cho tôi biết ngày mua và tình trạng hiện tại của máy không ạ?",
                "intent": "warranty_inquiry",
                "language": "vi",
                "context": "Khách hàng hỏi về bảo hành iPhone",
                "entities": [{"type": "brand", "value": "iPhone"}]
            },
            {
                "user_message": "Laptop của tôi bị treo thường xuyên, có thể sửa không?",
                "assistant_response": "Laptop bị treo có thể do nhiều nguyên nhân: phần mềm, driver, hoặc phần cứng. Nếu còn bảo hành, chúng tôi sẽ kiểm tra và sửa miễn phí. Bạn có thể mang máy đến cửa hàng hoặc tôi có thể hướng dẫn bạn reset lại hệ thống. Bạn muốn cách nào?",
                "intent": "warranty_inquiry",
                "language": "vi",
                "context": "Khách hàng gặp vấn đề với laptop",
                "entities": []
            },
            
            # General Chat
            {
                "user_message": "Cảm ơn bạn đã giúp đỡ!",
                "assistant_response": "Không có gì ạ! Tôi rất vui được giúp bạn. Nếu bạn có thêm câu hỏi gì về sản phẩm hoặc dịch vụ, đừng ngại liên hệ nhé. Chúc bạn một ngày tốt lành!",
                "intent": "general_chat",
                "language": "vi",
                "context": "Khách hàng cảm ơn",
                "entities": []
            },
            {
                "user_message": "Tạm biệt!",
                "assistant_response": "Tạm biệt! Cảm ơn bạn đã tin tưởng và sử dụng dịch vụ của chúng tôi. Hẹn gặp lại bạn lần sau. Chúc bạn một ngày vui vẻ!",
                "intent": "general_chat",
                "language": "vi",
                "context": "Khách hàng chào tạm biệt",
                "entities": []
            },
            
            # Complex scenarios
            {
                "user_message": "Tôi muốn mua laptop cho con học online, con 10 tuổi, ngân sách 15 triệu, không cần quá mạnh",
                "assistant_response": "Với nhu cầu học online cho bé 10 tuổi, tôi gợi ý laptop nhẹ, dễ sử dụng như ASUS VivoBook S15 (14.990.000 VNĐ) hoặc HP Pavilion 15 (13.990.000 VNĐ). Cả hai đều có màn hình 15.6 inch, webcam tốt cho học online, và đủ mạnh cho các ứng dụng học tập. Bạn có ưu tiên thương hiệu nào không?",
                "intent": "product_recommendation",
                "language": "vi",
                "context": "Khách hàng tìm laptop cho trẻ em học online",
                "entities": [{"type": "price_range", "value": "15 triệu"}, {"type": "age", "value": "10 tuổi"}]
            },
            {
                "user_message": "Tôi cần so sánh iPhone 14 Pro và Samsung Galaxy S23 Ultra, cái nào tốt hơn?",
                "assistant_response": "Đây là hai flagship rất tốt! iPhone 14 Pro có chip A16 mạnh, camera 48MP với ProRAW, và hệ sinh thái iOS ổn định. Galaxy S23 Ultra có camera 200MP, S Pen, màn hình 120Hz, và Android linh hoạt. iPhone phù hợp nếu bạn dùng Mac/iPad, Galaxy tốt hơn nếu bạn thích tùy chỉnh. Bạn ưu tiên camera hay hiệu năng?",
                "intent": "product_recommendation",
                "language": "vi",
                "context": "Khách hàng so sánh hai flagship",
                "entities": [{"type": "brand", "value": "iPhone 14 Pro"}, {"type": "brand", "value": "Samsung Galaxy S23 Ultra"}]
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
            
            logger.info("Loaded %d conversations from dataset", len(conversations))
            return conversations
            
        except Exception as e:
            logger.error("Error loading dataset: %s", e)
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
        
        logger.info("Data split: Train=%d, Val=%d, Test=%d", len(train_data), len(val_data), len(test_data))
        return train_data, val_data, test_data
    
    def generate_synthetic_data(self, base_conversations: List[Dict[str, Any]], multiplier: int = 3) -> List[Dict[str, Any]]:
        """Generate synthetic training data by varying existing conversations"""
        synthetic_data = []
        
        for conv in base_conversations:
            # Add original
            synthetic_data.append(conv)
            
            # Generate variations
            for _ in range(multiplier - 1):
                variation = self.create_variation(conv)
                if variation:
                    synthetic_data.append(variation)
        
        logger.info("Generated %d synthetic conversations from %d base conversations", len(synthetic_data), len(base_conversations))
        return synthetic_data
    
    def create_variation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Create a variation of a conversation - Enhanced"""
        user_msg = conversation["user_message"]
        assistant_resp = conversation["assistant_response"]
        intent = conversation.get("intent", "general_chat")
        
        # Enhanced variations based on intent
        variations = []
        
        if intent == "product_search":
            variations = [
                lambda x: f"Tôi muốn tìm hiểu về {x.lower()}",
                lambda x: f"Bạn có thể cho tôi biết về {x.lower()} không?",
                lambda x: f"Tôi quan tâm đến {x.lower()}, bạn có thông tin gì không?",
                lambda x: f"Tôi đang cân nhắc mua {x.lower()}, tư vấn giúp tôi",
                lambda x: f"Xin chào, tôi cần thông tin về {x.lower()}"
            ]
        elif intent == "product_recommendation":
            variations = [
                lambda x: f"Bạn có thể gợi ý {x.lower()} không?",
                lambda x: f"Tôi cần tư vấn về {x.lower()}",
                lambda x: f"Bạn nghĩ sao về {x.lower()}?",
                lambda x: f"Tôi đang phân vân về {x.lower()}, giúp tôi quyết định",
                lambda x: f"Cho tôi lời khuyên về {x.lower()}"
            ]
        elif intent == "order_inquiry":
            variations = [
                lambda x: f"Tôi cần kiểm tra {x.lower()}",
                lambda x: f"Bạn có thể giúp tôi với {x.lower()} không?",
                lambda x: f"Tôi có vấn đề với {x.lower()}",
                lambda x: f"Xin hỗ trợ về {x.lower()}",
                lambda x: f"Tôi muốn biết về {x.lower()}"
            ]
        elif intent == "payment_inquiry":
            variations = [
                lambda x: f"Tôi cần hỗ trợ về {x.lower()}",
                lambda x: f"Bạn có thể giúp tôi với {x.lower()} không?",
                lambda x: f"Tôi gặp vấn đề với {x.lower()}",
                lambda x: f"Xin tư vấn về {x.lower()}",
                lambda x: f"Tôi cần thông tin về {x.lower()}"
            ]
        elif intent == "warranty_inquiry":
            variations = [
                lambda x: f"Tôi cần hỗ trợ về {x.lower()}",
                lambda x: f"Bạn có thể giúp tôi với {x.lower()} không?",
                lambda x: f"Tôi gặp vấn đề với {x.lower()}",
                lambda x: f"Xin tư vấn về {x.lower()}",
                lambda x: f"Tôi cần thông tin về {x.lower()}"
            ]
        else:
            # General variations
            variations = [
                lambda x: f"Xin chào, {x.lower()}",
                lambda x: f"Bạn có thể giúp tôi {x.lower()} không?",
                lambda x: f"Tôi cần {x.lower()} gấp",
                lambda x: f"Tôi đang tìm hiểu về {x.lower()}",
                lambda x: f"Cho tôi biết về {x.lower()}"
            ]
        
        import random
        variation_func = random.choice(variations)
        new_user_msg = variation_func(user_msg)
        
        # Add some randomness to assistant response
        response_variations = [
            assistant_resp,
            assistant_resp.replace("Tôi", "Mình"),
            assistant_resp.replace("bạn", "anh/chị"),
            assistant_resp.replace("ạ", ""),
            assistant_resp + " Nếu bạn cần thêm thông tin gì, đừng ngại hỏi nhé!"
        ]
        
        new_assistant_resp = random.choice(response_variations)
        
        return {
            "user_message": new_user_msg,
            "assistant_response": new_assistant_resp,
            "intent": intent,
            "language": conversation.get("language", "vi"),
            "context": conversation.get("context", ""),
            "entities": conversation.get("entities", []),
            "metadata": {
                "synthetic": True,
                "base_conversation": conversation.get("metadata", {}),
                "variation_type": "enhanced_text_variation",
                "original_intent": intent
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
    logger.info("Total conversations: %d", stats['total_conversations'])
    logger.info("Training samples: %d", stats['train_samples'])
    logger.info("Validation samples: %d", stats['val_samples'])
    logger.info("Test samples: %d", stats['test_samples'])
    logger.info("Intent distribution: %s", stats['intent_distribution'])

if __name__ == "__main__":
    main()
