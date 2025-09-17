# """
# RAG Model - Retrieval-Augmented Generation for product search
# Handles vector search and response generation
# """

# import asyncio
# import logging
# from typing import List, Dict, Any, Optional, Tuple
# import numpy as np
# from sentence_transformers import SentenceTransformer

# logger = logging.getLogger(__name__)

# class RAGModel:
#     """
#     RAG Model for product search and knowledge retrieval
    
#     Features:
#     - Vector similarity search using Pinecone
#     - Product filtering and ranking
#     - Context-aware response generation
#     - User personalization integration
#     """
    
#     def __init__(
#         self,
#         pinecone_client,
#         model_loader,
#         embedding_model_name: str = "intfloat/multilingual-e5-base"
#     ):
#         self.pinecone_client = pinecone_client
#         self.model_loader = model_loader
#         self.embedding_model_name = embedding_model_name
#         self.embedding_model = None
        
#     async def initialize(self):
#         """Initialize RAG model components"""
#         try:
#             logger.info("Initializing RAG model...")
            
#             # Initialize embedding model
#             await self._initialize_embedding_model()
            
#             logger.info("RAG model initialized successfully")
            
#         except Exception as e:
#             logger.error(f"Failed to initialize RAG model: {e}")
#             raise
    
#     async def _initialize_embedding_model(self):
#         """Initialize sentence transformer for embeddings"""
#         try:
#             logger.info(f"Loading embedding model: {self.embedding_model_name}")
            
#             # Load sentence transformer model
#             self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
#             logger.info("Embedding model loaded successfully")
            
#         except Exception as e:
#             logger.error(f"Failed to load embedding model: {e}")
#             raise
    
#     async def search_products(
#         self,
#         query: str,
#         user_id: Optional[str] = None,
#         top_k: int = 5,
#         price_range: Optional[Tuple[float, float]] = None,
#         brand: Optional[str] = None,
#         category: Optional[str] = None
#     ) -> List[Dict[str, Any]]:
#         """
#         Search for products using RAG
        
#         Args:
#             query: Search query
#             user_id: User ID for personalization
#             top_k: Number of results to return
#             price_range: Optional price range filter
#             brand: Optional brand filter
#             category: Optional category filter
        
#         Returns:
#             List of product search results
#         """
#         try:
#             logger.info(f"Searching products for query: {query}")
            
#             # Generate query embedding
#             query_embedding = await self._generate_embedding(query)
            
#             # Search in Pinecone
#             search_results = await self.pinecone_client.search_products(
#                 query_vector=query_embedding,
#                 top_k=top_k,
#                 price_range=price_range,
#                 brand=brand,
#                 category=category
#             )
            
#             # Process and format results
#             products = await self._process_search_results(search_results, user_id)
            
#             logger.info(f"Found {len(products)} products")
#             return products
            
#         except Exception as e:
#             logger.error(f"Failed to search products: {e}")
#             raise
    
#     async def _generate_embedding(self, text: str) -> List[float]:
#         """Generate embedding for text"""
#         try:
#             if not self.embedding_model:
#                 raise ValueError("Embedding model not initialized")
            
#             # Generate embedding
#             embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            
#             # Convert to list if needed
#             if isinstance(embedding, np.ndarray):
#                 embedding = embedding.tolist()
            
#             return embedding
            
#         except Exception as e:
#             logger.error(f"Failed to generate embedding: {e}")
#             raise
    
#     async def _process_search_results(
#         self,
#         search_results: List[Dict[str, Any]],
#         user_id: Optional[str] = None
#     ) -> List[Dict[str, Any]]:
#         """Process and format search results"""
#         try:
#             products = []
            
#             for result in search_results:
#                 product_info = result.get("product_info", {})
                
#                 # Format product data
#                 product = {
#                     "id": result["id"],
#                     "name": product_info.get("name", "Unknown Product"),
#                     "brand": product_info.get("brand", "Unknown Brand"),
#                     "price": product_info.get("price", 0),
#                     "description": product_info.get("description", ""),
#                     "category": product_info.get("category", "Unknown"),
#                     "image_url": product_info.get("image_url", ""),
#                     "rating": product_info.get("rating", 0),
#                     "reviews_count": product_info.get("reviews_count", 0),
#                     "availability": product_info.get("availability", "In Stock"),
#                     "specifications": product_info.get("specifications", {}),
#                     "similarity_score": result["score"],
#                     "relevance_score": await self._calculate_relevance_score(
#                         product_info, user_id
#                     )
#                 }
                
#                 products.append(product)
            
#             # Sort by relevance score
#             products.sort(key=lambda x: x["relevance_score"], reverse=True)
            
#             return products
            
#         except Exception as e:
#             logger.error(f"Failed to process search results: {e}")
#             raise
    
#     async def _calculate_relevance_score(
#         self,
#         product_info: Dict[str, Any],
#         user_id: Optional[str] = None
#     ) -> float:
#         """Calculate relevance score for product"""
#         try:
#             # Base score from similarity
#             base_score = 0.5
            
#             # Boost for high ratings
#             rating = product_info.get("rating", 0)
#             if rating >= 4.5:
#                 base_score += 0.2
#             elif rating >= 4.0:
#                 base_score += 0.1
            
#             # Boost for popular products (high review count)
#             reviews_count = product_info.get("reviews_count", 0)
#             if reviews_count >= 1000:
#                 base_score += 0.1
#             elif reviews_count >= 100:
#                 base_score += 0.05
            
#             # TODO: Add user personalization scoring
#             if user_id:
#                 # This would integrate with personalization model
#                 pass
            
#             return min(base_score, 1.0)
            
#         except Exception as e:
#             logger.error(f"Failed to calculate relevance score: {e}")
#             return 0.5
    
#     async def generate_product_summary(
#         self,
#         products: List[Dict[str, Any]],
#         query: str
#     ) -> str:
#         """Generate a summary of search results"""
#         try:
#             if not products:
#                 return "Không tìm thấy sản phẩm phù hợp với yêu cầu của bạn."
            
#             # Create product summary
#             summary_parts = []
            
#             # Add query context
#             summary_parts.append(f"Dựa trên yêu cầu '{query}', tôi tìm thấy {len(products)} sản phẩm phù hợp:")
            
#             # Add top products
#             for i, product in enumerate(products[:3], 1):
#                 name = product["name"]
#                 brand = product["brand"]
#                 price = product["price"]
#                 rating = product["rating"]
                
#                 summary_parts.append(
#                     f"{i}. {name} ({brand}) - {price:,} VNĐ - ⭐ {rating}/5"
#                 )
            
#             if len(products) > 3:
#                 summary_parts.append(f"... và {len(products) - 3} sản phẩm khác")
            
#             return "\n".join(summary_parts)
            
#         except Exception as e:
#             logger.error(f"Failed to generate product summary: {e}")
#             return "Có lỗi khi tạo tóm tắt sản phẩm."
    
#     async def upsert_product(
#         self,
#         product_id: str,
#         product_data: Dict[str, Any],
#         namespace: str = "default"
#     ) -> bool:
#         """
#         Upsert product to vector database
        
#         Args:
#             product_id: Unique product identifier
#             product_data: Product information
#             namespace: Pinecone namespace
        
#         Returns:
#             Success status
#         """
#         try:
#             logger.info(f"Upserting product: {product_id}")
            
#             # Create product text for embedding
#             product_text = self._create_product_text(product_data)
            
#             # Generate embedding
#             embedding = await self._generate_embedding(product_text)
            
#             # Prepare vector data
#             vector_data = {
#                 "id": product_id,
#                 "values": embedding,
#                 "metadata": {
#                     "name": product_data.get("name", ""),
#                     "brand": product_data.get("brand", ""),
#                     "price": product_data.get("price", 0),
#                     "description": product_data.get("description", ""),
#                     "category": product_data.get("category", ""),
#                     "image_url": product_data.get("image_url", ""),
#                     "rating": product_data.get("rating", 0),
#                     "reviews_count": product_data.get("reviews_count", 0),
#                     "availability": product_data.get("availability", "In Stock"),
#                     "specifications": product_data.get("specifications", {}),
#                     "product_text": product_text
#                 }
#             }
            
#             # Upsert to Pinecone
#             await self.pinecone_client.upsert_vectors(
#                 vectors=[vector_data],
#                 namespace=namespace
#             )
            
#             logger.info(f"Successfully upserted product: {product_id}")
#             return True
            
#         except Exception as e:
#             logger.error(f"Failed to upsert product: {e}")
#             return False
    
#     def _create_product_text(self, product_data: Dict[str, Any]) -> str:
#         """Create text representation of product for embedding"""
#         try:
#             text_parts = []
            
#             # Add basic info
#             if product_data.get("name"):
#                 text_parts.append(product_data["name"])
            
#             if product_data.get("brand"):
#                 text_parts.append(f"thương hiệu {product_data['brand']}")
            
#             if product_data.get("description"):
#                 text_parts.append(product_data["description"])
            
#             # Add specifications
#             specs = product_data.get("specifications", {})
#             if specs:
#                 spec_text = []
#                 for key, value in specs.items():
#                     spec_text.append(f"{key}: {value}")
#                 text_parts.append(" ".join(spec_text))
            
#             # Add category
#             if product_data.get("category"):
#                 text_parts.append(f"danh mục {product_data['category']}")
            
#             return " ".join(text_parts)
            
#         except Exception as e:
#             logger.error(f"Failed to create product text: {e}")
#             return ""
    
#     async def cleanup(self):
#         """Cleanup resources"""
#         try:
#             logger.info("Cleaning up RAG model...")
#             # No explicit cleanup needed for sentence transformer
#             logger.info("RAG model cleanup completed")
            
#         except Exception as e:
#             logger.error(f"Error during RAG model cleanup: {e}")
"""
RAG Model - Retrieval-Augmented Generation for product search
Handles vector search and response generation
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class RAGModel:
    """
    RAG Model for product search and knowledge retrieval
    
    Features:
    - Vector similarity search using Pinecone
    - Product filtering and ranking
    - Context-aware response generation
    - User personalization integration
    """

    def __init__(self, pinecone_client, model_loader):
        self.pinecone_client = pinecone_client
        self.model_loader = model_loader
        self.dimension = 1024   # fix theo index product-search
        self.embedding_model_name = "llama-text-embed-v2"

    async def initialize(self):
        """Initialize RAG model components"""
        try:
            logger.info("Initializing RAG model with Pinecone Cloud embeddings...")
            logger.info(f"Embedding model: {self.embedding_model_name}")
            logger.info("RAG model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG model: {e}")
            raise
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding via Pinecone managed model"""
        try:
            pc = self.pinecone_client.pc  # đã khởi tạo từ adapters/pinecone_client
            response = pc.inference.embed(
                model=self.embedding_model_name,
                inputs=[text],
                parameters={"input_type": "passage"}
            )
            return response[0].values  # 1 vector (1024-dim)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def search_products(
        self, 
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        price_range: Optional[Tuple[float, float]] = None,
        brand: Optional[str] = None,
        category: Optional[str] = None,
        specs: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for products using RAG"""
        try:
            logger.info(f"Searching products for query: {query}")

            # Extract metadata from query
            extracted_metadata = await self._extract_metadata_from_query(query)
            
            # Merge with provided metadata
            final_price_range = price_range or extracted_metadata.get("price_range")
            final_brand = brand or extracted_metadata.get("brand")
            final_category = category or extracted_metadata.get("category")
            final_specs = specs or extracted_metadata.get("specs", {})

            # Generate query embedding
            query_embedding = await self._generate_embedding(query)

            # Search in Pinecone
            search_results = await self.pinecone_client.search_products(
                query_vector=query_embedding,
                top_k=top_k,
                price_range=final_price_range,
                brand=final_brand,
                category=final_category
            )

            # Process and format results
            products = await self._process_search_results(search_results, user_id)
            
            # Apply additional filtering based on extracted specs
            if final_specs:
                products = await self._filter_by_specs(products, final_specs)
            
            # If no products found and we have strict filters, try relaxed search
            if not products and (final_price_range or final_brand or final_specs):
                logger.info("No products found with strict filters, trying relaxed search...")
                relaxed_results = await self.pinecone_client.search_products(
                    query_vector=query_embedding,
                    top_k=top_k * 2,  # Get more results
                    price_range=None,  # Remove price filter
                    brand=None,  # Remove brand filter
                    category=None
                )
                products = await self._process_search_results(relaxed_results, user_id)
                # Apply only essential filters
                if final_specs and any(spec in final_specs for spec in ['pin', 'camera', 'chơi game']):
                    products = await self._filter_by_specs(products, final_specs)
            
            logger.info(f"Found {len(products)} products")
            return products
            
        except Exception as e:
            logger.error(f"Failed to search products: {e}")
            raise
    
    async def _process_search_results(
        self, 
        search_results: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Process and format search results"""
        try:
            products = []
            for result in search_results:
                product_info = result.get("product_info", {})

                product = {
                    "id": result["id"],
                    "name": product_info.get("name", "Unknown Product"),
                    "brand": product_info.get("brand", "Unknown Brand"),
                    "price": product_info.get("price", 0),
                    "description": product_info.get("description", ""),
                    "category": product_info.get("category", "Unknown"),
                    "image_url": product_info.get("image_url", ""),
                    "rating": product_info.get("rating", 0),
                    "reviews_count": product_info.get("reviews_count", 0),
                    "availability": product_info.get("availability", "In Stock"),
                    "specifications": product_info.get("specifications", {}),
                    "similarity_score": result["score"],
                    "relevance_score": await self._calculate_relevance_score(
                        product_info, user_id
                    )
                }
                products.append(product)

            # Sort by relevance score
            products.sort(key=lambda x: x["relevance_score"], reverse=True)
            return products

        except Exception as e:
            logger.error(f"Failed to process search results: {e}")
            raise

    async def _calculate_relevance_score(
        self,
        product_info: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> float:
        """Calculate relevance score for product"""
        try:
            base_score = 0.5
            rating = product_info.get("rating", 0)
            if rating >= 4.5:
                base_score += 0.2
            elif rating >= 4.0:
                base_score += 0.1

            reviews_count = product_info.get("reviews_count", 0)
            if reviews_count >= 1000:
                base_score += 0.1
            elif reviews_count >= 100:
                base_score += 0.05

            # TODO: add personalization here
            return min(base_score, 1.0)
        except Exception:
            return 0.5

    async def generate_product_summary(
        self,
        products: List[Dict[str, Any]],
        query: str
    ) -> str:
        """Generate a summary of search results"""
        try:
            if not products:
                return "Không tìm thấy sản phẩm phù hợp với yêu cầu của bạn."

            parts = [f"Dựa trên yêu cầu '{query}', tôi tìm thấy {len(products)} sản phẩm phù hợp:"]
            for i, product in enumerate(products[:3], 1):
                parts.append(
                    f"{i}. {product['name']} ({product['brand']}) - {product['price']:,} VNĐ - ⭐ {product['rating']}/5"
                )
            if len(products) > 3:
                parts.append(f"... và {len(products) - 3} sản phẩm khác")
            return "\n".join(parts)
        except Exception as e:
            logger.error(f"Failed to generate product summary: {e}")
            return "Có lỗi khi tạo tóm tắt sản phẩm."

    async def upsert_product(
        self,
        product_id: str,
        product_data: Dict[str, Any],
        namespace: str = "default"
    ) -> bool:
        """Upsert product to vector database"""
        try:
            logger.info(f"Upserting product: {product_id}")

            product_text = self._create_product_text(product_data)
            embedding = await self._generate_embedding(product_text)

            vector_data = {
                "id": product_id,
                "values": embedding,
                "metadata": {
                    "name": product_data.get("name", ""),
                    "brand": product_data.get("brand", ""),
                    "price": product_data.get("price", 0),
                    "description": product_data.get("description", ""),
                    "category": product_data.get("category", ""),
                    "image_url": product_data.get("image_url", ""),
                    "rating": product_data.get("rating", 0),
                    "reviews_count": product_data.get("reviews_count", 0),
                    "availability": product_data.get("availability", "In Stock"),
                    # ✅ Fix: serialize specifications dict
                    "specifications": "; ".join(
                        [f"{k}: {v}" for k, v in product_data.get("specifications", {}).items()]
                    ),
                    "product_text": product_text,
                },
            }

            await self.pinecone_client.upsert_vectors([vector_data], namespace=namespace)
            logger.info(f"Successfully upserted product: {product_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert product: {e}")
            return False

    def _create_product_text(self, product_data: Dict[str, Any]) -> str:
        """Create text representation of product for embedding"""
        try:
            parts = []
            if product_data.get("name"):
                parts.append(product_data["name"])
            if product_data.get("brand"):
                parts.append(f"thương hiệu {product_data['brand']}")
            if product_data.get("description"):
                parts.append(product_data["description"])
            specs = product_data.get("specifications", {})
            if specs:
                parts.append(" ".join([f"{k}: {v}" for k, v in specs.items()]))
            if product_data.get("category"):
                parts.append(f"danh mục {product_data['category']}")
            return " ".join(parts)
        except Exception:
            return ""

    async def _extract_metadata_from_query(self, query: str) -> Dict[str, Any]:
        """Extract metadata from user query"""
        try:
            metadata = {
                "price_range": None,
                "brand": None,
                "category": None,
                "specs": {}
            }
            
            query_lower = query.lower()
            
            # Extract price range
            import re
            
            # Price patterns
            price_patterns = [
                r'dưới\s+(\d+)\s*tr(?:iệu)?',
                r'trên\s+(\d+)\s*tr(?:iệu)?',
                r'khoảng\s+(\d+)\s*tr(?:iệu)?',
                r'từ\s+(\d+)\s*đến\s+(\d+)\s*tr(?:iệu)?',
                r'(\d+)\s*tr(?:iệu)?\s*trở\s+xuống',
                r'(\d+)\s*tr(?:iệu)?\s*trở\s+lên'
            ]
            
            for pattern in price_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    if 'từ' in pattern and 'đến' in pattern:
                        # Range pattern
                        min_price = int(match.group(1)) * 1000000
                        max_price = int(match.group(2)) * 1000000
                        metadata["price_range"] = (min_price, max_price)
                    elif 'dưới' in pattern or 'trở xuống' in pattern:
                        # Max price
                        max_price = int(match.group(1)) * 1000000
                        metadata["price_range"] = (0, max_price)
                    elif 'trên' in pattern or 'trở lên' in pattern:
                        # Min price
                        min_price = int(match.group(1)) * 1000000
                        metadata["price_range"] = (min_price, float('inf'))
                    elif 'khoảng' in pattern:
                        # Approximate price
                        price = int(match.group(1)) * 1000000
                        tolerance = price * 0.2  # 20% tolerance
                        metadata["price_range"] = (price - tolerance, price + tolerance)
                    break
            
            # Extract brand
            brands = [
                'iphone', 'apple', 'samsung', 'xiaomi', 'oppo', 'vivo', 
                'realme', 'oneplus', 'huawei', 'nokia', 'motorola', 'lg', 'sony'
            ]
            
            for brand in brands:
                if brand in query_lower:
                    metadata["brand"] = brand.title()
                    break
            
            # Extract specs
            specs_patterns = {
                'pin': r'pin\s+(khỏe|tốt|lâu|dài)',
                'camera': r'camera\s+(tốt|đẹp|chụp\s+ảnh)',
                'ram': r'(\d+)\s*gb\s*ram',
                'rom': r'(\d+)\s*gb\s*rom',
                'màn hình': r'màn\s+hình\s+(\d+\.?\d*)\s*inch',
                'chơi game': r'chơi\s+game',
                'chụp ảnh': r'chụp\s+ảnh'
            }
            
            for spec, pattern in specs_patterns.items():
                match = re.search(pattern, query_lower)
                if match:
                    if spec in ['ram', 'rom', 'màn hình']:
                        metadata["specs"][spec] = match.group(1)
                    else:
                        metadata["specs"][spec] = True
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from query: {e}")
            return {}
    
    async def _filter_by_specs(self, products: List[Dict[str, Any]], specs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter products by extracted specifications"""
        try:
            filtered_products = []
            
            for product in products:
                product_specs = product.get("specifications", {})
                matches = True
                
                for spec_key, spec_value in specs.items():
                    if spec_key == 'pin' and spec_value:
                        # Check for battery-related keywords
                        battery_text = str(product_specs.get('pin', '')).lower()
                        if not any(keyword in battery_text for keyword in ['khỏe', 'tốt', 'lâu', 'dài', 'mAh']):
                            matches = False
                            break
                    
                    elif spec_key == 'camera' and spec_value:
                        # Check for camera quality
                        camera_text = str(product_specs.get('camera', '')).lower()
                        if not any(keyword in camera_text for keyword in ['mp', 'mega', 'tốt', 'đẹp']):
                            matches = False
                            break
                    
                    elif spec_key == 'ram' and spec_value:
                        # Check RAM
                        ram_text = str(product_specs.get('ram', '')).lower()
                        if spec_value not in ram_text:
                            matches = False
                            break
                    
                    elif spec_key == 'rom' and spec_value:
                        # Check ROM
                        rom_text = str(product_specs.get('rom', '')).lower()
                        if spec_value not in rom_text:
                            matches = False
                            break
                    
                    elif spec_key == 'màn hình' and spec_value:
                        # Check screen size
                        screen_text = str(product_specs.get('màn hình', '')).lower()
                        if spec_value not in screen_text:
                            matches = False
                            break
                    
                    elif spec_key in ['chơi game', 'chụp ảnh'] and spec_value:
                        # Check for gaming or photography features
                        description = str(product.get('description', '')).lower()
                        if spec_key == 'chơi game' and not any(keyword in description for keyword in ['game', 'gaming', 'chơi']):
                            matches = False
                            break
                        elif spec_key == 'chụp ảnh' and not any(keyword in description for keyword in ['camera', 'chụp', 'ảnh', 'photo']):
                            matches = False
                            break
                
                if matches:
                    filtered_products.append(product)
            
            return filtered_products
            
        except Exception as e:
            logger.error(f"Failed to filter by specs: {e}")
            return products

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up RAG model... (nothing to release)")
