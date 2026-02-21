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

import asyncio
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Giá tối đa hợp lý cho filter "trên X triệu" (tránh 999999999)
MAX_PRICE_VND = 500_000_000  # 500 triệu
RELAXED_TOP_K_MULTIPLIER = 4
RELAXED_TOP_K_MAX = 60
RELAXED_PRICE_TOLERANCE = 0.25  # mở rộng 25% khi relaxed


class RAGModel:
    """
    RAG Model for product search and knowledge retrieval
    
    Features:
    - Vector similarity search using Pinecone
    - Product filtering and ranking
    - Context-aware response generation
    - User personalization integration
    """

    def __init__(self, pinecone_client, model_loader, dimension: Optional[int] = None):
        self.pinecone_client = pinecone_client
        self.model_loader = model_loader
        # Dimension từ config (Pinecone index), không hardcode
        self.dimension = dimension if dimension is not None else 1024
        self.embedding_model_name = "llama-text-embed-v2"
        self._dimension_validated = False

    async def initialize(self):
        """Initialize RAG model components; validate dimension on first embed if needed."""
        try:
            logger.info("Initializing RAG model with Pinecone Cloud embeddings...")
            logger.info(f"Embedding model: {self.embedding_model_name}, dimension={self.dimension}")
            logger.info("RAG model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG model: {e}")
            raise

    def _embed_sync(self, text: str, input_type: str) -> List[float]:
        """Sync embedding call (Pinecone inference is blocking). Run via to_thread."""
        pc = self.pinecone_client.pc
        response = pc.inference.embed(
            model=self.embedding_model_name,
            inputs=[text],
            parameters={"input_type": input_type}
        )
        return list(response[0].values)

    async def _generate_embedding(self, text: str, input_type: str = "passage") -> List[float]:
        """Generate embedding via Pinecone managed model. Use input_type='query' for search."""
        try:
            embedding = await asyncio.to_thread(
                self._embed_sync, text, input_type
            )
            if not self._dimension_validated:
                if len(embedding) != self.dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: got {len(embedding)}, expected {self.dimension}"
                    )
                self._dimension_validated = True
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query (query vs passage)."""
        return await self._generate_embedding(query, input_type="query")
    
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

            # Generate query embedding (query type, not passage)
            query_embedding = await self._generate_query_embedding(query)

            # Search in Pinecone
            search_results = await self.pinecone_client.search_products(
                query_vector=query_embedding,
                top_k=top_k,
                price_range=final_price_range,
                brand=final_brand,
                category=final_category
            )

            # Process and format results (ranking fuses similarity + quality)
            products = await self._process_search_results(search_results, user_id)
            if final_specs:
                products = await self._filter_by_specs(products, final_specs)

            # Relaxed search: chỉ khi không có kết quả; mở rộng price, tăng top_k, giữ brand
            if not products and (final_price_range or final_brand or final_specs):
                relaxed_top_k = min(top_k * RELAXED_TOP_K_MULTIPLIER, RELAXED_TOP_K_MAX)
                relaxed_price = None
                if final_price_range:
                    lo, hi = final_price_range
                    delta = (hi - lo) * RELAXED_PRICE_TOLERANCE if hi > lo else lo * RELAXED_PRICE_TOLERANCE
                    relaxed_price = (
                        max(0, lo - delta),
                        min(MAX_PRICE_VND, hi + delta)
                    )
                logger.info("No products with strict filters, trying relaxed search (wider price, top_k=%s)...", relaxed_top_k)
                relaxed_results = await self.pinecone_client.search_products(
                    query_vector=query_embedding,
                    top_k=relaxed_top_k,
                    price_range=relaxed_price,
                    brand=final_brand,
                    category=final_category
                )
                products = await self._process_search_results(relaxed_results, user_id)
                if final_specs:
                    products = await self._filter_by_specs(products, final_specs)
                products = products[:top_k]

            logger.info(f"Found {len(products)} products")
            return products
            
        except Exception as e:
            logger.error(f"Failed to search products: {e}")
            raise
    
    def _parse_specifications(self, raw: Any) -> Dict[str, Any]:
        """Parse specifications from metadata: JSON string or 'key: value;' fallback."""
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
            parsed = {}
            for item in raw.split(";"):
                item = item.strip()
                if ":" in item:
                    key, val = item.split(":", 1)
                    parsed[key.strip()] = val.strip()
            return parsed
        return {}

    async def _process_search_results(
        self, 
        search_results: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Process and format search results; ranking fuses similarity + quality."""
        try:
            products = []
            for result in search_results:
                product_info = result.get("product_info", {})
                similarity_score = float(result.get("score", 0) or 0)
                specs = self._parse_specifications(product_info.get("specifications"))

                product = {
                    "id": result["id"],
                    "name": product_info.get("name", "Unknown Product"),
                    "brand": product_info.get("brand", "Unknown Brand"),
                    "price": float(product_info.get("price", 0)),
                    "description": product_info.get("description", ""),
                    "category": product_info.get("category", "Unknown"),
                    "image_url": product_info.get("image_url", ""),
                    "rating": float(product_info.get("rating", 0)),
                    "reviews_count": int(product_info.get("reviews_count", 0)),
                    "availability": product_info.get("availability", "In Stock"),
                    "specifications": specs,
                    "similarity_score": similarity_score,
                    "relevance_score": await self._calculate_relevance_score(
                        product_info, user_id, similarity_score=similarity_score
                    )
                }
                products.append(product)

            products.sort(key=lambda x: x["relevance_score"], reverse=True)
            return products
        except Exception as e:
            logger.error(f"Failed to process search results: {e}")
            raise

    async def _calculate_relevance_score(
        self,
        product_info: Dict[str, Any],
        user_id: Optional[str] = None,
        similarity_score: float = 0.0
    ) -> float:
        """Relevance = similarity (semantic) + rating/reviews boost. Không 'mù similarity'."""
        try:
            base = max(0.0, min(1.0, float(similarity_score)))
            rating = product_info.get("rating", 0)
            if rating >= 4.5:
                base += 0.15
            elif rating >= 4.0:
                base += 0.08
            reviews_count = product_info.get("reviews_count", 0)
            if reviews_count >= 1000:
                base += 0.05
            elif reviews_count >= 100:
                base += 0.03
            return min(base, 1.0)
        except Exception:
            return max(0.0, min(1.0, float(similarity_score)))

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
        """Upsert product to vector database. Passage embedding; specs stored as JSON."""
        try:
            logger.info(f"Upserting product: {product_id}")
            product_text = self._create_product_text(product_data)
            embedding = await self._generate_embedding(product_text, input_type="passage")
            specs_dict = product_data.get("specifications") or {}
            if isinstance(specs_dict, str):
                specs_dict = self._parse_specifications(specs_dict)
            specs_json = json.dumps(specs_dict, ensure_ascii=False) if specs_dict else "{}"

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
                    "specifications": specs_json,
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
        """Structured prompt for stable passage embedding (searchable specs)."""
        try:
            parts = []
            if product_data.get("name"):
                parts.append(f"Tên sản phẩm: {product_data['name']}.")
            if product_data.get("brand"):
                parts.append(f"Thương hiệu: {product_data['brand']}.")
            if product_data.get("category"):
                parts.append(f"Danh mục: {product_data['category']}.")
            price = product_data.get("price")
            if price is not None:
                parts.append(f"Giá: {price:,.0f} VNĐ.")
            if product_data.get("description"):
                parts.append(f"Mô tả: {product_data['description']}")
            specs = product_data.get("specifications") or {}
            if isinstance(specs, dict) and specs:
                spec_parts = [f"{k}: {v}" for k, v in specs.items()]
                parts.append("Thông số: " + ", ".join(spec_parts) + ".")
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

            # Price patterns - updated to better handle Vietnamese
            price_patterns = [
                (r'từ\s+(\d+)\s*(?:đến|tới)\s+(\d+)\s*tr(?:iệu)?', 'range'),  # "từ 10 đến 20 triệu"
                (r'dưới\s+(\d+)\s*tr(?:iệu)?(?:\s|$)', 'max'),  # "dưới 20 triệu"
                (r'trên\s+(\d+)\s*tr(?:iệu)?(?:\s|$)', 'min'),  # "trên 20 triệu"
                (r'khoảng\s+(\d+)\s*tr(?:iệu)?(?:\s|$)', 'approx'),  # "khoảng 15 triệu"
                (r'(\d+)\s*tr(?:iệu)?\s*trở\s+xuống', 'max'),  # "20 triệu trở xuống"
                (r'(\d+)\s*tr(?:iệu)?\s*trở\s+lên', 'min'),  # "20 triệu trở lên"
            ]
            
            for pattern, pattern_type in price_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    if pattern_type == 'range':
                        # Range pattern: từ X đến Y
                        min_price = int(match.group(1)) * 1000000
                        max_price = int(match.group(2)) * 1000000
                        metadata["price_range"] = (min_price, max_price)
                    elif pattern_type == 'max':
                        # Max price: dưới X hoặc X trở xuống
                        max_price = int(match.group(1)) * 1000000
                        metadata["price_range"] = (0, max_price)
                    elif pattern_type == 'min':
                        min_price = int(match.group(1)) * 1000000
                        metadata["price_range"] = (min_price, MAX_PRICE_VND)
                        logger.info(f"Extracted 'trên' price - min: {min_price}, max: {MAX_PRICE_VND}")
                    elif pattern_type == 'approx':
                        # Approximate price: khoảng X
                        price = int(match.group(1)) * 1000000
                        tolerance = price * 0.2  # 20% tolerance
                        metadata["price_range"] = (price - tolerance, price + tolerance)
                    logger.info(f"Extracted price range from query: {metadata['price_range']}")
                    break
            
            # Extract brand: map query keywords to dataset brand names (e.g. CSV "Company Name")
            # iPhone/Apple: dataset uses "Apple", so "iphone" must map to "Apple" for Pinecone filter
            brand_keywords_to_canonical = {
                "iphone": "Apple", "apple": "Apple",
                "samsung": "Samsung", "xiaomi": "Xiaomi", "oppo": "Oppo", "vivo": "Vivo",
                "realme": "Realme", "oneplus": "OnePlus", "huawei": "Huawei", "nokia": "Nokia",
                "motorola": "Motorola", "lg": "LG", "sony": "Sony",
            }
            for keyword, canonical_brand in brand_keywords_to_canonical.items():
                if keyword in query_lower:
                    metadata["brand"] = canonical_brand
                    logger.info(f"Extracted brand from query: '{keyword}' -> {canonical_brand}")
                    break
            
            # Extract specs tối thiểu cho điện thoại: RAM, ROM, mAh, W (sạc), 5G, NFC
            specs_patterns = [
                # RAM (gb)
                ('ram', r'(\d+)\s*gb\s*ram|\bram\s+(\d+)\s*gb', 1),
                # ROM (gb)
                ('rom', r'(\d+)\s*gb\s*(?:rom|bộ nhớ|storage)|\brom\s+(\d+)\s*gb', 1),
                # Pin mAh
                ('mah', r'(\d+)\s*mah|\bpin\s+(\d+)\s*mah|battery\s+(\d+)\s*mah', 1),
                # Sạc nhanh W
                ('charging_w', r'(\d+)\s*w\s*(?:sạc|fast|charge)|sạc\s+nhanh\s+(\d+)\s*w|(\d+)\s*w', 1),
                # 5G, NFC (boolean)
                ('5g', r'\b5g\b', None),
                ('nfc', r'\bnfc\b', None),
                # Các spec bổ sung
                ('pin', r'pin\s+(khỏe|tốt|lâu|dài|cao|trâu|mạnh|bền)|pin\s+trâu|trâu\s+pin|pin\s+dự\s+phòng|dùng\s+2\s+ngày', None),
                ('camera', r'camera\s+(tốt|đẹp|chụp\s+ảnh|chất\s+lượng)|chụp\s+ảnh\s+đẹp', None),
                ('màn hình', r'màn\s+hình\s+(\d+\.?\d*)\s*(?:inch|")', 1),
                ('chơi game', r'chơi\s+game|gaming|pubg|genshin|liên\s+quân', None),
                ('chụp ảnh', r'chụp\s+ảnh|photography|photo|quay\s+video', None),
                ('esim', r'\besim\b', None),
                ('chip', r'snapdragon|dimensity|(\w+\s*\d+\s*gen)', 1),
                ('nhỏ gọn', r'nhỏ\s+gọn|dễ\s+cầm|gọn\s+nhẹ', None),
                ('bền', r'bền|ít\s+lỗi|lâu\s+bền', None),
                ('ois', r'\bois\b|chống\s+rung', None),
                ('sạc nhanh', r'sạc\s+nhanh', None),
                ('ip67', r'ip67|ip68|chống\s+nước', None),
                ('jack_35', r'jack\s+3\.5|3\.5\s*mm', None),
                ('amoled', r'amoled|oled|120hz|90hz', None),
            ]
            for spec, pattern, group in specs_patterns:
                match = re.search(pattern, query_lower, re.IGNORECASE)
                if match:
                    if group is not None:
                        # Hỗ trợ nhiều group (vd: ram 8gb | 8gb ram): lấy group đầu tiên khác None
                        val = next((g for g in match.groups() if g is not None), None)
                        if val is not None:
                            metadata["specs"][spec] = (val.strip() if isinstance(val, str) else str(val).strip())
                    else:
                        metadata["specs"][spec] = True
                    logger.info("Extracted spec: %s", spec)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from query: {e}")
            return {}
    
    def _extract_number_from_spec(self, text: str) -> Optional[float]:
        """Extract first number from spec string (e.g. '8GB' -> 8, '6.1 inch' -> 6.1)."""
        if not text:
            return None
        m = re.search(r"(\d+\.?\d*)", str(text).strip())
        return float(m.group(1)) if m else None

    def _normalize_specs_dict(self, d: Dict[str, Any]) -> Dict[str, str]:
        """Lowercase keys for case-insensitive match."""
        if not d or not isinstance(d, dict):
            return {}
        return {k.strip().lower(): (v if isinstance(v, str) else str(v)).strip() for k, v in d.items()}

    async def _filter_by_specs(self, products: List[Dict[str, Any]], specs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter by specs: soft match (description + specs), ram/rom numeric >=."""
        if not specs:
            return products
        try:
            filtered = []
            for product in products:
                product_specs = self._normalize_specs_dict(product.get("specifications") or {})
                desc = str(product.get("description") or "").lower()
                all_text = " ".join(product_specs.values()).lower() + " " + desc
                matches = True
                for spec_key, spec_value in specs.items():
                    if not spec_value:
                        continue
                    if spec_key == "pin":
                        battery_text = product_specs.get("pin", "")
                        has_battery = (
                            any(k in battery_text.lower() for k in ["khỏe", "tốt", "lâu", "dài", "trâu", "mạnh", "mah", "mah"])
                            or re.search(r"\d+\s*mah", battery_text, re.I)
                        )
                        has_battery = has_battery or re.search(r"\d+\s*mah|pin\s+\d+|battery", desc)
                        if not has_battery:
                            matches = False
                            break
                    elif spec_key == "camera":
                        camera_text = product_specs.get("camera", "") + " " + desc
                        if not any(k in camera_text for k in ["mp", "mega", "tốt", "đẹp", "camera", "chụp"]):
                            matches = False
                            break
                    elif spec_key == "ram":
                        req_num = self._extract_number_from_spec(str(spec_value))
                        if req_num is None:
                            continue
                        ram_val = product_specs.get("ram") or ""
                        prod_num = self._extract_number_from_spec(ram_val)
                        if prod_num is None or prod_num < req_num:
                            matches = False
                            break
                    elif spec_key == "rom":
                        req_num = self._extract_number_from_spec(str(spec_value))
                        if req_num is None:
                            continue
                        rom_val = product_specs.get("rom") or product_specs.get("bộ nhớ", "") or ""
                        prod_num = self._extract_number_from_spec(rom_val)
                        if prod_num is None or prod_num < req_num:
                            matches = False
                            break
                    elif spec_key == "màn hình":
                        req_num = self._extract_number_from_spec(str(spec_value))
                        if req_num is None:
                            continue
                        screen_val = product_specs.get("màn hình", "") or product_specs.get("screen", "") or ""
                        prod_num = self._extract_number_from_spec(screen_val)
                        if prod_num is None:
                            matches = False
                            break
                        if abs(prod_num - req_num) > 1.5:
                            matches = False
                            break
                    elif spec_key == "chơi game":
                        if not any(k in all_text for k in ["game", "gaming", "chơi", "pubg", "genshin"]):
                            matches = False
                            break
                    elif spec_key == "chụp ảnh":
                        if not any(k in all_text for k in ["camera", "chụp", "ảnh", "photo", "quay"]):
                            matches = False
                            break
                    elif spec_key in ("5g", "nfc", "esim", "ois", "ip67", "jack_35", "amoled", "nhỏ gọn", "bền", "sạc nhanh"):
                        search_terms = {"5g": ["5g"], "nfc": ["nfc"], "esim": ["esim"], "ois": ["ois", "chống rung"], "ip67": ["ip67", "ip68", "chống nước"], "jack_35": ["jack", "3.5"], "amoled": ["amoled", "oled", "120hz", "90hz"], "nhỏ gọn": ["nhỏ", "gọn", "dễ cầm"], "bền": ["bền"], "sạc nhanh": ["sạc nhanh", "fast charging"]}.get(spec_key, [spec_key])
                        if not any(term in all_text for term in search_terms):
                            matches = False
                            break
                    elif spec_key == "chip":
                        chip_val = (product_specs.get("chip", "") or product_specs.get("cpu", "") or "") + " " + desc
                        chip_val_lower = chip_val.lower()
                        if not any(x in chip_val_lower for x in ["snapdragon", "dimensity", "gen", "chip"]):
                            matches = False
                            break
                if matches:
                    filtered.append(product)
            return filtered
        except Exception as e:
            logger.error(f"Failed to filter by specs: {e}")
            return products

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up RAG model... (nothing to release)")
