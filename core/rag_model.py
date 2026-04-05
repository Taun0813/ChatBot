"""
RAG Model - Retrieval-Augmented Generation for product search
Handles vector search and response generation
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote

from config import get_settings

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
        self.dimension = 1024   
        self.embedding_model_name = "llama-text-embed-v2"
        settings = get_settings()
        product_service_url = getattr(settings, "product_service_url", "http://localhost:8181/api/products")
        self.product_service_url = f"{product_service_url}".rstrip("/")
        self.rag_live_only = bool(getattr(settings, "rag_live_only", True))

    async def initialize(self):
        """Initialize RAG model components"""
        try:
            logger.info("Initializing RAG model with Pinecone Cloud embeddings...")
            logger.info("Embedding model: %s", self.embedding_model_name)
            logger.info("RAG model initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize RAG model: %s", e)
            raise
    
    async def _generate_embedding(self, text: str, input_type: str = "query") -> List[float]:
        """Generate embedding via Pinecone managed model.

        Uses query mode for retrieval by default and falls back to passage mode
        if provider/model rejects the requested input_type.
        """
        try:
            pc = self.pinecone_client.pc  # đã khởi tạo từ adapters/pinecone_client
            try:
                response = pc.inference.embed(
                    model=self.embedding_model_name,
                    inputs=[text],
                    parameters={"input_type": input_type}
                )
            except Exception as primary_error:
                if input_type != "passage":
                    logger.warning(
                        "Embedding input_type '%s' failed (%s), fallback to 'passage'",
                        input_type,
                        primary_error,
                    )
                    response = pc.inference.embed(
                        model=self.embedding_model_name,
                        inputs=[text],
                        parameters={"input_type": "passage"}
                    )
                else:
                    raise
            return response[0].values  # 1 vector (1024-dim)
        except Exception as e:
            logger.error("Failed to generate embedding: %s", e)
            raise
    
    async def search_products(
        self, 
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        price_range: Optional[Tuple[float, float]] = None,
        brand: Optional[str] = None,
        category: Optional[str] = None,
        specs: Optional[Dict[str, Any]] = None,
        live_only: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Search for products using RAG"""
        try:
            logger.info("Searching products for query: %s", query)

            # Extract metadata from query
            extracted_metadata = await self._extract_metadata_from_query(query)
            
            # Merge with provided metadata
            final_price_range = price_range or extracted_metadata.get("price_range")
            final_brand = brand or extracted_metadata.get("brand")
            final_category = category or extracted_metadata.get("category")
            final_specs = specs or extracted_metadata.get("specs", {})
            effective_live_only = self.rag_live_only if live_only is None else bool(live_only)

            # Generate query embedding
            query_embedding = await self._generate_embedding(query, input_type="query")

            # Search in Pinecone
            search_results = await self.pinecone_client.search_products(
                query_vector=query_embedding,
                top_k=top_k,
                price_range=final_price_range,
                brand=final_brand,
                category=final_category,
                only_live_products=effective_live_only,
            )

            # Process and format results
            products = await self._process_search_results(search_results, user_id)

            # Enforce brand match with case-insensitive local filtering.
            # This protects against inconsistent brand metadata values in vector store.
            if final_brand:
                products = self._filter_by_brand(products, final_brand)
            
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
                    # Do not apply exact brand filter here; metadata values can vary
                    # (e.g., "Samsung Electronics"). We apply tolerant local brand
                    # filtering right after retrieval to avoid unrelated products.
                    brand=None,
                    category=None,
                    only_live_products=effective_live_only,
                )
                products = await self._process_search_results(relaxed_results, user_id)
                if final_brand:
                    products = self._filter_by_brand(products, final_brand)
                # Apply only essential filters
                if final_specs and any(spec in final_specs for spec in ['pin', 'camera', 'chơi game']):
                    products = await self._filter_by_specs(products, final_specs)
            
            logger.info("Found %s products", len(products))
            return products
            
        except Exception as e:
            logger.error("Failed to search products: %s", e)
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
                backend_id = product_info.get("backend_id") or result.get("id")
                product_url, specs_url = self._build_product_urls(backend_id, product_info)

                # ✅ Fix: Parse specifications string back to dict if needed
                specs = product_info.get("specifications", {})
                if isinstance(specs, str):
                    # Try to parse from "key: value; key: value" format
                    parsed_specs = {}
                    try:
                        for item in specs.split(';'):
                            if ':' in item:
                                key, val = item.split(':', 1)
                                parsed_specs[key.strip()] = val.strip()
                    except Exception:
                        parsed_specs = specs
                    specs = parsed_specs

                product = {
                    "id": backend_id,
                    "vector_id": result["id"],
                    "backend_id": backend_id,
                    "name": product_info.get("name", "Unknown Product"),
                    "brand": product_info.get("brand", "Unknown Brand"),
                    "price": float(product_info.get("price", 0)),
                    "price_vnd": int(float(product_info.get("price_vnd", product_info.get("price", 0)) or 0)),
                    "currency": product_info.get("currency", "VND"),
                    "source_price": float(product_info.get("source_price", product_info.get("price", 0)) or 0),
                    "source_currency": product_info.get("source_currency", "VND"),
                    "description": product_info.get("description", ""),
                    "category": product_info.get("category", "Unknown"),
                    "image_url": product_info.get("image_url", ""),
                    "rating": float(product_info.get("rating", 0)),
                    "reviews_count": int(product_info.get("reviews_count", 0)),
                    "availability": product_info.get("availability", "In Stock"),
                    "stock": self._availability_to_stock(product_info.get("availability")),
                    "is_live": bool(product_info.get("is_live", True)),
                    "source": product_info.get("source", "vector_store"),
                    "source_id": product_info.get("source_id"),
                    "specifications": specs,
                    "product_url": product_url,
                    "specs_url": specs_url,
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
            logger.error("Failed to process search results: %s", e)
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
            logger.error("Failed to generate product summary: %s", e)
            return "Có lỗi khi tạo tóm tắt sản phẩm."

    async def upsert_product(
        self,
        product_id: str,
        product_data: Dict[str, Any],
        namespace: str = "default"
    ) -> bool:
        """Upsert product to vector database"""
        try:
            logger.info("Upserting product: %s", product_id)

            product_text = self._create_product_text(product_data)
            embedding = await self._generate_embedding(product_text, input_type="passage")
            product_url, specs_url = self._build_product_urls(product_id, product_data)

            vector_data = {
                "id": product_id,
                "values": embedding,
                "metadata": {
                    "backend_id": product_data.get("backend_id") or product_id,
                    "name": product_data.get("name", ""),
                    "brand": product_data.get("brand", ""),
                    "price": product_data.get("price", 0),
                    "price_vnd": int(product_data.get("price_vnd", product_data.get("price", 0)) or 0),
                    "currency": product_data.get("currency", "VND"),
                    "source_price": product_data.get("source_price", product_data.get("price", 0)),
                    "source_currency": product_data.get("source_currency", "VND"),
                    "description": product_data.get("description", ""),
                    "category": product_data.get("category", ""),
                    "image_url": product_data.get("image_url", ""),
                    "rating": product_data.get("rating", 0),
                    "reviews_count": product_data.get("reviews_count", 0),
                    "availability": product_data.get("availability", "In Stock"),
                    "is_live": bool(product_data.get("is_live", True)),
                    "source": product_data.get("source", "ingestion"),
                    "source_id": product_data.get("source_id"),
                    # ✅ Fix: serialize specifications dict
                    "specifications": "; ".join(
                        [f"{k}: {v}" for k, v in product_data.get("specifications", {}).items()]
                    ),
                    "product_url": product_data.get("product_url") or product_url,
                    "specs_url": product_data.get("specs_url") or specs_url,
                    "product_text": product_text,
                },
            }

            await self.pinecone_client.upsert_vectors([vector_data], namespace=namespace)
            logger.info("Successfully upserted product: %s", product_id)
            return True
        except Exception as e:
            logger.error("Failed to upsert product: %s", e)
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

    def _build_product_urls(self, product_id: Optional[str], product_info: Dict[str, Any]) -> Tuple[str, str]:
        """Build product detail/specification URLs for frontend navigation."""
        if not product_id:
            return "", ""

        existing_product_url = (
            product_info.get("product_url")
            or product_info.get("detail_url")
            or product_info.get("url")
        )
        existing_specs_url = (
            product_info.get("specs_url")
            or product_info.get("specifications_url")
        )

        if not existing_product_url:
            encoded_product_id = quote(str(product_id), safe="")
            existing_product_url = f"{self.product_service_url}/{encoded_product_id}"

        if not existing_specs_url:
            existing_specs_url = f"{existing_product_url}#specifications"

        return existing_product_url, existing_specs_url

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
                        # Min price: trên X hoặc X trở lên
                        min_price = int(match.group(1)) * 1000000
                        metadata["price_range"] = (min_price, 999999999)  # Large finite value
                        logger.info("Extracted 'trên' price - min: %s, max: 999999999", min_price)
                    elif pattern_type == 'approx':
                        # Approximate price: khoảng X
                        price = int(match.group(1)) * 1000000
                        tolerance = price * 0.2  # 20% tolerance
                        metadata["price_range"] = (price - tolerance, price + tolerance)
                    logger.info("Extracted price range from query: %s", metadata["price_range"])
                    break
            
            # Extract brand (with basic synonym mapping, e.g. "iphone" -> "Apple")
            brands = [
                'iphone', 'apple', 'samsung', 'xiaomi', 'oppo', 'vivo', 
                'realme', 'oneplus', 'huawei', 'nokia', 'motorola', 'lg', 'sony'
            ]

            # Map user-mentioned brand keywords to normalized brand values used in dataset
            brand_synonyms = {
                # iPhone is a product line of Apple, dataset brand is usually "Apple"
                'iphone': 'Apple',
            }
            
            for brand in brands:
                if brand in query_lower:
                    normalized_brand = brand_synonyms.get(brand, brand.title())
                    metadata["brand"] = normalized_brand
                    logger.info("Extracted brand from query: %s", metadata["brand"])
                    break
            
            # Extract specs
            specs_patterns = {
                'pin': r'pin\s+(khỏe|tốt|lâu|dài|cao)',
                'camera': r'camera\s+(tốt|đẹp|chụp\s+ảnh|chất\s+lượng)',
                'ram': r'(\d+)\s*gb\s*ram',
                'rom': r'(\d+)\s*gb\s*(?:rom|bộ nhớ|storage)',
                'màn hình': r'màn\s+hình\s+(\d+\.?\d*)\s*(?:inch|")',
                'chơi game': r'chơi\s+game|gaming',
                'chụp ảnh': r'chụp\s+ảnh|photography|photo'
            }
            
            for spec, pattern in specs_patterns.items():
                match = re.search(pattern, query_lower)
                if match:
                    if spec in ['ram', 'rom', 'màn hình']:
                        metadata["specs"][spec] = match.group(1)
                        logger.info("Extracted %s from query: %s", spec, match.group(1))
                    else:
                        metadata["specs"][spec] = True
                        logger.info("Extracted %s requirement from query", spec)
            
            return metadata
            
        except Exception as e:
            logger.error("Failed to extract metadata from query: %s", e)
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
            logger.error("Failed to filter by specs: %s", e)
            return products

    def _filter_by_brand(self, products: List[Dict[str, Any]], brand: str) -> List[Dict[str, Any]]:
        """Apply tolerant brand filtering on already-retrieved products."""
        if not products or not brand:
            return products

        normalized_brand = self._normalize_brand_keyword(brand)
        filtered: List[Dict[str, Any]] = []
        for product in products:
            product_brand = str(product.get("brand", "")).strip()
            normalized_product_brand = self._normalize_brand_keyword(product_brand)

            # Match exact normalized brand and allow common variant containment
            # (e.g., "Samsung Electronics" for requested "Samsung").
            if (
                normalized_product_brand == normalized_brand
                or normalized_brand in normalized_product_brand
                or normalized_product_brand in normalized_brand
            ):
                filtered.append(product)

        return filtered

    def _normalize_brand_keyword(self, value: str) -> str:
        """Normalize brand text for robust matching."""
        normalized = (value or "").strip().lower()

        # Normalize common aliases to a single canonical token.
        if "iphone" in normalized or normalized == "apple":
            return "apple"

        return normalized

    def _availability_to_stock(self, availability: Optional[str]) -> int:
        """Map availability text to estimated stock quantity."""
        if not availability:
            return 0
        availability_lower = str(availability).strip().lower()
        if "in stock" in availability_lower or "còn hàng" in availability_lower:
            return 10
        if "limited" in availability_lower or "sắp hết" in availability_lower:
            return 3
        if "out of stock" in availability_lower or "hết hàng" in availability_lower:
            return 0
        return 0

    def _extract_comparison_targets(self, query: str) -> List[str]:
        """Extract up to 2 product target phrases from a comparison query."""
        if not (query or "").strip():
            return []

        normalized_query = re.sub(r"\s+", " ", query.strip())
        split_pattern = r"\s+(?:và|vs\.?|so với|hay)\s+"
        parts = re.split(split_pattern, normalized_query, flags=re.IGNORECASE)

        targets: List[str] = []
        for part in parts:
            cleaned = re.sub(
                r"^(so\s*sánh|so\s*sanh|compare|đối\s*chiếu)\s+",
                "",
                part.strip(),
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(r"(nào\s*tốt\s*hơn\??|với\s*nhau\??)$", "", cleaned, flags=re.IGNORECASE).strip(" ?.,")
            if cleaned and len(cleaned) >= 3:
                targets.append(cleaned)

        # De-duplicate while preserving order
        unique_targets: List[str] = []
        seen = set()
        for target in targets:
            key = target.lower()
            if key not in seen:
                seen.add(key)
                unique_targets.append(target)

        return unique_targets[:2]

    async def resolve_products_for_comparison(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Resolve 2 best products for comparison from user query."""
        targets = self._extract_comparison_targets(query)
        selected_products: List[Dict[str, Any]] = []
        selected_ids = set()

        for target in targets:
            candidate_products = await self.search_products(
                query=target,
                user_id=user_id,
                top_k=3,
            )
            for product in candidate_products:
                product_id = product.get("id")
                if product_id and product_id not in selected_ids:
                    selected_ids.add(product_id)
                    selected_products.append(product)
                    break

        if len(selected_products) < 2:
            fallback_products = await self.search_products(
                query=query,
                user_id=user_id,
                top_k=max(top_k, 6),
            )
            for product in fallback_products:
                product_id = product.get("id")
                if product_id and product_id not in selected_ids:
                    selected_ids.add(product_id)
                    selected_products.append(product)
                if len(selected_products) >= 2:
                    break

        return selected_products[:2]

    async def get_spec_clarification_question(self, query: str) -> Optional[str]:
        """Return a follow-up question when query asks specs but is still ambiguous."""
        if not (query or "").strip():
            return None

        query_lower = query.lower()
        metadata = await self._extract_metadata_from_query(query)
        extracted_specs = metadata.get("specs", {}) if metadata else {}

        has_generic_spec_intent = any(
            token in query_lower
            for token in ["thông số", "cấu hình", "spec", "chi tiết", "mạnh", "tốt"]
        )

        if has_generic_spec_intent and not extracted_specs:
            return (
                "Bạn muốn mình ưu tiên thông số nào để lọc chính xác hơn: "
                "RAM, bộ nhớ (ROM), pin, camera hay màn hình? "
                "Nếu được, bạn cho mình luôn mức cụ thể (ví dụ RAM 8GB, pin 5000mAh)."
            )

        if "ram" in query_lower and "ram" not in extracted_specs:
            return "Bạn muốn RAM tối thiểu bao nhiêu GB (ví dụ 8GB hoặc 12GB)?"
        if any(k in query_lower for k in ["rom", "bộ nhớ", "storage"]) and "rom" not in extracted_specs:
            return "Bạn muốn bộ nhớ trong tối thiểu bao nhiêu GB (ví dụ 128GB hoặc 256GB)?"
        if "màn hình" in query_lower and "màn hình" not in extracted_specs:
            return "Bạn muốn màn hình khoảng bao nhiêu inch?"

        return None

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up RAG model... (nothing to release)")
