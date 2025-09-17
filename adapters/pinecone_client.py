"""
Pinecone Vector Database Client
Handles vector operations for RAG system
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pinecone
from pinecone.core.client.exceptions import PineconeException

# Handle different Pinecone versions
Pinecone = pinecone.Pinecone
ServerlessSpec = getattr(pinecone, 'ServerlessSpec', None)

logger = logging.getLogger(__name__)

class PineconeClient:
    """
    Pinecone vector database client for AI Agent system
    
    Handles:
    - Index creation and management
    - Vector upsert operations
    - Similarity search queries
    - Metadata filtering
    """
    
    def __init__(
        self,
        api_key: str,
        environment: str = "us-west1-gcp-free",
        index_name: str = "product-search",
        dimension: int = 768,
        metric: str = "cosine"
    ):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.pc = None
        self.index = None
        
    async def initialize(self):
        """Initialize Pinecone client and index"""
        try:
            if not self.api_key:
                raise ValueError("Pinecone API key is required")
            
            logger.info(f"Initializing Pinecone client for index: {self.index_name}")
            
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name not in index_names:
                logger.info(f"Creating new index: {self.index_name}")
                await self._create_index()
            else:
                logger.info(f"Using existing index: {self.index_name}")
                self.index = self.pc.Index(self.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    async def _create_index(self):
        """Create Pinecone index"""
        try:
            if ServerlessSpec is not None:
                # New API with ServerlessSpec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            else:
                # Old API without ServerlessSpec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric
                )
            
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                await asyncio.sleep(1)
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Index {self.index_name} created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    async def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """
        Upsert vectors to Pinecone index
        
        Args:
            vectors: List of vectors with id, values, and metadata
            namespace: Namespace for the vectors
        
        Returns:
            Upsert response from Pinecone
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            
            logger.info(f"Upserting {len(vectors)} vectors to namespace: {namespace}")
            
            # Validate vectors
            for vector in vectors:
                if "id" not in vector or "values" not in vector:
                    raise ValueError("Vector must have 'id' and 'values' fields")
                
                if len(vector["values"]) != self.dimension:
                    raise ValueError(f"Vector dimension must be {self.dimension}")
            
            # Upsert vectors
            response = self.index.upsert(
                vectors=vectors,
                namespace=namespace
            )
            
            logger.info(f"Successfully upserted {response.upserted_count} vectors")
            return response
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise
    
    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector for similarity search
            top_k: Number of results to return
            namespace: Namespace to search in
            filter_dict: Metadata filter
            include_metadata: Whether to include metadata in results
        
        Returns:
            List of search results
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            
            if len(query_vector) != self.dimension:
                raise ValueError(f"Query vector dimension must be {self.dimension}")
            
            logger.info(f"Searching for {top_k} similar vectors")
            
            # Perform search
            search_response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict,
                include_metadata=include_metadata
            )
            
            # Format results
            results = []
            for match in search_response.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata if include_metadata else {}
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise
    
    async def search_products(
        self,
        query_vector: List[float],
        top_k: int = 5,
        price_range: Optional[Tuple[float, float]] = None,
        brand: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for products using vector similarity
        
        Args:
            query_vector: Query vector for product search
            top_k: Number of products to return
            price_range: Optional price range filter (min_price, max_price)
            brand: Optional brand filter
            category: Optional category filter
        
        Returns:
            List of product search results
        """
        try:
            # Build filter dictionary
            filter_dict = {}
            
            if price_range:
                min_price, max_price = price_range
                filter_dict["price"] = {"$gte": min_price, "$lte": max_price}
            
            if brand:
                filter_dict["brand"] = {"$eq": brand}
            
            if category:
                filter_dict["category"] = {"$eq": category}
            
            # Search vectors
            results = await self.search_vectors(
                query_vector=query_vector,
                top_k=top_k,
                filter_dict=filter_dict if filter_dict else None,
                include_metadata=True
            )
            
            # Format product results
            products = []
            for result in results:
                product = {
                    "id": result["id"],
                    "score": result["score"],
                    "product_info": result["metadata"]
                }
                products.append(product)
            
            logger.info(f"Found {len(products)} products")
            return products
            
        except Exception as e:
            logger.error(f"Failed to search products: {e}")
            raise
    
    async def get_vector_by_id(
        self,
        vector_id: str,
        namespace: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Get vector by ID
        
        Args:
            vector_id: ID of the vector to retrieve
            namespace: Namespace to search in
        
        Returns:
            Vector data or None if not found
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            
            response = self.index.fetch(
                ids=[vector_id],
                namespace=namespace
            )
            
            if vector_id in response.vectors:
                vector_data = response.vectors[vector_id]
                return {
                    "id": vector_data.id,
                    "values": vector_data.values,
                    "metadata": vector_data.metadata
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get vector by ID: {e}")
            raise
    
    async def delete_vectors(
        self,
        vector_ids: List[str],
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """
        Delete vectors by IDs
        
        Args:
            vector_ids: List of vector IDs to delete
            namespace: Namespace to delete from
        
        Returns:
            Delete response from Pinecone
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            
            logger.info(f"Deleting {len(vector_ids)} vectors")
            
            response = self.index.delete(
                ids=vector_ids,
                namespace=namespace
            )
            
            logger.info(f"Successfully deleted vectors")
            return response
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            if not self.index:
                raise ValueError("Index not initialized")
            
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("Cleaning up Pinecone client...")
            # Pinecone client doesn't need explicit cleanup
            logger.info("Pinecone client cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Pinecone cleanup: {e}")