"""
Recommender System - Rule-based product recommendations
Handles personalized product recommendations based on user profiles
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Recommender:
    """
    Rule-based Recommender System
    
    Features:
    - Brand preference scoring
    - Price range filtering
    - Feature-based recommendations
    - Purchase history analysis
    - Search pattern analysis
    - Collaborative filtering (basic)
    """
    
    def __init__(self, profile_manager=None):
        self.profile_manager = profile_manager
        self.brand_weights = {
            "iphone": 1.0,
            "apple": 1.0,
            "samsung": 0.9,
            "xiaomi": 0.8,
            "oppo": 0.7,
            "vivo": 0.7,
            "realme": 0.6,
            "oneplus": 0.8,
            "huawei": 0.6,
            "nokia": 0.5,
            "motorola": 0.5,
            "lg": 0.6,
            "sony": 0.7
        }
        
        self.feature_weights = {
            "pin khỏe": 0.3,
            "camera tốt": 0.4,
            "chơi game": 0.2,
            "màn hình lớn": 0.1,
            "ram cao": 0.2,
            "rom lớn": 0.1,
            "chụp ảnh": 0.3,
            "pin lâu": 0.3,
            "battery": 0.3,
            "gaming": 0.2,
            "camera": 0.4,
            "display": 0.1,
            "storage": 0.1
        }
    
    async def get_personalized_recommendations(
        self,
        user_id: str,
        query: str,
        search_results: List[Dict[str, Any]],
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations based on user profile
        
        Args:
            user_id: User identifier
            query: Search query
            search_results: Raw search results from RAG
            max_recommendations: Maximum number of recommendations
        
        Returns:
            List of personalized recommendations
        """
        try:
            logger.info(f"Generating personalized recommendations for user: {user_id}")
            
            if not search_results:
                return []
            
            # Get user profile
            profile = await self.profile_manager.get_user_profile(user_id) if self.profile_manager else {}
            user_insights = await self.profile_manager.get_user_insights(user_id) if self.profile_manager else {}
            
            # Score each product
            scored_products = []
            
            for product in search_results:
                score = await self._calculate_personalization_score(
                    product, profile, user_insights, query
                )
                
                scored_products.append({
                    **product,
                    "personalization_score": score,
                    "original_score": product.get("score", 0)
                })
            
            # Sort by personalization score
            scored_products.sort(key=lambda x: x["personalization_score"], reverse=True)
            
            # Take top recommendations
            recommendations = scored_products[:max_recommendations]
            
            # Add recommendation reasons
            for rec in recommendations:
                rec["recommendation_reasons"] = await self._get_recommendation_reasons(
                    rec, profile, user_insights
                )
            
            logger.info(f"Generated {len(recommendations)} personalized recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate personalized recommendations: {e}")
            return search_results[:max_recommendations]  # Fallback to original results
    
    async def _calculate_personalization_score(
        self,
        product: Dict[str, Any],
        profile: Dict[str, Any],
        user_insights: Dict[str, Any],
        query: str
    ) -> float:
        """Calculate personalization score for a product"""
        try:
            base_score = product.get("score", 0.5)
            personalization_multiplier = 1.0
            
            # Brand preference scoring
            brand_score = await self._calculate_brand_score(product, profile, user_insights)
            personalization_multiplier *= brand_score
            
            # Price preference scoring
            price_score = await self._calculate_price_score(product, profile, user_insights)
            personalization_multiplier *= price_score
            
            # Feature preference scoring
            feature_score = await self._calculate_feature_score(product, profile, user_insights, query)
            personalization_multiplier *= feature_score
            
            # Purchase history scoring
            history_score = await self._calculate_history_score(product, profile, user_insights)
            personalization_multiplier *= history_score
            
            # Search pattern scoring
            pattern_score = await self._calculate_pattern_score(product, profile, user_insights, query)
            personalization_multiplier *= pattern_score
            
            # Calculate final score
            final_score = base_score * personalization_multiplier
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate personalization score: {e}")
            return product.get("score", 0.5)
    
    async def _calculate_brand_score(
        self,
        product: Dict[str, Any],
        profile: Dict[str, Any],
        user_insights: Dict[str, Any]
    ) -> float:
        """Calculate brand preference score"""
        try:
            product_brand = product.get("brand", "").lower()
            if not product_brand:
                return 1.0
            
            # Get user's favorite brands
            favorite_brands = user_insights.get("favorite_brands", [])
            
            if not favorite_brands:
                # Use default brand weights
                return self.brand_weights.get(product_brand, 0.5)
            
            # Check if brand is in favorites
            for i, fav_brand in enumerate(favorite_brands):
                if fav_brand.lower() in product_brand or product_brand in fav_brand.lower():
                    # Higher position = higher score
                    return 1.0 - (i * 0.1)
            
            # Brand not in favorites, use default weight
            return self.brand_weights.get(product_brand, 0.5)
            
        except Exception as e:
            logger.error(f"Failed to calculate brand score: {e}")
            return 1.0
    
    async def _calculate_price_score(
        self,
        product: Dict[str, Any],
        profile: Dict[str, Any],
        user_insights: Dict[str, Any]
    ) -> float:
        """Calculate price preference score"""
        try:
            product_price = product.get("price", 0)
            if product_price <= 0:
                return 1.0
            
            # Get user's budget preference
            budget_preference = user_insights.get("budget_preference")
            
            if not budget_preference:
                return 1.0
            
            min_budget = budget_preference.get("min", 0)
            max_budget = budget_preference.get("max", float('inf'))
            
            # Check if price is within budget
            if min_budget <= product_price <= max_budget:
                return 1.0
            elif product_price < min_budget:
                # Too cheap, might be low quality
                return 0.7
            else:
                # Too expensive
                return 0.3
            
        except Exception as e:
            logger.error(f"Failed to calculate price score: {e}")
            return 1.0
    
    async def _calculate_feature_score(
        self,
        product: Dict[str, Any],
        profile: Dict[str, Any],
        user_insights: Dict[str, Any],
        query: str
    ) -> float:
        """Calculate feature preference score"""
        try:
            product_features = product.get("features", [])
            if not product_features:
                return 1.0
            
            # Get user's preferred features
            preferred_features = user_insights.get("preferred_features", [])
            
            if not preferred_features:
                # Use query-based feature matching
                return await self._calculate_query_feature_score(product, query)
            
            # Calculate feature match score
            total_score = 0.0
            matched_features = 0
            
            for feature in preferred_features:
                feature_lower = feature.lower()
                for product_feature in product_features:
                    product_feature_lower = str(product_feature).lower()
                    
                    if feature_lower in product_feature_lower or product_feature_lower in feature_lower:
                        weight = self.feature_weights.get(feature_lower, 0.1)
                        total_score += weight
                        matched_features += 1
                        break
            
            if matched_features == 0:
                return 1.0
            
            # Normalize score
            return min(1.0, total_score / len(preferred_features))
            
        except Exception as e:
            logger.error(f"Failed to calculate feature score: {e}")
            return 1.0
    
    async def _calculate_query_feature_score(
        self,
        product: Dict[str, Any],
        query: str
    ) -> float:
        """Calculate feature score based on query"""
        try:
            query_lower = query.lower()
            product_features = product.get("features", [])
            
            if not product_features:
                return 1.0
            
            total_score = 0.0
            matched_features = 0
            
            for feature_key, weight in self.feature_weights.items():
                if feature_key in query_lower:
                    for product_feature in product_features:
                        product_feature_lower = str(product_feature).lower()
                        if feature_key in product_feature_lower or product_feature_lower in feature_key:
                            total_score += weight
                            matched_features += 1
                            break
            
            if matched_features == 0:
                return 1.0
            
            return min(1.0, total_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate query feature score: {e}")
            return 1.0
    
    async def _calculate_history_score(
        self,
        product: Dict[str, Any],
        profile: Dict[str, Any],
        user_insights: Dict[str, Any]
    ) -> float:
        """Calculate score based on purchase history"""
        try:
            product_brand = product.get("brand", "").lower()
            product_category = product.get("category", "").lower()
            
            if not product_brand and not product_category:
                return 1.0
            
            # Get purchase patterns
            purchase_patterns = user_insights.get("purchase_patterns", {})
            recent_purchases = purchase_patterns.get("recent_purchases", [])
            
            if not recent_purchases:
                return 1.0
            
            # Check if user has purchased similar products
            brand_matches = 0
            category_matches = 0
            
            for purchase in recent_purchases:
                # This would need product info to check brand/category
                # For now, just return neutral score
                pass
            
            # Simple scoring based on purchase frequency
            total_purchases = purchase_patterns.get("total_purchases", 0)
            if total_purchases > 0:
                return 1.0 + (total_purchases * 0.1)  # Slight boost for active users
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Failed to calculate history score: {e}")
            return 1.0
    
    async def _calculate_pattern_score(
        self,
        product: Dict[str, Any],
        profile: Dict[str, Any],
        user_insights: Dict[str, Any],
        query: str
    ) -> float:
        """Calculate score based on search patterns"""
        try:
            search_patterns = user_insights.get("search_patterns", {})
            most_searched_queries = search_patterns.get("most_searched_queries", [])
            
            if not most_searched_queries:
                return 1.0
            
            # Check if current query matches previous searches
            query_lower = query.lower()
            product_name = product.get("name", "").lower()
            product_brand = product.get("brand", "").lower()
            
            for prev_query, frequency in most_searched_queries:
                prev_query_lower = prev_query.lower()
                
                # Check for brand matches
                if product_brand and product_brand in prev_query_lower:
                    return 1.0 + (frequency * 0.1)
                
                # Check for product name matches
                if product_name and any(word in prev_query_lower for word in product_name.split()):
                    return 1.0 + (frequency * 0.05)
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Failed to calculate pattern score: {e}")
            return 1.0
    
    async def _get_recommendation_reasons(
        self,
        product: Dict[str, Any],
        profile: Dict[str, Any],
        user_insights: Dict[str, Any]
    ) -> List[str]:
        """Get reasons why this product was recommended"""
        try:
            reasons = []
            
            # Brand preference reason
            favorite_brands = user_insights.get("favorite_brands", [])
            product_brand = product.get("brand", "").lower()
            
            if favorite_brands and any(fav_brand.lower() in product_brand for fav_brand in favorite_brands):
                reasons.append(f"Phù hợp với thương hiệu bạn yêu thích: {product_brand.title()}")
            
            # Price reason
            budget_preference = user_insights.get("budget_preference")
            product_price = product.get("price", 0)
            
            if budget_preference and product_price > 0:
                min_budget = budget_preference.get("min", 0)
                max_budget = budget_preference.get("max", float('inf'))
                
                if min_budget <= product_price <= max_budget:
                    reasons.append("Nằm trong ngân sách phù hợp")
                elif product_price < min_budget:
                    reasons.append("Giá rẻ hơn ngân sách dự kiến")
            
            # Feature reason
            preferred_features = user_insights.get("preferred_features", [])
            if preferred_features:
                matched_features = []
                product_features = product.get("features", [])
                
                for feature in preferred_features:
                    for product_feature in product_features:
                        if feature.lower() in str(product_feature).lower():
                            matched_features.append(feature)
                            break
                
                if matched_features:
                    reasons.append(f"Có các tính năng bạn quan tâm: {', '.join(matched_features[:3])}")
            
            # Purchase history reason
            purchase_patterns = user_insights.get("purchase_patterns", {})
            total_purchases = purchase_patterns.get("total_purchases", 0)
            
            if total_purchases > 0:
                reasons.append("Dựa trên lịch sử mua hàng của bạn")
            
            # Default reason if no specific reasons
            if not reasons:
                reasons.append("Sản phẩm phù hợp với tìm kiếm của bạn")
            
            return reasons[:3]  # Limit to 3 reasons
            
        except Exception as e:
            logger.error(f"Failed to get recommendation reasons: {e}")
            return ["Sản phẩm phù hợp với tìm kiếm của bạn"]
    
    async def get_trending_recommendations(
        self,
        user_id: str,
        category: Optional[str] = None,
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """Get trending recommendations (not implemented yet)"""
        try:
            # This would integrate with analytics to get trending products
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Failed to get trending recommendations: {e}")
            return []
    
    async def get_collaborative_recommendations(
        self,
        user_id: str,
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """Get collaborative filtering recommendations (not implemented yet)"""
        try:
            # This would use user similarity to recommend products
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Failed to get collaborative recommendations: {e}")
            return []