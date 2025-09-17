"""
Personalization Model - User preference and recommendation handling
Handles user profiling and personalized recommendations
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PersonalizationModel:
    """
    Personalization Model for user preference handling with Spring Boot integration
    
    Features:
    - User profile management (SQLite + JSON backup)
    - Advanced product recommendations
    - Preference learning from interactions
    - Behavioral analysis
    - Spring Boot service integration
    """
    
    def __init__(
        self,
        enable_personalization: bool = True,
        enable_recommendations: bool = True,
        enable_rl_learning: bool = True,
        profiles_dir: str = "./data/profiles",
        models_dir: str = "./data/models",
        profile_manager=None,
        recommender=None
    ):
        self.enable_personalization = enable_personalization
        self.enable_recommendations = enable_recommendations
        self.enable_rl_learning = enable_rl_learning
        self.profiles_dir = profiles_dir
        self.models_dir = models_dir
        
        # Initialize components
        self.profile_manager = profile_manager
        self.recommender = recommender
        
        # Ensure directories exist
        os.makedirs(profiles_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
    
    async def initialize(self):
        """Initialize personalization model"""
        try:
            if self.profile_manager:
                logger.info("Personalization Model initialized with Profile Manager")
            if self.recommender:
                logger.info("Personalization Model initialized with Recommender")
            
            logger.info("Personalization Model initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize Personalization Model: {e}")
            raise
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get user profile using Profile Manager
        
        Args:
            user_id: User identifier
        
        Returns:
            User profile data
        """
        try:
            if not self.enable_personalization:
                return self._get_default_profile()
            
            if self.profile_manager:
                return await self.profile_manager.get_user_profile(user_id)
            else:
                # Fallback to file-based storage
                return await self._get_profile_from_file(user_id)
                
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return self._get_default_profile()
    
    async def _get_profile_from_file(self, user_id: str) -> Dict[str, Any]:
        """Fallback method to get profile from file"""
        try:
            profile_file = os.path.join(self.profiles_dir, f"{user_id}.json")
            
            if os.path.exists(profile_file):
                with open(profile_file, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                
                # Update last accessed
                profile["last_accessed"] = datetime.now().isoformat()
                
                # Save updated profile
                await self._save_user_profile(user_id, profile)
                
                return profile
            else:
                # Create new profile
                profile = self._create_new_profile(user_id)
                await self._save_user_profile(user_id, profile)
                return profile
                
        except Exception as e:
            logger.error(f"Failed to get profile from file: {e}")
            return self._get_default_profile()
    
    async def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """
        Update user preferences
        
        Args:
            user_id: User identifier
            preferences: New preferences
        
        Returns:
            Success status
        """
        try:
            if not self.enable_personalization:
                return True
            
            profile = await self.get_user_profile(user_id)
            
            # Update preferences
            if "preferences" not in profile:
                profile["preferences"] = {}
            
            profile["preferences"].update(preferences)
            profile["last_updated"] = datetime.now().isoformat()
            
            # Save updated profile
            await self._save_user_profile(user_id, profile)
            
            logger.info(f"Updated preferences for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user preferences: {e}")
            return False
    
    async def record_user_interaction(
        self,
        user_id: str,
        interaction_type: str,
        product_id: Optional[str] = None,
        query: Optional[str] = None,
        rating: Optional[float] = None
    ) -> bool:
        """
        Record user interaction for learning
        
        Args:
            user_id: User identifier
            interaction_type: Type of interaction (search, view, purchase, etc.)
            product_id: Product identifier (if applicable)
            query: Search query (if applicable)
            rating: User rating (if applicable)
        
        Returns:
            Success status
        """
        try:
            if not self.enable_personalization:
                return True
            
            profile = await self.get_user_profile(user_id)
            
            # Add interaction to history
            interaction = {
                "type": interaction_type,
                "product_id": product_id,
                "query": query,
                "rating": rating,
                "timestamp": datetime.now().isoformat()
            }
            
            if "interaction_history" not in profile:
                profile["interaction_history"] = []
            
            profile["interaction_history"].append(interaction)
            
            # Keep only last 1000 interactions
            if len(profile["interaction_history"]) > 1000:
                profile["interaction_history"] = profile["interaction_history"][-1000:]
            
            # Update profile based on interaction
            await self._update_profile_from_interaction(profile, interaction)
            
            # Save updated profile
            await self._save_user_profile(user_id, profile)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record user interaction: {e}")
            return False
    
    async def get_personalized_recommendations(
        self,
        user_id: str,
        query: str,
        search_results: List[Dict[str, Any]],
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get personalized product recommendations using Recommender
        
        Args:
            user_id: User identifier
            query: Search query
            search_results: List of products from RAG search
            max_recommendations: Maximum number of recommendations
        
        Returns:
            Ranked list of recommended products
        """
        try:
            if not self.enable_recommendations:
                return search_results[:max_recommendations]
            
            if self.recommender:
                # Use advanced recommender
                return await self.recommender.get_personalized_recommendations(
                    user_id=user_id,
                    query=query,
                    search_results=search_results,
                    max_recommendations=max_recommendations
                )
            else:
                # Fallback to basic personalization
                return await self._get_basic_recommendations(
                    user_id, search_results, max_recommendations
                )
            
        except Exception as e:
            logger.error(f"Failed to get personalized recommendations: {e}")
            return search_results[:max_recommendations]
    
    async def _get_basic_recommendations(
        self,
        user_id: str,
        products: List[Dict[str, Any]],
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """Basic recommendation fallback"""
        try:
            profile = await self.get_user_profile(user_id)
            
            # Calculate personalized scores
            scored_products = []
            for product in products:
                score = await self._calculate_personalization_score(
                    product, profile
                )
                product_copy = product.copy()
                product_copy["personalization_score"] = score
                scored_products.append(product_copy)
            
            # Sort by personalization score
            scored_products.sort(
                key=lambda x: x["personalization_score"], 
                reverse=True
            )
            
            return scored_products[:max_recommendations]
            
        except Exception as e:
            logger.error(f"Failed to get basic recommendations: {e}")
            return products[:max_recommendations]
    
    async def _calculate_personalization_score(
        self,
        product: Dict[str, Any],
        profile: Dict[str, Any]
    ) -> float:
        """Calculate personalization score for product"""
        try:
            score = 0.0
            
            # Base score from similarity (if available)
            if "similarity_score" in product:
                score += product["similarity_score"] * 0.4
            
            # Brand preference
            brand = product.get("brand", "").lower()
            brand_preferences = profile.get("preferences", {}).get("brands", {})
            if brand in brand_preferences:
                score += brand_preferences[brand] * 0.2
            
            # Price preference
            price = product.get("price", 0)
            price_preference = profile.get("preferences", {}).get("price_range", {})
            if price_preference:
                min_price = price_preference.get("min", 0)
                max_price = price_preference.get("max", float('inf'))
                if min_price <= price <= max_price:
                    score += 0.2
            
            # Category preference
            category = product.get("category", "").lower()
            category_preferences = profile.get("preferences", {}).get("categories", {})
            if category in category_preferences:
                score += category_preferences[category] * 0.1
            
            # Rating preference
            rating = product.get("rating", 0)
            min_rating = profile.get("preferences", {}).get("min_rating", 0)
            if rating >= min_rating:
                score += (rating / 5.0) * 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate personalization score: {e}")
            return 0.5
    
    async def _update_profile_from_interaction(
        self,
        profile: Dict[str, Any],
        interaction: Dict[str, Any]
    ):
        """Update profile based on user interaction"""
        try:
            interaction_type = interaction.get("type")
            
            if interaction_type == "search":
                # Update search preferences
                query = interaction.get("query", "")
                if query:
                    search_history = profile.get("search_history", [])
                    search_history.append({
                        "query": query,
                        "timestamp": interaction["timestamp"]
                    })
                    profile["search_history"] = search_history[-100:]  # Keep last 100 searches
            
            elif interaction_type == "view":
                # Update product preferences
                product_id = interaction.get("product_id")
                if product_id:
                    viewed_products = profile.get("viewed_products", [])
                    if product_id not in viewed_products:
                        viewed_products.append(product_id)
                    profile["viewed_products"] = viewed_products[-50:]  # Keep last 50 viewed
            
            elif interaction_type == "purchase":
                # Update purchase history
                product_id = interaction.get("product_id")
                if product_id:
                    purchase_history = profile.get("purchase_history", [])
                    purchase_history.append({
                        "product_id": product_id,
                        "timestamp": interaction["timestamp"]
                    })
                    profile["purchase_history"] = purchase_history[-20:]  # Keep last 20 purchases
            
            elif interaction_type == "rating":
                # Update rating preferences
                rating = interaction.get("rating")
                product_id = interaction.get("product_id")
                if rating and product_id:
                    ratings = profile.get("ratings", {})
                    ratings[product_id] = rating
                    profile["ratings"] = ratings
            
        except Exception as e:
            logger.error(f"Failed to update profile from interaction: {e}")
    
    def _create_new_profile(self, user_id: str) -> Dict[str, Any]:
        """Create new user profile"""
        return {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "preferences": {
                "brands": {},
                "categories": {},
                "price_range": {},
                "min_rating": 0
            },
            "interaction_history": [],
            "search_history": [],
            "viewed_products": [],
            "purchase_history": [],
            "ratings": {}
        }
    
    def _get_default_profile(self) -> Dict[str, Any]:
        """Get default profile for users without personalization"""
        return {
            "user_id": "default",
            "preferences": {},
            "personalization_enabled": False
        }
    
    async def _save_user_profile(self, user_id: str, profile: Dict[str, Any]) -> bool:
        """Save user profile to file"""
        try:
            profile_file = os.path.join(self.profiles_dir, f"{user_id}.json")
            
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save user profile: {e}")
            return False