"""
Profile Manager - User profile and preference management
Handles user data storage and retrieval for personalization
"""

import asyncio
import json
import logging
import os
import sqlite3
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class ProfileManager:
    """
    Profile Manager for user personalization
    
    Features:
    - User profile storage (SQLite/JSON)
    - Query history tracking
    - Preference learning
    - Behavior analysis
    - Profile persistence
    """
    
    def __init__(
        self,
        db_path: str = "data/profiles/profiles.db",
        json_backup: bool = True
    ):
        self.db_path = db_path
        self.json_backup = json_backup
        self.json_dir = "data/profiles/json"
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if json_backup:
            os.makedirs(self.json_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    preferences TEXT,
                    search_history TEXT,
                    purchase_history TEXT,
                    interaction_history TEXT,
                    profile_data TEXT
                )
            ''')
            
            # Create query history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query TEXT,
                    intent TEXT,
                    results_count INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES profiles (user_id)
                )
            ''')
            
            # Create interaction events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interaction_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    event_type TEXT,
                    product_id TEXT,
                    query TEXT,
                    rating REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES profiles (user_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Profile database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, created_at, last_accessed, preferences, 
                       search_history, purchase_history, interaction_history, profile_data
                FROM profiles WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                profile = {
                    "user_id": row[0],
                    "created_at": row[1],
                    "last_accessed": row[2],
                    "preferences": json.loads(row[3]) if row[3] else {},
                    "search_history": json.loads(row[4]) if row[4] else [],
                    "purchase_history": json.loads(row[5]) if row[5] else [],
                    "interaction_history": json.loads(row[6]) if row[6] else [],
                    "profile_data": json.loads(row[7]) if row[7] else {}
                }
                
                # Update last accessed
                await self._update_last_accessed(user_id)
                
                return profile
            else:
                # Create new profile
                return await self._create_new_profile(user_id)
                
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return self._get_default_profile()
    
    async def _create_new_profile(self, user_id: str) -> Dict[str, Any]:
        """Create new user profile"""
        try:
            profile = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "preferences": {
                    "brands": {},
                    "categories": {},
                    "price_range": {},
                    "min_rating": 0,
                    "features": {}
                },
                "search_history": [],
                "purchase_history": [],
                "interaction_history": [],
                "profile_data": {
                    "total_searches": 0,
                    "total_purchases": 0,
                    "favorite_brands": [],
                    "budget_range": None,
                    "preferred_features": []
                }
            }
            
            # Save to database
            await self._save_profile_to_db(profile)
            
            # Save JSON backup
            if self.json_backup:
                await self._save_profile_to_json(profile)
            
            logger.info(f"Created new profile for user: {user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to create new profile: {e}")
            return self._get_default_profile()
    
    async def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """Update user preferences"""
        try:
            profile = await self.get_user_profile(user_id)
            
            # Update preferences
            if "preferences" not in profile:
                profile["preferences"] = {}
            
            profile["preferences"].update(preferences)
            profile["last_accessed"] = datetime.now().isoformat()
            
            # Save updated profile
            await self._save_profile_to_db(profile)
            
            if self.json_backup:
                await self._save_profile_to_json(profile)
            
            logger.info(f"Updated preferences for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user preferences: {e}")
            return False
    
    async def record_query(
        self,
        user_id: str,
        query: str,
        intent: str,
        results_count: int
    ) -> bool:
        """Record user query"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert query history
            cursor.execute('''
                INSERT INTO query_history (user_id, query, intent, results_count)
                VALUES (?, ?, ?, ?)
            ''', (user_id, query, intent, results_count))
            
            # Update profile search history
            profile = await self.get_user_profile(user_id)
            search_history = profile.get("search_history", [])
            
            search_entry = {
                "query": query,
                "intent": intent,
                "results_count": results_count,
                "timestamp": datetime.now().isoformat()
            }
            
            search_history.append(search_entry)
            
            # Keep only last 100 searches
            if len(search_history) > 100:
                search_history = search_history[-100:]
            
            profile["search_history"] = search_history
            profile["profile_data"]["total_searches"] = len(search_history)
            
            await self._save_profile_to_db(profile)
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record query: {e}")
            return False
    
    async def record_interaction(
        self,
        user_id: str,
        event_type: str,
        product_id: Optional[str] = None,
        query: Optional[str] = None,
        rating: Optional[float] = None
    ) -> bool:
        """Record user interaction"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert interaction event
            cursor.execute('''
                INSERT INTO interaction_events (user_id, event_type, product_id, query, rating)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, event_type, product_id, query, rating))
            
            # Update profile
            profile = await self.get_user_profile(user_id)
            interaction_history = profile.get("interaction_history", [])
            
            interaction = {
                "event_type": event_type,
                "product_id": product_id,
                "query": query,
                "rating": rating,
                "timestamp": datetime.now().isoformat()
            }
            
            interaction_history.append(interaction)
            
            # Keep only last 1000 interactions
            if len(interaction_history) > 1000:
                interaction_history = interaction_history[-1000:]
            
            profile["interaction_history"] = interaction_history
            
            # Update profile data based on interaction
            await self._update_profile_from_interaction(profile, interaction)
            
            await self._save_profile_to_db(profile)
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")
            return False
    
    async def _update_profile_from_interaction(
        self,
        profile: Dict[str, Any],
        interaction: Dict[str, Any]
    ):
        """Update profile based on interaction"""
        try:
            event_type = interaction.get("event_type")
            product_id = interaction.get("product_id")
            rating = interaction.get("rating")
            
            if event_type == "view":
                # Update viewed products
                viewed_products = profile.get("profile_data", {}).get("viewed_products", [])
                if product_id and product_id not in viewed_products:
                    viewed_products.append(product_id)
                profile["profile_data"]["viewed_products"] = viewed_products[-50:]  # Keep last 50
            
            elif event_type == "purchase":
                # Update purchase history
                purchase_history = profile.get("purchase_history", [])
                purchase_entry = {
                    "product_id": product_id,
                    "timestamp": interaction["timestamp"]
                }
                purchase_history.append(purchase_entry)
                profile["purchase_history"] = purchase_history[-20:]  # Keep last 20
                profile["profile_data"]["total_purchases"] = len(purchase_history)
            
            elif event_type == "rating" and rating:
                # Update brand preferences based on rating
                if product_id:
                    # This would need product info to extract brand
                    # For now, just store the rating
                    ratings = profile.get("profile_data", {}).get("ratings", {})
                    ratings[product_id] = rating
                    profile["profile_data"]["ratings"] = ratings
            
        except Exception as e:
            logger.error(f"Failed to update profile from interaction: {e}")
    
    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get user insights for personalization"""
        try:
            profile = await self.get_user_profile(user_id)
            
            insights = {
                "total_searches": profile.get("profile_data", {}).get("total_searches", 0),
                "total_purchases": profile.get("profile_data", {}).get("total_purchases", 0),
                "favorite_brands": self._extract_favorite_brands(profile),
                "budget_preference": self._extract_budget_preference(profile),
                "preferred_features": self._extract_preferred_features(profile),
                "search_patterns": self._analyze_search_patterns(profile),
                "purchase_patterns": self._analyze_purchase_patterns(profile)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get user insights: {e}")
            return {}
    
    def _extract_favorite_brands(self, profile: Dict[str, Any]) -> List[str]:
        """Extract favorite brands from profile"""
        try:
            brands = profile.get("preferences", {}).get("brands", {})
            return sorted(brands.keys(), key=lambda x: brands[x], reverse=True)[:5]
        except:
            return []
    
    def _extract_budget_preference(self, profile: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract budget preference from profile"""
        try:
            price_range = profile.get("preferences", {}).get("price_range", {})
            if price_range:
                return {
                    "min": price_range.get("min", 0),
                    "max": price_range.get("max", float('inf'))
                }
            return None
        except:
            return None
    
    def _extract_preferred_features(self, profile: Dict[str, Any]) -> List[str]:
        """Extract preferred features from profile"""
        try:
            features = profile.get("preferences", {}).get("features", {})
            return sorted(features.keys(), key=lambda x: features[x], reverse=True)[:10]
        except:
            return []
    
    def _analyze_search_patterns(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze search patterns"""
        try:
            search_history = profile.get("search_history", [])
            
            if not search_history:
                return {}
            
            # Analyze query frequency
            query_frequency = {}
            intent_frequency = {}
            
            for search in search_history:
                query = search.get("query", "")
                intent = search.get("intent", "")
                
                query_frequency[query] = query_frequency.get(query, 0) + 1
                intent_frequency[intent] = intent_frequency.get(intent, 0) + 1
            
            return {
                "most_searched_queries": sorted(query_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
                "intent_distribution": intent_frequency,
                "total_queries": len(search_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze search patterns: {e}")
            return {}
    
    def _analyze_purchase_patterns(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze purchase patterns"""
        try:
            purchase_history = profile.get("purchase_history", [])
            
            if not purchase_history:
                return {}
            
            return {
                "total_purchases": len(purchase_history),
                "purchase_frequency": len(purchase_history) / 30,  # Purchases per month
                "recent_purchases": purchase_history[-5:]  # Last 5 purchases
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze purchase patterns: {e}")
            return {}
    
    async def _save_profile_to_db(self, profile: Dict[str, Any]) -> bool:
        """Save profile to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO profiles 
                (user_id, created_at, last_accessed, preferences, search_history, 
                 purchase_history, interaction_history, profile_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile["user_id"],
                profile["created_at"],
                profile["last_accessed"],
                json.dumps(profile["preferences"]),
                json.dumps(profile["search_history"]),
                json.dumps(profile["purchase_history"]),
                json.dumps(profile["interaction_history"]),
                json.dumps(profile["profile_data"])
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save profile to database: {e}")
            return False
    
    async def _save_profile_to_json(self, profile: Dict[str, Any]) -> bool:
        """Save profile to JSON backup"""
        try:
            json_file = os.path.join(self.json_dir, f"{profile['user_id']}.json")
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save profile to JSON: {e}")
            return False
    
    async def _update_last_accessed(self, user_id: str) -> bool:
        """Update last accessed timestamp"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE profiles SET last_accessed = ? WHERE user_id = ?
            ''', (datetime.now().isoformat(), user_id))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update last accessed: {e}")
            return False
    
    def _get_default_profile(self) -> Dict[str, Any]:
        """Get default profile for new users"""
        return {
            "user_id": "default",
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "preferences": {},
            "search_history": [],
            "purchase_history": [],
            "interaction_history": [],
            "profile_data": {
                "total_searches": 0,
                "total_purchases": 0,
                "favorite_brands": [],
                "budget_range": None,
                "preferred_features": []
            }
        }