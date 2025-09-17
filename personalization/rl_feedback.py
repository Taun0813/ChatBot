"""
Reinforcement Learning from user feedback
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

@dataclass
class Feedback:
    """User feedback data"""
    user_id: str
    product_id: str
    action: str  # "click", "purchase", "dismiss", "like", "dislike"
    reward: float  # -1 to 1
    context: Dict[str, Any]
    timestamp: str

class RLFeedbackSystem:
    """Reinforcement Learning system for learning from user feedback"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.feedback_history: List[Feedback] = []
        self.user_models: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.product_features: Dict[str, Dict[str, float]] = {}
        self.action_rewards = {
            "purchase": 1.0,
            "like": 0.8,
            "click": 0.3,
            "dismiss": -0.2,
            "dislike": -0.5
        }
    
    async def record_feedback(
        self, 
        user_id: str, 
        product_id: str, 
        action: str, 
        context: Dict[str, Any]
    ) -> bool:
        """Record user feedback"""
        try:
            reward = self.action_rewards.get(action, 0.0)
            
            feedback = Feedback(
                user_id=user_id,
                product_id=product_id,
                action=action,
                reward=reward,
                context=context,
                timestamp=context.get("timestamp", "")
            )
            
            self.feedback_history.append(feedback)
            
            # Update user model
            await self._update_user_model(user_id, product_id, reward, context)
            
            # Update product features
            await self._update_product_features(product_id, reward, context)
            
            self.logger.info(
                f"Recorded feedback: user={user_id}, product={product_id}, "
                f"action={action}, reward={reward}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording feedback: {e}")
            return False
    
    async def _update_user_model(
        self, 
        user_id: str, 
        product_id: str, 
        reward: float, 
        context: Dict[str, Any]
    ) -> None:
        """Update user preference model based on feedback"""
        try:
            # Learning rate
            alpha = 0.1
            
            # Get product features
            product_features = self.product_features.get(product_id, {})
            
            # Update user preferences for each feature
            for feature, value in product_features.items():
                if feature not in self.user_models[user_id]:
                    self.user_models[user_id][feature] = 0.0
                
                # Update preference based on reward
                self.user_models[user_id][feature] += alpha * reward * value
            
            # Normalize preferences
            self._normalize_user_preferences(user_id)
            
        except Exception as e:
            self.logger.error(f"Error updating user model: {e}")
    
    async def _update_product_features(
        self, 
        product_id: str, 
        reward: float, 
        context: Dict[str, Any]
    ) -> None:
        """Update product feature weights based on feedback"""
        try:
            if product_id not in self.product_features:
                self.product_features[product_id] = {}
            
            # Extract features from context
            features = self._extract_features_from_context(context)
            
            # Update feature weights
            for feature, value in features.items():
                if feature not in self.product_features[product_id]:
                    self.product_features[product_id][feature] = 0.0
                
                # Update weight based on reward
                self.product_features[product_id][feature] += 0.05 * reward * value
            
        except Exception as e:
            self.logger.error(f"Error updating product features: {e}")
    
    def _extract_features_from_context(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from context"""
        features = {}
        
        # Time-based features
        hour = context.get("hour", 12)
        features["time_morning"] = 1.0 if 6 <= hour < 12 else 0.0
        features["time_afternoon"] = 1.0 if 12 <= hour < 18 else 0.0
        features["time_evening"] = 1.0 if 18 <= hour < 24 else 0.0
        
        # Device features
        device = context.get("device", "desktop")
        features[f"device_{device}"] = 1.0
        
        # Session features
        session_length = context.get("session_length", 0)
        features["short_session"] = 1.0 if session_length < 300 else 0.0
        features["long_session"] = 1.0 if session_length > 1800 else 0.0
        
        return features
    
    def _normalize_user_preferences(self, user_id: str) -> None:
        """Normalize user preferences to prevent explosion"""
        preferences = self.user_models[user_id]
        
        if not preferences:
            return
        
        # L2 normalization
        norm = np.sqrt(sum(v**2 for v in preferences.values()))
        if norm > 0:
            for feature in preferences:
                preferences[feature] /= norm
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, float]:
        """Get learned user preferences"""
        return dict(self.user_models[user_id])
    
    async def predict_reward(
        self, 
        user_id: str, 
        product_id: str, 
        context: Dict[str, Any]
    ) -> float:
        """Predict reward for user-product interaction"""
        try:
            user_prefs = self.user_models[user_id]
            product_features = self.product_features.get(product_id, {})
            
            if not user_prefs or not product_features:
                return 0.0
            
            # Calculate dot product of user preferences and product features
            reward = sum(
                user_prefs.get(feature, 0) * value 
                for feature, value in product_features.items()
            )
            
            # Apply context features
            context_features = self._extract_features_from_context(context)
            for feature, value in context_features.items():
                if feature in user_prefs:
                    reward += user_prefs[feature] * value * 0.1
            
            return max(-1.0, min(1.0, reward))  # Clamp to [-1, 1]
            
        except Exception as e:
            self.logger.error(f"Error predicting reward: {e}")
            return 0.0
    
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        try:
            total_feedback = len(self.feedback_history)
            
            if total_feedback == 0:
                return {"total_feedback": 0}
            
            # Calculate average reward by action
            action_rewards = defaultdict(list)
            for feedback in self.feedback_history:
                action_rewards[feedback.action].append(feedback.reward)
            
            avg_rewards = {
                action: np.mean(rewards) 
                for action, rewards in action_rewards.items()
            }
            
            # Calculate learning progress
            recent_feedback = self.feedback_history[-100:] if len(self.feedback_history) > 100 else self.feedback_history
            recent_avg_reward = np.mean([f.reward for f in recent_feedback])
            
            return {
                "total_feedback": total_feedback,
                "unique_users": len(self.user_models),
                "unique_products": len(self.product_features),
                "average_rewards_by_action": avg_rewards,
                "recent_average_reward": recent_avg_reward,
                "learning_progress": self._calculate_learning_progress()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating learning stats: {e}")
            return {"error": str(e)}
    
    def _calculate_learning_progress(self) -> float:
        """Calculate learning progress (0-1)"""
        if len(self.feedback_history) < 10:
            return 0.0
        
        # Calculate reward trend over time
        window_size = min(50, len(self.feedback_history) // 2)
        recent_window = self.feedback_history[-window_size:]
        older_window = self.feedback_history[-(window_size*2):-window_size]
        
        if not older_window:
            return 0.0
        
        recent_avg = np.mean([f.reward for f in recent_window])
        older_avg = np.mean([f.reward for f in older_window])
        
        # Progress is improvement in average reward
        improvement = recent_avg - older_avg
        progress = max(0.0, min(1.0, (improvement + 1) / 2))  # Normalize to [0, 1]
        
        return progress
    
    async def save_models(self, filepath: str) -> bool:
        """Save learned models to file"""
        try:
            models_data = {
                "user_models": dict(self.user_models),
                "product_features": self.product_features,
                "feedback_count": len(self.feedback_history)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(models_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved RL models to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            return False
    
    async def load_models(self, filepath: str) -> bool:
        """Load learned models from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                models_data = json.load(f)
            
            self.user_models = defaultdict(dict, models_data.get("user_models", {}))
            self.product_features = models_data.get("product_features", {})
            
            self.logger.info(f"Loaded RL models from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
