"""
Training Pipeline Integration
Kết nối training system với main application cho continuous improvement
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import threading
import time

from .prepare_data import DataPreparator
from .finetune import FineTuner
from .evaluate import ModelEvaluator

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Training pipeline integration cho continuous improvement"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_preparator = DataPreparator()
        self.finetuner = FineTuner(config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
        self.evaluator = ModelEvaluator()
        
        # Training state
        self.is_training = False
        self.training_progress = {}
        self.last_training_time = None
        self.model_version = "1.0.0"
        
        # Conversation collection
        self.conversation_buffer = []
        self.buffer_size = config.get("conversation_buffer_size", 100)
        self.auto_retrain_threshold = config.get("auto_retrain_threshold", 1000)
        
        # Training schedule
        self.auto_retrain_enabled = config.get("auto_retrain_enabled", True)
        self.retrain_interval = config.get("retrain_interval", 86400)  # 24 hours
        self.last_retrain_time = 0
        
    def collect_conversation(self, conversation: Dict[str, Any]) -> None:
        """Thu thập conversation data cho training"""
        try:
            # Normalize conversation
            normalized_conv = self.data_preparator.normalize_conversation(conversation)
            
            # Add to buffer
            self.conversation_buffer.append({
                **normalized_conv,
                "collected_at": datetime.now().isoformat(),
                "source": "live_conversation"
            })
            
            # Auto-save if buffer is full
            if len(self.conversation_buffer) >= self.buffer_size:
                self._save_conversation_buffer()
            
            # Check if auto-retrain is needed
            if (self.auto_retrain_enabled and 
                len(self.conversation_buffer) >= self.auto_retrain_threshold):
                asyncio.create_task(self._auto_retrain())
                
        except Exception as e:
            logger.error(f"Error collecting conversation: {e}")
    
    def _save_conversation_buffer(self) -> None:
        """Lưu conversation buffer vào file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversations_{timestamp}.json"
            filepath = Path("training/dataset") / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_buffer, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(self.conversation_buffer)} conversations to {filepath}")
            self.conversation_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error saving conversation buffer: {e}")
    
    async def _auto_retrain(self) -> None:
        """Tự động retrain model khi có đủ data"""
        if self.is_training:
            logger.info("Training already in progress, skipping auto-retrain")
            return
        
        try:
            logger.info("Starting auto-retrain process...")
            await self.start_training_pipeline(
                data_source="collected_conversations",
                auto_mode=True
            )
        except Exception as e:
            logger.error(f"Auto-retrain failed: {e}")
    
    async def start_training_pipeline(
        self, 
        data_source: str = "dataset",
        auto_mode: bool = False
    ) -> Dict[str, Any]:
        """Bắt đầu training pipeline hoàn chỉnh"""
        if self.is_training:
            return {"status": "error", "message": "Training already in progress"}
        
        self.is_training = True
        self.training_progress = {
            "status": "running",
            "current_step": "preparing_data",
            "progress": 0,
            "start_time": time.time(),
            "steps": {
                "data_preparation": {"status": "pending", "progress": 0},
                "fine_tuning": {"status": "pending", "progress": 0},
                "evaluation": {"status": "pending", "progress": 0},
                "deployment": {"status": "pending", "progress": 0}
            }
        }
        
        try:
            # Step 1: Data Preparation
            logger.info("Step 1: Preparing training data...")
            self.training_progress["steps"]["data_preparation"]["status"] = "running"
            self.training_progress["progress"] = 10
            
            if data_source == "collected_conversations":
                # Use collected conversations
                conversations = self.conversation_buffer.copy()
                if not conversations:
                    # Load from saved files
                    conversation_files = list(Path("training/dataset").glob("conversations_*.json"))
                    if conversation_files:
                        latest_file = max(conversation_files, key=lambda x: x.stat().st_mtime)
                        with open(latest_file, 'r', encoding='utf-8') as f:
                            conversations = json.load(f)
            else:
                # Use existing dataset
                conversations = self.data_preparator.load_dataset("training/dataset/dataset.json")
            
            # Prepare training data
            training_data = self.data_preparator.prepare_conversation_data(conversations)
            train_data, val_data, test_data = self.data_preparator.create_training_splits(training_data)
            
            # Save prepared data
            self.data_preparator.save_training_data(train_data, "train_conversations.json")
            self.data_preparator.save_training_data(val_data, "val_conversations.json")
            self.data_preparator.save_training_data(test_data, "test_conversations.json")
            
            self.training_progress["steps"]["data_preparation"]["status"] = "completed"
            self.training_progress["steps"]["data_preparation"]["progress"] = 100
            self.training_progress["progress"] = 25
            
            # Step 2: Fine-tuning
            logger.info("Step 2: Fine-tuning model...")
            self.training_progress["steps"]["fine_tuning"]["status"] = "running"
            self.training_progress["progress"] = 30
            
            # Start fine-tuning in background thread
            training_thread = threading.Thread(
                target=self._run_fine_tuning,
                args=(train_data, val_data)
            )
            training_thread.start()
            
            # Wait for training to complete
            training_thread.join()
            
            self.training_progress["steps"]["fine_tuning"]["status"] = "completed"
            self.training_progress["steps"]["fine_tuning"]["progress"] = 100
            self.training_progress["progress"] = 60
            
            # Step 3: Evaluation
            logger.info("Step 3: Evaluating model...")
            self.training_progress["steps"]["evaluation"]["status"] = "running"
            self.training_progress["progress"] = 70
            
            # Load and evaluate model
            model_path = "training/checkpoints"
            if self.evaluator.load_model(model_path):
                evaluation_results = self.evaluator.evaluate_model(test_data)
                
                # Save evaluation results
                self.evaluator.save_evaluation_results(
                    evaluation_results, 
                    "training/evaluation_results.json"
                )
                
                # Generate report
                report = self.evaluator.generate_evaluation_report(evaluation_results)
                with open("training/evaluation_report.md", 'w', encoding='utf-8') as f:
                    f.write(report)
                
                self.training_progress["steps"]["evaluation"]["status"] = "completed"
                self.training_progress["steps"]["evaluation"]["progress"] = 100
                self.training_progress["progress"] = 85
            else:
                logger.warning("Failed to load model for evaluation")
                self.training_progress["steps"]["evaluation"]["status"] = "failed"
            
            # Step 4: Deployment
            logger.info("Step 4: Deploying model...")
            self.training_progress["steps"]["deployment"]["status"] = "running"
            self.training_progress["progress"] = 90
            
            # Update model version
            self.model_version = f"1.{int(time.time())}"
            self.last_training_time = time.time()
            
            # Save training metadata
            training_metadata = {
                "model_version": self.model_version,
                "training_time": self.last_training_time,
                "data_source": data_source,
                "auto_mode": auto_mode,
                "conversations_used": len(conversations),
                "training_data_size": len(training_data),
                "evaluation_results": evaluation_results if 'evaluation_results' in locals() else {}
            }
            
            with open("training/training_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(training_metadata, f, indent=2, ensure_ascii=False)
            
            self.training_progress["steps"]["deployment"]["status"] = "completed"
            self.training_progress["steps"]["deployment"]["progress"] = 100
            self.training_progress["progress"] = 100
            self.training_progress["status"] = "completed"
            
            logger.info("Training pipeline completed successfully")
            
            return {
                "status": "success",
                "message": "Training pipeline completed",
                "model_version": self.model_version,
                "training_metadata": training_metadata
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            self.training_progress["status"] = "failed"
            self.training_progress["error"] = str(e)
            return {
                "status": "error",
                "message": f"Training pipeline failed: {e}"
            }
        finally:
            self.is_training = False
    
    def _run_fine_tuning(self, train_data: List[Dict[str, Any]], val_data: List[Dict[str, Any]]) -> None:
        """Chạy fine-tuning trong background thread"""
        try:
            # Save data to files
            self.finetuner.save_training_data(
                self.finetuner.create_training_dataset(train_data), 
                "train_dataset.jsonl"
            )
            
            if val_data:
                self.finetuner.save_training_data(
                    self.finetuner.create_training_dataset(val_data), 
                    "val_dataset.jsonl"
                )
            
            # Start fine-tuning
            config = self.finetuner.start_training(
                "training/checkpoints/train_dataset.jsonl",
                "training/checkpoints/val_dataset.jsonl" if val_data else None
            )
            
            logger.info("Fine-tuning completed")
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            raise
    
    def get_training_status(self) -> Dict[str, Any]:
        """Lấy trạng thái training hiện tại"""
        return {
            "is_training": self.is_training,
            "training_progress": self.training_progress,
            "model_version": self.model_version,
            "last_training_time": self.last_training_time,
            "conversation_buffer_size": len(self.conversation_buffer),
            "auto_retrain_enabled": self.auto_retrain_enabled
        }
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Lấy lịch sử training"""
        try:
            metadata_file = Path("training/training_metadata.json")
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return [json.load(f)]
            return []
        except Exception as e:
            logger.error(f"Error loading training history: {e}")
            return []
    
    def enable_auto_retrain(self, enabled: bool = True) -> None:
        """Bật/tắt auto-retrain"""
        self.auto_retrain_enabled = enabled
        logger.info(f"Auto-retrain {'enabled' if enabled else 'disabled'}")
    
    def set_retrain_interval(self, interval_seconds: int) -> None:
        """Thiết lập interval cho auto-retrain"""
        self.retrain_interval = interval_seconds
        logger.info(f"Retrain interval set to {interval_seconds} seconds")
    
    def clear_conversation_buffer(self) -> None:
        """Xóa conversation buffer"""
        self.conversation_buffer.clear()
        logger.info("Conversation buffer cleared")
    
    async def scheduled_retrain(self) -> None:
        """Scheduled retrain task"""
        while True:
            try:
                if (self.auto_retrain_enabled and 
                    time.time() - self.last_retrain_time > self.retrain_interval):
                    
                    logger.info("Starting scheduled retrain...")
                    await self._auto_retrain()
                    self.last_retrain_time = time.time()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in scheduled retrain: {e}")
                await asyncio.sleep(3600)

# Global training pipeline instance
training_pipeline = None

def get_training_pipeline(config: Dict[str, Any] = None) -> TrainingPipeline:
    """Get global training pipeline instance"""
    global training_pipeline
    if training_pipeline is None:
        training_pipeline = TrainingPipeline(config or {})
    return training_pipeline
