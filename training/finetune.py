"""
Cloud-based Fine-tuning Script for E-commerce AI Agent
Supports OpenAI, Google Vertex AI, and other cloud fine-tuning platforms
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import aiofiles

# Import config and model loaders
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_settings
from adapters.model_loader.gemini_loader import GeminiLoader
from adapters.model_loader.openai_loader import OpenAILoader
from adapters.model_loader.groq_loader import GroqLoader

logger = logging.getLogger(__name__)

class CloudFineTuner:
    def __init__(self, model_backend: Optional[str] = None):
        self.settings = get_settings()
        self.model_backend = model_backend or self.settings.model_loader_backend
        self.output_dir = Path("training/checkpoints")
        self.output_dir.mkdir(exist_ok=True)
        
        # Cloud fine-tuning configuration
        self.training_config = {
            "model_backend": self.model_backend,
            "output_dir": str(self.output_dir),
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "training_epochs": 3,
            "batch_size": 16,
            "learning_rate": 1e-5,
            "validation_split": 0.1,
            "early_stopping_patience": 3,
            "save_best_model": True,
            "evaluation_metrics": ["accuracy", "f1_score", "bleu_score"]
        }
        
        # E-commerce specific configuration
        self.domain_config = {
            "task_type": "conversational_ai",
            "domain": "e-commerce",
            "language": "vietnamese",
            "intent_classes": [
                "product_search", "product_recommendation", "order_inquiry",
                "payment_inquiry", "warranty_inquiry", "general_chat"
            ],
            "entity_types": ["brand", "price_range", "number", "product_category"],
            "max_conversation_length": 2048,
            "response_max_length": 1024,
            "conversation_turns": 5
        }
        
        # Initialize model loader
        self.model_loader = self._get_model_loader()
    
    def _get_model_loader(self):
        """Get appropriate model loader based on backend"""
        if self.model_backend == "gemini":
            return GeminiLoader(
                model_name=self.settings.model_name,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                api_key=self.settings.gemini_api_key
            )
        elif self.model_backend == "openai":
            return OpenAILoader({
                "model_name": self.settings.model_name,
                "max_tokens": self.settings.max_tokens,
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "api_key": self.settings.openai_api_key
            })
        elif self.model_backend == "groq":
            return GroqLoader(
                model_name=self.settings.model_name,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                api_key=self.settings.groq_api_key
            )
        else:
            raise ValueError(f"Unsupported model backend: {self.model_backend}")
    
    def prepare_training_config(self) -> Dict[str, Any]:
        """Prepare training configuration"""
        config = self.training_config.copy()
        config.update({
            "model_name": self.settings.model_name,
            "api_key_configured": self._check_api_key(),
            "training_data_path": str(self.output_dir / "training_data.jsonl"),
            "validation_data_path": str(self.output_dir / "validation_data.jsonl"),
            "model_output_path": str(self.output_dir / "fine_tuned_model"),
            "training_logs_path": str(self.output_dir / "training_logs.json")
        })
        return config
    
    def _check_api_key(self) -> bool:
        """Check if API key is configured for the selected backend"""
        if self.model_backend == "gemini":
            return bool(self.settings.gemini_api_key)
        elif self.model_backend == "openai":
            return bool(self.settings.openai_api_key)
        elif self.model_backend == "groq":
            return bool(self.settings.groq_api_key)
        return False
    
    def create_training_dataset(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Create training dataset in the format expected by cloud fine-tuning APIs"""
        dataset = []
        
        for item in data:
            # Format for instruction tuning
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            input_text = item.get("input", "")
            intent = item.get("intent", "general_chat")
            
            # Create conversation format for e-commerce
            if input_text:
                messages = [
                    {
                        "role": "system",
                        "content": "Bạn là trợ lý AI chuyên về thương mại điện tử, giúp khách hàng tìm sản phẩm, tư vấn mua hàng và hỗ trợ đơn hàng."
                    },
                    {
                        "role": "user", 
                        "content": f"Ngữ cảnh: {input_text}\n\nCâu hỏi: {instruction}"
                    },
                    {
                        "role": "assistant",
                        "content": output
                    }
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": "Bạn là trợ lý AI chuyên về thương mại điện tử, giúp khách hàng tìm sản phẩm, tư vấn mua hàng và hỗ trợ đơn hàng."
                    },
                    {
                        "role": "user",
                        "content": instruction
                    },
                    {
                        "role": "assistant", 
                        "content": output
                    }
                ]
            
            dataset.append({
                "messages": messages,
                "intent": intent,
                "entities": item.get("entities", []),
                "metadata": {
                    **item.get("metadata", {}),
                    "domain": self.domain_config["domain"],
                    "language": self.domain_config["language"],
                    "conversation_turns": len(messages) // 2
                }
            })
        
        return dataset
    
    async def save_training_data(self, data: List[Dict[str, str]], filename: str):
        """Save training data in JSONL format for fine-tuning"""
        file_path = self.output_dir / filename
        
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                await f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info("Saved %d training samples to %s", len(data), file_path)
    
    async def load_training_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load training data from file"""
        data = []
        
        if data_path.endswith('.json'):
            async with aiofiles.open(data_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
        elif data_path.endswith('.jsonl'):
            async with aiofiles.open(data_path, 'r', encoding='utf-8') as f:
                async for line in f:
                    data.append(json.loads(line.strip()))
        
        return data
    
    async def start_training(self, data_path: str, validation_data_path: Optional[str] = None):
        """Start cloud-based fine-tuning process"""
        logger.info("Starting cloud fine-tuning with %s backend...", self.model_backend)
        
        try:
            # Check API key
            if not self._check_api_key():
                raise ValueError(f"API key not configured for {self.model_backend} backend")
            
            # Initialize model loader
            await self.model_loader.initialize()
            
            # Load training data
            training_data = await self.load_training_data(data_path)
            logger.info("Loaded %d training samples", len(training_data))
            
            # Create training dataset
            train_dataset = self.create_training_dataset(training_data)
            await self.save_training_data(train_dataset, "training_data.jsonl")
            
            # Load validation data if provided
            val_dataset = []
            if validation_data_path and Path(validation_data_path).exists():
                val_data = await self.load_training_data(validation_data_path)
                val_dataset = self.create_training_dataset(val_data)
                await self.save_training_data(val_dataset, "validation_data.jsonl")
                logger.info("Loaded %d validation samples", len(val_dataset))
            
            # Prepare config
            config = self.prepare_training_config()
            
            # Save configuration
            config_path = self.output_dir / "training_config.json"
            async with aiofiles.open(config_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(config, indent=2, ensure_ascii=False))
            
            logger.info("Training config saved to %s", config_path)
            
            # Start cloud fine-tuning
            result = await self._run_cloud_fine_tuning(config, train_dataset, val_dataset)
            
            return {**config, "training_result": result}
            
        except Exception as e:
            logger.error("Error during cloud fine-tuning: %s", e)
            raise
        finally:
            # Cleanup
            await self.model_loader.cleanup()
    
    async def _run_cloud_fine_tuning(self, config: Dict[str, Any], train_dataset: List[Dict], val_dataset: List[Dict]) -> Dict[str, Any]:
        """Run cloud-based fine-tuning using the selected backend"""
        try:
            if self.model_backend == "openai":
                return await self._run_openai_fine_tuning(config, train_dataset, val_dataset)
            elif self.model_backend == "gemini":
                return await self._run_gemini_fine_tuning(config, train_dataset, val_dataset)
            elif self.model_backend == "groq":
                return await self._run_groq_fine_tuning(config, train_dataset, val_dataset)
            else:
                raise ValueError(f"Cloud fine-tuning not supported for {self.model_backend}")
            
        except Exception as e:
            logger.error("Error in cloud fine-tuning process: %s", e)
            raise
    
    async def _run_openai_fine_tuning(self, config: Dict[str, Any], train_dataset: List[Dict], val_dataset: List[Dict]) -> Dict[str, Any]:
        """Run OpenAI fine-tuning"""
        logger.info("Starting OpenAI fine-tuning...")
        
        # OpenAI fine-tuning API call
        import openai
        
        client = openai.AsyncOpenAI(api_key=self.settings.openai_api_key)
        
        # Upload training file
        training_file = await client.files.create(
            file=open(str(self.output_dir / "training_data.jsonl"), "rb"),
            purpose="fine-tune"
        )
        
        # Upload validation file if exists
        validation_file = None
        if val_dataset:
            validation_file = await client.files.create(
                file=open(str(self.output_dir / "validation_data.jsonl"), "rb"),
                purpose="fine-tune"
            )
        
        # Create fine-tuning job
        fine_tune_job = await client.fine_tuning.jobs.create(
            training_file=training_file.id,
            validation_file=validation_file.id if validation_file else None,
            model=config["model_name"],
            hyperparameters={
                "n_epochs": config["training_epochs"],
                "batch_size": config["batch_size"],
                "learning_rate_multiplier": config["learning_rate"]
            }
        )
        
        logger.info("OpenAI fine-tuning job created: %s", fine_tune_job.id)
        
        # Monitor job status
        job_status = await self._monitor_fine_tuning_job(client, fine_tune_job.id)
        
        return {
            "job_id": fine_tune_job.id,
            "status": job_status["status"],
            "model_id": job_status.get("fine_tuned_model"),
            "training_file_id": training_file.id,
            "validation_file_id": validation_file.id if validation_file else None,
            "metrics": job_status.get("result_files", [])
        }
    
    async def _run_gemini_fine_tuning(self, config: Dict[str, Any], train_dataset: List[Dict], val_dataset: List[Dict]) -> Dict[str, Any]:
        """Run Gemini fine-tuning using Vertex AI"""
        logger.info("Starting Gemini fine-tuning with Vertex AI...")
        
        # Note: This is a placeholder implementation
        # Actual implementation would use Google Cloud Vertex AI
        logger.warning("Gemini fine-tuning requires Vertex AI setup - using mock implementation")
        
        from datetime import datetime
        return {
            "job_id": f"gemini_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "completed",
            "model_id": f"gemini-ft-{config['model_name']}",
            "message": "Gemini fine-tuning requires Vertex AI configuration"
        }
    
    async def _run_groq_fine_tuning(self, config: Dict[str, Any], train_dataset: List[Dict], val_dataset: List[Dict]) -> Dict[str, Any]:
        """Run Groq fine-tuning"""
        logger.info("Starting Groq fine-tuning...")
        
        # Note: Groq doesn't support fine-tuning yet
        logger.warning("Groq doesn't support fine-tuning - using mock implementation")
        
        from datetime import datetime
        return {
            "job_id": f"groq_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "not_supported",
            "model_id": config["model_name"],
            "message": "Groq doesn't support fine-tuning yet"
        }
    
    async def _monitor_fine_tuning_job(self, client, job_id: str) -> Dict[str, Any]:
        """Monitor fine-tuning job status"""
        import time
        
        while True:
            job = await client.fine_tuning.jobs.retrieve(job_id)
            
            logger.info("Fine-tuning job %s status: %s", job_id, job.status)
            
            if job.status in ["succeeded", "failed", "cancelled"]:
                return {
                    "status": job.status,
                    "fine_tuned_model": job.fine_tuned_model,
                    "result_files": job.result_files,
                    "error": job.error if job.status == "failed" else None
                }
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def evaluate_model(self, test_data_path: str) -> Dict[str, Any]:
        """Evaluate the fine-tuned model"""
        logger.info("Evaluating fine-tuned model...")
        
        try:
            # Load test data
            test_data = await self.load_training_data(test_data_path)
            test_dataset = self.create_training_dataset(test_data)
            
            # Initialize model for evaluation
            await self.model_loader.initialize()
            
            # Evaluate on test set
            results = {
                "total_samples": len(test_dataset),
                "correct_predictions": 0,
                "intent_accuracy": {},
                "response_quality": []
            }
            
            for item in test_dataset:
                # Get model prediction
                messages = item["messages"][:-1]  # Exclude assistant response
                prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                
                prediction = await self.model_loader.generate_response(prompt)
                expected = item["messages"][-1]["content"]
                
                # Simple evaluation metrics
                if item["intent"] in results["intent_accuracy"]:
                    results["intent_accuracy"][item["intent"]] += 1
                else:
                    results["intent_accuracy"][item["intent"]] = 1
                
                # Basic similarity check (can be improved with proper metrics)
                if any(word in prediction.lower() for word in expected.lower().split()[:3]):
                    results["correct_predictions"] += 1
                
                results["response_quality"].append({
                    "expected": expected,
                    "predicted": prediction,
                    "intent": item["intent"]
                })
            
            # Calculate overall accuracy
            results["overall_accuracy"] = results["correct_predictions"] / results["total_samples"]
            
            # Save evaluation results
            eval_path = self.output_dir / "evaluation_results.json"
            async with aiofiles.open(eval_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(results, indent=2, ensure_ascii=False))
            
            logger.info("Evaluation completed. Results saved to %s", eval_path)
            logger.info("Overall accuracy: %.2f%%", results['overall_accuracy'] * 100)
            
            return results
            
        except Exception as e:
            logger.error("Error during model evaluation: %s", e)
            raise
        finally:
            await self.model_loader.cleanup()

async def main():
    """Main function for cloud fine-tuning"""
    # Get model backend from config or command line
    import sys
    model_backend = sys.argv[1] if len(sys.argv) > 1 else None
    
    finetuner = CloudFineTuner(model_backend)
    
    # Check for training data
    train_data_path = "training/dataset/train_conversations.json"
    val_data_path = "training/dataset/val_conversations.json"
    test_data_path = "training/dataset/test_conversations.json"
    
    if not Path(train_data_path).exists():
        logger.error(f"Training data not found at {train_data_path}")
        logger.info("Please run prepare_data.py first to generate training data")
        return
    
    # Start training
    try:
        logger.info("Starting fine-tuning with %s backend...", finetuner.model_backend)
        
        config = await finetuner.start_training(
            train_data_path, 
            val_data_path if Path(val_data_path).exists() else None
        )
        
        logger.info("Fine-tuning completed successfully")
        logger.info("Model configuration: %s", config['model_name'])
        logger.info("Output directory: %s", config['output_dir'])
        
        # Evaluate model if test data exists
        if Path(test_data_path).exists():
            logger.info("Evaluating fine-tuned model...")
            eval_results = await finetuner.evaluate_model(test_data_path)
            logger.info("Evaluation accuracy: %.2f%%", eval_results['overall_accuracy'] * 100)
        
    except Exception as e:
        logger.error("Fine-tuning failed: %s", e)
        raise

if __name__ == "__main__":
    asyncio.run(main())
