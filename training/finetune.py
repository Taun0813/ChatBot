"""
Fine-tuning Script for InteractionModel
Enhanced for e-commerce domain fine-tuning
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
from datetime import datetime

logger = logging.getLogger(__name__)

class FineTuner:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.output_dir = Path("training/checkpoints")
        self.output_dir.mkdir(exist_ok=True)
        
        # Training configuration
        self.training_config = {
            "model_name": model_name,
            "output_dir": str(self.output_dir),
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "max_seq_length": 512,
            "warmup_steps": 100,
            "logging_steps": 25,
            "save_steps": 500,
            "eval_steps": 500,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "fp16": True,
            "dataloader_num_workers": 4,
            "remove_unused_columns": False,
            "push_to_hub": False,
            "report_to": "none"
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
            "max_conversation_length": 512,
            "response_max_length": 256
        }
    
    def prepare_training_config(self) -> Dict[str, Any]:
        """Prepare training configuration"""
        return self.training_config.copy()
    
    def create_training_dataset(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Create training dataset in the format expected by transformers"""
        dataset = []
        
        for item in data:
            # Format for instruction tuning
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            input_text = item.get("input", "")
            
            # Create prompt template for e-commerce conversations
            if input_text:
                prompt = f"### Hệ thống:\nBạn là trợ lý AI chuyên về thương mại điện tử, giúp khách hàng tìm sản phẩm, tư vấn mua hàng và hỗ trợ đơn hàng.\n\n### Ngữ cảnh:\n{input_text}\n\n### Khách hàng:\n{instruction}\n\n### Trợ lý:"
            else:
                prompt = f"### Hệ thống:\nBạn là trợ lý AI chuyên về thương mại điện tử, giúp khách hàng tìm sản phẩm, tư vấn mua hàng và hỗ trợ đơn hàng.\n\n### Khách hàng:\n{instruction}\n\n### Trợ lý:"
            
            dataset.append({
                "prompt": prompt,
                "completion": output,
                "intent": item.get("intent", "general_chat"),
                "entities": item.get("entities", []),
                "metadata": item.get("metadata", {})
            })
        
        return dataset
    
    def save_training_data(self, data: List[Dict[str, str]], filename: str):
        """Save training data in JSONL format for fine-tuning"""
        file_path = self.output_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(data)} training samples to {file_path}")
    
    def load_training_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load training data from file"""
        data = []
        
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        
        return data
    
    def start_training(self, data_path: str, validation_data_path: Optional[str] = None):
        """Start fine-tuning process"""
        logger.info("Starting fine-tuning process...")
        
        try:
            # Load training data
            training_data = self.load_training_data(data_path)
            logger.info(f"Loaded {len(training_data)} training samples")
            
            # Create training dataset
            train_dataset = self.create_training_dataset(training_data)
            self.save_training_data(train_dataset, "train_dataset.jsonl")
            
            # Load validation data if provided
            val_dataset = []
            if validation_data_path and Path(validation_data_path).exists():
                val_data = self.load_training_data(validation_data_path)
                val_dataset = self.create_training_dataset(val_data)
                self.save_training_data(val_dataset, "val_dataset.jsonl")
                logger.info(f"Loaded {len(val_dataset)} validation samples")
            
            # Prepare config
            config = self.prepare_training_config()
            config["train_file"] = str(self.output_dir / "train_dataset.jsonl")
            if val_dataset:
                config["validation_file"] = str(self.output_dir / "val_dataset.jsonl")
            
            # Save configuration
            config_path = self.output_dir / "training_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Training config saved to {config_path}")
            
            # Start actual fine-tuning
            self._run_fine_tuning(config)
            
            return config
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise
    
    def _run_fine_tuning(self, config: Dict[str, Any]):
        """Run the actual fine-tuning process"""
        try:
            # Check if transformers and peft are available
            try:
                from transformers import (
                    AutoTokenizer, AutoModelForCausalLM, 
                    TrainingArguments, Trainer, DataCollatorForLanguageModeling
                )
                from peft import LoraConfig, get_peft_model, TaskType
                from datasets import Dataset
            except ImportError as e:
                logger.error(f"Required packages not available: {e}")
                logger.info("Please install: pip install transformers peft datasets accelerate")
                return
            
            logger.info("Starting fine-tuning with transformers and PEFT...")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.float16 if config.get("fp16", False) else torch.float32,
                device_map="auto"
            )
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config["lora_r"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Load datasets
            train_dataset = self._load_dataset(config["train_file"], tokenizer)
            val_dataset = None
            if "validation_file" in config:
                val_dataset = self._load_dataset(config["validation_file"], tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=config["output_dir"],
                num_train_epochs=config["num_train_epochs"],
                per_device_train_batch_size=config["per_device_train_batch_size"],
                gradient_accumulation_steps=config["gradient_accumulation_steps"],
                learning_rate=config["learning_rate"],
                warmup_steps=config["warmup_steps"],
                logging_steps=config["logging_steps"],
                save_steps=config["save_steps"],
                eval_steps=config["eval_steps"],
                evaluation_strategy="steps" if val_dataset else "no",
                save_strategy="steps",
                load_best_model_at_end=True if val_dataset else False,
                fp16=config.get("fp16", False),
                dataloader_num_workers=config["dataloader_num_workers"],
                remove_unused_columns=config["remove_unused_columns"],
                report_to=config["report_to"],
                push_to_hub=config["push_to_hub"]
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer
            )
            
            # Start training
            logger.info("Starting training...")
            trainer.train()
            
            # Save model
            trainer.save_model()
            tokenizer.save_pretrained(config["output_dir"])
            
            logger.info(f"Fine-tuning completed. Model saved to {config['output_dir']}")
            
        except Exception as e:
            logger.error(f"Error in fine-tuning process: {e}")
            raise
    
    def _load_dataset(self, file_path: str, tokenizer) -> Dataset:
        """Load and tokenize dataset"""
        from datasets import Dataset
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        
        def tokenize_function(examples):
            # Combine prompt and completion
            texts = [f"{item['prompt']} {item['completion']}" for item in examples]
            return tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        
        dataset = Dataset.from_list(data)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset

def main():
    finetuner = FineTuner()
    
    # Check for training data
    train_data_path = "training/dataset/train_conversations.json"
    val_data_path = "training/dataset/val_conversations.json"
    
    if not Path(train_data_path).exists():
        logger.error(f"Training data not found at {train_data_path}")
        logger.info("Please run prepare_data.py first to generate training data")
        return
    
    # Start training
    try:
        config = finetuner.start_training(
            train_data_path, 
            val_data_path if Path(val_data_path).exists() else None
        )
        
        logger.info("Fine-tuning completed successfully")
        logger.info(f"Model saved to: {config['output_dir']}")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise

if __name__ == "__main__":
    main()
