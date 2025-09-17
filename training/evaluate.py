"""
Evaluation Script for InteractionModel
Enhanced for e-commerce domain evaluation
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.metrics = {}
        self.model = None
        self.tokenizer = None
        
        # Evaluation metrics
        self.evaluation_metrics = {
            "accuracy": 0.0,
            "bleu_score": 0.0,
            "rouge_scores": {},
            "intent_accuracy": 0.0,
            "response_time": 0.0,
            "perplexity": 0.0,
            "semantic_similarity": 0.0
        }
    
    def load_model(self, model_path: str):
        """Load the fine-tuned model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """Generate response using the model"""
        if not self.model or not self.tokenizer:
            return "Model not loaded"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the response
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def calculate_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            smoothie = SmoothingFunction().method4
            bleu_scores = []
            
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                
                if len(ref_tokens) > 0:
                    bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
                    bleu_scores.append(bleu)
            
            return np.mean(bleu_scores) if bleu_scores else 0.0
            
        except ImportError:
            logger.warning("NLTK not available for BLEU calculation")
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def calculate_rouge_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE score"""
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                for metric in rouge_scores:
                    rouge_scores[metric].append(scores[metric].fmeasure)
            
            return {
                'rouge-1': np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0.0,
                'rouge-2': np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0.0,
                'rouge-l': np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0.0
            }
            
        except ImportError:
            logger.warning("rouge_score not available for ROUGE calculation")
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
        except Exception as e:
            logger.error(f"Error calculating ROUGE score: {e}")
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    
    def calculate_intent_accuracy(self, predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> float:
        """Calculate intent classification accuracy"""
        correct = 0
        total = 0
        
        for pred, ref in zip(predictions, references):
            pred_intent = pred.get("intent", "unknown")
            ref_intent = ref.get("intent", "unknown")
            
            if pred_intent == ref_intent:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def calculate_semantic_similarity(self, predictions: List[str], references: List[str]) -> float:
        """Calculate semantic similarity using sentence transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            pred_embeddings = model.encode(predictions)
            ref_embeddings = model.encode(references)
            
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
                similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.0
            
        except ImportError:
            logger.warning("sentence-transformers not available for semantic similarity")
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_perplexity(self, predictions: List[str]) -> float:
        """Calculate perplexity of generated text"""
        if not self.model or not self.tokenizer:
            return 0.0
        
        try:
            total_loss = 0
            total_tokens = 0
            
            for text in predictions:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    total_loss += loss.item() * inputs["input_ids"].size(1)
                    total_tokens += inputs["input_ids"].size(1)
            
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            perplexity = np.exp(avg_loss)
            
            return perplexity
            
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return 0.0
    
    def evaluate_model(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate model performance comprehensively"""
        logger.info("Starting comprehensive model evaluation...")
        
        predictions = []
        references = []
        prediction_metadata = []
        reference_metadata = []
        
        # Generate predictions
        start_time = time.time()
        for i, item in enumerate(test_data):
            logger.info(f"Evaluating sample {i+1}/{len(test_data)}")
            
            # Create prompt
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            
            if input_text:
                prompt = f"### Hệ thống:\nBạn là trợ lý AI chuyên về thương mại điện tử.\n\n### Ngữ cảnh:\n{input_text}\n\n### Khách hàng:\n{instruction}\n\n### Trợ lý:"
            else:
                prompt = f"### Hệ thống:\nBạn là trợ lý AI chuyên về thương mại điện tử.\n\n### Khách hàng:\n{instruction}\n\n### Trợ lý:"
            
            # Generate response
            response = self.generate_response(prompt)
            predictions.append(response)
            
            # Store metadata
            prediction_metadata.append({
                "intent": item.get("intent", "unknown"),
                "entities": item.get("entities", []),
                "response_time": time.time() - start_time
            })
            
            # Store reference
            references.append(item.get("output", ""))
            reference_metadata.append({
                "intent": item.get("intent", "unknown"),
                "entities": item.get("entities", [])
            })
        
        total_time = time.time() - start_time
        avg_response_time = total_time / len(test_data) if test_data else 0
        
        # Calculate metrics
        logger.info("Calculating evaluation metrics...")
        
        bleu_score = self.calculate_bleu_score(predictions, references)
        rouge_scores = self.calculate_rouge_score(predictions, references)
        intent_accuracy = self.calculate_intent_accuracy(prediction_metadata, reference_metadata)
        semantic_similarity = self.calculate_semantic_similarity(predictions, references)
        perplexity = self.calculate_perplexity(predictions)
        
        # Calculate overall accuracy (simple word overlap)
        accuracy_scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            if ref_words:
                overlap = len(pred_words.intersection(ref_words)) / len(ref_words)
                accuracy_scores.append(overlap)
        
        accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
        
        results = {
            "accuracy": accuracy,
            "bleu_score": bleu_score,
            "rouge_scores": rouge_scores,
            "intent_accuracy": intent_accuracy,
            "response_time": avg_response_time,
            "perplexity": perplexity,
            "semantic_similarity": semantic_similarity,
            "total_samples": len(test_data),
            "total_time": total_time
        }
        
        self.evaluation_metrics = results
        
        logger.info(f"Evaluation completed. Results: {results}")
        return results
    
    def save_evaluation_results(self, results: Dict[str, float], output_path: str):
        """Save evaluation results to file"""
        output_data = {
            "evaluation_results": results,
            "timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "evaluation_metadata": {
                "evaluator_version": "1.0.0",
                "metrics_calculated": list(results.keys())
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def generate_evaluation_report(self, results: Dict[str, float]) -> str:
        """Generate a human-readable evaluation report"""
        report = f"""
# Model Evaluation Report

## Overview
- **Model Path**: {self.model_path or 'Not specified'}
- **Total Samples**: {results.get('total_samples', 0)}
- **Evaluation Time**: {results.get('total_time', 0):.2f} seconds

## Performance Metrics

### Text Quality
- **BLEU Score**: {results.get('bleu_score', 0):.4f}
- **ROUGE-1**: {results.get('rouge_scores', {}).get('rouge-1', 0):.4f}
- **ROUGE-2**: {results.get('rouge_scores', {}).get('rouge-2', 0):.4f}
- **ROUGE-L**: {results.get('rouge_scores', {}).get('rouge-l', 0):.4f}
- **Semantic Similarity**: {results.get('semantic_similarity', 0):.4f}

### Task Performance
- **Overall Accuracy**: {results.get('accuracy', 0):.4f}
- **Intent Accuracy**: {results.get('intent_accuracy', 0):.4f}

### Efficiency
- **Average Response Time**: {results.get('response_time', 0):.4f} seconds
- **Perplexity**: {results.get('perplexity', 0):.4f}

## Interpretation
- BLEU Score > 0.3: Good translation quality
- ROUGE-1 > 0.4: Good content overlap
- Intent Accuracy > 0.8: Good intent classification
- Response Time < 2.0s: Good response speed
- Perplexity < 10: Good language model quality
"""
        return report

def main():
    # Check for model and test data
    model_path = "training/checkpoints"
    test_data_path = "training/dataset/test_conversations.json"
    
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        logger.info("Please run finetune.py first to train a model")
        return
    
    if not Path(test_data_path).exists():
        logger.error(f"Test data not found at {test_data_path}")
        logger.info("Please run prepare_data.py first to generate test data")
        return
    
    # Load test data
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Create evaluator and load model
    evaluator = ModelEvaluator(model_path)
    
    if not evaluator.load_model(model_path):
        logger.error("Failed to load model")
        return
    
    # Evaluate model
    try:
        results = evaluator.evaluate_model(test_data)
        
        # Save results
        output_path = "training/evaluation_results.json"
        evaluator.save_evaluation_results(results, output_path)
        
        # Generate and save report
        report = evaluator.generate_evaluation_report(results)
        report_path = "training/evaluation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Evaluation completed. Results saved to {output_path}")
        logger.info(f"Report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Accuracy: {results.get('accuracy', 0):.4f}")
        print(f"BLEU Score: {results.get('bleu_score', 0):.4f}")
        print(f"ROUGE-1: {results.get('rouge_scores', {}).get('rouge-1', 0):.4f}")
        print(f"Intent Accuracy: {results.get('intent_accuracy', 0):.4f}")
        print(f"Response Time: {results.get('response_time', 0):.4f}s")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
