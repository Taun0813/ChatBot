"""
Cloud-based Evaluation Script for E-commerce AI Agent
Enhanced for comprehensive model evaluation
"""

import logging
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from datetime import datetime

# Import config and model loaders
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_settings
from adapters.model_loader.gemini_loader import GeminiLoader
from adapters.model_loader.openai_loader import OpenAILoader
from adapters.model_loader.groq_loader import GroqLoader

logger = logging.getLogger(__name__)

class CloudModelEvaluator:
    def __init__(self, model_backend: Optional[str] = None):
        self.settings = get_settings()
        self.model_backend = model_backend or self.settings.model_loader_backend
        self.model_loader = None
        self.metrics = {}
        
        # Enhanced evaluation metrics for e-commerce
        self.evaluation_metrics = {
            "accuracy": 0.0,
            "bleu_score": 0.0,
            "rouge_scores": {},
            "intent_accuracy": 0.0,
            "response_time": 0.0,
            "semantic_similarity": 0.0,
            "customer_satisfaction": 0.0,
            "product_knowledge": 0.0,
            "conversation_flow": 0.0,
            "entity_extraction": 0.0,
            "price_accuracy": 0.0,
            "recommendation_quality": 0.0
        }
    
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
    
    async def initialize_model(self):
        """Initialize the cloud model"""
        try:
            self.model_loader = self._get_model_loader()
            await self.model_loader.initialize()
            logger.info(f"Cloud model {self.model_backend} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            return False
    
    async def generate_response(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate response using the cloud model"""
        if not self.model_loader:
            return "Model not initialized"
        
        try:
            response = await self.model_loader.generate_response(
                prompt=prompt,
                max_tokens=max_tokens,
                    temperature=0.7,
                top_p=0.9
                )
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
    
    def calculate_customer_satisfaction(self, predictions: List[str], references: List[str]) -> float:
        """Calculate customer satisfaction based on response quality"""
        satisfaction_keywords = [
            "cảm ơn", "vui lòng", "tôi hiểu", "tôi sẽ giúp", "không có gì",
            "rất vui", "hân hạnh", "xin lỗi", "thông cảm", "hỗ trợ"
        ]
        
        satisfaction_scores = []
        for pred in predictions:
            pred_lower = pred.lower()
            score = sum(1 for keyword in satisfaction_keywords if keyword in pred_lower)
            satisfaction_scores.append(min(score / len(satisfaction_keywords), 1.0))
        
        return np.mean(satisfaction_scores) if satisfaction_scores else 0.0
    
    def calculate_product_knowledge(self, predictions: List[str], test_data: List[Dict[str, Any]]) -> float:
        """Calculate product knowledge accuracy"""
        product_terms = [
            "iPhone", "Samsung", "laptop", "camera", "RAM", "ROM", "CPU", "GPU",
            "pin", "màn hình", "âm thanh", "wifi", "bluetooth", "5G", "4G"
        ]
        
        knowledge_scores = []
        for i, pred in enumerate(predictions):
            pred_lower = pred.lower()
            item = test_data[i]
            intent = item.get("intent", "")
            
            # Check if response contains relevant product information
            if intent in ["product_search", "product_recommendation"]:
                product_mentions = sum(1 for term in product_terms if term.lower() in pred_lower)
                # Check for price mentions
                price_mentions = len([word for word in pred.split() if word.replace('.', '').replace(',', '').isdigit() and len(word) >= 3])
                score = min((product_mentions + price_mentions) / 10, 1.0)
                knowledge_scores.append(score)
        
        return np.mean(knowledge_scores) if knowledge_scores else 0.0
    
    def calculate_conversation_flow(self, predictions: List[str]) -> float:
        """Calculate conversation flow quality"""
        flow_indicators = [
            "bạn có", "bạn muốn", "bạn cần", "bạn có thể", "bạn có ngân sách",
            "bạn ưu tiên", "bạn quan tâm", "bạn có lý do", "bạn có thể cho tôi biết"
        ]
        
        flow_scores = []
        for pred in predictions:
            pred_lower = pred.lower()
            # Check for question asking (good conversation flow)
            question_count = pred.count('?') + pred.count('không') + pred.count('gì')
            # Check for flow indicators
            flow_count = sum(1 for indicator in flow_indicators if indicator in pred_lower)
            score = min((question_count + flow_count) / 5, 1.0)
            flow_scores.append(score)
        
        return np.mean(flow_scores) if flow_scores else 0.0
    
    def calculate_entity_extraction(self, predictions: List[str], test_data: List[Dict[str, Any]]) -> float:
        """Calculate entity extraction accuracy"""
        entity_scores = []
        
        for i, pred in enumerate(predictions):
            item = test_data[i]
            expected_entities = item.get("entities", [])
            
            if not expected_entities:
                entity_scores.append(1.0)  # No entities to extract
                continue
            
            pred_lower = pred.lower()
            extracted_entities = 0
            
            for entity in expected_entities:
                entity_value = entity.get("value", "").lower()
                if entity_value in pred_lower:
                    extracted_entities += 1
            
            score = extracted_entities / len(expected_entities) if expected_entities else 1.0
            entity_scores.append(score)
        
        return np.mean(entity_scores) if entity_scores else 0.0
    
    def calculate_price_accuracy(self, predictions: List[str], test_data: List[Dict[str, Any]]) -> float:
        """Calculate price information accuracy"""
        price_scores = []
        
        for i, pred in enumerate(predictions):
            item = test_data[i]
            intent = item.get("intent", "")
            
            if intent in ["product_search", "product_recommendation"]:
                pred_lower = pred.lower()
                # Check for price mentions (Vietnamese format)
                price_patterns = [
                    r'\d+\.\d+\.\d+',  # 1.000.000
                    r'\d+\s*triệu',    # 20 triệu
                    r'\d+\s*VNĐ',      # 20000000 VNĐ
                    r'giá.*\d+',       # giá 20 triệu
                ]
                
                price_mentions = 0
                for pattern in price_patterns:
                    import re
                    if re.search(pattern, pred_lower):
                        price_mentions += 1
                
                score = min(price_mentions / 2, 1.0)  # At least 2 price mentions
                price_scores.append(score)
        
        return np.mean(price_scores) if price_scores else 0.0
    
    def calculate_recommendation_quality(self, predictions: List[str], test_data: List[Dict[str, Any]]) -> float:
        """Calculate recommendation quality"""
        recommendation_scores = []
        
        for i, pred in enumerate(predictions):
            item = test_data[i]
            intent = item.get("intent", "")
            
            if intent == "product_recommendation":
                pred_lower = pred.lower()
                # Check for recommendation indicators
                rec_indicators = [
                    "gợi ý", "khuyến nghị", "phù hợp", "tốt nhất", "nên chọn",
                    "có thể", "tôi gợi ý", "bạn nên", "phù hợp với", "tốt cho"
                ]
                
                rec_count = sum(1 for indicator in rec_indicators if indicator in pred_lower)
                # Check for multiple options
                option_count = pred.count('hoặc') + pred.count(',') + pred.count(';')
                score = min((rec_count + option_count) / 5, 1.0)
                recommendation_scores.append(score)
        
        return np.mean(recommendation_scores) if recommendation_scores else 0.0
    
    async def evaluate_model(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate model performance comprehensively with e-commerce metrics"""
        logger.info("Starting comprehensive e-commerce model evaluation...")
        
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
                prompt = f"### Hệ thống:\nBạn là trợ lý AI chuyên về thương mại điện tử, giúp khách hàng tìm sản phẩm, tư vấn mua hàng và hỗ trợ đơn hàng.\n\n### Ngữ cảnh:\n{input_text}\n\n### Khách hàng:\n{instruction}\n\n### Trợ lý:"
            else:
                prompt = f"### Hệ thống:\nBạn là trợ lý AI chuyên về thương mại điện tử, giúp khách hàng tìm sản phẩm, tư vấn mua hàng và hỗ trợ đơn hàng.\n\n### Khách hàng:\n{instruction}\n\n### Trợ lý:"
            
            # Generate response
            response_start = time.time()
            response = await self.generate_response(prompt)
            response_time = time.time() - response_start
            
            predictions.append(response)
            
            # Store metadata
            prediction_metadata.append({
                "intent": item.get("intent", "unknown"),
                "entities": item.get("entities", []),
                "response_time": response_time
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
        
        # Basic metrics
        bleu_score = self.calculate_bleu_score(predictions, references)
        rouge_scores = self.calculate_rouge_score(predictions, references)
        intent_accuracy = self.calculate_intent_accuracy(prediction_metadata, reference_metadata)
        semantic_similarity = self.calculate_semantic_similarity(predictions, references)
        
        # E-commerce specific metrics
        customer_satisfaction = self.calculate_customer_satisfaction(predictions, references)
        product_knowledge = self.calculate_product_knowledge(predictions, test_data)
        conversation_flow = self.calculate_conversation_flow(predictions)
        entity_extraction = self.calculate_entity_extraction(predictions, test_data)
        price_accuracy = self.calculate_price_accuracy(predictions, test_data)
        recommendation_quality = self.calculate_recommendation_quality(predictions, test_data)
        
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
            "semantic_similarity": semantic_similarity,
            "customer_satisfaction": customer_satisfaction,
            "product_knowledge": product_knowledge,
            "conversation_flow": conversation_flow,
            "entity_extraction": entity_extraction,
            "price_accuracy": price_accuracy,
            "recommendation_quality": recommendation_quality,
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
        """Generate a human-readable evaluation report for e-commerce AI agent"""
        report = f"""
# E-commerce AI Agent Evaluation Report

## Overview
- **Model Backend**: {self.model_backend}
- **Total Samples**: {results.get('total_samples', 0)}
- **Evaluation Time**: {results.get('total_time', 0):.2f} seconds

## Performance Metrics

### Basic Text Quality
- **BLEU Score**: {results.get('bleu_score', 0):.4f}
- **ROUGE-1**: {results.get('rouge_scores', {}).get('rouge-1', 0):.4f}
- **ROUGE-2**: {results.get('rouge_scores', {}).get('rouge-2', 0):.4f}
- **ROUGE-L**: {results.get('rouge_scores', {}).get('rouge-l', 0):.4f}
- **Semantic Similarity**: {results.get('semantic_similarity', 0):.4f}

### Task Performance
- **Overall Accuracy**: {results.get('accuracy', 0):.4f}
- **Intent Accuracy**: {results.get('intent_accuracy', 0):.4f}

### E-commerce Specific Metrics
- **Customer Satisfaction**: {results.get('customer_satisfaction', 0):.4f}
- **Product Knowledge**: {results.get('product_knowledge', 0):.4f}
- **Conversation Flow**: {results.get('conversation_flow', 0):.4f}
- **Entity Extraction**: {results.get('entity_extraction', 0):.4f}
- **Price Accuracy**: {results.get('price_accuracy', 0):.4f}
- **Recommendation Quality**: {results.get('recommendation_quality', 0):.4f}

### Efficiency
- **Average Response Time**: {results.get('response_time', 0):.4f} seconds

## E-commerce Performance Interpretation

### Customer Service Quality
- **Customer Satisfaction > 0.7**: Excellent customer service tone
- **Conversation Flow > 0.6**: Good interactive conversation
- **Response Time < 2.0s**: Good response speed

### Product Expertise
- **Product Knowledge > 0.8**: Excellent product understanding
- **Entity Extraction > 0.7**: Good at identifying product details
- **Price Accuracy > 0.6**: Good at providing price information

### Recommendation Quality
- **Recommendation Quality > 0.7**: Excellent product recommendations
- **Intent Accuracy > 0.8**: Good at understanding customer needs

## Overall Assessment
- **Excellent (0.8+)**: Model performs very well for e-commerce tasks
- **Good (0.6-0.8)**: Model performs well with room for improvement
- **Fair (0.4-0.6)**: Model needs significant improvement
- **Poor (<0.4)**: Model requires major retraining or architecture changes

## Recommendations
1. **High Customer Satisfaction**: Focus on polite, helpful responses
2. **Strong Product Knowledge**: Ensure accurate product information
3. **Good Conversation Flow**: Maintain interactive dialogue
4. **Accurate Entity Extraction**: Properly identify customer needs
5. **Reliable Price Information**: Provide accurate pricing
6. **Quality Recommendations**: Offer relevant product suggestions
"""
        return report

async def main():
    """Main function for cloud model evaluation"""
    # Get model backend from config or command line
    import sys
    model_backend = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Check for test data
    test_data_path = "training/dataset/test_conversations.json"
    
    if not Path(test_data_path).exists():
        logger.error(f"Test data not found at {test_data_path}")
        logger.info("Please run prepare_data.py first to generate test data")
        return
    
    # Load test data
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Create evaluator and initialize model
    evaluator = CloudModelEvaluator(model_backend)
    
    if not await evaluator.initialize_model():
        logger.error("Failed to initialize model")
        return
    
    # Evaluate model
    try:
        results = await evaluator.evaluate_model(test_data)
        
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
        print("\n" + "="*60)
        print("E-COMMERCE AI AGENT EVALUATION SUMMARY")
        print("="*60)
        print(f"Model Backend: {evaluator.model_backend}")
        print(f"Total Samples: {results.get('total_samples', 0)}")
        print(f"Total Time: {results.get('total_time', 0):.2f}s")
        print()
        print("BASIC METRICS:")
        print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
        print(f"  BLEU Score: {results.get('bleu_score', 0):.4f}")
        print(f"  ROUGE-1: {results.get('rouge_scores', {}).get('rouge-1', 0):.4f}")
        print(f"  Intent Accuracy: {results.get('intent_accuracy', 0):.4f}")
        print(f"  Response Time: {results.get('response_time', 0):.4f}s")
        print()
        print("E-COMMERCE METRICS:")
        print(f"  Customer Satisfaction: {results.get('customer_satisfaction', 0):.4f}")
        print(f"  Product Knowledge: {results.get('product_knowledge', 0):.4f}")
        print(f"  Conversation Flow: {results.get('conversation_flow', 0):.4f}")
        print(f"  Entity Extraction: {results.get('entity_extraction', 0):.4f}")
        print(f"  Price Accuracy: {results.get('price_accuracy', 0):.4f}")
        print(f"  Recommendation Quality: {results.get('recommendation_quality', 0):.4f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        # Cleanup
        if evaluator.model_loader:
            await evaluator.model_loader.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
