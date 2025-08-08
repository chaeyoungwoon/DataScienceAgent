#!/usr/bin/env python3
"""
Local LLM Manager for Data Science Agent Swarm

This module provides specialized Hugging Face Transformers for data science tasks,
replacing API-based LLMs with local models optimized for:
- Dataset discovery and scoring
- Technical report generation
- Code generation for data science
- Statistical analysis and insights
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import json
import re

# Hugging Face Transformers imports
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline,
    TextGenerationPipeline,
    SummarizationPipeline,
    TextClassificationPipeline
)
import torch

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Available local LLM providers."""
    TEXT_GENERATION = "text_generation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    CODE_GENERATION = "code_generation"


class LocalLLMManager:
    """
    Specialized LLM manager using Hugging Face Transformers for data science tasks.
    
    Features:
    - Local processing (no API costs or rate limits)
    - Specialized models for data science tasks
    - Optimized for dataset discovery, report generation, and code generation
    - Privacy-first approach
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize local LLM manager with specialized models.
        
        Args:
            config: Configuration with model paths and settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize specialized models
        self.text_generator = None
        self.summarizer = None
        self.classifier = None
        self.code_generator = None
        
        self._setup_models()
        
        # Track usage for optimization
        self.usage_stats = {
            'text_generation': {'calls': 0, 'errors': 0},
            'summarization': {'calls': 0, 'errors': 0},
            'classification': {'calls': 0, 'errors': 0},
            'code_generation': {'calls': 0, 'errors': 0}
        }
        
        self.logger.info("Local LLM Manager initialized with specialized data science models")
    
    def _setup_models(self):
        """Setup specialized models for data science tasks."""
        try:
            # Advanced text generation for reasoning and insights - using a more capable model
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2",  # More reliable and widely used
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info("✅ Text generation model loaded (GPT-2)")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load text generation model: {e}")
            # Fallback to DialoGPT
            try:
                self.text_generator = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-medium",
                    device=0 if torch.cuda.is_available() else -1
                )
                self.logger.info("✅ Text generation model loaded (DialoGPT-medium - fallback)")
            except Exception as e2:
                self.logger.error(f"❌ Failed to load fallback text generation model: {e2}")
        
        try:
            # Better summarization for report generation
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info("✅ Summarization model loaded (BART-large-CNN)")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load summarization model: {e}")
        
        try:
            # Better classification for dataset relevance scoring - using a more sophisticated model
            self.classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",  # Better for understanding context
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info("✅ Classification model loaded (RoBERTa-base-sentiment)")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load classification model: {e}")
            # Fallback to distilbert
            try:
                self.classifier = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased",
                    device=0 if torch.cuda.is_available() else -1
                )
                self.logger.info("✅ Classification model loaded (DistilBERT - fallback)")
            except Exception as e2:
                self.logger.error(f"❌ Failed to load fallback classification model: {e2}")
        
        try:
            # Better code generation for data science tasks
            self.code_generator = pipeline(
                "text-generation",
                model="Salesforce/codegen-350M-mono",
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info("✅ Code generation model loaded (CodeGen-350M)")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load code generation model: {e}")
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available local providers."""
        available = []
        
        if self.text_generator:
            available.append(LLMProvider.TEXT_GENERATION)
        
        if self.summarizer:
            available.append(LLMProvider.SUMMARIZATION)
        
        if self.classifier:
            available.append(LLMProvider.CLASSIFICATION)
        
        if self.code_generator:
            available.append(LLMProvider.CODE_GENERATION)
        
        return available
    
    def get_preferred_provider(self) -> Optional[LLMProvider]:
        """Get the preferred available provider for general text generation."""
        available = self.get_available_providers()
        return LLMProvider.TEXT_GENERATION if LLMProvider.TEXT_GENERATION in available else None
    
    async def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate text using local models.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if not self.text_generator:
            return "Local text generation model not available"
        
        try:
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: self.text_generator(
                    prompt, 
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.text_generator.tokenizer.eos_token_id,
                    truncation=True
                )
            )
            
            generated_text = result[0]['generated_text']
            
            # Clean up the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            self.usage_stats['text_generation']['calls'] += 1
            return generated_text
            
        except Exception as e:
            self.usage_stats['text_generation']['errors'] += 1
            self.logger.error(f"Error in text generation: {e}")
            return f"Error generating text: {str(e)}"
    
    async def generate_code(self, task_description: str, context: Dict[str, Any] = None, language: str = "python") -> Dict[str, Any]:
        """Generate code using specialized code generation model."""
        if not self.code_generator:
            return {
                'code': f"# {task_description}\n# Local code generation model not available",
                'language': language,
                'status': 'error',
                'error': 'Code generation model not available'
            }
        
        try:
            # Create code generation prompt
            code_prompt = f"""
# {task_description}
# Language: {language}
# Context: {context or 'No additional context'}

"""
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.code_generator(
                    code_prompt,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.3,
                    pad_token_id=self.code_generator.tokenizer.eos_token_id,
                    truncation=True
                )
            )
            
            generated_code = result[0]['generated_text']
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(generated_code, language)
            
            self.usage_stats['code_generation']['calls'] += 1
            
            return {
                'code': code_blocks,
                'language': language,
                'task': task_description,
                'status': 'success'
            }
            
        except Exception as e:
            self.usage_stats['code_generation']['errors'] += 1
            return {
                'code': f"# {task_description}\n# Error: {str(e)}",
                'language': language,
                'status': 'error',
                'error': str(e)
            }
    
    async def generate_search_queries(self, research_question: str, domain: str = "general", max_queries: int = 10) -> List[str]:
        """Generate search queries for dataset discovery using simple keyword extraction."""
        try:
            # Simple keyword extraction approach - much more reliable than LLM
            research_lower = research_question.lower()
            
            # Extract key terms from research question
            key_terms = []
            
            # Common data science terms to remove
            stop_words = {
                'analyze', 'analysis', 'dataset', 'data', 'about', 'the', 'a', 'an', 'and', 'or', 'but', 
                'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 
                'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 
                'may', 'might', 'can', 'this', 'that', 'these', 'those', 'study', 'research', 'predict',
                'prediction', 'model', 'machine', 'learning', 'ai', 'artificial', 'intelligence'
            }
            
            # Extract meaningful words
            words = research_lower.split()
            for word in words:
                # Clean the word
                word = word.strip('.,!?()[]{}":;')
                if len(word) > 2 and word not in stop_words:
                    key_terms.append(word)
            
            # If no key terms found, use the original words
            if not key_terms:
                for word in words:
                    word = word.strip('.,!?()[]{}":;')
                    if len(word) > 2:
                        key_terms.append(word)
            
            # Generate search queries using key terms
            queries = []
            
            # Add the original key terms
            for term in key_terms[:max_queries//2]:
                queries.append(f"{term} dataset")
                queries.append(f"{term} data")
            
            # Add combinations of key terms
            if len(key_terms) >= 2:
                for i in range(min(len(key_terms)-1, max_queries//4)):
                    queries.append(f"{key_terms[i]} {key_terms[i+1]} dataset")
            
            # Add domain-specific variations
            for term in key_terms[:max_queries//4]:
                queries.append(f"{term} csv")
                queries.append(f"{term} analysis")
            
            # Remove duplicates and limit
            unique_queries = list(dict.fromkeys(queries))  # Preserves order
            

            
            return unique_queries[:max_queries]
            
        except Exception as e:
            self.logger.error(f"Error generating search queries: {e}")
            # Fallback to simple approach
            research_words = research_question.lower().split()
            fallback_queries = []
            for word in research_words:
                if len(word) > 3:
                    fallback_queries.append(f"{word} dataset")
            return fallback_queries[:max_queries]
            

            
        except Exception as e:
            self.usage_stats['text_generation']['errors'] += 1
            self.logger.error(f"Error generating search queries: {e}")
            return []
    
    async def score_dataset_relevance(self, dataset_description: str, research_question: str, requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """Score dataset relevance using classification model."""
        if not self.classifier:
            return {"relevance_score": 0.5, "reasoning": "Classification model not available"}
        
        try:
            # Create classification prompt
            classification_text = f"Dataset: {dataset_description} | Research: {research_question}"
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.classifier(classification_text)
            )
            
            # Create enhanced classification result with context
            enhanced_result = {
                'label': result[0]['label'],
                'score': result[0]['score'],
                'dataset_text': dataset_description.lower(),
                'research_text': research_question.lower()
            }
            
            # Map classification result to relevance score
            relevance_score = self._map_classification_to_score(enhanced_result)
            
            self.usage_stats['classification']['calls'] += 1
            
            return {
                "relevance_score": relevance_score,
                "reasoning": f"Classified as {result[0]['label']} with confidence {result[0]['score']:.3f}",
                "confidence": result[0]['score'],
                "status": "success"
            }
            
        except Exception as e:
            self.usage_stats['classification']['errors'] += 1
            return {"relevance_score": 0.5, "reasoning": f"Error: {str(e)}"}
    
    async def generate_insights(self, analysis_results: Dict[str, Any], research_question: str) -> Dict[str, Any]:
        """Generate insights using summarization model."""
        if not self.summarizer:
            return {"error": "Summarization model not available"}
        
        try:
            # Convert analysis results to text
            analysis_text = json.dumps(analysis_results, indent=2)
            
            # Create insight generation prompt
            insight_prompt = f"""
Research Question: {research_question}

Analysis Results:
{analysis_text}

Generate key insights from this data science analysis.
"""
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.summarizer(
                    insight_prompt,
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )
            )
            
            insights = result[0]['summary_text']
            
            self.usage_stats['summarization']['calls'] += 1
            
            return {
                "insights": [insights],
                "research_question": research_question,
                "status": "success"
            }
            
        except Exception as e:
            self.usage_stats['summarization']['errors'] += 1
            return {"error": str(e)}
    
    def _extract_code_blocks(self, text: str, language: str = "python") -> List[str]:
        """Extract code blocks from generated text."""
        # Look for code blocks marked with ```
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', text, re.DOTALL)
        
        # Also look for code blocks without language specification
        if not code_blocks:
            code_blocks = re.findall(r'```\n(.*?)\n```', text, re.DOTALL)
        
        # If no code blocks found, return the entire text
        if not code_blocks:
            return [text.strip()]
        
        return [block.strip() for block in code_blocks]
    
    def _map_classification_to_score(self, classification_result: Dict[str, Any]) -> float:
        """Map classification result to relevance score (0.0 to 1.0)."""
        dataset_text = classification_result.get('dataset_text', '').lower()
        research_text = classification_result.get('research_text', '').lower()
        
        # Simple keyword matching approach
        research_words = set()
        for word in research_text.split():
            # Clean and filter words
            word = word.strip('.,!?()[]{}":;').lower()
            if len(word) > 2 and word not in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'about', 'this', 'that', 'these', 'those', 'analyze', 'analysis', 'dataset', 'data']:
                research_words.add(word)
        
        # Count matches
        exact_matches = sum(1 for word in research_words if word in dataset_text)
        
        # Calculate simple relevance score
        if len(research_words) == 0:
            return 0.5  # Default score if no meaningful words found
        
        relevance_ratio = exact_matches / len(research_words)
        
        # Boost for having any matches
        if exact_matches > 0:
            relevance_ratio += 0.2
        
        # Cap at 1.0
        return min(relevance_ratio, 1.0)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all models."""
        return {
            'models': {
                'text_generation': {
                    'available': self.text_generator is not None,
                    'stats': self.usage_stats['text_generation']
                },
                'summarization': {
                    'available': self.summarizer is not None,
                    'stats': self.usage_stats['summarization']
                },
                'classification': {
                    'available': self.classifier is not None,
                    'stats': self.usage_stats['classification']
                },
                'code_generation': {
                    'available': self.code_generator is not None,
                    'stats': self.usage_stats['code_generation']
                }
            },
            'preferred_provider': self.get_preferred_provider().value if self.get_preferred_provider() else None,
            'available_providers': [p.value for p in self.get_available_providers()],
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    def is_available(self) -> bool:
        """Check if any models are available."""
        return len(self.get_available_providers()) > 0


# Alias for backward compatibility
LLMManager = LocalLLMManager
