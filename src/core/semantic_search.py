#!/usr/bin/env python3
"""
Semantic Search Module for Dataset Discovery

This module provides semantic search capabilities using Hugging Face transformers
to find the most relevant datasets based on research questions.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import sentence transformers for semantic search
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# Import Kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    logging.warning("kaggle-api-extended not available. Install with: pip install kaggle-api-extended")

logger = logging.getLogger(__name__)


class SemanticDatasetSearch:
    """
    Semantic search system for finding relevant datasets using sentence transformers.
    
    Features:
    - Semantic similarity using sentence-transformers/all-MiniLM-L6-v2
    - Kaggle API integration for dataset discovery
    - Cosine similarity ranking
    - Metadata extraction and embedding
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize semantic search system.
        
        Args:
            model_name: Hugging Face model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self.kaggle_api = None
        
        # Initialize components
        self._setup_sentence_transformer()
        self._setup_kaggle_api()
        
        # Create context directory
        self.context_dir = Path("context")
        self.context_dir.mkdir(exist_ok=True)
        
        logger.info(f"Semantic Dataset Search initialized with model: {model_name}")
    
    def _setup_sentence_transformer(self):
        """Setup sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available")
            return
        
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Sentence transformer loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load sentence transformer: {e}")
    
    def _setup_kaggle_api(self):
        """Setup Kaggle API connection."""
        if not KAGGLE_AVAILABLE:
            logger.error("Kaggle API not available")
            return
        
        try:
            # Get credentials from environment
            username = os.getenv('KAGGLE_USERNAME')
            key = os.getenv('KAGGLE_KEY')
            
            if not username or not key:
                logger.error("Kaggle credentials not found in environment variables")
                return
            
            # Set environment variables for Kaggle API
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = key
            
            self.kaggle_api = KaggleApi()
            self.kaggle_api.authenticate()
            logger.info("✅ Kaggle API authenticated successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Kaggle API: {e}")
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Embed text using sentence transformer.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        if self.model is None:
            logger.error("Sentence transformer model not available")
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            return None
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    async def search_datasets(self, research_question: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant datasets using semantic similarity.
        
        Args:
            research_question: User's research question or project description
            max_results: Maximum number of datasets to return
            
        Returns:
            List of relevant datasets with metadata and similarity scores
        """
        if self.kaggle_api is None:
            logger.error("Kaggle API not available")
            return []
        
        if self.model is None:
            logger.error("Sentence transformer not available")
            return []
        
        try:
            # Embed the research question
            logger.info(f"Embedding research question: {research_question}")
            question_embedding = self.embed_text(research_question)
            if question_embedding is None:
                logger.error("Failed to embed research question")
                return []
            
            # Search Kaggle datasets
            logger.info(f"Searching Kaggle datasets for: {research_question}")
            datasets = await self._search_kaggle_datasets(research_question, max_results * 2)
            logger.info(f"Retrieved {len(datasets)} datasets from Kaggle")
            
            if not datasets:
                logger.warning("No datasets found from Kaggle API")
                return []
            
            # Compute similarity scores
            scored_datasets = []
            logger.info("Computing similarity scores...")
            for i, dataset in enumerate(datasets):
                logger.info(f"Processing dataset {i+1}/{len(datasets)}: {dataset.get('title', 'Unknown')}")
                
                # Create dataset text for embedding
                dataset_text = self._create_dataset_text(dataset)
                logger.info(f"Dataset text length: {len(dataset_text)} characters")
                
                dataset_embedding = self.embed_text(dataset_text)
                
                if dataset_embedding is not None:
                    similarity = self.compute_similarity(question_embedding, dataset_embedding)
                    dataset['similarity_score'] = similarity
                    scored_datasets.append(dataset)
                    logger.info(f"Similarity score: {similarity:.3f}")
                else:
                    logger.warning(f"Failed to embed dataset: {dataset.get('title', 'Unknown')}")
            
            logger.info(f"Successfully scored {len(scored_datasets)} datasets")
            
            # Sort by similarity score
            scored_datasets.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Return top results
            top_datasets = scored_datasets[:max_results]
            
            # If no datasets found, return empty list - no fallbacks
            if not top_datasets:
                logger.warning(f"No datasets found via semantic search for: {research_question}")
                return []
            
            logger.info(f"Found {len(top_datasets)} relevant datasets")
            for dataset in top_datasets:
                logger.info(f"Dataset: {dataset.get('title', 'Unknown')} (score: {dataset.get('similarity_score', 0):.3f})")
            
            return top_datasets
            
        except Exception as e:
            logger.error(f"Failed to search datasets: {e}")
            return []
    
    async def _search_kaggle_datasets(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Search Kaggle datasets using the API with intelligent query expansion.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of dataset metadata
        """
        try:
            # Generate multiple search queries from the research question
            search_queries = self._generate_search_queries(query)
            logger.info(f"Generated search queries: {search_queries}")
            
            all_datasets = []
            seen_refs = set()  # Track unique datasets
            
            for search_query in search_queries:
                logger.info(f"Searching Kaggle with query: '{search_query}'")
                datasets = self.kaggle_api.dataset_list(search=search_query)
                logger.info(f"Query '{search_query}' returned {len(datasets)} datasets")
                
                # Process datasets from this query
                logger.info(f"Processing {len(datasets)} datasets from query '{search_query}'")
                for i, dataset in enumerate(datasets):
                    try:
                        logger.info(f"Processing dataset {i+1}/{len(datasets)} from query '{search_query}'")
                        
                        # Convert dataset object to dictionary for easier access
                        dataset_dict = dataset.__dict__ if hasattr(dataset, '__dict__') else {}
                        
                        # Extract basic info using multiple possible attribute names
                        title = (getattr(dataset, 'title', None) or 
                                getattr(dataset, 'name', None) or 
                                dataset_dict.get('title', '') or 
                                dataset_dict.get('name', ''))
                        
                        # Try to get the dataset reference directly first
                        dataset_ref = (getattr(dataset, 'ref', None) or 
                                     dataset_dict.get('ref', ''))
                        
                        # If no direct ref, try to construct from owner and slug
                        if not dataset_ref:
                            owner = (getattr(dataset, 'owner_name', None) or 
                                   getattr(dataset, 'ownerName', None) or 
                                   getattr(dataset, 'owner', None) or 
                                   dataset_dict.get('owner_name', '') or 
                                   dataset_dict.get('ownerName', '') or 
                                   dataset_dict.get('owner', ''))
                            
                            slug = (getattr(dataset, 'datasetSlug', None) or 
                                   getattr(dataset, 'slug', None) or 
                                   dataset_dict.get('datasetSlug', '') or 
                                   dataset_dict.get('slug', ''))
                            
                            if owner and slug:
                                dataset_ref = f"{owner}/{slug}"
                        
                        logger.info(f"Dataset: {title} | Ref: {dataset_ref}")
                        
                        if not dataset_ref:
                            logger.warning(f"Skipping dataset {title}: missing dataset reference")
                            continue
                        
                        # Skip if we've already seen this dataset
                        if dataset_ref in seen_refs:
                            logger.info(f"Skipping duplicate dataset: {title}")
                            continue
                        
                        seen_refs.add(dataset_ref)
                        
                        # Get detailed view
                        logger.info(f"Getting detailed info for: {dataset_ref}")
                        try:
                            detailed_info = self.kaggle_api.dataset_metadata(dataset_ref)
                        except AttributeError:
                            # Fallback: use basic dataset info if dataset_metadata doesn't exist
                            logger.warning(f"dataset_metadata method not available, using basic info for {dataset_ref}")
                            detailed_info = dataset
                        
                        # Convert detailed info to dictionary
                        detailed_dict = detailed_info.__dict__ if hasattr(detailed_info, '__dict__') else {}
                        
                        # Combine basic and detailed info
                        dataset_info = {
                            'ref': dataset_ref,
                            'title': title,
                            'description': (getattr(detailed_info, 'description', None) or 
                                          detailed_dict.get('description', '')),
                            'tags': (getattr(detailed_info, 'tags', None) or 
                                    detailed_dict.get('tags', [])),
                            'size': (getattr(dataset, 'size', None) or 
                                    dataset_dict.get('size', 0)),
                            'download_count': (getattr(dataset, 'downloadCount', None) or 
                                             getattr(dataset, 'download_count', None) or 
                                             dataset_dict.get('downloadCount', 0) or 
                                             dataset_dict.get('download_count', 0)),
                            'vote_count': (getattr(dataset, 'voteCount', None) or 
                                         getattr(dataset, 'vote_count', None) or 
                                         dataset_dict.get('voteCount', 0) or 
                                         dataset_dict.get('vote_count', 0)),
                            'last_updated': (getattr(dataset, 'lastUpdated', None) or 
                                           getattr(dataset, 'last_updated', None) or 
                                           dataset_dict.get('lastUpdated', '') or 
                                           dataset_dict.get('last_updated', '')),
                            'owner_name': owner,
                            'dataset_slug': slug,
                            'license_name': (getattr(dataset, 'licenseName', None) or 
                                           getattr(dataset, 'license_name', None) or 
                                           dataset_dict.get('licenseName', '') or 
                                           dataset_dict.get('license_name', '')),
                            'subtitle': (getattr(dataset, 'subtitle', None) or 
                                       dataset_dict.get('subtitle', '')),
                            'categories': (getattr(detailed_info, 'categories', None) or 
                                         detailed_dict.get('categories', [])),
                            'keywords': (getattr(detailed_info, 'keywords', None) or 
                                       detailed_dict.get('keywords', []))
                        }
                        
                        all_datasets.append(dataset_info)
                        logger.info(f"✅ Successfully added dataset: {title}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to get detailed info for dataset: {e}")
                        continue
                
                # Stop if we have enough datasets
                if len(all_datasets) >= max_results * 3:
                    break
            
            logger.info(f"Successfully processed {len(all_datasets)} unique datasets")
            return all_datasets
            
        except Exception as e:
            logger.error(f"Failed to search Kaggle datasets: {e}")
            return []
    
    def _generate_search_queries(self, research_question: str) -> List[str]:
        """
        Generate multiple search queries from a research question.
        
        Args:
            research_question: The original research question
            
        Returns:
            List of search queries to try
        """
        # Extract key terms from the research question
        words = research_question.lower().split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'about', 'above',
            'after', 'again', 'against', 'all', 'am', 'any', 'as', 'because', 'before', 'being',
            'below', 'between', 'both', 'during', 'each', 'few', 'from', 'further', 'having',
            'here', 'how', 'if', 'into', 'just', 'more', 'most', 'no', 'nor', 'not', 'now',
            'only', 'other', 'our', 'out', 'over', 'own', 'same', 'so', 'some', 'such', 'than',
            'that', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'under', 'until',
            'very', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'you'
        }
        
        # Extract meaningful words (longer than 2 chars, not stop words)
        key_terms = [word.strip('.,!?()[]{}\":;') for word in words 
                    if len(word.strip('.,!?()[]{}\":;')) > 2 and word.strip('.,!?()[]{}\":;') not in stop_words]
        
        # If no key terms found, use original words
        if not key_terms:
            key_terms = [word.strip('.,!?()[]{}\":;') for word in words if len(word.strip('.,!?()[]{}\":;')) > 2]
        
        queries = []
        
        # Add the original research question
        queries.append(research_question)
        
        # Add individual key terms
        for term in key_terms:
            queries.append(term)
            queries.append(f"{term} dataset")
            queries.append(f"{term} data")
        
        # Add combinations of key terms
        if len(key_terms) >= 2:
            queries.append(f"{key_terms[0]} {key_terms[1]}")
            queries.append(f"{key_terms[0]} {key_terms[1]} dataset")
            if len(key_terms) >= 3:
                queries.append(f"{key_terms[0]} {key_terms[1]} {key_terms[2]}")
        
        # Remove duplicates while preserving order
        unique_queries = list(dict.fromkeys(queries))
        
        # Limit to reasonable number of queries
        return unique_queries[:15]
    
    def _create_dataset_text(self, dataset: Dict[str, Any]) -> str:
        """
        Create text representation of dataset for embedding.
        
        Args:
            dataset: Dataset metadata
            
        Returns:
            Text representation for embedding
        """
        text_parts = []
        
        # Add title
        if dataset.get('title'):
            text_parts.append(dataset['title'])
        
        # Add description
        if dataset.get('description'):
            text_parts.append(dataset['description'])
        
        # Add tags
        if dataset.get('tags'):
            text_parts.extend(dataset['tags'])
        
        # Add categories
        if dataset.get('categories'):
            text_parts.extend(dataset['categories'])
        
        # Add keywords
        if dataset.get('keywords'):
            text_parts.extend(dataset['keywords'])
        
        # Add subtitle
        if dataset.get('subtitle'):
            text_parts.append(dataset['subtitle'])
        
        return " ".join(text_parts)
    
    def save_context(self, data: Dict[str, Any], filename: str = "context_output.json"):
        """
        Save context data to JSON file.
        
        Args:
            data: Data to save
            filename: Output filename
        """
        try:
            output_path = self.context_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Context saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save context: {e}")
    
    def load_context(self, filename: str = "context_output.json") -> Optional[Dict[str, Any]]:
        """
        Load context data from JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded data or None if failed
        """
        try:
            input_path = self.context_dir / filename
            if not input_path.exists():
                return None
            
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load context: {e}")
            return None


# Global instance for reuse
semantic_search = None

def get_semantic_search() -> SemanticDatasetSearch:
    """Get or create global semantic search instance."""
    global semantic_search
    if semantic_search is None:
        semantic_search = SemanticDatasetSearch()
    return semantic_search
