"""
Dataset Discovery Agent

Uses semantic search with Hugging Face Transformers (BAAI/bge-base-en-v1.5) 
to find the most semantically relevant datasets from Kaggle based on research questions.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import kaggle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.core.context_manager import (
    read_context, write_context, log_step, 
    update_context_chain, get_research_question, add_dataset_ref
)

# Load environment variables
load_dotenv()

class DatasetDiscoveryAgent:
    """
    Dataset Discovery Agent for finding relevant datasets using semantic search.
    
    Uses BAAI/bge-base-en-v1.5 for semantic embeddings to match research questions
    to dataset descriptions even when wording differs.
    """
    
    def __init__(self):
        """Initialize Dataset Discovery Agent."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize Kaggle API
        self._setup_kaggle_api()
        
        # Initialize sentence transformer model
        self.model_name = "BAAI/bge-base-en-v1.5"
        self.model = SentenceTransformer(self.model_name)
        
        # Create output directory
        self.output_dir = Path("output/dataset_discovery_01")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Dataset Discovery Agent initialized with model: {self.model_name}")
    
    def _setup_kaggle_api(self):
        """Setup Kaggle API authentication."""
        try:
            # Check if credentials are set
            if not os.getenv('KAGGLE_USERNAME') or not os.getenv('KAGGLE_KEY'):
                raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY must be set in .env file")
            
            # Authenticate with Kaggle
            kaggle.api.authenticate()
            self.logger.info("Kaggle API authenticated successfully")
        except Exception as e:
            self.logger.error(f"Failed to authenticate with Kaggle API: {e}")
            raise
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute dataset discovery task.
        
        Returns:
            Dict containing discovery results and metadata
        """
        try:
            # Read context
            context = read_context()
            
            # Get research question
            research_question = get_research_question(context)
            if not research_question:
                raise ValueError("No research question found in context")
            
            self.logger.info(f"Starting dataset discovery for: {research_question}")
            
            # Search for datasets
            datasets = self._search_kaggle_datasets(research_question)
            
            if not datasets:
                raise ValueError("No datasets found for the research question")
            
            # Perform semantic search and ranking
            ranked_datasets = self._rank_datasets_semantically(research_question, datasets)
            
            # Select top datasets
            selected_datasets = self._select_best_datasets(ranked_datasets, max_select=3)
            
            # Prepare results
            results = {
                'research_question': research_question,
                'search_timestamp': datetime.now().isoformat(),
                'total_datasets_found': len(datasets),
                'selected_datasets': selected_datasets,
                'all_candidates': ranked_datasets,
                'search_metadata': {
                    'model_used': self.model_name,
                    'search_method': 'semantic_similarity',
                    'max_results_requested': 10,
                    'max_selected': 3
                }
            }
            
            # Update context
            update_context_chain(context, 'dataset_discovery', results)
            
            # Add dataset references to project metadata
            for dataset in selected_datasets:
                add_dataset_ref(context, dataset['ref'])
            
            # Log completion
            log_step(context, 'dataset_discovery', 
                    f"Found {len(selected_datasets)} relevant datasets for research question")
            
            # Write updated context
            write_context(context)
            
            # Save detailed results
            self._save_discovery_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Dataset discovery failed: {e}")
            # Update context with error
            context = read_context()
            update_context_chain(context, 'dataset_discovery', {'error': str(e)})
            log_step(context, 'dataset_discovery', f"Error: {str(e)}")
            write_context(context)
            raise
    
    def _search_kaggle_datasets(self, research_question: str) -> List[Dict[str, Any]]:
        """
        Search Kaggle for datasets using keywords from research question.
        
        Args:
            research_question: The research question to search for
            
        Returns:
            List of dataset dictionaries
        """
        # Extract keywords from research question
        keywords = self._extract_keywords(research_question)
        
        datasets = []
        for keyword in keywords[:3]:  # Use top 3 keywords
            try:
                # Search Kaggle datasets
                search_results = kaggle.api.dataset_list(search=keyword)
                
                for dataset in search_results:
                    try:
                        dataset_info = {
                            'ref': dataset.ref,
                            'title': dataset.title,
                            'description': getattr(dataset, 'description', '') or "",
                            'size': getattr(dataset, 'size', 'Unknown'),
                            'lastUpdated': getattr(dataset, 'lastUpdated', 'Unknown'),
                            'downloadCount': getattr(dataset, 'downloadCount', 0),
                            'voteCount': getattr(dataset, 'voteCount', 0),
                            'usabilityRating': getattr(dataset, 'usabilityRating', 0),
                            'search_keyword': keyword
                        }
                        datasets.append(dataset_info)
                    except Exception as e:
                        self.logger.warning(f"Failed to process dataset {dataset.ref}: {e}")
                        continue
                    
            except Exception as e:
                self.logger.warning(f"Failed to search for keyword '{keyword}': {e}")
        
        return datasets
    
    def _extract_keywords(self, research_question: str) -> List[str]:
        """
        Extract relevant keywords from research question.
        
        Args:
            research_question: The research question
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - could be enhanced with NLP
        words = research_question.lower().split()
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:5]  # Return top 5 keywords
    
    def _rank_datasets_semantically(self, research_question: str, datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank datasets using semantic similarity with the research question.
        
        Args:
            research_question: The research question
            datasets: List of dataset dictionaries
            
        Returns:
            List of datasets ranked by similarity score
        """
        # Prepare texts for embedding
        texts = [research_question] + [f"{d['title']} {d['description']}" for d in datasets]
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        # Calculate similarities
        question_embedding = embeddings[0].reshape(1, -1)
        dataset_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(question_embedding, dataset_embeddings)[0]
        
        # Add similarity scores to datasets
        for i, dataset in enumerate(datasets):
            dataset['similarity_score'] = float(similarities[i])
        
        # Sort by similarity score (descending)
        ranked_datasets = sorted(datasets, key=lambda x: x['similarity_score'], reverse=True)
        
        return ranked_datasets
    
    def _select_best_datasets(self, ranked_datasets: List[Dict[str, Any]], max_select: int = 3) -> List[Dict[str, Any]]:
        """
        Select the best datasets based on similarity score and quality metrics.
        
        Args:
            ranked_datasets: Datasets ranked by similarity
            max_select: Maximum number of datasets to select
            
        Returns:
            List of selected datasets
        """
        selected = []
        
        for dataset in ranked_datasets[:max_select]:
            # Add selection reason
            dataset['selection_reason'] = self._generate_selection_reason(dataset)
            selected.append(dataset)
        
        return selected
    
    def _generate_selection_reason(self, dataset: Dict[str, Any]) -> str:
        """
        Generate a reason for selecting this dataset.
        
        Args:
            dataset: Dataset dictionary
            
        Returns:
            Selection reason string
        """
        reasons = []
        
        if dataset.get('similarity_score', 0) > 0.7:
            reasons.append("High semantic similarity to research question")
        
        if dataset.get('downloadCount', 0) > 1000:
            reasons.append("Popular dataset with high download count")
        
        if dataset.get('usabilityRating', 0) > 7:
            reasons.append("High usability rating")
        
        if dataset.get('voteCount', 0) > 10:
            reasons.append("Well-voted dataset")
        
        if not reasons:
            reasons.append("Selected based on semantic similarity")
        
        return "; ".join(reasons)
    
    def _save_discovery_results(self, results: Dict[str, Any]):
        """
        Save detailed discovery results to output directory.
        
        Args:
            results: Discovery results dictionary
        """
        # Save full results
        results_file = self.output_dir / "discovery_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary_file = self.output_dir / "discovery_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Dataset Discovery Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Research Question: {results['research_question']}\n")
            f.write(f"Total Datasets Found: {results['total_datasets_found']}\n")
            f.write(f"Selected Datasets: {len(results['selected_datasets'])}\n\n")
            
            for i, dataset in enumerate(results['selected_datasets'], 1):
                f.write(f"{i}. {dataset['title']}\n")
                f.write(f"   Ref: {dataset['ref']}\n")
                f.write(f"   Similarity Score: {dataset['similarity_score']:.3f}\n")
                f.write(f"   Reason: {dataset['selection_reason']}\n\n")
        
        self.logger.info(f"Discovery results saved to {self.output_dir}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run agent
    agent = DatasetDiscoveryAgent()
    results = agent.execute()
    print(f"Dataset discovery completed. Found {len(results['selected_datasets'])} datasets.") 