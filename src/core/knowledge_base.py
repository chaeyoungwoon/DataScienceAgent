"""
Knowledge Base for Data Science Agent Swarm

This module provides persistent storage and retrieval capabilities for agent
knowledge, including project history, learned patterns, and shared insights.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class KnowledgeBase:
    """
    Knowledge base for storing and retrieving agent knowledge.
    
    This class provides persistent storage for:
    - Project history and results
    - Learned patterns and insights
    - Dataset metadata and quality scores
    - Model performance records
    - Agent collaboration history
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the knowledge base.
        
        Args:
            config: Configuration dictionary containing storage settings
        """
        self.logger = logging.getLogger("knowledge_base")
        self.config = config
        
        # Initialize storage backends
        self.vector_db = self.initialize_vector_db()
        self.cache_db = self.initialize_cache_db()
        
        # In-memory storage for development
        self.memory_storage = {
            'projects': {},
            'datasets': {},
            'models': {},
            'insights': {},
            'patterns': {}
        }
        
        self.logger.info("Knowledge base initialized")
    
    def initialize_vector_db(self):
        """Initialize vector database for semantic search."""
        if CHROMA_AVAILABLE:
            try:
                client = chromadb.Client()
                collection = client.create_collection(
                    name="agent_knowledge",
                    metadata={"description": "Agent knowledge and insights"}
                )
                return {
                    'type': 'chromadb',
                    'client': client,
                    'collection': collection
                }
            except Exception as e:
                self.logger.error(f"Failed to initialize ChromaDB: {e}")
                return None
        else:
            self.logger.warning("ChromaDB not available, using in-memory storage")
            return None
    
    def initialize_cache_db(self):
        """Initialize cache database for fast access."""
        if REDIS_AVAILABLE:
            try:
                redis_config = self.config.get('redis', {})
                client = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 1),  # Use different DB for cache
                    decode_responses=True
                )
                return {
                    'type': 'redis',
                    'client': client
                }
            except Exception as e:
                self.logger.error(f"Failed to initialize Redis cache: {e}")
                return None
        else:
            self.logger.warning("Redis not available, using in-memory cache")
            return None
    
    async def store_project_result(self, project_id: str, result: Dict[str, Any]) -> bool:
        """
        Store project result in knowledge base.
        
        Args:
            project_id: Unique project identifier
            result: Project result data
            
        Returns:
            True if stored successfully
        """
        try:
            # Add metadata
            result_with_metadata = {
                'project_id': project_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data': result
            }
            
            # Store in memory
            self.memory_storage['projects'][project_id] = result_with_metadata
            
            # Store in cache if available
            if self.cache_db and self.cache_db['type'] == 'redis':
                self.cache_db['client'].setex(
                    f"project:{project_id}",
                    3600,  # 1 hour TTL
                    json.dumps(result_with_metadata)
                )
            
            # Store in vector DB if available
            if self.vector_db and self.vector_db['type'] == 'chromadb':
                # Extract text for vector storage
                text_content = self.extract_text_from_result(result)
                self.vector_db['collection'].add(
                    documents=[text_content],
                    metadatas=[{'project_id': project_id, 'type': 'project_result'}],
                    ids=[f"project_{project_id}"]
                )
            
            self.logger.info(f"Stored project result for {project_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store project result for {project_id}: {e}")
            return False
    
    async def retrieve_project_result(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve project result from knowledge base.
        
        Args:
            project_id: Unique project identifier
            
        Returns:
            Project result data or None if not found
        """
        try:
            # Check memory first
            if project_id in self.memory_storage['projects']:
                return self.memory_storage['projects'][project_id]
            
            # Check cache
            if self.cache_db and self.cache_db['type'] == 'redis':
                cached_data = self.cache_db['client'].get(f"project:{project_id}")
                if cached_data:
                    return json.loads(cached_data)
            
            self.logger.warning(f"Project result not found for {project_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve project result for {project_id}: {e}")
            return None
    
    async def store_dataset_metadata(self, dataset_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Store dataset metadata in knowledge base.
        
        Args:
            dataset_id: Unique dataset identifier
            metadata: Dataset metadata
            
        Returns:
            True if stored successfully
        """
        try:
            metadata_with_timestamp = {
                'dataset_id': dataset_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data': metadata
            }
            
            self.memory_storage['datasets'][dataset_id] = metadata_with_timestamp
            
            if self.cache_db and self.cache_db['type'] == 'redis':
                self.cache_db['client'].setex(
                    f"dataset:{dataset_id}",
                    7200,  # 2 hours TTL
                    json.dumps(metadata_with_timestamp)
                )
            
            self.logger.info(f"Stored dataset metadata for {dataset_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store dataset metadata for {dataset_id}: {e}")
            return False
    
    async def retrieve_dataset_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve dataset metadata from knowledge base.
        
        Args:
            dataset_id: Unique dataset identifier
            
        Returns:
            Dataset metadata or None if not found
        """
        try:
            if dataset_id in self.memory_storage['datasets']:
                return self.memory_storage['datasets'][dataset_id]
            
            if self.cache_db and self.cache_db['type'] == 'redis':
                cached_data = self.cache_db['client'].get(f"dataset:{dataset_id}")
                if cached_data:
                    return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve dataset metadata for {dataset_id}: {e}")
            return None
    
    async def store_model_performance(self, model_id: str, performance: Dict[str, Any]) -> bool:
        """
        Store model performance data in knowledge base.
        
        Args:
            model_id: Unique model identifier
            performance: Model performance data
            
        Returns:
            True if stored successfully
        """
        try:
            performance_with_timestamp = {
                'model_id': model_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data': performance
            }
            
            self.memory_storage['models'][model_id] = performance_with_timestamp
            
            if self.cache_db and self.cache_db['type'] == 'redis':
                self.cache_db['client'].setex(
                    f"model:{model_id}",
                    3600,  # 1 hour TTL
                    json.dumps(performance_with_timestamp)
                )
            
            self.logger.info(f"Stored model performance for {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store model performance for {model_id}: {e}")
            return False
    
    async def retrieve_model_performance(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve model performance data from knowledge base.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            Model performance data or None if not found
        """
        try:
            if model_id in self.memory_storage['models']:
                return self.memory_storage['models'][model_id]
            
            if self.cache_db and self.cache_db['type'] == 'redis':
                cached_data = self.cache_db['client'].get(f"model:{model_id}")
                if cached_data:
                    return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve model performance for {model_id}: {e}")
            return None
    
    async def store_insight(self, insight_id: str, insight: Dict[str, Any]) -> bool:
        """
        Store insight in knowledge base.
        
        Args:
            insight_id: Unique insight identifier
            insight: Insight data
            
        Returns:
            True if stored successfully
        """
        try:
            insight_with_timestamp = {
                'insight_id': insight_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data': insight
            }
            
            self.memory_storage['insights'][insight_id] = insight_with_timestamp
            
            # Store in vector DB for semantic search
            if self.vector_db and self.vector_db['type'] == 'chromadb':
                text_content = self.extract_text_from_insight(insight)
                self.vector_db['collection'].add(
                    documents=[text_content],
                    metadatas=[{'insight_id': insight_id, 'type': 'insight'}],
                    ids=[f"insight_{insight_id}"]
                )
            
            self.logger.info(f"Stored insight {insight_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store insight {insight_id}: {e}")
            return False
    
    async def search_similar_insights(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar insights using semantic search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of similar insights
        """
        try:
            if self.vector_db and self.vector_db['type'] == 'chromadb':
                results = self.vector_db['collection'].query(
                    query_texts=[query],
                    n_results=limit,
                    where={"type": "insight"}
                )
                
                insights = []
                for i, doc_id in enumerate(results['ids'][0]):
                    insight_id = doc_id.replace('insight_', '')
                    if insight_id in self.memory_storage['insights']:
                        insights.append(self.memory_storage['insights'][insight_id])
                
                return insights
            else:
                # Fallback to simple text search
                return self.simple_text_search(query, limit)
                
        except Exception as e:
            self.logger.error(f"Failed to search insights: {e}")
            return []
    
    def simple_text_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Simple text-based search fallback."""
        query_lower = query.lower()
        results = []
        
        for insight_id, insight_data in self.memory_storage['insights'].items():
            insight_text = self.extract_text_from_insight(insight_data['data'])
            if query_lower in insight_text.lower():
                results.append(insight_data)
                if len(results) >= limit:
                    break
        
        return results
    
    def extract_text_from_result(self, result: Dict[str, Any]) -> str:
        """Extract text content from project result for vector storage."""
        text_parts = []
        
        if 'research_question' in result:
            text_parts.append(f"Research question: {result['research_question']}")
        
        if 'insights' in result:
            for insight in result['insights']:
                text_parts.append(f"Insight: {insight.get('description', '')}")
        
        if 'recommendations' in result:
            for rec in result['recommendations']:
                text_parts.append(f"Recommendation: {rec.get('description', '')}")
        
        return " ".join(text_parts)
    
    def extract_text_from_insight(self, insight: Dict[str, Any]) -> str:
        """Extract text content from insight for vector storage."""
        text_parts = []
        
        if 'title' in insight:
            text_parts.append(insight['title'])
        
        if 'description' in insight:
            text_parts.append(insight['description'])
        
        if 'category' in insight:
            text_parts.append(f"Category: {insight['category']}")
        
        return " ".join(text_parts)
    
    async def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge base contents."""
        return {
            'total_projects': len(self.memory_storage['projects']),
            'total_datasets': len(self.memory_storage['datasets']),
            'total_models': len(self.memory_storage['models']),
            'total_insights': len(self.memory_storage['insights']),
            'vector_db_available': self.vector_db is not None,
            'cache_db_available': self.cache_db is not None
        }
    
    async def clear_knowledge_base(self):
        """Clear all stored knowledge (use with caution)."""
        self.memory_storage = {
            'projects': {},
            'datasets': {},
            'models': {},
            'insights': {},
            'patterns': {}
        }
        
        if self.cache_db and self.cache_db['type'] == 'redis':
            self.cache_db['client'].flushdb()
        
        if self.vector_db and self.vector_db['type'] == 'chromadb':
            # Note: This would delete the entire collection
            pass
        
        self.logger.warning("Knowledge base cleared")
    
    def get_status(self) -> Dict[str, Any]:
        """Get knowledge base status."""
        return {
            'vector_db_type': self.vector_db['type'] if self.vector_db else None,
            'cache_db_type': self.cache_db['type'] if self.cache_db else None,
            'memory_storage_size': sum(len(storage) for storage in self.memory_storage.values())
        } 