"""
Secure Configuration Manager for Data Science Agent Swarm

Handles API keys, security settings, and download limits to protect system resources.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
import hashlib
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    max_download_size_mb: int = 100
    cache_enabled: bool = True
    log_level: str = "INFO"
    api_rate_limit_per_minute: int = 60
    max_concurrent_downloads: int = 3
    allowed_file_extensions: tuple = ('.csv', '.json', '.parquet', '.xlsx', '.xls')
    max_dataset_size_mb: int = 50

@dataclass
class APIConfig:
    """Configuration settings for local models and external APIs."""
    local_models: Dict[str, str] = None
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None
    github_token: Optional[str] = None

class ConfigManager:
    """
    Secure configuration manager with download limits and security features.
    """
    
    def __init__(self, config_path: str = "config/"):
        """Initialize configuration manager."""
        self.config_path = Path(config_path)
        self.config_path.mkdir(exist_ok=True)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize configurations
        self.security = SecurityConfig()
        self.api = APIConfig()
        
        # Load configurations
        self._load_api_config()
        self._load_security_config()
        
        # Setup logging
        self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Configuration manager initialized")
    
    def _load_api_config(self):
        """Load configuration for local models and external APIs."""
        # Local model configuration
        self.api.local_models = {
            'text_generation': os.getenv('TEXT_GENERATION_MODEL', 'microsoft/DialoGPT-medium'),
            'summarization': os.getenv('SUMMARIZATION_MODEL', 'facebook/bart-large-cnn'),
            'classification': os.getenv('CLASSIFICATION_MODEL', 'distilbert-base-uncased'),
            'code_generation': os.getenv('CODE_GENERATION_MODEL', 'Salesforce/codegen-350M-mono')
        }
        
        # External API configuration (for Kaggle, GitHub, etc.)
        self.api.kaggle_username = os.getenv('KAGGLE_USERNAME')
        self.api.kaggle_key = os.getenv('KAGGLE_KEY')
        self.api.github_token = os.getenv('GITHUB_TOKEN')
        
        # Debug logging
        logger = logging.getLogger(__name__)
        logger.info(f"Local Models Config loaded - Text Generation: {self.api.local_models['text_generation']}")
        logger.info(f"Local Models Config loaded - Summarization: {self.api.local_models['summarization']}")
        logger.info(f"Local Models Config loaded - Classification: {self.api.local_models['classification']}")
        logger.info(f"Local Models Config loaded - Code Generation: {self.api.local_models['code_generation']}")
        logger.info(f"API Config loaded - Kaggle Username: {'✓' if self.api.kaggle_username else '✗'}")
        logger.info(f"API Config loaded - Kaggle Key: {'✓' if self.api.kaggle_key else '✗'}")
    
    def _load_security_config(self):
        """Load security configuration."""
        self.security.max_download_size_mb = int(os.getenv('MAX_DOWNLOAD_SIZE_MB', 100))
        self.security.cache_enabled = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        self.security.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.security.api_rate_limit_per_minute = int(os.getenv('API_RATE_LIMIT_PER_MINUTE', 60))
        self.security.max_concurrent_downloads = int(os.getenv('MAX_CONCURRENT_DOWNLOADS', 3))
        self.security.max_dataset_size_mb = int(os.getenv('MAX_DATASET_SIZE_MB', 50))
    
    def _setup_logging(self):
        """Setup secure logging configuration."""
        log_level = getattr(logging, self.security.log_level.upper())
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(logs_dir / 'app.log')
            ]
        )
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for local models and external APIs."""
        return {
            'local_models': self.api.local_models,
            'kaggle': {
                'username': self.api.kaggle_username,
                'key': self.api.kaggle_key
            },
            'github_token': self.api.github_token
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration dictionary."""
        return {
            'max_download_size_mb': self.security.max_download_size_mb,
            'cache_enabled': self.security.cache_enabled,
            'api_rate_limit_per_minute': self.security.api_rate_limit_per_minute,
            'max_concurrent_downloads': self.security.max_concurrent_downloads,
            'allowed_file_extensions': self.security.allowed_file_extensions,
            'max_dataset_size_mb': self.security.max_dataset_size_mb
        }
    
    def validate_download_size(self, size_bytes: int) -> bool:
        """Validate if download size is within limits."""
        size_mb = size_bytes / (1024 * 1024)
        return size_mb <= self.security.max_download_size_mb
    
    def validate_file_extension(self, filename: str) -> bool:
        """Validate if file extension is allowed."""
        return any(filename.lower().endswith(ext) for ext in self.security.allowed_file_extensions)
    
    def get_cache_dir(self) -> Path:
        """Get cache directory path."""
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        return cache_dir
    
    def is_api_available(self, api_name: str) -> bool:
        """Check if specific API is available."""
        api_configs = {
            'kaggle': bool(self.api.kaggle_username and self.api.kaggle_key),
            'github': bool(self.api.github_token),
            'local_models': bool(self.api.local_models)
        }
        return api_configs.get(api_name, False)
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get complete system configuration."""
        return {
            'kafka_servers': ['localhost:9092'],
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'agents': {
                'max_concurrent': 10,
                'timeout': 3600
            },
            **self.get_api_config(),
            **self.get_security_config()
        }
