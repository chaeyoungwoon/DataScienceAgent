"""
Secure Download Manager for Data Science Agent Swarm

Handles secure downloads with size limits, caching, and file validation.
"""

import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import Dict, Any
import hashlib
import json
from datetime import datetime, timedelta


class SecureDownloadManager:
    """
    Secure download manager with size limits, caching, and validation.
    """
    
    def __init__(self, config_manager):
        """Initialize secure download manager."""
        self.config = config_manager
        self.cache_dir = self.config.get_cache_dir()
        self.logger = logging.getLogger(__name__)
        self.download_semaphore = asyncio.Semaphore(
            self.config.security.max_concurrent_downloads
        )
        
        # Create cache metadata file
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = self._load_cache_metadata()
        
        self.logger.info("Secure download manager initialized")
    
    def _load_cache_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_cache_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, url: str, filename: str) -> str:
        """Generate cache key for URL and filename."""
        key_string = f"{url}_{filename}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached file is still valid."""
        if cache_key not in self.cache_metadata:
            return False
        
        metadata = self.cache_metadata[cache_key]
        cache_time = datetime.fromisoformat(metadata['timestamp'])
        max_age = timedelta(hours=24)  # Cache for 24 hours
        
        return datetime.now() - cache_time < max_age
    
    async def download_file(
        self, 
        url: str, 
        filename: str,
        source_type: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Download file with security checks and caching.
        
        Args:
            url: Download URL
            filename: Target filename
            source_type: Source type (kaggle, github, etc.)
            
        Returns:
            Dictionary with download result
        """
        # Validate file extension
        if not self.config.validate_file_extension(filename):
            return {
                'status': 'error',
                'error': f'File extension not allowed: {filename}',
                'file_path': None
            }
        
        # Check cache first
        cache_key = self._get_cache_key(url, filename)
        cached_path = self.cache_dir / f"{cache_key}_{filename}"
        
        if self.config.security.cache_enabled and cached_path.exists():
            if self._is_cache_valid(cache_key):
                self.logger.info(f"Using cached file: {filename}")
                return {
                    'status': 'success',
                    'file_path': str(cached_path),
                    'cached': True,
                    'size_bytes': cached_path.stat().st_size
                }
        
        # Download with size validation
        async with self.download_semaphore:
            try:
                return await self._perform_download(url, filename, cache_key, cached_path)
            except Exception as e:
                self.logger.error(f"Download failed: {e}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'file_path': None
                }
    
    async def _perform_download(
        self, 
        url: str, 
        filename: str, 
        cache_key: str, 
        cached_path: Path
    ) -> Dict[str, Any]:
        """Perform the actual download with security checks."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return {
                        'status': 'error',
                        'error': f'HTTP {response.status}: {response.reason}',
                        'file_path': None
                    }
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length:
                    size_bytes = int(content_length)
                    if not self.config.validate_download_size(size_bytes):
                        return {
                            'status': 'error',
                            'error': f'File too large: {size_bytes / (1024*1024):.1f}MB',
                            'file_path': None
                        }
                
                # Download with progress tracking
                downloaded_size = 0
                chunks = []
                
                async for chunk in response.content.iter_chunked(8192):
                    downloaded_size += len(chunk)
                    chunks.append(chunk)
                    
                    # Check size during download
                    if not self.config.validate_download_size(downloaded_size):
                        return {
                            'status': 'error',
                            'error': f'File too large during download: {downloaded_size / (1024*1024):.1f}MB',
                            'file_path': None
                        }
                
                # Save file
                file_content = b''.join(chunks)
                cached_path.write_bytes(file_content)
                
                # Update cache metadata
                self.cache_metadata[cache_key] = {
                    'url': url,
                    'filename': filename,
                    'timestamp': datetime.now().isoformat(),
                    'size_bytes': downloaded_size,
                    'source_type': source_type
                }
                self._save_cache_metadata()
                
                self.logger.info(f"Downloaded: {filename} ({downloaded_size / (1024*1024):.1f}MB)")
                
                return {
                    'status': 'success',
                    'file_path': str(cached_path),
                    'cached': False,
                    'size_bytes': downloaded_size
                }
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cached files."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        files_to_remove = []
        
        for cache_key, metadata in self.cache_metadata.items():
            cache_time = datetime.fromisoformat(metadata['timestamp'])
            if cache_time < cutoff_time:
                filename = metadata['filename']
                cached_path = self.cache_dir / f"{cache_key}_{filename}"
                if cached_path.exists():
                    files_to_remove.append((cache_key, cached_path))
        
        for cache_key, file_path in files_to_remove:
            try:
                file_path.unlink()
                del self.cache_metadata[cache_key]
                self.logger.info(f"Removed cached file: {file_path.name}")
            except Exception as e:
                self.logger.error(f"Failed to remove cached file {file_path}: {e}")
        
        self._save_cache_metadata()
        self.logger.info(f"Cache cleanup completed. Removed {len(files_to_remove)} files")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = 0
        file_count = 0
        
        for metadata in self.cache_metadata.values():
            total_size += metadata.get('size_bytes', 0)
            file_count += 1
        
        return {
            'file_count': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }
