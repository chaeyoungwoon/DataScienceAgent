"""
Data Acquisition Agent

Downloads and prepares datasets for local processing using Kaggle API.
Handles large file downloads with streaming and maintains file paths and types.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import zipfile
import shutil

import kaggle
from dotenv import load_dotenv
import pandas as pd

from src.core.context_manager import (
    read_context, write_context, log_step, 
    update_context_chain, get_context_chain_data
)

# Load environment variables
load_dotenv()

class DataAcquisitionAgent:
    """
    Data Acquisition Agent for downloading and preparing datasets.
    
    Downloads datasets using Kaggle API, unzips them into data/raw/,
    and maintains a list of local file paths and types.
    """
    
    def __init__(self):
        """Initialize Data Acquisition Agent."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize Kaggle API
        self._setup_kaggle_api()
        
        # Create output directories
        self.output_dir = Path("output/data_acquisition_01")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_data_dir = Path("data/raw")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Data Acquisition Agent initialized")
    
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
        Execute data acquisition task.
        
        Returns:
            Dict containing acquisition results and metadata
        """
        try:
            # Read context
            context = read_context()
            
            # Get selected datasets from dataset discovery
            discovery_data = get_context_chain_data(context, 'dataset_discovery')
            if not discovery_data or 'selected_datasets' not in discovery_data:
                raise ValueError("No selected datasets found in context")
            
            selected_datasets = discovery_data['selected_datasets']
            self.logger.info(f"Starting data acquisition for {len(selected_datasets)} datasets")
            
            # Download datasets
            acquisition_results = {
                'download_timestamp': datetime.now().isoformat(),
                'datasets_downloaded': [],
                'total_datasets': len(selected_datasets),
                'successful_downloads': 0,
                'failed_downloads': 0,
                'file_paths': [],
                'file_types': []
            }
            
            for dataset in selected_datasets:
                try:
                    dataset_result = self._download_dataset(dataset)
                    acquisition_results['datasets_downloaded'].append(dataset_result)
                    acquisition_results['successful_downloads'] += 1
                    
                    # Add file paths and types
                    if 'local_files' in dataset_result:
                        acquisition_results['file_paths'].extend(dataset_result['local_files'])
                        acquisition_results['file_types'].extend(dataset_result['file_types'])
                        
                except Exception as e:
                    self.logger.error(f"Failed to download dataset {dataset['ref']}: {e}")
                    acquisition_results['failed_downloads'] += 1
                    acquisition_results['datasets_downloaded'].append({
                        'ref': dataset['ref'],
                        'title': dataset['title'],
                        'error': str(e),
                        'status': 'failed'
                    })
            
            # Update context
            update_context_chain(context, 'data_acquisition', acquisition_results)
            
            # Log completion
            log_step(context, 'data_acquisition', 
                    f"Downloaded {acquisition_results['successful_downloads']} datasets successfully")
            
            # Write updated context
            write_context(context)
            
            # Save detailed results
            self._save_acquisition_results(acquisition_results)
            
            return acquisition_results
            
        except Exception as e:
            self.logger.error(f"Data acquisition failed: {e}")
            # Update context with error
            context = read_context()
            update_context_chain(context, 'data_acquisition', {'error': str(e)})
            log_step(context, 'data_acquisition', f"Error: {str(e)}")
            write_context(context)
            raise
    
    def _download_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Download a single dataset from Kaggle.
        
        Args:
            dataset: Dataset information from discovery
            
        Returns:
            Dict containing download results
        """
        dataset_ref = dataset['ref']
        dataset_title = dataset['title']
        
        self.logger.info(f"Downloading dataset: {dataset_title} ({dataset_ref})")
        
        # Create dataset-specific directory
        dataset_dir = self.raw_data_dir / dataset_ref.replace('/', '_')
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            # Download dataset
            kaggle.api.dataset_download_files(
                dataset_ref,
                path=str(dataset_dir),
                unzip=True
            )
            
            # Analyze downloaded files
            local_files, file_types = self._analyze_downloaded_files(dataset_dir)
            
            result = {
                'ref': dataset_ref,
                'title': dataset_title,
                'status': 'success',
                'download_path': str(dataset_dir),
                'local_files': local_files,
                'file_types': file_types,
                'file_count': len(local_files),
                'download_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Successfully downloaded {len(local_files)} files for {dataset_title}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to download {dataset_ref}: {e}")
            raise
    
    def _analyze_downloaded_files(self, dataset_dir: Path) -> tuple[List[str], List[str]]:
        """
        Analyze downloaded files to determine paths and types.
        
        Args:
            dataset_dir: Directory containing downloaded files
            
        Returns:
            Tuple of (file_paths, file_types)
        """
        file_paths = []
        file_types = []
        
        # Walk through all files in the dataset directory
        for file_path in dataset_dir.rglob('*'):
            if file_path.is_file():
                # Get relative path from raw data directory
                relative_path = str(file_path.relative_to(self.raw_data_dir))
                file_paths.append(relative_path)
                
                # Determine file type
                file_type = self._determine_file_type(file_path)
                file_types.append(file_type)
        
        return file_paths, file_types
    
    def _determine_file_type(self, file_path: Path) -> str:
        """
        Determine the type of a file based on its extension and content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type string
        """
        extension = file_path.suffix.lower()
        
        # Common data file extensions
        if extension in ['.csv']:
            return 'csv'
        elif extension in ['.xlsx', '.xls']:
            return 'excel'
        elif extension in ['.json']:
            return 'json'
        elif extension in ['.parquet']:
            return 'parquet'
        elif extension in ['.txt', '.tsv']:
            return 'text'
        elif extension in ['.zip', '.tar', '.gz']:
            return 'archive'
        else:
            # Try to determine type by reading first few bytes
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(10)
                    
                # Check for common file signatures
                if header.startswith(b'PK'):
                    return 'zip'
                elif header.startswith(b'\x1f\x8b'):
                    return 'gzip'
                elif header.startswith(b'{"'):
                    return 'json'
                else:
                    return 'unknown'
            except:
                return 'unknown'
    
    def _save_acquisition_results(self, results: Dict[str, Any]):
        """
        Save detailed acquisition results to output directory.
        
        Args:
            results: Acquisition results dictionary
        """
        # Save full results
        results_file = self.output_dir / "acquisition_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary_file = self.output_dir / "acquisition_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Data Acquisition Summary\n")
            f.write(f"=======================\n\n")
            f.write(f"Total Datasets: {results['total_datasets']}\n")
            f.write(f"Successful Downloads: {results['successful_downloads']}\n")
            f.write(f"Failed Downloads: {results['failed_downloads']}\n")
            f.write(f"Total Files: {len(results['file_paths'])}\n\n")
            
            f.write(f"Downloaded Datasets:\n")
            for dataset in results['datasets_downloaded']:
                f.write(f"- {dataset['title']} ({dataset['ref']})\n")
                f.write(f"  Status: {dataset.get('status', 'unknown')}\n")
                if 'local_files' in dataset:
                    f.write(f"  Files: {len(dataset['local_files'])}\n")
                if 'error' in dataset:
                    f.write(f"  Error: {dataset['error']}\n")
                f.write("\n")
            
            f.write(f"File Types Found:\n")
            type_counts = {}
            for file_type in results['file_types']:
                type_counts[file_type] = type_counts.get(file_type, 0) + 1
            
            for file_type, count in type_counts.items():
                f.write(f"- {file_type}: {count} files\n")
        
        self.logger.info(f"Acquisition results saved to {self.output_dir}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run agent
    agent = DataAcquisitionAgent()
    results = agent.execute()
    print(f"Data acquisition completed. Downloaded {results['successful_downloads']} datasets.") 