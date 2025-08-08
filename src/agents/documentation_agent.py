"""
Documentation Agent

Automatically generates readable dataset documentation using Hugging Face Transformers.
Uses facebook/bart-large-cnn for summarization to produce concise, human-friendly descriptions.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

from transformers import BartTokenizer, BartForConditionalGeneration
import torch

from src.core.context_manager import (
    read_context, write_context, log_step, 
    update_context_chain, get_context_chain_data
)

class DocumentationAgent:
    """
    Documentation Agent for generating dataset documentation.
    
    Uses facebook/bart-large-cnn for summarization to produce concise,
    human-friendly descriptions of datasets and their contents.
    """
    
    def __init__(self):
        """Initialize Documentation Agent."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize BART model for summarization
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        
        # Create output directories
        self.output_dir = Path("output/documentation_01")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Documentation Agent initialized with model: {self.model_name}")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute documentation generation task.
        
        Returns:
            Dict containing documentation results and metadata
        """
        try:
            # Read context
            context = read_context()
            
            # Get cleaned dataset paths from data quality
            quality_data = get_context_chain_data(context, 'data_quality')
            if not quality_data or 'cleaned_file_paths' not in quality_data:
                raise ValueError("No cleaned file paths found in context")
            
            cleaned_file_paths = quality_data['cleaned_file_paths']
            self.logger.info(f"Starting documentation generation for {len(cleaned_file_paths)} files")
            
            # Generate documentation for each file
            documentation_results = {
                'generation_timestamp': datetime.now().isoformat(),
                'files_documented': [],
                'total_files': len(cleaned_file_paths),
                'successful_documentations': 0,
                'failed_documentations': 0,
                'overall_summary': ''
            }
            
            for file_path in cleaned_file_paths:
                try:
                    file_doc = self._generate_file_documentation(file_path)
                    documentation_results['files_documented'].append(file_doc)
                    documentation_results['successful_documentations'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to document file {file_path}: {e}")
                    documentation_results['failed_documentations'] += 1
                    documentation_results['files_documented'].append({
                        'file_path': file_path,
                        'error': str(e),
                        'status': 'failed'
                    })
            
            # Generate overall dataset summary
            if documentation_results['successful_documentations'] > 0:
                documentation_results['overall_summary'] = self._generate_overall_summary(
                    documentation_results['files_documented']
                )
            
            # Update context
            update_context_chain(context, 'documentation', documentation_results)
            
            # Log completion
            log_step(context, 'documentation', 
                    f"Generated documentation for {documentation_results['successful_documentations']} files")
            
            # Write updated context
            write_context(context)
            
            # Save detailed results
            self._save_documentation_results(documentation_results)
            
            return documentation_results
            
        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}")
            # Update context with error
            context = read_context()
            update_context_chain(context, 'documentation', {'error': str(e)})
            log_step(context, 'documentation', f"Error: {str(e)}")
            write_context(context)
            raise
    
    def _generate_file_documentation(self, file_path: str) -> Dict[str, Any]:
        """
        Generate documentation for a single file.
        
        Args:
            file_path: Path to the cleaned file
            
        Returns:
            Dict containing file documentation
        """
        # Load the data
        data = self._load_data_file(file_path)
        if data is None:
            raise ValueError(f"Could not load data from {file_path}")
        
        self.logger.info(f"Generating documentation for: {file_path}")
        
        # Generate column information
        column_info = self._generate_column_info(data)
        
        # Generate sample data
        sample_data = self._generate_sample_data(data)
        
        # Generate dataset summary using BART
        dataset_summary = self._generate_dataset_summary(data, column_info)
        
        # Create documentation structure
        documentation = {
            'file_path': file_path,
            'status': 'success',
            'dataset_info': {
                'shape': data.shape,
                'memory_usage': data.memory_usage(deep=True).sum(),
                'data_types': data.dtypes.to_dict(),
                'missing_values': data.isnull().sum().to_dict()
            },
            'column_info': column_info,
            'sample_data': sample_data,
            'dataset_summary': dataset_summary,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        return documentation
    
    def _load_data_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load data file from cleaned directory.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame or None if failed
        """
        try:
            full_path = Path("data/cleaned") / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {full_path}")
            
            file_ext = full_path.suffix.lower()
            
            if file_ext == '.csv':
                return pd.read_csv(full_path)
            elif file_ext == '.json':
                return pd.read_json(full_path)
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(full_path)
            elif file_ext == '.parquet':
                return pd.read_parquet(full_path)
            else:
                # Try to read as CSV with different separators
                return pd.read_csv(full_path, sep=None, engine='python')
                
        except Exception as e:
            self.logger.warning(f"Failed to load file {file_path}: {e}")
            return None
    
    def _generate_column_info(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate detailed information for each column.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            List of column information dictionaries
        """
        column_info = []
        
        for column in data.columns:
            col_data = data[column]
            
            # Basic column info
            col_info = {
                'name': column,
                'data_type': str(col_data.dtype),
                'missing_count': col_data.isnull().sum(),
                'missing_percentage': (col_data.isnull().sum() / len(data)) * 100,
                'unique_count': col_data.nunique(),
                'description': self._generate_column_description(column, col_data)
            }
            
            # Add type-specific information
            if col_data.dtype in ['int64', 'float64']:
                col_info.update({
                    'min_value': float(col_data.min()) if not col_data.empty else None,
                    'max_value': float(col_data.max()) if not col_data.empty else None,
                    'mean_value': float(col_data.mean()) if not col_data.empty else None,
                    'median_value': float(col_data.median()) if not col_data.empty else None,
                    'std_value': float(col_data.std()) if not col_data.empty else None
                })
            elif col_data.dtype == 'object':
                # For categorical/text columns
                value_counts = col_data.value_counts()
                col_info.update({
                    'top_values': value_counts.head(5).to_dict(),
                    'most_common': value_counts.index[0] if not value_counts.empty else None
                })
            elif col_data.dtype == 'datetime64[ns]':
                col_info.update({
                    'min_date': str(col_data.min()) if not col_data.empty else None,
                    'max_date': str(col_data.max()) if not col_data.empty else None
                })
            
            column_info.append(col_info)
        
        return column_info
    
    def _generate_column_description(self, column_name: str, column_data: pd.Series) -> str:
        """
        Generate a description for a column based on its content and name.
        
        Args:
            column_name: Name of the column
            column_data: Series containing column data
            
        Returns:
            Description string
        """
        # Simple heuristic-based description generation
        name_lower = column_name.lower()
        
        if 'id' in name_lower:
            return "Unique identifier for the record"
        elif 'date' in name_lower or 'time' in name_lower:
            return "Date/time information"
        elif 'price' in name_lower or 'cost' in name_lower or 'amount' in name_lower:
            return "Monetary value or price information"
        elif 'name' in name_lower or 'title' in name_lower:
            return "Name or title information"
        elif 'email' in name_lower:
            return "Email address"
        elif 'phone' in name_lower:
            return "Phone number"
        elif 'address' in name_lower:
            return "Address information"
        elif column_data.dtype in ['int64', 'float64']:
            return "Numeric value"
        elif column_data.dtype == 'object':
            return "Text or categorical information"
        else:
            return "Data field"
    
    def _generate_sample_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate sample data for documentation.
        
        Args:
            data: DataFrame to sample
            
        Returns:
            Dict containing sample data information
        """
        # Take a sample of the data
        sample_size = min(5, len(data))
        sample = data.head(sample_size)
        
        return {
            'sample_size': sample_size,
            'sample_data': sample.to_dict('records'),
            'total_rows': len(data)
        }
    
    def _generate_dataset_summary(self, data: pd.DataFrame, column_info: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of the dataset using BART model.
        
        Args:
            data: DataFrame to summarize
            column_info: Information about columns
            
        Returns:
            Generated summary text
        """
        try:
            # Create a text description of the dataset
            dataset_description = self._create_dataset_description(data, column_info)
            
            # Use BART to generate summary
            inputs = self.tokenizer(dataset_description, max_length=1024, truncation=True, return_tensors="pt")
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"], 
                    max_length=150, 
                    min_length=40, 
                    length_penalty=2.0, 
                    num_beams=4, 
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"BART summarization failed: {e}")
            # Fallback to heuristic summary
            return self._create_heuristic_summary(data, column_info)
    
    def _create_dataset_description(self, data: pd.DataFrame, column_info: List[Dict[str, Any]]) -> str:
        """
        Create a text description of the dataset for summarization.
        
        Args:
            data: DataFrame to describe
            column_info: Information about columns
            
        Returns:
            Text description
        """
        description = f"This dataset contains {len(data)} rows and {len(data.columns)} columns. "
        
        # Add column information
        numeric_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
        categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
        datetime_cols = [col for col in data.columns if data[col].dtype == 'datetime64[ns]']
        
        if numeric_cols:
            description += f"It has {len(numeric_cols)} numeric columns including {', '.join(numeric_cols[:3])}. "
        
        if categorical_cols:
            description += f"It has {len(categorical_cols)} categorical columns including {', '.join(categorical_cols[:3])}. "
        
        if datetime_cols:
            description += f"It has {len(datetime_cols)} datetime columns including {', '.join(datetime_cols)}. "
        
        # Add missing value information
        total_missing = data.isnull().sum().sum()
        if total_missing > 0:
            description += f"The dataset has {total_missing} missing values across all columns. "
        
        # Add data type information
        description += f"The data types include {', '.join(set(str(dtype) for dtype in data.dtypes))}. "
        
        return description
    
    def _create_heuristic_summary(self, data: pd.DataFrame, column_info: List[Dict[str, Any]]) -> str:
        """
        Create a heuristic summary when BART fails.
        
        Args:
            data: DataFrame to summarize
            column_info: Information about columns
            
        Returns:
            Heuristic summary text
        """
        summary = f"This dataset contains {len(data)} records with {len(data.columns)} columns. "
        
        # Count data types
        numeric_count = len([col for col in data.columns if data[col].dtype in ['int64', 'float64']])
        categorical_count = len([col for col in data.columns if data[col].dtype == 'object'])
        datetime_count = len([col for col in data.columns if data[col].dtype == 'datetime64[ns]'])
        
        summary += f"It includes {numeric_count} numeric, {categorical_count} categorical, and {datetime_count} datetime columns. "
        # Add key insights
        missing_total = data.isnull().sum().sum()
        if missing_total > 0:
            summary += f"There are {missing_total} missing values that may need attention. "
        
        return summary
    
    def _generate_overall_summary(self, file_documentations: List[Dict[str, Any]]) -> str:
        """
        Generate an overall summary of all documented files.
        
        Args:
            file_documentations: List of file documentation results
            
        Returns:
            Overall summary text
        """
        try:
            # Combine all dataset summaries
            all_summaries = []
            for doc in file_documentations:
                if doc.get('status') == 'success' and 'dataset_summary' in doc:
                    all_summaries.append(doc['dataset_summary'])
            
            if not all_summaries:
                return "No dataset summaries available."
            
            # Combine summaries
            combined_text = " ".join(all_summaries)
            
            # Use BART to generate overall summary
            inputs = self.tokenizer(combined_text, max_length=1024, truncation=True, return_tensors="pt")
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"], 
                    max_length=200, 
                    min_length=50, 
                    length_penalty=2.0, 
                    num_beams=4, 
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"Overall summary generation failed: {e}")
            return f"Successfully documented {len([d for d in file_documentations if d.get('status') == 'success'])} datasets."
    
    def _save_documentation_results(self, results: Dict[str, Any]):
        """
        Save detailed documentation results to output directory.
        
        Args:
            results: Documentation results dictionary
        """
        # Save full results
        results_file = self.output_dir / "documentation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary_file = self.output_dir / "documentation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Dataset Documentation Summary\n")
            f.write(f"===========================\n\n")
            f.write(f"Total Files: {results['total_files']}\n")
            f.write(f"Successful Documentations: {results['successful_documentations']}\n")
            f.write(f"Failed Documentations: {results['failed_documentations']}\n\n")
            
            f.write(f"Overall Summary:\n")
            f.write(f"{results['overall_summary']}\n\n")
            
            f.write(f"Documented Files:\n")
            for doc in results['files_documented']:
                f.write(f"- {doc['file_path']}\n")
                f.write(f"  Status: {doc.get('status', 'unknown')}\n")
                if 'dataset_info' in doc:
                    info = doc['dataset_info']
                    f.write(f"  Shape: {info['shape']}\n")
                    f.write(f"  Columns: {len(info['data_types'])}\n")
                if 'error' in doc:
                    f.write(f"  Error: {doc['error']}\n")
                f.write("\n")
        
        # Save individual documentation files
        docs_dir = self.output_dir / "documentation"
        docs_dir.mkdir(exist_ok=True)
        
        for doc in results['files_documented']:
            if doc.get('status') == 'success':
                filename = Path(doc['file_path']).stem + "_documentation.json"
                doc_file = docs_dir / filename
                with open(doc_file, 'w') as f:
                    json.dump(doc, f, indent=2)
        
        self.logger.info(f"Documentation results saved to {self.output_dir}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run agent
    agent = DocumentationAgent()
    results = agent.execute()
    print(f"Documentation generation completed. Documented {results['successful_documentations']} files.") 