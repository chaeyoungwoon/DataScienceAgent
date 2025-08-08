"""
Data Quality Agent

Ensures dataset is consistent, complete, and analysis-ready.
Detects and removes duplicate rows, handles missing values, and ensures data types are correct.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from src.core.context_manager import (
    read_context, write_context, log_step, 
    update_context_chain, get_context_chain_data
)

class DataQualityAgent:
    """
    Data Quality Agent for cleaning and validating datasets.
    
    Detects and removes duplicate rows, identifies missing values and applies
    imputation or removal strategies, ensures data types are consistent and correct.
    """
    
    def __init__(self):
        """Initialize Data Quality Agent."""
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = Path("output/data_quality_01")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cleaned_data_dir = Path("data/cleaned")
        self.cleaned_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Data Quality Agent initialized")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute data quality task.
        
        Returns:
            Dict containing quality results and metadata
        """
        try:
            # Read context
            context = read_context()
            
            # Get dataset paths from data acquisition
            acquisition_data = get_context_chain_data(context, 'data_acquisition')
            if not acquisition_data or 'file_paths' not in acquisition_data:
                raise ValueError("No file paths found in context")
            
            file_paths = acquisition_data['file_paths']
            self.logger.info(f"Starting data quality check for {len(file_paths)} files")
            
            # Process each file
            quality_results = {
                'processing_timestamp': datetime.now().isoformat(),
                'files_processed': [],
                'total_files': len(file_paths),
                'successful_cleanings': 0,
                'failed_cleanings': 0,
                'cleaned_file_paths': [],
                'quality_metrics': {}
            }
            
            for file_path in file_paths:
                try:
                    file_result = self._process_file(file_path)
                    quality_results['files_processed'].append(file_result)
                    quality_results['successful_cleanings'] += 1
                    
                    if 'cleaned_path' in file_result:
                        quality_results['cleaned_file_paths'].append(file_result['cleaned_path'])
                    
                except Exception as e:
                    self.logger.error(f"Failed to process file {file_path}: {e}")
                    quality_results['failed_cleanings'] += 1
                    quality_results['files_processed'].append({
                        'original_path': file_path,
                        'error': str(e),
                        'status': 'failed'
                    })
            
            # Calculate overall quality metrics
            quality_results['quality_metrics'] = self._calculate_overall_metrics(quality_results['files_processed'])
            
            # Convert numpy types to native Python types for JSON serialization
            quality_results = self._convert_numpy_types(quality_results)
            
            # Update context
            update_context_chain(context, 'data_quality', quality_results)
            
            # Log completion
            log_step(context, 'data_quality', 
                    f"Cleaned {quality_results['successful_cleanings']} files successfully")
            
            # Write updated context
            write_context(context)
            
            # Save detailed results
            self._save_quality_results(quality_results)
            
            return quality_results
            
        except Exception as e:
            self.logger.error(f"Data quality check failed: {e}")
            # Update context with error
            context = read_context()
            update_context_chain(context, 'data_quality', {'error': str(e)})
            log_step(context, 'data_quality', f"Error: {str(e)}")
            write_context(context)
            raise
    
    def _process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file for data quality.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dict containing processing results
        """
        # Construct full path
        full_path = Path("data/raw") / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        
        self.logger.info(f"Processing file: {file_path}")
        
        # Load data
        data = self._load_data_file(full_path)
        if data is None:
            raise ValueError(f"Could not load data from {file_path}")
        
        # Analyze data quality
        quality_analysis = self._analyze_data_quality(data)
        
        # Clean data
        cleaned_data = self._clean_data(data, quality_analysis)
        
        # Save cleaned data
        cleaned_path = self._save_cleaned_data(cleaned_data, file_path)
        
        result = {
            'original_path': file_path,
            'cleaned_path': str(cleaned_path),
            'status': 'success',
            'quality_analysis': quality_analysis,
            'cleaning_summary': {
                'original_rows': len(data),
                'cleaned_rows': len(cleaned_data),
                'duplicates_removed': quality_analysis['duplicates_count'],
                'missing_values_handled': quality_analysis['missing_values_total'],
                'data_types_corrected': quality_analysis['data_type_issues']
            }
        }
        
        return result
    
    def _load_data_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load data file based on extension.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame or None if failed
        """
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.csv':
                return pd.read_csv(file_path)
            elif file_ext == '.json':
                return pd.read_json(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif file_ext == '.parquet':
                return pd.read_parquet(file_path)
            elif file_ext in ['.txt', '.tsv']:
                return pd.read_csv(file_path, sep='\t')
            else:
                # Try to read as CSV with different separators
                return pd.read_csv(file_path, sep=None, engine='python')
                
        except Exception as e:
            self.logger.warning(f"Failed to load file {file_path}: {e}")
            return None
    
    def _analyze_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data quality issues.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dict containing quality analysis results
        """
        analysis = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'duplicates_count': data.duplicated().sum(),
            'missing_values': data.isnull().sum().to_dict(),
            'missing_values_total': data.isnull().sum().sum(),
            'data_types': data.dtypes.to_dict(),
            'data_type_issues': 0,
            'memory_usage': data.memory_usage(deep=True).sum(),
            'column_info': {}
        }
        
        # Analyze each column
        for column in data.columns:
            col_info = {
                'dtype': str(data[column].dtype),
                'missing_count': data[column].isnull().sum(),
                'missing_percentage': (data[column].isnull().sum() / len(data)) * 100,
                'unique_values': data[column].nunique(),
                'sample_values': data[column].dropna().head(3).tolist()
            }
            
            # Check for data type issues
            if data[column].dtype == 'object':
                # Try to convert to more appropriate type
                try:
                    # Try numeric conversion
                    pd.to_numeric(data[column], errors='raise')
                    col_info['suggested_dtype'] = 'numeric'
                    analysis['data_type_issues'] += 1
                except:
                    # Try datetime conversion
                    try:
                        pd.to_datetime(data[column], errors='raise')
                        col_info['suggested_dtype'] = 'datetime'
                        analysis['data_type_issues'] += 1
                    except:
                        col_info['suggested_dtype'] = 'object'
            
            analysis['column_info'][column] = col_info
        
        return analysis
    
    def _clean_data(self, data: pd.DataFrame, quality_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        Clean data based on quality analysis.
        
        Args:
            data: Original DataFrame
            quality_analysis: Quality analysis results
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        
        # Remove duplicates
        if quality_analysis['duplicates_count'] > 0:
            cleaned_data = cleaned_data.drop_duplicates()
            self.logger.info(f"Removed {quality_analysis['duplicates_count']} duplicate rows")
        
        # Handle missing values
        if quality_analysis['missing_values_total'] > 0:
            cleaned_data = self._handle_missing_values(cleaned_data, quality_analysis)
        
        # Fix data types
        if quality_analysis['data_type_issues'] > 0:
            cleaned_data = self._fix_data_types(cleaned_data, quality_analysis)
        
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame, quality_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data: DataFrame to clean
            quality_analysis: Quality analysis results
            
        Returns:
            DataFrame with missing values handled
        """
        cleaned_data = data.copy()
        
        for column in data.columns:
            missing_count = quality_analysis['missing_values'][column]
            missing_percentage = (missing_count / len(data)) * 100
            
            if missing_count > 0:
                if missing_percentage > 50:
                    # If more than 50% missing, drop the column
                    cleaned_data = cleaned_data.drop(columns=[column])
                    self.logger.info(f"Dropped column '{column}' due to {missing_percentage:.1f}% missing values")
                else:
                    # Impute missing values based on data type
                    if data[column].dtype in ['int64', 'float64']:
                        # Numeric column - use median
                        imputer = SimpleImputer(strategy='median')
                        cleaned_data[column] = imputer.fit_transform(cleaned_data[[column]])
                    else:
                        # Categorical column - use mode
                        imputer = SimpleImputer(strategy='most_frequent')
                        cleaned_data[column] = imputer.fit_transform(cleaned_data[[column]])
                    
                    self.logger.info(f"Imputed {missing_count} missing values in column '{column}'")
        
        return cleaned_data
    
    def _fix_data_types(self, data: pd.DataFrame, quality_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        Fix data types based on analysis.
        
        Args:
            data: DataFrame to fix
            quality_analysis: Quality analysis results
            
        Returns:
            DataFrame with corrected data types
        """
        cleaned_data = data.copy()
        
        for column in data.columns:
            col_info = quality_analysis['column_info'][column]
            suggested_dtype = col_info.get('suggested_dtype')
            
            if suggested_dtype == 'numeric':
                try:
                    cleaned_data[column] = pd.to_numeric(cleaned_data[column], errors='coerce')
                    self.logger.info(f"Converted column '{column}' to numeric")
                except:
                    pass
            elif suggested_dtype == 'datetime':
                try:
                    cleaned_data[column] = pd.to_datetime(cleaned_data[column], errors='coerce')
                    self.logger.info(f"Converted column '{column}' to datetime")
                except:
                    pass
        
        return cleaned_data
    
    def _save_cleaned_data(self, data: pd.DataFrame, original_path: str) -> Path:
        """
        Save cleaned data to cleaned directory.
        
        Args:
            data: Cleaned DataFrame
            original_path: Original file path
            
        Returns:
            Path to saved cleaned file
        """
        # Create filename for cleaned data
        original_name = Path(original_path).stem
        cleaned_filename = f"{original_name}_cleaned.csv"
        cleaned_path = self.cleaned_data_dir / cleaned_filename
        
        # Save as CSV
        data.to_csv(cleaned_path, index=False)
        
        self.logger.info(f"Saved cleaned data to {cleaned_path}")
        return cleaned_path
    
    def _calculate_overall_metrics(self, files_processed: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall quality metrics.
        
        Args:
            files_processed: List of file processing results
            
        Returns:
            Overall quality metrics
        """
        total_original_rows = 0
        total_cleaned_rows = 0
        total_duplicates_removed = 0
        total_missing_handled = 0
        total_type_issues = 0
        successful_files = 0
        
        for file_result in files_processed:
            if file_result.get('status') == 'success':
                successful_files += 1
                summary = file_result.get('cleaning_summary', {})
                total_original_rows += summary.get('original_rows', 0)
                total_cleaned_rows += summary.get('cleaned_rows', 0)
                total_duplicates_removed += summary.get('duplicates_removed', 0)
                total_missing_handled += summary.get('missing_values_handled', 0)
                total_type_issues += summary.get('data_types_corrected', 0)
        
        return {
            'total_files_processed': len(files_processed),
            'successful_files': successful_files,
            'total_original_rows': total_original_rows,
            'total_cleaned_rows': total_cleaned_rows,
            'total_duplicates_removed': total_duplicates_removed,
            'total_missing_handled': total_missing_handled,
            'total_type_issues_fixed': total_type_issues,
            'data_reduction_percentage': ((total_original_rows - total_cleaned_rows) / total_original_rows * 100) if total_original_rows > 0 else 0
        }
    
    def _save_quality_results(self, results: Dict[str, Any]):
        """
        Save detailed quality results to output directory.
        
        Args:
            results: Quality results dictionary
        """
        # Save full results
        results_file = self.output_dir / "quality_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary_file = self.output_dir / "quality_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Data Quality Summary\n")
            f.write(f"===================\n\n")
            f.write(f"Total Files: {results['total_files']}\n")
            f.write(f"Successful Cleanings: {results['successful_cleanings']}\n")
            f.write(f"Failed Cleanings: {results['failed_cleanings']}\n\n")
            
            metrics = results['quality_metrics']
            f.write(f"Overall Metrics:\n")
            f.write(f"- Total Original Rows: {metrics['total_original_rows']:,}\n")
            f.write(f"- Total Cleaned Rows: {metrics['total_cleaned_rows']:,}\n")
            f.write(f"- Duplicates Removed: {metrics['total_duplicates_removed']:,}\n")
            f.write(f"- Missing Values Handled: {metrics['total_missing_handled']:,}\n")
            f.write(f"- Data Type Issues Fixed: {metrics['total_type_issues_fixed']}\n")
            f.write(f"- Data Reduction: {metrics['data_reduction_percentage']:.1f}%\n\n")
            
            f.write(f"Processed Files:\n")
            for file_result in results['files_processed']:
                f.write(f"- {file_result['original_path']}\n")
                f.write(f"  Status: {file_result.get('status', 'unknown')}\n")
                if 'cleaning_summary' in file_result:
                    summary = file_result['cleaning_summary']
                    f.write(f"  Original Rows: {summary['original_rows']:,}\n")
                    f.write(f"  Cleaned Rows: {summary['cleaned_rows']:,}\n")
                    f.write(f"  Duplicates Removed: {summary['duplicates_removed']}\n")
                    f.write(f"  Missing Values Handled: {summary['missing_values_handled']}\n")
                if 'error' in file_result:
                    f.write(f"  Error: {file_result['error']}\n")
                f.write("\n")
        
        self.logger.info(f"Quality results saved to {self.output_dir}")
    
    def _convert_numpy_types(self, obj):
        """
        Convert numpy types to native Python types for JSON serialization.
        
        Args:
            obj: Object that may contain numpy types
            
        Returns:
            Object with numpy types converted to native Python types
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, np.dtype):
            return str(obj)
        elif hasattr(obj, 'dtype'):  # Handle pandas dtypes
            return str(obj)
        elif hasattr(obj, '__dict__'):  # Handle objects with __dict__
            return str(obj)
        else:
            return obj

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run agent
    agent = DataQualityAgent()
    results = agent.execute()
    print(f"Data quality check completed. Cleaned {results['successful_cleanings']} files.") 