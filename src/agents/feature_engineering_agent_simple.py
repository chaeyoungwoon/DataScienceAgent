"""
Feature Engineering Agent - Simple Implementation

Prepares features for modeling through transformation and encoding.
Follows the master specification for agent implementation.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.core.context_manager import (
    read_context, write_context, log_step, 
    update_context_chain, get_context_chain_data
)

class FeatureEngineeringAgent:
    """
    Feature Engineering Agent for data transformation and encoding.
    
    Encodes categorical variables, scales numeric variables, and generates
    new features for machine learning modeling.
    """
    
    def __init__(self):
        """Initialize Feature Engineering Agent."""
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = Path("output/feature_engineering_01")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.processed_data_dir = Path("data/processed")
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Feature Engineering Agent initialized")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute feature engineering task.
        
        Returns:
            Dict containing feature engineering results and metadata
        """
        try:
            # Read context
            context = read_context()
            
            # Get cleaned dataset paths from data quality
            quality_data = get_context_chain_data(context, 'data_quality')
            if not quality_data or 'cleaned_file_paths' not in quality_data:
                raise ValueError("No cleaned file paths found in context")
            
            cleaned_file_paths = quality_data['cleaned_file_paths']
            self.logger.info(f"Starting feature engineering for {len(cleaned_file_paths)} files")
            
            # Process each file
            feature_results = {
                'processing_timestamp': datetime.now().isoformat(),
                'files_processed': [],
                'total_files': len(cleaned_file_paths),
                'successful_transformations': 0,
                'failed_transformations': 0,
                'processed_file_paths': [],
                'feature_info': {}
            }
            
            for file_path in cleaned_file_paths:
                try:
                    file_result = self._process_file(file_path)
                    feature_results['files_processed'].append(file_result)
                    feature_results['successful_transformations'] += 1
                    
                    if 'processed_path' in file_result:
                        feature_results['processed_file_paths'].append(file_result['processed_path'])
                    
                    if 'feature_info' in file_result:
                        feature_results['feature_info'][file_path] = file_result['feature_info']
                    
                except Exception as e:
                    self.logger.error(f"Failed to process file {file_path}: {e}")
                    feature_results['failed_transformations'] += 1
                    feature_results['files_processed'].append({
                        'original_path': file_path,
                        'error': str(e),
                        'status': 'failed'
                    })
            
            # Convert numpy types for JSON serialization
            feature_results = self._convert_numpy_types(feature_results)
            
            # Update context
            update_context_chain(context, 'feature_engineering', feature_results)
            
            # Log completion
            log_step(context, 'feature_engineering', 
                    f"Processed {feature_results['successful_transformations']} files successfully")
            write_context(context)
            
            # Save detailed results
            self._save_feature_results(feature_results)
            
            self.logger.info(f"Feature engineering completed successfully for {feature_results['successful_transformations']} files")
            
            return feature_results
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            
            # Log error in context
            context = read_context()
            error_data = {
                'error': str(e),
                'status': 'failed',
                'processing_timestamp': datetime.now().isoformat()
            }
            update_context_chain(context, 'feature_engineering', error_data)
            log_step(context, 'feature_engineering', f"Error: {str(e)}")
            write_context(context)
            
            raise
    
    def _process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file for feature engineering.
        
        Args:
            file_path: Path to the cleaned data file
            
        Returns:
            Dict containing processing results
        """
        # Load data
        data = pd.read_csv(file_path)
        self.logger.info(f"Processing features for file: {file_path} with {len(data)} rows")
        
        # Analyze data types
        feature_info = self._analyze_features(data)
        
        # Transform features
        transformed_data = self._transform_features(data, feature_info)
        
        # Save processed data
        processed_path = self._save_processed_data(transformed_data, file_path)
        
        result = {
            'original_path': file_path,
            'processed_path': str(processed_path),
            'status': 'success',
            'feature_info': feature_info,
            'transformation_summary': {
                'original_shape': data.shape,
                'processed_shape': transformed_data.shape,
                'categorical_encoded': feature_info['categorical_columns'],
                'numeric_scaled': feature_info['numeric_columns'],
                'new_features_created': len(feature_info['new_features'])
            }
        }
        
        return result
    
    def _analyze_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze features and determine transformation strategy."""
        feature_info = {
            'numeric_columns': [],
            'categorical_columns': [],
            'binary_columns': [],
            'new_features': [],
            'transformation_plan': {}
        }
        
        # Identify column types
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                feature_info['numeric_columns'].append(col)
            elif data[col].dtype == 'object':
                unique_count = data[col].nunique()
                if unique_count == 2:
                    feature_info['binary_columns'].append(col)
                else:
                    feature_info['categorical_columns'].append(col)
        
        # Create transformation plan
        feature_info['transformation_plan'] = {
            'scale_numeric': feature_info['numeric_columns'],
            'encode_categorical': feature_info['categorical_columns'],
            'encode_binary': feature_info['binary_columns']
        }
        
        return feature_info
    
    def _transform_features(self, data: pd.DataFrame, feature_info: Dict[str, Any]) -> pd.DataFrame:
        """Transform features according to the analysis."""
        transformed_data = data.copy()
        
        # Handle binary columns
        for col in feature_info['binary_columns']:
            if transformed_data[col].dtype == 'object':
                # Convert to numeric (0/1)
                transformed_data[col] = (transformed_data[col] == transformed_data[col].mode()[0]).astype(int)
        
        # Handle categorical columns
        for col in feature_info['categorical_columns']:
            if transformed_data[col].dtype == 'object':
                # Use label encoding for simplicity
                le = LabelEncoder()
                transformed_data[col] = le.fit_transform(transformed_data[col].astype(str))
        
        # Scale numeric columns
        numeric_cols = feature_info['numeric_columns']
        if numeric_cols:
            scaler = StandardScaler()
            transformed_data[numeric_cols] = scaler.fit_transform(transformed_data[numeric_cols])
        
        # Create new features
        transformed_data = self._create_new_features(transformed_data, feature_info)
        
        return transformed_data
    
    def _create_new_features(self, data: pd.DataFrame, feature_info: Dict[str, Any]) -> pd.DataFrame:
        """Create new features from existing ones."""
        new_features = []
        
        # Create interaction features for numeric columns
        numeric_cols = feature_info['numeric_columns']
        if len(numeric_cols) >= 2:
            # Create ratio features
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    if data[col2].abs().min() > 0:  # Avoid division by zero
                        ratio_name = f"{col1}_div_{col2}"
                        data[ratio_name] = data[col1] / data[col2]
                        new_features.append(ratio_name)
        
        # Create polynomial features for important numeric columns
        if numeric_cols:
            # Use first numeric column for polynomial features
            col = numeric_cols[0]
            data[f"{col}_squared"] = data[col] ** 2
            new_features.append(f"{col}_squared")
        
        feature_info['new_features'] = new_features
        return data
    
    def _save_processed_data(self, data: pd.DataFrame, original_path: str) -> Path:
        """Save processed data to file."""
        file_name = Path(original_path).stem
        processed_path = self.processed_data_dir / f"{file_name}_processed.csv"
        
        data.to_csv(processed_path, index=False)
        self.logger.info(f"Saved processed data to {processed_path}")
        
        return processed_path
    
    def _save_feature_results(self, results: Dict[str, Any]):
        """Save detailed feature engineering results to output directory."""
        # Save full results
        results_file = self.output_dir / "feature_engineering_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = self.output_dir / "feature_engineering_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Feature Engineering Summary\n")
            f.write(f"==========================\n\n")
            f.write(f"Processing Timestamp: {results['processing_timestamp']}\n")
            f.write(f"Total Files: {results['total_files']}\n")
            f.write(f"Successful Transformations: {results['successful_transformations']}\n")
            f.write(f"Failed Transformations: {results['failed_transformations']}\n\n")
            
            f.write(f"Processed Files:\n")
            for file_result in results['files_processed']:
                f.write(f"- {file_result['original_path']}\n")
                f.write(f"  Status: {file_result.get('status', 'unknown')}\n")
                if 'transformation_summary' in file_result:
                    summary = file_result['transformation_summary']
                    f.write(f"  Original Shape: {summary['original_shape']}\n")
                    f.write(f"  Processed Shape: {summary['processed_shape']}\n")
                    f.write(f"  Categorical Encoded: {summary['categorical_encoded']}\n")
                    f.write(f"  Numeric Scaled: {summary['numeric_scaled']}\n")
                    f.write(f"  New Features: {summary['new_features_created']}\n")
                if 'error' in file_result:
                    f.write(f"  Error: {file_result['error']}\n")
                f.write("\n")
        
        self.logger.info(f"Feature engineering results saved to {self.output_dir}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
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
        elif isinstance(obj, bool):
            return bool(obj)
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
    agent = FeatureEngineeringAgent()
    results = agent.execute()
    print(f"Feature engineering completed. Processed {results['successful_transformations']} files.")
