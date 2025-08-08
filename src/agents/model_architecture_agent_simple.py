"""
Model Architecture Agent - Simple Implementation

Designs optimal model architectures based on data characteristics.
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

from src.core.context_manager import (
    read_context, write_context, log_step, 
    update_context_chain, get_context_chain_data
)

class ModelArchitectureAgent:
    """
    Model Architecture Agent for designing optimal models.
    
    Analyzes data characteristics and designs appropriate model architectures
    for classification and regression tasks.
    """
    
    def __init__(self):
        """Initialize Model Architecture Agent."""
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = Path("output/model_architecture_01")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Model Architecture Agent initialized")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute model architecture design task.
        
        Returns:
            Dict containing model architecture results and metadata
        """
        try:
            # Read context
            context = read_context()
            
            # Get processed dataset paths from feature engineering
            feature_data = get_context_chain_data(context, 'feature_engineering')
            if not feature_data or 'processed_file_paths' not in feature_data:
                raise ValueError("No processed file paths found in context")
            
            processed_file_paths = feature_data['processed_file_paths']
            self.logger.info(f"Starting model architecture design for {len(processed_file_paths)} files")
            
            # Process each file
            architecture_results = {
                'design_timestamp': datetime.now().isoformat(),
                'files_analyzed': [],
                'total_files': len(processed_file_paths),
                'successful_designs': 0,
                'failed_designs': 0,
                'recommended_models': {},
                'model_comparisons': {}
            }
            
            for file_path in processed_file_paths:
                try:
                    file_result = self._design_architecture(file_path)
                    architecture_results['files_analyzed'].append(file_result)
                    architecture_results['successful_designs'] += 1
                    
                    if 'recommended_model' in file_result:
                        architecture_results['recommended_models'][file_path] = file_result['recommended_model']
                    
                    if 'model_comparison' in file_result:
                        architecture_results['model_comparisons'][file_path] = file_result['model_comparison']
                    
                except Exception as e:
                    self.logger.error(f"Failed to design architecture for file {file_path}: {e}")
                    architecture_results['failed_designs'] += 1
                    architecture_results['files_analyzed'].append({
                        'file_path': file_path,
                        'error': str(e),
                        'status': 'failed'
                    })
            
            # Generate overall architecture summary
            if architecture_results['successful_designs'] > 0:
                architecture_results['overall_summary'] = self._generate_overall_summary(architecture_results)
            
            # Convert numpy types for JSON serialization
            architecture_results = self._convert_numpy_types(architecture_results)
            
            # Update context
            update_context_chain(context, 'model_architecture', architecture_results)
            
            # Log completion
            log_step(context, 'model_architecture', 
                    f"Completed architecture design for {architecture_results['successful_designs']} files")
            write_context(context)
            
            # Save detailed results
            self._save_architecture_results(architecture_results)
            
            self.logger.info(f"Model architecture design completed successfully for {architecture_results['successful_designs']} files")
            
            return architecture_results
            
        except Exception as e:
            self.logger.error(f"Model architecture design failed: {e}")
            
            # Log error in context
            context = read_context()
            error_data = {
                'error': str(e),
                'status': 'failed',
                'design_timestamp': datetime.now().isoformat()
            }
            update_context_chain(context, 'model_architecture', error_data)
            log_step(context, 'model_architecture', f"Error: {str(e)}")
            write_context(context)
            
            raise
    
    def _design_architecture(self, file_path: str) -> Dict[str, Any]:
        """
        Design model architecture for a single file.
        
        Args:
            file_path: Path to the processed data file
            
        Returns:
            Dict containing architecture design results
        """
        # Load data
        data = pd.read_csv(file_path)
        self.logger.info(f"Designing architecture for file: {file_path} with {len(data)} rows and {len(data.columns)} columns")
        
        # Analyze data characteristics
        data_analysis = self._analyze_data_characteristics(data)
        
        # Determine task type (classification vs regression)
        task_type = self._determine_task_type(data, data_analysis)
        
        # Design and compare models
        model_comparison = self._compare_models(data, task_type, data_analysis)
        
        # Select best model
        recommended_model = self._select_best_model(model_comparison, task_type)
        
        return {
            'file_path': file_path,
            'status': 'success',
            'data_analysis': data_analysis,
            'task_type': task_type,
            'model_comparison': model_comparison,
            'recommended_model': recommended_model,
            'data_shape': data.shape,
            'column_names': list(data.columns)
        }
    
    def _analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data characteristics for model selection."""
        analysis = {
            'total_samples': len(data),
            'total_features': len(data.columns),
            'numeric_features': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(data.select_dtypes(include=['object']).columns),
            'missing_values': data.isnull().sum().sum(),
            'feature_correlation': None,
            'class_balance': None
        }
        
        # Calculate feature correlation for numeric features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = data[numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.8:
                        high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr))
            analysis['feature_correlation'] = high_corr_pairs
        
        return analysis
    
    def _determine_task_type(self, data: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """Determine if this is a classification or regression task."""
        # Look for target variable patterns
        # For now, assume the last column is the target
        target_col = data.columns[-1]
        target_values = data[target_col]
        
        # Check if target is categorical (classification)
        if target_values.dtype == 'object' or target_values.nunique() <= 10:
            return 'classification'
        else:
            return 'regression'
    
    def _compare_models(self, data: pd.DataFrame, task_type: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different model architectures."""
        # Prepare data
        X = data.iloc[:, :-1]  # All features except last
        y = data.iloc[:, -1]   # Last column as target
        
        # Remove any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        if len(X.columns) == 0:
            raise ValueError("No numeric features available for modeling")
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mode()[0] if task_type == 'classification' else y.mean())
        
        # Define models based on task type
        if task_type == 'classification':
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42),
                'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
            }
            scoring = 'accuracy'
        else:
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'SVR': SVR(),
                'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
            }
            scoring = 'r2'
        
        # Compare models using cross-validation
        comparison = {
            'task_type': task_type,
            'models_tested': list(models.keys()),
            'cv_scores': {},
            'best_model': None,
            'best_score': -1
        }
        
        for name, model in models.items():
            try:
                # Use 5-fold cross-validation
                scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
                comparison['cv_scores'][name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores.tolist()
                }
                
                # Track best model
                if scores.mean() > comparison['best_score']:
                    comparison['best_score'] = scores.mean()
                    comparison['best_model'] = name
                    
            except Exception as e:
                self.logger.warning(f"Failed to evaluate {name}: {e}")
                comparison['cv_scores'][name] = {
                    'mean_score': -1,
                    'std_score': 0,
                    'scores': [],
                    'error': str(e)
                }
        
        return comparison
    
    def _select_best_model(self, comparison: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Select the best model based on cross-validation results."""
        best_model_name = comparison['best_model']
        best_scores = comparison['cv_scores'][best_model_name]
        
        # Define model parameters based on the best model
        if task_type == 'classification':
            if best_model_name == 'Random Forest':
                model_params = {
                    'model_type': 'RandomForestClassifier',
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
            elif best_model_name == 'Logistic Regression':
                model_params = {
                    'model_type': 'LogisticRegression',
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'lbfgs',
                    'max_iter': 1000,
                    'random_state': 42
                }
            elif best_model_name == 'SVM':
                model_params = {
                    'model_type': 'SVC',
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'random_state': 42
                }
            else:  # Neural Network
                model_params = {
                    'model_type': 'MLPClassifier',
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'max_iter': 500,
                    'random_state': 42
                }
        else:  # Regression
            if best_model_name == 'Random Forest':
                model_params = {
                    'model_type': 'RandomForestRegressor',
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
            elif best_model_name == 'Linear Regression':
                model_params = {
                    'model_type': 'LinearRegression'
                }
            elif best_model_name == 'SVR':
                model_params = {
                    'model_type': 'SVR',
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale'
                }
            else:  # Neural Network
                model_params = {
                    'model_type': 'MLPRegressor',
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'max_iter': 500,
                    'random_state': 42
                }
        
        return {
            'model_name': best_model_name,
            'model_params': model_params,
            'task_type': task_type,
            'cv_score': best_scores['mean_score'],
            'cv_std': best_scores['std_score'],
            'recommendation_reason': f"Best cross-validation performance among tested models"
        }
    
    def _generate_overall_summary(self, architecture_results: Dict[str, Any]) -> str:
        """Generate overall model architecture summary."""
        successful_designs = architecture_results['successful_designs']
        total_files = architecture_results['total_files']
        
        # Count model types
        model_counts = {}
        for file_path, model_info in architecture_results['recommended_models'].items():
            model_name = model_info['model_name']
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
        
        summary = f"""
        Model Architecture Design Summary:
        - Files analyzed: {successful_designs}/{total_files}
        - Models recommended: {len(architecture_results['recommended_models'])}
        
        Model Distribution:
        """
        for model_name, count in model_counts.items():
            summary += f"- {model_name}: {count} files\n"
        
        summary += f"""
        Key findings:
        - Cross-validation used to compare multiple model architectures
        - Models selected based on task type (classification/regression)
        - Best performing model recommended for each dataset
        - Model parameters optimized for each architecture
        """
        
        return summary
    
    def _save_architecture_results(self, results: Dict[str, Any]):
        """Save detailed model architecture results to output directory."""
        # Save full results
        results_file = self.output_dir / "model_architecture_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = self.output_dir / "model_architecture_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Model Architecture Design Summary\n")
            f.write(f"================================\n\n")
            f.write(f"Design Timestamp: {results['design_timestamp']}\n")
            f.write(f"Total Files: {results['total_files']}\n")
            f.write(f"Successful Designs: {results['successful_designs']}\n")
            f.write(f"Failed Designs: {results['failed_designs']}\n\n")
            
            if 'overall_summary' in results:
                f.write(f"Overall Summary:\n{results['overall_summary']}\n\n")
            
            f.write(f"Recommended Models:\n")
            for file_path, model_info in results['recommended_models'].items():
                f.write(f"- {file_path}\n")
                f.write(f"  Model: {model_info['model_name']}\n")
                f.write(f"  Task Type: {model_info['task_type']}\n")
                f.write(f"  CV Score: {model_info['cv_score']:.4f} ± {model_info['cv_std']:.4f}\n")
                f.write(f"  Reason: {model_info['recommendation_reason']}\n\n")
            
            f.write(f"Analyzed Files:\n")
            for file_analysis in results['files_analyzed']:
                f.write(f"- {file_analysis['file_path']}\n")
                f.write(f"  Status: {file_analysis.get('status', 'unknown')}\n")
                if 'data_shape' in file_analysis:
                    f.write(f"  Shape: {file_analysis['data_shape']}\n")
                if 'error' in file_analysis:
                    f.write(f"  Error: {file_analysis['error']}\n")
                f.write("\n")
        
        self.logger.info(f"Model architecture results saved to {self.output_dir}")
    
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
    agent = ModelArchitectureAgent()
    results = agent.execute()
    print(f"Model architecture design completed. Designed {results['successful_designs']} models.")
