"""
Hyperparameter Optimization Agent - Simple Implementation

Optimizes model hyperparameters using grid search and cross-validation.
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

from src.core.context_manager import (
    read_context, write_context, log_step, 
    update_context_chain, get_context_chain_data
)

class HyperparameterOptimizationAgent:
    """
    Hyperparameter Optimization Agent for model tuning.
    
    Uses grid search and cross-validation to find optimal hyperparameters
    for the recommended models from the model architecture agent.
    """
    
    def __init__(self):
        """Initialize Hyperparameter Optimization Agent."""
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = Path("output/hyperparameter_optimization_01")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Hyperparameter Optimization Agent initialized")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute hyperparameter optimization task.
        
        Returns:
            Dict containing optimization results and metadata
        """
        try:
            # Read context
            context = read_context()
            
            # Get recommended models from model architecture
            architecture_data = get_context_chain_data(context, 'model_architecture')
            if not architecture_data or 'recommended_models' not in architecture_data:
                raise ValueError("No recommended models found in context")
            
            recommended_models = architecture_data['recommended_models']
            self.logger.info(f"Starting hyperparameter optimization for {len(recommended_models)} models")
            
            # Process each model
            optimization_results = {
                'optimization_timestamp': datetime.now().isoformat(),
                'models_optimized': [],
                'total_models': len(recommended_models),
                'successful_optimizations': 0,
                'failed_optimizations': 0,
                'optimized_models': {},
                'performance_improvements': {}
            }
            
            for file_path, model_info in recommended_models.items():
                try:
                    # Get processed data for this model
                    feature_data = get_context_chain_data(context, 'feature_engineering')
                    if not feature_data or 'processed_file_paths' not in feature_data:
                        continue
                    
                    # Find the corresponding processed file
                    processed_file = None
                    for processed_path in feature_data['processed_file_paths']:
                        if Path(processed_path).stem.replace('_processed', '') in file_path:
                            processed_file = processed_path
                            break
                    
                    if not processed_file:
                        self.logger.warning(f"No processed file found for {file_path}")
                        continue
                    
                    optimization_result = self._optimize_model(processed_file, model_info)
                    optimization_results['models_optimized'].append(optimization_result)
                    optimization_results['successful_optimizations'] += 1
                    
                    if 'optimized_params' in optimization_result:
                        optimization_results['optimized_models'][file_path] = optimization_result['optimized_params']
                    
                    if 'performance_improvement' in optimization_result:
                        optimization_results['performance_improvements'][file_path] = optimization_result['performance_improvement']
                    
                except Exception as e:
                    self.logger.error(f"Failed to optimize model for file {file_path}: {e}")
                    optimization_results['failed_optimizations'] += 1
                    optimization_results['models_optimized'].append({
                        'file_path': file_path,
                        'error': str(e),
                        'status': 'failed'
                    })
            
            # Generate overall optimization summary
            if optimization_results['successful_optimizations'] > 0:
                optimization_results['overall_summary'] = self._generate_overall_summary(optimization_results)
            
            # Convert numpy types for JSON serialization
            optimization_results = self._convert_numpy_types(optimization_results)
            
            # Update context
            update_context_chain(context, 'hyperparameter_optimization', optimization_results)
            
            # Log completion
            log_step(context, 'hyperparameter_optimization', 
                    f"Completed optimization for {optimization_results['successful_optimizations']} models")
            write_context(context)
            
            # Save detailed results
            self._save_optimization_results(optimization_results)
            
            self.logger.info(f"Hyperparameter optimization completed successfully for {optimization_results['successful_optimizations']} models")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {e}")
            
            # Log error in context
            context = read_context()
            error_data = {
                'error': str(e),
                'status': 'failed',
                'optimization_timestamp': datetime.now().isoformat()
            }
            update_context_chain(context, 'hyperparameter_optimization', error_data)
            log_step(context, 'hyperparameter_optimization', f"Error: {str(e)}")
            write_context(context)
            
            raise
    
    def _optimize_model(self, data_file: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a single model.
        
        Args:
            data_file: Path to the processed data file
            model_info: Information about the recommended model
            
        Returns:
            Dict containing optimization results
        """
        # Load data
        data = pd.read_csv(data_file)
        self.logger.info(f"Optimizing model for file: {data_file}")
        
        # Prepare data
        X = data.iloc[:, :-1].select_dtypes(include=[np.number])
        y = data.iloc[:, -1]
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mode()[0] if model_info['task_type'] == 'classification' else y.mean())
        
        # Get original model performance
        original_score = model_info['cv_score']
        
        # Define parameter grids based on model type
        param_grid = self._get_parameter_grid(model_info['model_name'], model_info['task_type'])
        
        # Perform optimization
        best_params, best_score = self._perform_optimization(X, y, model_info, param_grid)
        
        # Calculate performance improvement
        performance_improvement = best_score - original_score if best_score > original_score else 0
        
        return {
            'file_path': data_file,
            'status': 'success',
            'original_score': original_score,
            'optimized_score': best_score,
            'performance_improvement': performance_improvement,
            'optimized_params': best_params,
            'task_type': model_info['task_type'],
            'model_name': model_info['model_name']
        }
    
    def _get_parameter_grid(self, model_name: str, task_type: str) -> Dict[str, List]:
        """Get parameter grid for hyperparameter optimization."""
        if task_type == 'classification':
            if model_name == 'Random Forest':
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_name == 'Logistic Regression':
                return {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            elif model_name == 'SVM':
                return {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            elif model_name == 'Neural Network':
                return {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
        else:  # Regression
            if model_name == 'Random Forest':
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_name == 'Linear Regression':
                return {}  # No hyperparameters to optimize
            elif model_name == 'SVR':
                return {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            elif model_name == 'Neural Network':
                return {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
        
        return {}
    
    def _perform_optimization(self, X: pd.DataFrame, y: pd.Series, model_info: Dict[str, Any], param_grid: Dict[str, List]) -> tuple:
        """Perform hyperparameter optimization using grid search."""
        # Create base model
        base_model = self._create_model(model_info['model_name'], model_info['task_type'])
        
        # Define scoring metric
        scoring = 'accuracy' if model_info['task_type'] == 'classification' else 'r2'
        
        # Perform grid search
        if param_grid:
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring=scoring,
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X, y)
            
            return grid_search.best_params_, grid_search.best_score_
        else:
            # No parameters to optimize, return default
            return {}, model_info['cv_score']
    
    def _create_model(self, model_name: str, task_type: str):
        """Create a model instance based on name and task type."""
        if task_type == 'classification':
            if model_name == 'Random Forest':
                return RandomForestClassifier(random_state=42)
            elif model_name == 'Logistic Regression':
                return LogisticRegression(random_state=42, max_iter=1000)
            elif model_name == 'SVM':
                return SVC(random_state=42)
            elif model_name == 'Neural Network':
                return MLPClassifier(random_state=42, max_iter=500)
        else:  # Regression
            if model_name == 'Random Forest':
                return RandomForestRegressor(random_state=42)
            elif model_name == 'Linear Regression':
                return LinearRegression()
            elif model_name == 'SVR':
                return SVR()
            elif model_name == 'Neural Network':
                return MLPRegressor(random_state=42, max_iter=500)
        
        # Default fallback
        return RandomForestClassifier(random_state=42) if task_type == 'classification' else RandomForestRegressor(random_state=42)
    
    def _generate_overall_summary(self, optimization_results: Dict[str, Any]) -> str:
        """Generate overall hyperparameter optimization summary."""
        successful_optimizations = optimization_results['successful_optimizations']
        total_models = optimization_results['total_models']
        
        # Calculate average improvement
        improvements = list(optimization_results['performance_improvements'].values())
        avg_improvement = np.mean(improvements) if improvements else 0
        
        summary = f"""
        Hyperparameter Optimization Summary:
        - Models optimized: {successful_optimizations}/{total_models}
        - Average performance improvement: {avg_improvement:.4f}
        - Models with improvement: {len([imp for imp in improvements if imp > 0])}
        
        Key findings:
        - Grid search performed for each recommended model
        - Cross-validation used to ensure robust parameter selection
        - Performance improvements measured against baseline models
        - Optimal parameters identified for each model architecture
        """
        
        return summary
    
    def _save_optimization_results(self, results: Dict[str, Any]):
        """Save detailed hyperparameter optimization results to output directory."""
        # Save full results
        results_file = self.output_dir / "hyperparameter_optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = self.output_dir / "hyperparameter_optimization_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Hyperparameter Optimization Summary\n")
            f.write(f"==================================\n\n")
            f.write(f"Optimization Timestamp: {results['optimization_timestamp']}\n")
            f.write(f"Total Models: {results['total_models']}\n")
            f.write(f"Successful Optimizations: {results['successful_optimizations']}\n")
            f.write(f"Failed Optimizations: {results['failed_optimizations']}\n\n")
            
            if 'overall_summary' in results:
                f.write(f"Overall Summary:\n{results['overall_summary']}\n\n")
            
            f.write(f"Optimized Models:\n")
            for file_path, optimized_params in results['optimized_models'].items():
                f.write(f"- {file_path}\n")
                f.write(f"  Optimized Parameters: {optimized_params}\n")
                if file_path in results['performance_improvements']:
                    improvement = results['performance_improvements'][file_path]
                    f.write(f"  Performance Improvement: {improvement:.4f}\n")
                f.write("\n")
            
            f.write(f"Optimized Files:\n")
            for model_result in results['models_optimized']:
                f.write(f"- {model_result['file_path']}\n")
                f.write(f"  Status: {model_result.get('status', 'unknown')}\n")
                if 'original_score' in model_result:
                    f.write(f"  Original Score: {model_result['original_score']:.4f}\n")
                    f.write(f"  Optimized Score: {model_result['optimized_score']:.4f}\n")
                    f.write(f"  Improvement: {model_result['performance_improvement']:.4f}\n")
                if 'error' in model_result:
                    f.write(f"  Error: {model_result['error']}\n")
                f.write("\n")
        
        self.logger.info(f"Hyperparameter optimization results saved to {self.output_dir}")
    
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
    agent = HyperparameterOptimizationAgent()
    results = agent.execute()
    print(f"Hyperparameter optimization completed. Optimized {results['successful_optimizations']} models.")
