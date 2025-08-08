"""
Model Validation Agent - Simple Implementation

Validates models using holdout sets and generates performance metrics.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

from src.core.context_manager import (
    read_context, write_context, log_step, 
    update_context_chain, get_context_chain_data
)

class ModelValidationAgent:
    """
    Model Validation Agent for model evaluation.
    
    Trains models on training data, validates on holdout sets, and
    generates comprehensive performance metrics and validation reports.
    """
    
    def __init__(self):
        """Initialize Model Validation Agent."""
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = Path("output/model_validation_01")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Model Validation Agent initialized")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute model validation task.
        
        Returns:
            Dict containing validation results and metadata
        """
        try:
            # Read context
            context = read_context()
            
            # Get optimized models from hyperparameter optimization
            optimization_data = get_context_chain_data(context, 'hyperparameter_optimization')
            if not optimization_data or 'optimized_models' not in optimization_data:
                raise ValueError("No optimized models found in context")
            
            optimized_models = optimization_data['optimized_models']
            self.logger.info(f"Starting model validation for {len(optimized_models)} models")
            
            # Process each model
            validation_results = {
                'validation_timestamp': datetime.now().isoformat(),
                'models_validated': [],
                'total_models': len(optimized_models),
                'successful_validations': 0,
                'failed_validations': 0,
                'validation_metrics': {},
                'model_performance': {}
            }
            
            for file_path, optimized_params in optimized_models.items():
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
                    
                    # Get model info from architecture
                    architecture_data = get_context_chain_data(context, 'model_architecture')
                    if not architecture_data or 'recommended_models' not in architecture_data:
                        continue
                    
                    model_info = architecture_data['recommended_models'].get(file_path, {})
                    if not model_info:
                        continue
                    
                    validation_result = self._validate_model(processed_file, model_info, optimized_params)
                    validation_results['models_validated'].append(validation_result)
                    validation_results['successful_validations'] += 1
                    
                    if 'metrics' in validation_result:
                        validation_results['validation_metrics'][file_path] = validation_result['metrics']
                    
                    if 'performance_summary' in validation_result:
                        validation_results['model_performance'][file_path] = validation_result['performance_summary']
                    
                except Exception as e:
                    self.logger.error(f"Failed to validate model for file {file_path}: {e}")
                    validation_results['failed_validations'] += 1
                    validation_results['models_validated'].append({
                        'file_path': file_path,
                        'error': str(e),
                        'status': 'failed'
                    })
            
            # Generate overall validation summary
            if validation_results['successful_validations'] > 0:
                validation_results['overall_summary'] = self._generate_overall_summary(validation_results)
            
            # Convert numpy types for JSON serialization
            validation_results = self._convert_numpy_types(validation_results)
            
            # Update context
            update_context_chain(context, 'model_validation', validation_results)
            
            # Log completion
            log_step(context, 'model_validation', 
                    f"Completed validation for {validation_results['successful_validations']} models")
            write_context(context)
            
            # Save detailed results
            self._save_validation_results(validation_results)
            
            self.logger.info(f"Model validation completed successfully for {validation_results['successful_validations']} models")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            
            # Log error in context
            context = read_context()
            error_data = {
                'error': str(e),
                'status': 'failed',
                'validation_timestamp': datetime.now().isoformat()
            }
            update_context_chain(context, 'model_validation', error_data)
            log_step(context, 'model_validation', f"Error: {str(e)}")
            write_context(context)
            
            raise
    
    def _validate_model(self, data_file: str, model_info: Dict[str, Any], optimized_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single model.
        
        Args:
            data_file: Path to the processed data file
            model_info: Information about the model
            optimized_params: Optimized hyperparameters
            
        Returns:
            Dict containing validation results
        """
        # Load data
        data = pd.read_csv(data_file)
        self.logger.info(f"Validating model for file: {data_file}")
        
        # Prepare data
        X = data.iloc[:, :-1].select_dtypes(include=[np.number])
        y = data.iloc[:, -1]
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mode()[0] if model_info['task_type'] == 'classification' else y.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if model_info['task_type'] == 'classification' else None
        )
        
        # Create and train model
        model = self._create_model(model_info['model_name'], model_info['task_type'], optimized_params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, model_info['task_type'])
        
        # Generate performance summary
        performance_summary = self._generate_performance_summary(metrics, model_info)
        
        return {
            'file_path': data_file,
            'status': 'success',
            'metrics': metrics,
            'performance_summary': performance_summary,
            'model_name': model_info['model_name'],
            'task_type': model_info['task_type'],
            'test_size': len(X_test),
            'train_size': len(X_train)
        }
    
    def _create_model(self, model_name: str, task_type: str, optimized_params: Dict[str, Any]):
        """Create a model instance with optimized parameters."""
        if task_type == 'classification':
            if model_name == 'Random Forest':
                return RandomForestClassifier(**optimized_params, random_state=42)
            elif model_name == 'Logistic Regression':
                return LogisticRegression(**optimized_params, random_state=42, max_iter=1000)
            elif model_name == 'SVM':
                return SVC(**optimized_params, random_state=42)
            elif model_name == 'Neural Network':
                return MLPClassifier(**optimized_params, random_state=42, max_iter=500)
        else:  # Regression
            if model_name == 'Random Forest':
                return RandomForestRegressor(**optimized_params, random_state=42)
            elif model_name == 'Linear Regression':
                return LinearRegression(**optimized_params)
            elif model_name == 'SVR':
                return SVR(**optimized_params)
            elif model_name == 'Neural Network':
                return MLPRegressor(**optimized_params, random_state=42, max_iter=500)
        
        # Default fallback
        return RandomForestClassifier(random_state=42) if task_type == 'classification' else RandomForestRegressor(random_state=42)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
        """Calculate performance metrics based on task type."""
        metrics = {}
        
        if task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        else:  # Regression
            metrics['r2_score'] = r2_score(y_true, y_pred)
            metrics['mean_squared_error'] = mean_squared_error(y_true, y_pred)
            metrics['root_mean_squared_error'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
        
        return metrics
    
    def _generate_performance_summary(self, metrics: Dict[str, float], model_info: Dict[str, Any]) -> str:
        """Generate a performance summary for the model."""
        task_type = model_info['task_type']
        model_name = model_info['model_name']
        
        if task_type == 'classification':
            summary = f"""
            Model: {model_name}
            Task Type: Classification
            Performance Metrics:
            - Accuracy: {metrics['accuracy']:.4f}
            - Precision: {metrics['precision']:.4f}
            - Recall: {metrics['recall']:.4f}
            - F1 Score: {metrics['f1_score']:.4f}
            """
        else:
            summary = f"""
            Model: {model_name}
            Task Type: Regression
            Performance Metrics:
            - R² Score: {metrics['r2_score']:.4f}
            - Mean Squared Error: {metrics['mean_squared_error']:.4f}
            - Root Mean Squared Error: {metrics['root_mean_squared_error']:.4f}
            - Mean Absolute Error: {metrics['mean_absolute_error']:.4f}
            """
        
        return summary
    
    def _generate_overall_summary(self, validation_results: Dict[str, Any]) -> str:
        """Generate overall model validation summary."""
        successful_validations = validation_results['successful_validations']
        total_models = validation_results['total_models']
        
        # Calculate average metrics
        avg_metrics = {}
        for file_path, metrics in validation_results['validation_metrics'].items():
            for metric_name, value in metrics.items():
                if metric_name not in avg_metrics:
                    avg_metrics[metric_name] = []
                avg_metrics[metric_name].append(value)
        
        # Calculate averages
        for metric_name in avg_metrics:
            avg_metrics[metric_name] = np.mean(avg_metrics[metric_name])
        
        summary = f"""
        Model Validation Summary:
        - Models validated: {successful_validations}/{total_models}
        - Average metrics across all models:
        """
        
        for metric_name, avg_value in avg_metrics.items():
            summary += f"- {metric_name}: {avg_value:.4f}\n"
        
        summary += f"""
        Key findings:
        - Holdout validation used for unbiased performance estimation
        - Comprehensive metrics calculated for each model
        - Performance summaries generated for model comparison
        - Validation reports saved for detailed analysis
        """
        
        return summary
    
    def _save_validation_results(self, results: Dict[str, Any]):
        """Save detailed model validation results to output directory."""
        # Save full results
        results_file = self.output_dir / "model_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = self.output_dir / "model_validation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Model Validation Summary\n")
            f.write(f"======================\n\n")
            f.write(f"Validation Timestamp: {results['validation_timestamp']}\n")
            f.write(f"Total Models: {results['total_models']}\n")
            f.write(f"Successful Validations: {results['successful_validations']}\n")
            f.write(f"Failed Validations: {results['failed_validations']}\n\n")
            
            if 'overall_summary' in results:
                f.write(f"Overall Summary:\n{results['overall_summary']}\n\n")
            
            f.write(f"Validation Results:\n")
            for file_path, metrics in results['validation_metrics'].items():
                f.write(f"- {file_path}\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")
                f.write("\n")
            
            f.write(f"Validated Files:\n")
            for model_result in results['models_validated']:
                f.write(f"- {model_result['file_path']}\n")
                f.write(f"  Status: {model_result.get('status', 'unknown')}\n")
                if 'model_name' in model_result:
                    f.write(f"  Model: {model_result['model_name']}\n")
                    f.write(f"  Task Type: {model_result['task_type']}\n")
                if 'test_size' in model_result:
                    f.write(f"  Test Size: {model_result['test_size']}\n")
                    f.write(f"  Train Size: {model_result['train_size']}\n")
                if 'error' in model_result:
                    f.write(f"  Error: {model_result['error']}\n")
                f.write("\n")
        
        self.logger.info(f"Model validation results saved to {self.output_dir}")
    
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
    agent = ModelValidationAgent()
    results = agent.execute()
    print(f"Model validation completed. Validated {results['successful_validations']} models.")
