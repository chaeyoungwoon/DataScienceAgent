"""
EDA Agent - Simple Implementation

Performs exploratory data analysis with descriptive statistics and visualizations.
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
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.context_manager import (
    read_context, write_context, log_step, 
    update_context_chain, get_context_chain_data
)

class EDAAgent:
    """
    EDA Agent for exploratory data analysis.
    
    Performs descriptive statistics, creates visualizations, and generates
    comprehensive data analysis reports.
    """
    
    def __init__(self):
        """Initialize EDA Agent."""
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = Path("output/eda_01")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.logger.info("EDA Agent initialized")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute EDA task.
        
        Returns:
            Dict containing EDA results and metadata
        """
        try:
            # Read context
            context = read_context()
            
            # Get cleaned dataset paths from data quality
            quality_data = get_context_chain_data(context, 'data_quality')
            if not quality_data or 'cleaned_file_paths' not in quality_data:
                raise ValueError("No cleaned file paths found in context")
            
            cleaned_file_paths = quality_data['cleaned_file_paths']
            self.logger.info(f"Starting EDA for {len(cleaned_file_paths)} files")
            
            # Process each file
            eda_results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'files_analyzed': [],
                'total_files': len(cleaned_file_paths),
                'successful_analyses': 0,
                'failed_analyses': 0,
                'summary_statistics': {},
                'visualizations': [],
                'correlation_matrices': {}
            }
            
            for file_path in cleaned_file_paths:
                try:
                    file_analysis = self._analyze_file(file_path)
                    eda_results['files_analyzed'].append(file_analysis)
                    eda_results['successful_analyses'] += 1
                    
                    # Aggregate summary statistics
                    if 'summary_stats' in file_analysis:
                        eda_results['summary_statistics'][file_path] = file_analysis['summary_stats']
                    
                    # Add visualizations
                    if 'visualizations' in file_analysis:
                        eda_results['visualizations'].extend(file_analysis['visualizations'])
                    
                    # Add correlation matrix
                    if 'correlation_matrix' in file_analysis:
                        eda_results['correlation_matrices'][file_path] = file_analysis['correlation_matrix']
                    
                except Exception as e:
                    self.logger.error(f"Failed to analyze file {file_path}: {e}")
                    eda_results['failed_analyses'] += 1
                    eda_results['files_analyzed'].append({
                        'file_path': file_path,
                        'error': str(e),
                        'status': 'failed'
                    })
            
            # Generate overall EDA summary
            if eda_results['successful_analyses'] > 0:
                eda_results['overall_summary'] = self._generate_overall_summary(eda_results)
            
            # Convert numpy types for JSON serialization
            eda_results = self._convert_numpy_types(eda_results)
            
            # Update context
            update_context_chain(context, 'eda', eda_results)
            
            # Log completion
            log_step(context, 'eda', 
                    f"Completed EDA for {eda_results['successful_analyses']} files")
            write_context(context)
            
            # Save detailed results
            self._save_eda_results(eda_results)
            
            self.logger.info(f"EDA completed successfully for {eda_results['successful_analyses']} files")
            
            return eda_results
            
        except Exception as e:
            self.logger.error(f"EDA failed: {e}")
            
            # Log error in context
            context = read_context()
            error_data = {
                'error': str(e),
                'status': 'failed',
                'analysis_timestamp': datetime.now().isoformat()
            }
            update_context_chain(context, 'eda', error_data)
            log_step(context, 'eda', f"Error: {str(e)}")
            write_context(context)
            
            raise
    
    def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single data file.
        
        Args:
            file_path: Path to the cleaned data file
            
        Returns:
            Dict containing analysis results
        """
        # Load data
        data = pd.read_csv(file_path)
        self.logger.info(f"Analyzing file: {file_path} with {len(data)} rows and {len(data.columns)} columns")
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(data)
        
        # Create visualizations
        visualizations = self._create_visualizations(data, file_path)
        
        # Generate correlation matrix
        correlation_matrix = self._generate_correlation_matrix(data)
        
        return {
            'file_path': file_path,
            'status': 'success',
            'summary_stats': summary_stats,
            'visualizations': visualizations,
            'correlation_matrix': correlation_matrix,
            'data_shape': data.shape,
            'column_names': list(data.columns)
        }
    
    def _generate_summary_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        stats = {
            'basic_info': {
                'rows': len(data),
                'columns': len(data.columns),
                'memory_usage': data.memory_usage(deep=True).sum(),
                'dtypes': data.dtypes.to_dict()
            },
            'descriptive_stats': {},
            'missing_values': {},
            'unique_counts': {}
        }
        
        # Descriptive statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['descriptive_stats'] = data[numeric_cols].describe().to_dict()
        
        # Missing values
        missing_values = data.isnull().sum()
        stats['missing_values'] = missing_values[missing_values > 0].to_dict()
        
        # Unique counts for categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            stats['unique_counts'][col] = data[col].nunique()
        
        return stats
    
    def _create_visualizations(self, data: pd.DataFrame, file_path: str) -> List[str]:
        """Create various visualizations for the data."""
        visualizations = []
        file_name = Path(file_path).stem
        
        # Set figure size for better quality
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Distribution plots for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, col in enumerate(numeric_cols[:4]):  # Limit to 4 columns
                if i < len(axes):
                    data[col].hist(ax=axes[i], bins=30, alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            dist_plot_path = self.output_dir / f"{file_name}_distributions.png"
            plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(dist_plot_path))
        
        # 2. Correlation heatmap for numeric columns
        if len(numeric_cols) > 1:
            correlation_matrix = data[numeric_cols].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title(f'Correlation Matrix - {file_name}')
            plt.tight_layout()
            
            corr_plot_path = self.output_dir / f"{file_name}_correlation.png"
            plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(corr_plot_path))
        
        # 3. Box plots for numeric columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, col in enumerate(numeric_cols[:4]):
                if i < len(axes):
                    data.boxplot(column=col, ax=axes[i])
                    axes[i].set_title(f'Box Plot of {col}')
                    axes[i].set_ylabel(col)
            
            # Hide empty subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            box_plot_path = self.output_dir / f"{file_name}_boxplots.png"
            plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(box_plot_path))
        
        # 4. Missing values visualization
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            plt.figure(figsize=(10, 6))
            missing_data[missing_data > 0].plot(kind='bar')
            plt.title(f'Missing Values - {file_name}')
            plt.xlabel('Columns')
            plt.ylabel('Missing Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            missing_plot_path = self.output_dir / f"{file_name}_missing_values.png"
            plt.savefig(missing_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(missing_plot_path))
        
        return visualizations
    
    def _generate_correlation_matrix(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate correlation matrix for numeric columns."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}
        
        correlation_matrix = data[numeric_cols].corr()
        
        # Convert to dict for JSON serialization
        return {
            'columns': list(correlation_matrix.columns),
            'correlations': correlation_matrix.to_dict()
        }
    
    def _generate_overall_summary(self, eda_results: Dict[str, Any]) -> str:
        """Generate overall EDA summary."""
        successful_files = eda_results['successful_analyses']
        total_files = eda_results['total_files']
        total_visualizations = len(eda_results['visualizations'])
        
        summary = f"""
        EDA Analysis Summary:
        - Files analyzed: {successful_files}/{total_files}
        - Visualizations created: {total_visualizations}
        - Correlation matrices generated: {len(eda_results['correlation_matrices'])}
        
        Key findings:
        - Comprehensive statistical summaries generated for all numeric variables
        - Distribution plots created to understand data patterns
        - Correlation analysis performed to identify relationships
        - Missing value analysis completed
        - Box plots generated to identify outliers
        """
        
        return summary
    
    def _save_eda_results(self, results: Dict[str, Any]):
        """Save detailed EDA results to output directory."""
        # Save full results
        results_file = self.output_dir / "eda_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = self.output_dir / "eda_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"EDA Analysis Summary\n")
            f.write(f"===================\n\n")
            f.write(f"Analysis Timestamp: {results['analysis_timestamp']}\n")
            f.write(f"Total Files: {results['total_files']}\n")
            f.write(f"Successful Analyses: {results['successful_analyses']}\n")
            f.write(f"Failed Analyses: {results['failed_analyses']}\n")
            f.write(f"Visualizations Created: {len(results['visualizations'])}\n\n")
            
            if 'overall_summary' in results:
                f.write(f"Overall Summary:\n{results['overall_summary']}\n\n")
            
            f.write(f"Analyzed Files:\n")
            for file_analysis in results['files_analyzed']:
                f.write(f"- {file_analysis['file_path']}\n")
                f.write(f"  Status: {file_analysis.get('status', 'unknown')}\n")
                if 'data_shape' in file_analysis:
                    f.write(f"  Shape: {file_analysis['data_shape']}\n")
                if 'error' in file_analysis:
                    f.write(f"  Error: {file_analysis['error']}\n")
                f.write("\n")
        
        self.logger.info(f"EDA results saved to {self.output_dir}")
    
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
    agent = EDAAgent()
    results = agent.execute()
    print(f"EDA completed. Analyzed {results['successful_analyses']} files.")
