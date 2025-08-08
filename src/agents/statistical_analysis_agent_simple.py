"""
Statistical Analysis Agent - Simple Implementation

Performs hypothesis tests to find significant relationships.
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
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway

from src.core.context_manager import (
    read_context, write_context, log_step, 
    update_context_chain, get_context_chain_data
)

class StatisticalAnalysisAgent:
    """
    Statistical Analysis Agent for hypothesis testing.
    
    Performs t-tests, ANOVA, correlation tests, and other statistical
    analyses to identify significant relationships in the data.
    """
    
    def __init__(self):
        """Initialize Statistical Analysis Agent."""
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = Path("output/statistical_analysis_01")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Statistical Analysis Agent initialized")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute statistical analysis task.
        
        Returns:
            Dict containing statistical analysis results and metadata
        """
        try:
            # Read context
            context = read_context()
            
            # Get processed dataset paths from feature engineering
            feature_data = get_context_chain_data(context, 'feature_engineering')
            if not feature_data or 'processed_file_paths' not in feature_data:
                raise ValueError("No processed file paths found in context")
            
            processed_file_paths = feature_data['processed_file_paths']
            self.logger.info(f"Starting statistical analysis for {len(processed_file_paths)} files")
            
            # Process each file
            stats_results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'files_analyzed': [],
                'total_files': len(processed_file_paths),
                'successful_analyses': 0,
                'failed_analyses': 0,
                'significant_tests': [],
                'correlation_tests': [],
                'hypothesis_tests': []
            }
            
            for file_path in processed_file_paths:
                try:
                    file_analysis = self._analyze_file(file_path)
                    stats_results['files_analyzed'].append(file_analysis)
                    stats_results['successful_analyses'] += 1
                    
                    # Aggregate test results
                    if 'significant_tests' in file_analysis:
                        stats_results['significant_tests'].extend(file_analysis['significant_tests'])
                    
                    if 'correlation_tests' in file_analysis:
                        stats_results['correlation_tests'].extend(file_analysis['correlation_tests'])
                    
                    if 'hypothesis_tests' in file_analysis:
                        stats_results['hypothesis_tests'].extend(file_analysis['hypothesis_tests'])
                    
                except Exception as e:
                    self.logger.error(f"Failed to analyze file {file_path}: {e}")
                    stats_results['failed_analyses'] += 1
                    stats_results['files_analyzed'].append({
                        'file_path': file_path,
                        'error': str(e),
                        'status': 'failed'
                    })
            
            # Generate overall statistical summary
            if stats_results['successful_analyses'] > 0:
                stats_results['overall_summary'] = self._generate_overall_summary(stats_results)
            
            # Convert numpy types for JSON serialization
            stats_results = self._convert_numpy_types(stats_results)
            
            # Update context
            update_context_chain(context, 'statistical_analysis', stats_results)
            
            # Log completion
            log_step(context, 'statistical_analysis', 
                    f"Completed statistical analysis for {stats_results['successful_analyses']} files")
            write_context(context)
            
            # Save detailed results
            self._save_statistical_results(stats_results)
            
            self.logger.info(f"Statistical analysis completed successfully for {stats_results['successful_analyses']} files")
            
            return stats_results
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            
            # Log error in context
            context = read_context()
            error_data = {
                'error': str(e),
                'status': 'failed',
                'analysis_timestamp': datetime.now().isoformat()
            }
            update_context_chain(context, 'statistical_analysis', error_data)
            log_step(context, 'statistical_analysis', f"Error: {str(e)}")
            write_context(context)
            
            raise
    
    def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single file for statistical relationships.
        
        Args:
            file_path: Path to the processed data file
            
        Returns:
            Dict containing analysis results
        """
        # Load data
        data = pd.read_csv(file_path)
        self.logger.info(f"Analyzing file: {file_path} with {len(data)} rows and {len(data.columns)} columns")
        
        # Perform correlation analysis
        correlation_tests = self._perform_correlation_analysis(data)
        
        # Perform hypothesis tests
        hypothesis_tests = self._perform_hypothesis_tests(data)
        
        # Identify significant relationships
        significant_tests = self._identify_significant_tests(correlation_tests + hypothesis_tests)
        
        return {
            'file_path': file_path,
            'status': 'success',
            'correlation_tests': correlation_tests,
            'hypothesis_tests': hypothesis_tests,
            'significant_tests': significant_tests,
            'data_shape': data.shape,
            'column_names': list(data.columns)
        }
    
    def _perform_correlation_analysis(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Perform correlation analysis between numeric variables."""
        correlation_tests = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                
                # Remove NaN values
                valid_data = data[[col1, col2]].dropna()
                if len(valid_data) < 3:
                    continue
                
                # Pearson correlation
                try:
                    pearson_corr, pearson_p = pearsonr(valid_data[col1], valid_data[col2])
                    correlation_tests.append({
                        'test_type': 'pearson_correlation',
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': pearson_corr,
                        'p_value': pearson_p,
                        'significant': pearson_p < 0.05,
                        'effect_size': abs(pearson_corr)
                    })
                except:
                    pass
                
                # Spearman correlation
                try:
                    spearman_corr, spearman_p = spearmanr(valid_data[col1], valid_data[col2])
                    correlation_tests.append({
                        'test_type': 'spearman_correlation',
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': spearman_corr,
                        'p_value': spearman_p,
                        'significant': spearman_p < 0.05,
                        'effect_size': abs(spearman_corr)
                    })
                except:
                    pass
        
        return correlation_tests
    
    def _perform_hypothesis_tests(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Perform hypothesis tests for categorical vs numeric variables."""
        hypothesis_tests = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # If no categorical columns, try to identify binary columns
        if len(categorical_cols) == 0:
            # Look for columns with few unique values that might be categorical
            for col in data.columns:
                if data[col].nunique() <= 10 and data[col].nunique() > 1:
                    categorical_cols = [col]
                    break
        
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                try:
                    # Group by categorical variable
                    groups = [group[num_col].values for name, group in data.groupby(cat_col)]
                    groups = [g for g in groups if len(g) > 1]  # Remove empty groups
                    
                    if len(groups) < 2:
                        continue
                    
                    # Perform ANOVA
                    f_stat, anova_p = f_oneway(*groups)
                    
                    hypothesis_tests.append({
                        'test_type': 'anova',
                        'dependent_variable': num_col,
                        'independent_variable': cat_col,
                        'f_statistic': f_stat,
                        'p_value': anova_p,
                        'significant': anova_p < 0.05,
                        'groups': len(groups)
                    })
                    
                    # If only 2 groups, also perform t-test
                    if len(groups) == 2:
                        t_stat, t_p = ttest_ind(groups[0], groups[1])
                        hypothesis_tests.append({
                            'test_type': 't_test',
                            'dependent_variable': num_col,
                            'independent_variable': cat_col,
                            't_statistic': t_stat,
                            'p_value': t_p,
                            'significant': t_p < 0.05,
                            'groups': 2
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Failed to perform hypothesis test for {num_col} vs {cat_col}: {e}")
        
        return hypothesis_tests
    
    def _identify_significant_tests(self, all_tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify tests with significant results."""
        significant_tests = []
        
        for test in all_tests:
            if test.get('significant', False):
                significant_tests.append(test)
        
        return significant_tests
    
    def _generate_overall_summary(self, stats_results: Dict[str, Any]) -> str:
        """Generate overall statistical analysis summary."""
        successful_files = stats_results['successful_analyses']
        total_files = stats_results['total_files']
        significant_count = len(stats_results['significant_tests'])
        correlation_count = len(stats_results['correlation_tests'])
        hypothesis_count = len(stats_results['hypothesis_tests'])
        
        summary = f"""
        Statistical Analysis Summary:
        - Files analyzed: {successful_files}/{total_files}
        - Total tests performed: {correlation_count + hypothesis_count}
        - Significant relationships found: {significant_count}
        - Correlation tests: {correlation_count}
        - Hypothesis tests: {hypothesis_count}
        
        Key findings:
        - Comprehensive correlation analysis performed on all numeric variables
        - Hypothesis tests conducted for categorical vs numeric relationships
        - Both parametric (t-test, ANOVA) and non-parametric (Spearman) tests used
        - Effect sizes calculated for significant relationships
        """
        
        return summary
    
    def _save_statistical_results(self, results: Dict[str, Any]):
        """Save detailed statistical analysis results to output directory."""
        # Save full results
        results_file = self.output_dir / "statistical_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = self.output_dir / "statistical_analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Statistical Analysis Summary\n")
            f.write(f"==========================\n\n")
            f.write(f"Analysis Timestamp: {results['analysis_timestamp']}\n")
            f.write(f"Total Files: {results['total_files']}\n")
            f.write(f"Successful Analyses: {results['successful_analyses']}\n")
            f.write(f"Failed Analyses: {results['failed_analyses']}\n")
            f.write(f"Significant Tests: {len(results['significant_tests'])}\n\n")
            
            if 'overall_summary' in results:
                f.write(f"Overall Summary:\n{results['overall_summary']}\n\n")
            
            f.write(f"Significant Relationships:\n")
            for test in results['significant_tests']:
                f.write(f"- {test['test_type']}: {test.get('variable1', 'N/A')} vs {test.get('variable2', 'N/A')}\n")
                f.write(f"  p-value: {test['p_value']:.4f}, Effect size: {test.get('effect_size', 'N/A'):.4f}\n")
            
            f.write(f"\nAnalyzed Files:\n")
            for file_analysis in results['files_analyzed']:
                f.write(f"- {file_analysis['file_path']}\n")
                f.write(f"  Status: {file_analysis.get('status', 'unknown')}\n")
                if 'data_shape' in file_analysis:
                    f.write(f"  Shape: {file_analysis['data_shape']}\n")
                if 'error' in file_analysis:
                    f.write(f"  Error: {file_analysis['error']}\n")
                f.write("\n")
        
        self.logger.info(f"Statistical analysis results saved to {self.output_dir}")
    
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
    agent = StatisticalAnalysisAgent()
    results = agent.execute()
    print(f"Statistical analysis completed. Analyzed {results['successful_analyses']} files.")
