"""
Visualization Agent - Simple Implementation

Creates comprehensive visualizations for all pipeline results.
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

class VisualizationAgent:
    """
    Visualization Agent for creating comprehensive charts and plots.
    
    Generates visualizations for EDA results, model performance,
    statistical findings, and overall pipeline results.
    """
    
    def __init__(self):
        """Initialize Visualization Agent."""
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = Path("output/visualization_01")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.logger.info("Visualization Agent initialized")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute visualization task.
        
        Returns:
            Dict containing visualization results and metadata
        """
        try:
            # Read context
            context = read_context()
            
            # Collect data from all previous agents
            pipeline_data = self._collect_pipeline_data(context)
            
            # Generate visualizations
            visualization_results = {
                'visualization_timestamp': datetime.now().isoformat(),
                'visualizations_created': [],
                'total_visualizations': 0,
                'successful_creations': 0,
                'failed_creations': 0,
                'visualization_paths': []
            }
            
            # Create different types of visualizations
            viz_types = [
                ('pipeline_summary', self._create_pipeline_summary_viz),
                ('data_quality', self._create_data_quality_viz),
                ('model_performance', self._create_model_performance_viz),
                ('statistical_findings', self._create_statistical_findings_viz),
                ('insight_synthesis', self._create_insight_synthesis_viz)
            ]
            
            for viz_type, viz_function in viz_types:
                try:
                    viz_path = viz_function(pipeline_data)
                    if viz_path:
                        visualization_results['visualizations_created'].append({
                            'type': viz_type,
                            'path': str(viz_path),
                            'status': 'success'
                        })
                        visualization_results['visualization_paths'].append(str(viz_path))
                        visualization_results['successful_creations'] += 1
                        visualization_results['total_visualizations'] += 1
                except Exception as e:
                    self.logger.error(f"Failed to create {viz_type} visualization: {e}")
                    visualization_results['visualizations_created'].append({
                        'type': viz_type,
                        'error': str(e),
                        'status': 'failed'
                    })
                    visualization_results['failed_creations'] += 1
                    visualization_results['total_visualizations'] += 1
            
            # Generate overall visualization summary
            if visualization_results['successful_creations'] > 0:
                visualization_results['overall_summary'] = self._generate_overall_summary(visualization_results)
            
            # Update context
            update_context_chain(context, 'visualization', visualization_results)
            
            # Log completion
            log_step(context, 'visualization', 
                    f"Created {visualization_results['successful_creations']} visualizations successfully")
            write_context(context)
            
            # Save detailed results
            self._save_visualization_results(visualization_results)
            
            self.logger.info(f"Visualization completed successfully with {visualization_results['successful_creations']} visualizations")
            
            return visualization_results
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            
            # Log error in context
            context = read_context()
            error_data = {
                'error': str(e),
                'status': 'failed',
                'visualization_timestamp': datetime.now().isoformat()
            }
            update_context_chain(context, 'visualization', error_data)
            log_step(context, 'visualization', f"Error: {str(e)}")
            write_context(context)
            
            raise
    
    def _collect_pipeline_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect data from all pipeline agents."""
        pipeline_data = {}
        
        # Collect from each agent
        agents = [
            'data_quality', 'eda', 'feature_engineering', 'statistical_analysis',
            'model_architecture', 'hyperparameter_optimization', 'model_validation',
            'insight_synthesis'
        ]
        
        for agent in agents:
            agent_data = get_context_chain_data(context, agent)
            if agent_data and agent_data.get('status') != 'failed':
                pipeline_data[agent] = agent_data
        
        return pipeline_data
    
    def _create_pipeline_summary_viz(self, pipeline_data: Dict[str, Any]) -> Optional[Path]:
        """Create pipeline summary visualization."""
        # Create a summary dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AI Research Pipeline Summary Dashboard', fontsize=16, fontweight='bold')
        
        # Agent completion status
        agent_status = []
        agent_names = []
        for agent, data in pipeline_data.items():
            if data and data.get('status') != 'failed':
                agent_status.append(1)
                agent_names.append(agent.replace('_', ' ').title())
            else:
                agent_status.append(0)
                agent_names.append(agent.replace('_', ' ').title())
        
        # Plot 1: Agent completion status
        axes[0, 0].bar(range(len(agent_names)), agent_status, color=['green' if s == 1 else 'red' for s in agent_status])
        axes[0, 0].set_title('Agent Completion Status')
        axes[0, 0].set_ylabel('Status (1=Success, 0=Failed)')
        axes[0, 0].set_xticks(range(len(agent_names)))
        axes[0, 0].set_xticklabels(agent_names, rotation=45, ha='right')
        
        # Plot 2: Data processing summary
        if 'data_quality' in pipeline_data:
            dq_data = pipeline_data['data_quality']
            successful = dq_data.get('successful_cleanings', 0)
            failed = dq_data.get('failed_cleanings', 0)
            total = dq_data.get('total_files', 0)
            
            if total > 0:
                axes[0, 1].pie([successful, failed], labels=['Successful', 'Failed'], 
                              autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
                axes[0, 1].set_title('Data Quality Processing Results')
        
        # Plot 3: Model performance (if available)
        if 'model_validation' in pipeline_data:
            mv_data = pipeline_data['model_validation']
            if 'validation_metrics' in mv_data:
                metrics = list(mv_data['validation_metrics'].values())
                if metrics:
                    # Extract accuracy or R² scores
                    scores = []
                    for metric in metrics:
                        score = metric.get('accuracy', metric.get('r2_score', 0))
                        scores.append(score)
                    
                    if scores:
                        axes[1, 0].hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                        axes[1, 0].set_title('Model Performance Distribution')
                        axes[1, 0].set_xlabel('Performance Score')
                        axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Statistical findings
        if 'statistical_analysis' in pipeline_data:
            stats_data = pipeline_data['statistical_analysis']
            significant = stats_data.get('significant_tests', [])
            total_tests = len(stats_data.get('correlation_tests', [])) + len(stats_data.get('hypothesis_tests', []))
            
            if total_tests > 0:
                significant_count = len(significant)
                non_significant = total_tests - significant_count
                
                axes[1, 1].pie([significant_count, non_significant], 
                              labels=['Significant', 'Non-significant'], 
                              autopct='%1.1f%%', colors=['lightblue', 'lightgray'])
                axes[1, 1].set_title('Statistical Test Results')
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = self.output_dir / "pipeline_summary_dashboard.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
    
    def _create_data_quality_viz(self, pipeline_data: Dict[str, Any]) -> Optional[Path]:
        """Create data quality visualization."""
        if 'data_quality' not in pipeline_data:
            return None
        
        dq_data = pipeline_data['data_quality']
        
        # Create data quality summary
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Data Quality Analysis', fontsize=14, fontweight='bold')
        
        # Plot 1: Processing results
        successful = dq_data.get('successful_cleanings', 0)
        failed = dq_data.get('failed_cleanings', 0)
        
        axes[0].bar(['Successful', 'Failed'], [successful, failed], 
                   color=['green', 'red'], alpha=0.7)
        axes[0].set_title('Data Processing Results')
        axes[0].set_ylabel('Number of Files')
        
        # Plot 2: Quality metrics (if available)
        if 'quality_metrics' in dq_data:
            metrics = dq_data['quality_metrics']
            if 'overall_quality_score' in metrics:
                score = metrics['overall_quality_score']
                axes[1].bar(['Overall Quality'], [score], color='blue', alpha=0.7)
                axes[1].set_ylim(0, 1)
                axes[1].set_title('Overall Data Quality Score')
                axes[1].set_ylabel('Quality Score')
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = self.output_dir / "data_quality_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
    
    def _create_model_performance_viz(self, pipeline_data: Dict[str, Any]) -> Optional[Path]:
        """Create model performance visualization."""
        if 'model_validation' not in pipeline_data:
            return None
        
        mv_data = pipeline_data['model_validation']
        
        if 'validation_metrics' not in mv_data:
            return None
        
        # Create model performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Analysis', fontsize=14, fontweight='bold')
        
        metrics_data = mv_data['validation_metrics']
        
        # Extract metrics
        model_names = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for file_path, metrics in metrics_data.items():
            model_name = Path(file_path).stem
            model_names.append(model_name)
            accuracy_scores.append(metrics.get('accuracy', 0))
            precision_scores.append(metrics.get('precision', 0))
            recall_scores.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1_score', 0))
        
        if model_names:
            x = range(len(model_names))
            
            # Plot 1: Accuracy comparison
            axes[0, 0].bar(x, accuracy_scores, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Model Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
            
            # Plot 2: Precision comparison
            axes[0, 1].bar(x, precision_scores, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Model Precision Comparison')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
            
            # Plot 3: Recall comparison
            axes[1, 0].bar(x, recall_scores, color='lightcoral', alpha=0.7)
            axes[1, 0].set_title('Model Recall Comparison')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
            
            # Plot 4: F1 Score comparison
            axes[1, 1].bar(x, f1_scores, color='gold', alpha=0.7)
            axes[1, 1].set_title('Model F1 Score Comparison')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = self.output_dir / "model_performance_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
    
    def _create_statistical_findings_viz(self, pipeline_data: Dict[str, Any]) -> Optional[Path]:
        """Create statistical findings visualization."""
        if 'statistical_analysis' not in pipeline_data:
            return None
        
        stats_data = pipeline_data['statistical_analysis']
        
        # Create statistical findings summary
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Statistical Analysis Results', fontsize=14, fontweight='bold')
        
        # Plot 1: Test results summary
        correlation_tests = len(stats_data.get('correlation_tests', []))
        hypothesis_tests = len(stats_data.get('hypothesis_tests', []))
        significant_tests = len(stats_data.get('significant_tests', []))
        
        test_types = ['Correlation', 'Hypothesis', 'Significant']
        test_counts = [correlation_tests, hypothesis_tests, significant_tests]
        
        axes[0].bar(test_types, test_counts, color=['blue', 'green', 'red'], alpha=0.7)
        axes[0].set_title('Statistical Tests Summary')
        axes[0].set_ylabel('Number of Tests')
        
        # Plot 2: Significance rate
        total_tests = correlation_tests + hypothesis_tests
        if total_tests > 0:
            significance_rate = significant_tests / total_tests
            axes[1].pie([significance_rate, 1-significance_rate], 
                       labels=['Significant', 'Non-significant'], 
                       autopct='%1.1f%%', colors=['lightblue', 'lightgray'])
            axes[1].set_title('Significance Rate')
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = self.output_dir / "statistical_findings_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
    
    def _create_insight_synthesis_viz(self, pipeline_data: Dict[str, Any]) -> Optional[Path]:
        """Create insight synthesis visualization."""
        if 'insight_synthesis' not in pipeline_data:
            return None
        
        synthesis_data = pipeline_data['insight_synthesis']
        
        # Create insight synthesis summary
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Insight Synthesis Summary', fontsize=14, fontweight='bold')
        
        # Plot 1: Key findings count
        key_findings = len(synthesis_data.get('key_findings', []))
        recommendations = len(synthesis_data.get('recommendations', []))
        
        categories = ['Key Findings', 'Recommendations']
        counts = [key_findings, recommendations]
        
        axes[0].bar(categories, counts, color=['orange', 'purple'], alpha=0.7)
        axes[0].set_title('Insight Categories')
        axes[0].set_ylabel('Count')
        
        # Plot 2: Overall conclusions
        conclusions = synthesis_data.get('overall_conclusions', {})
        if conclusions:
            conclusion_names = list(conclusions.keys())
            conclusion_values = [1 if v else 0 for v in conclusions.values()]
            
            axes[1].bar(conclusion_names, conclusion_values, 
                       color=['green' if v else 'red' for v in conclusion_values], alpha=0.7)
            axes[1].set_title('Overall Conclusions')
            axes[1].set_ylabel('Status (1=Success, 0=Failed)')
            axes[1].set_xticklabels(conclusion_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = self.output_dir / "insight_synthesis_summary.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
    
    def _generate_overall_summary(self, visualization_results: Dict[str, Any]) -> str:
        """Generate overall visualization summary."""
        successful_creations = visualization_results['successful_creations']
        total_visualizations = visualization_results['total_visualizations']
        
        summary = f"""
        Visualization Summary:
        - Visualizations created: {successful_creations}/{total_visualizations}
        - Dashboard created: Pipeline summary dashboard
        - Analysis charts: Data quality, model performance, statistical findings
        - Insight synthesis: Overall conclusions and recommendations
        
        Key visualizations:
        - Pipeline summary dashboard with agent completion status
        - Data quality analysis with processing results
        - Model performance comparison across all validated models
        - Statistical findings summary with significance rates
        - Insight synthesis summary with key findings and recommendations
        """
        
        return summary
    
    def _save_visualization_results(self, results: Dict[str, Any]):
        """Save detailed visualization results to output directory."""
        # Save full results
        results_file = self.output_dir / "visualization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = self.output_dir / "visualization_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Visualization Summary\n")
            f.write(f"===================\n\n")
            f.write(f"Visualization Timestamp: {results['visualization_timestamp']}\n")
            f.write(f"Total Visualizations: {results['total_visualizations']}\n")
            f.write(f"Successful Creations: {results['successful_creations']}\n")
            f.write(f"Failed Creations: {results['failed_creations']}\n\n")
            
            if 'overall_summary' in results:
                f.write(f"Overall Summary:\n{results['overall_summary']}\n\n")
            
            f.write(f"Created Visualizations:\n")
            for viz in results['visualizations_created']:
                f.write(f"- {viz['type']}: {viz.get('path', 'N/A')}\n")
                f.write(f"  Status: {viz.get('status', 'unknown')}\n")
                if 'error' in viz:
                    f.write(f"  Error: {viz['error']}\n")
                f.write("\n")
        
        self.logger.info(f"Visualization results saved to {self.output_dir}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run agent
    agent = VisualizationAgent()
    results = agent.execute()
    print(f"Visualization completed with {results['successful_creations']} visualizations.")
