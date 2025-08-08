"""
Insight Synthesis Agent - Simple Implementation

Synthesizes insights from all previous agents into actionable conclusions.
Follows the master specification for agent implementation.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.core.context_manager import (
    read_context, write_context, log_step, 
    update_context_chain, get_context_chain_data
)

class InsightSynthesisAgent:
    """
    Insight Synthesis Agent for combining all pipeline results.
    
    Synthesizes insights from EDA, statistical analysis, model performance,
    and other agents to generate actionable conclusions and recommendations.
    """
    
    def __init__(self):
        """Initialize Insight Synthesis Agent."""
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = Path("output/insight_synthesis_01")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Insight Synthesis Agent initialized")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute insight synthesis task.
        
        Returns:
            Dict containing synthesized insights and recommendations
        """
        try:
            # Read context
            context = read_context()
            
            # Collect insights from all previous agents
            insights = self._collect_insights(context)
            
            # Synthesize insights
            synthesis_results = {
                'synthesis_timestamp': datetime.now().isoformat(),
                'insights_collected': len(insights),
                'key_findings': [],
                'recommendations': [],
                'data_quality_summary': {},
                'statistical_insights': {},
                'model_performance_insights': {},
                'overall_conclusions': {}
            }
            
            # Analyze data quality insights
            if 'data_quality' in insights:
                synthesis_results['data_quality_summary'] = self._analyze_data_quality_insights(insights['data_quality'])
            
            # Analyze statistical insights
            if 'statistical_analysis' in insights:
                synthesis_results['statistical_insights'] = self._analyze_statistical_insights(insights['statistical_analysis'])
            
            # Analyze model performance insights
            if 'model_validation' in insights:
                synthesis_results['model_performance_insights'] = self._analyze_model_performance_insights(insights['model_validation'])
            
            # Generate key findings
            synthesis_results['key_findings'] = self._generate_key_findings(synthesis_results)
            
            # Generate recommendations
            synthesis_results['recommendations'] = self._generate_recommendations(synthesis_results)
            
            # Generate overall conclusions
            synthesis_results['overall_conclusions'] = self._generate_overall_conclusions(synthesis_results)
            
            # Update context
            update_context_chain(context, 'insight_synthesis', synthesis_results)
            
            # Log completion
            log_step(context, 'insight_synthesis', 
                    f"Completed insight synthesis with {len(synthesis_results['key_findings'])} key findings")
            write_context(context)
            
            # Save detailed results
            self._save_synthesis_results(synthesis_results)
            
            self.logger.info(f"Insight synthesis completed successfully")
            
            return synthesis_results
            
        except Exception as e:
            self.logger.error(f"Insight synthesis failed: {e}")
            
            # Log error in context
            context = read_context()
            error_data = {
                'error': str(e),
                'status': 'failed',
                'synthesis_timestamp': datetime.now().isoformat()
            }
            update_context_chain(context, 'insight_synthesis', error_data)
            log_step(context, 'insight_synthesis', f"Error: {str(e)}")
            write_context(context)
            
            raise
    
    def _collect_insights(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect insights from all previous agents."""
        insights = {}
        
        # Collect from each agent
        agents = [
            'data_quality', 'eda', 'feature_engineering', 'statistical_analysis',
            'model_architecture', 'hyperparameter_optimization', 'model_validation'
        ]
        
        for agent in agents:
            agent_data = get_context_chain_data(context, agent)
            if agent_data and agent_data.get('status') != 'failed':
                insights[agent] = agent_data
        
        return insights
    
    def _analyze_data_quality_insights(self, data_quality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data quality insights."""
        analysis = {
            'total_files_processed': data_quality_data.get('total_files', 0),
            'successful_cleanings': data_quality_data.get('successful_cleanings', 0),
            'failed_cleanings': data_quality_data.get('failed_cleanings', 0),
            'data_quality_score': 0,
            'key_issues': []
        }
        
        if analysis['total_files_processed'] > 0:
            analysis['data_quality_score'] = analysis['successful_cleanings'] / analysis['total_files_processed']
        
        # Identify key issues
        if 'files_processed' in data_quality_data:
            for file_result in data_quality_data['files_processed']:
                if file_result.get('status') == 'failed':
                    analysis['key_issues'].append(f"Failed to clean {file_result.get('original_path', 'unknown file')}")
        
        return analysis
    
    def _analyze_statistical_insights(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze statistical analysis insights."""
        analysis = {
            'total_tests_performed': 0,
            'significant_relationships': 0,
            'correlation_tests': 0,
            'hypothesis_tests': 0,
            'key_statistical_findings': []
        }
        
        if 'correlation_tests' in stats_data:
            analysis['correlation_tests'] = len(stats_data['correlation_tests'])
            analysis['total_tests_performed'] += analysis['correlation_tests']
        
        if 'hypothesis_tests' in stats_data:
            analysis['hypothesis_tests'] = len(stats_data['hypothesis_tests'])
            analysis['total_tests_performed'] += analysis['hypothesis_tests']
        
        if 'significant_tests' in stats_data:
            analysis['significant_relationships'] = len(stats_data['significant_tests'])
            
            # Extract key findings from significant tests
            for test in stats_data['significant_tests'][:5]:  # Top 5 significant findings
                if test.get('test_type') == 'pearson_correlation':
                    analysis['key_statistical_findings'].append(
                        f"Strong correlation ({test.get('correlation', 0):.3f}) between "
                        f"{test.get('variable1', 'N/A')} and {test.get('variable2', 'N/A')}"
                    )
                elif test.get('test_type') == 't_test':
                    analysis['key_statistical_findings'].append(
                        f"Significant difference in {test.get('dependent_variable', 'N/A')} "
                        f"across {test.get('independent_variable', 'N/A')} groups"
                    )
        
        return analysis
    
    def _analyze_model_performance_insights(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model performance insights."""
        analysis = {
            'total_models_validated': validation_data.get('total_models', 0),
            'successful_validations': validation_data.get('successful_validations', 0),
            'average_performance': {},
            'best_performing_model': None,
            'performance_insights': []
        }
        
        # Calculate average performance metrics
        if 'validation_metrics' in validation_data:
            metrics_summary = {}
            for file_path, metrics in validation_data['validation_metrics'].items():
                for metric_name, value in metrics.items():
                    if metric_name not in metrics_summary:
                        metrics_summary[metric_name] = []
                    metrics_summary[metric_name].append(value)
            
            # Calculate averages
            for metric_name, values in metrics_summary.items():
                if values:
                    analysis['average_performance'][metric_name] = sum(values) / len(values)
        
        # Find best performing model
        if 'validation_metrics' in validation_data:
            best_score = -1
            for file_path, metrics in validation_data['validation_metrics'].items():
                # Use accuracy for classification or R² for regression
                score = metrics.get('accuracy', metrics.get('r2_score', 0))
                if score > best_score:
                    best_score = score
                    analysis['best_performing_model'] = file_path
        
        return analysis
    
    def _generate_key_findings(self, synthesis_results: Dict[str, Any]) -> List[str]:
        """Generate key findings from all analyses."""
        findings = []
        
        # Data quality findings
        data_quality = synthesis_results['data_quality_summary']
        if data_quality.get('data_quality_score', 0) > 0.8:
            findings.append("High data quality achieved with minimal cleaning required")
        elif data_quality.get('data_quality_score', 0) > 0.5:
            findings.append("Moderate data quality with some cleaning issues resolved")
        else:
            findings.append("Data quality issues identified requiring attention")
        
        # Statistical findings
        stats = synthesis_results['statistical_insights']
        if stats.get('significant_relationships', 0) > 0:
            findings.append(f"Found {stats['significant_relationships']} statistically significant relationships")
        
        # Model performance findings
        model_perf = synthesis_results['model_performance_insights']
        if model_perf.get('successful_validations', 0) > 0:
            findings.append(f"Successfully validated {model_perf['successful_validations']} models")
            
            # Add performance insights
            avg_perf = model_perf.get('average_performance', {})
            if 'accuracy' in avg_perf and avg_perf['accuracy'] > 0.8:
                findings.append("High model accuracy achieved across validated models")
            elif 'r2_score' in avg_perf and avg_perf['r2_score'] > 0.7:
                findings.append("Good model performance with high R² scores")
        
        return findings
    
    def _generate_recommendations(self, synthesis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Data quality recommendations
        data_quality = synthesis_results['data_quality_summary']
        if data_quality.get('failed_cleanings', 0) > 0:
            recommendations.append("Address data quality issues in failed cleaning operations")
        
        # Statistical recommendations
        stats = synthesis_results['statistical_insights']
        if stats.get('significant_relationships', 0) > 0:
            recommendations.append("Focus on variables with significant relationships for feature engineering")
        
        # Model performance recommendations
        model_perf = synthesis_results['model_performance_insights']
        if model_perf.get('successful_validations', 0) > 0:
            recommendations.append("Consider ensemble methods to improve model performance")
            recommendations.append("Implement cross-validation for more robust model evaluation")
        
        # General recommendations
        recommendations.append("Document all preprocessing steps for reproducibility")
        recommendations.append("Consider feature importance analysis for model interpretability")
        
        return recommendations
    
    def _generate_overall_conclusions(self, synthesis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall conclusions."""
        conclusions = {
            'research_question_addressed': True,
            'data_suitable_for_analysis': True,
            'models_performed_adequately': True,
            'insights_generated': len(synthesis_results['key_findings']) > 0,
            'recommendations_provided': len(synthesis_results['recommendations']) > 0,
            'pipeline_success': True
        }
        
        # Assess overall success
        data_quality = synthesis_results['data_quality_summary']
        if data_quality.get('data_quality_score', 0) < 0.5:
            conclusions['data_suitable_for_analysis'] = False
            conclusions['pipeline_success'] = False
        
        model_perf = synthesis_results['model_performance_insights']
        if model_perf.get('successful_validations', 0) == 0:
            conclusions['models_performed_adequately'] = False
            conclusions['pipeline_success'] = False
        
        return conclusions
    
    def _save_synthesis_results(self, results: Dict[str, Any]):
        """Save detailed insight synthesis results to output directory."""
        # Save full results
        results_file = self.output_dir / "insight_synthesis_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = self.output_dir / "insight_synthesis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Insight Synthesis Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Synthesis Timestamp: {results['synthesis_timestamp']}\n")
            f.write(f"Insights Collected: {results['insights_collected']}\n")
            f.write(f"Key Findings: {len(results['key_findings'])}\n")
            f.write(f"Recommendations: {len(results['recommendations'])}\n\n")
            
            f.write(f"Key Findings:\n")
            for i, finding in enumerate(results['key_findings'], 1):
                f.write(f"{i}. {finding}\n")
            f.write("\n")
            
            f.write(f"Recommendations:\n")
            for i, recommendation in enumerate(results['recommendations'], 1):
                f.write(f"{i}. {recommendation}\n")
            f.write("\n")
            
            f.write(f"Overall Conclusions:\n")
            conclusions = results['overall_conclusions']
            for key, value in conclusions.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
            
            # Add detailed analysis summaries
            if results['data_quality_summary']:
                f.write(f"Data Quality Summary:\n")
                dq = results['data_quality_summary']
                f.write(f"- Total files processed: {dq.get('total_files_processed', 0)}\n")
                f.write(f"- Successful cleanings: {dq.get('successful_cleanings', 0)}\n")
                f.write(f"- Data quality score: {dq.get('data_quality_score', 0):.3f}\n")
                f.write("\n")
            
            if results['statistical_insights']:
                f.write(f"Statistical Insights:\n")
                stats = results['statistical_insights']
                f.write(f"- Total tests performed: {stats.get('total_tests_performed', 0)}\n")
                f.write(f"- Significant relationships: {stats.get('significant_relationships', 0)}\n")
                f.write(f"- Key findings: {len(stats.get('key_statistical_findings', []))}\n")
                f.write("\n")
            
            if results['model_performance_insights']:
                f.write(f"Model Performance Insights:\n")
                perf = results['model_performance_insights']
                f.write(f"- Models validated: {perf.get('successful_validations', 0)}\n")
                f.write(f"- Best performing model: {perf.get('best_performing_model', 'N/A')}\n")
                f.write(f"- Average performance metrics: {len(perf.get('average_performance', {}))}\n")
                f.write("\n")
        
        self.logger.info(f"Insight synthesis results saved to {self.output_dir}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run agent
    agent = InsightSynthesisAgent()
    results = agent.execute()
    print(f"Insight synthesis completed with {len(results['key_findings'])} key findings.")
