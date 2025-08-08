"""
AI Research Pipeline - Main Orchestrator

Runs all agents in the exact order specified in the master specification.
Each agent reads context/context_output.json, modifies only its relevant keys, and writes back the updated file.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.context_manager import (
    read_context, write_context, log_step, set_research_question
)
from src.agents.dataset_discovery_agent import DatasetDiscoveryAgent
from src.agents.data_acquisition_agent import DataAcquisitionAgent
from src.agents.data_quality_agent import DataQualityAgent
from src.agents.documentation_agent import DocumentationAgent
from src.agents.eda_agent_simple import EDAAgent
from src.agents.feature_engineering_agent_simple import FeatureEngineeringAgent
from src.agents.statistical_analysis_agent_simple import StatisticalAnalysisAgent
from src.agents.model_architecture_agent_simple import ModelArchitectureAgent
from src.agents.hyperparameter_optimization_agent_simple import HyperparameterOptimizationAgent
from src.agents.model_validation_agent_simple import ModelValidationAgent
from src.agents.insight_synthesis_agent_simple import InsightSynthesisAgent
from src.agents.visualization_agent_simple import VisualizationAgent
from src.agents.final_report_generator import FinalReportGenerator

class PipelineOrchestrator:
    """
    Main orchestrator for the AI Research Pipeline.
    
    Executes all agents in the exact order specified in the master specification.
    Each agent reads context, modifies only its relevant keys, and writes back the updated file.
    """
    
    def __init__(self):
        """Initialize the pipeline orchestrator."""
        self.logger = logging.getLogger(__name__)
        
        # Define the pipeline order as specified in the master specification
        self.pipeline_order = [
            ('dataset_discovery', DatasetDiscoveryAgent),
            ('data_acquisition', DataAcquisitionAgent),
            ('data_quality', DataQualityAgent),
            ('documentation', DocumentationAgent),
            ('eda', EDAAgent),
            ('feature_engineering', FeatureEngineeringAgent),
            ('statistical_analysis', StatisticalAnalysisAgent),
            ('model_architecture', ModelArchitectureAgent),
            ('hyperparameter_optimization', HyperparameterOptimizationAgent),
            ('model_validation', ModelValidationAgent),
            ('insight_synthesis', InsightSynthesisAgent),
            ('visualization', VisualizationAgent),
            ('final_report', FinalReportGenerator),
        ]
        
        # Pipeline execution results
        self.execution_results = {
            'start_time': None,
            'end_time': None,
            'total_agents': len(self.pipeline_order),
            'successful_agents': 0,
            'failed_agents': 0,
            'agent_results': {},
            'pipeline_status': 'not_started'
        }
        
        self.logger.info(f"Pipeline Orchestrator initialized with {len(self.pipeline_order)} agents")
    
    def run_pipeline(self, research_question: str) -> Dict[str, Any]:
        """
        Run the complete AI research pipeline.
        
        Args:
            research_question: The research question to investigate
            
        Returns:
            Dict containing pipeline execution results
        """
        self.logger.info(f"Starting AI Research Pipeline for: {research_question}")
        
        # Initialize context with research question
        self._initialize_context(research_question)
        
        # Record start time
        self.execution_results['start_time'] = datetime.now().isoformat()
        self.execution_results['pipeline_status'] = 'running'
        
        try:
            # Execute each agent in order
            for agent_name, agent_class in self.pipeline_order:
                self.logger.info(f"Executing agent: {agent_name}")
                
                try:
                    # Create and execute agent
                    agent = agent_class()
                    result = agent.execute()
                    
                    # Record successful execution
                    self.execution_results['agent_results'][agent_name] = {
                        'status': 'success',
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.execution_results['successful_agents'] += 1
                    
                    self.logger.info(f"Agent {agent_name} completed successfully")
                    
                except Exception as e:
                    self.logger.error(f"Agent {agent_name} failed: {e}")
                    
                    # Record failed execution
                    self.execution_results['agent_results'][agent_name] = {
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    self.execution_results['failed_agents'] += 1
                    
                    # Log the error in context
                    context = read_context()
                    log_step(context, 'pipeline_orchestrator', f"Agent {agent_name} failed: {str(e)}")
                    write_context(context)
                    
                    # Continue with next agent (don't stop the pipeline)
                    continue
            
            # Record end time and final status
            self.execution_results['end_time'] = datetime.now().isoformat()
            self.execution_results['pipeline_status'] = 'completed'
            
            # Log pipeline completion
            context = read_context()
            log_step(context, 'pipeline_orchestrator', 
                    f"Pipeline completed. {self.execution_results['successful_agents']} agents succeeded, "
                    f"{self.execution_results['failed_agents']} agents failed")
            write_context(context)
            
            self.logger.info(f"Pipeline completed. {self.execution_results['successful_agents']} agents succeeded, "
                           f"{self.execution_results['failed_agents']} agents failed")
            
            return self.execution_results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            self.execution_results['pipeline_status'] = 'failed'
            self.execution_results['end_time'] = datetime.now().isoformat()
            
            # Log the error in context
            context = read_context()
            log_step(context, 'pipeline_orchestrator', f"Pipeline failed: {str(e)}")
            write_context(context)
            
            raise
    
    def _initialize_context(self, research_question: str):
        """
        Initialize the context with the research question.
        
        Args:
            research_question: The research question to investigate
        """
        context = read_context()
        set_research_question(context, research_question)
        log_step(context, 'pipeline_orchestrator', f"Pipeline started for research question: {research_question}")
        write_context(context)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get the current status of the pipeline.
        
        Returns:
            Dict containing pipeline status information
        """
        context = read_context()
        
        status = {
            'pipeline_status': self.execution_results['pipeline_status'],
            'total_agents': self.execution_results['total_agents'],
            'successful_agents': self.execution_results['successful_agents'],
            'failed_agents': self.execution_results['failed_agents'],
            'research_question': context.get('project_metadata', {}).get('research_question', ''),
            'pipeline_log': context.get('pipeline_log', [])
        }
        
        return status
    
    def save_pipeline_results(self):
        """Save pipeline execution results to file."""
        results_dir = Path("output/pipeline_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save execution results
        results_file = results_dir / "pipeline_execution_results.json"
        with open(results_file, 'w') as f:
            import json
            json.dump(self.execution_results, f, indent=2)
        
        # Save final context
        context = read_context()
        context_file = results_dir / "final_context.json"
        with open(context_file, 'w') as f:
            json.dump(context, f, indent=2)
        
        self.logger.info(f"Pipeline results saved to {results_dir}")


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/pipeline.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description='AI Research Pipeline')
    parser.add_argument('--research-question', '-r', required=True,
                       help='The research question to investigate')
    parser.add_argument('--status', '-s', action='store_true',
                       help='Check pipeline status')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator()
    
    if args.status:
        # Check pipeline status
        status = orchestrator.get_pipeline_status()
        print(f"Pipeline Status: {status['pipeline_status']}")
        print(f"Research Question: {status['research_question']}")
        print(f"Successful Agents: {status['successful_agents']}/{status['total_agents']}")
        print(f"Failed Agents: {status['failed_agents']}")
        
        if status['pipeline_log']:
            print("\nRecent Pipeline Log:")
            for log_entry in status['pipeline_log'][-5:]:  # Show last 5 entries
                print(f"  {log_entry['timestamp']} - {log_entry['agent']}: {log_entry['message']}")
    else:
        # Run the pipeline
        try:
            results = orchestrator.run_pipeline(args.research_question)
            orchestrator.save_pipeline_results()
            
            print(f"\nPipeline completed!")
            print(f"Status: {results['pipeline_status']}")
            print(f"Successful Agents: {results['successful_agents']}/{results['total_agents']}")
            print(f"Failed Agents: {results['failed_agents']}")
            
            if results['failed_agents'] > 0:
                print("\nFailed Agents:")
                for agent_name, agent_result in results['agent_results'].items():
                    if agent_result['status'] == 'failed':
                        print(f"  {agent_name}: {agent_result['error']}")
            
        except Exception as e:
            print(f"Pipeline execution failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
