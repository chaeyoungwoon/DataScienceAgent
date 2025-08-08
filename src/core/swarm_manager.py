"""
Swarm Manager - Main Orchestrator for Data Science Agent Swarm

This module contains the SwarmManager class that orchestrates all agents,
manages the research workflow, and coordinates the entire data science
research process from data discovery to final report generation.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent, AgentContext


class ProjectStatus(Enum):
    """Enumeration of project status states."""
    INITIALIZING = "initializing"
    DATA_DISCOVERY = "data_discovery"
    DATA_ANALYSIS = "data_analysis"
    MODELING = "modeling"
    SYNTHESIS = "synthesis"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ResearchProject:
    """Data class representing a research project."""
    project_id: str
    research_question: str
    domain: str
    analysis_type: str
    parameters: Dict[str, Any]
    status: ProjectStatus
    created_at: datetime
    updated_at: datetime
    results: Dict[str, Any] = None
    agents_involved: List[str] = None


class SwarmManager:
    """
    Main orchestrator for the Data Science Agent Swarm.
    
    This class manages the entire research workflow, from initial project
    creation through data discovery, analysis, modeling, and final report
    generation. It coordinates all agents and ensures proper task distribution
    and result aggregation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the swarm manager.
        
        Args:
            config: Configuration dictionary for the swarm manager
        """
        self.config = config
        self.logger = self.setup_logging()
        
        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        
        # Project management
        self.active_projects: Dict[str, ResearchProject] = {}
        self.project_history: List[ResearchProject] = []
        
        # Task management
        self.task_queue: List[Dict[str, Any]] = []
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Communication
        self.message_bus = self.setup_message_bus()
        
        # Metrics and monitoring
        self.metrics_collector = self.setup_metrics_collector()
        
        self.logger.info("Swarm Manager initialized successfully")
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging for the swarm manager."""
        logger = logging.getLogger("swarm_manager")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler
        file_handler = logging.FileHandler('logs/swarm_manager.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def setup_message_bus(self):
        """Setup message bus for inter-agent communication."""
        # This would be implemented with Kafka or similar
        return None
    
    def setup_metrics_collector(self):
        """Setup metrics collection for monitoring."""
        try:
            from src.monitoring.metrics import MetricsCollector
            return MetricsCollector()
        except ImportError:
            self.logger.warning("Metrics collection not available")
            return None
    
    async def register_agent(self, agent: BaseAgent) -> bool:
        """
        Register an agent with the swarm manager.
        
        Args:
            agent: Agent instance to register
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            agent_info = agent.get_agent_info()
            agent_id = agent_info['agent_id']
            
            self.agents[agent_id] = agent
            self.agent_capabilities[agent_id] = agent.get_capabilities()
            
            self.logger.info(f"Registered agent: {agent_id} with capabilities: {agent.get_capabilities()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the swarm manager.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                await agent.stop()
                del self.agents[agent_id]
                del self.agent_capabilities[agent_id]
                
                self.logger.info(f"Unregistered agent: {agent_id}")
                return True
            else:
                self.logger.warning(f"Agent {agent_id} not found for unregistration")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def create_research_project(self, project_config: Dict[str, Any]) -> str:
        """
        Create a new research project.
        
        Args:
            project_config: Configuration for the research project
            
        Returns:
            Project ID for the created project
        """
        import uuid
        
        project_id = str(uuid.uuid4())
        
        project = ResearchProject(
            project_id=project_id,
            research_question=project_config['research_question'],
            domain=project_config.get('domain', 'general'),
            analysis_type=project_config.get('analysis_type', 'exploratory'),
            parameters=project_config.get('parameters', {}),
            status=ProjectStatus.INITIALIZING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            results={},
            agents_involved=[]
        )
        
        self.active_projects[project_id] = project
        self.logger.info(f"Created research project: {project_id}")
        
        return project_id
    
    async def execute_research_project(self, project_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complete research project from start to finish.
        
        This is the main entry point for research execution. It orchestrates
        the entire workflow from data discovery through final report generation.
        
        Args:
            project_config: Configuration for the research project
            
        Returns:
            Complete research results including all findings and recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            # Create project
            project_id = await self.create_research_project(project_config)
            project = self.active_projects[project_id]
            
            self.logger.info(f"Starting research project: {project_id}")
            self.logger.info(f"Research question: {project.research_question}")
            
            # Phase 1: Data Discovery
            await self._update_project_status(project_id, ProjectStatus.DATA_DISCOVERY)
            data_discovery_results = await self._orchestrate_data_discovery(project)
            
            # Phase 2: Data Acquisition
            await self._update_project_status(project_id, ProjectStatus.DATA_ANALYSIS)
            data_acquisition_results = await self._orchestrate_data_acquisition(project, data_discovery_results)
            
            # Phase 3: Data Analysis
            await self._update_project_status(project_id, ProjectStatus.DATA_ANALYSIS)
            analysis_results = await self._orchestrate_data_analysis(project, data_acquisition_results)
            
            # Phase 3: Modeling
            await self._update_project_status(project_id, ProjectStatus.MODELING)
            modeling_results = await self._orchestrate_modeling(project, data_discovery_results, analysis_results)
            
            # Phase 4: Synthesis and Communication
            await self._update_project_status(project_id, ProjectStatus.SYNTHESIS)
            synthesis_results = await self._orchestrate_synthesis(
                project, data_discovery_results, analysis_results, modeling_results
            )
            
            # Compile final results
            final_results = {
                'project_id': project_id,
                'research_question': project.research_question,
                'execution_time': (datetime.utcnow() - start_time).total_seconds(),
                'data_discovery': data_discovery_results,
                'analysis_results': analysis_results,
                'modeling_results': modeling_results,
                'synthesis_results': synthesis_results,
                'insights': synthesis_results.get('insights', {}),
                'recommendations': synthesis_results.get('recommendations', []),
                'visualizations': synthesis_results.get('visualizations', []),
                'documentation': synthesis_results.get('documentation', {})
            }
            
            # Update project with results
            project.results = final_results
            project.status = ProjectStatus.COMPLETED
            project.updated_at = datetime.utcnow()
            
            # Move to history
            self.project_history.append(project)
            del self.active_projects[project_id]
            
            self.logger.info(f"Research project {project_id} completed successfully")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Research project failed: {e}")
            
            if project_id in self.active_projects:
                project = self.active_projects[project_id]
                project.status = ProjectStatus.FAILED
                project.updated_at = datetime.utcnow()
            
            raise e
    
    async def _orchestrate_data_discovery(self, project: ResearchProject) -> Dict[str, Any]:
        """
        Orchestrate the data discovery phase.
        
        Args:
            project: Research project to execute data discovery for
            
        Returns:
            Data discovery results
        """
        self.logger.info(f"Starting data discovery for project {project.project_id}")
        
        # Find data scout agents
        data_scout_agents = self._find_agents_by_capability('dataset_discovery')
        
        if not data_scout_agents:
            raise RuntimeError("No data discovery agents available")
        
        # Create data discovery tasks
        tasks = []
        for agent in data_scout_agents:
            task = {
                'task_id': f"{project.project_id}_data_discovery_{agent.name}",
                'task_type': 'discover_datasets',
                'research_question': project.research_question,
                'domain': project.domain,
                'data_requirements': project.parameters.get('data_requirements', {}),
                'max_datasets': project.parameters.get('max_datasets', 5),
                'metadata': {
                    'user_prompt': project.research_question,
                    'domain': project.domain,
                    'analysis_type': project.analysis_type
                }
            }
            tasks.append((agent, task))
        
        # Execute tasks
        results = await self._execute_tasks_parallel(tasks)
        
        # Aggregate results
        aggregated_results = self._aggregate_data_discovery_results(results)
        
        self.logger.info(f"Data discovery completed. Found {len(aggregated_results.get('datasets', []))} datasets")
        
        return aggregated_results
    
    async def _orchestrate_data_acquisition(self, project: ResearchProject, data_discovery_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate the data acquisition phase.
        
        Args:
            project: Research project
            data_discovery_results: Results from data discovery phase
            
        Returns:
            Data acquisition results with processed datasets
        """
        self.logger.info(f"Starting data acquisition for project {project.project_id}")
        
        # Find data acquisition agents
        acquisition_agents = self._find_agents_by_capability('data_acquisition')
        
        if not acquisition_agents:
            raise RuntimeError("No data acquisition agents available")
        
        # Create data acquisition tasks for each dataset
        tasks = []
        datasets = data_discovery_results.get('datasets', [])
        
        for dataset in datasets[:project.parameters.get('max_datasets', 5)]:
            for agent in acquisition_agents:
                task = {
                    'task_id': f"{project.project_id}_acquisition_{agent.name}_{dataset.get('ref', 'unknown')}",
                    'task_type': 'acquire_dataset',
                    'dataset': dataset,
                    'acquisition_requirements': {
                        'download_dataset': True,
                        'validate_data': True,
                        'preprocess_data': True,
                        'save_processed_data': True
                    },
                    'research_question': project.research_question,
                    'metadata': {
                        'dataset_info': dataset,
                        'source_type': 'kaggle',
                        'preprocessing_steps': ['handle_missing_values', 'convert_data_types', 'normalize_numeric'],
                        'validation_checks': True
                    }
                }
                tasks.append((agent, task))
        
        # Execute tasks
        results = await self._execute_tasks_parallel(tasks)
        
        # Aggregate results
        aggregated_results = self._aggregate_data_acquisition_results(results)
        
        self.logger.info(f"Data acquisition completed for {len(datasets)} datasets")
        
        return aggregated_results
    
    async def _orchestrate_data_analysis(self, project: ResearchProject, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate the data analysis phase.
        
        Args:
            project: Research project
            data_results: Results from data discovery phase
            
        Returns:
            Data analysis results
        """
        self.logger.info(f"Starting data analysis for project {project.project_id}")
        
        # Find analysis agents
        analysis_agents = self._find_agents_by_capability('exploratory_data_analysis')
        
        if not analysis_agents:
            raise RuntimeError("No data analysis agents available")
        
        # Create analysis tasks for each processed dataset
        tasks = []
        processed_datasets = data_results.get('processed_datasets', {})
        
        for dataset_id, processed_data in processed_datasets.items():
            for agent in analysis_agents:
                task = {
                    'task_id': f"{project.project_id}_analysis_{agent.name}_{dataset_id}",
                    'task_type': 'analyze_dataset',
                    'dataset': {
                        'dataset_id': dataset_id,
                        'processed_data': processed_data,
                        'metadata': data_results.get('acquisition_metadata', {}).get(dataset_id, {})
                    },
                    'analysis_requirements': {
                        'exploratory_analysis': True,
                        'statistical_analysis': True,
                        'feature_engineering': True
                    },
                    'research_question': project.research_question,
                    'metadata': {
                        'dataset_path': data_results.get('dataset_paths', {}).get(dataset_id, ''),
                        'validation_report': data_results.get('validation_reports', {}).get(dataset_id, {})
                    }
                }
                tasks.append((agent, task))
        
        # Execute tasks
        results = await self._execute_tasks_parallel(tasks)
        
        # Aggregate results
        aggregated_results = self._aggregate_analysis_results(results)
        
        self.logger.info(f"Data analysis completed for {len(processed_datasets)} datasets")
        
        return aggregated_results
    
    async def _orchestrate_modeling(self, project: ResearchProject, data_results: Dict[str, Any], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate the modeling phase.
        
        Args:
            project: Research project
            data_results: Results from data discovery phase
            analysis_results: Results from analysis phase
            
        Returns:
            Modeling results
        """
        self.logger.info(f"Starting modeling for project {project.project_id}")
        
        # Find modeling agents
        modeling_agents = self._find_agents_by_capability('modeling')
        
        if not modeling_agents:
            raise RuntimeError("No modeling agents available")
        
        # Create modeling tasks
        tasks = []
        for agent in modeling_agents:
            task = {
                'task_id': f"{project.project_id}_modeling_{agent.name}",
                'task_type': 'develop_models',
                'datasets': data_results.get('datasets', []),
                'analysis_results': analysis_results,
                'modeling_requirements': {
                    'problem_type': project.analysis_type,
                    'optimization_metric': project.parameters.get('optimization_metric', 'accuracy'),
                    'validation_strategy': 'cross_validation'
                }
            }
            tasks.append((agent, task))
        
        # Execute tasks
        results = await self._execute_tasks_parallel(tasks)
        
        # Aggregate results
        aggregated_results = self._aggregate_modeling_results(results)
        
        self.logger.info(f"Modeling completed with {len(aggregated_results.get('models', []))} models")
        
        return aggregated_results
    
    async def _orchestrate_synthesis(self, project: ResearchProject, data_results: Dict[str, Any], 
                                   analysis_results: Dict[str, Any], modeling_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate the synthesis and communication phase.
        
        Args:
            project: Research project
            data_results: Results from data discovery phase
            analysis_results: Results from analysis phase
            modeling_results: Results from modeling phase
            
        Returns:
            Synthesis results including insights, visualizations, and documentation
        """
        self.logger.info(f"Starting synthesis for project {project.project_id}")
        
        # Find synthesis agents
        synthesis_agents = self._find_agents_by_capability('synthesis')
        
        if not synthesis_agents:
            raise RuntimeError("No synthesis agents available")
        
        # Create synthesis tasks
        tasks = []
        for agent in synthesis_agents:
            task = {
                'task_id': f"{project.project_id}_synthesis_{agent.name}",
                'task_type': 'synthesize_results',
                'data_results': data_results,
                'analysis_results': analysis_results,
                'modeling_results': modeling_results,
                'research_question': project.research_question,
                'synthesis_requirements': {
                    'generate_insights': True,
                    'create_visualizations': True,
                    'generate_documentation': True
                }
            }
            tasks.append((agent, task))
        
        # Execute tasks
        results = await self._execute_tasks_parallel(tasks)
        
        # Aggregate results
        aggregated_results = self._aggregate_synthesis_results(results)
        
        self.logger.info(f"Synthesis completed with {len(aggregated_results.get('insights', []))} insights")
        
        return aggregated_results
    
    def _find_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """
        Find agents that have a specific capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of agents with the specified capability
        """
        matching_agents = []
        
        for agent_id, capabilities in self.agent_capabilities.items():
            if capability in capabilities:
                matching_agents.append(self.agents[agent_id])
        
        return matching_agents
    
    async def _execute_tasks_parallel(self, tasks: List[tuple]) -> List[Dict[str, Any]]:
        """
        Execute multiple tasks in parallel.
        
        Args:
            tasks: List of (agent, task) tuples
            
        Returns:
            List of task results
        """
        async def execute_single_task(agent: BaseAgent, task: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Create AgentContext from task
                context = AgentContext(
                    research_question=task.get('research_question', ''),
                    analysis_type=task.get('analysis_type', 'general'),
                    metadata=task.get('metadata', {}),
                    user_prompt=task.get('research_question', '')
                )
                
                result = await agent.act(context)
                return {
                    'agent_id': agent.name,
                    'task_id': task['task_id'],
                    'status': 'success' if result.success else 'failed',
                    'result': result.data if result.success else None,
                    'metadata': result.metadata,
                    'error': result.error if not result.success else None
                }
            except Exception as e:
                self.logger.error(f"Task {task['task_id']} failed: {e}")
                return {
                    'agent_id': agent.name,
                    'task_id': task['task_id'],
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Execute tasks in parallel
        tasks_to_execute = [execute_single_task(agent, task) for agent, task in tasks]
        results = await asyncio.gather(*tasks_to_execute, return_exceptions=True)
        
        return results
    
    def _aggregate_data_discovery_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate data discovery results from multiple agents."""
        aggregated = {
            'datasets': [],
            'total_datasets_found': 0,
            'sources_searched': set(),
            'quality_scores': []
        }
        
        for result in results:
            if result['status'] == 'success':
                agent_result = result['result']
                if agent_result:
                    # Extract datasets from the agent result
                    if 'all_candidates' in agent_result:
                        aggregated['datasets'].extend(agent_result['all_candidates'])
                    elif 'datasets' in agent_result:
                        aggregated['datasets'].extend(agent_result['datasets'])
                    
                    # Get metadata from the result
                    metadata = result.get('metadata', {})
                    aggregated['total_datasets_found'] += metadata.get('datasets_found', 0)
                    
                    # Add selected dataset if available
                    if 'dataset_info' in agent_result:
                        selected_dataset = agent_result['dataset_info']
                        if selected_dataset and selected_dataset.get('ref'):
                            aggregated['datasets'].append(selected_dataset)
        
        # Remove duplicates and sort by relevance
        unique_datasets = {}
        for dataset in aggregated['datasets']:
            dataset_id = dataset.get('dataset_id')
            if dataset_id not in unique_datasets or dataset.get('relevance_score', 0) > unique_datasets[dataset_id].get('relevance_score', 0):
                unique_datasets[dataset_id] = dataset
        
        aggregated['datasets'] = sorted(
            unique_datasets.values(),
            key=lambda x: x.get('relevance_score', 0),
            reverse=True
        )
        aggregated['sources_searched'] = list(aggregated['sources_searched'])
        
        return aggregated
    
    def _aggregate_data_acquisition_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate data acquisition results from multiple agents."""
        aggregated = {
            'processed_datasets': {},
            'dataset_paths': {},
            'acquisition_metadata': {},
            'validation_reports': {},
            'total_datasets_acquired': 0
        }
        
        for result in results:
            if result['status'] == 'success':
                agent_result = result['result']
                if agent_result:
                    # Extract dataset information
                    dataset_id = agent_result.get('dataset_id', 'unknown')
                    
                    # Store processed data
                    if 'data' in agent_result:
                        aggregated['processed_datasets'][dataset_id] = agent_result['data']
                    
                    # Store dataset path
                    if 'dataset_path' in agent_result:
                        aggregated['dataset_paths'][dataset_id] = agent_result['dataset_path']
                    
                    # Store metadata
                    if 'metadata' in agent_result:
                        aggregated['acquisition_metadata'][dataset_id] = agent_result['metadata']
                    
                    # Store validation reports
                    metadata = result.get('metadata', {})
                    if 'validation_results' in metadata:
                        aggregated['validation_reports'][dataset_id] = metadata['validation_results']
                    
                    aggregated['total_datasets_acquired'] += 1
        
        return aggregated
    
    def _aggregate_analysis_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate analysis results from multiple agents."""
        aggregated = {
            'dataset_analyses': {},
            'statistical_tests': {},
            'feature_insights': {},
            'data_quality_reports': {}
        }
        
        for result in results:
            if result['status'] == 'success':
                agent_result = result['result']
                dataset_id = agent_result.get('dataset_id')
                
                if dataset_id:
                    aggregated['dataset_analyses'][dataset_id] = agent_result.get('analysis', {})
                    aggregated['statistical_tests'][dataset_id] = agent_result.get('statistical_tests', {})
                    aggregated['feature_insights'][dataset_id] = agent_result.get('feature_insights', {})
                    aggregated['data_quality_reports'][dataset_id] = agent_result.get('quality_report', {})
        
        return aggregated
    
    def _aggregate_modeling_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate modeling results from multiple agents."""
        aggregated = {
            'models': [],
            'best_model': None,
            'model_comparison': {},
            'performance_metrics': {}
        }
        
        best_performance = 0
        best_model = None
        
        for result in results:
            if result['status'] == 'success':
                agent_result = result['result']
                models = agent_result.get('models', [])
                
                for model in models:
                    aggregated['models'].append(model)
                    
                    # Track best model
                    performance = model.get('performance_metrics', {}).get('overall_score', 0)
                    if performance > best_performance:
                        best_performance = performance
                        best_model = model
        
        aggregated['best_model'] = best_model
        
        return aggregated
    
    def _aggregate_synthesis_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate synthesis results from multiple agents."""
        aggregated = {
            'insights': [],
            'recommendations': [],
            'visualizations': [],
            'documentation': {}
        }
        
        for result in results:
            if result['status'] == 'success':
                agent_result = result['result']
                
                aggregated['insights'].extend(agent_result.get('insights', []))
                aggregated['recommendations'].extend(agent_result.get('recommendations', []))
                aggregated['visualizations'].extend(agent_result.get('visualizations', []))
                
                # Merge documentation
                doc = agent_result.get('documentation', {})
                for key, value in doc.items():
                    if key not in aggregated['documentation']:
                        aggregated['documentation'][key] = value
                    elif isinstance(value, list):
                        aggregated['documentation'][key].extend(value)
        
        return aggregated
    
    async def _update_project_status(self, project_id: str, status: ProjectStatus):
        """Update project status."""
        if project_id in self.active_projects:
            project = self.active_projects[project_id]
            project.status = status
            project.updated_at = datetime.utcnow()
            
            self.logger.info(f"Project {project_id} status updated to: {status.value}")
    
    def get_project_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a project."""
        if project_id in self.active_projects:
            project = self.active_projects[project_id]
            return {
                'project_id': project.project_id,
                'status': project.status.value,
                'research_question': project.research_question,
                'created_at': project.created_at.isoformat(),
                'updated_at': project.updated_at.isoformat(),
                'agents_involved': project.agents_involved
            }
        return None
    
    def get_all_projects(self) -> Dict[str, Any]:
        """Get information about all projects."""
        return {
            'active_projects': len(self.active_projects),
            'completed_projects': len(self.project_history),
            'active_project_ids': list(self.active_projects.keys()),
            'recent_projects': [
                {
                    'project_id': p.project_id,
                    'status': p.status.value,
                    'research_question': p.research_question,
                    'created_at': p.created_at.isoformat()
                }
                for p in list(self.project_history)[-5:]  # Last 5 projects
            ]
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents."""
        agent_status = {}
        
        for agent_id, agent in self.agents.items():
            agent_status[agent_id] = {
                'agent_type': agent.__class__.__name__,
                'capabilities': agent.get_capabilities(),
                'is_active': agent.is_active,
                'current_task': agent.current_task.get('task_id') if agent.current_task else None,
                'task_history_count': len(agent.task_history)
            }
        
        return agent_status 