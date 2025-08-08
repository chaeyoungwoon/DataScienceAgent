"""
Base Agent Class for Data Science Agent Swarm

Provides a foundation for all specialized agents with coordination,
reasoning, and communication capabilities.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

@dataclass
class AgentContext:
    """Context passed between agents during execution."""
    research_question: str
    domain: str = "data_science"
    analysis_type: str = "exploratory"
    user_prompt: Optional[str] = None
    dataset_path: Optional[str] = None
    data_path: Optional[str] = None
    cleaned_data: Optional[Any] = None
    processed_data: Optional[Any] = None
    eda_results: Optional[Dict[str, Any]] = None
    model_results: Optional[Dict[str, Any]] = None
    insights: Optional[List[str]] = None
    charts_dir: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AgentResult:
    """Standardized result from agent execution."""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    error: Optional[str] = None
    execution_time: Optional[float] = None

class BaseAgent(ABC):
    """
    Base class for all agents in the Data Science Agent Swarm.
    
    Features:
    - Standardized communication protocol
    - Context passing between agents
    - Error handling and logging
    - Performance monitoring
    - LLM reasoning capabilities
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize base agent."""
        self.name = name
        self.agent_id = name  # Add agent_id as a regular attribute
        self.config = config
        self.logger = logging.getLogger(f"agent.{name}")
        self.execution_history = []
        self.performance_metrics = {}
        
        # Agent status attributes
        self.is_active = True
        self.current_task = None
        self.task_history = []
        
        # Initialize agent-specific components
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize agent-specific components."""
        # Create output directory for this agent
        self.output_dir = Path(f"output/{self.name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize performance tracking
        self.start_time = None
        self.end_time = None
        
        self.logger.info(f"Agent {self.name} initialized")
    
    @abstractmethod
    async def act(self, context: AgentContext) -> AgentResult:
        """
        Execute the agent's primary action.
        
        Args:
            context: Current execution context with data from previous agents
            
        Returns:
            AgentResult: Standardized result with data and metadata
        """
        pass
    
    async def execute_with_reasoning(self, context: AgentContext) -> AgentResult:
        """
        Execute agent with LLM reasoning for decision making.
        
        Args:
            context: Current execution context
            
        Returns:
            AgentResult: Result with reasoning included
        """
        try:
            # Pre-execution reasoning
            reasoning = await self._reason_about_task(context)
            
            # Execute main action
            result = await self.act(context)
            
            # Add reasoning to metadata
            result.metadata['reasoning'] = reasoning
            result.metadata['agent_name'] = self.name
            result.metadata['execution_timestamp'] = datetime.utcnow().isoformat()
            
            # Log performance
            self._log_performance(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Agent {self.name} failed: {e}")
            return AgentResult(
                success=False,
                data=None,
                metadata={'agent_name': self.name, 'error': str(e)},
                error=str(e)
            )
    
    async def _reason_about_task(self, context: AgentContext) -> str:
        """
        Use LLM to reason about the current task.
        
        Args:
            context: Current execution context
            
        Returns:
            str: Reasoning about the current task
        """
        # This can be overridden by specific agents
        return f"Agent {self.name} executing task for {context.analysis_type} analysis"
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task using the agent's act method.
        
        Args:
            task: Task dictionary with parameters
            
        Returns:
            Dict: Task execution results
        """
        try:
            # Create context from task
            context = AgentContext(
                research_question=task.get('research_question', ''),
                domain=task.get('domain', 'data_science'),
                analysis_type=task.get('task_type', 'exploratory'),
                user_prompt=task.get('research_question'),
                metadata=task
            )
            
            # Execute the agent's act method
            result = await self.act(context)
            
            # Convert AgentResult to dictionary format expected by swarm manager
            return {
                'success': result.success,
                'data': result.data,
                'metadata': result.metadata,
                'error': result.error,
                'execution_time': result.execution_time
            }
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return {
                'success': False,
                'data': None,
                'metadata': {'error': str(e)},
                'error': str(e),
                'execution_time': None
            }
    
    def _log_performance(self, result: AgentResult):
        """Log performance metrics for the agent."""
        if result.execution_time:
            self.performance_metrics['last_execution_time'] = result.execution_time
            self.performance_metrics['total_executions'] = len(self.execution_history) + 1
        
        self.execution_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'success': result.success,
            'execution_time': result.execution_time,
            'metadata': result.metadata
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this agent."""
        if not self.execution_history:
            return {'status': 'No executions yet'}
        
        successful_executions = [e for e in self.execution_history if e['success']]
        avg_execution_time = sum(e.get('execution_time', 0) for e in self.execution_history) / len(self.execution_history)
        
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': len(successful_executions),
            'success_rate': len(successful_executions) / len(self.execution_history),
            'average_execution_time': avg_execution_time,
            'last_execution': self.execution_history[-1] if self.execution_history else None
        }
    
    def save_output(self, data: Any, filename: str) -> str:
        """
        Save agent output to file.
        
        Args:
            data: Data to save
            filename: Output filename
            
        Returns:
            str: Path to saved file
        """
        output_path = self.output_dir / filename
        
        if isinstance(data, (dict, list)):
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif hasattr(data, 'to_csv'):
            data.to_csv(output_path, index=False)
        else:
            with open(output_path, 'w') as f:
                f.write(str(data))
        
        self.logger.info(f"Saved output to {output_path}")
        return str(output_path)
    
    def update_context(self, context: AgentContext, key: str, value: Any) -> AgentContext:
        """
        Update the execution context with new data.
        
        Args:
            context: Current context
            key: Key to update
            value: New value
            
        Returns:
            AgentContext: Updated context
        """
        if context.metadata is None:
            context.metadata = {}
        
        context.metadata[key] = value
        return context
    
    async def coordinate_with_other_agents(self, context: AgentContext, agent_results: Dict[str, AgentResult]) -> AgentContext:
        """
        Coordinate with results from other agents.
        
        Args:
            context: Current execution context
            agent_results: Results from other agents
            
        Returns:
            AgentContext: Updated context with coordination data
        """
        # This can be overridden by specific agents for complex coordination
        for agent_name, result in agent_results.items():
            if result.success:
                context = self.update_context(context, f"{agent_name}_result", result.data)
        
        return context
    
    def validate_input(self, context: AgentContext) -> bool:
        """
        Validate input context for this agent.
        
        Args:
            context: Execution context to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Base validation - can be overridden by specific agents
        required_fields = self.get_required_context_fields()
        
        for field in required_fields:
            if not hasattr(context, field) or getattr(context, field) is None:
                self.logger.warning(f"Missing required field: {field}")
                return False
        
        return True
    
    def get_required_context_fields(self) -> List[str]:
        """Get list of required context fields for this agent."""
        return []  # Override in specific agents
    
    def cleanup(self):
        """Clean up agent resources."""
        self.logger.info(f"Agent {self.name} cleanup completed")
    
    def __str__(self):
        return f"Agent({self.name})"
    
    def __repr__(self):
        return f"Agent(name='{self.name}', config={self.config})" 