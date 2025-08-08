"""
Task Scheduler for Data Science Agent Swarm

This module provides task scheduling and management capabilities for coordinating
agent activities and ensuring optimal resource utilization.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
import uuid
from enum import Enum
from dataclasses import dataclass


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task status states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task data structure."""
    id: str
    agent_id: str
    task_type: str
    priority: TaskPriority
    data: Dict[str, Any]
    created_at: datetime
    scheduled_for: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskScheduler:
    """
    Task scheduler for managing agent tasks and resource allocation.
    
    This class provides:
    - Task queuing and prioritization
    - Resource management and load balancing
    - Task execution monitoring
    - Failure handling and retry logic
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the task scheduler.
        
        Args:
            config: Configuration dictionary containing scheduler settings
        """
        self.logger = logging.getLogger("task_scheduler")
        self.config = config
        
        # Task queues by priority
        self.task_queues = {
            TaskPriority.LOW: asyncio.Queue(),
            TaskPriority.NORMAL: asyncio.Queue(),
            TaskPriority.HIGH: asyncio.Queue(),
            TaskPriority.CRITICAL: asyncio.Queue()
        }
        
        # Active tasks
        self.active_tasks = {}
        
        # Task history
        self.task_history = []
        
        # Resource limits
        self.max_concurrent_tasks = config.get('max_concurrent_tasks', 10)
        self.max_tasks_per_agent = config.get('max_tasks_per_agent', 3)
        
        # Agent task counts
        self.agent_task_counts = {}
        
        # Scheduler state
        self.is_running = False
        self.scheduler_task = None
        
        self.logger.info("Task scheduler initialized")
    
    async def start(self):
        """Start the task scheduler."""
        self.is_running = True
        self.scheduler_task = asyncio.create_task(self.scheduler_loop())
        self.logger.info("Task scheduler started")
    
    async def stop(self):
        """Stop the task scheduler."""
        self.is_running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Task scheduler stopped")
    
    async def submit_task(self, agent_id: str, task_type: str, data: Dict[str, Any], 
                         priority: TaskPriority = TaskPriority.NORMAL,
                         scheduled_for: Optional[datetime] = None) -> str:
        """
        Submit a task for execution.
        
        Args:
            agent_id: ID of the agent to execute the task
            task_type: Type of task to execute
            data: Task data
            priority: Task priority
            scheduled_for: When to schedule the task (None for immediate)
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            agent_id=agent_id,
            task_type=task_type,
            priority=priority,
            data=data,
            created_at=datetime.utcnow(),
            scheduled_for=scheduled_for
        )
        
        # Add to appropriate queue
        if scheduled_for and scheduled_for > datetime.utcnow():
            # Schedule for later
            await self.schedule_delayed_task(task)
        else:
            # Add to immediate queue
            await self.task_queues[priority].put(task)
        
        self.logger.info(f"Task {task_id} submitted for agent {agent_id}")
        return task_id
    
    async def schedule_delayed_task(self, task: Task):
        """Schedule a task for later execution."""
        delay = (task.scheduled_for - datetime.utcnow()).total_seconds()
        
        async def delayed_task():
            await asyncio.sleep(delay)
            await self.task_queues[task.priority].put(task)
        
        asyncio.create_task(delayed_task())
    
    async def scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                # Check if we can start new tasks
                if len(self.active_tasks) < self.max_concurrent_tasks:
                    # Try to get a task from highest priority queue
                    task = await self.get_next_task()
                    if task:
                        await self.execute_task(task)
                
                # Clean up completed tasks
                await self.cleanup_completed_tasks()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1)
    
    async def get_next_task(self) -> Optional[Task]:
        """Get the next task from the highest priority queue."""
        # Check queues in priority order (highest first)
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                        TaskPriority.NORMAL, TaskPriority.LOW]:
            if not self.task_queues[priority].empty():
                task = self.task_queues[priority].get_nowait()
                
                # Check if agent can handle more tasks
                if self.can_agent_accept_task(task.agent_id):
                    return task
                else:
                    # Put task back in queue
                    await self.task_queues[priority].put(task)
        
        return None
    
    def can_agent_accept_task(self, agent_id: str) -> bool:
        """Check if an agent can accept a new task."""
        current_tasks = self.agent_task_counts.get(agent_id, 0)
        return current_tasks < self.max_tasks_per_agent
    
    async def execute_task(self, task: Task):
        """Execute a task."""
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            
            # Add to active tasks
            self.active_tasks[task.id] = task
            
            # Update agent task count
            self.agent_task_counts[task.agent_id] = \
                self.agent_task_counts.get(task.agent_id, 0) + 1
            
            self.logger.info(f"Starting task {task.id} for agent {task.agent_id}")
            
            # Execute task (this would call the agent's execute_task method)
            # For now, we'll simulate task execution
            await self.simulate_task_execution(task)
            
        except Exception as e:
            self.logger.error(f"Error executing task {task.id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
    
    async def simulate_task_execution(self, task: Task):
        """Simulate task execution (placeholder)."""
        # Simulate some processing time
        await asyncio.sleep(2)
        
        # Simulate task completion
        task.status = TaskStatus.COMPLETED
        task.result = {
            'task_id': task.id,
            'agent_id': task.agent_id,
            'task_type': task.task_type,
            'status': 'completed',
            'timestamp': datetime.utcnow().isoformat()
        }
        task.completed_at = datetime.utcnow()
        
        self.logger.info(f"Task {task.id} completed successfully")
    
    async def cleanup_completed_tasks(self):
        """Clean up completed tasks from active tasks."""
        completed_tasks = []
        
        for task_id, task in self.active_tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                completed_tasks.append(task_id)
                
                # Update agent task count
                self.agent_task_counts[task.agent_id] = \
                    max(0, self.agent_task_counts.get(task.agent_id, 1) - 1)
                
                # Add to history
                self.task_history.append(task)
        
        # Remove completed tasks from active tasks
        for task_id in completed_tasks:
            del self.active_tasks[task_id]
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if task was cancelled successfully
        """
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow()
            self.logger.info(f"Task {task_id} cancelled")
            return True
        
        # Check queues for pending tasks
        for priority, queue in self.task_queues.items():
            # This is a simplified implementation
            # In a real implementation, you'd need to search through the queue
            pass
        
        return False
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task status information or None if not found
        """
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return self.task_to_dict(task)
        
        # Check history
        for task in self.task_history:
            if task.id == task_id:
                return self.task_to_dict(task)
        
        return None
    
    def task_to_dict(self, task: Task) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'id': task.id,
            'agent_id': task.agent_id,
            'task_type': task.task_type,
            'priority': task.priority.value,
            'status': task.status.value,
            'created_at': task.created_at.isoformat(),
            'scheduled_for': task.scheduled_for.isoformat() if task.scheduled_for else None,
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'result': task.result,
            'error': task.error
        }
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status information."""
        queue_sizes = {
            priority.name: queue.qsize() 
            for priority, queue in self.task_queues.items()
        }
        
        return {
            'is_running': self.is_running,
            'active_tasks': len(self.active_tasks),
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'queue_sizes': queue_sizes,
            'agent_task_counts': self.agent_task_counts,
            'total_completed_tasks': len(self.task_history)
        }
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get task execution statistics."""
        if not self.task_history:
            return {
                'total_tasks': 0,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'average_execution_time': 0
            }
        
        completed_tasks = [t for t in self.task_history if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in self.task_history if t.status == TaskStatus.FAILED]
        
        # Calculate average execution time
        execution_times = []
        for task in completed_tasks:
            if task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                execution_times.append(duration)
        
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            'total_tasks': len(self.task_history),
            'completed_tasks': len(completed_tasks),
            'failed_tasks': len(failed_tasks),
            'success_rate': len(completed_tasks) / len(self.task_history),
            'average_execution_time': avg_execution_time
        }
    
    async def get_agent_tasks(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get all tasks for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of task information
        """
        tasks = []
        
        # Check active tasks
        for task in self.active_tasks.values():
            if task.agent_id == agent_id:
                tasks.append(self.task_to_dict(task))
        
        # Check history
        for task in self.task_history:
            if task.agent_id == agent_id:
                tasks.append(self.task_to_dict(task))
        
        return tasks
    
    async def clear_task_history(self):
        """Clear task history (use with caution)."""
        self.task_history.clear()
        self.logger.warning("Task history cleared")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get detailed queue status."""
        return {
            'queue_sizes': {
                priority.name: queue.qsize() 
                for priority, queue in self.task_queues.items()
            },
            'total_queued_tasks': sum(queue.qsize() for queue in self.task_queues.values())
        } 