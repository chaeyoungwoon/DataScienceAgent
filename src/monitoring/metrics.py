"""
Metrics Collection for Data Science Agent Swarm

This module provides comprehensive metrics collection and monitoring
capabilities for the agent swarm system, including performance tracking,
error monitoring, and system health metrics.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for when prometheus is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self


class MetricsCollector:
    """
    Collect and export metrics for monitoring the agent swarm system.
    
    This class provides comprehensive metrics collection including:
    - Agent task execution metrics
    - System performance metrics
    - Error tracking and alerting
    - Resource utilization monitoring
    """
    
    def __init__(self, port: int = 9090, enable_prometheus: bool = True):
        """
        Initialize the metrics collector.
        
        Args:
            port: Port for Prometheus metrics server
            enable_prometheus: Whether to enable Prometheus metrics
        """
        self.logger = logging.getLogger("metrics_collector")
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        
        # Initialize metrics
        self.initialize_metrics()
        
        # Start Prometheus server if enabled
        if self.enable_prometheus:
            try:
                start_http_server(port)
                self.logger.info(f"Prometheus metrics server started on port {port}")
            except Exception as e:
                self.logger.error(f"Failed to start Prometheus server: {e}")
                self.enable_prometheus = False
        
        # Internal metrics storage for when Prometheus is not available
        self.internal_metrics = {
            'task_executions': {},
            'errors': {},
            'performance': {},
            'system_health': {}
        }
    
    def initialize_metrics(self):
        """Initialize all metrics counters and gauges."""
        if self.enable_prometheus:
            # Task execution metrics
            self.AGENT_TASKS_TOTAL = Counter(
                'agent_tasks_total', 
                'Total tasks executed by agents', 
                ['agent_type', 'task_type', 'status']
            )
            
            self.AGENT_TASK_DURATION = Histogram(
                'agent_task_duration_seconds', 
                'Time spent on agent tasks', 
                ['agent_type', 'task_type']
            )
            
            # System health metrics
            self.ACTIVE_AGENTS = Gauge(
                'active_agents_total', 
                'Number of currently active agents', 
                ['agent_type']
            )
            
            self.PROJECT_COMPLETION_TIME = Histogram(
                'project_completion_time_seconds', 
                'Total time to complete research projects'
            )
            
            self.DATA_QUALITY_SCORE = Gauge(
                'data_quality_score', 
                'Average data quality score of discovered datasets'
            )
            
            # Error tracking
            self.ERROR_COUNTER = Counter(
                'agent_errors_total',
                'Total errors encountered by agents',
                ['agent_type', 'error_type']
            )
            
            # Performance metrics
            self.MEMORY_USAGE = Gauge(
                'agent_memory_usage_bytes',
                'Memory usage by agent type',
                ['agent_type']
            )
            
            self.CPU_USAGE = Gauge(
                'agent_cpu_usage_percent',
                'CPU usage by agent type',
                ['agent_type']
            )
        else:
            # Create dummy metrics for when Prometheus is not available
            self.AGENT_TASKS_TOTAL = Counter()
            self.AGENT_TASK_DURATION = Histogram()
            self.ACTIVE_AGENTS = Gauge()
            self.PROJECT_COMPLETION_TIME = Histogram()
            self.DATA_QUALITY_SCORE = Gauge()
            self.ERROR_COUNTER = Counter()
            self.MEMORY_USAGE = Gauge()
            self.CPU_USAGE = Gauge()
    
    def record_task_completion(self, agent_type: str, task_type: str, duration: float, success: bool):
        """
        Record task completion metrics.
        
        Args:
            agent_type: Type of agent that executed the task
            task_type: Type of task that was executed
            duration: Duration of task execution in seconds
            success: Whether the task completed successfully
        """
        status = 'success' if success else 'failure'
        
        if self.enable_prometheus:
            self.AGENT_TASKS_TOTAL.labels(
                agent_type=agent_type, 
                task_type=task_type, 
                status=status
            ).inc()
            
            self.AGENT_TASK_DURATION.labels(
                agent_type=agent_type, 
                task_type=task_type
            ).observe(duration)
        else:
            # Store in internal metrics
            key = f"{agent_type}_{task_type}"
            if key not in self.internal_metrics['task_executions']:
                self.internal_metrics['task_executions'][key] = {
                    'total': 0,
                    'success': 0,
                    'failure': 0,
                    'total_duration': 0.0,
                    'avg_duration': 0.0
                }
            
            self.internal_metrics['task_executions'][key]['total'] += 1
            self.internal_metrics['task_executions'][key][status] += 1
            self.internal_metrics['task_executions'][key]['total_duration'] += duration
            self.internal_metrics['task_executions'][key]['avg_duration'] = (
                self.internal_metrics['task_executions'][key]['total_duration'] / 
                self.internal_metrics['task_executions'][key]['total']
            )
        
        self.logger.info(f"Task completed: {agent_type} - {task_type} - {status} - {duration:.2f}s")
    
    def update_active_agents(self, agent_type: str, count: int):
        """
        Update active agent count.
        
        Args:
            agent_type: Type of agent
            count: Number of active agents
        """
        if self.enable_prometheus:
            self.ACTIVE_AGENTS.labels(agent_type=agent_type).set(count)
        else:
            self.internal_metrics['system_health'][f'active_{agent_type}_agents'] = count
        
        self.logger.debug(f"Active {agent_type} agents: {count}")
    
    def record_project_completion(self, duration: float):
        """
        Record project completion time.
        
        Args:
            duration: Total time to complete the project in seconds
        """
        if self.enable_prometheus:
            self.PROJECT_COMPLETION_TIME.observe(duration)
        else:
            if 'project_completion_times' not in self.internal_metrics['performance']:
                self.internal_metrics['performance']['project_completion_times'] = []
            self.internal_metrics['performance']['project_completion_times'].append(duration)
        
        self.logger.info(f"Project completed in {duration:.2f} seconds")
    
    def update_data_quality(self, score: float):
        """
        Update average data quality score.
        
        Args:
            score: Data quality score (0-1)
        """
        if self.enable_prometheus:
            self.DATA_QUALITY_SCORE.set(score)
        else:
            self.internal_metrics['performance']['data_quality_score'] = score
        
        self.logger.debug(f"Data quality score updated: {score:.3f}")
    
    def record_error(self, agent_type: str, error_type: str, error_message: str = ""):
        """
        Record an error occurrence.
        
        Args:
            agent_type: Type of agent that encountered the error
            error_type: Type of error
            error_message: Optional error message
        """
        if self.enable_prometheus:
            self.ERROR_COUNTER.labels(agent_type=agent_type, error_type=error_type).inc()
        else:
            key = f"{agent_type}_{error_type}"
            if key not in self.internal_metrics['errors']:
                self.internal_metrics['errors'][key] = 0
            self.internal_metrics['errors'][key] += 1
        
        self.logger.error(f"Error recorded: {agent_type} - {error_type} - {error_message}")
    
    def update_memory_usage(self, agent_type: str, memory_bytes: int):
        """
        Update memory usage for an agent type.
        
        Args:
            agent_type: Type of agent
            memory_bytes: Memory usage in bytes
        """
        if self.enable_prometheus:
            self.MEMORY_USAGE.labels(agent_type=agent_type).set(memory_bytes)
        else:
            self.internal_metrics['system_health'][f'{agent_type}_memory_bytes'] = memory_bytes
        
        self.logger.debug(f"Memory usage for {agent_type}: {memory_bytes} bytes")
    
    def update_cpu_usage(self, agent_type: str, cpu_percent: float):
        """
        Update CPU usage for an agent type.
        
        Args:
            agent_type: Type of agent
            cpu_percent: CPU usage percentage
        """
        if self.enable_prometheus:
            self.CPU_USAGE.labels(agent_type=agent_type).set(cpu_percent)
        else:
            self.internal_metrics['system_health'][f'{agent_type}_cpu_percent'] = cpu_percent
        
        self.logger.debug(f"CPU usage for {agent_type}: {cpu_percent}%")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all collected metrics.
        
        Returns:
            Dictionary containing metrics summary
        """
        if self.enable_prometheus:
            # For Prometheus, we can't easily get current values
            # This would require querying the Prometheus API
            return {
                'prometheus_enabled': True,
                'note': 'Metrics available via Prometheus endpoint'
            }
        else:
            return {
                'prometheus_enabled': False,
                'task_executions': self.internal_metrics['task_executions'],
                'errors': self.internal_metrics['errors'],
                'performance': self.internal_metrics['performance'],
                'system_health': self.internal_metrics['system_health']
            }
    
    def get_task_execution_stats(self, agent_type: str = None, task_type: str = None) -> Dict[str, Any]:
        """
        Get task execution statistics.
        
        Args:
            agent_type: Optional filter by agent type
            task_type: Optional filter by task type
            
        Returns:
            Dictionary containing task execution statistics
        """
        if self.enable_prometheus:
            return {
                'note': 'Detailed metrics available via Prometheus queries'
            }
        else:
            stats = {}
            for key, data in self.internal_metrics['task_executions'].items():
                if agent_type and agent_type not in key:
                    continue
                if task_type and task_type not in key:
                    continue
                
                stats[key] = {
                    'total_executions': data['total'],
                    'success_rate': data['success'] / data['total'] if data['total'] > 0 else 0,
                    'average_duration': data['avg_duration'],
                    'total_duration': data['total_duration']
                }
            
            return stats
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get error summary statistics.
        
        Returns:
            Dictionary containing error summary
        """
        if self.enable_prometheus:
            return {
                'note': 'Error metrics available via Prometheus queries'
            }
        else:
            return {
                'total_errors': sum(self.internal_metrics['errors'].values()),
                'error_breakdown': self.internal_metrics['errors']
            }
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """
        Get system health summary.
        
        Returns:
            Dictionary containing system health metrics
        """
        if self.enable_prometheus:
            return {
                'note': 'System health metrics available via Prometheus queries'
            }
        else:
            return {
                'active_agents': {
                    k: v for k, v in self.internal_metrics['system_health'].items() 
                    if 'active_' in k
                },
                'resource_usage': {
                    k: v for k, v in self.internal_metrics['system_health'].items() 
                    if 'memory' in k or 'cpu' in k
                }
            }
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        if not self.enable_prometheus:
            self.internal_metrics = {
                'task_executions': {},
                'errors': {},
                'performance': {},
                'system_health': {}
            }
            self.logger.info("Metrics reset")
    
    def export_metrics(self, format: str = 'json') -> str:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format ('json', 'csv', 'prometheus')
            
        Returns:
            String representation of metrics in specified format
        """
        if format == 'json':
            import json
            return json.dumps(self.get_metrics_summary(), indent=2)
        elif format == 'csv':
            # Simple CSV export
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            summary = self.get_metrics_summary()
            if 'task_executions' in summary:
                writer.writerow(['Metric', 'Value'])
                for key, data in summary['task_executions'].items():
                    writer.writerow([f'task_{key}_total', data['total']])
                    writer.writerow([f'task_{key}_success_rate', data['success'] / data['total'] if data['total'] > 0 else 0])
            
            return output.getvalue()
        elif format == 'prometheus':
            if self.enable_prometheus:
                return "# Prometheus metrics available via /metrics endpoint"
            else:
                return "# Prometheus not enabled"
        else:
            raise ValueError(f"Unsupported format: {format}")


# Decorator for monitoring function execution
def monitor_execution(func):
    """
    Decorator to monitor function execution with metrics collection.
    
    This decorator automatically records execution time, success/failure,
    and other relevant metrics for the decorated function.
    """
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        agent_type = self.__class__.__name__
        task_type = kwargs.get('task', {}).get('task_type', 'unknown')
        
        try:
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time
            
            # Record successful execution
            if hasattr(self, 'metrics_collector') and self.metrics_collector:
                self.metrics_collector.record_task_completion(agent_type, task_type, duration, True)
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            
            # Record failed execution
            if hasattr(self, 'metrics_collector') and self.metrics_collector:
                self.metrics_collector.record_task_completion(agent_type, task_type, duration, False)
                self.metrics_collector.record_error(agent_type, type(e).__name__, str(e))
            
            raise e
    
    return wrapper 