"""
Specialized agents for the Data Science Agent Swarm.

Agents are imported lazily to avoid triggering heavy dependencies
(kaggle, sentence-transformers, torch) at package import time.
"""

__all__ = [
    'DatasetDiscoveryAgent',
    'DataAcquisitionAgent',
    'DataQualityAgent',
    'DocumentationAgent',
    'EDAAgent',
    'FeatureEngineeringAgent',
    'StatisticalAnalysisAgent',
    'ModelArchitectureAgent',
    'HyperparameterOptimizationAgent',
    'ModelValidationAgent',
    'InsightSynthesisAgent',
    'VisualizationAgent',
    'FinalReportGenerator',
]


def __getattr__(name):
    """Lazy import so heavy deps (kaggle, torch) are only loaded when the agent is used."""
    _map = {
        'DatasetDiscoveryAgent':           ('dataset_discovery_agent',                'DatasetDiscoveryAgent'),
        'DataAcquisitionAgent':            ('data_acquisition_agent',                 'DataAcquisitionAgent'),
        'DataQualityAgent':                ('data_quality_agent',                     'DataQualityAgent'),
        'DocumentationAgent':              ('documentation_agent',                    'DocumentationAgent'),
        'EDAAgent':                        ('eda_agent_simple',                       'EDAAgent'),
        'FeatureEngineeringAgent':         ('feature_engineering_agent_simple',       'FeatureEngineeringAgent'),
        'StatisticalAnalysisAgent':        ('statistical_analysis_agent_simple',      'StatisticalAnalysisAgent'),
        'ModelArchitectureAgent':          ('model_architecture_agent_simple',        'ModelArchitectureAgent'),
        'HyperparameterOptimizationAgent': ('hyperparameter_optimization_agent_simple', 'HyperparameterOptimizationAgent'),
        'ModelValidationAgent':            ('model_validation_agent_simple',          'ModelValidationAgent'),
        'InsightSynthesisAgent':           ('insight_synthesis_agent_simple',         'InsightSynthesisAgent'),
        'VisualizationAgent':              ('visualization_agent_simple',             'VisualizationAgent'),
        'FinalReportGenerator':            ('final_report_generator',                 'FinalReportGenerator'),
    }
    if name in _map:
        module_name, class_name = _map[name]
        import importlib
        mod = importlib.import_module(f'.{module_name}', package=__name__)
        return getattr(mod, class_name)
    raise AttributeError(f"module 'src.agents' has no attribute {name!r}")
