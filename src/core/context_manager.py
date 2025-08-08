"""
Standardized Context Manager for AI Research Pipeline

Implements the standardized context boilerplate specified in the master specification.
All agents use this module for reading, writing, and logging context.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

CONTEXT_PATH = "context/context_output.json"

def read_context() -> Dict[str, Any]:
    """
    Read the current context from context_output.json.
    
    Returns:
        Dict containing the current context or default structure if file doesn't exist
    """
    if os.path.exists(CONTEXT_PATH):
        try:
            with open(CONTEXT_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading context file: {e}")
    
    # Return default context structure
    return {
        "project_metadata": {
            "research_question": "",
            "created_at": datetime.now().isoformat(),
            "dataset_refs": []
        },
        "context_chain": {
            "dataset_discovery": {},
            "data_acquisition": {},
            "data_quality": {},
            "documentation": {},
            "eda": {},
            "feature_engineering": {},
            "statistical_analysis": {},
            "model_architecture": {},
            "hyperparameter_optimization": {},
            "model_validation": {},
            "insight_synthesis": {},
            "visualization": {},
            "final_report": {}
        },
        "pipeline_log": []
    }

def write_context(context: Dict[str, Any]) -> None:
    """
    Write context to context_output.json.
    
    Args:
        context: The context dictionary to write
    """
    # Ensure context directory exists
    context_dir = Path(CONTEXT_PATH).parent
    context_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(CONTEXT_PATH, "w") as f:
            json.dump(context, f, indent=2)
    except IOError as e:
        print(f"Error writing context file: {e}")

def log_step(context: Dict[str, Any], agent_name: str, message: str) -> None:
    """
    Log a pipeline step to the context.
    
    Args:
        context: The current context dictionary
        agent_name: Name of the agent that completed the step
        message: Description of what was accomplished
    """
    context["pipeline_log"].append({
        "agent": agent_name,
        "timestamp": datetime.now().isoformat(),
        "message": message
    })

def update_context_chain(context: Dict[str, Any], agent_name: str, data: Dict[str, Any]) -> None:
    """
    Update a specific section in the context_chain.
    
    Args:
        context: The current context dictionary
        agent_name: Name of the agent (must match a key in context_chain)
        data: Data to store in the agent's section
    """
    if agent_name in context["context_chain"]:
        context["context_chain"][agent_name] = data
    else:
        print(f"Warning: Unknown agent '{agent_name}' in context_chain")

def get_context_chain_data(context: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """
    Get data from a specific section in the context_chain.
    
    Args:
        context: The current context dictionary
        agent_name: Name of the agent
        
    Returns:
        Data from the agent's section or empty dict if not found
    """
    return context["context_chain"].get(agent_name, {})

def set_research_question(context: Dict[str, Any], question: str) -> None:
    """
    Set the research question in project_metadata.
    
    Args:
        context: The current context dictionary
        question: The research question
    """
    context["project_metadata"]["research_question"] = question
    if not context["project_metadata"]["created_at"]:
        context["project_metadata"]["created_at"] = datetime.now().isoformat()

def get_research_question(context: Dict[str, Any]) -> str:
    """
    Get the research question from project_metadata.
    
    Args:
        context: The current context dictionary
        
    Returns:
        The research question string
    """
    return context["project_metadata"].get("research_question", "")

def add_dataset_ref(context: Dict[str, Any], dataset_ref: str) -> None:
    """
    Add a dataset reference to project_metadata.
    
    Args:
        context: The current context dictionary
        dataset_ref: Dataset reference to add
    """
    if dataset_ref not in context["project_metadata"]["dataset_refs"]:
        context["project_metadata"]["dataset_refs"].append(dataset_ref)
