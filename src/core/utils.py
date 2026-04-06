"""
Shared utilities for the AI Research Pipeline.
"""

import math
import numpy as np
import pandas as pd
from typing import Any


def safe_json_convert(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas types to JSON-safe Python types.
    Handles: numpy integers/floats/arrays, pandas Series/DataFrames,
    tuples, dicts, lists, and float NaN/Inf values.
    """
    if isinstance(obj, dict):
        return {key: safe_json_convert(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_convert(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    elif isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    elif isinstance(obj, np.ndarray):
        return [safe_json_convert(x) for x in obj.tolist()]
    elif isinstance(obj, (pd.Series, pd.Index)):
        return [safe_json_convert(x) for x in obj.tolist()]
    elif isinstance(obj, pd.DataFrame):
        return [safe_json_convert(row) for row in obj.to_dict('records')]
    elif isinstance(obj, np.dtype):
        return str(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def find_processed_file(file_path: str, processed_file_paths: list) -> str | None:
    """
    Find a processed file path that corresponds to the given file_path key.
    Handles both exact matches and stem-based matching.
    """
    from pathlib import Path

    # Exact match first
    if file_path in processed_file_paths:
        return file_path

    # Stem-based fuzzy match
    key_stem = Path(file_path).stem.replace('_processed', '').replace('_cleaned', '')
    for p in processed_file_paths:
        candidate_stem = Path(p).stem.replace('_processed', '').replace('_cleaned', '')
        if key_stem == candidate_stem or key_stem in Path(p).name or Path(p).name in file_path:
            return p

    return None
