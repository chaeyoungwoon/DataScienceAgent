"""
Shared utilities for the AI Research Pipeline.
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Optional


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


def safe_read_csv(file_path, **kwargs) -> Optional[pd.DataFrame]:
    """
    Read a CSV file with automatic encoding detection.
    Tries UTF-8, then latin-1, then cp1252 before failing.
    Also handles tab-separated files and auto-detection.
    """
    path = Path(file_path)
    sep = kwargs.pop('sep', ',')
    encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-8-sig']

    # For TSV files
    if path.suffix.lower() in ('.tsv', '.txt') and sep == ',':
        sep = '\t'

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, sep=sep, **kwargs)
        except UnicodeDecodeError:
            continue
        except Exception:
            break

    # Last resort: auto-detect separator
    try:
        return pd.read_csv(path, encoding='latin-1', sep=None, engine='python', **kwargs)
    except Exception:
        return None


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
