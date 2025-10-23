"""
Helper utility functions for the YOLO benchmarking system.
"""

import time
from pathlib import Path
from typing import Any, Dict, Union
import hashlib
import json


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def format_size(bytes_size: Union[int, float]) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        bytes_size: Size in bytes
    
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def create_experiment_name(params: Dict[str, Any]) -> str:
    """
    Create a unique experiment name from parameters.
    
    Args:
        params: Dictionary of parameters
    
    Returns:
        Unique experiment name
    """
    if not params:
        return "baseline"
    
    # Create sorted parameter string
    param_parts = []
    for key, value in sorted(params.items()):
        if isinstance(value, float):
            param_parts.append(f"{key}_{value:.4f}")
        else:
            param_parts.append(f"{key}_{value}")
    
    # Create hash for uniqueness
    param_str = "_".join(param_parts)
    hash_obj = hashlib.md5(param_str.encode())
    short_hash = hash_obj.hexdigest()[:8]
    
    return f"exp_{short_hash}"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe filesystem usage.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Ensure not empty
    if not filename:
        filename = "unnamed"
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename


def save_json_safe(obj: Any, filepath: Path, indent: int = 2) -> None:
    """
    Save object to JSON file safely.
    
    Args:
        obj: Object to save
        filepath: Output file path
        indent: JSON indentation
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def json_serializer(obj):
        """JSON serializer for objects not serializable by default."""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        else:
            return str(obj)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent, default=json_serializer)


def load_json_safe(filepath: Path) -> Any:
    """
    Load JSON file safely.
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded object
    """
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise ValueError(f"Error loading JSON file {filepath}: {e}")


def get_file_hash(filepath: Path) -> str:
    """
    Calculate MD5 hash of file contents.
    
    Args:
        filepath: File path
    
    Returns:
        MD5 hash as hex string
    """
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except IOError:
        return ""
    return hash_md5.hexdigest()


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object for directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_directory_size(path: Union[str, Path]) -> int:
    """
    Get total size of directory in bytes.
    
    Args:
        path: Directory path
    
    Returns:
        Total size in bytes
    """
    path = Path(path)
    if not path.exists():
        return 0
    
    total_size = 0
    for item in path.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size
    
    return total_size


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
