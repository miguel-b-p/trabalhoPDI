"""
Utility functions for the YOLO Benchmark System.

Provides helper functions for device detection, file operations,
and common transformations.
"""

import torch
import platform
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import json


def detect_device() -> str:
    """
    Automatically detect the best available device for training.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed device information.
    
    Returns:
        Dictionary with device details
    """
    info = {
        'device': detect_device(),
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
    }
    
    if info['device'] == 'cuda':
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
        info['cuda_version'] = torch.version.cuda
        
        # Get memory info
        for i in range(torch.cuda.device_count()):
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            info[f'gpu_{i}_memory_gb'] = round(mem_total, 2)
    
    return info


def estimate_batch_size(
    device: str,
    imgsz: int,
    model_size: str = 'm'
) -> int:
    """
    Estimate optimal batch size based on available memory.
    
    Args:
        device: Device type ('cuda', 'mps', 'cpu')
        imgsz: Input image size
        model_size: Model variant ('n', 's', 'm', 'l', 'x')
    
    Returns:
        Recommended batch size
    """
    if device == 'cpu':
        return 4  # Conservative for CPU
    
    # Base memory requirements (GB) per image for different models
    memory_per_image = {
        'n': 0.05,
        's': 0.08,
        'm': 0.12,
        'l': 0.18,
        'x': 0.25
    }
    
    base_mem = memory_per_image.get(model_size, 0.12)
    
    # Scale by image size (baseline: 640)
    size_factor = (imgsz / 640) ** 2
    mem_per_img = base_mem * size_factor
    
    if device == 'cuda':
        # Get available GPU memory
        available_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        # Reserve 2GB for system
        usable_mem = available_mem - 2.0
        
        batch_size = int(usable_mem / mem_per_img)
        # Clamp to reasonable values
        return max(4, min(batch_size, 64))
    
    elif device == 'mps':
        # MPS has shared memory with system
        import psutil
        total_mem = psutil.virtual_memory().total / (1024 ** 3)
        usable_mem = total_mem * 0.3  # Use 30% of system memory
        
        batch_size = int(usable_mem / mem_per_img)
        return max(4, min(batch_size, 32))
    
    return 16  # Default


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        path: Path to YAML file
    
    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml_config(config: Dict[str, Any], path: Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        path: Path to save YAML file
    """
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        path: Path to JSON file
    
    Returns:
        Data dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data dictionary
        path: Path to save JSON file
        indent: JSON indentation
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "02:15:30")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def format_memory(bytes_val: float) -> str:
    """
    Format bytes to human-readable memory string.
    
    Args:
        bytes_val: Memory in bytes
    
    Returns:
        Formatted string (e.g., "4.2 GB")
    """
    gb = bytes_val / (1024 ** 3)
    
    if gb >= 1:
        return f"{gb:.2f} GB"
    
    mb = bytes_val / (1024 ** 2)
    return f"{mb:.2f} MB"


def get_model_variant_from_path(model_path: str) -> str:
    """
    Extract model variant (n/s/m/l/x) from model path.
    
    Args:
        model_path: Path to model file (e.g., "yolov8m.pt")
    
    Returns:
        Model variant letter
    """
    path_str = str(model_path).lower()
    
    for variant in ['n', 's', 'm', 'l', 'x']:
        if f'yolov8{variant}' in path_str:
            return variant
    
    return 'm'  # Default to medium


def ensure_path_exists(path: Path, is_file: bool = False) -> Path:
    """
    Ensure a path exists, creating directories if needed.
    
    Args:
        path: Path to ensure
        is_file: If True, create parent directory; if False, create directory
    
    Returns:
        The path (for chaining)
    """
    if is_file:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    
    return path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    return filename
