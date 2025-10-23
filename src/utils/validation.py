"""
Validation utilities for system requirements and configurations.
"""

import sys
import psutil
import GPUtil
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def check_system_requirements() -> Dict[str, Any]:
    """
    Check system requirements and capabilities.
    
    Returns:
        Dictionary with system information and requirements check
    """
    requirements = {
        "python_version": sys.version_info,
        "cpu_cores": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "requirements_met": True,
        "issues": []
    }
    
    # Check Python version
    if sys.version_info < (3, 8):
        requirements["requirements_met"] = False
        requirements["issues"].append("Python 3.8+ required")
    
    # Check memory
    if requirements["memory_gb"] < 4:
        requirements["requirements_met"] = False
        requirements["issues"].append("At least 4GB RAM recommended")
    
    # Check GPU
    if not requirements["gpu_available"]:
        requirements["issues"].append("GPU not available - training will be slow")
    
    # Check GPU memory
    if requirements["gpu_available"]:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = []
                for gpu in gpus:
                    gpu_info.append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_gb": gpu.memoryTotal / 1024,
                        "memory_free_gb": gpu.memoryFree / 1024
                    })
                requirements["gpu_details"] = gpu_info
        except Exception as e:
            logger.warning(f"Could not get GPU details: {e}")
    
    return requirements


def validate_config(config_dict: Dict[str, Any]) -> List[str]:
    """
    Validate configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary to validate
    
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    # Required fields
    required_fields = ["name", "dataset_path", "base_model"]
    for field in required_fields:
        if field not in config_dict:
            issues.append(f"Missing required field: {field}")
    
    # Dataset path validation
    if "dataset_path" in config_dict:
        dataset_path = Path(config_dict["dataset_path"])
        if not dataset_path.exists():
            issues.append(f"Dataset path does not exist: {dataset_path}")
    
    # Numeric validations
    numeric_fields = {
        "max_epochs": (1, 1000),
        "max_batch_size": (1, 1024),
        "repetitions": (1, 10),
        "memory_limit_gb": (1, 128),
        "max_time_hours": (0.1, 168)
    }
    
    for field, (min_val, max_val) in numeric_fields.items():
        if field in config_dict:
            value = config_dict[field]
            if not isinstance(value, (int, float)):
                issues.append(f"{field} must be numeric")
            elif value < min_val or value > max_val:
                issues.append(f"{field} must be between {min_val} and {max_val}")
    
    # Fractions validation
    if "fractions" in config_dict:
        fractions = config_dict["fractions"]
        if not isinstance(fractions, list):
            issues.append("fractions must be a list")
        elif not fractions:
            issues.append("fractions cannot be empty")
        elif not all(0 < f <= 1.0 for f in fractions):
            issues.append("all fractions must be between 0 and 1")
    
    # Parameters validation
    if "selected_parameters" in config_dict:
        from ..config.parameters import HyperparameterSpace
        param_space = HyperparameterSpace()
        valid_params = set(param_space.parameters.keys())
        
        selected_params = config_dict["selected_parameters"]
        if not isinstance(selected_params, list):
            issues.append("selected_parameters must be a list")
        else:
            invalid_params = set(selected_params) - valid_params
            if invalid_params:
                issues.append(f"Invalid parameters: {invalid_params}")
    
    return issues


def validate_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Validate dataset structure and content.
    
    Args:
        dataset_path: Path to dataset configuration file
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        "valid": True,
        "issues": [],
        "info": {}
    }
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        validation["valid"] = False
        validation["issues"].append(f"Dataset file not found: {dataset_path}")
        return validation
    
    try:
        import yaml
        with open(dataset_path) as f:
            dataset_config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ["train", "val", "nc", "names"]
        for field in required_fields:
            if field not in dataset_config:
                validation["valid"] = False
                validation["issues"].append(f"Missing dataset field: {field}")
        
        # Check paths
        for split in ["train", "val"]:
            if split in dataset_config:
                split_path = Path(dataset_config[split])
                if not split_path.exists():
                    validation["valid"] = False
                    validation["issues"].append(f"{split} path not found: {split_path}")
        
        # Check class names
        if "names" in dataset_config:
            names = dataset_config["names"]
            if isinstance(names, dict):
                names = list(names.values())
            validation["info"]["num_classes"] = len(names)
            validation["info"]["class_names"] = names[:5]  # First 5 classes
        
    except Exception as e:
        validation["valid"] = False
        validation["issues"].append(f"Error reading dataset: {e}")
    
    return validation


def check_disk_space(required_gb: float = 10) -> Dict[str, Any]:
    """
    Check available disk space.
    
    Args:
        required_gb: Required space in GB
    
    Returns:
        Dictionary with disk space information
    """
    disk_info = {}
    
    try:
        # Check current directory
        current_path = Path.cwd()
        disk_usage = psutil.disk_usage(current_path)
        
        disk_info = {
            "total_gb": disk_usage.total / (1024**3),
            "used_gb": disk_usage.used / (1024**3),
            "free_gb": disk_usage.free / (1024**3),
            "required_gb": required_gb,
            "sufficient_space": disk_usage.free / (1024**3) >= required_gb
        }
        
        if not disk_info["sufficient_space"]:
            logger.warning(f"Insufficient disk space: {disk_info['free_gb']:.1f}GB available, {required_gb}GB required")
    
    except Exception as e:
        logger.error(f"Error checking disk space: {e}")
        disk_info["error"] = str(e)
    
    return disk_info


def print_system_report():
    """Print comprehensive system report."""
    print("=" * 60)
    print("SYSTEM REQUIREMENTS REPORT")
    print("=" * 60)
    
    # System info
    requirements = check_system_requirements()
    disk_info = check_disk_space()
    
    print(f"Python Version: {requirements['python_version'].major}.{requirements['python_version'].minor}")
    print(f"CPU Cores: {requirements['cpu_cores']}")
    print(f"Memory: {requirements['memory_gb']:.1f} GB")
    print(f"GPU Available: {requirements['gpu_available']}")
    
    if requirements['gpu_available'] and 'gpu_details' in requirements:
        for gpu in requirements['gpu_details']:
            print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
    
    print(f"Disk Space: {disk_info['free_gb']:.1f} GB free")
    
    # Issues
    if requirements['issues']:
        print("\nWARNINGS:")
        for issue in requirements['issues']:
            print(f"  ⚠️  {issue}")
    
    if not requirements['requirements_met']:
        print("\n❌ Some requirements not met")
    else:
        print("\n✅ System requirements satisfied")
    
    print("=" * 60)
