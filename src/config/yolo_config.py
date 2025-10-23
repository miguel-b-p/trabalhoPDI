"""
YOLO configuration management for training sessions.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import yaml
import os
from pathlib import Path


@dataclass
class YOLOConfig:
    """Complete YOLO training configuration."""
    
    # Model settings
    model: str = "yolov8m.pt"
    task: str = "detect"
    verbose: bool = True
    
    # Training parameters
    data: str = "coco128.yaml"
    epochs: int = 100
    patience: int = 50
    batch: int = 16
    imgsz: int = 640
    save: bool = True
    save_period: int = -1
    cache: bool = False
    device: str = ""
    workers: int = 8
    project: str = "runs/train"
    name: str = "exp"
    exist_ok: bool = False
    pretrained: bool = True
    optimizer: str = "auto"
    
    # Learning rate parameters
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # Loss parameters
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    
    # NMS parameters
    nms: bool = True
    iou_thres: float = 0.7
    conf_thres: float = 0.25
    
    # Augmentation parameters
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    
    # Validation parameters
    val: bool = True
    split: str = "val"
    
    # Export parameters
    format: str = "torchscript"
    keras: bool = False
    optimize: bool = False
    int8: bool = False
    dynamic: bool = False
    simplify: bool = False
    opset: int = None
    workspace: int = 4
    nms: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_yaml(self) -> str:
        """Export config to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'YOLOConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'YOLOConfig':
        """Create config from YAML string."""
        config_dict = yaml.safe_load(yaml_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'YOLOConfig':
        """Load config from YAML file."""
        import yaml
        with open(file_path, 'r') as f:
            return cls.from_yaml(f.read())
    
    def save_to_file(self, file_path: str):
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(self.to_yaml())
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_benchmark_params(self) -> Dict[str, Any]:
        """Get parameters relevant for benchmarking."""
        benchmark_keys = [
            'lr0', 'lrf', 'batch', 'epochs', 'patience', 'imgsz',
            'momentum', 'weight_decay', 'hsv_h', 'hsv_s', 'hsv_v',
            'degrees', 'translate', 'scale', 'shear', 'perspective',
            'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste',
            'box', 'cls', 'dfl', 'iou_thres', 'conf_thres'
        ]
        
        return {k: getattr(self, k) for k in benchmark_keys if hasattr(self, k)}
    
    def create_experiment_name(self, param_changes: Dict[str, Any]) -> str:
        """Generate experiment name based on parameter changes."""
        if not param_changes:
            return "baseline"
        
        parts = []
        for key, value in param_changes.items():
            if isinstance(value, float):
                parts.append(f"{key}_{value:.4f}")
            else:
                parts.append(f"{key}_{value}")
        
        return "_".join(parts)
