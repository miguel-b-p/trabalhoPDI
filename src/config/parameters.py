"""
Hyperparameter space definitions for YOLO benchmarking.
Based on the research parameters from "Influência de Hiperparâmetros no Treinamento do YOLO".
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np


@dataclass
class ParameterRange:
    """Defines a parameter range for benchmarking."""
    name: str
    min_value: float
    max_value: float
    step: float = None
    values: List[Any] = None
    type: str = "float"
    
    def get_fractional_values(self, fractions: List[float]) -> List[float]:
        """Get parameter values for given fractions of the range."""
        if self.values:
            indices = [int(f * (len(self.values) - 1)) for f in fractions]
            return [self.values[i] for i in indices]
        
        range_size = self.max_value - self.min_value
        values = [self.min_value + f * range_size for f in fractions]
        
        if self.type == "int":
            values = [int(round(v)) for v in values]
        
        return values


class HyperparameterSpace:
    """Complete hyperparameter space for YOLOv8m benchmarking."""
    
    def __init__(self):
        self.parameters = {
            # Learning parameters
            "lr0": ParameterRange("initial_learning_rate", 0.0001, 0.1, type="float"),
            "lrf": ParameterRange("final_learning_rate_factor", 0.01, 1.0, type="float"),
            
            # Batch and accumulation
            "batch": ParameterRange("batch_size", 4, 64, type="int"),
            "accumulate": ParameterRange("gradient_accumulation", 1, 8, type="int"),
            
            # Optimizer parameters
            "momentum": ParameterRange("momentum", 0.6, 0.98, type="float"),
            "weight_decay": ParameterRange("weight_decay", 0.0001, 0.001, type="float"),
            
            # Training duration
            "epochs": ParameterRange("epochs", 50, 300, type="int"),
            "patience": ParameterRange("early_stopping_patience", 10, 100, type="int"),
            
            # Input processing
            "imgsz": ParameterRange("image_size", 320, 1280, step=32, type="int"),
            "hsv_h": ParameterRange("hsv_hue", 0.0, 0.1, type="float"),
            "hsv_s": ParameterRange("hsv_saturation", 0.0, 0.9, type="float"),
            "hsv_v": ParameterRange("hsv_value", 0.0, 0.9, type="float"),
            
            # Augmentation parameters
            "degrees": ParameterRange("rotation_degrees", 0.0, 45.0, type="float"),
            "translate": ParameterRange("translation", 0.0, 0.5, type="float"),
            "scale": ParameterRange("scale", 0.0, 0.5, type="float"),
            "shear": ParameterRange("shear", 0.0, 10.0, type="float"),
            "perspective": ParameterRange("perspective", 0.0, 0.001, type="float"),
            
            # Flip and mosaic
            "flipud": ParameterRange("flip_up_down", 0.0, 1.0, type="float"),
            "fliplr": ParameterRange("flip_left_right", 0.0, 1.0, type="float"),
            "mosaic": ParameterRange("mosaic", 0.0, 1.0, type="float"),
            "mixup": ParameterRange("mixup", 0.0, 1.0, type="float"),
            
            # Copy-paste augmentation
            "copy_paste": ParameterRange("copy_paste", 0.0, 1.0, type="float"),
            
            # Loss parameters
            "box": ParameterRange("box_loss_gain", 0.02, 0.2, type="float"),
            "cls": ParameterRange("cls_loss_gain", 0.2, 4.0, type="float"),
            "dfl": ParameterRange("dfl_loss_gain", 0.4, 6.0, type="float"),
            
            # NMS parameters
            "iou_thres": ParameterRange("iou_threshold", 0.1, 0.7, type="float"),
            "conf_thres": ParameterRange("confidence_threshold", 0.001, 0.5, type="float"),
        }
    
    def get_parameter(self, name: str) -> ParameterRange:
        """Get parameter range by name."""
        return self.parameters[name]
    
    def get_all_parameters(self) -> Dict[str, ParameterRange]:
        """Get all parameter ranges."""
        return self.parameters
    
    def get_benchmark_parameters(self, selected: List[str] = None) -> Dict[str, ParameterRange]:
        """Get parameters for benchmarking."""
        if selected is None:
            return self.parameters
        return {k: v for k, v in self.parameters.items() if k in selected}
    
    def generate_benchmark_configs(self, fractions: List[float], 
                                 selected_params: List[str] = None) -> List[Dict[str, Any]]:
        """Generate benchmark configurations with fractional parameter values."""
        if selected_params is None:
            selected_params = list(self.parameters.keys())
        
        configs = []
        
        # Single parameter variation
        for param_name in selected_params:
            if param_name not in self.parameters:
                continue
                
            param = self.parameters[param_name]
            values = param.get_fractional_values(fractions)
            
            for value in values:
                config = {"type": "single", "parameter": param_name, "value": value}
                configs.append(config)
        
        return configs
