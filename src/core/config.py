"""
Configuration module for YOLO benchmark parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BenchmarkConfig:
    # Max values for parameters
    max_epochs: int = 100
    max_batch_size: int = 32
    max_image_size: int = 640
    max_learning_rate: float = 0.01
    max_data_quantity: float = 1.0  # fraction of total data

    # Other fixed params
    model: str = 'yolov8n.pt'  # path or name
    data: str = 'data.yaml'  # dataset config
    device: str = 'cpu'  # or 'cuda'

    # Benchmark settings
    quantity: int = 5  # number of steps, e.g., 5 for 1/5,2/5,...,5/5


def get_param_steps(max_value: Any, quantity: int) -> list:
    """Generate steps for a parameter: 1/quantity to quantity/quantity of max_value."""
    if isinstance(max_value, int):
        return [int((i / quantity) * max_value) for i in range(1, quantity + 1)]
    elif isinstance(max_value, float):
        return [(i / quantity) * max_value for i in range(1, quantity + 1)]
    else:
        # For non-numeric, perhaps just repeat or something, but assume numeric
        raise ValueError("Parameter must be numeric")
