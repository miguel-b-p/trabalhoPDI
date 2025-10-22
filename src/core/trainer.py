"""
YOLO training module using Ultralytics.
"""

from ultralytics import YOLO
from pathlib import Path
from src.core.config import BenchmarkConfig
import os
from typing import Dict, Any


class YOLOTrainer:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.models_dir = Path(__file__).parent.parent / 'models'
        self.models_dir.mkdir(exist_ok=True)

    def train(self, params: Dict[str, Any]) -> str:
        """Train YOLO model with given parameters and return model path."""
        model = YOLO(self.config.model)

        # Override config with params
        train_args = {
            'data': self.config.data,
            'epochs': params.get('epochs', self.config.max_epochs),
            'batch': params.get('batch_size', self.config.max_batch_size),
            'imgsz': params.get('image_size', self.config.max_image_size),
            'lr0': params.get('learning_rate', self.config.max_learning_rate),
            'device': self.config.device,
            'fraction': params.get('data_quantity', self.config.max_data_quantity),  # Use fraction for data subset
            'name': f"benchmark_{'_'.join(f'{k}_{v}' for k,v in params.items())}",
            'project': str(self.models_dir),
        }

        # Run training
        results = model.train(**train_args)

        # Return path to best model
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        return str(best_model_path)

    def validate(self, model_path: str) -> Dict[str, float]:
        """Validate model and return metrics."""
        model = YOLO(model_path)
        results = model.val(data=self.config.data, device=self.config.device)
        # Extract key metrics
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
        }
        return metrics
