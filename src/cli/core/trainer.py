"""
YOLO training wrapper with comprehensive monitoring and metrics collection.
"""

import os
import time
import json
import psutil
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import logging

from config import YOLOConfig
from .monitor import TrainingMonitor
from .metrics import MetricsCollector


class YOLOTrainer:
    """Enhanced YOLO trainer with comprehensive monitoring."""
    
    def __init__(self, config: YOLOConfig, experiment_name: str = None):
        self.config = config
        self.experiment_name = experiment_name or "default"
        self.model = None
        self.results = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitoring
        self.monitor = TrainingMonitor()
        self.metrics = MetricsCollector()
        
        # Setup output directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for training."""
        self.output_dir = Path(self.config.project) / self.config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.weights_dir = self.output_dir / "weights"
        self.weights_dir.mkdir(exist_ok=True)
        
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
    def load_model(self, model_path: str = None) -> YOLO:
        """Load or create YOLO model."""
        model_path = model_path or self.config.model
        
        if Path(model_path).exists():
            self.model = YOLO(model_path)
            self.logger.info(f"Loaded model from {model_path}")
        else:
            self.model = YOLO(self.config.model)
            self.logger.info(f"Created new {self.config.model} model")
            
        return self.model
    
    def train(self, 
              param_overrides: Dict[str, Any] = None,
              callbacks: List = None) -> Dict[str, Any]:
        """
        Train YOLO model with monitoring and metrics collection.
        
        Args:
            param_overrides: Parameter changes for this training run
            callbacks: Optional training callbacks
            
        Returns:
            Dictionary with training results and metrics
        """
        start_time = time.time()
        
        # Apply parameter overrides
        if param_overrides:
            self.config.update_from_dict(param_overrides)
            
        # Save configuration
        config_path = self.output_dir / "config.yaml"
        self.config.save_to_file(str(config_path))
        
        # Initialize model
        self.load_model()
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        try:
            # Train model
            self.logger.info(f"Starting training: {self.experiment_name}")
            self.logger.info(f"Parameters: {param_overrides or 'baseline'}")
            
            results = self.model.train(
                data=self.config.data,
                epochs=self.config.epochs,
                patience=self.config.patience,
                batch=self.config.batch,
                imgsz=self.config.imgsz,
                save=self.config.save,
                save_period=self.config.save_period,
                cache=self.config.cache,
                device=self.config.device,
                workers=self.config.workers,
                project=str(self.output_dir),
                name="train",
                exist_ok=self.config.exist_ok,
                pretrained=self.config.pretrained,
                optimizer=self.config.optimizer,
                lr0=self.config.lr0,
                lrf=self.config.lrf,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                warmup_epochs=self.config.warmup_epochs,
                warmup_momentum=self.config.warmup_momentum,
                warmup_bias_lr=self.config.warmup_bias_lr,
                box=self.config.box,
                cls=self.config.cls,
                dfl=self.config.dfl,
                hsv_h=self.config.hsv_h,
                hsv_s=self.config.hsv_s,
                hsv_v=self.config.hsv_v,
                degrees=self.config.degrees,
                translate=self.config.translate,
                scale=self.config.scale,
                shear=self.config.shear,
                perspective=self.config.perspective,
                flipud=self.config.flipud,
                fliplr=self.config.fliplr,
                mosaic=self.config.mosaic,
                mixup=self.config.mixup,
                copy_paste=self.config.copy_paste,
                val=self.config.val,
                split=self.config.split,
                verbose=self.config.verbose,
            )
            
            self.results = results
            
            # Collect final metrics
            final_metrics = self.collect_final_metrics()
            
            # Stop monitoring
            system_stats = self.monitor.stop_monitoring()
            
            # Compile comprehensive results
            training_results = {
                "experiment_name": self.experiment_name,
                "parameters": self.config.to_dict(),
                "param_changes": param_overrides or {},
                "metrics": final_metrics,
                "system_stats": system_stats,
                "training_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "model_size_mb": self.get_model_size(),
                "status": "completed"
            }
            
            # Save results
            self.save_results(training_results)
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            
            # Collect partial results if available
            system_stats = self.monitor.stop_monitoring()
            
            error_results = {
                "experiment_name": self.experiment_name,
                "parameters": self.config.to_dict(),
                "param_changes": param_overrides or {},
                "error": str(e),
                "system_stats": system_stats,
                "training_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            }
            
            self.save_results(error_results)
            raise
    
    def collect_final_metrics(self) -> Dict[str, Any]:
        """Collect final training and validation metrics."""
        if not self.results:
            return {}
            
        metrics = {}
        
        # Training metrics from results
        if hasattr(self.results, 'results_dict'):
            metrics.update(self.results.results_dict)
        
        # Validation metrics
        if hasattr(self.results, 'metrics'):
            metrics['val_metrics'] = self.results.metrics
        
        # Best metrics
        if hasattr(self.results, 'best_fitness'):
            metrics['best_fitness'] = self.results.best_fitness
        
        # Add mAP calculations
        if 'metrics/mAP50(B)' in metrics:
            metrics['mAP50'] = metrics['metrics/mAP50(B)']
        if 'metrics/mAP50-95(B)' in metrics:
            metrics['mAP50_95'] = metrics['metrics/mAP50-95(B)']
        
        return metrics
    
    def get_model_size(self) -> float:
        """Get model size in MB."""
        if self.model is None:
            return 0.0
            
        try:
            model_path = self.weights_dir / "best.pt"
            if model_path.exists():
                return model_path.stat().st_size / (1024 * 1024)
        except:
            pass
            
        return 0.0
    
    def validate(self, data_path: str = None) -> Dict[str, Any]:
        """Run validation on trained model."""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        data_path = data_path or self.config.data
        
        val_results = self.model.val(
            data=data_path,
            imgsz=self.config.imgsz,
            batch=self.config.batch,
            device=self.config.device,
            workers=self.config.workers,
            project=str(self.output_dir),
            name="val",
            exist_ok=True,
        )
        
        return {
            "val_metrics": val_results.results_dict,
            "val_time": time.time(),
            "model_path": str(self.weights_dir / "best.pt")
        }
    
    def predict(self, source: str, **kwargs) -> List[Dict[str, Any]]:
        """Run inference on images or videos."""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        results = self.model.predict(
            source=source,
            imgsz=self.config.imgsz,
            conf=self.config.conf_thres,
            iou=self.config.iou_thres,
            device=self.config.device,
            **kwargs
        )
        
        return [
            {
                "path": r.path,
                "boxes": r.boxes.xyxy.tolist() if r.boxes else [],
                "scores": r.boxes.conf.tolist() if r.boxes else [],
                "classes": r.boxes.cls.tolist() if r.boxes else [],
                "speed": r.speed
            }
            for r in results
        ]
    
    def save_results(self, results: Dict[str, Any]):
        """Save training results to JSON file."""
        results_file = self.output_dir / "results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def export_model(self, format: str = "onnx") -> str:
        """Export trained model to specified format."""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        export_path = self.model.export(
            format=format,
            imgsz=self.config.imgsz,
            device=self.config.device,
            optimize=True,
            half=True,
            simplify=True
        )
        
        return str(export_path)
