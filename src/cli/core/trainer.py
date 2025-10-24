"""
YOLO training wrapper with comprehensive metrics collection.

This module provides a high-level interface to ultralytics YOLO training
with integrated metrics collection, resource monitoring, and result tracking.

Implements the training methodology described in the research paper:
"Influência de Hiperparâmetros no Treinamento do YOLO" (UNIP, 2025)
"""

from typing import Optional, Dict, Any, Callable
from pathlib import Path
import time
import torch
from ultralytics import YOLO

from .config import YOLOHyperparameters, ProjectConfig
from .metrics import (
    TrainingMetrics,
    PerformanceMetrics,
    MetricsCollector,
    LossCurves
)
from .logger import BenchmarkLogger, get_logger
from .utils import detect_device, estimate_batch_size
from .validator import validate_and_sanitize_args


class YOLOTrainer:
    """
    High-level YOLO training wrapper with metrics collection.
    
    This class orchestrates the training process, collecting comprehensive
    metrics throughout training for academic analysis.
    """
    
    def __init__(
        self,
        model_path: str,
        dataset_path: Path,
        project_config: ProjectConfig,
        logger: Optional[BenchmarkLogger] = None
    ):
        """
        Initialize YOLO trainer.
        
        Args:
            model_path: Path to YOLO model weights (e.g., 'yolov8m.pt')
            dataset_path: Path to dataset configuration (data.yaml)
            project_config: Project configuration
            logger: Logger instance (None = use global logger)
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.project_config = project_config
        self.logger = logger or get_logger()
        
        # Initialize model
        self.model: Optional[YOLO] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        
        # Training state
        self._current_epoch: int = 0
        self._total_epochs: int = 0
        
        self.logger.info(f"Initialized YOLOTrainer with model: {model_path}")
        self.logger.info(f"Dataset: {dataset_path}")
    
    def train(
        self,
        hyperparameters: YOLOHyperparameters,
        run_name: str = "experiment",
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None
    ) -> TrainingMetrics:
        """
        Train YOLO model with specified hyperparameters.
        
        Args:
            hyperparameters: Complete hyperparameter configuration
            run_name: Name for this training run (for organization)
            progress_callback: Optional callback for progress updates
                               callback(current_epoch, total_epochs, metrics_dict)
        
        Returns:
            TrainingMetrics with comprehensive results
        """
        self.logger.section(f"Starting Training: {run_name}")
        
        # Initialize model
        self.model = YOLO(self.model_path)
        
        # Setup metrics collection
        self.metrics_collector = MetricsCollector()
        
        # Get device
        device = self.project_config.device or detect_device()
        self.logger.info(f"Training device: {device}")
        
        # Auto-adjust batch size if needed
        batch_size = hyperparameters.batch.batch
        if batch_size == -1:
            model_variant = self.model_path.split('yolov8')[-1].split('.')[0]
            batch_size = estimate_batch_size(
                device,
                hyperparameters.architecture.imgsz,
                model_variant
            )
            hyperparameters.batch.batch = batch_size
            self.logger.info(f"Auto-detected batch size: {batch_size}")
        
        # Prepare training arguments
        train_args = self._prepare_training_args(
            hyperparameters,
            run_name,
            device
        )
        
        self.logger.info("Hyperparameters:")
        for key, value in train_args.items():
            if key not in ['data', 'project', 'name']:
                self.logger.info(f"  {key}: {value}")
        
        # Store epochs for tracking
        self._total_epochs = hyperparameters.architecture.epochs
        
        # Train model
        try:
            self.logger.info("Starting training...")
            self.metrics_collector.start_epoch()
            
            results = self.model.train(**train_args)
            
            self.logger.success("Training completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        
        # Collect final metrics
        metrics = self._collect_final_metrics(results, run_name)
        
        # Measure inference time
        inference_time = self._measure_inference_time(
            hyperparameters.architecture.imgsz
        )
        metrics.operational.inference_time = inference_time
        
        # Log summary
        self._log_metrics_summary(metrics)
        
        return metrics
    
    def _prepare_training_args(
        self,
        hyperparameters: YOLOHyperparameters,
        run_name: str,
        device: str
    ) -> Dict[str, Any]:
        """
        Prepare arguments dictionary for YOLO training.
        
        Args:
            hyperparameters: Hyperparameter configuration
            run_name: Run name for organization
            device: Training device
        
        Returns:
            Arguments dictionary for model.train()
        """
        # Convert to ultralytics format
        args = hyperparameters.to_ultralytics_dict()
        
        # Add required training parameters
        args.update({
            'data': str(self.dataset_path),
            'project': str(self.project_config.models_dir),
            'name': run_name,
            'device': device,
            'workers': self.project_config.workers,
            'verbose': self.project_config.verbose,
            'exist_ok': True,
            'pretrained': True,
            'save': True,
            'save_period': -1,  # Only save last and best
            'plots': True,
            'val': True,
        })
        
        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL: Validate and sanitize arguments for Ultralytics compatibility
        # Removes deprecated arguments (accumulate, label_smoothing, etc.)
        # and ensures all arguments are valid for current YOLO version
        # ═══════════════════════════════════════════════════════════════════
        args = validate_and_sanitize_args(
            args,
            strict=False,  # Filter invalid args instead of raising errors
            verbose=self.project_config.verbose
        )
        
        return args
    
    def _collect_final_metrics(
        self,
        results: Any,
        run_name: str
    ) -> TrainingMetrics:
        """
        Collect final metrics after training completion.
        
        Args:
            results: Training results from ultralytics
            run_name: Run name
        
        Returns:
            Complete TrainingMetrics object
        """
        # Get performance metrics from validation
        validator = self.model.val()
        
        performance = PerformanceMetrics(
            map50=float(validator.results_dict.get('metrics/mAP50(B)', 0.0)),
            map50_95=float(validator.results_dict.get('metrics/mAP50-95(B)', 0.0)),
            precision=float(validator.results_dict.get('metrics/precision(B)', 0.0)),
            recall=float(validator.results_dict.get('metrics/recall(B)', 0.0)),
            f1_score=0.0  # Will calculate
        )
        
        # Calculate F1 score
        if performance.precision + performance.recall > 0:
            performance.f1_score = (
                2 * (performance.precision * performance.recall) /
                (performance.precision + performance.recall)
            )
        
        # Get operational metrics from collector
        operational = self.metrics_collector.get_operational_metrics()
        
        # Extract loss curves from results
        loss_curves = self._extract_loss_curves(results)
        
        # Get model path
        model_path = self.project_config.models_dir / run_name / "weights" / "best.pt"
        
        metrics = TrainingMetrics(
            performance=performance,
            operational=operational,
            loss_curves=loss_curves,
            model_path=model_path if model_path.exists() else None
        )
        
        return metrics
    
    def _extract_loss_curves(self, results: Any) -> LossCurves:
        """
        Extract loss curves from training results.
        
        Args:
            results: Training results object
        
        Returns:
            LossCurves object
        """
        loss_curves = LossCurves()
        
        try:
            # Try to get loss data from results
            if hasattr(results, 'results_dict'):
                # Ultralytics stores losses differently depending on version
                # Try to extract what we can
                pass
            
            # Fallback: read from CSV if available
            csv_path = self.project_config.models_dir / "results.csv"
            if csv_path.exists():
                import pandas as pd
                df = pd.read_csv(csv_path)
                
                if 'box_loss' in df.columns:
                    loss_curves.box_loss = df['box_loss'].tolist()
                if 'cls_loss' in df.columns:
                    loss_curves.cls_loss = df['cls_loss'].tolist()
                if 'dfl_loss' in df.columns:
                    loss_curves.dfl_loss = df['dfl_loss'].tolist()
                if 'val_loss' in df.columns:
                    loss_curves.val_loss = df['val_loss'].tolist()
        
        except Exception as e:
            self.logger.warning(f"Could not extract loss curves: {e}")
        
        return loss_curves
    
    def _measure_inference_time(
        self,
        imgsz: int,
        num_iterations: int = 100
    ) -> float:
        """
        Measure average inference time per image.
        
        Args:
            imgsz: Image size for testing
            num_iterations: Number of inference iterations
        
        Returns:
            Average inference time in milliseconds
        """
        self.logger.info("Measuring inference time...")
        
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, imgsz, imgsz)
            
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                self.model.model.cuda()
            
            # Warmup
            for _ in range(10):
                _ = self.model.predict(dummy_input, verbose=False)
            
            # Measure
            start_time = time.time()
            for _ in range(num_iterations):
                _ = self.model.predict(dummy_input, verbose=False)
            
            total_time = time.time() - start_time
            avg_time_ms = (total_time / num_iterations) * 1000
            
            self.logger.info(f"Average inference time: {avg_time_ms:.2f}ms")
            
            return avg_time_ms
        
        except Exception as e:
            self.logger.warning(f"Could not measure inference time: {e}")
            return 0.0
    
    def _log_metrics_summary(self, metrics: TrainingMetrics) -> None:
        """
        Log a summary of training metrics.
        
        Args:
            metrics: Training metrics to summarize
        """
        self.logger.section("Training Results Summary")
        
        # Performance metrics
        self.logger.info("Performance Metrics:")
        self.logger.info(f"  mAP@0.5:     {metrics.performance.map50:.4f}")
        self.logger.info(f"  mAP@0.5:0.95: {metrics.performance.map50_95:.4f}")
        self.logger.info(f"  Precision:   {metrics.performance.precision:.4f}")
        self.logger.info(f"  Recall:      {metrics.performance.recall:.4f}")
        self.logger.info(f"  F1-Score:    {metrics.performance.f1_score:.4f}")
        
        # Operational metrics
        self.logger.info("\nOperational Metrics:")
        self.logger.info(f"  Time/Epoch:  {metrics.operational.time_per_epoch:.2f}s")
        self.logger.info(f"  Total Time:  {metrics.operational.total_train_time:.2f}s")
        self.logger.info(f"  Inference:   {metrics.operational.inference_time:.2f}ms")
        self.logger.info(f"  Memory Peak: {metrics.operational.memory_peak:.2f}GB")
        self.logger.info(f"  Memory Avg:  {metrics.operational.memory_avg:.2f}GB")
        
        if metrics.operational.gpu_utilization_avg:
            self.logger.info(
                f"  GPU Util:    {metrics.operational.gpu_utilization_avg:.1f}%"
            )
        
        # Loss
        final_loss = metrics.loss_curves.get_final_loss()
        if final_loss > 0:
            self.logger.info(f"\nFinal Loss: {final_loss:.4f}")
    
    def validate(self) -> PerformanceMetrics:
        """
        Run validation on the trained model.
        
        Returns:
            PerformanceMetrics from validation
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Train a model first.")
        
        self.logger.info("Running validation...")
        
        validator = self.model.val()
        
        performance = PerformanceMetrics(
            map50=float(validator.results_dict.get('metrics/mAP50(B)', 0.0)),
            map50_95=float(validator.results_dict.get('metrics/mAP50-95(B)', 0.0)),
            precision=float(validator.results_dict.get('metrics/precision(B)', 0.0)),
            recall=float(validator.results_dict.get('metrics/recall(B)', 0.0)),
            f1_score=0.0
        )
        
        # Calculate F1
        if performance.precision + performance.recall > 0:
            performance.f1_score = (
                2 * (performance.precision * performance.recall) /
                (performance.precision + performance.recall)
            )
        
        return performance
