"""
Metrics collection, calculation, and aggregation system.

This module provides comprehensive metrics tracking for YOLO training,
including performance metrics (mAP, precision, recall) and operational
metrics (time, memory usage).

Implements the data collection methodology described in the research paper.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import psutil
import numpy as np
from datetime import datetime
from pydantic import BaseModel, Field


class PerformanceMetrics(BaseModel):
    """
    Model performance metrics.
    
    These are the primary metrics for evaluating object detection quality,
    following COCO evaluation standards.
    """
    
    map50: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean Average Precision @ IoU=0.5"
    )
    
    map50_95: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean Average Precision @ IoU=0.5:0.95"
    )
    
    precision: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Precision"
    )
    
    recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Recall"
    )
    
    f1_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="F1-Score (harmonic mean of precision and recall)"
    )
    
    @classmethod
    def from_ultralytics_results(cls, results: Dict[str, Any]) -> 'PerformanceMetrics':
        """
        Create PerformanceMetrics from ultralytics training results.
        
        Args:
            results: Results dictionary from YOLO validator
        
        Returns:
            PerformanceMetrics instance
        """
        precision = results.get('metrics/precision(B)', 0.0)
        recall = results.get('metrics/recall(B)', 0.0)
        
        # Calculate F1-Score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return cls(
            map50=results.get('metrics/mAP50(B)', 0.0),
            map50_95=results.get('metrics/mAP50-95(B)', 0.0),
            precision=precision,
            recall=recall,
            f1_score=f1
        )


class OperationalMetrics(BaseModel):
    """
    Operational metrics tracking computational costs.
    
    Critical for understanding the practical trade-offs between
    model performance and resource requirements.
    """
    
    time_per_epoch: float = Field(
        default=0.0,
        ge=0.0,
        description="Average time per epoch (seconds)"
    )
    
    total_train_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Total training time (seconds)"
    )
    
    inference_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Average inference time per image (milliseconds)"
    )
    
    memory_peak: float = Field(
        default=0.0,
        ge=0.0,
        description="Peak memory usage (GB)"
    )
    
    memory_avg: float = Field(
        default=0.0,
        ge=0.0,
        description="Average memory usage (GB)"
    )
    
    gpu_utilization_avg: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Average GPU utilization (%)"
    )
    
    gpu_memory_used: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="GPU memory used (GB)"
    )


class LossCurves(BaseModel):
    """
    Training loss curves for analysis.
    
    Stores loss values across epochs for visualization and
    convergence analysis.
    """
    
    box_loss: List[float] = Field(
        default_factory=list,
        description="Box regression loss per epoch"
    )
    
    cls_loss: List[float] = Field(
        default_factory=list,
        description="Classification loss per epoch"
    )
    
    dfl_loss: List[float] = Field(
        default_factory=list,
        description="Distribution Focal Loss per epoch"
    )
    
    val_loss: List[float] = Field(
        default_factory=list,
        description="Validation loss per epoch"
    )
    
    def add_epoch(
        self,
        box: float,
        cls: float,
        dfl: float,
        val: Optional[float] = None
    ) -> None:
        """Add loss values for an epoch."""
        self.box_loss.append(box)
        self.cls_loss.append(cls)
        self.dfl_loss.append(dfl)
        if val is not None:
            self.val_loss.append(val)
    
    def get_final_loss(self) -> float:
        """Get final combined loss."""
        if not self.box_loss:
            return 0.0
        return self.box_loss[-1] + self.cls_loss[-1] + self.dfl_loss[-1]


class TrainingMetrics(BaseModel):
    """
    Complete metrics for a single training run.
    
    Aggregates performance, operational, and loss metrics with
    associated configuration metadata.
    """
    
    performance: PerformanceMetrics = Field(
        default_factory=PerformanceMetrics
    )
    
    operational: OperationalMetrics = Field(
        default_factory=OperationalMetrics
    )
    
    loss_curves: LossCurves = Field(
        default_factory=LossCurves
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Training completion timestamp"
    )
    
    model_path: Optional[Path] = Field(
        default=None,
        description="Path to saved model weights"
    )
    
    def to_summary_dict(self) -> Dict[str, float]:
        """
        Get summary dictionary of key metrics.
        
        Returns:
            Dictionary with primary metrics for quick comparison
        """
        return {
            'mAP@0.5': self.performance.map50,
            'mAP@0.5:0.95': self.performance.map50_95,
            'f1_score': self.performance.f1_score,
            'time_per_epoch': self.operational.time_per_epoch,
            'total_time': self.operational.total_train_time,
            'memory_peak': self.operational.memory_peak,
            'final_loss': self.loss_curves.get_final_loss()
        }


@dataclass
class MetricsCollector:
    """
    Real-time metrics collection during training.
    
    Monitors system resources and collects metrics as training progresses.
    This class is designed to be non-intrusive and minimal overhead.
    """
    
    start_time: float = field(default_factory=time.time)
    epoch_times: List[float] = field(default_factory=list)
    memory_samples: List[float] = field(default_factory=list)
    gpu_samples: List[float] = field(default_factory=list)
    
    _epoch_start: Optional[float] = field(default=None, init=False)
    _process: Optional[psutil.Process] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize process monitoring."""
        self._process = psutil.Process()
    
    def start_epoch(self) -> None:
        """Mark the start of an epoch."""
        self._epoch_start = time.time()
        self._sample_resources()
    
    def end_epoch(self) -> None:
        """Mark the end of an epoch."""
        if self._epoch_start is not None:
            epoch_time = time.time() - self._epoch_start
            self.epoch_times.append(epoch_time)
            self._sample_resources()
    
    def _sample_resources(self) -> None:
        """Sample current resource usage."""
        try:
            # Memory usage in GB
            memory_info = self._process.memory_info()
            memory_gb = memory_info.rss / (1024 ** 3)
            self.memory_samples.append(memory_gb)
            
            # Try to get GPU stats (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
                    gpu_util = torch.cuda.utilization()
                    self.gpu_samples.append({
                        'memory': gpu_memory,
                        'utilization': gpu_util
                    })
            except (ImportError, Exception):
                pass  # GPU monitoring not available
                
        except Exception as e:
            # Don't let monitoring errors crash training
            pass
    
    def get_operational_metrics(self) -> OperationalMetrics:
        """
        Calculate final operational metrics.
        
        Returns:
            OperationalMetrics with aggregated data
        """
        total_time = time.time() - self.start_time
        
        avg_epoch_time = (
            np.mean(self.epoch_times) if self.epoch_times else 0.0
        )
        
        memory_peak = (
            max(self.memory_samples) if self.memory_samples else 0.0
        )
        
        memory_avg = (
            np.mean(self.memory_samples) if self.memory_samples else 0.0
        )
        
        # GPU metrics (if available)
        gpu_util_avg = None
        gpu_memory = None
        
        if self.gpu_samples:
            gpu_util_avg = np.mean([s['utilization'] for s in self.gpu_samples])
            gpu_memory = np.mean([s['memory'] for s in self.gpu_samples])
        
        return OperationalMetrics(
            time_per_epoch=avg_epoch_time,
            total_train_time=total_time,
            inference_time=0.0,  # Will be measured separately
            memory_peak=memory_peak,
            memory_avg=memory_avg,
            gpu_utilization_avg=gpu_util_avg,
            gpu_memory_used=gpu_memory
        )


class BenchmarkResult(BaseModel):
    """
    Result of a single benchmark test.
    
    Contains configuration, metrics, and metadata for one fraction
    in the benchmark experiment.
    """
    
    test_number: int = Field(
        ge=1,
        description="Test number in sequence"
    )
    
    fraction: str = Field(
        description="Fraction label (e.g., '3/5')"
    )
    
    variable_name: str = Field(
        description="Name of the benchmarked variable"
    )
    
    variable_value: float = Field(
        description="Value of the variable for this test"
    )
    
    config: Dict[str, Any] = Field(
        description="Complete hyperparameter configuration"
    )
    
    metrics: TrainingMetrics = Field(
        description="Collected metrics from training"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'test_number': self.test_number,
            'fraction': self.fraction,
            'variable_name': self.variable_name,
            'variable_value': self.variable_value,
            'config': self.config,
            'metrics': {
                'performance': self.metrics.performance.model_dump(),
                'operational': self.metrics.operational.model_dump(),
                'loss_curves': self.metrics.loss_curves.model_dump(),
                'timestamp': self.metrics.timestamp.isoformat(),
            }
        }


class BenchmarkResults(BaseModel):
    """
    Complete results from a benchmark experiment.
    
    Aggregates all test results with metadata about the benchmark run.
    """
    
    benchmark_id: str = Field(
        description="Unique identifier for this benchmark"
    )
    
    variable_name: str = Field(
        description="Hyperparameter that was benchmarked"
    )
    
    num_fractions: int = Field(
        ge=2,
        description="Number of fractions tested"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Benchmark start timestamp"
    )
    
    tests: List[BenchmarkResult] = Field(
        default_factory=list,
        description="Results from each test"
    )
    
    def add_result(self, result: BenchmarkResult) -> None:
        """Add a test result to the benchmark."""
        self.tests.append(result)
    
    def save_to_json(self, path: Path) -> None:
        """
        Save benchmark results to JSON file.
        
        Args:
            path: Path to save JSON file
        """
        data = {
            'benchmark_id': self.benchmark_id,
            'variable_name': self.variable_name,
            'num_fractions': self.num_fractions,
            'timestamp': self.timestamp.isoformat(),
            'tests': [test.to_dict() for test in self.tests]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_json(cls, path: Path) -> 'BenchmarkResults':
        """
        Load benchmark results from JSON file.
        
        Args:
            path: Path to JSON file
        
        Returns:
            BenchmarkResults instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct tests
        tests = []
        for test_data in data['tests']:
            metrics = TrainingMetrics(
                performance=PerformanceMetrics(**test_data['metrics']['performance']),
                operational=OperationalMetrics(**test_data['metrics']['operational']),
                loss_curves=LossCurves(**test_data['metrics']['loss_curves']),
                timestamp=datetime.fromisoformat(test_data['metrics']['timestamp'])
            )
            
            result = BenchmarkResult(
                test_number=test_data['test_number'],
                fraction=test_data['fraction'],
                variable_name=test_data['variable_name'],
                variable_value=test_data['variable_value'],
                config=test_data['config'],
                metrics=metrics
            )
            tests.append(result)
        
        return cls(
            benchmark_id=data['benchmark_id'],
            variable_name=data['variable_name'],
            num_fractions=data['num_fractions'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            tests=tests
        )
    
    def get_summary_dataframe(self) -> Dict[str, List[Any]]:
        """
        Get summary data suitable for pandas DataFrame or Bokeh plotting.
        
        Returns:
            Dictionary with columns for visualization
        """
        summary = {
            'fraction': [],
            'variable_value': [],
            'map50': [],
            'map50_95': [],
            'f1_score': [],
            'precision': [],
            'recall': [],
            'time_per_epoch': [],
            'total_time': [],
            'memory_peak': [],
            'final_loss': []
        }
        
        for test in self.tests:
            summary['fraction'].append(test.fraction)
            summary['variable_value'].append(test.variable_value)
            summary['map50'].append(test.metrics.performance.map50)
            summary['map50_95'].append(test.metrics.performance.map50_95)
            summary['f1_score'].append(test.metrics.performance.f1_score)
            summary['precision'].append(test.metrics.performance.precision)
            summary['recall'].append(test.metrics.performance.recall)
            summary['time_per_epoch'].append(test.metrics.operational.time_per_epoch)
            summary['total_time'].append(test.metrics.operational.total_train_time)
            summary['memory_peak'].append(test.metrics.operational.memory_peak)
            summary['final_loss'].append(test.metrics.loss_curves.get_final_loss())
        
        return summary
