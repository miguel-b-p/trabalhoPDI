"""
Benchmark configuration for systematic hyperparameter testing.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import yaml
from datetime import datetime
import os


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # Basic settings
    name: str = "yolo_benchmark"
    description: str = "YOLO hyperparameter benchmark"
    
    # Dataset settings
    dataset_path: str = "data/coco128.yaml"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    
    # Model settings
    base_model: str = "yolov8m.pt"
    model_size: str = "m"
    
    # Benchmark parameters
    fractions: List[float] = None
    selected_parameters: List[str] = None
    repetitions: int = 3
    
    # Resource limits
    max_epochs: int = 100
    max_batch_size: int = 32
    memory_limit_gb: float = 8.0
    max_time_hours: float = 24.0
    
    # Output settings
    output_dir: str = "results"
    save_models: bool = True
    save_checkpoints: bool = False
    generate_plots: bool = True
    generate_report: bool = True
    
    # Performance tracking
    track_memory: bool = True
    track_time: bool = True
    track_gpu: bool = True
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Parallel processing
    max_parallel_jobs: int = 1
    use_distributed: bool = False
    
    def __post_init__(self):
        """Initialize default values."""
        if self.fractions is None:
            self.fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        if self.selected_parameters is None:
            self.selected_parameters = [
                'lr0', 'batch', 'epochs', 'momentum', 'weight_decay',
                'imgsz', 'mosaic', 'mixup', 'degrees', 'scale'
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def get_output_dir(self) -> str:
        """Get output directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"{self.name}_{timestamp}")
    
    def get_experiment_dir(self, experiment_name: str) -> str:
        """Get directory for specific experiment."""
        return os.path.join(self.get_output_dir(), experiment_name)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if not os.path.exists(self.dataset_path):
            issues.append(f"Dataset path does not exist: {self.dataset_path}")
        
        if self.repetitions < 1:
            issues.append("Repetitions must be at least 1")
        
        if self.max_epochs < 1:
            issues.append("Max epochs must be at least 1")
        
        if self.max_batch_size < 1:
            issues.append("Max batch size must be at least 1")
        
        if self.memory_limit_gb < 1.0:
            issues.append("Memory limit must be at least 1GB")
        
        if self.max_time_hours < 0.1:
            issues.append("Max time must be at least 0.1 hours")
        
        if not self.fractions:
            issues.append("Fractions list cannot be empty")
        
        if not all(0 < f <= 1.0 for f in self.fractions):
            issues.append("All fractions must be between 0 and 1")
        
        return issues
    
    def estimate_total_runs(self) -> int:
        """Estimate total number of training runs."""
        return len(self.selected_parameters) * len(self.fractions) * self.repetitions
    
    def estimate_total_time(self) -> float:
        """Estimate total time in hours."""
        # Rough estimation: 30 minutes per run on average
        avg_time_per_run = 0.5
        return self.estimate_total_runs() * avg_time_per_run
    
    @classmethod
    def from_file(cls, file_path: str) -> 'BenchmarkConfig':
        """Load config from YAML file."""
        import yaml
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BenchmarkConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource requirements."""
        return {
            "total_runs": self.estimate_total_runs(),
            "estimated_time_hours": self.estimate_total_time(),
            "memory_limit_gb": self.memory_limit_gb,
            "max_parallel_jobs": self.max_parallel_jobs,
            "repetitions": self.repetitions,
            "parameters": len(self.selected_parameters),
            "fractions": len(self.fractions)
        }
