"""
Configuration models using Pydantic for type validation and serialization.

This module defines all hyperparameter configurations for YOLOv8 training,
following the academic research methodology defined in the paper:
"Influência de Hiperparâmetros no Treinamento do YOLO" (UNIP, 2025)

All models use Pydantic v2 for robust validation and serialization.
"""

from typing import Literal, Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
import uuid


class OptimizationConfig(BaseModel):
    """
    Hyperparameters related to optimization and learning rate.
    
    Based on research investigating the impact of learning rate schedules
    and optimization algorithms on YOLO convergence.
    """
    
    lr0: float = Field(
        default=0.01,
        ge=0.0001,
        le=0.1,
        description="Initial learning rate (i.e. SGD=1E-2, Adam=1E-3)"
    )
    
    lrf: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Final learning rate (lr0 * lrf)"
    )
    
    momentum: float = Field(
        default=0.937,
        ge=0.0,
        le=1.0,
        description="SGD momentum/Adam beta1"
    )
    
    weight_decay: float = Field(
        default=0.0005,
        ge=0.0,
        le=0.001,
        description="Optimizer weight decay (L2 regularization)"
    )
    
    warmup_epochs: float = Field(
        default=3.0,
        ge=0.0,
        le=10.0,
        description="Warmup epochs (fractions ok)"
    )
    
    warmup_momentum: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Warmup initial momentum"
    )
    
    warmup_bias_lr: float = Field(
        default=0.1,
        ge=0.0,
        le=0.2,
        description="Warmup initial bias lr"
    )
    
    optimizer: Literal['SGD', 'Adam', 'AdamW', 'RMSProp'] = Field(
        default='SGD',
        description="Optimizer algorithm"
    )
    
    model_config = ConfigDict(validate_assignment=True)


class BatchConfig(BaseModel):
    """
    Batch size and loss normalization configuration.
    
    Critical for memory management and training stability.
    
    NOTE: 'accumulate' was replaced with 'nbs' (Nominal Batch Size) in Ultralytics v8+.
    nbs controls loss normalization and is typically set to 64.
    """
    
    batch: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Batch size (auto-batch if -1)"
    )
    
    nbs: int = Field(
        default=64,
        ge=1,
        le=256,
        description="Nominal batch size for loss normalization (replaces accumulate)"
    )
    
    model_config = ConfigDict(validate_assignment=True)


class ArchitectureConfig(BaseModel):
    """
    Model architecture and input configuration.
    
    Defines image resolution and training duration - key factors
    affecting both performance and computational cost.
    """
    
    imgsz: int = Field(
        default=640,
        ge=32,
        description="Input image size (pixels)"
    )
    
    epochs: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of training epochs"
    )
    
    patience: int = Field(
        default=50,
        ge=0,
        description="Early stopping patience (epochs without improvement)"
    )
    
    @field_validator('imgsz')
    @classmethod
    def validate_imgsz_multiple_32(cls, v: int) -> int:
        """Ensure image size is multiple of 32 (YOLO requirement)."""
        if v % 32 != 0:
            raise ValueError(f"imgsz must be multiple of 32, got {v}")
        return v
    
    model_config = ConfigDict(validate_assignment=True)


class DataAugmentationConfig(BaseModel):
    """
    Data augmentation hyperparameters.
    
    Based on research investigating the trade-off between augmentation
    strength and overfitting prevention vs. training time.
    """
    
    # Color space augmentations
    hsv_h: float = Field(
        default=0.015,
        ge=0.0,
        le=0.1,
        description="HSV-Hue augmentation (fraction)"
    )
    
    hsv_s: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="HSV-Saturation augmentation (fraction)"
    )
    
    hsv_v: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="HSV-Value augmentation (fraction)"
    )
    
    # Geometric augmentations
    degrees: float = Field(
        default=0.0,
        ge=0.0,
        le=45.0,
        description="Image rotation (+/- deg)"
    )
    
    translate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Image translation (+/- fraction)"
    )
    
    scale: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Image scale (+/- gain)"
    )
    
    shear: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Image shear (+/- deg)"
    )
    
    perspective: float = Field(
        default=0.0,
        ge=0.0,
        le=0.001,
        description="Image perspective (+/- fraction)"
    )
    
    # Flip augmentations
    flipud: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Vertical flip probability"
    )
    
    fliplr: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Horizontal flip probability"
    )
    
    # Advanced augmentations
    mosaic: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Mosaic augmentation probability"
    )
    
    mixup: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="MixUp augmentation probability"
    )
    
    copy_paste: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Copy-paste augmentation probability"
    )
    
    model_config = ConfigDict(validate_assignment=True)


class RegularizationConfig(BaseModel):
    """
    Regularization techniques configuration.
    
    NOTE: 'label_smoothing' was REMOVED in Ultralytics v8+ (no replacement).
    Use dropout for regularization.
    """
    
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Dropout probability (classification head)"
    )
    
    model_config = ConfigDict(validate_assignment=True)


class PostProcessingConfig(BaseModel):
    """
    Non-Maximum Suppression (NMS) configuration.
    
    Critical hyperparameters affecting inference quality and speed.
    """
    
    conf: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Object confidence threshold"
    )
    
    iou: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS"
    )
    
    max_det: int = Field(
        default=300,
        ge=1,
        description="Maximum number of detections per image"
    )
    
    model_config = ConfigDict(validate_assignment=True)


class YOLOHyperparameters(BaseModel):
    """
    Complete YOLO hyperparameter configuration.
    
    Aggregates all hyperparameter categories into a single validated model.
    This is the main configuration object used throughout the system.
    """
    
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    architecture: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
    augmentation: DataAugmentationConfig = Field(default_factory=DataAugmentationConfig)
    regularization: RegularizationConfig = Field(default_factory=RegularizationConfig)
    postprocessing: PostProcessingConfig = Field(default_factory=PostProcessingConfig)
    
    def to_ultralytics_dict(self) -> Dict[str, Any]:
        """
        Convert to ultralytics YOLO training arguments dictionary.
        
        Returns:
            Dictionary compatible with YOLO().train() method
        """
        config = {}
        
        # Flatten all nested configs
        for category in [
            self.optimization,
            self.batch,
            self.architecture,
            self.augmentation,
            self.regularization,
            self.postprocessing
        ]:
            config.update(category.model_dump())
        
        return config
    
    def get_variable_value(self, variable_name: str) -> Any:
        """
        Get value of a specific hyperparameter by name.
        
        Args:
            variable_name: Name of the hyperparameter (e.g., 'epochs', 'lr0')
        
        Returns:
            Current value of the hyperparameter
        
        Raises:
            ValueError: If variable_name doesn't exist
        """
        for category in [
            self.optimization,
            self.batch,
            self.architecture,
            self.augmentation,
            self.regularization,
            self.postprocessing
        ]:
            if hasattr(category, variable_name):
                return getattr(category, variable_name)
        
        raise ValueError(f"Unknown hyperparameter: {variable_name}")
    
    def set_variable_value(self, variable_name: str, value: Any) -> None:
        """
        Set value of a specific hyperparameter by name.
        
        Args:
            variable_name: Name of the hyperparameter
            value: New value to set
        
        Raises:
            ValueError: If variable_name doesn't exist
        """
        for category in [
            self.optimization,
            self.batch,
            self.architecture,
            self.augmentation,
            self.regularization,
            self.postprocessing
        ]:
            if hasattr(category, variable_name):
                setattr(category, variable_name, value)
                return
        
        raise ValueError(f"Unknown hyperparameter: {variable_name}")
    
    @classmethod
    def get_all_variables(cls) -> Dict[str, type]:
        """
        Get dictionary of all available hyperparameters and their types.
        
        Returns:
            Dictionary mapping variable names to their Python types
        """
        variables = {}
        
        for config_class in [
            OptimizationConfig,
            BatchConfig,
            ArchitectureConfig,
            DataAugmentationConfig,
            RegularizationConfig,
            PostProcessingConfig
        ]:
            for field_name, field_info in config_class.model_fields.items():
                variables[field_name] = field_info.annotation
        
        return variables
    
    @classmethod
    def from_yaml_dict(cls, yaml_dict: Dict[str, Any]) -> 'YOLOHyperparameters':
        """
        Create YOLOHyperparameters from YAML dictionary.
        
        Args:
            yaml_dict: Dictionary from YAML file (typically under 'default_hyperparameters' key)
        
        Returns:
            YOLOHyperparameters instance with values from YAML
        
        Note:
            This method properly loads hyperparameters from config.yaml,
            preventing the bug where Pydantic defaults override YAML values.
        """
        # Separate parameters by category
        optimization_params = {}
        batch_params = {}
        architecture_params = {}
        augmentation_params = {}
        regularization_params = {}
        postprocessing_params = {}
        
        # Map parameters to their respective categories
        for key, value in yaml_dict.items():
            # Optimization parameters
            if key in OptimizationConfig.model_fields:
                optimization_params[key] = value
            # Batch parameters
            elif key in BatchConfig.model_fields:
                batch_params[key] = value
            # Architecture parameters
            elif key in ArchitectureConfig.model_fields:
                architecture_params[key] = value
            # Augmentation parameters
            elif key in DataAugmentationConfig.model_fields:
                augmentation_params[key] = value
            # Regularization parameters
            elif key in RegularizationConfig.model_fields:
                regularization_params[key] = value
            # Post-processing parameters
            elif key in PostProcessingConfig.model_fields:
                postprocessing_params[key] = value
        
        # Create nested config objects
        return cls(
            optimization=OptimizationConfig(**optimization_params),
            batch=BatchConfig(**batch_params),
            architecture=ArchitectureConfig(**architecture_params),
            augmentation=DataAugmentationConfig(**augmentation_params),
            regularization=RegularizationConfig(**regularization_params),
            postprocessing=PostProcessingConfig(**postprocessing_params)
        )
    
    model_config = ConfigDict(validate_assignment=True)


class BenchmarkConfig(BaseModel):
    """
    Configuration for fractional benchmark experiments.
    
    Defines how to systematically vary hyperparameters across fractions
    (1/5, 2/5, 3/5, 4/5, 5/5) to investigate their impact.
    """
    
    benchmark_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this benchmark run"
    )
    
    variable_name: str = Field(
        ...,
        description="Hyperparameter to benchmark (e.g., 'epochs', 'lr0')"
    )
    
    max_value: float = Field(
        ...,
        description="Maximum value for the variable (fraction 5/5)"
    )
    
    num_fractions: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of fractions to test"
    )
    
    base_config: YOLOHyperparameters = Field(
        default_factory=YOLOHyperparameters,
        description="Base hyperparameter configuration"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Benchmark start timestamp"
    )
    
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    
    def get_fraction_values(self) -> List[float]:
        """
        Calculate values for each fraction.
        
        Returns:
            List of values [1/n * max, 2/n * max, ..., n/n * max]
        """
        return [
            (i + 1) / self.num_fractions * self.max_value
            for i in range(self.num_fractions)
        ]
    
    def get_fraction_label(self, fraction_index: int) -> str:
        """
        Get human-readable label for a fraction.
        
        Args:
            fraction_index: Index of the fraction (0-based)
        
        Returns:
            Label like "1/5", "2/5", etc.
        """
        return f"{fraction_index + 1}/{self.num_fractions}"
    
    model_config = ConfigDict(validate_assignment=True)


class MultiBenchmarkConfig(BaseModel):
    """
    Configuration for benchmarking multiple variables simultaneously.
    
    Allows systematic exploration of multiple hyperparameters in a
    single experimental run.
    """
    
    benchmark_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this multi-benchmark run"
    )
    
    variables: List[str] = Field(
        ...,
        min_length=1,
        description="List of hyperparameters to benchmark"
    )
    
    max_values: Dict[str, float] = Field(
        ...,
        description="Maximum value for each variable"
    )
    
    num_fractions: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of fractions to test"
    )
    
    base_config: YOLOHyperparameters = Field(
        default_factory=YOLOHyperparameters,
        description="Base hyperparameter configuration"
    )
    
    parallel_jobs: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of parallel training jobs"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Multi-benchmark start timestamp"
    )
    
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    
    @field_validator('max_values')
    @classmethod
    def validate_max_values_match_variables(cls, v: Dict[str, float], info) -> Dict[str, float]:
        """Ensure max_values contains entry for each variable."""
        variables = info.data.get('variables', [])
        if set(v.keys()) != set(variables):
            raise ValueError("max_values keys must match variables list")
        return v
    
    model_config = ConfigDict(validate_assignment=True)


class ProjectConfig(BaseModel):
    """
    Overall project configuration.
    
    Defines paths, project metadata, and global settings.
    """
    
    project_name: str = Field(
        default="yolo_benchmark",
        description="Project name"
    )
    
    model_variant: str = Field(
        default="yolov8m.pt",
        description="YOLO model variant (n/s/m/l/x)"
    )
    
    dataset_path: Path = Field(
        ...,
        description="Path to dataset (data.yaml)"
    )
    
    models_dir: Path = Field(
        default=Path("src/models"),
        description="Directory for saved models"
    )
    
    results_dir: Path = Field(
        default=Path("src/results"),
        description="Directory for results"
    )
    
    device: Optional[str] = Field(
        default=None,
        description="Device (cuda/mps/cpu, None=auto-detect)"
    )
    
    workers: int = Field(
        default=8,
        ge=0,
        description="Number of dataloader workers"
    )
    
    verbose: bool = Field(
        default=True,
        description="Verbose logging"
    )
    
    @field_validator('dataset_path')
    @classmethod
    def validate_dataset_exists(cls, v: Path) -> Path:
        """Validate dataset path exists."""
        if not v.exists():
            raise ValueError(f"Dataset path does not exist: {v}")
        return v
    
    def ensure_directories(self) -> None:
        """Create all necessary directories."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / "checkpoints").mkdir(exist_ok=True)
        (self.models_dir / "best_models").mkdir(exist_ok=True)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "benchmarks").mkdir(exist_ok=True)
        (self.results_dir / "graphs").mkdir(exist_ok=True)
        (self.results_dir / "reports").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
    
    model_config = ConfigDict(validate_assignment=True)
