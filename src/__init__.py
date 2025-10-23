"""
YOLO Hyperparameter Benchmarking System

A comprehensive system for benchmarking and analyzing how YOLO hyperparameters
affect model performance, based on the research article "Influência de Hiperparâmetros no Treinamento do YOLO".

This package provides:
- YOLO training with configurable hyperparameters
- Systematic benchmarking with fractional testing
- Rich CLI interface for parameter configuration
- Bokeh-based visualization of results
- Comprehensive performance analysis

Author: Miguel Batista Pinotti
Research: UNIVERSIDADE PAULISTA - Processamento de Imagem
"""

__version__ = "1.0.0"
__author__ = "Miguel Batista Pinotti"
__email__ = "miguel.pinotti@unip.br"

from .config import YOLOConfig, BenchmarkConfig
from .cli.core import YOLOTrainer, BenchmarkOrchestrator
from .utils import setup_logging, validate_config

__all__ = [
    "YOLOConfig",
    "BenchmarkConfig", 
    "YOLOTrainer",
    "BenchmarkOrchestrator",
    "setup_logging",
    "validate_config"
]
