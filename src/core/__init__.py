"""
Core Module
===========
Módulo principal contendo toda a lógica do sistema.
"""

from .config import Config, BenchmarkConfig
from .trainer import YOLOTrainer
from .benchmark import BenchmarkRunner
from .data_manager import DataManager

__all__ = [
    'Config',
    'BenchmarkConfig',
    'YOLOTrainer',
    'BenchmarkRunner',
    'DataManager'
]
