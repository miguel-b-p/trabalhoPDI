"""
Simple test script to check imports and basic functionality.
"""

import sys
sys.path.insert(0, '/home/mingas/projetos/trabalhoPDI')

from src.core.config import BenchmarkConfig, get_param_steps
from src.core.trainer import YOLOTrainer
from src.core.benchmark import BenchmarkRunner
from src.results.plotter import ResultsPlotter

def test_config():
    config = BenchmarkConfig()
    print("Config:", config)
    steps = get_param_steps(100, 5)
    print("Steps for 100:", steps)
    assert steps == [20, 40, 60, 80, 100]

def test_imports():
    print("All imports successful")

if __name__ == "__main__":
    test_imports()
    test_config()
    print("Basic tests passed")
