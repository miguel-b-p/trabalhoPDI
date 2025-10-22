"""
Benchmark module to run parameter variations.
"""

from src.core.config import BenchmarkConfig, get_param_steps
from src.core.trainer import YOLOTrainer
from pathlib import Path
import json
from typing import Dict, List, Any


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.trainer = YOLOTrainer(config)
        self.results_dir = Path(__file__).parent.parent / 'results'
        self.results_dir.mkdir(exist_ok=True)

    def run_benchmark(self, param_to_vary: str) -> List[Dict[str, Any]]:
        """Run benchmark by varying one parameter, others fixed at max."""
        results = []
        steps = get_param_steps(getattr(self.config, f'max_{param_to_vary}'), self.config.quantity)

        fixed_params = {
            'epochs': self.config.max_epochs,
            'batch_size': self.config.max_batch_size,
            'image_size': self.config.max_image_size,
            'learning_rate': self.config.max_learning_rate,
            'data_quantity': self.config.max_data_quantity,  # Note: handling data subset might need custom logic
        }

        for step_value in steps:
            params = fixed_params.copy()
            params[param_to_vary] = step_value

            # For data_quantity, might need to subset data, but for simplicity, assume it's handled elsewhere or skip for now
            if param_to_vary == 'data_quantity':
                # Placeholder: in real, modify data.yaml or use subset
                pass

            print(f"Training with {param_to_vary}={step_value}")
            model_path = self.trainer.train(params)
            metrics = self.trainer.validate(model_path)

            result = {
                'param': param_to_vary,
                'value': step_value,
                'metrics': metrics,
                'model_path': model_path,
            }
            results.append(result)

        # Save results
        results_file = self.results_dir / f'benchmark_{param_to_vary}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        return results

    def run_all_benchmarks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run benchmarks for all parameters."""
        params_to_benchmark = ['epochs', 'batch_size', 'image_size', 'learning_rate', 'data_quantity']
        all_results = {}
        for param in params_to_benchmark:
            all_results[param] = self.run_benchmark(param)
        return all_results
