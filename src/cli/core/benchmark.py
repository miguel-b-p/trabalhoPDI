"""
Benchmark orchestrator for systematic hyperparameter testing with fractional increments.
"""

import os
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import shutil

from config import YOLOConfig, BenchmarkConfig, HyperparameterSpace
from .trainer import YOLOTrainer
from .metrics import MetricsCollector


class BenchmarkOrchestrator:
    """Orchestrates systematic YOLO hyperparameter benchmarking."""
    
    def __init__(self, benchmark_config: BenchmarkConfig):
        self.config = benchmark_config
        self.parameter_space = HyperparameterSpace()
        self.metrics_collector = MetricsCollector()
        self.logger = logging.getLogger(__name__)
        
        # Setup output directory
        self.output_dir = Path(self.config.get_output_dir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = []
        self.experiments = []
        
    def generate_experiments(self) -> List[Dict[str, Any]]:
        """Generate all experiments for benchmarking."""
        experiments = []
        
        # Create baseline experiment
        baseline = {
            "id": "baseline",
            "name": "baseline",
            "type": "baseline",
            "parameter": None,
            "fraction": 1.0,
            "value": None,
            "param_changes": {}
        }
        experiments.append(baseline)
        
        # Generate fractional experiments for each parameter
        for param_name in self.config.selected_parameters:
            if param_name not in self.parameter_space.parameters:
                continue
                
            param = self.parameter_space.parameters[param_name]
            values = param.get_fractional_values(self.config.fractions)
            
            for fraction, value in zip(self.config.fractions, values):
                experiment = {
                    "id": f"{param_name}_{fraction:.1f}",
                    "name": f"{param_name}_{value}",
                    "type": "parameter_variation",
                    "parameter": param_name,
                    "fraction": fraction,
                    "value": value,
                    "param_changes": {param_name: value}
                }
                experiments.append(experiment)
        
        # Add repetitions
        all_experiments = []
        for exp in experiments:
            for rep in range(self.config.repetitions):
                rep_exp = exp.copy()
                rep_exp["id"] = f"{exp['id']}_rep{rep}"
                rep_exp["repetition"] = rep
                all_experiments.append(rep_exp)
        
        # Shuffle experiments for better statistical distribution
        random.shuffle(all_experiments)
        
        self.experiments = all_experiments
        self.save_experiment_plan()
        
        return all_experiments
    
    def save_experiment_plan(self):
        """Save experiment plan to file."""
        plan_file = self.output_dir / "experiment_plan.json"
        
        plan_data = {
            "config": self.config.to_dict(),
            "experiments": self.experiments,
            "total_experiments": len(self.experiments),
            "generated_at": datetime.now().isoformat()
        }
        
        with open(plan_file, 'w') as f:
            json.dump(plan_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved experiment plan: {plan_file}")
    
    def run_single_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment."""
        experiment_id = experiment["id"]
        experiment_name = experiment["name"]
        
        self.logger.info(f"Starting experiment: {experiment_name}")
        
        # Create experiment directory
        exp_dir = self.output_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Create base configuration
        base_config = YOLOConfig()
        base_config.project = str(exp_dir)
        base_config.name = "experiment"
        
        # Apply parameter changes
        param_changes = experiment["param_changes"]
        
        # Ensure epochs don't exceed maximum
        if "epochs" in param_changes:
            param_changes["epochs"] = min(param_changes["epochs"], self.config.max_epochs)
        
        # Ensure batch size doesn't exceed maximum
        if "batch" in param_changes:
            param_changes["batch"] = min(param_changes["batch"], self.config.max_batch_size)
        
        # Create trainer
        trainer = YOLOTrainer(base_config, experiment_name)
        
        try:
            # Run training
            start_time = time.time()
            results = trainer.train(param_overrides=param_changes)
            
            # Add experiment metadata
            results.update({
                "experiment_id": experiment_id,
                "experiment_metadata": experiment,
                "run_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save individual results
            results_file = exp_dir / "experiment_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Completed experiment: {experiment_name}")
            return results
            
        except Exception as e:
            error_result = {
                "experiment_id": experiment_id,
                "experiment_metadata": experiment,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            }
            
            # Save error results
            results_file = exp_dir / "experiment_error.json"
            with open(results_file, 'w') as f:
                json.dump(error_result, f, indent=2, default=str)
            
            self.logger.error(f"Failed experiment {experiment_name}: {e}")
            return error_result
    
    def run_benchmark(self, max_workers: int = 1) -> Dict[str, Any]:
        """Run complete benchmark with all experiments."""
        self.logger.info("Starting benchmark orchestration")
        
        # Generate experiments
        experiments = self.generate_experiments()
        
        # Validate configuration
        validation_issues = self.config.validate()
        if validation_issues:
            return {
                "status": "validation_failed",
                "issues": validation_issues
            }
        
        # Run experiments
        start_time = time.time()
        
        if max_workers == 1:
            # Sequential execution
            results = []
            for exp in experiments:
                result = self.run_single_experiment(exp)
                results.append(result)
        else:
            # Parallel execution
            results = self._run_parallel(experiments, max_workers)
        
        # Save all results
        self.results = results
        self.save_benchmark_results()
        
        # Generate analysis
        analysis = self.metrics_collector.generate_summary_report(results)
        
        benchmark_summary = {
            "benchmark_id": f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "config": self.config.to_dict(),
            "total_experiments": len(experiments),
            "successful_experiments": len([r for r in results if r.get('status') != 'failed']),
            "failed_experiments": len([r for r in results if r.get('status') == 'failed']),
            "total_time": time.time() - start_time,
            "results": results,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save benchmark summary
        summary_file = self.output_dir / "benchmark_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(benchmark_summary, f, indent=2, default=str)
        
        self.logger.info(f"Benchmark completed. Results saved to {self.output_dir}")
        return benchmark_summary
    
    def _run_parallel(self, experiments: List[Dict], max_workers: int) -> List[Dict]:
        """Run experiments in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_experiment = {
                executor.submit(self.run_single_experiment, exp): exp
                for exp in experiments
            }
            
            # Collect results
            for future in as_completed(future_to_experiment):
                experiment = future_to_experiment[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    error_result = {
                        "experiment_id": experiment["id"],
                        "experiment_metadata": experiment,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "status": "failed"
                    }
                    results.append(error_result)
        
        return results
    
    def save_benchmark_results(self):
        """Save all benchmark results."""
        # Save raw results
        results_file = self.output_dir / "all_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save parameter analysis
        analysis = self.metrics_collector.analyze_parameter_impact(
            [r for r in self.results if r.get('status') == 'completed']
        )
        
        analysis_file = self.output_dir / "parameter_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info("Saved benchmark results")
    
    def get_parameter_impact_analysis(self) -> Dict[str, Any]:
        """Get detailed parameter impact analysis."""
        successful_results = [r for r in self.results if r.get('status') == 'completed']
        return self.metrics_collector.analyze_parameter_impact(successful_results)
    
    def export_results(self, format: str = "csv") -> str:
        """Export results in specified format."""
        if format == "csv":
            return self._export_csv()
        elif format == "json":
            return self._export_json()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_csv(self) -> str:
        """Export results to CSV format."""
        import pandas as pd
        
        # Flatten results for CSV
        flat_results = []
        for result in self.results:
            if result.get('status') == 'completed':
                flat_result = {
                    'experiment_id': result.get('experiment_id'),
                    'parameter': result.get('experiment_metadata', {}).get('parameter'),
                    'fraction': result.get('experiment_metadata', {}).get('fraction'),
                    'value': result.get('experiment_metadata', {}).get('value'),
                    'repetition': result.get('experiment_metadata', {}).get('repetition'),
                    'training_time': result.get('training_time'),
                    'status': result.get('status')
                }
                
                # Add metrics
                metrics = result.get('metrics', {})
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        flat_result[key] = value
                
                flat_results.append(flat_result)
        
        df = pd.DataFrame(flat_results)
        csv_path = self.output_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def _export_json(self) -> str:
        """Export results to JSON format."""
        json_path = self.output_dir / "benchmark_results.json"
        
        export_data = {
            "config": self.config.to_dict(),
            "results": self.results,
            "analysis": self.get_parameter_impact_analysis()
        }
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return str(json_path)
