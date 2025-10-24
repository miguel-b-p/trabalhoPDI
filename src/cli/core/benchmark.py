"""
Fractional benchmark engine for systematic hyperparameter exploration.

This module implements the core benchmark methodology described in the paper:
"Influência de Hiperparâmetros no Treinamento do YOLO" (UNIP, 2025)

The fractional approach tests hyperparameters at values corresponding to
fractions (1/N, 2/N, ..., N/N) of a maximum value, enabling systematic
investigation of hyperparameter effects.
"""

from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import uuid
from datetime import datetime
import concurrent.futures
from copy import deepcopy

from .config import (
    YOLOHyperparameters,
    ProjectConfig,
    BenchmarkConfig,
    MultiBenchmarkConfig
)
from .trainer import YOLOTrainer
from .metrics import BenchmarkResult, BenchmarkResults, TrainingMetrics
from .logger import BenchmarkLogger, get_logger


class BenchmarkEngine:
    """
    Engine for executing fractional benchmark experiments.
    
    Systematically varies hyperparameters across fractions to investigate
    their impact on model performance and computational cost.
    """
    
    def __init__(
        self,
        project_config: ProjectConfig,
        logger: Optional[BenchmarkLogger] = None
    ):
        """
        Initialize benchmark engine.
        
        Args:
            project_config: Project configuration
            logger: Logger instance (None = use global logger)
        """
        self.project_config = project_config
        self.logger = logger or get_logger()
        
        self.logger.info("Initialized BenchmarkEngine")
    
    def run_single_variable_benchmark(
        self,
        benchmark_config: BenchmarkConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> BenchmarkResults:
        """
        Run benchmark for a single hyperparameter variable.
        
        This is the core fractional benchmark implementation:
        1. Takes a variable and its maximum value
        2. Generates N test configurations at fractions 1/N, 2/N, ..., N/N
        3. Trains model for each configuration
        4. Collects comprehensive metrics
        5. Returns structured results
        
        Args:
            benchmark_config: Configuration for this benchmark
            progress_callback: Optional callback for progress updates
                               callback(test_number, total_tests, status_message)
        
        Returns:
            BenchmarkResults with all test results
        """
        self.logger.section(
            f"Starting Benchmark: {benchmark_config.variable_name}"
        )
        
        # Initialize results container
        results = BenchmarkResults(
            benchmark_id=benchmark_config.benchmark_id,
            variable_name=benchmark_config.variable_name,
            num_fractions=benchmark_config.num_fractions,
            timestamp=benchmark_config.timestamp
        )
        
        # Get fraction values
        fraction_values = benchmark_config.get_fraction_values()
        
        self.logger.info(
            f"Testing {len(fraction_values)} fractions: {fraction_values}"
        )
        
        # Run each test
        for i, value in enumerate(fraction_values):
            test_number = i + 1
            fraction_label = benchmark_config.get_fraction_label(i)
            
            self.logger.section(
                f"Test {test_number}/{len(fraction_values)} - "
                f"Fraction {fraction_label} - "
                f"{benchmark_config.variable_name}={value}"
            )
            
            # Update progress callback
            if progress_callback:
                progress_callback(
                    test_number,
                    len(fraction_values),
                    f"Training with {benchmark_config.variable_name}={value}"
                )
            
            # Create configuration for this test
            test_config = deepcopy(benchmark_config.base_config)
            test_config.set_variable_value(benchmark_config.variable_name, value)
            
            # Train model
            trainer = YOLOTrainer(
                model_path=self.project_config.model_variant,
                dataset_path=self.project_config.dataset_path,
                project_config=self.project_config,
                logger=self.logger
            )
            
            run_name = (
                f"benchmark_{benchmark_config.variable_name}_"
                f"fraction_{test_number}_of_{len(fraction_values)}"
            )
            
            try:
                metrics = trainer.train(
                    hyperparameters=test_config,
                    run_name=run_name
                )
                
                # Create benchmark result
                result = BenchmarkResult(
                    test_number=test_number,
                    fraction=fraction_label,
                    variable_name=benchmark_config.variable_name,
                    variable_value=value,
                    config=test_config.to_ultralytics_dict(),
                    metrics=metrics
                )
                
                results.add_result(result)
                
                self.logger.success(
                    f"Test {test_number} completed - "
                    f"mAP@0.5: {metrics.performance.map50:.4f}"
                )
                
            except Exception as e:
                self.logger.error(f"Test {test_number} failed: {str(e)}")
                # Continue with next test
                continue
        
        # Save results
        self._save_benchmark_results(results)
        
        self.logger.success(
            f"Benchmark completed! Results saved with ID: {results.benchmark_id}"
        )
        
        return results
    
    def run_multi_variable_benchmark(
        self,
        multi_config: MultiBenchmarkConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[BenchmarkResults]:
        """
        Run benchmarks for multiple variables sequentially or in parallel.
        
        Args:
            multi_config: Multi-variable benchmark configuration
            progress_callback: Progress callback
        
        Returns:
            List of BenchmarkResults, one per variable
        """
        self.logger.section(
            f"Starting Multi-Variable Benchmark: {', '.join(multi_config.variables)}"
        )
        
        all_results = []
        
        # Create individual benchmark configs
        benchmark_configs = []
        for variable in multi_config.variables:
            config = BenchmarkConfig(
                benchmark_id=str(uuid.uuid4()),
                variable_name=variable,
                max_value=multi_config.max_values[variable],
                num_fractions=multi_config.num_fractions,
                base_config=deepcopy(multi_config.base_config),
                timestamp=multi_config.timestamp,
                seed=multi_config.seed
            )
            benchmark_configs.append(config)
        
        if multi_config.parallel_jobs > 1:
            # Parallel execution
            self.logger.info(
                f"Running {len(benchmark_configs)} benchmarks in parallel "
                f"(max {multi_config.parallel_jobs} jobs)"
            )
            
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=multi_config.parallel_jobs
            ) as executor:
                futures = {
                    executor.submit(
                        self.run_single_variable_benchmark,
                        config
                    ): config.variable_name
                    for config in benchmark_configs
                }
                
                for future in concurrent.futures.as_completed(futures):
                    variable_name = futures[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        self.logger.success(
                            f"Completed benchmark for: {variable_name}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Benchmark failed for {variable_name}: {str(e)}"
                        )
        else:
            # Sequential execution
            self.logger.info(
                f"Running {len(benchmark_configs)} benchmarks sequentially"
            )
            
            for i, config in enumerate(benchmark_configs):
                self.logger.info(
                    f"Benchmark {i+1}/{len(benchmark_configs)}: "
                    f"{config.variable_name}"
                )
                
                try:
                    result = self.run_single_variable_benchmark(
                        config,
                        progress_callback
                    )
                    all_results.append(result)
                except Exception as e:
                    self.logger.error(
                        f"Benchmark failed for {config.variable_name}: {str(e)}"
                    )
                    continue
        
        self.logger.success(
            f"Multi-variable benchmark completed! "
            f"{len(all_results)}/{len(benchmark_configs)} succeeded"
        )
        
        return all_results
    
    def _save_benchmark_results(self, results: BenchmarkResults) -> None:
        """
        Save benchmark results to JSON file.
        
        Args:
            results: Benchmark results to save
        """
        results_dir = self.project_config.results_dir / "benchmarks"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filename = (
            f"{results.variable_name}_"
            f"{results.timestamp.strftime('%Y%m%d_%H%M%S')}_"
            f"{results.benchmark_id[:8]}.json"
        )
        
        filepath = results_dir / filename
        results.save_to_json(filepath)
        
        self.logger.info(f"Results saved to: {filepath}")
    
    def load_benchmark_results(
        self,
        benchmark_id: Optional[str] = None,
        variable_name: Optional[str] = None
    ) -> List[BenchmarkResults]:
        """
        Load benchmark results from disk.
        
        Args:
            benchmark_id: Specific benchmark ID to load (None = load all)
            variable_name: Filter by variable name
        
        Returns:
            List of BenchmarkResults matching criteria
        """
        results_dir = self.project_config.results_dir / "benchmarks"
        
        if not results_dir.exists():
            self.logger.warning("No benchmark results directory found")
            return []
        
        all_results = []
        
        for json_file in results_dir.glob("*.json"):
            try:
                result = BenchmarkResults.load_from_json(json_file)
                
                # Filter by criteria
                if benchmark_id and result.benchmark_id != benchmark_id:
                    continue
                
                if variable_name and result.variable_name != variable_name:
                    continue
                
                all_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Could not load {json_file}: {e}")
                continue
        
        self.logger.info(f"Loaded {len(all_results)} benchmark results")
        
        return all_results
    
    def compare_benchmarks(
        self,
        benchmark_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple benchmark results.
        
        Args:
            benchmark_ids: List of benchmark IDs to compare
        
        Returns:
            Comparison dictionary with statistics
        """
        benchmarks = []
        for bid in benchmark_ids:
            results = self.load_benchmark_results(benchmark_id=bid)
            if results:
                benchmarks.append(results[0])
        
        if not benchmarks:
            self.logger.warning("No benchmarks found to compare")
            return {}
        
        comparison = {
            'variables': [b.variable_name for b in benchmarks],
            'num_tests': [len(b.tests) for b in benchmarks],
            'best_map50': {},
            'avg_time_per_epoch': {},
            'total_train_time': {}
        }
        
        for benchmark in benchmarks:
            var = benchmark.variable_name
            
            # Find best mAP@0.5
            best_map = max(
                test.metrics.performance.map50
                for test in benchmark.tests
            )
            comparison['best_map50'][var] = best_map
            
            # Average time per epoch across all tests
            avg_time = sum(
                test.metrics.operational.time_per_epoch
                for test in benchmark.tests
            ) / len(benchmark.tests)
            comparison['avg_time_per_epoch'][var] = avg_time
            
            # Total training time
            total_time = sum(
                test.metrics.operational.total_train_time
                for test in benchmark.tests
            )
            comparison['total_train_time'][var] = total_time
        
        return comparison


class QuickBenchmark:
    """
    Quick benchmark utility for rapid hyperparameter testing.
    
    Useful for preliminary exploration before full benchmark runs.
    """
    
    @staticmethod
    def quick_test(
        variable_name: str,
        values: List[float],
        base_config: YOLOHyperparameters,
        project_config: ProjectConfig
    ) -> Dict[float, TrainingMetrics]:
        """
        Quick test of specific values for a hyperparameter.
        
        Args:
            variable_name: Hyperparameter to test
            values: List of values to test
            base_config: Base configuration
            project_config: Project configuration
        
        Returns:
            Dictionary mapping values to metrics
        """
        logger = get_logger()
        logger.section(f"Quick Test: {variable_name}")
        
        results = {}
        
        for value in values:
            logger.info(f"Testing {variable_name}={value}")
            
            test_config = deepcopy(base_config)
            test_config.set_variable_value(variable_name, value)
            
            trainer = YOLOTrainer(
                model_path=project_config.model_variant,
                dataset_path=project_config.dataset_path,
                project_config=project_config,
                logger=logger
            )
            
            try:
                metrics = trainer.train(
                    hyperparameters=test_config,
                    run_name=f"quick_{variable_name}_{value}"
                )
                results[value] = metrics
                
            except Exception as e:
                logger.error(f"Test failed for {value}: {e}")
        
        return results
