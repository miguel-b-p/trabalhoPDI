"""
Rich visualization components for configuration preview and results display.

Provides beautiful displays of configurations, benchmarks, and system information
using Rich components like Syntax, Pretty, Tree, and Tables.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.syntax import Syntax
from rich.pretty import Pretty
from rich.columns import Columns
from rich.text import Text
import yaml
import json

from .core.config import YOLOHyperparameters, ProjectConfig
from .core.metrics import BenchmarkResults


class ConfigVisualizer:
    """
    Visualizer for YOLO hyperparameter configurations.
    
    Provides multiple views: table, tree, YAML syntax highlighting.
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize config visualizer.
        
        Args:
            console: Rich console (None = create new)
        """
        self.console = console or Console()
    
    def show_as_table(
        self,
        config: YOLOHyperparameters,
        title: str = "YOLO Hyperparameters"
    ) -> None:
        """
        Display configuration as a table.
        
        Args:
            config: Hyperparameter configuration
            title: Table title
        """
        table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta",
            border_style="blue"
        )
        
        table.add_column("Category", style="cyan", width=20)
        table.add_column("Parameter", style="green", width=25)
        table.add_column("Value", style="yellow", width=15)
        table.add_column("Description", style="white", width=40)
        
        # Optimization
        for field_name, field in config.optimization.model_fields.items():
            value = getattr(config.optimization, field_name)
            table.add_row(
                "Optimization",
                field_name,
                str(value),
                field.description or ""
            )
        
        # Batch
        for field_name, field in config.batch.model_fields.items():
            value = getattr(config.batch, field_name)
            table.add_row(
                "Batch",
                field_name,
                str(value),
                field.description or ""
            )
        
        # Architecture
        for field_name, field in config.architecture.model_fields.items():
            value = getattr(config.architecture, field_name)
            table.add_row(
                "Architecture",
                field_name,
                str(value),
                field.description or ""
            )
        
        # Augmentation
        for field_name, field in config.augmentation.model_fields.items():
            value = getattr(config.augmentation, field_name)
            table.add_row(
                "Augmentation",
                field_name,
                str(value),
                field.description or ""
            )
        
        # Regularization
        for field_name, field in config.regularization.model_fields.items():
            value = getattr(config.regularization, field_name)
            table.add_row(
                "Regularization",
                field_name,
                str(value),
                field.description or ""
            )
        
        # Post-processing
        for field_name, field in config.postprocessing.model_fields.items():
            value = getattr(config.postprocessing, field_name)
            table.add_row(
                "Post-processing",
                field_name,
                str(value),
                field.description or ""
            )
        
        self.console.print(table)
    
    def show_as_tree(
        self,
        config: YOLOHyperparameters,
        title: str = "YOLO Configuration"
    ) -> None:
        """
        Display configuration as a tree.
        
        Args:
            config: Hyperparameter configuration
            title: Tree title
        """
        tree = Tree(f"[bold cyan]{title}[/bold cyan]")
        
        # Optimization branch
        opt_branch = tree.add("[bold yellow]Optimization[/bold yellow]")
        for field_name in config.optimization.model_fields.keys():
            value = getattr(config.optimization, field_name)
            opt_branch.add(f"[green]{field_name}[/green]: [white]{value}[/white]")
        
        # Batch branch
        batch_branch = tree.add("[bold yellow]Batch[/bold yellow]")
        for field_name in config.batch.model_fields.keys():
            value = getattr(config.batch, field_name)
            batch_branch.add(f"[green]{field_name}[/green]: [white]{value}[/white]")
        
        # Architecture branch
        arch_branch = tree.add("[bold yellow]Architecture[/bold yellow]")
        for field_name in config.architecture.model_fields.keys():
            value = getattr(config.architecture, field_name)
            arch_branch.add(f"[green]{field_name}[/green]: [white]{value}[/white]")
        
        # Augmentation branch
        aug_branch = tree.add("[bold yellow]Augmentation[/bold yellow]")
        for field_name in config.augmentation.model_fields.keys():
            value = getattr(config.augmentation, field_name)
            aug_branch.add(f"[green]{field_name}[/green]: [white]{value}[/white]")
        
        # Regularization branch
        reg_branch = tree.add("[bold yellow]Regularization[/bold yellow]")
        for field_name in config.regularization.model_fields.keys():
            value = getattr(config.regularization, field_name)
            reg_branch.add(f"[green]{field_name}[/green]: [white]{value}[/white]")
        
        # Post-processing branch
        post_branch = tree.add("[bold yellow]Post-processing[/bold yellow]")
        for field_name in config.postprocessing.model_fields.keys():
            value = getattr(config.postprocessing, field_name)
            post_branch.add(f"[green]{field_name}[/green]: [white]{value}[/white]")
        
        self.console.print(tree)
    
    def show_as_yaml(
        self,
        config: YOLOHyperparameters,
        title: str = "YOLO Configuration (YAML)"
    ) -> None:
        """
        Display configuration as syntax-highlighted YAML.
        
        Args:
            config: Hyperparameter configuration
            title: Display title
        """
        config_dict = config.to_ultralytics_dict()
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        
        syntax = Syntax(
            yaml_str,
            "yaml",
            theme="monokai",
            line_numbers=True,
            word_wrap=True
        )
        
        panel = Panel(
            syntax,
            title=f"[bold]{title}[/bold]",
            border_style="cyan"
        )
        
        self.console.print(panel)
    
    def show_comparison(
        self,
        configs: Dict[str, YOLOHyperparameters],
        highlight_differences: bool = True
    ) -> None:
        """
        Display side-by-side comparison of multiple configurations.
        
        Args:
            configs: Dictionary mapping names to configurations
            highlight_differences: Highlight different values
        """
        if not configs:
            self.console.print("[yellow]No configurations to compare[/yellow]")
            return
        
        table = Table(
            title="Configuration Comparison",
            show_header=True,
            header_style="bold magenta",
            border_style="blue"
        )
        
        table.add_column("Parameter", style="cyan", width=25)
        for name in configs.keys():
            table.add_column(name, style="green", width=15)
        
        # Get all parameters from first config
        first_config = list(configs.values())[0]
        all_params = first_config.to_ultralytics_dict()
        
        for param in all_params.keys():
            values = []
            for config in configs.values():
                config_dict = config.to_ultralytics_dict()
                values.append(config_dict.get(param, "N/A"))
            
            # Check if all values are the same
            all_same = len(set(str(v) for v in values)) == 1
            
            row_values = [param]
            for value in values:
                if highlight_differences and not all_same:
                    row_values.append(f"[bold yellow]{value}[/bold yellow]")
                else:
                    row_values.append(str(value))
            
            table.add_row(*row_values)
        
        self.console.print(table)


class BenchmarkVisualizer:
    """
    Visualizer for benchmark results.
    
    Displays benchmark results in various formats: tables, summaries, rankings.
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize benchmark visualizer.
        
        Args:
            console: Rich console (None = create new)
        """
        self.console = console or Console()
    
    def show_results_table(
        self,
        results: BenchmarkResults,
        sort_by: str = "fraction"
    ) -> None:
        """
        Display benchmark results as a table.
        
        Args:
            results: Benchmark results
            sort_by: Column to sort by
        """
        table = Table(
            title=f"Benchmark Results: {results.variable_name}",
            show_header=True,
            header_style="bold magenta",
            border_style="blue"
        )
        
        table.add_column("Fraction", style="cyan", justify="center")
        table.add_column("Value", style="green", justify="right")
        table.add_column("mAP@0.5", style="yellow", justify="right")
        table.add_column("mAP@0.5:0.95", style="yellow", justify="right")
        table.add_column("F1-Score", style="yellow", justify="right")
        table.add_column("Time/Epoch", style="magenta", justify="right")
        table.add_column("Total Time", style="magenta", justify="right")
        table.add_column("Memory Peak", style="red", justify="right")
        
        for test in results.tests:
            table.add_row(
                test.fraction,
                f"{test.variable_value:.4f}",
                f"{test.metrics.performance.map50:.4f}",
                f"{test.metrics.performance.map50_95:.4f}",
                f"{test.metrics.performance.f1_score:.4f}",
                f"{test.metrics.operational.time_per_epoch:.1f}s",
                f"{test.metrics.operational.total_train_time:.0f}s",
                f"{test.metrics.operational.memory_peak:.2f}GB"
            )
        
        self.console.print(table)
    
    def show_summary(self, results: BenchmarkResults) -> None:
        """
        Display summary statistics of benchmark.
        
        Args:
            results: Benchmark results
        """
        # Calculate statistics
        map50_values = [t.metrics.performance.map50 for t in results.tests]
        times = [t.metrics.operational.total_train_time for t in results.tests]
        
        best_test = max(results.tests, key=lambda t: t.metrics.performance.map50)
        fastest_test = min(results.tests, key=lambda t: t.metrics.operational.total_train_time)
        
        # Create summary table
        summary = Table(
            title=f"Benchmark Summary: {results.variable_name}",
            show_header=False,
            border_style="green",
            padding=(0, 2)
        )
        
        summary.add_column(style="cyan", justify="right")
        summary.add_column(style="yellow")
        
        summary.add_row("Variable:", results.variable_name)
        summary.add_row("Tests:", str(len(results.tests)))
        summary.add_row("Best mAP@0.5:", f"{max(map50_values):.4f} (fraction {best_test.fraction})")
        summary.add_row("Average mAP@0.5:", f"{sum(map50_values)/len(map50_values):.4f}")
        summary.add_row("Total Time:", f"{sum(times):.0f}s ({sum(times)/3600:.1f}h)")
        summary.add_row("Fastest Test:", f"{fastest_test.fraction} ({fastest_test.metrics.operational.total_train_time:.0f}s)")
        
        self.console.print(summary)
    
    def show_ranking(
        self,
        results: BenchmarkResults,
        metric: str = "map50"
    ) -> None:
        """
        Display ranked results by specified metric.
        
        Args:
            results: Benchmark results
            metric: Metric to rank by
        """
        # Sort tests by metric
        if metric == "map50":
            sorted_tests = sorted(
                results.tests,
                key=lambda t: t.metrics.performance.map50,
                reverse=True
            )
            metric_name = "mAP@0.5"
        elif metric == "f1_score":
            sorted_tests = sorted(
                results.tests,
                key=lambda t: t.metrics.performance.f1_score,
                reverse=True
            )
            metric_name = "F1-Score"
        else:
            sorted_tests = results.tests
            metric_name = metric
        
        table = Table(
            title=f"Ranking by {metric_name}",
            show_header=True,
            header_style="bold magenta",
            border_style="blue"
        )
        
        table.add_column("Rank", style="cyan", justify="center")
        table.add_column("Fraction", style="green")
        table.add_column("Value", style="yellow", justify="right")
        table.add_column(metric_name, style="bold yellow", justify="right")
        
        for i, test in enumerate(sorted_tests, 1):
            if metric == "map50":
                value = f"{test.metrics.performance.map50:.4f}"
            elif metric == "f1_score":
                value = f"{test.metrics.performance.f1_score:.4f}"
            else:
                value = "N/A"
            
            # Highlight top 3
            rank_str = str(i)
            if i == 1:
                rank_str = f"[bold gold1]🥇 {i}[/bold gold1]"
            elif i == 2:
                rank_str = f"[bold silver]🥈 {i}[/bold silver]"
            elif i == 3:
                rank_str = f"[bold orange3]🥉 {i}[/bold orange3]"
            
            table.add_row(
                rank_str,
                test.fraction,
                f"{test.variable_value:.4f}",
                value
            )
        
        self.console.print(table)


class SystemInfoVisualizer:
    """
    Visualizer for system information and device details.
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize system info visualizer.
        
        Args:
            console: Rich console (None = create new)
        """
        self.console = console or Console()
    
    def show_device_info(self, device_info: Dict[str, Any]) -> None:
        """
        Display device information.
        
        Args:
            device_info: Device information dictionary
        """
        table = Table(
            title="System Information",
            show_header=False,
            border_style="cyan",
            padding=(0, 2)
        )
        
        table.add_column(style="cyan", justify="right")
        table.add_column(style="yellow")
        
        for key, value in device_info.items():
            table.add_row(f"{key}:", str(value))
        
        panel = Panel(
            table,
            title="[bold]Hardware Configuration[/bold]",
            border_style="green"
        )
        
        self.console.print(panel)
    
    def show_project_info(self, project_config: ProjectConfig) -> None:
        """
        Display project configuration.
        
        Args:
            project_config: Project configuration
        """
        table = Table(
            title="Project Configuration",
            show_header=False,
            border_style="blue",
            padding=(0, 2)
        )
        
        table.add_column(style="cyan", justify="right")
        table.add_column(style="yellow")
        
        table.add_row("Project Name:", project_config.project_name)
        table.add_row("Model Variant:", project_config.model_variant)
        table.add_row("Dataset Path:", str(project_config.dataset_path))
        table.add_row("Models Directory:", str(project_config.models_dir))
        table.add_row("Results Directory:", str(project_config.results_dir))
        table.add_row("Workers:", str(project_config.workers))
        
        if project_config.device:
            table.add_row("Device:", project_config.device)
        
        self.console.print(table)
