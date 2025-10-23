"""
CLI commands for YOLO benchmarking system.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
import json
import yaml
from pathlib import Path
import sys
from datetime import datetime

from config import YOLOConfig, BenchmarkConfig, HyperparameterSpace
from cli.core import BenchmarkOrchestrator
from results_visualizer import ResultsVisualizer

console = Console()


@click.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--dry-run', is_flag=True, help='Show what would be run without executing')
@click.option('--max-workers', '-w', type=int, default=1, help='Maximum parallel workers')
@click.option('--resume', is_flag=True, help='Resume interrupted benchmark')
@click.option('--parameter', '-p', multiple=True, help='Specific parameters to test')
@click.option('--fractions', '-f', help='Comma-separated fractions (e.g., 0.2,0.4,0.6,0.8,1.0)')
def benchmark_cmd(config, dry_run, max_workers, resume, parameter, fractions):
    """Run hyperparameter benchmarking experiments."""
    
    console.print(Panel("🎯 YOLO Hyperparameter Benchmark", style="bold blue"))
    
    # Load configuration
    if config:
        benchmark_config = BenchmarkConfig.from_file(config)
    else:
        benchmark_config = BenchmarkConfig()
    
    # Override parameters if specified
    if parameter:
        benchmark_config.selected_parameters = list(parameter)
    
    if fractions:
        try:
            benchmark_config.fractions = [float(f.strip()) for f in fractions.split(',')]
        except ValueError:
            console.print("[red]Invalid fractions format[/red]")
            return
    
    # Validate configuration
    issues = benchmark_config.validate()
    if issues:
        console.print("[red]Configuration issues:[/red]")
        for issue in issues:
            console.print(f"  • {issue}")
        return
    
    # Show benchmark overview
    resource_summary = benchmark_config.get_resource_summary()
    
    overview_table = Table(title="Benchmark Overview", show_header=True, header_style="bold cyan")
    overview_table.add_column("Metric", style="white")
    overview_table.add_column("Value", style="green")
    
    overview_table.add_row("Total Experiments", str(resource_summary["total_runs"]))
    overview_table.add_row("Parameters", str(resource_summary["parameters"]))
    overview_table.add_row("Fractions", str(resource_summary["fractions"]))
    overview_table.add_row("Repetitions", str(resource_summary["repetitions"]))
    overview_table.add_row("Estimated Time", f"{resource_summary['estimated_time_hours']:.1f} hours")
    overview_table.add_row("Output Directory", benchmark_config.get_output_dir())
    
    console.print(overview_table)
    
    if dry_run:
        console.print("\n[yellow]Dry run mode - showing experiments to run:[/yellow]")
        
        # Generate experiments to show
        orchestrator = BenchmarkOrchestrator(benchmark_config)
        experiments = orchestrator.generate_experiments()
        
        experiments_table = Table(title="Experiments to Run")
        experiments_table.add_column("ID")
        experiments_table.add_column("Parameter")
        experiments_table.add_column("Fraction")
        experiments_table.add_column("Value")
        experiments_table.add_column("Repetition")
        
        for exp in experiments[:20]:  # Show first 20
            experiments_table.add_row(
                exp["id"],
                exp.get("parameter", "baseline"),
                str(exp.get("fraction", "-")),
                str(exp.get("value", "-")),
                str(exp.get("repetition", 0))
            )
        
        if len(experiments) > 20:
            experiments_table.add_row("...", "...", "...", "...", "...")
        
        console.print(experiments_table)
        console.print(f"[dim]Total experiments: {len(experiments)}[/dim]")
        return
    
    # Confirm before running
    if not Confirm.ask("\nProceed with benchmark?"):
        console.print("[yellow]Benchmark cancelled[/yellow]")
        return
    
    # Run benchmark
    try:
        orchestrator = BenchmarkOrchestrator(benchmark_config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Running benchmark...", total=resource_summary["total_runs"])
            
            # Run benchmark
            results = orchestrator.run_benchmark(max_workers=max_workers)
            
            progress.update(task, completed=resource_summary["total_runs"])
        
        # Display results summary
        console.print("\n[bold green]Benchmark completed![/bold green]")
        
        summary_table = Table(title="Results Summary")
        summary_table.add_column("Metric")
        summary_table.add_column("Value")
        
        summary_table.add_row("Successful Experiments", str(results.get("successful_experiments", 0)))
        summary_table.add_row("Failed Experiments", str(results.get("failed_experiments", 0)))
        summary_table.add_row("Total Time", f"{results.get('total_time', 0)/3600:.1f} hours")
        summary_table.add_row("Results Directory", str(results.get("benchmark_id", "")))
        
        console.print(summary_table)
        
        # Save final configuration
        final_config_path = Path(benchmark_config.get_output_dir()) / "final_config.yaml"
        with open(final_config_path, 'w') as f:
            yaml.dump(benchmark_config.to_dict(), f, default_flow_style=False)
        
    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")


@click.command()
@click.option('--create', is_flag=True, help='Create new configuration file')
@click.option('--edit', type=click.Path(exists=True), help='Edit existing configuration')
@click.option('--validate', type=click.Path(exists=True), help='Validate configuration file')
@click.option('--show-defaults', is_flag=True, help='Show default configuration')
def config_cmd(create, edit, validate, show_defaults):
    """Manage benchmark configuration files."""
    
    console.print(Panel("⚙️ Configuration Management", style="bold yellow"))
    
    if show_defaults:
        config = BenchmarkConfig()
        console.print("[bold]Default Configuration:[/bold]")
        console.print(yaml.dump(config.to_dict(), default_flow_style=False))
        return
    
    if create:
        config = BenchmarkConfig()
        
        # Interactive configuration
        console.print("\n[bold blue]Interactive Configuration Setup[/bold blue]")
        
        config.name = Prompt.ask("Benchmark name", default=config.name)
        config.description = Prompt.ask("Description", default=config.description)
        config.dataset_path = Prompt.ask("Dataset path", default=config.dataset_path)
        
        config.max_epochs = int(Prompt.ask("Max epochs", default=str(config.max_epochs)))
        config.max_batch_size = int(Prompt.ask("Max batch size", default=str(config.max_batch_size)))
        config.repetitions = int(Prompt.ask("Repetitions", default=str(config.repetitions)))
        
        # Show parameter space
        param_space = HyperparameterSpace()
        console.print("\n[bold]Available Parameters:[/bold]")
        for param_name in param_space.parameters:
            param = param_space.parameters[param_name]
            console.print(f"  • {param_name}: {param.min_value} - {param.max_value}")
        
        selected_params = Prompt.ask(
            "Parameters to test (comma-separated, or 'all')",
            default=",".join(config.selected_parameters)
        )
        
        if selected_params.lower() != 'all':
            config.selected_parameters = [p.strip() for p in selected_params.split(',')]
        
        # Save configuration
        config_path = Path("config/custom_benchmark.yaml")
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        
        console.print(f"[green]Configuration saved to {config_path}[/green]")
        return
    
    if edit:
        config_path = Path(edit)
        
        try:
            config = BenchmarkConfig.from_file(str(config_path))
            
            console.print(f"[bold]Editing: {config_path}[/bold]")
            
            # Allow editing key parameters
            config.name = Prompt.ask("Benchmark name", default=config.name)
            config.max_epochs = int(Prompt.ask("Max epochs", default=str(config.max_epochs)))
            config.repetitions = int(Prompt.ask("Repetitions", default=str(config.repetitions)))
            
            with open(config_path, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
            
            console.print("[green]Configuration updated[/green]")
            
        except Exception as e:
            console.print(f"[red]Error editing configuration: {e}[/red]")
        return
    
    if validate:
        try:
            config = BenchmarkConfig.from_file(validate)
            issues = config.validate()
            
            if issues:
                console.print("[red]Validation issues:[/red]")
                for issue in issues:
                    console.print(f"  • {issue}")
            else:
                console.print("[green]Configuration is valid[/green]")
                
        except Exception as e:
            console.print(f"[red]Error validating configuration: {e}[/red]")
        return
    
    # Show help
    console.print("[dim]Use --create, --edit, --validate, or --show-defaults[/dim]")


@click.command()
@click.option('--results-dir', type=click.Path(exists=True), help='Results directory to analyze')
@click.option('--output', '-o', type=click.Path(), help='Output directory for analysis')
@click.option('--format', type=click.Choice(['html', 'png', 'svg']), default='html', help='Output format')
@click.option('--interactive', is_flag=True, help='Launch interactive dashboard')
def analyze_cmd(results_dir, output, format, interactive):
    """Analyze and visualize benchmark results."""
    
    console.print(Panel("📊 Results Analysis", style="bold magenta"))
    
    if not results_dir:
        # Find latest results
        results_path = Path("results")
        if results_path.exists():
            dirs = [d for d in results_path.iterdir() if d.is_dir()]
            if dirs:
                results_dir = str(max(dirs, key=lambda d: d.stat().st_mtime))
                console.print(f"[dim]Using latest results: {results_dir}[/dim]")
            else:
                console.print("[red]No results found in results/ directory[/red]")
                return
        else:
            console.print("[red]No results directory found[/red]")
            return
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        console.print(f"[red]Results directory not found: {results_dir}[/red]")
        return
    
    # Check for required files
    summary_file = results_path / "benchmark_summary.json"
    if not summary_file.exists():
        console.print("[red]No benchmark summary found in results directory[/red]")
        return
    
    try:
        # Load results
        with open(summary_file) as f:
            summary = json.load(f)
        
        console.print("[green]Results loaded successfully[/green]")
        
        # Show summary
        summary_table = Table(title="Benchmark Summary")
        summary_table.add_column("Metric")
        summary_table.add_column("Value")
        
        summary_table.add_row("Total Experiments", str(summary.get("total_experiments", 0)))
        summary_table.add_row("Successful", str(summary.get("successful_experiments", 0)))
        summary_table.add_row("Failed", str(summary.get("failed_experiments", 0)))
        summary_table.add_row("Total Time", f"{summary.get('total_time', 0)/3600:.1f} hours")
        
        console.print(summary_table)
        
        # Create visualizer
        visualizer = ResultsVisualizer()
        
        if interactive:
            # Launch interactive dashboard
            console.print("[blue]Launching interactive dashboard...[/blue]")
            visualizer.launch_dashboard(results_path)
        else:
            # Generate static reports
            output_dir = Path(output) if output else results_path / "analysis"
            output_dir.mkdir(exist_ok=True)
            
            console.print("[blue]Generating analysis reports...[/blue]")
            
            # Generate various plots
            plots = visualizer.generate_all_plots(results_path, output_dir, format)
            
            console.print(f"[green]Analysis complete![/green]")
            console.print(f"Results saved to: {output_dir}")
            
            # List generated files
            files_table = Table(title="Generated Files")
            files_table.add_column("File")
            files_table.add_column("Type")
            
            for plot_file in plots:
                files_table.add_row(str(plot_file), format.upper())
            
            console.print(files_table)
            
    except Exception as e:
        console.print(f"[red]Error analyzing results: {e}[/red]")
