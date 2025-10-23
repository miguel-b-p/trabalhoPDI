"""
Main CLI interface using Rich for YOLO hyperparameter benchmarking.
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.live import Live
import sys
from pathlib import Path

from .commands import benchmark_cmd, config_cmd, analyze_cmd

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="YOLO Benchmark")
@click.pass_context
def main(ctx):
    """
    YOLO Hyperparameter Benchmarking System
    
    A comprehensive tool for analyzing how YOLO hyperparameters affect model performance.
    Based on the research: "Influência de Hiperparâmetros no Treinamento do YOLO"
    """
    ctx.ensure_object(dict)


@main.command()
def welcome():
    """Display welcome screen with system overview."""
    
    # Create welcome banner
    welcome_text = Text()
    welcome_text.append("🎯 YOLO Hyperparameter Benchmarking System\n", style="bold cyan")
    welcome_text.append("Based on UNIVERSIDADE PAULISTA Research\n", style="italic")
    welcome_text.append("\"Influência de Hiperparâmetros no Treinamento do YOLO\"\n", style="dim")
    
    # Create features table
    features_table = Table(title="System Features", show_header=True, header_style="bold magenta")
    features_table.add_column("Feature", style="cyan", no_wrap=True)
    features_table.add_column("Description", style="white")
    
    features_table.add_row("📊 Benchmark Engine", "Systematic parameter testing with fractional increments")
    features_table.add_row("🎛️  Rich CLI", "Interactive configuration with beautiful terminal UI")
    features_table.add_row("📈 Bokeh Charts", "Interactive visualization of parameter impacts")
    features_table.add_row("🔍 Real-time Monitoring", "CPU, GPU, and memory usage tracking")
    features_table.add_row("📋 Comprehensive Reports", "Detailed analysis and recommendations")
    
    # Create commands table
    commands_table = Table(title="Available Commands", show_header=True, header_style="bold green")
    commands_table.add_column("Command", style="yellow", no_wrap=True)
    commands_table.add_column("Description", style="white")
    
    commands_table.add_row("benchmark", "Run parameter benchmarking experiments")
    commands_table.add_row("config", "Configure benchmark parameters")
    commands_table.add_row("analyze", "Analyze and visualize results")
    commands_table.add_row("welcome", "Show this welcome screen")
    
    # Display everything
    console.print(Panel(welcome_text, title="Welcome", border_style="blue"))
    console.print()
    console.print(features_table)
    console.print()
    console.print(commands_table)


@main.command()
def quickstart():
    """Interactive quick start guide."""
    
    console.print(Panel("🚀 Quick Start Guide", style="bold green"))
    
    # Step 1: Check system
    console.print("\n[bold blue]Step 1: System Check[/bold blue]")
    
    from ..config import BenchmarkConfig
    config = BenchmarkConfig()
    
    issues = config.validate()
    if issues:
        console.print("[red]❌ Configuration issues found:[/red]")
        for issue in issues:
            console.print(f"  • {issue}")
    else:
        console.print("[green]✅ System configuration valid[/green]")
    
    # Step 2: Show resource estimate
    console.print("\n[bold blue]Step 2: Resource Estimate[/bold blue]")
    
    resource_summary = config.get_resource_summary()
    
    resource_table = Table(show_header=True, header_style="bold cyan")
    resource_table.add_column("Metric")
    resource_table.add_column("Value")
    
    resource_table.add_row("Total Experiments", str(resource_summary["total_runs"]))
    resource_table.add_row("Estimated Time", f"{resource_summary['estimated_time_hours']:.1f} hours")
    resource_table.add_row("Memory Limit", f"{resource_summary['memory_limit_gb']} GB")
    resource_table.add_row("Parallel Jobs", str(resource_summary["max_parallel_jobs"]))
    
    console.print(resource_table)
    
    # Step 3: Interactive setup
    console.print("\n[bold blue]Step 3: Interactive Setup[/bold blue]")
    
    if Confirm.ask("Would you like to customize the benchmark configuration?"):
        # Allow user to modify key parameters
        new_epochs = Prompt.ask(
            "Maximum epochs per experiment",
            default=str(config.max_epochs),
            show_default=True
        )
        try:
            config.max_epochs = int(new_epochs)
        except ValueError:
            console.print("[yellow]Using default epochs[/yellow]")
        
        new_batch = Prompt.ask(
            "Maximum batch size",
            default=str(config.max_batch_size),
            show_default=True
        )
        try:
            config.max_batch_size = int(new_batch)
        except ValueError:
            console.print("[yellow]Using default batch size[/yellow]")
        
        # Save configuration
        config_path = Path("config/custom_benchmark.yaml")
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        
        console.print(f"[green]Configuration saved to {config_path}[/green]")
    
    # Step 4: Run recommendation
    console.print("\n[bold blue]Step 4: Ready to Run[/bold blue]")
    console.print("\nTo start benchmarking:")
    console.print("  [cyan]yolo-benchmark benchmark[/cyan] - Run with default config")
    console.print("  [cyan]yolo-benchmark benchmark --config config/custom_benchmark.yaml[/cyan] - Run with custom config")
    console.print("  [cyan]yolo-benchmark analyze[/cyan] - Analyze existing results")


# Add commands to main group
main.add_command(benchmark_cmd)
main.add_command(config_cmd)
main.add_command(analyze_cmd)


if __name__ == "__main__":
    main()
