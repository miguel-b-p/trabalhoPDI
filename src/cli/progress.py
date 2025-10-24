"""
Rich progress tracking components for training and benchmarking.

Provides beautiful real-time progress displays with metrics, resource usage,
and estimated completion times.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text


class TrainingProgressDisplay:
    """
    Real-time training progress display with Rich.
    
    Shows:
    - Current epoch progress
    - Loss metrics
    - Resource usage (GPU/CPU, memory)
    - ETA
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize training progress display.
        
        Args:
            console: Rich console (None = create new)
        """
        self.console = console or Console()
        
        # Create progress bars
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        )
        
        self.task_id: Optional[int] = None
        self.live: Optional[Live] = None
        
        # Metrics
        self.current_metrics: Dict[str, Any] = {}
    
    def start(
        self,
        total_epochs: int,
        description: str = "Training"
    ) -> None:
        """
        Start progress display.
        
        Args:
            total_epochs: Total number of epochs
            description: Task description
        """
        self.task_id = self.progress.add_task(
            description,
            total=total_epochs
        )
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="progress"),
            Layout(name="metrics", size=8)
        )
        
        layout["progress"].update(self.progress)
        layout["metrics"].update(self._create_metrics_panel())
        
        self.live = Live(
            layout,
            console=self.console,
            refresh_per_second=4
        )
        self.live.start()
    
    def update(
        self,
        current_epoch: int,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update progress display.
        
        Args:
            current_epoch: Current epoch number
            metrics: Dictionary of metrics to display
        """
        if self.task_id is not None:
            self.progress.update(self.task_id, completed=current_epoch)
        
        if metrics:
            self.current_metrics.update(metrics)
        
        # Update layout
        if self.live:
            layout = self.live.renderable
            layout["metrics"].update(self._create_metrics_panel())
    
    def stop(self) -> None:
        """Stop progress display."""
        if self.live:
            self.live.stop()
    
    def _create_metrics_panel(self) -> Panel:
        """Create panel with current metrics."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="magenta")
        
        # Add metrics
        for key, value in self.current_metrics.items():
            if isinstance(value, float):
                table.add_row(f"{key}:", f"{value:.4f}")
            else:
                table.add_row(f"{key}:", str(value))
        
        return Panel(
            table,
            title="[bold]Current Metrics[/bold]",
            border_style="blue"
        )


class BenchmarkProgressDisplay:
    """
    Progress display for benchmark experiments.
    
    Shows:
    - Overall benchmark progress (which test)
    - Current training progress
    - Test configuration
    - Summary of completed tests
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize benchmark progress display.
        
        Args:
            console: Rich console (None = create new)
        """
        self.console = console or Console()
        
        # Create progress bars
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        )
        
        self.benchmark_task_id: Optional[int] = None
        self.training_task_id: Optional[int] = None
        
        self.live: Optional[Live] = None
        self.current_config: Dict[str, Any] = {}
        self.completed_tests: list = []
    
    def start(
        self,
        total_tests: int,
        variable_name: str
    ) -> None:
        """
        Start benchmark progress display.
        
        Args:
            total_tests: Total number of benchmark tests
            variable_name: Name of variable being benchmarked
        """
        self.benchmark_task_id = self.progress.add_task(
            f"Benchmark: {variable_name}",
            total=total_tests
        )
        
        self.training_task_id = self.progress.add_task(
            "Current Training",
            total=100,
            visible=False
        )
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=6),
            Layout(name="config", size=8),
            Layout(name="results", size=10)
        )
        
        layout["header"].update(
            Panel(
                f"[bold cyan]Fractional Benchmark: {variable_name}[/bold cyan]",
                style="blue"
            )
        )
        layout["progress"].update(self.progress)
        layout["config"].update(self._create_config_panel())
        layout["results"].update(self._create_results_panel())
        
        self.live = Live(
            layout,
            console=self.console,
            refresh_per_second=2
        )
        self.live.start()
    
    def update_test(
        self,
        test_number: int,
        config: Dict[str, Any]
    ) -> None:
        """
        Update to new test.
        
        Args:
            test_number: Current test number
            config: Test configuration
        """
        if self.benchmark_task_id is not None:
            self.progress.update(
                self.benchmark_task_id,
                completed=test_number - 1
            )
        
        self.current_config = config
        
        # Update layout
        if self.live:
            layout = self.live.renderable
            layout["config"].update(self._create_config_panel())
    
    def update_training(
        self,
        current_epoch: int,
        total_epochs: int
    ) -> None:
        """
        Update training progress within current test.
        
        Args:
            current_epoch: Current epoch
            total_epochs: Total epochs
        """
        if self.training_task_id is not None:
            progress_pct = int((current_epoch / total_epochs) * 100)
            self.progress.update(
                self.training_task_id,
                completed=progress_pct,
                visible=True,
                description=f"Training: Epoch {current_epoch}/{total_epochs}"
            )
    
    def complete_test(
        self,
        test_number: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Mark test as completed and record results.
        
        Args:
            test_number: Test number
            metrics: Test metrics
        """
        self.completed_tests.append({
            'test': test_number,
            'metrics': metrics
        })
        
        if self.benchmark_task_id is not None:
            self.progress.update(
                self.benchmark_task_id,
                completed=test_number
            )
        
        # Hide training progress
        if self.training_task_id is not None:
            self.progress.update(
                self.training_task_id,
                visible=False
            )
        
        # Update results panel
        if self.live:
            layout = self.live.renderable
            layout["results"].update(self._create_results_panel())
    
    def stop(self) -> None:
        """Stop progress display."""
        if self.live:
            self.live.stop()
    
    def _create_config_panel(self) -> Panel:
        """Create panel with current test configuration."""
        if not self.current_config:
            return Panel("Waiting...", title="Current Configuration")
        
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="yellow")
        
        for key, value in self.current_config.items():
            if isinstance(value, float):
                table.add_row(f"{key}:", f"{value:.4f}")
            else:
                table.add_row(f"{key}:", str(value))
        
        return Panel(
            table,
            title="[bold]Current Test Configuration[/bold]",
            border_style="green"
        )
    
    def _create_results_panel(self) -> Panel:
        """Create panel with completed test results."""
        if not self.completed_tests:
            return Panel(
                "No tests completed yet",
                title="Completed Tests"
            )
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Test", style="cyan", justify="center")
        table.add_column("mAP@0.5", justify="right")
        table.add_column("F1-Score", justify="right")
        table.add_column("Time/Epoch", justify="right")
        
        for test in self.completed_tests[-5:]:  # Show last 5
            metrics = test['metrics']
            table.add_row(
                str(test['test']),
                f"{metrics.get('mAP@0.5', 0):.4f}",
                f"{metrics.get('f1_score', 0):.4f}",
                f"{metrics.get('time_per_epoch', 0):.1f}s"
            )
        
        return Panel(
            table,
            title="[bold]Latest Results[/bold]",
            border_style="blue"
        )


def create_simple_progress() -> Progress:
    """
    Create a simple progress bar for general use.
    
    Returns:
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn()
    )
