"""
Rich-based logging system for the YOLO Benchmark System.

Provides beautiful console output and file logging with different
levels and formatting.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text


class BenchmarkLogger:
    """
    Custom logger with Rich console output and file logging.
    
    Provides multiple log levels with beautiful formatting and
    automatic file rotation.
    """
    
    def __init__(
        self,
        name: str = "yolo_benchmark",
        log_dir: Optional[Path] = None,
        console: Optional[Console] = None,
        level: int = logging.INFO
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files (None = no file logging)
            console: Rich console instance (None = create new)
            level: Logging level
        """
        self.name = name
        self.console = console or Console()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Add Rich console handler
        console_handler = RichHandler(
            console=self.console,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True
        )
        console_handler.setLevel(level)
        self.logger.addHandler(console_handler)
        
        # Add file handler if log_dir specified
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # File gets everything
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.info(f"Logging to file: {log_file}")
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
    
    def success(self, message: str) -> None:
        """Log success message (info level with green color)."""
        self.console.print(f"[bold green]✓[/bold green] {message}")
        self.logger.info(f"SUCCESS: {message}")
    
    def section(self, title: str) -> None:
        """
        Print a section header.
        
        Args:
            title: Section title
        """
        self.console.rule(f"[bold cyan]{title}[/bold cyan]")
        self.logger.info(f"=== {title} ===")
    
    def panel(
        self,
        content: str,
        title: Optional[str] = None,
        style: str = "cyan"
    ) -> None:
        """
        Print content in a panel.
        
        Args:
            content: Panel content
            title: Panel title
            style: Panel border style
        """
        panel = Panel(
            content,
            title=title,
            border_style=style,
            expand=False
        )
        self.console.print(panel)
        self.logger.info(f"PANEL [{title}]: {content}")
    
    def banner(self, text: str, style: str = "bold magenta") -> None:
        """
        Print a banner message.
        
        Args:
            text: Banner text
            style: Rich style string
        """
        self.console.print(f"\n[{style}]{text}[/{style}]\n")
        self.logger.info(f"BANNER: {text}")


# Global logger instance (singleton pattern)
_global_logger: Optional[BenchmarkLogger] = None


def get_logger(
    name: str = "yolo_benchmark",
    log_dir: Optional[Path] = None,
    console: Optional[Console] = None,
    level: int = logging.INFO,
    force_new: bool = False
) -> BenchmarkLogger:
    """
    Get global logger instance (singleton).
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        console: Rich console instance
        level: Logging level
        force_new: Force creation of new logger
    
    Returns:
        BenchmarkLogger instance
    """
    global _global_logger
    
    if _global_logger is None or force_new:
        _global_logger = BenchmarkLogger(
            name=name,
            log_dir=log_dir,
            console=console,
            level=level
        )
    
    return _global_logger


def setup_logger(
    log_dir: Path,
    console: Optional[Console] = None,
    level: int = logging.INFO
) -> BenchmarkLogger:
    """
    Setup and configure the global logger.
    
    Args:
        log_dir: Directory for log files
        console: Rich console instance
        level: Logging level
    
    Returns:
        Configured BenchmarkLogger
    """
    return get_logger(
        log_dir=log_dir,
        console=console,
        level=level,
        force_new=True
    )
