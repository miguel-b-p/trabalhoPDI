#!/usr/bin/env python3
"""
Main entry point for the YOLO Hyperparameter Benchmark System.

This is the CLI application that researchers use to conduct systematic
hyperparameter experiments as described in the paper:
"Influência de Hiperparâmetros no Treinamento do YOLO" (UNIP, 2025)

Usage:
    python -m src.cli.main
    
    Or with custom configuration:
    python -m src.cli.main --config path/to/config.yaml
"""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.traceback import install

# Install rich tracebacks for better error messages
install(show_locals=True)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cli.core.config import ProjectConfig, YOLOHyperparameters
from src.cli.core.logger import setup_logger
from src.cli.core.utils import load_yaml_config, get_device_info
from src.cli.menu import MainMenu


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="YOLO Hyperparameter Benchmark System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python -m src.cli.main
  
  # Run with custom config
  python -m src.cli.main --config my_config.yaml
  
  # Specify dataset
  python -m src.cli.main --dataset /path/to/data.yaml
  
  # Enable verbose logging
  python -m src.cli.main --verbose

For more information, see the README.md file.
        """
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration YAML file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to dataset YAML file (overrides config)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="YOLO model variant (default: yolov8m.pt)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU usage (disable GPU)"
    )
    
    return parser.parse_args()


def load_project_config(args: argparse.Namespace) -> tuple[ProjectConfig, dict]:
    """
    Load project configuration from file or create default.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Tuple of (ProjectConfig instance, full config_dict from YAML)
    """
    console = Console()
    
    # Try to load from config file
    if args.config.exists():
        try:
            console.print(f"[cyan]Loading configuration from:[/cyan] {args.config}")
            config_dict = load_yaml_config(args.config)
            
            # Create ProjectConfig
            project_config = ProjectConfig(
                project_name=config_dict.get("project_name", "yolo_benchmark"),
                model_variant=args.model or config_dict.get("model_variant", "yolov8m.pt"),
                dataset_path=args.dataset or Path(config_dict["dataset_path"]),
                models_dir=Path(config_dict.get("models_dir", "src/models")),
                results_dir=Path(config_dict.get("results_dir", "src/results")),
                device=None if args.no_gpu else config_dict.get("device"),
                workers=config_dict.get("workers", 8),
                verbose=args.verbose or config_dict.get("verbose", True)
            )
            
            return project_config, config_dict
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load config file: {e}[/yellow]")
            console.print("[yellow]Using default configuration[/yellow]")
            project_config = create_default_config(args)
            return project_config, {}
    else:
        console.print(f"[yellow]Config file not found: {args.config}[/yellow]")
        console.print("[yellow]Using default configuration[/yellow]")
        project_config = create_default_config(args)
        return project_config, {}


def create_default_config(args: argparse.Namespace) -> ProjectConfig:
    """
    Create default project configuration.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Default ProjectConfig
    """
    # Check for dataset
    if not args.dataset:
        console = Console()
        console.print("[bold red]Error:[/bold red] Dataset path not specified!")
        console.print("Please provide --dataset argument or create config.yaml")
        sys.exit(1)
    
    return ProjectConfig(
        project_name="yolo_benchmark",
        model_variant=args.model,
        dataset_path=args.dataset,
        models_dir=Path("src/models"),
        results_dir=Path("src/results"),
        device=None if args.no_gpu else None,  # Auto-detect
        workers=8,
        verbose=args.verbose
    )


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Create console
    console = Console()
    
    try:
        # Load project configuration
        project_config, config_dict = load_project_config(args)
        
        # Ensure directories exist
        project_config.ensure_directories()
        
        # Setup logger
        logger = setup_logger(
            log_dir=project_config.results_dir / "logs",
            console=console,
            level=10 if args.verbose else 20  # DEBUG if verbose, else INFO
        )
        
        logger.section("YOLO Hyperparameter Benchmark System")
        logger.info(f"Project: {project_config.project_name}")
        logger.info(f"Model: {project_config.model_variant}")
        logger.info(f"Dataset: {project_config.dataset_path}")
        
        # Show device info
        device_info = get_device_info()
        logger.info(f"Device: {device_info['device']}")
        if device_info['device'] == 'cuda':
            logger.info(f"GPU: {device_info.get('gpu_name', 'Unknown')}")
        
        # Create and run main menu
        menu = MainMenu(
            project_config=project_config,
            logger=logger,
            console=console,
            yaml_config_dict=config_dict
        )
        
        menu.run()
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Program interrupted by user[/yellow]")
        sys.exit(0)
    
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {str(e)}")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
