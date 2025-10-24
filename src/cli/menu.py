"""
Main menu and navigation for the CLI interface.

Provides the main menu system and navigation flow for the YOLO Benchmark System.
"""

from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.markdown import Markdown

from .core.config import ProjectConfig, YOLOHyperparameters
from .core.logger import BenchmarkLogger
from .core.utils import get_device_info, load_yaml_config, save_yaml_config
from .visualizer import SystemInfoVisualizer, ConfigVisualizer
from .config_menu import ConfigurationMenu
from .benchmark_menu import BenchmarkMenu


class MainMenu:
    """
    Main menu controller for the YOLO Benchmark System.
    
    Implements the primary navigation and orchestration logic.
    """
    
    def __init__(
        self,
        project_config: ProjectConfig,
        logger: BenchmarkLogger,
        console: Optional[Console] = None,
        yaml_config_dict: Optional[dict] = None
    ):
        """
        Initialize main menu.
        
        Args:
            project_config: Project configuration
            logger: Logger instance
            console: Rich console (None = create new)
            yaml_config_dict: Full YAML configuration dictionary (optional)
        """
        self.project_config = project_config
        self.logger = logger
        self.console = console or Console()
        
        # Current hyperparameter configuration
        # Load from YAML if available, otherwise use defaults
        if yaml_config_dict and 'default_hyperparameters' in yaml_config_dict:
            self.current_config = YOLOHyperparameters.from_yaml_dict(
                yaml_config_dict['default_hyperparameters']
            )
            self.logger.info("Loaded hyperparameters from config.yaml")
        else:
            self.current_config = YOLOHyperparameters()
            self.logger.info("Using default hyperparameters")
        
        # Sub-menus
        self.config_menu = ConfigurationMenu(
            self.current_config,
            self.console,
            self.logger
        )
        
        self.benchmark_menu = BenchmarkMenu(
            self.project_config,
            self.console,
            self.logger
        )
        
        # Visualizers
        self.system_viz = SystemInfoVisualizer(self.console)
        self.config_viz = ConfigVisualizer(self.console)
    
    def show_banner(self) -> None:
        """Display welcome banner."""
        banner_text = """
[bold cyan]╔══════════════════════════════════════════════════════════════╗
║        🔬 YOLO Hyperparameter Benchmark System v1.0         ║
║              Pesquisa UNIP - Processamento de Imagem         ║
╚══════════════════════════════════════════════════════════════╝[/bold cyan]

[italic]Investigação Sistemática da Influência de Hiperparâmetros
no Treinamento do YOLOv8 para Detecção de Objetos[/italic]
"""
        self.console.print(banner_text)
        self.console.print()
    
    def show_main_menu(self) -> str:
        """
        Display main menu and get user choice.
        
        Returns:
            User's menu choice
        """
        self.console.print()
        
        menu_panel = Panel(
            """[cyan]1[/cyan] 🎯 Treinar modelo único (configuração customizada)
[cyan]2[/cyan] 📊 Executar Benchmark Fracionado
[cyan]3[/cyan] 📈 Visualizar Resultados (Bokeh)
[cyan]4[/cyan] ⚙️  Configurar Hiperparâmetros
[cyan]5[/cyan] 💾 Salvar/Carregar Configuração
[cyan]6[/cyan] ℹ️  Informações do Sistema
[cyan]7[/cyan] 📖 Ajuda e Documentação
[cyan]0[/cyan] ❌ Sair""",
            title="[bold]Menu Principal[/bold]",
            border_style="blue"
        )
        
        self.console.print(menu_panel)
        
        choice = Prompt.ask(
            "\n[bold cyan]Escolha uma opção[/bold cyan]",
            choices=["0", "1", "2", "3", "4", "5", "6", "7"],
            default="0"
        )
        
        return choice
    
    def run(self) -> None:
        """Run the main menu loop."""
        self.show_banner()
        
        while True:
            try:
                choice = self.show_main_menu()
                
                if choice == "0":
                    if Confirm.ask("\n[yellow]Deseja realmente sair?[/yellow]"):
                        self.console.print("\n[bold green]Até logo! 👋[/bold green]\n")
                        break
                
                elif choice == "1":
                    self._handle_single_training()
                
                elif choice == "2":
                    self._handle_benchmark()
                
                elif choice == "3":
                    self._handle_visualization()
                
                elif choice == "4":
                    self._handle_configuration()
                
                elif choice == "5":
                    self._handle_save_load()
                
                elif choice == "6":
                    self._handle_system_info()
                
                elif choice == "7":
                    self._handle_help()
                
            except KeyboardInterrupt:
                self.console.print("\n\n[yellow]Operação cancelada pelo usuário[/yellow]")
                continue
            except Exception as e:
                self.logger.error(f"Erro: {str(e)}")
                self.console.print(f"\n[bold red]Erro:[/bold red] {str(e)}\n")
                continue
    
    def _handle_single_training(self) -> None:
        """Handle single model training."""
        from .core.trainer import YOLOTrainer
        
        self.console.rule("[bold cyan]Treinamento de Modelo Único[/bold cyan]")
        
        # Show current configuration
        self.console.print("\n[bold]Configuração Atual:[/bold]")
        self.config_viz.show_as_tree(self.current_config)
        
        # Confirm training
        if not Confirm.ask("\n[cyan]Deseja treinar com esta configuração?[/cyan]"):
            return
        
        # Get run name
        run_name = Prompt.ask(
            "[cyan]Nome para este treinamento[/cyan]",
            default="single_training"
        )
        
        # Train
        self.logger.section("Iniciando Treinamento")
        
        trainer = YOLOTrainer(
            model_path=self.project_config.model_variant,
            dataset_path=self.project_config.dataset_path,
            project_config=self.project_config,
            logger=self.logger
        )
        
        try:
            metrics = trainer.train(
                hyperparameters=self.current_config,
                run_name=run_name
            )
            
            self.console.print("\n[bold green]✓ Treinamento concluído com sucesso![/bold green]")
            
            # Show results
            results_table = Table(title="Resultados do Treinamento", border_style="green")
            results_table.add_column("Métrica", style="cyan")
            results_table.add_column("Valor", style="yellow", justify="right")
            
            results_table.add_row("mAP@0.5", f"{metrics.performance.map50:.4f}")
            results_table.add_row("mAP@0.5:0.95", f"{metrics.performance.map50_95:.4f}")
            results_table.add_row("F1-Score", f"{metrics.performance.f1_score:.4f}")
            results_table.add_row("Tempo Total", f"{metrics.operational.total_train_time:.1f}s")
            results_table.add_row("Tempo/Época", f"{metrics.operational.time_per_epoch:.1f}s")
            results_table.add_row("Memória Pico", f"{metrics.operational.memory_peak:.2f}GB")
            
            self.console.print(results_table)
            
        except Exception as e:
            self.console.print(f"\n[bold red]Erro no treinamento:[/bold red] {str(e)}")
        
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
    
    def _handle_benchmark(self) -> None:
        """Handle benchmark execution."""
        self.console.rule("[bold cyan]Benchmark Fracionado[/bold cyan]")
        
        # Use benchmark menu
        self.benchmark_menu.set_base_config(self.current_config)
        self.benchmark_menu.run()
    
    def _handle_visualization(self) -> None:
        """Handle results visualization."""
        from .core.benchmark import BenchmarkEngine
        from ..visualization.bokeh_plots import BokehVisualizer
        
        self.console.rule("[bold cyan]Visualização de Resultados[/bold cyan]")
        
        # Load available benchmarks
        engine = BenchmarkEngine(self.project_config, self.logger)
        all_results = engine.load_benchmark_results()
        
        if not all_results:
            self.console.print("\n[yellow]Nenhum resultado de benchmark encontrado.[/yellow]")
            self.console.print("[dim]Execute um benchmark primeiro (opção 2)[/dim]\n")
            Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
            return
        
        # Show available benchmarks
        table = Table(title="Benchmarks Disponíveis", border_style="blue")
        table.add_column("#", style="cyan", justify="center")
        table.add_column("Variável", style="green")
        table.add_column("Testes", style="yellow", justify="center")
        table.add_column("Data", style="magenta")
        
        for i, result in enumerate(all_results, 1):
            table.add_row(
                str(i),
                result.variable_name,
                str(len(result.tests)),
                result.timestamp.strftime("%Y-%m-%d %H:%M")
            )
        
        self.console.print(table)
        
        # Select benchmark
        choice = Prompt.ask(
            "\n[cyan]Selecione o benchmark para visualizar[/cyan]",
            default="1"
        )
        
        try:
            index = int(choice) - 1
            if 0 <= index < len(all_results):
                selected = all_results[index]
                
                # Generate visualizations
                viz = BokehVisualizer(self.project_config.results_dir / "graphs")
                output_file = viz.create_full_dashboard(selected)
                
                self.console.print(f"\n[bold green]✓ Dashboard gerado:[/bold green] {output_file}")
                self.console.print(f"[dim]Abra o arquivo em um navegador para visualizar[/dim]\n")
            else:
                self.console.print("[red]Opção inválida[/red]")
        except ValueError:
            self.console.print("[red]Entrada inválida[/red]")
        
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
    
    def _handle_configuration(self) -> None:
        """Handle hyperparameter configuration."""
        self.console.rule("[bold cyan]Configuração de Hiperparâmetros[/bold cyan]")
        
        # Use configuration menu
        self.config_menu.config = self.current_config
        self.config_menu.run()
        
        # Update current config from menu
        self.current_config = self.config_menu.config
    
    def _handle_save_load(self) -> None:
        """Handle save/load configuration."""
        self.console.rule("[bold cyan]Salvar/Carregar Configuração[/bold cyan]")
        
        action = Prompt.ask(
            "\n[cyan]Escolha uma ação[/cyan]",
            choices=["salvar", "carregar", "cancelar"],
            default="cancelar"
        )
        
        if action == "salvar":
            filename = Prompt.ask(
                "[cyan]Nome do arquivo[/cyan]",
                default="my_config.yaml"
            )
            
            if not filename.endswith('.yaml'):
                filename += '.yaml'
            
            filepath = Path(filename)
            
            config_dict = self.current_config.to_ultralytics_dict()
            save_yaml_config(config_dict, filepath)
            
            self.console.print(f"\n[bold green]✓ Configuração salva em:[/bold green] {filepath}\n")
        
        elif action == "carregar":
            filename = Prompt.ask(
                "[cyan]Nome do arquivo[/cyan]",
                default="config.yaml"
            )
            
            filepath = Path(filename)
            
            if not filepath.exists():
                self.console.print(f"\n[bold red]Arquivo não encontrado:[/bold red] {filepath}\n")
            else:
                try:
                    config_dict = load_yaml_config(filepath)
                    
                    # Update current config
                    for key, value in config_dict.items():
                        try:
                            self.current_config.set_variable_value(key, value)
                        except ValueError:
                            pass  # Skip unknown keys
                    
                    self.console.print(f"\n[bold green]✓ Configuração carregada de:[/bold green] {filepath}\n")
                    
                    # Show loaded config
                    self.config_viz.show_as_tree(self.current_config)
                
                except Exception as e:
                    self.console.print(f"\n[bold red]Erro ao carregar:[/bold red] {str(e)}\n")
        
        if action != "cancelar":
            Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
    
    def _handle_system_info(self) -> None:
        """Handle system information display."""
        self.console.rule("[bold cyan]Informações do Sistema[/bold cyan]")
        
        # Get device info
        device_info = get_device_info()
        
        # Display
        self.system_viz.show_device_info(device_info)
        self.console.print()
        self.system_viz.show_project_info(self.project_config)
        
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
    
    def _handle_help(self) -> None:
        """Handle help and documentation."""
        self.console.rule("[bold cyan]Ajuda e Documentação[/bold cyan]")
        
        help_text = """
# Sistema de Benchmark de Hiperparâmetros YOLO

## Sobre
Este sistema implementa a metodologia de benchmark fracionado para investigação
sistemática do impacto de hiperparâmetros no treinamento do YOLOv8.

## Benchmark Fracionado
O benchmark fracionado testa um hiperparâmetro em valores correspondentes a
frações (1/N, 2/N, ..., N/N) de um valor máximo, permitindo análise sistemática
do seu impacto em performance e custo computacional.

### Exemplo:
- Variável: `epochs`
- Valor Máximo: 200
- Frações: 5
- Valores Testados: 40, 80, 120, 160, 200 épocas

## Opções do Menu

### 1. Treinar Modelo Único
Treina um modelo YOLO com a configuração atual de hiperparâmetros.

### 2. Executar Benchmark Fracionado
Executa benchmark sistemático de um ou mais hiperparâmetros.

### 3. Visualizar Resultados
Gera visualizações interativas (Bokeh) dos resultados do benchmark.

### 4. Configurar Hiperparâmetros
Interface interativa para ajustar todos os hiperparâmetros do YOLO.

### 5. Salvar/Carregar Configuração
Salva ou carrega configurações de hiperparâmetros em YAML.

### 6. Informações do Sistema
Exibe informações sobre hardware e configuração do projeto.

## Referência
Baseado no artigo: "Influência de Hiperparâmetros no Treinamento do YOLO" (UNIP, 2025)
"""
        
        md = Markdown(help_text)
        self.console.print(md)
        
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
