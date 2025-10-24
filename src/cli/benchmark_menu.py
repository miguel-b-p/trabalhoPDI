"""
Benchmark menu for configuring and executing fractional benchmarks.

Implements the interactive interface for the fractional benchmark methodology.
"""

from typing import Optional, List, Dict
from pathlib import Path
from copy import deepcopy
from rich.console import Console
from rich.prompt import Prompt, FloatPrompt, IntPrompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

from .core.config import (
    ProjectConfig,
    YOLOHyperparameters,
    BenchmarkConfig,
    MultiBenchmarkConfig
)
from .core.benchmark import BenchmarkEngine
from .core.logger import BenchmarkLogger
from .progress import BenchmarkProgressDisplay
from .visualizer import ConfigVisualizer, BenchmarkVisualizer


class BenchmarkMenu:
    """
    Interactive menu for benchmark configuration and execution.
    
    Implements the fractional benchmark interface with step-by-step
    configuration and real-time progress tracking.
    """
    
    def __init__(
        self,
        project_config: ProjectConfig,
        console: Optional[Console] = None,
        logger: Optional[BenchmarkLogger] = None
    ):
        """
        Initialize benchmark menu.
        
        Args:
            project_config: Project configuration
            console: Rich console (None = create new)
            logger: Logger instance
        """
        self.project_config = project_config
        self.console = console or Console()
        self.logger = logger
        
        self.base_config = YOLOHyperparameters()
        self.benchmark_engine = BenchmarkEngine(project_config, logger)
        
        self.config_viz = ConfigVisualizer(console)
        self.benchmark_viz = BenchmarkVisualizer(console)
    
    def set_base_config(self, config: YOLOHyperparameters) -> None:
        """
        Set base configuration for benchmarks.
        
        Args:
            config: Base hyperparameter configuration
        """
        self.base_config = deepcopy(config)
    
    def run(self) -> None:
        """Run the benchmark menu loop."""
        while True:
            choice = self._show_menu()
            
            if choice == "0":
                break
            elif choice == "1":
                self._run_single_variable_benchmark()
            elif choice == "2":
                self._run_multi_variable_benchmark()
            elif choice == "3":
                self._view_benchmark_results()
            elif choice == "4":
                self._compare_benchmarks()
    
    def _show_menu(self) -> str:
        """
        Show benchmark menu and get choice.
        
        Returns:
            User's choice
        """
        self.console.print()
        
        menu_panel = Panel(
            """[cyan]1[/cyan] 📊 Executar Benchmark de Variável Única
[cyan]2[/cyan] 📈 Executar Benchmark Multi-Variável
[cyan]3[/cyan] 👁️  Visualizar Resultados Salvos
[cyan]4[/cyan] ⚖️  Comparar Benchmarks
[cyan]0[/cyan] ◀️  Voltar""",
            title="[bold]Menu de Benchmark[/bold]",
            border_style="blue"
        )
        
        self.console.print(menu_panel)
        
        choice = Prompt.ask(
            "\n[bold cyan]Escolha uma opção[/bold cyan]",
            choices=["0", "1", "2", "3", "4"],
            default="0"
        )
        
        return choice
    
    def _run_single_variable_benchmark(self) -> None:
        """Configure and run single variable benchmark."""
        self.console.rule("[bold cyan]Benchmark de Variável Única[/bold cyan]")
        
        # Step 1: Select variable
        variable_name = self._select_variable()
        if not variable_name:
            return
        
        # Step 2: Configure max value
        max_value = self._configure_max_value(variable_name)
        if max_value is None:
            return
        
        # Step 3: Configure number of fractions
        num_fractions = IntPrompt.ask(
            "\n[cyan]Número de frações a testar[/cyan]",
            default=5
        )
        
        # Step 4: Show preview
        self._show_benchmark_preview(variable_name, max_value, num_fractions)
        
        # Step 5: Confirm execution
        if not Confirm.ask("\n[cyan]Executar este benchmark?[/cyan]"):
            self.console.print("[yellow]Benchmark cancelado[/yellow]")
            return
        
        # Step 6: Create benchmark config
        benchmark_config = BenchmarkConfig(
            variable_name=variable_name,
            max_value=max_value,
            num_fractions=num_fractions,
            base_config=deepcopy(self.base_config)
        )
        
        # Step 7: Execute benchmark
        self._execute_benchmark(benchmark_config)
    
    def _run_multi_variable_benchmark(self) -> None:
        """Configure and run multi-variable benchmark."""
        self.console.rule("[bold cyan]Benchmark Multi-Variável[/bold cyan]")
        
        # Step 1: Select variables
        variables = self._select_multiple_variables()
        if not variables:
            return
        
        # Step 2: Configure max values for each variable
        max_values = {}
        for var in variables:
            max_val = self._configure_max_value(var)
            if max_val is None:
                return
            max_values[var] = max_val
        
        # Step 3: Configure number of fractions
        num_fractions = IntPrompt.ask(
            "\n[cyan]Número de frações para cada variável[/cyan]",
            default=5
        )
        
        # Step 4: Configure parallel jobs
        parallel_jobs = IntPrompt.ask(
            "\n[cyan]Número de jobs paralelos (1 = sequencial)[/cyan]",
            default=1
        )
        
        # Step 5: Show preview
        self.console.print("\n[bold]Resumo do Benchmark Multi-Variável:[/bold]")
        
        summary_table = Table(border_style="green")
        summary_table.add_column("Variável", style="cyan")
        summary_table.add_column("Valor Máximo", style="yellow", justify="right")
        summary_table.add_column("Testes", style="magenta", justify="center")
        
        for var, max_val in max_values.items():
            summary_table.add_row(var, f"{max_val:.4f}", str(num_fractions))
        
        self.console.print(summary_table)
        
        total_tests = len(variables) * num_fractions
        self.console.print(f"\n[yellow]Total de testes:[/yellow] {total_tests}")
        
        # Step 6: Confirm execution
        if not Confirm.ask("\n[cyan]Executar este benchmark?[/cyan]"):
            self.console.print("[yellow]Benchmark cancelado[/yellow]")
            return
        
        # Step 7: Create multi-benchmark config
        multi_config = MultiBenchmarkConfig(
            variables=variables,
            max_values=max_values,
            num_fractions=num_fractions,
            base_config=deepcopy(self.base_config),
            parallel_jobs=parallel_jobs
        )
        
        # Step 8: Execute benchmark
        self._execute_multi_benchmark(multi_config)
    
    def _select_variable(self) -> Optional[str]:
        """
        Let user select a variable to benchmark.
        
        Returns:
            Selected variable name or None if cancelled
        """
        # Get all available variables
        all_vars = YOLOHyperparameters.get_all_variables()
        
        # Organize by category
        categories = {
            "Otimização": ["lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "optimizer"],
            "Batch": ["batch", "accumulate"],
            "Arquitetura": ["imgsz", "epochs", "patience"],
            "Augmentação": ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
                           "mosaic", "mixup", "fliplr", "flipud"],
            "Regularização": ["label_smoothing", "dropout"],
            "Pós-processamento": ["conf", "iou"]
        }
        
        # Show menu
        self.console.print("\n[bold]Selecione a variável para benchmarkar:[/bold]\n")
        
        var_list = []
        index = 1
        
        for category, vars_in_cat in categories.items():
            self.console.print(f"[bold cyan]{category}:[/bold cyan]")
            for var in vars_in_cat:
                if var in all_vars:
                    self.console.print(f"  [{index}] {var}")
                    var_list.append(var)
                    index += 1
            self.console.print()
        
        self.console.print(f"  [0] Cancelar\n")
        
        choice = Prompt.ask(
            "[cyan]Escolha o número da variável[/cyan]",
            default="0"
        )
        
        try:
            choice_idx = int(choice)
            if choice_idx == 0:
                return None
            if 1 <= choice_idx <= len(var_list):
                return var_list[choice_idx - 1]
        except ValueError:
            pass
        
        self.console.print("[red]Opção inválida[/red]")
        return None
    
    def _select_multiple_variables(self) -> List[str]:
        """
        Let user select multiple variables to benchmark.
        
        Returns:
            List of selected variable names
        """
        selected = []
        
        self.console.print("\n[bold]Selecione variáveis (digite números separados por vírgula):[/bold]")
        self.console.print("[dim]Exemplo: 1,5,8[/dim]\n")
        
        # Get all available variables
        all_vars = YOLOHyperparameters.get_all_variables()
        var_list = list(all_vars.keys())
        
        # Show options
        for i, var in enumerate(var_list, 1):
            self.console.print(f"  [{i}] {var}")
        
        choice = Prompt.ask(
            "\n[cyan]Variáveis (números separados por vírgula)[/cyan]"
        )
        
        try:
            indices = [int(x.strip()) for x in choice.split(",")]
            for idx in indices:
                if 1 <= idx <= len(var_list):
                    selected.append(var_list[idx - 1])
        except ValueError:
            self.console.print("[red]Formato inválido[/red]")
            return []
        
        if selected:
            self.console.print(f"\n[green]Selecionadas:[/green] {', '.join(selected)}")
        
        return selected
    
    def _configure_max_value(self, variable_name: str) -> Optional[float]:
        """
        Configure maximum value for a variable.
        
        Args:
            variable_name: Variable name
        
        Returns:
            Maximum value or None if cancelled
        """
        # Get current value
        current_value = self.base_config.get_variable_value(variable_name)
        
        self.console.print(f"\n[bold]Configurando valor máximo para:[/bold] [cyan]{variable_name}[/cyan]")
        self.console.print(f"[dim]Valor atual na configuração base: {current_value}[/dim]")
        
        # Suggest reasonable defaults based on variable
        suggestions = {
            "epochs": 200,
            "batch": 64,
            "lr0": 0.1,
            "imgsz": 1280,
            "mosaic": 1.0,
            "mixup": 1.0,
        }
        
        default_max = suggestions.get(variable_name, current_value * 2)
        
        max_value = FloatPrompt.ask(
            f"[cyan]Valor MÁXIMO para {variable_name} (fração 5/5)[/cyan]",
            default=float(default_max)
        )
        
        return max_value
    
    def _show_benchmark_preview(
        self,
        variable_name: str,
        max_value: float,
        num_fractions: int
    ) -> None:
        """
        Show preview of benchmark tests.
        
        Automatically displays rounded values for integer-type variables
        (e.g., epochs, batch, imgsz) to match what will actually be used.
        
        Args:
            variable_name: Variable to benchmark
            max_value: Maximum value
            num_fractions: Number of fractions
        """
        self.console.print("\n[bold]Preview do Benchmark:[/bold]\n")
        
        # Check if this variable requires integer values
        all_vars = YOLOHyperparameters.get_all_variables()
        is_integer_type = all_vars.get(variable_name) == int
        
        table = Table(border_style="blue")
        table.add_column("Teste", style="cyan", justify="center")
        table.add_column("Fração", style="green", justify="center")
        table.add_column(variable_name, style="yellow", justify="right")
        
        # Add note if values will be rounded
        if is_integer_type:
            table.add_column("Info", style="dim", justify="left")
        
        for i in range(num_fractions):
            fraction = (i + 1) / num_fractions
            value = fraction * max_value
            
            # Round if integer type
            if is_integer_type:
                rounded_value = round(value)
                
                # Show both original and rounded if different
                if value != rounded_value:
                    table.add_row(
                        str(i + 1),
                        f"{i + 1}/{num_fractions}",
                        f"{rounded_value}",
                        f"(~{value:.2f})"
                    )
                else:
                    table.add_row(
                        str(i + 1),
                        f"{i + 1}/{num_fractions}",
                        f"{rounded_value}",
                        ""
                    )
            else:
                # Show decimal values for float types
                table.add_row(
                    str(i + 1),
                    f"{i + 1}/{num_fractions}",
                    f"{value:.4f}"
                )
        
        self.console.print(table)
        
        # Show rounding note if applicable
        if is_integer_type:
            self.console.print(
                "\n[dim]ℹ️  Valores serão automaticamente arredondados "
                f"('{variable_name}' requer integers)[/dim]"
            )
        
        # Estimate time
        avg_time_per_test = 30 * 60  # Estimate 30 min per test
        total_time_estimate = num_fractions * avg_time_per_test
        hours = total_time_estimate / 3600
        
        self.console.print(f"\n[yellow]Estimativa de tempo:[/yellow] ~{hours:.1f} horas")
    
    def _execute_benchmark(self, benchmark_config: BenchmarkConfig) -> None:
        """
        Execute single variable benchmark with progress display.
        
        Args:
            benchmark_config: Benchmark configuration
        """
        self.console.rule("[bold green]Executando Benchmark[/bold green]")
        
        # Create progress display
        progress_display = BenchmarkProgressDisplay(self.console)
        
        def progress_callback(test_num: int, total: int, message: str):
            """Callback for progress updates."""
            # This would be called by the benchmark engine
            pass
        
        try:
            # Run benchmark
            results = self.benchmark_engine.run_single_variable_benchmark(
                benchmark_config,
                progress_callback=progress_callback
            )
            
            self.console.print("\n[bold green]✓ Benchmark concluído com sucesso![/bold green]\n")
            
            # Show results
            self.benchmark_viz.show_summary(results)
            self.console.print()
            self.benchmark_viz.show_results_table(results)
            
        except Exception as e:
            self.console.print(f"\n[bold red]Erro no benchmark:[/bold red] {str(e)}\n")
        
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
    
    def _execute_multi_benchmark(self, multi_config: MultiBenchmarkConfig) -> None:
        """
        Execute multi-variable benchmark.
        
        Args:
            multi_config: Multi-benchmark configuration
        """
        self.console.rule("[bold green]Executando Benchmark Multi-Variável[/bold green]")
        
        try:
            # Run benchmarks
            all_results = self.benchmark_engine.run_multi_variable_benchmark(multi_config)
            
            self.console.print("\n[bold green]✓ Benchmarks concluídos![/bold green]\n")
            
            # Show summary for each
            for results in all_results:
                self.benchmark_viz.show_summary(results)
                self.console.print()
            
        except Exception as e:
            self.console.print(f"\n[bold red]Erro no benchmark:[/bold red] {str(e)}\n")
        
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
    
    def _view_benchmark_results(self) -> None:
        """View saved benchmark results."""
        self.console.rule("[bold cyan]Resultados Salvos[/bold cyan]")
        
        # Load all results
        all_results = self.benchmark_engine.load_benchmark_results()
        
        if not all_results:
            self.console.print("\n[yellow]Nenhum resultado encontrado[/yellow]\n")
            Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
            return
        
        # Show list
        table = Table(title="Benchmarks Disponíveis", border_style="blue")
        table.add_column("#", style="cyan", justify="center")
        table.add_column("Variável", style="green")
        table.add_column("Testes", style="yellow", justify="center")
        table.add_column("Data", style="magenta")
        table.add_column("Melhor mAP", style="bold yellow", justify="right")
        
        for i, result in enumerate(all_results, 1):
            best_map = max(t.metrics.performance.map50 for t in result.tests)
            table.add_row(
                str(i),
                result.variable_name,
                str(len(result.tests)),
                result.timestamp.strftime("%Y-%m-%d %H:%M"),
                f"{best_map:.4f}"
            )
        
        self.console.print(table)
        
        # Select to view details
        choice = Prompt.ask(
            "\n[cyan]Selecione um benchmark para ver detalhes (0 = voltar)[/cyan]",
            default="0"
        )
        
        try:
            idx = int(choice) - 1
            if idx >= 0 and idx < len(all_results):
                selected = all_results[idx]
                self.console.print()
                self.benchmark_viz.show_results_table(selected)
                self.console.print()
                self.benchmark_viz.show_ranking(selected, metric="map50")
        except ValueError:
            pass
        
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
    
    def _compare_benchmarks(self) -> None:
        """Compare multiple benchmarks."""
        self.console.rule("[bold cyan]Comparar Benchmarks[/bold cyan]")
        
        all_results = self.benchmark_engine.load_benchmark_results()
        
        if len(all_results) < 2:
            self.console.print("\n[yellow]Necessário pelo menos 2 benchmarks para comparar[/yellow]\n")
            Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
            return
        
        # Show available
        table = Table(border_style="blue")
        table.add_column("#", style="cyan", justify="center")
        table.add_column("Variável", style="green")
        table.add_column("ID", style="dim")
        
        for i, result in enumerate(all_results, 1):
            table.add_row(
                str(i),
                result.variable_name,
                result.benchmark_id[:8]
            )
        
        self.console.print(table)
        
        # Select benchmarks
        choice = Prompt.ask(
            "\n[cyan]Selecione benchmarks (números separados por vírgula)[/cyan]"
        )
        
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            selected_ids = [all_results[i].benchmark_id for i in indices if 0 <= i < len(all_results)]
            
            if len(selected_ids) >= 2:
                comparison = self.benchmark_engine.compare_benchmarks(selected_ids)
                
                # Show comparison
                comp_table = Table(title="Comparação de Benchmarks", border_style="green")
                comp_table.add_column("Métrica", style="cyan")
                
                for var in comparison['variables']:
                    comp_table.add_column(var, style="yellow", justify="right")
                
                comp_table.add_row(
                    "Melhor mAP@0.5",
                    *[f"{comparison['best_map50'][v]:.4f}" for v in comparison['variables']]
                )
                
                comp_table.add_row(
                    "Tempo Médio/Época",
                    *[f"{comparison['avg_time_per_epoch'][v]:.1f}s" for v in comparison['variables']]
                )
                
                comp_table.add_row(
                    "Tempo Total",
                    *[f"{comparison['total_train_time'][v]/3600:.1f}h" for v in comparison['variables']]
                )
                
                self.console.print()
                self.console.print(comp_table)
        
        except (ValueError, IndexError, KeyError) as e:
            self.console.print(f"[red]Erro: {str(e)}[/red]")
        
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
