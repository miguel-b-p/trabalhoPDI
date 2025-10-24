"""
Configuration menu for interactive hyperparameter editing.

Provides an intuitive interface for configuring all YOLO hyperparameters
with validation and preview.
"""

from typing import Optional
from rich.console import Console
from rich.prompt import Prompt, FloatPrompt, IntPrompt, Confirm
from rich.panel import Panel
from rich.table import Table

from .core.config import YOLOHyperparameters
from .core.logger import BenchmarkLogger
from .visualizer import ConfigVisualizer


class ConfigurationMenu:
    """
    Interactive menu for hyperparameter configuration.
    
    Allows users to configure all hyperparameters with validation
    and real-time preview.
    """
    
    def __init__(
        self,
        config: YOLOHyperparameters,
        console: Optional[Console] = None,
        logger: Optional[BenchmarkLogger] = None
    ):
        """
        Initialize configuration menu.
        
        Args:
            config: Hyperparameter configuration to edit
            console: Rich console (None = create new)
            logger: Logger instance
        """
        self.config = config
        self.console = console or Console()
        self.logger = logger
        self.visualizer = ConfigVisualizer(self.console)
    
    def run(self) -> None:
        """Run the configuration menu loop."""
        while True:
            choice = self._show_menu()
            
            if choice == "0":
                break
            elif choice == "1":
                self._configure_optimization()
            elif choice == "2":
                self._configure_batch()
            elif choice == "3":
                self._configure_architecture()
            elif choice == "4":
                self._configure_augmentation()
            elif choice == "5":
                self._configure_regularization()
            elif choice == "6":
                self._configure_postprocessing()
            elif choice == "7":
                self._view_current_config()
            elif choice == "8":
                self._reset_to_defaults()
    
    def _show_menu(self) -> str:
        """
        Show configuration menu and get choice.
        
        Returns:
            User's choice
        """
        self.console.print()
        
        menu_panel = Panel(
            """[cyan]1[/cyan] ⚡ Otimização (lr0, momentum, optimizer, etc.)
[cyan]2[/cyan] 📦 Batch (batch size, accumulation)
[cyan]3[/cyan] 🏗️  Arquitetura (imgsz, epochs, patience)
[cyan]4[/cyan] 🎨 Augmentação de Dados (hsv, flip, mosaic, etc.)
[cyan]5[/cyan] 🛡️  Regularização (dropout, label smoothing)
[cyan]6[/cyan] 🎯 Pós-processamento (NMS conf, iou)
[cyan]7[/cyan] 👁️  Visualizar Configuração Atual
[cyan]8[/cyan] 🔄 Resetar para Padrões
[cyan]0[/cyan] ◀️  Voltar""",
            title="[bold]Menu de Configuração[/bold]",
            border_style="green"
        )
        
        self.console.print(menu_panel)
        
        choice = Prompt.ask(
            "\n[bold cyan]Escolha uma categoria[/bold cyan]",
            choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"],
            default="0"
        )
        
        return choice
    
    def _configure_optimization(self) -> None:
        """Configure optimization hyperparameters."""
        self.console.rule("[bold green]Otimização[/bold green]")
        
        # lr0
        self.config.optimization.lr0 = FloatPrompt.ask(
            "[cyan]lr0[/cyan] - Taxa de aprendizado inicial",
            default=self.config.optimization.lr0
        )
        
        # lrf
        self.config.optimization.lrf = FloatPrompt.ask(
            "[cyan]lrf[/cyan] - Taxa de aprendizado final (fração de lr0)",
            default=self.config.optimization.lrf
        )
        
        # momentum
        self.config.optimization.momentum = FloatPrompt.ask(
            "[cyan]momentum[/cyan] - Momentum do SGD/Adam beta1",
            default=self.config.optimization.momentum
        )
        
        # weight_decay
        self.config.optimization.weight_decay = FloatPrompt.ask(
            "[cyan]weight_decay[/cyan] - Regularização L2",
            default=self.config.optimization.weight_decay
        )
        
        # warmup_epochs
        self.config.optimization.warmup_epochs = FloatPrompt.ask(
            "[cyan]warmup_epochs[/cyan] - Épocas de warmup",
            default=self.config.optimization.warmup_epochs
        )
        
        # optimizer
        optimizer = Prompt.ask(
            "[cyan]optimizer[/cyan] - Algoritmo de otimização",
            choices=["SGD", "Adam", "AdamW", "RMSProp"],
            default=self.config.optimization.optimizer
        )
        self.config.optimization.optimizer = optimizer
        
        self.console.print("\n[bold green]✓ Configuração de otimização atualizada[/bold green]")
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
    
    def _configure_batch(self) -> None:
        """Configure batch hyperparameters."""
        self.console.rule("[bold green]Batch[/bold green]")
        
        # batch
        self.config.batch.batch = IntPrompt.ask(
            "[cyan]batch[/cyan] - Tamanho do batch (-1 para auto)",
            default=self.config.batch.batch
        )
        
        # accumulate
        self.config.batch.accumulate = IntPrompt.ask(
            "[cyan]accumulate[/cyan] - Gradient accumulation steps",
            default=self.config.batch.accumulate
        )
        
        # Show effective batch size
        effective = self.config.batch.effective_batch_size
        self.console.print(f"\n[yellow]Batch size efetivo:[/yellow] {effective}")
        
        self.console.print("\n[bold green]✓ Configuração de batch atualizada[/bold green]")
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
    
    def _configure_architecture(self) -> None:
        """Configure architecture hyperparameters."""
        self.console.rule("[bold green]Arquitetura[/bold green]")
        
        # imgsz
        imgsz = IntPrompt.ask(
            "[cyan]imgsz[/cyan] - Tamanho da imagem (múltiplo de 32)",
            default=self.config.architecture.imgsz
        )
        
        # Validate multiple of 32
        if imgsz % 32 != 0:
            self.console.print(f"[yellow]Ajustando para múltiplo de 32...[/yellow]")
            imgsz = ((imgsz // 32) + 1) * 32
            self.console.print(f"[yellow]Novo valor: {imgsz}[/yellow]")
        
        self.config.architecture.imgsz = imgsz
        
        # epochs
        self.config.architecture.epochs = IntPrompt.ask(
            "[cyan]epochs[/cyan] - Número de épocas de treinamento",
            default=self.config.architecture.epochs
        )
        
        # patience
        self.config.architecture.patience = IntPrompt.ask(
            "[cyan]patience[/cyan] - Early stopping patience",
            default=self.config.architecture.patience
        )
        
        self.console.print("\n[bold green]✓ Configuração de arquitetura atualizada[/bold green]")
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
    
    def _configure_augmentation(self) -> None:
        """Configure data augmentation hyperparameters."""
        self.console.rule("[bold green]Augmentação de Dados[/bold green]")
        
        self.console.print("\n[bold]Augmentações de Cor (HSV):[/bold]")
        
        self.config.augmentation.hsv_h = FloatPrompt.ask(
            "[cyan]hsv_h[/cyan] - HSV-Hue",
            default=self.config.augmentation.hsv_h
        )
        
        self.config.augmentation.hsv_s = FloatPrompt.ask(
            "[cyan]hsv_s[/cyan] - HSV-Saturation",
            default=self.config.augmentation.hsv_s
        )
        
        self.config.augmentation.hsv_v = FloatPrompt.ask(
            "[cyan]hsv_v[/cyan] - HSV-Value",
            default=self.config.augmentation.hsv_v
        )
        
        self.console.print("\n[bold]Augmentações Geométricas:[/bold]")
        
        self.config.augmentation.degrees = FloatPrompt.ask(
            "[cyan]degrees[/cyan] - Rotação (graus)",
            default=self.config.augmentation.degrees
        )
        
        self.config.augmentation.translate = FloatPrompt.ask(
            "[cyan]translate[/cyan] - Translação (fração)",
            default=self.config.augmentation.translate
        )
        
        self.config.augmentation.scale = FloatPrompt.ask(
            "[cyan]scale[/cyan] - Scale augmentation",
            default=self.config.augmentation.scale
        )
        
        self.console.print("\n[bold]Augmentações Avançadas:[/bold]")
        
        self.config.augmentation.mosaic = FloatPrompt.ask(
            "[cyan]mosaic[/cyan] - Mosaic probability",
            default=self.config.augmentation.mosaic
        )
        
        self.config.augmentation.mixup = FloatPrompt.ask(
            "[cyan]mixup[/cyan] - MixUp probability",
            default=self.config.augmentation.mixup
        )
        
        self.config.augmentation.fliplr = FloatPrompt.ask(
            "[cyan]fliplr[/cyan] - Horizontal flip probability",
            default=self.config.augmentation.fliplr
        )
        
        self.console.print("\n[bold green]✓ Configuração de augmentação atualizada[/bold green]")
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
    
    def _configure_regularization(self) -> None:
        """Configure regularization hyperparameters."""
        self.console.rule("[bold green]Regularização[/bold green]")
        
        # label_smoothing
        self.config.regularization.label_smoothing = FloatPrompt.ask(
            "[cyan]label_smoothing[/cyan] - Label smoothing epsilon",
            default=self.config.regularization.label_smoothing
        )
        
        # dropout
        self.config.regularization.dropout = FloatPrompt.ask(
            "[cyan]dropout[/cyan] - Dropout probability",
            default=self.config.regularization.dropout
        )
        
        self.console.print("\n[bold green]✓ Configuração de regularização atualizada[/bold green]")
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
    
    def _configure_postprocessing(self) -> None:
        """Configure post-processing (NMS) hyperparameters."""
        self.console.rule("[bold green]Pós-processamento (NMS)[/bold green]")
        
        # conf
        self.config.postprocessing.conf = FloatPrompt.ask(
            "[cyan]conf[/cyan] - Confidence threshold",
            default=self.config.postprocessing.conf
        )
        
        # iou
        self.config.postprocessing.iou = FloatPrompt.ask(
            "[cyan]iou[/cyan] - IoU threshold para NMS",
            default=self.config.postprocessing.iou
        )
        
        # max_det
        self.config.postprocessing.max_det = IntPrompt.ask(
            "[cyan]max_det[/cyan] - Detecções máximas por imagem",
            default=self.config.postprocessing.max_det
        )
        
        self.console.print("\n[bold green]✓ Configuração de pós-processamento atualizada[/bold green]")
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
    
    def _view_current_config(self) -> None:
        """View current configuration."""
        self.console.rule("[bold cyan]Configuração Atual[/bold cyan]")
        
        view_type = Prompt.ask(
            "\n[cyan]Formato de visualização[/cyan]",
            choices=["tabela", "árvore", "yaml"],
            default="árvore"
        )
        
        self.console.print()
        
        if view_type == "tabela":
            self.visualizer.show_as_table(self.config)
        elif view_type == "árvore":
            self.visualizer.show_as_tree(self.config)
        elif view_type == "yaml":
            self.visualizer.show_as_yaml(self.config)
        
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
    
    def _reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        if Confirm.ask("\n[yellow]Resetar todas as configurações para os valores padrão?[/yellow]"):
            self.config = YOLOHyperparameters()
            self.console.print("\n[bold green]✓ Configuração resetada para padrões[/bold green]")
        
        Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]")
