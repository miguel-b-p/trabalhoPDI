"""
Interface Jupyter
=================
Interface interativa completa para Jupyter usando IPython widgets.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import json

from ..core.config import Config, BenchmarkConfig
from ..core.trainer import YOLOTrainer
from ..core.benchmark import BenchmarkRunner
from ..core.data_manager import DataManager
from ..results.visualizer import BenchmarkVisualizer
from ..results.analyzer import ResultsAnalyzer


class YOLOBenchmarkInterface:
    """Interface interativa completa para benchmarks YOLO no Jupyter."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Inicializa a interface.
        
        Args:
            project_root: Raiz do projeto (None = detecta automaticamente)
        """
        if project_root is None:
            project_root = Path.cwd()
        
        self.project_root = Path(project_root)
        self.config = Config(project_root=self.project_root)
        self.benchmark_config = BenchmarkConfig()
        
        # Output widgets
        self.output = widgets.Output()
        
        # Estado
        self.current_tab = 'training'
        
        # Inicializa componentes
        self._init_widgets()
        
        # Habilita modo notebook para visualizações
        visualizer = BenchmarkVisualizer(self.config.results_path)
        visualizer.enable_notebook_mode()
    
    def _init_widgets(self):
        """Inicializa todos os widgets da interface."""
        # === ABA DE TREINAMENTO ===
        self.training_widgets = self._create_training_widgets()
        
        # === ABA DE BENCHMARK ===
        self.benchmark_widgets = self._create_benchmark_widgets()
        
        # === ABA DE RESULTADOS ===
        self.results_widgets = self._create_results_widgets()
        
        # === ABA DE CONFIGURAÇÕES ===
        self.config_widgets = self._create_config_widgets()
    
    def _create_training_widgets(self) -> Dict[str, widgets.Widget]:
        """Cria widgets para treinamento individual."""
        w = {}
        
        # Dataset
        w['dataset_path'] = widgets.Text(
            description='Dataset:',
            placeholder='/path/to/data.yaml',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='600px')
        )
        
        # Modelo
        w['model_name'] = widgets.Dropdown(
            description='Modelo:',
            options=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt',
                    'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
            value='yolo11n.pt',
            style={'description_width': '150px'}
        )
        
        # Epochs
        w['epochs'] = widgets.IntSlider(
            description='Epochs:',
            min=1, max=500, value=100,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='600px')
        )
        
        # Batch size
        w['batch_size'] = widgets.Dropdown(
            description='Batch Size:',
            options=[4, 8, 16, 32, 64],
            value=16,
            style={'description_width': '150px'}
        )
        
        # Image size
        w['imgsz'] = widgets.Dropdown(
            description='Image Size:',
            options=[320, 416, 512, 640, 768, 1024, 1280],
            value=640,
            style={'description_width': '150px'}
        )
        
        # Device
        w['device'] = widgets.Dropdown(
            description='Device:',
            options=['cuda', 'cpu', 'mps'],
            value='cuda',
            style={'description_width': '150px'}
        )
        
        # Optimizer
        w['optimizer'] = widgets.Dropdown(
            description='Optimizer:',
            options=['SGD', 'Adam', 'AdamW'],
            value='Adam',
            style={'description_width': '150px'}
        )
        
        # Learning rate
        w['lr0'] = widgets.FloatText(
            description='Learning Rate:',
            value=0.01,
            step=0.001,
            style={'description_width': '150px'}
        )
        
        # Augmentation
        w['augment'] = widgets.Checkbox(
            description='Data Augmentation',
            value=True,
            style={'description_width': '150px'}
        )
        
        # Botões
        w['train_button'] = widgets.Button(
            description='🚀 Iniciar Treinamento',
            button_style='success',
            layout=widgets.Layout(width='200px', height='40px')
        )
        w['train_button'].on_click(self._on_train_click)
        
        return w
    
    def _create_benchmark_widgets(self) -> Dict[str, widgets.Widget]:
        """Cria widgets para benchmark."""
        w = {}
        
        # Dataset
        w['dataset_path'] = widgets.Text(
            description='Dataset:',
            placeholder='/path/to/data.yaml',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='600px')
        )
        
        # Nome do benchmark
        w['benchmark_name'] = widgets.Text(
            description='Nome:',
            value='yolo_benchmark',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='600px')
        )
        
        # Número de divisões
        w['num_divisions'] = widgets.IntSlider(
            description='Divisões:',
            min=3, max=10, value=5,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='600px')
        )
        
        # Parâmetros para benchmark
        w['params_header'] = widgets.HTML("<h3>Parâmetros para Benchmark</h3>")
        
        # Epochs
        w['bench_epochs'] = widgets.Checkbox(
            description='Epochs',
            value=True,
            style={'description_width': '150px'}
        )
        w['epochs_min'] = widgets.IntText(description='Min:', value=10, layout=widgets.Layout(width='150px'))
        w['epochs_max'] = widgets.IntText(description='Max:', value=100, layout=widgets.Layout(width='150px'))
        
        # Batch size
        w['bench_batch'] = widgets.Checkbox(
            description='Batch Size',
            value=True,
            style={'description_width': '150px'}
        )
        w['batch_min'] = widgets.IntText(description='Min:', value=4, layout=widgets.Layout(width='150px'))
        w['batch_max'] = widgets.IntText(description='Max:', value=32, layout=widgets.Layout(width='150px'))
        
        # Image size
        w['bench_imgsz'] = widgets.Checkbox(
            description='Image Size',
            value=True,
            style={'description_width': '150px'}
        )
        w['imgsz_min'] = widgets.IntText(description='Min:', value=320, layout=widgets.Layout(width='150px'))
        w['imgsz_max'] = widgets.IntText(description='Max:', value=1280, layout=widgets.Layout(width='150px'))
        
        # Learning rate
        w['bench_lr'] = widgets.Checkbox(
            description='Learning Rate',
            value=True,
            style={'description_width': '150px'}
        )
        w['lr_min'] = widgets.FloatText(description='Min:', value=0.001, layout=widgets.Layout(width='150px'))
        w['lr_max'] = widgets.FloatText(description='Max:', value=0.1, layout=widgets.Layout(width='150px'))
        
        # Optimizer
        w['bench_optimizer'] = widgets.Checkbox(
            description='Optimizer',
            value=False,
            style={'description_width': '150px'}
        )
        
        # Augmentation
        w['bench_augment'] = widgets.Checkbox(
            description='Augmentation',
            value=False,
            style={'description_width': '150px'}
        )
        
        # Modo paralelo
        w['parallel'] = widgets.Checkbox(
            description='Execução Paralela',
            value=False,
            style={'description_width': '150px'}
        )
        
        w['max_workers'] = widgets.IntSlider(
            description='Workers:',
            min=1, max=8, value=2,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        
        # Botão
        w['benchmark_button'] = widgets.Button(
            description='🏁 Iniciar Benchmark',
            button_style='warning',
            layout=widgets.Layout(width='200px', height='40px')
        )
        w['benchmark_button'].on_click(self._on_benchmark_click)
        
        return w
    
    def _create_results_widgets(self) -> Dict[str, widgets.Widget]:
        """Cria widgets para visualização de resultados."""
        w = {}
        
        # Seletor de benchmark
        w['benchmark_selector'] = widgets.Dropdown(
            description='Benchmark:',
            options=self._list_benchmarks(),
            style={'description_width': '150px'},
            layout=widgets.Layout(width='600px')
        )
        
        # Botão refresh
        w['refresh_button'] = widgets.Button(
            description='🔄 Atualizar Lista',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        w['refresh_button'].on_click(self._on_refresh_benchmarks)
        
        # Botão visualizar
        w['visualize_button'] = widgets.Button(
            description='📊 Visualizar',
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        w['visualize_button'].on_click(self._on_visualize_click)
        
        # Botão relatório
        w['report_button'] = widgets.Button(
            description='📄 Gerar Relatório',
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        w['report_button'].on_click(self._on_report_click)
        
        return w
    
    def _create_config_widgets(self) -> Dict[str, widgets.Widget]:
        """Cria widgets para configurações."""
        w = {}
        
        # Configurações gerais
        w['config_header'] = widgets.HTML("<h3>Configurações do Projeto</h3>")
        
        w['project_root'] = widgets.Text(
            description='Raiz do Projeto:',
            value=str(self.project_root),
            style={'description_width': '150px'},
            layout=widgets.Layout(width='600px')
        )
        
        w['workers'] = widgets.IntSlider(
            description='Workers:',
            min=1, max=16, value=8,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='600px')
        )
        
        w['seed'] = widgets.IntText(
            description='Random Seed:',
            value=42,
            style={'description_width': '150px'}
        )
        
        w['verbose'] = widgets.Checkbox(
            description='Verbose Output',
            value=True,
            style={'description_width': '150px'}
        )
        
        # Botões
        w['save_config_button'] = widgets.Button(
            description='💾 Salvar Configuração',
            button_style='success',
            layout=widgets.Layout(width='200px')
        )
        w['save_config_button'].on_click(self._on_save_config)
        
        w['load_config_button'] = widgets.Button(
            description='📂 Carregar Configuração',
            button_style='info',
            layout=widgets.Layout(width='200px')
        )
        w['load_config_button'].on_click(self._on_load_config)
        
        return w
    
    def _on_train_click(self, button):
        """Handler para botão de treinamento."""
        with self.output:
            clear_output()
            
            # Atualiza config
            self.config.model_name = self.training_widgets['model_name'].value
            self.config.epochs = self.training_widgets['epochs'].value
            self.config.batch_size = self.training_widgets['batch_size'].value
            self.config.imgsz = self.training_widgets['imgsz'].value
            self.config.device = self.training_widgets['device'].value
            self.config.optimizer = self.training_widgets['optimizer'].value
            self.config.lr0 = self.training_widgets['lr0'].value
            self.config.augment = self.training_widgets['augment'].value
            
            dataset_path = self.training_widgets['dataset_path'].value
            
            if not dataset_path:
                print("❌ Erro: Especifique o caminho do dataset!")
                return
            
            try:
                # Treina
                trainer = YOLOTrainer(self.config)
                trainer.load_model()
                metrics = trainer.train(data=dataset_path)
                
                print("\n✅ Treinamento concluído com sucesso!")
                print(f"Métricas: {metrics}")
                
            except Exception as e:
                print(f"❌ Erro durante treinamento: {str(e)}")
    
    def _on_benchmark_click(self, button):
        """Handler para botão de benchmark."""
        with self.output:
            clear_output()
            
            dataset_path = self.benchmark_widgets['dataset_path'].value
            
            if not dataset_path:
                print("❌ Erro: Especifique o caminho do dataset!")
                return
            
            # Configura benchmark
            self.benchmark_config.benchmark_name = self.benchmark_widgets['benchmark_name'].value
            self.benchmark_config.num_divisions = self.benchmark_widgets['num_divisions'].value
            
            # Monta parâmetros para benchmark
            benchmark_params = {}
            
            if self.benchmark_widgets['bench_epochs'].value:
                benchmark_params['epochs'] = {
                    'min': self.benchmark_widgets['epochs_min'].value,
                    'max': self.benchmark_widgets['epochs_max'].value,
                    'type': 'int'
                }
            
            if self.benchmark_widgets['bench_batch'].value:
                benchmark_params['batch_size'] = {
                    'min': self.benchmark_widgets['batch_min'].value,
                    'max': self.benchmark_widgets['batch_max'].value,
                    'type': 'int'
                }
            
            if self.benchmark_widgets['bench_imgsz'].value:
                benchmark_params['imgsz'] = {
                    'min': self.benchmark_widgets['imgsz_min'].value,
                    'max': self.benchmark_widgets['imgsz_max'].value,
                    'type': 'int'
                }
            
            if self.benchmark_widgets['bench_lr'].value:
                benchmark_params['lr0'] = {
                    'min': self.benchmark_widgets['lr_min'].value,
                    'max': self.benchmark_widgets['lr_max'].value,
                    'type': 'float'
                }
            
            if self.benchmark_widgets['bench_optimizer'].value:
                benchmark_params['optimizer'] = {
                    'values': ['SGD', 'Adam', 'AdamW'],
                    'type': 'categorical'
                }
            
            if self.benchmark_widgets['bench_augment'].value:
                benchmark_params['augment'] = {
                    'values': [True, False],
                    'type': 'bool'
                }
            
            if not benchmark_params:
                print("❌ Erro: Selecione pelo menos um parâmetro para benchmark!")
                return
            
            self.benchmark_config.benchmark_params = benchmark_params
            
            try:
                # Executa benchmark
                runner = BenchmarkRunner(self.config, self.benchmark_config)
                
                parallel = self.benchmark_widgets['parallel'].value
                max_workers = self.benchmark_widgets['max_workers'].value
                
                results = runner.run_benchmark(
                    dataset_path=dataset_path,
                    parallel=parallel,
                    max_workers=max_workers
                )
                
                print("\n✅ Benchmark concluído com sucesso!")
                print(f"Total de testes: {results['total_tests']}")
                print(f"Testes bem-sucedidos: {results['successful_tests']}")
                
            except Exception as e:
                print(f"❌ Erro durante benchmark: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def _on_visualize_click(self, button):
        """Handler para visualização."""
        with self.output:
            clear_output()
            
            benchmark_name = self.results_widgets['benchmark_selector'].value
            
            if not benchmark_name:
                print("❌ Erro: Selecione um benchmark!")
                return
            
            try:
                # Carrega resultados
                analyzer = ResultsAnalyzer(self.config.results_path)
                benchmark_data = analyzer.load_benchmark(benchmark_name)
                
                # Cria visualização
                visualizer = BenchmarkVisualizer(self.config.results_path)
                output_html = visualizer.visualize_benchmark(
                    benchmark_data,
                    show_plot=False
                )
                
                print(f"✅ Visualização criada: {output_html}")
                
                # Exibe link
                display(HTML(f'<a href="{output_html}" target="_blank">Abrir Visualização</a>'))
                
            except Exception as e:
                print(f"❌ Erro ao visualizar: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def _on_report_click(self, button):
        """Handler para geração de relatório."""
        with self.output:
            clear_output()
            
            benchmark_name = self.results_widgets['benchmark_selector'].value
            
            if not benchmark_name:
                print("❌ Erro: Selecione um benchmark!")
                return
            
            try:
                # Carrega resultados
                analyzer = ResultsAnalyzer(self.config.results_path)
                benchmark_data = analyzer.load_benchmark(benchmark_name)
                
                # Gera relatório
                report_file = self.config.results_path / 'benchmarks' / benchmark_name / 'report.txt'
                report = analyzer.generate_report(benchmark_data, report_file)
                
                print(report)
                
            except Exception as e:
                print(f"❌ Erro ao gerar relatório: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def _on_refresh_benchmarks(self, button):
        """Handler para atualizar lista de benchmarks."""
        self.results_widgets['benchmark_selector'].options = self._list_benchmarks()
        with self.output:
            clear_output()
            print("✅ Lista atualizada!")
    
    def _on_save_config(self, button):
        """Handler para salvar configuração."""
        with self.output:
            clear_output()
            try:
                config_file = self.project_root / 'config.json'
                self.config.save(config_file)
                print(f"✅ Configuração salva em: {config_file}")
            except Exception as e:
                print(f"❌ Erro ao salvar configuração: {str(e)}")
    
    def _on_load_config(self, button):
        """Handler para carregar configuração."""
        with self.output:
            clear_output()
            try:
                config_file = self.project_root / 'config.json'
                if config_file.exists():
                    self.config = Config.load(config_file)
                    print(f"✅ Configuração carregada de: {config_file}")
                else:
                    print(f"❌ Arquivo não encontrado: {config_file}")
            except Exception as e:
                print(f"❌ Erro ao carregar configuração: {str(e)}")
    
    def _list_benchmarks(self) -> list:
        """Lista benchmarks disponíveis."""
        benchmarks_dir = self.config.results_path / 'benchmarks'
        if not benchmarks_dir.exists():
            return []
        
        return [d.name for d in benchmarks_dir.iterdir() if d.is_dir()]
    
    def show(self):
        """Exibe a interface completa."""
        # Cria abas
        training_tab = self._create_training_tab()
        benchmark_tab = self._create_benchmark_tab()
        results_tab = self._create_results_tab()
        config_tab = self._create_config_tab()
        
        tabs = widgets.Tab()
        tabs.children = [training_tab, benchmark_tab, results_tab, config_tab]
        tabs.titles = ['🎯 Treinamento', '🏁 Benchmark', '📊 Resultados', '⚙️ Configurações']
        
        # Layout final
        display(HTML("<h1>🚀 YOLO Benchmark System</h1>"))
        display(tabs)
        display(self.output)
    
    def _create_training_tab(self):
        """Cria aba de treinamento."""
        w = self.training_widgets
        
        return widgets.VBox([
            widgets.HTML("<h2>Treinamento Individual</h2>"),
            w['dataset_path'],
            widgets.HBox([w['model_name'], w['device']]),
            w['epochs'],
            widgets.HBox([w['batch_size'], w['imgsz']]),
            widgets.HBox([w['optimizer'], w['lr0']]),
            w['augment'],
            widgets.HTML("<br>"),
            w['train_button']
        ])
    
    def _create_benchmark_tab(self):
        """Cria aba de benchmark."""
        w = self.benchmark_widgets
        
        return widgets.VBox([
            widgets.HTML("<h2>Benchmark de Parâmetros</h2>"),
            w['dataset_path'],
            w['benchmark_name'],
            w['num_divisions'],
            w['params_header'],
            widgets.HBox([w['bench_epochs'], w['epochs_min'], w['epochs_max']]),
            widgets.HBox([w['bench_batch'], w['batch_min'], w['batch_max']]),
            widgets.HBox([w['bench_imgsz'], w['imgsz_min'], w['imgsz_max']]),
            widgets.HBox([w['bench_lr'], w['lr_min'], w['lr_max']]),
            w['bench_optimizer'],
            w['bench_augment'],
            widgets.HTML("<h3>Opções de Execução</h3>"),
            w['parallel'],
            w['max_workers'],
            widgets.HTML("<br>"),
            w['benchmark_button']
        ])
    
    def _create_results_tab(self):
        """Cria aba de resultados."""
        w = self.results_widgets
        
        return widgets.VBox([
            widgets.HTML("<h2>Visualização de Resultados</h2>"),
            widgets.HBox([w['benchmark_selector'], w['refresh_button']]),
            widgets.HTML("<br>"),
            widgets.HBox([w['visualize_button'], w['report_button']])
        ])
    
    def _create_config_tab(self):
        """Cria aba de configurações."""
        w = self.config_widgets
        
        return widgets.VBox([
            w['config_header'],
            w['project_root'],
            w['workers'],
            w['seed'],
            w['verbose'],
            widgets.HTML("<br>"),
            widgets.HBox([w['save_config_button'], w['load_config_button']])
        ])
