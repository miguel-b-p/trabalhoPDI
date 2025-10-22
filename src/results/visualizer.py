"""
Visualizador de Resultados
==========================
Cria visualizações interativas dos resultados de benchmark usando Bokeh.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from bokeh.plotting import figure, output_file, save, show
from bokeh.layouts import column, row, gridplot
from bokeh.models import (
    HoverTool, ColumnDataSource, Legend, LegendItem,
    Panel, Tabs, Range1d, LinearAxis
)
from bokeh.palettes import Category10_10, Viridis256
from bokeh.io import output_notebook, push_notebook
from bokeh.transform import factor_cmap


class BenchmarkVisualizer:
    """Cria visualizações interativas de resultados de benchmark."""
    
    def __init__(self, results_path: Path):
        """
        Inicializa o visualizador.
        
        Args:
            results_path: Caminho para a pasta de resultados
        """
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Configurações de estilo
        self.colors = Category10_10
        self.palette = Viridis256
    
    def visualize_benchmark(
        self,
        benchmark_data: Dict[str, Any],
        output_html: Optional[Path] = None,
        show_plot: bool = False
    ):
        """
        Cria visualização completa de um benchmark.
        
        Args:
            benchmark_data: Dados do benchmark
            output_html: Caminho para salvar HTML (None = auto)
            show_plot: Se deve exibir o plot
        """
        if output_html is None:
            benchmark_name = benchmark_data.get('benchmark_name', 'benchmark')
            output_html = self.results_path / f'{benchmark_name}_visualization.html'
        
        output_file(str(output_html))
        
        # Cria abas para diferentes visualizações
        tabs = []
        
        # 1. Visão geral
        overview_tab = self._create_overview_tab(benchmark_data)
        if overview_tab:
            tabs.append(overview_tab)
        
        # 2. Análise por parâmetro
        param_tabs = self._create_parameter_tabs(benchmark_data)
        tabs.extend(param_tabs)
        
        # 3. Comparação de métricas
        metrics_tab = self._create_metrics_comparison_tab(benchmark_data)
        if metrics_tab:
            tabs.append(metrics_tab)
        
        # 4. Timeline
        timeline_tab = self._create_timeline_tab(benchmark_data)
        if timeline_tab:
            tabs.append(timeline_tab)
        
        # Monta layout final
        final_layout = Tabs(tabs=tabs)
        
        # Salva
        save(final_layout)
        
        print(f"\nVisualização salva em: {output_html}")
        
        if show_plot:
            show(final_layout)
        
        return output_html
    
    def _create_overview_tab(self, benchmark_data: Dict[str, Any]) -> Optional[Panel]:
        """Cria aba com visão geral do benchmark."""
        try:
            # Prepara dados
            stats = {
                'Total de Testes': benchmark_data['total_tests'],
                'Testes Bem-sucedidos': benchmark_data['successful_tests'],
                'Testes Falhados': benchmark_data['failed_tests'],
                'Tempo Total (min)': benchmark_data['total_time'] / 60,
            }
            
            # Cria figura de barras
            p = figure(
                x_range=list(stats.keys()),
                height=400,
                title="Visão Geral do Benchmark",
                toolbar_location="above"
            )
            
            p.vbar(
                x=list(stats.keys()),
                top=list(stats.values()),
                width=0.7,
                color=self.colors[0],
                alpha=0.8
            )
            
            p.xgrid.grid_line_color = None
            p.y_range.start = 0
            p.xaxis.major_label_orientation = 0.8
            
            # Adiciona valores nas barras
            from bokeh.models import Label
            for i, (key, value) in enumerate(stats.items()):
                label = Label(
                    x=i, y=value,
                    text=str(round(value, 2)),
                    text_align='center',
                    y_offset=5
                )
                p.add_layout(label)
            
            return Panel(child=p, title="Visão Geral")
            
        except Exception as e:
            print(f"Erro ao criar aba de visão geral: {e}")
            return None
    
    def _create_parameter_tabs(self, benchmark_data: Dict[str, Any]) -> List[Panel]:
        """Cria abas para análise de cada parâmetro."""
        tabs = []
        
        results_by_param = benchmark_data.get('results_by_parameter', {})
        
        for param_name, param_results in results_by_param.items():
            try:
                tab = self._create_single_parameter_tab(param_name, param_results)
                if tab:
                    tabs.append(tab)
            except Exception as e:
                print(f"Erro ao criar aba para parâmetro {param_name}: {e}")
        
        return tabs
    
    def _create_single_parameter_tab(
        self,
        param_name: str,
        param_results: List[Dict[str, Any]]
    ) -> Optional[Panel]:
        """Cria aba para análise de um parâmetro específico."""
        # Extrai dados
        values = []
        training_times = []
        map50s = []
        map50_95s = []
        
        for result in param_results:
            values.append(result['benchmark_value'])
            training_times.append(result['training_time'] / 60)  # Converte para minutos
            
            # Extrai métricas
            val_metrics = result.get('val_metrics', {})
            map50s.append(val_metrics.get('mAP50', 0) * 100 if val_metrics.get('mAP50') else 0)
            map50_95s.append(val_metrics.get('mAP50-95', 0) * 100 if val_metrics.get('mAP50-95') else 0)
        
        # Ordena por valor do parâmetro
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
        values = [values[i] for i in sorted_indices]
        training_times = [training_times[i] for i in sorted_indices]
        map50s = [map50s[i] for i in sorted_indices]
        map50_95s = [map50_95s[i] for i in sorted_indices]
        
        # Converte valores para strings se forem booleanos ou categóricos
        if isinstance(values[0], bool):
            x_values = [str(v) for v in values]
            x_range = x_values
        elif isinstance(values[0], str):
            x_values = values
            x_range = values
        else:
            x_values = values
            x_range = None
        
        # Cria plots
        plots = []
        
        # 1. Tempo de treinamento
        if x_range:
            p1 = figure(
                x_range=x_range,
                height=300,
                title=f"Tempo de Treinamento vs {param_name}",
                x_axis_label=param_name,
                y_axis_label="Tempo (minutos)"
            )
            p1.vbar(x=x_values, top=training_times, width=0.7, color=self.colors[0], alpha=0.8)
        else:
            p1 = figure(
                height=300,
                title=f"Tempo de Treinamento vs {param_name}",
                x_axis_label=param_name,
                y_axis_label="Tempo (minutos)"
            )
            p1.line(x_values, training_times, line_width=2, color=self.colors[0], legend_label="Tempo")
            p1.circle(x_values, training_times, size=8, color=self.colors[0])
        
        p1.add_tools(HoverTool(tooltips=[
            (param_name, "@x"),
            ("Tempo", "@y{0.2f} min")
        ]))
        plots.append(p1)
        
        # 2. mAP Scores
        if any(map50s) or any(map50_95s):
            if x_range:
                p2 = figure(
                    x_range=x_range,
                    height=300,
                    title=f"mAP vs {param_name}",
                    x_axis_label=param_name,
                    y_axis_label="mAP (%)"
                )
            else:
                p2 = figure(
                    height=300,
                    title=f"mAP vs {param_name}",
                    x_axis_label=param_name,
                    y_axis_label="mAP (%)"
                )
            
            if not x_range:
                p2.line(x_values, map50s, line_width=2, color=self.colors[1], legend_label="mAP50")
                p2.circle(x_values, map50s, size=8, color=self.colors[1])
                
                p2.line(x_values, map50_95s, line_width=2, color=self.colors[2], legend_label="mAP50-95")
                p2.circle(x_values, map50_95s, size=8, color=self.colors[2])
            else:
                # Para categóricos, usa barras agrupadas
                from bokeh.models import FactorRange
                x_factors = [(str(v), 'mAP50') for v in x_values] + [(str(v), 'mAP50-95') for v in x_values]
                
                source = ColumnDataSource(data=dict(
                    x=x_factors,
                    y=map50s + map50_95s
                ))
                
                p2 = figure(
                    x_range=FactorRange(*x_factors),
                    height=300,
                    title=f"mAP vs {param_name}",
                    x_axis_label=param_name,
                    y_axis_label="mAP (%)"
                )
                
                p2.vbar(x='x', top='y', width=0.9, source=source,
                       color=factor_cmap('x', palette=[self.colors[1], self.colors[2]], factors=['mAP50', 'mAP50-95'], start=1, end=2))
            
            p2.legend.location = "top_left"
            p2.add_tools(HoverTool(tooltips=[
                (param_name, "@x"),
                ("Score", "@y{0.2f}%")
            ]))
            plots.append(p2)
        
        # Layout
        layout = column(*plots)
        
        return Panel(child=layout, title=f"Parâmetro: {param_name}")
    
    def _create_metrics_comparison_tab(self, benchmark_data: Dict[str, Any]) -> Optional[Panel]:
        """Cria aba com comparação de métricas entre todos os testes."""
        try:
            # Coleta todas as métricas
            all_results = benchmark_data.get('all_results', [])
            
            if not all_results:
                return None
            
            # Prepara dados
            data = {
                'test_id': [],
                'param': [],
                'value': [],
                'mAP50': [],
                'mAP50_95': [],
                'training_time': []
            }
            
            for result in all_results:
                if not result.get('success', False):
                    continue
                
                data['test_id'].append(result['test_id'])
                data['param'].append(result['benchmark_param'])
                data['value'].append(str(result['benchmark_value']))
                
                val_metrics = result.get('val_metrics', {})
                data['mAP50'].append(val_metrics.get('mAP50', 0) * 100 if val_metrics.get('mAP50') else 0)
                data['mAP50_95'].append(val_metrics.get('mAP50-95', 0) * 100 if val_metrics.get('mAP50-95') else 0)
                data['training_time'].append(result['training_time'] / 60)
            
            source = ColumnDataSource(data=data)
            
            # Scatter plot: mAP vs Training Time
            p = figure(
                height=400,
                title="mAP vs Tempo de Treinamento",
                x_axis_label="Tempo de Treinamento (min)",
                y_axis_label="mAP50-95 (%)",
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )
            
            # Agrupa por parâmetro
            unique_params = list(set(data['param']))
            for i, param in enumerate(unique_params):
                param_indices = [j for j, p in enumerate(data['param']) if p == param]
                
                p.circle(
                    x=[data['training_time'][j] for j in param_indices],
                    y=[data['mAP50_95'][j] for j in param_indices],
                    size=10,
                    color=self.colors[i % len(self.colors)],
                    alpha=0.6,
                    legend_label=param
                )
            
            p.legend.location = "top_left"
            p.legend.click_policy = "hide"
            
            hover = HoverTool(tooltips=[
                ("Teste ID", "@test_id"),
                ("Parâmetro", "@param"),
                ("Valor", "@value"),
                ("mAP50-95", "@mAP50_95{0.2f}%"),
                ("Tempo", "@training_time{0.2f} min")
            ])
            p.add_tools(hover)
            
            return Panel(child=p, title="Comparação de Métricas")
            
        except Exception as e:
            print(f"Erro ao criar aba de comparação: {e}")
            return None
    
    def _create_timeline_tab(self, benchmark_data: Dict[str, Any]) -> Optional[Panel]:
        """Cria aba com timeline dos testes."""
        try:
            all_results = benchmark_data.get('all_results', [])
            
            if not all_results:
                return None
            
            # Prepara dados
            test_ids = []
            times = []
            params = []
            
            for result in all_results:
                if result.get('success', False):
                    test_ids.append(result['test_id'])
                    times.append(result['training_time'] / 60)
                    params.append(f"{result['benchmark_param']}={result['benchmark_value']}")
            
            # Cria plot
            p = figure(
                height=400,
                title="Timeline de Execução dos Testes",
                x_axis_label="ID do Teste",
                y_axis_label="Tempo de Treinamento (min)"
            )
            
            p.vbar(x=test_ids, top=times, width=0.8, color=self.colors[3], alpha=0.7)
            
            hover = HoverTool(tooltips=[
                ("Teste", "@x"),
                ("Tempo", "@top{0.2f} min")
            ])
            p.add_tools(hover)
            
            return Panel(child=p, title="Timeline")
            
        except Exception as e:
            print(f"Erro ao criar aba de timeline: {e}")
            return None
    
    def create_comparison_plot(
        self,
        benchmark_names: List[str],
        results_dir: Path,
        metric: str = 'mAP50-95',
        output_html: Optional[Path] = None
    ):
        """
        Cria plot de comparação entre múltiplos benchmarks.
        
        Args:
            benchmark_names: Lista de nomes de benchmarks
            results_dir: Diretório com resultados
            metric: Métrica a comparar
            output_html: Caminho para salvar HTML
        """
        if output_html is None:
            output_html = self.results_path / 'benchmark_comparison.html'
        
        output_file(str(output_html))
        
        # Carrega dados de todos os benchmarks
        all_data = []
        
        for name in benchmark_names:
            benchmark_file = results_dir / name / 'benchmark_results.json'
            if not benchmark_file.exists():
                print(f"Aviso: {benchmark_file} não encontrado")
                continue
            
            import json
            with open(benchmark_file, 'r') as f:
                data = json.load(f)
                all_data.append((name, data))
        
        # Cria plot de comparação
        p = figure(
            height=500,
            title=f"Comparação de {metric} entre Benchmarks",
            x_axis_label="Benchmark",
            y_axis_label=metric,
            x_range=[name for name, _ in all_data]
        )
        
        # TODO: Implementar lógica de comparação
        
        save(p)
        print(f"Comparação salva em: {output_html}")
        
        return output_html
    
    def enable_notebook_mode(self):
        """Habilita modo notebook para visualização inline no Jupyter."""
        output_notebook()
