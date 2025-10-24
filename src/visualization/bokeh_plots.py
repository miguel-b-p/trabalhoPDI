"""
Bokeh interactive visualization for benchmark results.

Creates professional, interactive HTML dashboards with:
- Line plots (metrics vs fractions)
- Scatter plots (performance vs cost)
- Heatmaps (correlations)
- Bar charts (rankings)
- Multi-line plots (loss curves)

All visualizations are fully interactive with hover tools, pan, zoom,
and other Bokeh features.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import numpy as np

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot, column, row
from bokeh.models import (
    HoverTool,
    ColumnDataSource,
    LinearColorMapper,
    ColorBar,
    Legend,
    Title
)
from bokeh.palettes import Category20, Viridis256, RdYlGn
from bokeh.transform import linear_cmap

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cli.core.metrics import BenchmarkResults


class BokehVisualizer:
    """
    Bokeh-based visualizer for benchmark results.
    
    Creates comprehensive interactive HTML dashboards for analyzing
    hyperparameter effects on model performance and computational cost.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize Bokeh visualizer.
        
        Args:
            output_dir: Directory for saving HTML outputs
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_full_dashboard(
        self,
        results: BenchmarkResults,
        filename: Optional[str] = None
    ) -> Path:
        """
        Create complete dashboard with all visualizations.
        
        Args:
            results: Benchmark results to visualize
            filename: Output filename (None = auto-generate)
        
        Returns:
            Path to generated HTML file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_{results.variable_name}_{timestamp}.html"
        
        output_path = self.output_dir / filename
        output_file(str(output_path))
        
        # Create all plots
        metric_plot = self.create_metric_vs_fraction_plot(results)
        scatter_plot = self.create_performance_vs_cost_scatter(results)
        bar_plot = self.create_f1_score_comparison_bar(results)
        loss_plot = self.create_loss_curves_plot(results)
        ranking_plot = self.create_ranking_plot(results)
        
        # Create layout
        dashboard = column(
            row(metric_plot, scatter_plot),
            row(bar_plot, ranking_plot),
            loss_plot
        )
        
        # Save
        save(dashboard)
        
        return output_path
    
    def create_metric_vs_fraction_plot(
        self,
        results: BenchmarkResults
    ) -> figure:
        """
        Create line plot of metrics vs fractions.
        
        Shows how performance metrics change across different
        hyperparameter values.
        
        Args:
            results: Benchmark results
        
        Returns:
            Bokeh figure
        """
        # Prepare data
        summary = results.get_summary_dataframe()
        
        fractions = list(range(1, len(summary['fraction']) + 1))
        
        # Create figure
        p = figure(
            title=f"Performance Metrics vs {results.variable_name}",
            x_axis_label="Fraction",
            y_axis_label="Metric Value",
            width=600,
            height=400,
            toolbar_location="above"
        )
        
        # Plot mAP@0.5
        p.line(
            fractions,
            summary['map50'],
            legend_label="mAP@0.5",
            color=Category20[20][0],
            line_width=2
        )
        p.scatter(
            fractions,
            summary['map50'],
            size=8,
            color=Category20[20][0],
            marker="circle"
        )
        
        # Plot mAP@0.5:0.95
        p.line(
            fractions,
            summary['map50_95'],
            legend_label="mAP@0.5:0.95",
            color=Category20[20][1],
            line_width=2
        )
        p.scatter(
            fractions,
            summary['map50_95'],
            size=8,
            color=Category20[20][1],
            marker="circle"
        )
        
        # Plot F1-Score
        p.line(
            fractions,
            summary['f1_score'],
            legend_label="F1-Score",
            color=Category20[20][2],
            line_width=2
        )
        p.scatter(
            fractions,
            summary['f1_score'],
            size=8,
            color=Category20[20][2],
            marker="circle"
        )
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Fraction", "@x"),
            ("Value", "@y{0.0000}")
        ])
        p.add_tools(hover)
        
        # Configure legend
        p.legend.location = "bottom_right"
        p.legend.click_policy = "hide"
        
        return p
    
    def create_performance_vs_cost_scatter(
        self,
        results: BenchmarkResults
    ) -> figure:
        """
        Create scatter plot of performance vs computational cost.
        
        Visualizes the trade-off between model performance (mAP)
        and training time/memory usage.
        
        Args:
            results: Benchmark results
        
        Returns:
            Bokeh figure
        """
        # Prepare data
        summary = results.get_summary_dataframe()
        
        # Normalize memory for sizing (10-40 pixel range)
        mem_array = np.array(summary['memory_peak'])
        # Handle edge case: if all memory values are the same
        if mem_array.max() > 0 and mem_array.max() != mem_array.min():
            sizes = 10 + (mem_array / mem_array.max()) * 30
        else:
            sizes = np.full_like(mem_array, 20.0)  # Default size
        
        # ✅ CORREÇÃO: Adicionar sizes como coluna no ColumnDataSource
        source = ColumnDataSource(data=dict(
            time=summary['time_per_epoch'],
            map50=summary['map50'],
            memory=summary['memory_peak'],
            fraction=summary['fraction'],
            value=summary['variable_value'],
            sizes=sizes.tolist()  # ✅ Adicionar sizes como coluna!
        ))
        
        # Create figure
        p = figure(
            title=f"Performance vs Cost ({results.variable_name})",
            x_axis_label="Time per Epoch (seconds)",
            y_axis_label="mAP@0.5",
            width=600,
            height=400,
            toolbar_location="above"
        )
        
        # Scatter plot with size based on memory
        # ✅ CORREÇÃO: Referenciar 'sizes' como string (nome da coluna)
        p.scatter(
            'time',
            'map50',
            size='sizes',  # ✅ String referenciando coluna no source!
            source=source,
            color=linear_cmap('memory', Viridis256, 
                            min(summary['memory_peak']),
                            max(summary['memory_peak'])),
            alpha=0.7,
            legend_label="Tests",
            marker="circle"
        )
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Fraction", "@fraction"),
            ("Value", "@value{0.00}"),
            ("mAP@0.5", "@map50{0.0000}"),
            ("Time/Epoch", "@time{0.0}s"),
            ("Memory Peak", "@memory{0.00}GB")
        ])
        p.add_tools(hover)
        
        return p
    
    def create_f1_score_comparison_bar(
        self,
        results: BenchmarkResults
    ) -> figure:
        """
        Create bar chart comparing F1-Scores across configurations.
        
        Args:
            results: Benchmark results
        
        Returns:
            Bokeh figure
        """
        # Prepare data
        summary = results.get_summary_dataframe()
        
        source = ColumnDataSource(data=dict(
            fractions=summary['fraction'],
            f1_scores=summary['f1_score'],
            colors=Category20[20][:len(summary['fraction'])]
        ))
        
        # Create figure
        p = figure(
            x_range=summary['fraction'],
            title=f"F1-Score by Configuration ({results.variable_name})",
            x_axis_label="Fraction",
            y_axis_label="F1-Score",
            width=600,
            height=400,
            toolbar_location="above"
        )
        
        # Bar chart
        p.vbar(
            x='fractions',
            top='f1_scores',
            source=source,
            width=0.7,
            color='colors',
            alpha=0.8
        )
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Fraction", "@fractions"),
            ("F1-Score", "@f1_scores{0.0000}")
        ])
        p.add_tools(hover)
        
        p.y_range.start = 0
        p.xgrid.grid_line_color = None
        
        return p
    
    def create_loss_curves_plot(
        self,
        results: BenchmarkResults
    ) -> figure:
        """
        Create multi-line plot of training loss curves.
        
        Shows how different loss components evolve across fractions.
        
        Args:
            results: Benchmark results
        
        Returns:
            Bokeh figure
        """
        # Create figure
        p = figure(
            title=f"Loss Curves Across Fractions ({results.variable_name})",
            x_axis_label="Test Number",
            y_axis_label="Final Loss",
            width=1200,
            height=400,
            toolbar_location="above"
        )
        
        # Prepare data
        test_numbers = []
        final_losses = []
        
        for test in results.tests:
            test_numbers.append(test.test_number)
            final_losses.append(test.metrics.loss_curves.get_final_loss())
        
        # Plot final loss
        p.line(
            test_numbers,
            final_losses,
            legend_label="Final Loss",
            color=Category20[20][3],
            line_width=3
        )
        p.scatter(
            test_numbers,
            final_losses,
            size=10,
            color=Category20[20][3],
            marker="circle"
        )
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Test", "@x"),
            ("Loss", "@y{0.0000}")
        ])
        p.add_tools(hover)
        
        p.legend.location = "top_right"
        
        return p
    
    def create_ranking_plot(
        self,
        results: BenchmarkResults
    ) -> figure:
        """
        Create horizontal bar chart ranking configurations by mAP.
        
        Args:
            results: Benchmark results
        
        Returns:
            Bokeh figure
        """
        # Sort by mAP@0.5
        sorted_tests = sorted(
            results.tests,
            key=lambda t: t.metrics.performance.map50,
            reverse=True
        )
        
        # Prepare data
        fractions = [t.fraction for t in sorted_tests]
        map_scores = [t.metrics.performance.map50 for t in sorted_tests]
        
        # Create color gradient (best = green, worst = red)
        colors = RdYlGn[len(fractions)] if len(fractions) <= 11 else Viridis256[::256//len(fractions)]
        
        source = ColumnDataSource(data=dict(
            fractions=fractions,
            scores=map_scores,
            colors=colors[:len(fractions)]
        ))
        
        # Create figure
        p = figure(
            y_range=fractions,
            title=f"Ranking by mAP@0.5 ({results.variable_name})",
            x_axis_label="mAP@0.5",
            y_axis_label="Fraction",
            width=600,
            height=400,
            toolbar_location="above"
        )
        
        # Horizontal bars
        p.hbar(
            y='fractions',
            right='scores',
            source=source,
            height=0.7,
            color='colors',
            alpha=0.8
        )
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Fraction", "@fractions"),
            ("mAP@0.5", "@scores{0.0000}")
        ])
        p.add_tools(hover)
        
        p.x_range.start = 0
        p.ygrid.grid_line_color = None
        
        return p
    
    def create_comparison_dashboard(
        self,
        results_list: List[BenchmarkResults],
        filename: Optional[str] = None
    ) -> Path:
        """
        Create comparison dashboard for multiple benchmark results.
        
        Args:
            results_list: List of benchmark results to compare
            filename: Output filename (None = auto-generate)
        
        Returns:
            Path to generated HTML file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            variables = "_".join([r.variable_name for r in results_list[:3]])
            filename = f"comparison_{variables}_{timestamp}.html"
        
        output_path = self.output_dir / filename
        output_file(str(output_path))
        
        # Create comparison plot
        comparison_plot = self._create_multi_variable_comparison(results_list)
        
        # Create individual plots for each variable
        individual_plots = []
        for results in results_list:
            plot = self.create_metric_vs_fraction_plot(results)
            individual_plots.append(plot)
        
        # Create layout
        dashboard = column(
            comparison_plot,
            gridplot([individual_plots], ncols=2)
        )
        
        # Save
        save(dashboard)
        
        return output_path
    
    def _create_multi_variable_comparison(
        self,
        results_list: List[BenchmarkResults]
    ) -> figure:
        """
        Create plot comparing multiple variables.
        
        Args:
            results_list: List of benchmark results
        
        Returns:
            Bokeh figure
        """
        # Create figure
        p = figure(
            title="Multi-Variable Comparison: Best mAP@0.5",
            x_axis_label="Variable",
            y_axis_label="Best mAP@0.5",
            width=1200,
            height=400,
            toolbar_location="above"
        )
        
        # Prepare data
        variables = []
        best_maps = []
        
        for results in results_list:
            variables.append(results.variable_name)
            best_map = max(t.metrics.performance.map50 for t in results.tests)
            best_maps.append(best_map)
        
        p.x_range.range_padding = 0.1
        p.xgrid.grid_line_color = None
        
        source = ColumnDataSource(data=dict(
            variables=variables,
            best_maps=best_maps,
            colors=Category20[20][:len(variables)]
        ))
        
        # Bar chart
        p.vbar(
            x='variables',
            top='best_maps',
            source=source,
            width=0.7,
            color='colors',
            alpha=0.8
        )
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Variable", "@variables"),
            ("Best mAP@0.5", "@best_maps{0.0000}")
        ])
        p.add_tools(hover)
        
        return p
