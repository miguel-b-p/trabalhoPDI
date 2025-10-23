"""
Results visualization with Bokeh for YOLO benchmark analysis.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from bokeh.plotting import figure, save, output_file
from bokeh.layouts import column, row, gridplot
from bokeh.models import (
    ColumnDataSource, HoverTool, FactorRange, 
    Select, Slider, CustomJS, Div, Panel, Tabs,
    LinearColorMapper, ColorBar, BasicTicker
)
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Spectral6, Viridis256, Category20
from bokeh.io import curdoc, show
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.server.server import Server
from bokeh.embed import file_html
from bokeh.resources import CDN


class ResultsVisualizer:
    """Comprehensive results visualization for YOLO benchmark analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results_data = None
        self.benchmark_config = None
    
    def load_results(self, results_dir: Path) -> bool:
        """Load benchmark results from directory."""
        try:
            summary_file = results_dir / "benchmark_summary.json"
            if not summary_file.exists():
                self.logger.error(f"No benchmark summary found: {summary_file}")
                return False
            
            with open(summary_file) as f:
                summary = json.load(f)
            
            self.results_data = summary.get("results", [])
            self.benchmark_config = summary.get("config", {})
            
            # Filter successful runs
            self.results_data = [r for r in self.results_data if r.get("status") == "completed"]
            
            return len(self.results_data) > 0
            
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return False
    
    def prepare_data(self) -> pd.DataFrame:
        """Prepare data for visualization."""
        if not self.results_data:
            return pd.DataFrame()
        
        # Flatten results into DataFrame
        rows = []
        for result in self.results_data:
            row = {
                "experiment_id": result.get("experiment_id"),
                "parameter": result.get("experiment_metadata", {}).get("parameter"),
                "fraction": result.get("experiment_metadata", {}).get("fraction"),
                "value": result.get("experiment_metadata", {}).get("value"),
                "repetition": result.get("experiment_metadata", {}).get("repetition", 0),
                "training_time": result.get("training_time"),
            }
            
            # Add metrics
            metrics = result.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = value
            
            # Add system stats
            system_stats = result.get("system_stats", {})
            if "duration_seconds" in system_stats:
                row["duration"] = system_stats["duration_seconds"]
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_parameter_impact_plot(self, df: pd.DataFrame) -> figure:
        """Create parameter impact visualization."""
        if df.empty:
            return figure(title="No data available")
        
        # Group by parameter and fraction
        impact_data = []
        parameters = df["parameter"].dropna().unique()
        
        for param in parameters:
            param_df = df[df["parameter"] == param]
            for fraction in param_df["fraction"].unique():
                subset = param_df[param_df["fraction"] == fraction]
                if not subset.empty:
                    impact_data.append({
                        "parameter": param,
                        "fraction": fraction,
                        "mAP50_mean": subset["mAP50"].mean(),
                        "mAP50_std": subset["mAP50"].std(),
                        "mAP50_95_mean": subset["mAP50_95"].mean(),
                        "mAP50_95_std": subset["mAP50_95"].std(),
                        "count": len(subset)
                    })
        
        impact_df = pd.DataFrame(impact_data)
        
        if impact_df.empty:
            return figure(title="No parameter impact data")
        
        # Create plot
        p = figure(
            title="Parameter Impact on Model Performance",
            x_axis_label="Fraction of Parameter Range",
            y_axis_label="mAP Score",
            width=1000,
            height=600
        )
        
        # Color mapping for parameters
        colors = Category20[20]
        param_colors = {param: colors[i % 20] for i, param in enumerate(parameters)}
        
        for param in parameters:
            param_data = impact_df[impact_df["parameter"] == param]
            if not param_data.empty:
                source = ColumnDataSource(param_data)
                
                # mAP50 line
                p.line(
                    x="fraction", y="mAP50_mean",
                    source=source,
                    color=param_colors[param],
                    line_width=2,
                    legend_label=f"{param} (mAP50)"
                )
                
                # mAP50 error bars
                p.vbar(
                    x="fraction", width=0.02, top="mAP50_mean", bottom="mAP50_mean",
                    source=source,
                    color=param_colors[param],
                    alpha=0.3
                )
                
                # mAP50-95 line
                p.line(
                    x="fraction", y="mAP50_95_mean",
                    source=source,
                    color=param_colors[param],
                    line_width=2,
                    line_dash="dashed",
                    legend_label=f"{param} (mAP50-95)"
                )
        
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Parameter", "@parameter"),
            ("Fraction", "@fraction"),
            ("mAP50", "@mAP50_mean"),
            ("mAP50-95", "@mAP50_95_mean"),
            ("Samples", "@count")
        ])
        p.add_tools(hover)
        
        return p
    
    def create_performance_tradeoff_plot(self, df: pd.DataFrame) -> figure:
        """Create performance vs training time tradeoff plot."""
        if df.empty or "mAP50" not in df.columns or "training_time" not in df.columns:
            return figure(title="Performance vs Training Time")
        
        # Prepare data
        tradeoff_data = df[["mAP50", "mAP50_95", "training_time", "parameter", "fraction"]].dropna()
        
        if tradeoff_data.empty:
            return figure(title="No tradeoff data available")
        
        source = ColumnDataSource(tradeoff_data)
        
        p = figure(
            title="Performance vs Training Time Tradeoff",
            x_axis_label="Training Time (seconds)",
            y_axis_label="mAP50 Score",
            width=800,
            height=600
        )
        
        # Color by parameter
        parameters = tradeoff_data["parameter"].unique()
        colors = Category20[20]
        param_colors = {param: colors[i % 20] for i, param in enumerate(parameters)}
        
        color_map = [param_colors[p] for p in tradeoff_data["parameter"]]
        
        p.scatter(
            x="training_time", y="mAP50",
            source=source,
            size=8,
            color=color_map,
            alpha=0.7,
            legend_field="parameter"
        )
        
        # Add optimal frontier
        pareto_data = self._calculate_pareto_frontier(tradeoff_data)
        if not pareto_data.empty:
            pareto_source = ColumnDataSource(pareto_data)
            p.line(
                x="training_time", y="mAP50",
                source=pareto_source,
                color="red", line_width=3,
                legend_label="Pareto Optimal"
            )
        
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Parameter", "@parameter"),
            ("Fraction", "@fraction"),
            ("mAP50", "@mAP50"),
            ("mAP50-95", "@mAP50_95"),
            ("Training Time", "@training_time{0.0}")
        ])
        p.add_tools(hover)
        
        return p
    
    def _calculate_pareto_frontier(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Pareto optimal frontier."""
        # Sort by training time ascending, mAP descending
        sorted_df = df.sort_values(["training_time", "mAP50"], ascending=[True, False])
        
        pareto_points = []
        max_map = 0
        
        for _, row in sorted_df.iterrows():
            if row["mAP50"] > max_map:
                pareto_points.append(row)
                max_map = row["mAP50"]
        
        return pd.DataFrame(pareto_points)
    
    def create_parameter_heatmap(self, df: pd.DataFrame) -> figure:
        """Create heatmap of parameter performance."""
        if df.empty:
            return figure(title="Parameter Performance Heatmap")
        
        # Prepare heatmap data
        heatmap_data = []
        parameters = df["parameter"].dropna().unique()
        
        for param in parameters:
            param_df = df[df["parameter"] == param]
            for fraction in param_df["fraction"].unique():
                subset = param_df[param_df["fraction"] == fraction]
                if not subset.empty:
                    heatmap_data.append({
                        "parameter": param,
                        "fraction": fraction,
                        "performance": subset["mAP50"].mean()
                    })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        
        if heatmap_df.empty:
            return figure(title="No heatmap data available")
        
        # Create heatmap
        from bokeh.transform import linear_cmap
        from bokeh.palettes import Viridis256
        
        source = ColumnDataSource(heatmap_df)
        
        p = figure(
            title="Parameter Performance Heatmap",
            x_axis_label="Fraction",
            y_axis_label="Parameter",
            width=800,
            height=600
        )
        
        # Create color mapper
        color_mapper = linear_cmap(
            field_name="performance",
            palette=Viridis256,
            low=heatmap_df["performance"].min(),
            high=heatmap_df["performance"].max()
        )
        
        p.rect(
            x="fraction", y="parameter",
            width=0.1, height=1,
            source=source,
            fill_color=color_mapper,
            line_color="white"
        )
        
        # Add color bar
        color_bar = ColorBar(
            color_mapper=color_mapper["transform"],
            width=8, height=400,
            location="top_right"
        )
        p.add_layout(color_bar, "right")
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Parameter", "@parameter"),
            ("Fraction", "@fraction"),
            ("Performance", "@performance")
        ])
        p.add_tools(hover)
        
        return p
    
    def create_correlation_matrix(self, df: pd.DataFrame) -> figure:
        """Create correlation matrix of parameters and metrics."""
        if df.empty:
            return figure(title="Correlation Matrix")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return figure(title="Insufficient numeric data")
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Prepare data for heatmap
        corr_data = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                corr_data.append({
                    "x": col1,
                    "y": col2,
                    "correlation": corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_data)
        
        # Create plot
        source = ColumnDataSource(corr_df)
        
        p = figure(
            title="Parameter-Metric Correlation Matrix",
            x_range=numeric_cols,
            y_range=list(reversed(numeric_cols)),
            width=800,
            height=600
        )
        
        # Create color mapper
        color_mapper = linear_cmap(
            field_name="correlation",
            palette=Viridis256,
            low=-1, high=1
        )
        
        p.rect(
            x="x", y="y",
            width=1, height=1,
            source=source,
            fill_color=color_mapper,
            line_color="white"
        )
        
        # Add color bar
        color_bar = ColorBar(
            color_mapper=color_mapper["transform"],
            width=8, height=400,
            location="top_right"
        )
        p.add_layout(color_bar, "right")
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("X", "@x"),
            ("Y", "@y"),
            ("Correlation", "@correlation")
        ])
        p.add_tools(hover)
        
        return p
    
    def generate_all_plots(self, results_dir: Path, output_dir: Path, format: str = "html") -> List[Path]:
        """Generate all visualization plots."""
        if not self.load_results(results_dir):
            return []
        
        df = self.prepare_data()
        if df.empty:
            return []
        
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []
        
        # Generate plots
        plots = [
            ("parameter_impact", self.create_parameter_impact_plot(df)),
            ("performance_tradeoff", self.create_performance_tradeoff_plot(df)),
            ("parameter_heatmap", self.create_parameter_heatmap(df)),
            ("correlation_matrix", self.create_correlation_matrix(df))
        ]
        
        for name, plot in plots:
            if format == "html":
                output_path = output_dir / f"{name}.html"
                output_file(str(output_path))
                save(plot)
            else:
                # For other formats, use matplotlib backend
                from bokeh.io import export_png, export_svg
                output_path = output_dir / f"{name}.{format}"
                
                try:
                    if format == "png":
                        export_png(plot, filename=str(output_path))
                    elif format == "svg":
                        export_svg(plot, filename=str(output_path))
                except Exception as e:
                    self.logger.warning(f"Could not export {format}: {e}")
                    continue
            
            generated_files.append(output_path)
        
        return generated_files
    
    def create_dashboard(self, results_dir: Path) -> None:
        """Create interactive dashboard."""
        if not self.load_results(results_dir):
            return
        
        df = self.prepare_data()
        if df.empty:
            return
        
        # Create dashboard layout
        from bokeh.layouts import column, row
        
        # Create all plots
        impact_plot = self.create_parameter_impact_plot(df)
        tradeoff_plot = self.create_performance_tradeoff_plot(df)
        heatmap_plot = self.create_parameter_heatmap(df)
        correlation_plot = self.create_correlation_matrix(df)
        
        # Create dashboard layout
        layout = column(
            impact_plot,
            row(tradeoff_plot, heatmap_plot),
            correlation_plot
        )
        
        # Save dashboard
        dashboard_path = results_dir / "dashboard.html"
        output_file(str(dashboard_path))
        save(layout)
        
        return dashboard_path
    
    def launch_dashboard(self, results_dir: Path, port: int = 5006) -> None:
        """Launch interactive Bokeh dashboard."""
        if not self.load_results(results_dir):
            return
        
        def modify_doc(doc):
            df = self.prepare_data()
            
            # Create dashboard
            impact_plot = self.create_parameter_impact_plot(df)
            tradeoff_plot = self.create_performance_tradeoff_plot(df)
            heatmap_plot = self.create_parameter_heatmap(df)
            correlation_plot = self.create_correlation_matrix(df)
            
            layout = column(
                impact_plot,
                row(tradeoff_plot, heatmap_plot),
                correlation_plot
            )
            
            doc.add_root(layout)
            doc.title = "YOLO Benchmark Dashboard"
        
        # Start server
        apps = {'/': Application(FunctionHandler(modify_doc))}
        server = Server(apps, port=port)
        server.start()
        
        print(f"Dashboard available at: http://localhost:{port}")
        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()


class InteractiveDashboard:
    """Interactive dashboard for exploring results."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.visualizer = ResultsVisualizer()
        self.df = None
    
    def create_interactive_layout(self):
        """Create fully interactive dashboard."""
        if not self.visualizer.load_results(self.results_dir):
            return None
        
        self.df = self.visualizer.prepare_data()
        if self.df.empty:
            return None
        
        # Create widgets
        parameters = list(self.df["parameter"].dropna().unique())
        
        parameter_select = Select(title="Parameter:", value=parameters[0], options=parameters)
        
        # Create plots
        impact_plot = self.visualizer.create_parameter_impact_plot(self.df)
        
        # Update function
        def update_plot(attr, old, new):
            param = parameter_select.value
            filtered_df = self.df[self.df["parameter"] == param]
            
            # Update plots with filtered data
            # This would require more complex implementation
            pass
        
        parameter_select.on_change('value', update_plot)
        
        # Layout
        from bokeh.layouts import column
        layout = column(parameter_select, impact_plot)
        
        return layout
