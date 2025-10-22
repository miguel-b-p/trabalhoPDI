"""
Results plotting using Bokeh.
"""

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column
from pathlib import Path
import json
from typing import List, Dict, Any


class ResultsPlotter:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir

    def plot_benchmark(self, param: str, results: List[Dict[str, Any]]):
        """Plot metrics vs parameter value."""
        output_file(self.results_dir / f'plot_{param}.html')

        p = figure(title=f"Benchmark: {param} vs Metrics", x_axis_label=param, y_axis_label="Value")

        values = [r['value'] for r in results]
        map50 = [r['metrics']['mAP50'] for r in results]
        map95 = [r['metrics']['mAP50-95'] for r in results]
        precision = [r['metrics']['precision'] for r in results]
        recall = [r['metrics']['recall'] for r in results]

        p.line(values, map50, legend_label="mAP50", line_width=2, color="blue")
        p.line(values, map95, legend_label="mAP50-95", line_width=2, color="green")
        p.line(values, precision, legend_label="Precision", line_width=2, color="red")
        p.line(values, recall, legend_label="Recall", line_width=2, color="orange")

        save(p)

    def plot_all(self, all_results: Dict[str, List[Dict[str, Any]]]):
        """Plot all benchmarks."""
        for param, results in all_results.items():
            self.plot_benchmark(param, results)
