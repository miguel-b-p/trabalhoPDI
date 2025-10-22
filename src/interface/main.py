"""
TUI interface using Textual for YOLO benchmark configuration and execution.
"""

import sys
sys.path.insert(0, '/home/mingas/projetos/trabalhoPDI')

from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Header, Footer, Input, Button, Static, Label
from textual import events
from src.core.config import BenchmarkConfig
from src.core.benchmark import BenchmarkRunner
from src.results.plotter import ResultsPlotter
from pathlib import Path


class BenchmarkApp(App):
    """Textual app for YOLO benchmark."""

    CSS = """
    Screen {
        align: center middle;
    }

    #config {
        width: 80%;
        height: 80%;
        border: solid green;
    }

    Input {
        margin: 1;
    }

    Button {
        margin: 1;
    }
    """

    def __init__(self):
        super().__init__()
        self.config = BenchmarkConfig()

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="config"):
            yield Label("YOLO Benchmark Configuration")
            yield Input(placeholder="Model path", value=self.config.model, id="model")
            yield Input(placeholder="Data YAML", value=self.config.data, id="data")
            yield Input(placeholder="Max Epochs", value=str(self.config.max_epochs), id="max_epochs")
            yield Input(placeholder="Max Batch Size", value=str(self.config.max_batch_size), id="max_batch_size")
            yield Input(placeholder="Max Image Size", value=str(self.config.max_image_size), id="max_image_size")
            yield Input(placeholder="Max Learning Rate", value=str(self.config.max_learning_rate), id="max_learning_rate")
            yield Input(placeholder="Quantity (steps)", value=str(self.config.quantity), id="quantity")
            yield Input(placeholder="Device", value=self.config.device, id="device")
            yield Button("Run Benchmark", id="run")
            yield Static("", id="status")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run":
            self.update_config()
            self.run_benchmark()

    def update_config(self):
        self.config.model = self.query_one("#model", Input).value
        self.config.data = self.query_one("#data", Input).value
        self.config.max_epochs = int(self.query_one("#max_epochs", Input).value)
        self.config.max_batch_size = int(self.query_one("#max_batch_size", Input).value)
        self.config.max_image_size = int(self.query_one("#max_image_size", Input).value)
        self.config.max_learning_rate = float(self.query_one("#max_learning_rate", Input).value)
        self.config.quantity = int(self.query_one("#quantity", Input).value)
        self.config.device = self.query_one("#device", Input).value

    def run_benchmark(self):
        status = self.query_one("#status", Static)
        status.update("Running benchmarks... This may take a while.")

        runner = BenchmarkRunner(self.config)
        results = runner.run_all_benchmarks()

        plotter = ResultsPlotter(Path(__file__).parent.parent / 'results')
        plotter.plot_all(results)

        status.update("Benchmarks completed. Check results/ for outputs.")


if __name__ == "__main__":
    app = BenchmarkApp()
    app.run()
