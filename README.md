# YOLO Benchmark Project

This project benchmarks how various parameters affect YOLO model performance. It uses the MVC pattern with organized folders under `src/`.

## Structure

- `src/core/`: Backend logic for training, benchmarking, and parameter handling.
- `src/interface/`: TUI interface using Rich for user interaction and configuration.
- `src/models/`: Storage for trained YOLO models.
- `src/results/`: Benchmark results and graph generation using Bokeh.

## Installation

1. Install dependencies: `pip install -r requirements.txt`
2. Run the interface: `python src/interface/main.py`

## Usage

Use the TUI to configure parameters and run benchmarks. Results will be saved in `src/results/` with interactive graphs.

## Parameters Benchmarked

- Epochs
- Batch size
- Image size
- Learning rate
- Data quantity (subsets)

Each parameter is varied in fractions (e.g., 1/5, 2/5, ..., 5/5 of max value) while others are fixed.
