# 🎯 YOLO Hyperparameter Benchmarking System

A comprehensive system for analyzing how YOLO hyperparameters affect model performance, based on the research:

**"Influência de Hiperparâmetros no Treinamento do YOLO"**  
*UNIVERSIDADE PAULISTA - Processamento de Imagem*  
*Author: Miguel Batista Pinotti*

## 📋 Overview

This system provides a systematic approach to benchmark YOLOv8m hyperparameters and understand their impact on model performance. It implements fractional testing methodology where each parameter is tested across multiple fractions of its range (1/5, 2/5, 3/5, 4/5, 5/5).

## 🚀 Features

- **Systematic Benchmarking**: Fractional testing across hyperparameter ranges
- **Rich CLI Interface**: Beautiful terminal interface with interactive configuration
- **Real-time Monitoring**: CPU, GPU, and memory usage tracking
- **Comprehensive Visualization**: Interactive Bokeh charts and dashboards
- **MVC Architecture**: Clean separation of concerns with organized code structure
- **Performance Analysis**: Detailed metrics and correlation analysis
- **Reproducible Results**: Fixed random seeds and consistent data splits

## 🏗️ Architecture

```
src/
├── config/              # Configuration management
│   ├── yolo_config.py   # YOLO training parameters
│   ├── benchmark_config.py  # Benchmark settings
│   └── parameters.py    # Hyperparameter space definitions
├── cli/                 # CLI interface (MVC View)
│   ├── main.py          # Main CLI entry point
│   ├── commands.py      # CLI commands
│   └── core/            # Backend logic (MVC Controller)
│       ├── trainer.py   # YOLO training wrapper
│       ├── benchmark.py # Benchmark orchestrator
│       ├── monitor.py   # System monitoring
│       └── metrics.py   # Metrics collection
├── results_visualizer.py # Results visualization (MVC View)
└── utils/               # Utility functions
```

## 📊 Supported Hyperparameters

Based on the research article, the system benchmarks these key parameters:

### Learning Parameters
- **lr0**: Initial learning rate (0.0001 - 0.1)
- **lrf**: Final learning rate factor (0.01 - 1.0)
- **momentum**: SGD momentum (0.6 - 0.98)
- **weight_decay**: L2 regularization (0.0001 - 0.001)

### Training Configuration
- **batch**: Batch size (4 - 64)
- **epochs**: Training epochs (50 - 300)
- **imgsz**: Image size (320 - 1280)

### Data Augmentation
- **mosaic**: Mosaic augmentation (0.0 - 1.0)
- **mixup**: Mixup augmentation (0.0 - 1.0)
- **degrees**: Rotation degrees (0.0 - 45.0)
- **scale**: Scale augmentation (0.0 - 0.5)
- **hsv_h/s/v**: HSV color augmentation

### Loss Parameters
- **box**: Box loss gain (0.02 - 0.2)
- **cls**: Classification loss gain (0.2 - 4.0)
- **dfl**: Distribution focal loss gain (0.4 - 6.0)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev python3-pip

# macOS
brew install python3

# Windows
# Install Python from python.org
```

## 🚀 Quick Start

### 1. Interactive Setup
```bash
python -m src.cli.main quickstart
```

### 2. Run Benchmark
```bash
# Basic benchmark
python -m src.cli.main benchmark

# With custom configuration
python -m src.cli.main benchmark --config config/custom.yaml

# Dry run to see experiments
python -m src.cli.main benchmark --dry-run

# Parallel execution
python -m src.cli.main benchmark --max-workers 4
```

### 3. Analyze Results
```bash
# Interactive dashboard
python -m src.cli.main analyze --interactive

# Generate static reports
python -m src.cli.main analyze --format html --output analysis/
```

## 📈 Usage Examples

### Configuration Management
```bash
# Create new configuration
python -m src.cli.main config --create

# Edit existing configuration
python -m src.cli.main config --edit config/benchmark.yaml

# Validate configuration
python -m src.cli.main config --validate config/benchmark.yaml

# Show defaults
python -m src.cli.main config --show-defaults
```

### Custom Parameter Testing
```bash
# Test specific parameters
python -m src.cli.main benchmark --parameter lr0 batch epochs

# Custom fractions
python -m src.cli.main benchmark --fractions 0.25,0.5,0.75,1.0
```

## 📊 Visualization Features

### Available Plots
1. **Parameter Impact Analysis**: Line plots showing performance vs parameter values
2. **Performance Tradeoff**: Scatter plots of accuracy vs training time
3. **Parameter Heatmap**: Color-coded performance across parameters and fractions
4. **Correlation Matrix**: Relationships between parameters and metrics
5. **Interactive Dashboard**: Web-based exploration tool

### Dashboard Access
```bash
# Launch interactive dashboard
python -m src.cli.main analyze --interactive --port 8080
```

Access at: http://localhost:8080

## 📋 Configuration

### Example Configuration File
```yaml
# config/benchmark.yaml
name: "yolo_hyperparameter_study"
description: "Comprehensive YOLOv8m hyperparameter analysis"

# Dataset
dataset_path: "data/coco128.yaml"

# Model
base_model: "yolov8m.pt"
model_size: "m"

# Benchmark parameters
fractions: [0.2, 0.4, 0.6, 0.8, 1.0]
selected_parameters:
  - lr0
  - batch
  - epochs
  - momentum
  - mosaic
  - mixup
repetitions: 3

# Resource limits
max_epochs: 100
max_batch_size: 32
memory_limit_gb: 8.0
max_time_hours: 24.0

# Output
output_dir: "results"
save_models: true
generate_plots: true
generate_report: true

# Random seed for reproducibility
seed: 42
```

## 🔧 Advanced Usage

### Programmatic Usage
```python
from src.config import BenchmarkConfig, YOLOConfig
from src.cli.core import BenchmarkOrchestrator

# Create configuration
config = BenchmarkConfig()
config.selected_parameters = ['lr0', 'batch', 'epochs']
config.repetitions = 5

# Run benchmark
orchestrator = BenchmarkOrchestrator(config)
results = orchestrator.run_benchmark(max_workers=2)

# Analyze results
from src.results_visualizer import ResultsVisualizer
visualizer = ResultsVisualizer()
visualizer.generate_all_plots(Path("results"), Path("analysis"))
```

### Custom Parameter Ranges
```python
from src.config.parameters import HyperparameterSpace

param_space = HyperparameterSpace()
param_space.parameters['lr0'].min_value = 0.001
param_space.parameters['lr0'].max_value = 0.05
```

## 📊 Output Structure

```
results/
├── benchmark_YYYYMMDD_HHMMSS/
│   ├── experiment_plan.json
│   ├── benchmark_summary.json
│   ├── parameter_analysis.json
│   ├── benchmark_results.csv
│   ├── experiment_*/
│   │   ├── experiment_results.json
│   │   ├── config.yaml
│   │   └── weights/
│   └── analysis/
│       ├── parameter_impact.html
│       ├── performance_tradeoff.html
│       ├── parameter_heatmap.html
│       └── correlation_matrix.html
```

## 🎯 Research Methodology

### Fractional Testing Approach
The system implements the fractional testing methodology described in the research:

1. **Parameter Range Definition**: Each parameter has defined min/max values
2. **Fractional Sampling**: Test at 20%, 40%, 60%, 80%, and 100% of range
3. **Statistical Significance**: Multiple repetitions with different random seeds
4. **Performance Metrics**: mAP@0.5, mAP@0.5:0.95, precision, recall, F1-score
5. **Efficiency Metrics**: Training time, inference latency, memory usage

### Analysis Features
- **Parameter Sensitivity**: Impact analysis for each hyperparameter
- **Correlation Analysis**: Relationships between parameters and performance
- **Pareto Optimization**: Tradeoff analysis between accuracy and efficiency
- **Statistical Significance**: Confidence intervals and significance testing

## 🔍 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size in configuration
python -m src.cli.main config --edit config/benchmark.yaml
# Set max_batch_size to smaller value
```

#### Slow Training
```bash
# Reduce epochs or use smaller model
# Check GPU utilization with nvidia-smi
```

#### Missing Dependencies
```bash
pip install --upgrade ultralytics rich bokeh pandas numpy
```

### Performance Optimization
- Use SSD storage for datasets
- Enable GPU acceleration
- Adjust batch size based on GPU memory
- Use parallel processing with appropriate worker count

## 📚 Research References

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **Bokeh Visualization**: https://docs.bokeh.org/
- **Rich CLI Library**: https://rich.readthedocs.io/
- **Research Paper**: "Influência de Hiperparâmetros no Treinamento do YOLO"

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
- Create GitHub issue
- Contact: miguel.pinotti@unip.br
- Research Group: UNIVERSIDADE PAULISTA - Processamento de Imagem

---

**Built with ❤️ for computer vision research**
