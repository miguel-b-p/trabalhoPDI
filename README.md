# 🚀 YOLO Benchmark System

Sistema completo de benchmark e testes de parâmetros YOLO para análise de impacto no desempenho de modelos de detecção de objetos.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Características](#características)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Uso Rápido](#uso-rápido)
- [Documentação](#documentação)
- [Exemplos](#exemplos)

## 🎯 Visão Geral

Este projeto permite testar sistematicamente como diferentes parâmetros do YOLO (epochs, batch size, learning rate, etc.) afetam o desempenho final do modelo. O sistema implementa um mecanismo inteligente de benchmark que divide o intervalo de cada parâmetro em frações (1/5, 2/5, 3/5, 4/5, 5/5) para análise detalhada.

## ✨ Características

- **🏗️ Arquitetura MVC**: Código organizado e modular
- **📊 Visualizações Interativas**: Gráficos com Bokeh
- **🎮 Interface Jupyter**: Widgets interativos para controle total
- **📈 Análise Estatística**: Correlações e rankings de impacto
- **⚡ Execução Paralela**: Suporte para múltiplos workers
- **💾 Persistência**: Salva todos os resultados em JSON/CSV
- **📝 Relatórios Automáticos**: Geração de relatórios detalhados
- **🔧 Altamente Configurável**: Todos os parâmetros YOLO disponíveis

## 📁 Estrutura do Projeto

```
trabalhoPDI/
├── src/
│   ├── core/                 # Backend e lógica principal
│   │   ├── __init__.py
│   │   ├── config.py        # Configurações do sistema
│   │   ├── trainer.py       # Treinador YOLO
│   │   ├── benchmark.py     # Sistema de benchmark
│   │   └── data_manager.py  # Gerenciamento de datasets
│   │
│   ├── interface/           # Interface interativa
│   │   ├── __init__.py
│   │   └── jupyter_interface.py  # Interface Jupyter com widgets
│   │
│   ├── results/             # Análise e visualização
│   │   ├── __init__.py
│   │   ├── visualizer.py    # Visualizações Bokeh
│   │   └── analyzer.py      # Análise estatística
│   │
│   ├── models/              # Modelos treinados
│   └── data/                # Datasets
│
├── requirements.txt         # Dependências
├── README.md               # Este arquivo
└── notebooks/              # Notebooks de exemplo
    ├── 01_quick_start.ipynb
    ├── 02_training.ipynb
    ├── 03_benchmark.ipynb
    └── 04_analysis.ipynb
```

## 🔧 Instalação

### 1. Clone o repositório

```bash
cd /home/mingas/projetos/trabalhoPDI
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. (Opcional) Instale Euporie para terminal Jupyter

```bash
pip install euporie
```

## 🚀 Uso Rápido

### Interface Jupyter (Recomendado)

```python
from src.interface import YOLOBenchmarkInterface

# Inicializa interface
interface = YOLOBenchmarkInterface()

# Exibe interface interativa
interface.show()
```

### API Python

#### Treinamento Individual

```python
from src.core import Config, YOLOTrainer

# Configura
config = Config()
config.epochs = 100
config.batch_size = 16
config.imgsz = 640

# Treina
trainer = YOLOTrainer(config)
trainer.load_model('yolo11n.pt')
metrics = trainer.train(data='/path/to/data.yaml')

print(f"mAP50-95: {metrics['mAP50-95']:.2f}%")
```

#### Benchmark de Parâmetros

```python
from src.core import Config, BenchmarkConfig, BenchmarkRunner

# Configuração base
config = Config()

# Configuração do benchmark
benchmark_config = BenchmarkConfig()
benchmark_config.benchmark_name = "epochs_benchmark"
benchmark_config.num_divisions = 5

# Define parâmetros para testar
benchmark_config.benchmark_params = {
    'epochs': {'min': 10, 'max': 100, 'type': 'int'},
    'batch_size': {'min': 4, 'max': 32, 'type': 'int'},
    'lr0': {'min': 0.001, 'max': 0.1, 'type': 'float'}
}

# Executa benchmark
runner = BenchmarkRunner(config, benchmark_config)
results = runner.run_benchmark(
    dataset_path='/path/to/data.yaml',
    parallel=False
)

print(f"Testes concluídos: {results['successful_tests']}/{results['total_tests']}")
```

#### Visualização de Resultados

```python
from src.results import BenchmarkVisualizer, ResultsAnalyzer

# Carrega resultados
analyzer = ResultsAnalyzer(config.results_path)
benchmark_data = analyzer.load_benchmark('epochs_benchmark')

# Cria visualizações
visualizer = BenchmarkVisualizer(config.results_path)
output_html = visualizer.visualize_benchmark(benchmark_data)

print(f"Visualização salva em: {output_html}")

# Gera relatório
report = analyzer.generate_report(benchmark_data)
print(report)
```

## 📖 Documentação Detalhada

### Sistema de Benchmark

O sistema de benchmark funciona da seguinte forma:

1. **Definição de Parâmetros**: Você especifica quais parâmetros testar e seus intervalos (min/max)
2. **Divisões Automáticas**: O sistema divide cada intervalo em N partes iguais (padrão: 5)
   - 1/5: 20% do intervalo
   - 2/5: 40% do intervalo
   - 3/5: 60% do intervalo
   - 4/5: 80% do intervalo
   - 5/5: 100% do intervalo (valor máximo)
3. **Execução**: Cada combinação é testada independentemente
4. **Análise**: Gera estatísticas, correlações e rankings de impacto

### Parâmetros Disponíveis

#### Treinamento Básico
- `epochs`: Número de épocas (padrão: 100)
- `batch_size`: Tamanho do batch (padrão: 16)
- `imgsz`: Tamanho da imagem (padrão: 640)
- `device`: Dispositivo (cuda, cpu, mps)

#### Otimização
- `optimizer`: Otimizador (SGD, Adam, AdamW)
- `lr0`: Learning rate inicial (padrão: 0.01)
- `lrf`: Learning rate final (padrão: 0.01)
- `momentum`: Momentum (padrão: 0.937)
- `weight_decay`: Weight decay (padrão: 0.0005)

#### Data Augmentation
- `augment`: Habilita/desabilita augmentation
- `hsv_h`, `hsv_s`, `hsv_v`: Augmentation HSV
- `degrees`: Rotação
- `translate`: Translação
- `scale`: Escala
- `fliplr`: Flip horizontal
- `mosaic`: Mosaic augmentation
- `mixup`: Mixup augmentation

### Métricas Analisadas

- **mAP50**: Mean Average Precision @ IoU 0.50
- **mAP50-95**: Mean Average Precision @ IoU 0.50:0.95
- **Precision**: Precisão do modelo
- **Recall**: Recall do modelo
- **Training Time**: Tempo de treinamento

## 📊 Visualizações

O sistema gera visualizações interativas com Bokeh:

1. **Visão Geral**: Estatísticas do benchmark
2. **Análise por Parâmetro**: Impacto individual de cada parâmetro
3. **Comparação de Métricas**: Scatter plots e comparações
4. **Timeline**: Tempo de execução dos testes

## 🎓 Exemplos

### Exemplo 1: Benchmark de Epochs

```python
benchmark_config = BenchmarkConfig()
benchmark_config.benchmark_params = {
    'epochs': {'min': 10, 'max': 100, 'type': 'int'}
}
benchmark_config.num_divisions = 5

# Testa: 10, 28, 46, 64, 82, 100 epochs
```

### Exemplo 2: Benchmark de Múltiplos Parâmetros

```python
benchmark_config = BenchmarkConfig()
benchmark_config.benchmark_params = {
    'epochs': {'min': 10, 'max': 50, 'type': 'int'},
    'batch_size': {'min': 8, 'max': 32, 'type': 'int'},
    'optimizer': {'values': ['SGD', 'Adam', 'AdamW'], 'type': 'categorical'}
}

# Total de testes: 5 (epochs) + 5 (batch_size) + 3 (optimizer) = 13 testes
```

### Exemplo 3: Análise de Impacto

```python
analyzer = ResultsAnalyzer(config.results_path)
benchmark_data = analyzer.load_benchmark('my_benchmark')

# Analisa impacto de um parâmetro
impact = analyzer.analyze_parameter_impact(benchmark_data, 'epochs', 'mAP50-95')
print(f"Correlação: {impact['correlation']['coefficient']:.3f}")
print(f"Melhor valor: {impact['best']['value']} (Score: {impact['best']['score']:.2f}%)")

# Ranking de parâmetros
rankings = analyzer.rank_parameters(benchmark_data)
for i, rank in enumerate(rankings, 1):
    print(f"{i}. {rank['parameter']} (Impact: {rank['impact_score']:.3f})")

# Configuração ótima
optimal = analyzer.find_optimal_configuration(benchmark_data)
print(f"Configuração ótima: {optimal['parameter']}={optimal['value']}")
```

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT.

## 👥 Autores

- **YOLO Benchmark Team**

## 🙏 Agradecimentos

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Bokeh](https://bokeh.org/)
- [Jupyter Project](https://jupyter.org/)

## 📞 Suporte

Para questões e suporte, abra uma issue no repositório do projeto.

---

**Nota**: Este projeto foi desenvolvido seguindo as melhores práticas de programação Python e padrão MVC para garantir código limpo, organizado e de fácil manutenção.
