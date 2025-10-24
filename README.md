# 🔬 YOLO Hyperparameter Benchmark System

> **Investigação Sistemática da Influência de Hiperparâmetros no Treinamento do YOLOv8**  
> Projeto de Pesquisa Acadêmica - UNIP 2025

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Sobre o Projeto

Este sistema implementa uma metodologia rigorosa de **benchmark fracionado** para investigação empírica do impacto de hiperparâmetros no treinamento do YOLOv8m, seguindo a metodologia descrita no artigo acadêmico:

**"Influência de Hiperparâmetros no Treinamento do YOLO"** (UNIP, 2025)

### Objetivos Científicos

1. **Quantificar** o efeito individual de cada hiperparâmetro nas métricas de desempenho
2. **Identificar** interações e trade-offs entre hiperparâmetros
3. **Estabelecer** relações entre configuração, performance (mAP) e custo computacional

### Metodologia: Benchmark Fracionado

O sistema testa cada hiperparâmetro em valores correspondentes a **frações** (1/N, 2/N, ..., N/N) de um valor máximo:

```
Variável: epochs
Valor Máximo: 200
Frações: 5

Testes:
├─ 1/5 → 40 epochs
├─ 2/5 → 80 epochs
├─ 3/5 → 120 epochs  
├─ 4/5 → 160 epochs
└─ 5/5 → 200 epochs
```

## 🏗️ Arquitetura

O projeto segue rigorosamente o padrão **MVC** (Model-View-Controller):

```
src/
├── cli/                      # View & Controller
│   ├── main.py              # Entry point
│   ├── menu.py              # Menu principal
│   ├── config_menu.py       # Configuração de hiperparâmetros
│   ├── benchmark_menu.py    # Interface de benchmark
│   ├── progress.py          # Progress displays (Rich)
│   └── visualizer.py        # Visualização de configs
│
├── cli/core/                # Model
│   ├── config.py            # Modelos Pydantic
│   ├── trainer.py           # Treinamento YOLO
│   ├── benchmark.py         # Motor de benchmark
│   ├── metrics.py           # Coleta de métricas
│   ├── logger.py            # Sistema de logging
│   └── utils.py             # Utilitários
│
├── visualization/           # Visualizações Bokeh
│   └── bokeh_plots.py       # Gráficos interativos
│
├── models/                  # Modelos treinados
│   ├── checkpoints/
│   └── best_models/
│
└── results/                 # Resultados
    ├── benchmarks/          # JSONs
    ├── graphs/              # HTMLs Bokeh
    ├── reports/             # Relatórios
    └── logs/                # Logs
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.8+
- CUDA 11.8+ (para GPU NVIDIA) ou
- Apple Silicon com MPS (M1/M2)

### Passo a Passo

```bash
# 1. Clone o repositório
git clone <repository-url>
cd trabalhoPDI

# 2. Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Instale PyTorch (escolha sua plataforma)

# GPU NVIDIA (CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon (MPS):
pip install torch torchvision

# CPU apenas:
pip install torch torchvision

# 4. Instale dependências
pip install -r requirements.txt

# 5. Configure o dataset
# Edite config.yaml e ajuste dataset_path para seu data.yaml
```

## 💻 Uso

### Início Rápido

```bash
python -m src.cli.main --dataset /path/to/data.yaml
```

### Opções de Linha de Comando

```bash
# Especificar configuração custom
python -m src.cli.main --config my_config.yaml

# Usar modelo diferente
python -m src.cli.main --model yolov8l.pt

# Modo verbose
python -m src.cli.main --verbose

# Forçar uso de CPU
python -m src.cli.main --no-gpu
```

### Interface CLI

O sistema oferece um menu interativo completo:

```
╔══════════════════════════════════════════════════════════════╗
║        🔬 YOLO Hyperparameter Benchmark System v1.0         ║
║              Pesquisa UNIP - Processamento de Imagem         ║
╚══════════════════════════════════════════════════════════════╝

┌─ Menu Principal ─────────────────────────────────────────────┐
│  [1] 🎯 Treinar modelo único                                 │
│  [2] 📊 Executar Benchmark Fracionado                        │
│  [3] 📈 Visualizar Resultados (Bokeh)                        │
│  [4] ⚙️  Configurar Hiperparâmetros                          │
│  [5] 💾 Salvar/Carregar Configuração                         │
│  [6] ℹ️  Informações do Sistema                              │
│  [7] 📖 Ajuda                                                │
│  [0] ❌ Sair                                                 │
└──────────────────────────────────────────────────────────────┘
```

## 📊 Hiperparâmetros Suportados

O sistema permite benchmarkar **TODOS** estes hiperparâmetros:

### Otimização
- `lr0` - Taxa de aprendizado inicial
- `lrf` - Taxa de aprendizado final
- `momentum` - Momentum do SGD
- `weight_decay` - Regularização L2
- `warmup_epochs` - Épocas de warmup
- `optimizer` - Algoritmo (SGD/Adam/AdamW/RMSProp)

### Batch
- `batch` - Tamanho do batch
- `accumulate` - Gradient accumulation

### Arquitetura
- `imgsz` - Resolução da imagem
- `epochs` - Número de épocas

### Augmentação de Dados
- `hsv_h`, `hsv_s`, `hsv_v` - Augmentações HSV
- `degrees`, `translate`, `scale`, `shear` - Geométricas
- `mosaic`, `mixup`, `copy_paste` - Avançadas
- `fliplr`, `flipud` - Flips

### Regularização
- `label_smoothing` - Suavização de labels
- `dropout` - Dropout

### Pós-processamento
- `conf` - Confidence threshold
- `iou` - IoU threshold NMS

## 📈 Métricas Coletadas

### Performance
- **mAP@0.5** - Mean Average Precision @ IoU 0.5
- **mAP@0.5:0.95** - mAP em múltiplos IoUs
- **Precision** - Precisão
- **Recall** - Revocação  
- **F1-Score** - Média harmônica

### Operacionais
- **time_per_epoch** - Tempo médio por época
- **total_train_time** - Tempo total
- **inference_time** - Latência de inferência
- **memory_peak** - Pico de memória
- **memory_avg** - Memória média
- **gpu_utilization** - Utilização GPU

## 📊 Visualizações Bokeh

O sistema gera dashboards HTML interativos com:

1. **Métricas vs Frações** - Linhas mostrando evolução das métricas
2. **Performance vs Custo** - Scatter plot (mAP vs tempo)
3. **Ranking** - Barras horizontais ordenadas por mAP
4. **Comparação F1** - Barras comparando F1-Scores
5. **Curvas de Loss** - Evolução do loss

Todos os gráficos são **totalmente interativos** com:
- Hover tooltips detalhados
- Pan & Zoom
- Exportação de imagens
- Legendas clicáveis

## 🔬 Exemplo de Workflow

```python
# 1. Executar benchmark de epochs
Menu → [2] Benchmark Fracionado
→ Variável: epochs
→ Valor máximo: 200
→ Frações: 5

# Sistema executa 5 treinos:
# 40, 80, 120, 160, 200 epochs

# 2. Visualizar resultados
Menu → [3] Visualizar Resultados
→ Selecionar benchmark
→ Dashboard HTML gerado em src/results/graphs/

# 3. Comparar com outro hiperparâmetro
Menu → [2] Benchmark Fracionado
→ Variável: lr0
→ Valor máximo: 0.1
→ Frações: 5

# 4. Comparar benchmarks
Menu → [3] Visualizar → Comparar
```

## 📁 Estrutura de Resultados

```
src/results/
├── benchmarks/
│   ├── epochs_20250117_143022_abc123.json
│   └── lr0_20250117_150030_def456.json
│
├── graphs/
│   ├── dashboard_epochs_20250117_143500.html
│   └── comparison_epochs_lr0_20250117_151000.html
│
├── reports/
│   └── summary_report_20250117.md
│
└── logs/
    └── yolo_benchmark_20250117_140000.log
```

## 🎯 Boas Práticas para Pesquisa

1. **Reprodutibilidade**
   - Sempre use `seed` fixo (padrão: 42)
   - Documente versões de bibliotecas
   - Salve configurações em YAML

2. **Controle de Variáveis**
   - Varie apenas 1 hiperparâmetro por vez
   - Use configuração base consistente
   - Repita experimentos críticos

3. **Documentação**
   - Mantenha logs detalhados
   - Anote observações qualitativas
   - Relacione com teoria

4. **Análise Estatística**
   - Compare resultados com baseline
   - Calcule intervalos de confiança se possível
   - Identifique outliers

## 🐛 Troubleshooting

### Erro: "CUDA out of memory"
```bash
# Reduza batch size
Menu → [4] Configurar → [2] Batch → batch: 8
```

### Treinamento muito lento
```bash
# Verifique GPU
Menu → [6] Informações do Sistema
# Se não houver GPU, considere reduzir imgsz e epochs
```

### Dataset não encontrado
```bash
# Verifique config.yaml
dataset_path: "/caminho/completo/para/data.yaml"
```

## 📚 Referências

Este projeto implementa conceitos de:

1. **YOLOv8**: Ultralytics YOLO  
   [https://docs.ultralytics.com](https://docs.ultralytics.com)

2. **Fractional Factorial Design**: Montgomery, D. C. (2017)  
   "Design and Analysis of Experiments"

3. **Hyperparameter Optimization**: Bergstra & Bengio (2012)  
   "Random Search for Hyper-Parameter Optimization"

## 👥 Autores

**Equipe de Pesquisa UNIP**  
Projeto TCC - Processamento de Imagem  
2025

## 📄 Licença

Este projeto é desenvolvido para fins acadêmicos.

## 🤝 Contribuições

Para dúvidas ou sugestões sobre a metodologia, consulte o artigo de referência ou entre em contato com a equipe de pesquisa.

---

**Desenvolvido com** 🔬 **para pesquisa acadêmica rigorosa em Computer Vision**
