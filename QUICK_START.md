# 🚀 Guia de Início Rápido

## Instalação (5 minutos)

### Opção 1: Script Automático (Recomendado)

```bash
# Torna o script executável e executa
chmod +x install.sh
./install.sh
```

### Opção 2: Manual

```bash
# Cria ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instala dependências
pip install -r requirements.txt

# Cria diretórios
mkdir -p src/{data,models,results/benchmarks} notebooks
```

## Verificação do Sistema

```bash
# Verifica se tudo está funcionando
python3 check_system.py
```

Você deve ver algo como:
```
✅ Python 3.10.0
✅ Todas as dependências principais instaladas
✅ CUDA disponível
✅ Sistema pronto para uso!
```

## Primeiro Uso

### 1. Preparar seu Dataset

Seu dataset deve estar no formato YOLO com um arquivo `data.yaml`:

```yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images  # opcional

nc: 3  # número de classes
names: ['class1', 'class2', 'class3']
```

### 2. Opção A: Interface Interativa (Mais Fácil)

Abra Jupyter:
```bash
jupyter lab
```

Abra `notebooks/01_quick_start.ipynb` e execute:

```python
from src.interface import YOLOBenchmarkInterface

interface = YOLOBenchmarkInterface()
interface.show()
```

Agora você tem uma interface completa com abas para:
- 🎯 Treinamento individual
- 🏁 Benchmark de parâmetros
- 📊 Visualização de resultados
- ⚙️ Configurações

### 2. Opção B: Código Python

```python
from src.core import Config, BenchmarkConfig, BenchmarkRunner

# Configuração
config = Config()
config.model_name = 'yolo11n.pt'
config.epochs = 50
config.batch_size = 16

# Benchmark
benchmark_config = BenchmarkConfig()
benchmark_config.benchmark_name = 'meu_primeiro_benchmark'
benchmark_config.benchmark_params = {
    'epochs': {'min': 10, 'max': 100, 'type': 'int'},
    'batch_size': {'min': 8, 'max': 32, 'type': 'int'},
}

# Executa
runner = BenchmarkRunner(config, benchmark_config)
results = runner.run_benchmark(
    dataset_path='/path/to/data.yaml',
    parallel=False
)

print(f"Concluído! {results['successful_tests']} testes bem-sucedidos")
```

## Exemplos Prontos

Execute para ver exemplos de uso:
```bash
python3 example_usage.py
```

Ou explore os notebooks:
- `notebooks/01_quick_start.ipynb` - Introdução
- `notebooks/02_training.ipynb` - Treinamento individual
- `notebooks/03_benchmark.ipynb` - Sistema de benchmark
- `notebooks/04_analysis.ipynb` - Análise de resultados

## Como Funciona o Benchmark?

O sistema testa cada parâmetro em 5 divisões (por padrão):

**Exemplo**: `epochs` de 10 a 100
- Teste 1: 10 + (100-10) × 1/5 = **28 epochs**
- Teste 2: 10 + (100-10) × 2/5 = **46 epochs**
- Teste 3: 10 + (100-10) × 3/5 = **64 epochs**
- Teste 4: 10 + (100-10) × 4/5 = **82 epochs**
- Teste 5: 10 + (100-10) × 5/5 = **100 epochs**

## Resultados

Após executar um benchmark, você terá:

```
src/results/benchmarks/meu_primeiro_benchmark/
├── benchmark_results.json      # Dados completos
├── benchmark_results.csv       # Para análise em Excel/Pandas
├── meu_primeiro_benchmark_visualization.html  # Visualização interativa
├── report.txt                  # Relatório textual
└── intermediate/               # Resultados parciais
```

## Visualização dos Resultados

### No Jupyter

```python
from src.results import BenchmarkVisualizer, ResultsAnalyzer

analyzer = ResultsAnalyzer(config.results_path)
visualizer = BenchmarkVisualizer(config.results_path)

# Carrega benchmark
data = analyzer.load_benchmark('meu_primeiro_benchmark')

# Cria visualização interativa
visualizer.visualize_benchmark(data, show_plot=True)

# Gera relatório
report = analyzer.generate_report(data)
print(report)
```

### Análise de Impacto

```python
# Qual parâmetro tem mais impacto?
rankings = analyzer.rank_parameters(data)
for i, rank in enumerate(rankings, 1):
    print(f"{i}. {rank['parameter']} - Impacto: {rank['impact_score']:.3f}")

# Configuração ótima
optimal = analyzer.find_optimal_configuration(data)
print(f"Melhor: {optimal['parameter']}={optimal['value']}")
print(f"Score: {optimal['metric_score']:.2f}%")
```

## Dicas Importantes

### 🚀 Performance

- **GPU**: Configure `device='cuda'` para treinar na GPU
- **Paralelo**: Use `parallel=True` com cuidado (requer muita memória)
- **Workers**: Ajuste `workers` conforme sua CPU

### 💾 Gestão de Espaço

- Cada modelo treinado ocupa ~10-50MB
- Um benchmark com 20 testes = ~200MB-1GB
- Limpe modelos antigos periodicamente

### ⏱️ Tempo Estimado

Para um dataset pequeno (~1000 imagens):
- 10 epochs: ~2-5 minutos
- 50 epochs: ~10-25 minutos
- Benchmark completo (5 parâmetros × 5 divisões): ~2-5 horas

### 🐛 Troubleshooting

**Erro: "CUDA out of memory"**
- Reduza `batch_size`
- Use modelo menor (yolo11n.pt)

**Erro: "Dataset not found"**
- Verifique caminho do `data.yaml`
- Use caminhos absolutos

**Treinamento muito lento**
- Verifique se está usando GPU (`device='cuda'`)
- Reduza `imgsz` (640 → 416)

## Suporte

- 📖 Documentação completa: `README.md`
- 💡 Exemplos: `example_usage.py`
- 🔍 Verificação: `python3 check_system.py`
- 📓 Notebooks: pasta `notebooks/`

## Próximos Passos

1. ✅ Instale e verifique o sistema
2. ✅ Prepare seu dataset
3. ✅ Execute um treinamento teste
4. ✅ Execute um benchmark pequeno
5. ✅ Analise os resultados
6. ✅ Ajuste parâmetros e repita!

---

**Dica Final**: Comece sempre com epochs baixos (10-20) para testar rapidamente!
