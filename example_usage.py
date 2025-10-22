#!/usr/bin/env python3
"""
Exemplo de uso do YOLO Benchmark System
========================================

Este script demonstra os principais recursos do sistema.
"""

from pathlib import Path
from src.core import Config, BenchmarkConfig, BenchmarkRunner, YOLOTrainer
from src.results import BenchmarkVisualizer, ResultsAnalyzer


def example_training():
    """Exemplo de treinamento individual."""
    print("\n" + "="*80)
    print("EXEMPLO 1: TREINAMENTO INDIVIDUAL")
    print("="*80 + "\n")
    
    # Configuração
    config = Config()
    config.model_name = 'yolo11n.pt'
    config.epochs = 10  # Poucos epochs para teste rápido
    config.batch_size = 16
    config.imgsz = 640
    
    # Trainer
    trainer = YOLOTrainer(config)
    trainer.load_model()
    
    # IMPORTANTE: Atualize com seu dataset
    dataset_path = '/path/to/your/data.yaml'
    
    print(f"Dataset: {dataset_path}")
    print("⚠️  Certifique-se de atualizar o caminho do dataset!\n")
    
    # Descomente para executar:
    # metrics = trainer.train(data=dataset_path, name='example_training')
    # print(f"\nResultados: {metrics}")


def example_benchmark():
    """Exemplo de benchmark de parâmetros."""
    print("\n" + "="*80)
    print("EXEMPLO 2: BENCHMARK DE PARÂMETROS")
    print("="*80 + "\n")
    
    # Configuração base
    config = Config()
    config.model_name = 'yolo11n.pt'
    
    # Configuração do benchmark
    benchmark_config = BenchmarkConfig()
    benchmark_config.benchmark_name = 'example_benchmark'
    benchmark_config.num_divisions = 3  # Menos divisões para teste
    
    # Parâmetros para testar
    benchmark_config.benchmark_params = {
        'epochs': {'min': 5, 'max': 15, 'type': 'int'},
        'batch_size': {'min': 8, 'max': 16, 'type': 'int'},
    }
    
    # Mostra o que será testado
    print("Parâmetros a testar:")
    for param in benchmark_config.benchmark_params:
        values = benchmark_config.get_benchmark_values(param)
        print(f"  {param}: {values}")
    
    total = sum(len(benchmark_config.get_benchmark_values(p)) 
                for p in benchmark_config.benchmark_params)
    print(f"\nTotal de testes: {total}")
    
    # IMPORTANTE: Atualize com seu dataset
    dataset_path = '/path/to/your/data.yaml'
    
    # Descomente para executar:
    # runner = BenchmarkRunner(config, benchmark_config)
    # results = runner.run_benchmark(dataset_path=dataset_path, parallel=False)
    # print(f"\nBenchmark concluído: {results['successful_tests']}/{results['total_tests']} testes")


def example_analysis():
    """Exemplo de análise de resultados."""
    print("\n" + "="*80)
    print("EXEMPLO 3: ANÁLISE DE RESULTADOS")
    print("="*80 + "\n")
    
    config = Config()
    analyzer = ResultsAnalyzer(config.results_path)
    visualizer = BenchmarkVisualizer(config.results_path)
    
    # Lista benchmarks disponíveis
    benchmarks_dir = config.results_path / 'benchmarks'
    if benchmarks_dir.exists():
        benchmarks = [d.name for d in benchmarks_dir.iterdir() if d.is_dir()]
        print(f"Benchmarks disponíveis: {benchmarks}")
        
        if benchmarks:
            # Analisa primeiro benchmark
            benchmark_name = benchmarks[0]
            print(f"\nAnalisando: {benchmark_name}")
            
            # Carrega dados
            benchmark_data = analyzer.load_benchmark(benchmark_name)
            
            # Gera relatório
            print("\nGerando relatório...")
            report = analyzer.generate_report(benchmark_data)
            print(report[:500] + "...")  # Mostra início do relatório
            
            # Cria visualização
            print("\nCriando visualização...")
            output_html = visualizer.visualize_benchmark(
                benchmark_data,
                show_plot=False
            )
            print(f"Visualização salva em: {output_html}")
        else:
            print("⚠️  Nenhum benchmark encontrado. Execute um benchmark primeiro!")
    else:
        print("⚠️  Diretório de resultados não encontrado.")


def example_interface():
    """Exemplo de uso da interface Jupyter."""
    print("\n" + "="*80)
    print("EXEMPLO 4: INTERFACE JUPYTER")
    print("="*80 + "\n")
    
    print("Para usar a interface interativa, execute em um notebook Jupyter:")
    print("\n```python")
    print("from src.interface import YOLOBenchmarkInterface")
    print("")
    print("interface = YOLOBenchmarkInterface()")
    print("interface.show()")
    print("```\n")
    print("Ou abra um dos notebooks de exemplo em notebooks/")


def main():
    """Função principal."""
    print("\n" + "="*80)
    print("YOLO BENCHMARK SYSTEM - EXEMPLOS DE USO")
    print("="*80)
    
    # Executa exemplos
    example_training()
    example_benchmark()
    example_analysis()
    example_interface()
    
    print("\n" + "="*80)
    print("PRÓXIMOS PASSOS")
    print("="*80)
    print("\n1. Prepare seu dataset no formato YOLO com arquivo data.yaml")
    print("2. Atualize os caminhos nos exemplos acima")
    print("3. Execute os notebooks em notebooks/ para explorar o sistema")
    print("4. Use a interface interativa para facilitar o uso")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
