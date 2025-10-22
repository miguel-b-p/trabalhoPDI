"""
Sistema de Benchmark
====================
Executa benchmarks completos de parâmetros YOLO.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

from .config import Config, BenchmarkConfig
from .trainer import YOLOTrainer
from .data_manager import DataManager


class BenchmarkRunner:
    """Executa e gerencia benchmarks de parâmetros YOLO."""
    
    def __init__(self, config: Config, benchmark_config: BenchmarkConfig):
        """
        Inicializa o runner de benchmark.
        
        Args:
            config: Configuração principal
            benchmark_config: Configuração de benchmark
        """
        self.config = config
        self.benchmark_config = benchmark_config
        self.trainer = YOLOTrainer(config)
        self.data_manager = DataManager(config.data_path)
        
        # Resultados
        self.benchmark_results = []
        
        # Cria diretório de resultados
        self.benchmark_dir = config.results_path / 'benchmarks' / benchmark_config.benchmark_name
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    def run_benchmark(
        self,
        dataset_path: str,
        parallel: bool = False,
        max_workers: int = 2
    ) -> Dict[str, Any]:
        """
        Executa benchmark completo.
        
        Args:
            dataset_path: Caminho para o dataset
            parallel: Se deve executar testes em paralelo
            max_workers: Número máximo de workers paralelos
            
        Returns:
            Dicionário com resultados consolidados
        """
        print(f"\n{'='*80}")
        print(f"INICIANDO BENCHMARK: {self.benchmark_config.benchmark_name}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # Gera todas as combinações de parâmetros
        combinations = self.benchmark_config.get_all_benchmark_combinations()
        
        print(f"Total de testes a executar: {len(combinations)}")
        print(f"Modo paralelo: {'Sim' if parallel else 'Não'}")
        if parallel:
            print(f"Workers: {max_workers}")
        print(f"\n{'-'*80}\n")
        
        # Executa testes
        if parallel:
            results = self._run_parallel(dataset_path, combinations, max_workers)
        else:
            results = self._run_sequential(dataset_path, combinations)
        
        total_time = time.time() - start_time
        
        # Consolida resultados
        consolidated = self._consolidate_results(results, total_time)
        
        # Salva resultados
        self._save_benchmark_results(consolidated)
        
        print(f"\n{'='*80}")
        print(f"BENCHMARK CONCLUÍDO!")
        print(f"Tempo total: {total_time:.2f}s ({total_time/60:.2f}min)")
        print(f"Resultados salvos em: {self.benchmark_dir}")
        print(f"{'='*80}\n")
        
        return consolidated
    
    def _run_sequential(
        self,
        dataset_path: str,
        combinations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Executa testes sequencialmente."""
        results = []
        
        for idx, params in enumerate(combinations, 1):
            print(f"\n{'='*80}")
            print(f"Teste {idx}/{len(combinations)}")
            print(f"{'='*80}")
            
            result = self._run_single_test(dataset_path, params, idx)
            results.append(result)
            
            # Salva resultado intermediário
            self._save_intermediate_result(result, idx)
        
        return results
    
    def _run_parallel(
        self,
        dataset_path: str,
        combinations: List[Dict[str, Any]],
        max_workers: int
    ) -> List[Dict[str, Any]]:
        """Executa testes em paralelo."""
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submete todos os jobs
            future_to_params = {
                executor.submit(
                    self._run_single_test_static,
                    dataset_path,
                    params,
                    idx,
                    self.config,
                    self.benchmark_config
                ): (params, idx)
                for idx, params in enumerate(combinations, 1)
            }
            
            # Coleta resultados conforme completam
            for future in as_completed(future_to_params):
                params, idx = future_to_params[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    print(f"\nTeste {idx}/{len(combinations)} concluído")
                    
                    # Salva resultado intermediário
                    self._save_intermediate_result(result, idx)
                    
                except Exception as e:
                    print(f"Erro no teste {idx}: {str(e)}")
                    results.append({
                        'test_id': idx,
                        'error': str(e),
                        'params': params
                    })
        
        # Ordena por test_id
        results.sort(key=lambda x: x.get('test_id', 0))
        
        return results
    
    def _run_single_test(
        self,
        dataset_path: str,
        params: Dict[str, Any],
        test_id: int
    ) -> Dict[str, Any]:
        """Executa um único teste de benchmark."""
        benchmark_param = params.pop('_benchmark_param')
        benchmark_value = params.pop('_benchmark_value')
        
        print(f"\nParâmetro testado: {benchmark_param} = {benchmark_value}")
        print(f"Parâmetros fixos: {params}")
        
        # Atualiza configuração
        for key, value in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Define nome do experimento
        experiment_name = f"bench_{benchmark_param}_{benchmark_value}_{test_id}"
        
        try:
            # Treina modelo
            start_time = time.time()
            metrics = self.trainer.train(
                data=dataset_path,
                name=experiment_name,
                **{benchmark_param: benchmark_value}
            )
            training_time = time.time() - start_time
            
            # Validação (se configurado)
            val_metrics = {}
            if self.benchmark_config.run_validation:
                val_metrics = self.trainer.validate(data=dataset_path)
            
            # Monta resultado
            result = {
                'test_id': test_id,
                'timestamp': datetime.now().isoformat(),
                'benchmark_param': benchmark_param,
                'benchmark_value': benchmark_value,
                'fixed_params': params,
                'training_time': training_time,
                'train_metrics': metrics,
                'val_metrics': val_metrics,
                'experiment_name': experiment_name,
                'success': True
            }
            
            return result
            
        except Exception as e:
            print(f"Erro durante teste: {str(e)}")
            return {
                'test_id': test_id,
                'timestamp': datetime.now().isoformat(),
                'benchmark_param': benchmark_param,
                'benchmark_value': benchmark_value,
                'error': str(e),
                'success': False
            }
    
    @staticmethod
    def _run_single_test_static(
        dataset_path: str,
        params: Dict[str, Any],
        test_id: int,
        config: Config,
        benchmark_config: BenchmarkConfig
    ) -> Dict[str, Any]:
        """Versão estática para execução paralela."""
        # Recria objetos no processo filho
        runner = BenchmarkRunner(config, benchmark_config)
        return runner._run_single_test(dataset_path, params, test_id)
    
    def _consolidate_results(
        self,
        results: List[Dict[str, Any]],
        total_time: float
    ) -> Dict[str, Any]:
        """Consolida resultados do benchmark."""
        # Agrupa por parâmetro
        by_parameter = {}
        
        for result in results:
            if not result.get('success', False):
                continue
            
            param_name = result['benchmark_param']
            if param_name not in by_parameter:
                by_parameter[param_name] = []
            
            by_parameter[param_name].append(result)
        
        # Estatísticas
        consolidated = {
            'benchmark_name': self.benchmark_config.benchmark_name,
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'total_tests': len(results),
            'successful_tests': sum(1 for r in results if r.get('success', False)),
            'failed_tests': sum(1 for r in results if not r.get('success', False)),
            'results_by_parameter': by_parameter,
            'all_results': results,
            'config': self.config.to_dict(),
            'benchmark_config': self.benchmark_config.to_dict()
        }
        
        return consolidated
    
    def _save_benchmark_results(self, consolidated: Dict[str, Any]):
        """Salva resultados consolidados do benchmark."""
        # JSON completo
        json_path = self.benchmark_dir / 'benchmark_results.json'
        with open(json_path, 'w') as f:
            json.dump(consolidated, f, indent=4)
        
        print(f"\nResultados JSON salvos em: {json_path}")
        
        # CSV para fácil análise
        self._save_results_csv(consolidated)
    
    def _save_results_csv(self, consolidated: Dict[str, Any]):
        """Salva resultados em formato CSV."""
        rows = []
        
        for result in consolidated['all_results']:
            if not result.get('success', False):
                continue
            
            row = {
                'test_id': result['test_id'],
                'timestamp': result['timestamp'],
                'benchmark_param': result['benchmark_param'],
                'benchmark_value': result['benchmark_value'],
                'training_time': result['training_time'],
            }
            
            # Adiciona métricas de treino
            if 'train_metrics' in result:
                for key, value in result['train_metrics'].items():
                    row[f'train_{key}'] = value
            
            # Adiciona métricas de validação
            if 'val_metrics' in result:
                for key, value in result['val_metrics'].items():
                    row[f'val_{key}'] = value
            
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            csv_path = self.benchmark_dir / 'benchmark_results.csv'
            df.to_csv(csv_path, index=False)
            
            print(f"Resultados CSV salvos em: {csv_path}")
    
    def _save_intermediate_result(self, result: Dict[str, Any], test_id: int):
        """Salva resultado intermediário de um teste."""
        intermediate_dir = self.benchmark_dir / 'intermediate'
        intermediate_dir.mkdir(exist_ok=True)
        
        result_file = intermediate_dir / f'test_{test_id:03d}.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=4)
    
    def load_benchmark_results(self, benchmark_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Carrega resultados de um benchmark anterior.
        
        Args:
            benchmark_name: Nome do benchmark (usa atual se None)
            
        Returns:
            Resultados consolidados
        """
        if benchmark_name:
            results_dir = self.config.results_path / 'benchmarks' / benchmark_name
        else:
            results_dir = self.benchmark_dir
        
        results_file = results_dir / 'benchmark_results.json'
        
        if not results_file.exists():
            raise FileNotFoundError(f"Resultados não encontrados: {results_file}")
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def compare_benchmarks(self, benchmark_names: List[str]) -> pd.DataFrame:
        """
        Compara resultados de múltiplos benchmarks.
        
        Args:
            benchmark_names: Lista de nomes de benchmarks
            
        Returns:
            DataFrame com comparação
        """
        all_data = []
        
        for name in benchmark_names:
            results = self.load_benchmark_results(name)
            
            for result in results['all_results']:
                if result.get('success', False):
                    row = {
                        'benchmark': name,
                        'param': result['benchmark_param'],
                        'value': result['benchmark_value'],
                        'training_time': result['training_time'],
                    }
                    
                    # Adiciona métricas
                    if 'train_metrics' in result:
                        row.update({f'train_{k}': v for k, v in result['train_metrics'].items()})
                    if 'val_metrics' in result:
                        row.update({f'val_{k}': v for k, v in result['val_metrics'].items()})
                    
                    all_data.append(row)
        
        return pd.DataFrame(all_data)
