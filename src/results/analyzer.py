"""
Analisador de Resultados
========================
Análise estatística e insights dos resultados de benchmark.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import json


class ResultsAnalyzer:
    """Analisa resultados de benchmarks e gera insights."""
    
    def __init__(self, results_path: Path):
        """
        Inicializa o analisador.
        
        Args:
            results_path: Caminho para a pasta de resultados
        """
        self.results_path = Path(results_path)
    
    def load_benchmark(self, benchmark_name: str) -> Dict[str, Any]:
        """
        Carrega dados de um benchmark.
        
        Args:
            benchmark_name: Nome do benchmark
            
        Returns:
            Dados do benchmark
        """
        benchmark_file = self.results_path / 'benchmarks' / benchmark_name / 'benchmark_results.json'
        
        if not benchmark_file.exists():
            raise FileNotFoundError(f"Benchmark não encontrado: {benchmark_file}")
        
        with open(benchmark_file, 'r') as f:
            return json.load(f)
    
    def analyze_parameter_impact(
        self,
        benchmark_data: Dict[str, Any],
        param_name: str,
        metric: str = 'mAP50-95'
    ) -> Dict[str, Any]:
        """
        Analisa o impacto de um parâmetro específico em uma métrica.
        
        Args:
            benchmark_data: Dados do benchmark
            param_name: Nome do parâmetro
            metric: Métrica a analisar
            
        Returns:
            Análise do impacto
        """
        results_by_param = benchmark_data.get('results_by_parameter', {})
        
        if param_name not in results_by_param:
            raise ValueError(f"Parâmetro '{param_name}' não encontrado")
        
        param_results = results_by_param[param_name]
        
        # Extrai dados
        values = []
        metrics = []
        
        for result in param_results:
            values.append(result['benchmark_value'])
            
            val_metrics = result.get('val_metrics', {})
            metric_value = val_metrics.get(metric, 0)
            metrics.append(metric_value * 100 if metric_value else 0)
        
        # Ordena
        sorted_pairs = sorted(zip(values, metrics), key=lambda x: x[0])
        values, metrics = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        # Análise estatística
        analysis = {
            'parameter': param_name,
            'metric': metric,
            'n_samples': len(values),
            'values': list(values),
            'metric_scores': list(metrics),
            'statistics': {}
        }
        
        if metrics:
            analysis['statistics'] = {
                'mean': float(np.mean(metrics)),
                'std': float(np.std(metrics)),
                'min': float(np.min(metrics)),
                'max': float(np.max(metrics)),
                'median': float(np.median(metrics)),
                'range': float(np.max(metrics) - np.min(metrics))
            }
            
            # Correlação (se valores numéricos)
            try:
                numeric_values = [float(v) for v in values]
                correlation, p_value = stats.pearsonr(numeric_values, metrics)
                
                analysis['correlation'] = {
                    'coefficient': float(correlation),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'interpretation': self._interpret_correlation(correlation)
                }
            except (ValueError, TypeError):
                analysis['correlation'] = None
            
            # Melhor valor
            best_idx = np.argmax(metrics)
            analysis['best'] = {
                'value': values[best_idx],
                'score': metrics[best_idx]
            }
            
            # Pior valor
            worst_idx = np.argmin(metrics)
            analysis['worst'] = {
                'value': values[worst_idx],
                'score': metrics[worst_idx]
            }
        
        return analysis
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpreta coeficiente de correlação."""
        abs_corr = abs(correlation)
        
        if abs_corr < 0.3:
            strength = "fraca"
        elif abs_corr < 0.7:
            strength = "moderada"
        else:
            strength = "forte"
        
        direction = "positiva" if correlation > 0 else "negativa"
        
        return f"Correlação {strength} {direction}"
    
    def rank_parameters(
        self,
        benchmark_data: Dict[str, Any],
        metric: str = 'mAP50-95'
    ) -> List[Dict[str, Any]]:
        """
        Ranqueia parâmetros por impacto na métrica.
        
        Args:
            benchmark_data: Dados do benchmark
            metric: Métrica para ranquear
            
        Returns:
            Lista de parâmetros ranqueados
        """
        rankings = []
        
        results_by_param = benchmark_data.get('results_by_parameter', {})
        
        for param_name in results_by_param:
            analysis = self.analyze_parameter_impact(benchmark_data, param_name, metric)
            
            # Score de impacto baseado em range e correlação
            impact_score = 0
            
            if analysis['statistics']:
                # Range normalizado (0-1)
                range_score = analysis['statistics']['range'] / 100
                impact_score += range_score * 0.5
            
            if analysis.get('correlation'):
                # Correlação absoluta (0-1)
                corr_score = abs(analysis['correlation']['coefficient'])
                impact_score += corr_score * 0.5
            
            rankings.append({
                'parameter': param_name,
                'impact_score': impact_score,
                'range': analysis['statistics'].get('range', 0),
                'correlation': analysis.get('correlation', {}).get('coefficient', 0),
                'best_score': analysis.get('best', {}).get('score', 0)
            })
        
        # Ordena por impacto
        rankings.sort(key=lambda x: x['impact_score'], reverse=True)
        
        return rankings
    
    def find_optimal_configuration(
        self,
        benchmark_data: Dict[str, Any],
        metric: str = 'mAP50-95',
        time_weight: float = 0.0
    ) -> Dict[str, Any]:
        """
        Encontra configuração ótima balanceando métrica e tempo.
        
        Args:
            benchmark_data: Dados do benchmark
            metric: Métrica principal
            time_weight: Peso do tempo (0-1, 0=ignora tempo)
            
        Returns:
            Configuração ótima
        """
        all_results = benchmark_data.get('all_results', [])
        
        if not all_results:
            return {}
        
        # Calcula score combinado
        best_score = -float('inf')
        best_config = None
        
        for result in all_results:
            if not result.get('success', False):
                continue
            
            # Métrica principal
            val_metrics = result.get('val_metrics', {})
            metric_value = val_metrics.get(metric, 0) * 100 if val_metrics.get(metric) else 0
            
            # Tempo (normalizado e invertido)
            time_minutes = result['training_time'] / 60
            max_time = max(r['training_time'] / 60 for r in all_results if r.get('success'))
            time_score = (1 - time_minutes / max_time) * 100  # Menor tempo = maior score
            
            # Score combinado
            combined_score = (1 - time_weight) * metric_value + time_weight * time_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_config = {
                    'test_id': result['test_id'],
                    'parameter': result['benchmark_param'],
                    'value': result['benchmark_value'],
                    'metric_score': metric_value,
                    'training_time_min': time_minutes,
                    'combined_score': combined_score,
                    'fixed_params': result.get('fixed_params', {})
                }
        
        return best_config
    
    def generate_report(
        self,
        benchmark_data: Dict[str, Any],
        output_file: Optional[Path] = None
    ) -> str:
        """
        Gera relatório textual completo da análise.
        
        Args:
            benchmark_data: Dados do benchmark
            output_file: Arquivo para salvar (None = retorna string)
            
        Returns:
            Relatório em texto
        """
        report_lines = []
        
        # Cabeçalho
        report_lines.append("=" * 80)
        report_lines.append(f"RELATÓRIO DE ANÁLISE DE BENCHMARK")
        report_lines.append(f"Benchmark: {benchmark_data['benchmark_name']}")
        report_lines.append(f"Data: {benchmark_data['timestamp']}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Resumo
        report_lines.append("RESUMO EXECUTIVO")
        report_lines.append("-" * 80)
        report_lines.append(f"Total de testes: {benchmark_data['total_tests']}")
        report_lines.append(f"Testes bem-sucedidos: {benchmark_data['successful_tests']}")
        report_lines.append(f"Testes falhados: {benchmark_data['failed_tests']}")
        report_lines.append(f"Tempo total: {benchmark_data['total_time']/60:.2f} minutos")
        report_lines.append("")
        
        # Ranking de parâmetros
        report_lines.append("RANKING DE IMPACTO DOS PARÂMETROS")
        report_lines.append("-" * 80)
        rankings = self.rank_parameters(benchmark_data)
        
        for i, rank in enumerate(rankings, 1):
            report_lines.append(f"{i}. {rank['parameter']}")
            report_lines.append(f"   Score de Impacto: {rank['impact_score']:.3f}")
            report_lines.append(f"   Range: {rank['range']:.2f}%")
            report_lines.append(f"   Correlação: {rank['correlation']:.3f}")
            report_lines.append(f"   Melhor Score: {rank['best_score']:.2f}%")
            report_lines.append("")
        
        # Configuração ótima
        report_lines.append("CONFIGURAÇÃO ÓTIMA")
        report_lines.append("-" * 80)
        optimal = self.find_optimal_configuration(benchmark_data)
        
        if optimal:
            report_lines.append(f"Parâmetro: {optimal['parameter']} = {optimal['value']}")
            report_lines.append(f"Score: {optimal['metric_score']:.2f}%")
            report_lines.append(f"Tempo de treinamento: {optimal['training_time_min']:.2f} min")
            report_lines.append("")
        
        # Análises detalhadas por parâmetro
        report_lines.append("ANÁLISES DETALHADAS POR PARÂMETRO")
        report_lines.append("=" * 80)
        
        for param_name in benchmark_data.get('results_by_parameter', {}).keys():
            report_lines.append("")
            report_lines.append(f"Parâmetro: {param_name}")
            report_lines.append("-" * 80)
            
            analysis = self.analyze_parameter_impact(benchmark_data, param_name)
            
            report_lines.append(f"Amostras: {analysis['n_samples']}")
            
            if analysis['statistics']:
                stats = analysis['statistics']
                report_lines.append(f"Média: {stats['mean']:.2f}%")
                report_lines.append(f"Desvio padrão: {stats['std']:.2f}%")
                report_lines.append(f"Mínimo: {stats['min']:.2f}%")
                report_lines.append(f"Máximo: {stats['max']:.2f}%")
                report_lines.append(f"Mediana: {stats['median']:.2f}%")
                report_lines.append(f"Range: {stats['range']:.2f}%")
            
            if analysis.get('correlation'):
                corr = analysis['correlation']
                report_lines.append(f"Correlação: {corr['coefficient']:.3f} ({corr['interpretation']})")
                report_lines.append(f"P-value: {corr['p_value']:.4f}")
                report_lines.append(f"Significante: {'Sim' if corr['significant'] else 'Não'}")
            
            if analysis.get('best'):
                best = analysis['best']
                report_lines.append(f"Melhor valor: {best['value']} (Score: {best['score']:.2f}%)")
            
            if analysis.get('worst'):
                worst = analysis['worst']
                report_lines.append(f"Pior valor: {worst['value']} (Score: {worst['score']:.2f}%)")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("FIM DO RELATÓRIO")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Salva se solicitado
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Relatório salvo em: {output_file}")
        
        return report_text
    
    def compare_benchmarks(
        self,
        benchmark_names: List[str]
    ) -> pd.DataFrame:
        """
        Compara múltiplos benchmarks.
        
        Args:
            benchmark_names: Lista de nomes de benchmarks
            
        Returns:
            DataFrame com comparação
        """
        comparison_data = []
        
        for name in benchmark_names:
            try:
                data = self.load_benchmark(name)
                
                comparison_data.append({
                    'benchmark': name,
                    'total_tests': data['total_tests'],
                    'successful_tests': data['successful_tests'],
                    'total_time_min': data['total_time'] / 60,
                    'avg_time_per_test': data['total_time'] / data['total_tests'] / 60
                })
                
            except Exception as e:
                print(f"Erro ao carregar benchmark {name}: {e}")
        
        return pd.DataFrame(comparison_data)
