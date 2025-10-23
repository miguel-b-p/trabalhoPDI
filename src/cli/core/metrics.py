"""
Metrics collection and analysis for YOLO training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import json
from pathlib import Path
import logging


class MetricsCollector:
    """Collect and analyze training metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_data = []
    
    def extract_training_metrics(self, results) -> Dict[str, List[float]]:
        """Extract training metrics from YOLO results."""
        if not hasattr(results, 'csv'):
            return {}
        
        try:
            # Read CSV file with training metrics
            csv_path = Path(results.csv)
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                metrics = {}
                for col in df.columns:
                    if col.startswith(('train/', 'val/', 'metrics/')):
                        metrics[col] = df[col].dropna().tolist()
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error extracting metrics: {e}")
        
        return {}
    
    def calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Primary detection metrics
        if 'metrics/mAP50(B)' in results:
            metrics['mAP50'] = float(results['metrics/mAP50(B)'])
        
        if 'metrics/mAP50-95(B)' in results:
            metrics['mAP50_95'] = float(results['metrics/mAP50-95(B)'])
        
        if 'metrics/precision(B)' in results:
            metrics['precision'] = float(results['metrics/precision(B)'])
        
        if 'metrics/recall(B)' in results:
            metrics['recall'] = float(results['metrics/recall(B)'])
        
        if 'metrics/f1_score' in results:
            metrics['f1_score'] = float(results['metrics/f1_score'])
        
        # Training efficiency metrics
        if 'train/box_loss' in results:
            box_losses = results['train/box_loss']
            if box_losses:
                metrics['final_box_loss'] = float(box_losses[-1])
                metrics['min_box_loss'] = float(np.min(box_losses))
        
        if 'train/cls_loss' in results:
            cls_losses = results['train/cls_loss']
            if cls_losses:
                metrics['final_cls_loss'] = float(cls_losses[-1])
                metrics['min_cls_loss'] = float(np.min(cls_losses))
        
        if 'train/dfl_loss' in results:
            dfl_losses = results['train/dfl_loss']
            if dfl_losses:
                metrics['final_dfl_loss'] = float(dfl_losses[-1])
                metrics['min_dfl_loss'] = float(np.min(dfl_losses))
        
        # Validation metrics
        if 'val/box_loss' in results:
            val_box_losses = results['val/box_loss']
            if val_box_losses:
                metrics['final_val_box_loss'] = float(val_box_losses[-1])
                metrics['min_val_box_loss'] = float(np.min(val_box_losses))
        
        if 'val/cls_loss' in results:
            val_cls_losses = results['val/cls_loss']
            if val_cls_losses:
                metrics['final_val_cls_loss'] = float(val_cls_losses[-1])
                metrics['min_val_cls_loss'] = float(np.min(val_cls_losses))
        
        # Convergence metrics
        metrics['convergence_rate'] = self.calculate_convergence_rate(results)
        metrics['overfitting_score'] = self.calculate_overfitting_score(results)
        
        return metrics
    
    def calculate_convergence_rate(self, results: Dict[str, Any]) -> float:
        """Calculate how quickly the model converges."""
        if 'val/box_loss' not in results:
            return 0.0
        
        losses = results['val/box_loss']
        if len(losses) < 10:
            return 0.0
        
        # Calculate slope of loss curve
        x = np.arange(len(losses))
        slope, _ = np.polyfit(x, losses, 1)
        
        # Normalize by initial loss
        if losses[0] > 0:
            return abs(slope) / losses[0]
        
        return 0.0
    
    def calculate_overfitting_score(self, results: Dict[str, Any]) -> float:
        """Calculate overfitting score based on train/val loss divergence."""
        if 'train/box_loss' not in results or 'val/box_loss' not in results:
            return 0.0
        
        train_losses = results['train/box_loss']
        val_losses = results['val/box_loss']
        
        if len(train_losses) != len(val_losses) or len(train_losses) < 10:
            return 0.0
        
        # Calculate divergence
        train_final = np.mean(train_losses[-5:])
        val_final = np.mean(val_losses[-5:])
        
        if train_final > 0:
            return max(0, (val_final - train_final) / train_final)
        
        return 0.0
    
    def analyze_parameter_impact(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how parameter changes affect performance."""
        if not results_list:
            return {}
        
        analysis = {
            "parameter_effects": {},
            "correlations": {},
            "optimal_values": {},
            "sensitivity_analysis": {}
        }
        
        # Extract parameter values and metrics
        df_data = []
        for result in results_list:
            if result.get('status') == 'completed':
                row = {
                    **result.get('param_changes', {}),
                    **result.get('metrics', {})
                }
                df_data.append(row)
        
        if not df_data:
            return analysis
        
        df = pd.DataFrame(df_data)
        
        # Calculate correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Find correlations with performance metrics
            performance_metrics = ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1_score']
            for metric in performance_metrics:
                if metric in corr_matrix.columns:
                    correlations = corr_matrix[metric].drop(metric)
                    analysis["correlations"][metric] = correlations.to_dict()
        
        # Parameter effects analysis
        for param in df.columns:
            if param in ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1_score']:
                continue
                
            if df[param].dtype in ['float64', 'int64']:
                # Group by parameter ranges
                param_values = df[param]
                for metric in ['mAP50', 'mAP50_95']:
                    if metric in df.columns:
                        # Calculate trend
                        correlation = param_values.corr(df[metric])
                        analysis["parameter_effects"][f"{param}_{metric}"] = float(correlation)
        
        return analysis
    
    def generate_summary_report(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        if not results_list:
            return {}
        
        # Filter successful runs
        successful_runs = [r for r in results_list if r.get('status') == 'completed']
        
        if not successful_runs:
            return {"error": "No successful runs found"}
        
        # Extract key metrics
        mAP50_values = [r.get('metrics', {}).get('mAP50', 0) for r in successful_runs]
        mAP50_95_values = [r.get('metrics', {}).get('mAP50_95', 0) for r in successful_runs]
        training_times = [r.get('training_time', 0) for r in successful_runs]
        
        report = {
            "summary": {
                "total_runs": len(results_list),
                "successful_runs": len(successful_runs),
                "failed_runs": len(results_list) - len(successful_runs),
                "success_rate": len(successful_runs) / len(results_list) if results_list else 0
            },
            "performance": {
                "best_mAP50": max(mAP50_values) if mAP50_values else 0,
                "mean_mAP50": float(np.mean(mAP50_values)) if mAP50_values else 0,
                "std_mAP50": float(np.std(mAP50_values)) if mAP50_values else 0,
                "best_mAP50_95": max(mAP50_95_values) if mAP50_95_values else 0,
                "mean_mAP50_95": float(np.mean(mAP50_95_values)) if mAP50_95_values else 0,
                "std_mAP50_95": float(np.std(mAP50_95_values)) if mAP50_95_values else 0
            },
            "efficiency": {
                "mean_training_time": float(np.mean(training_times)) if training_times else 0,
                "std_training_time": float(np.std(training_times)) if training_times else 0,
                "min_training_time": min(training_times) if training_times else 0,
                "max_training_time": max(training_times) if training_times else 0
            },
            "parameter_analysis": self.analyze_parameter_impact(successful_runs)
        }
        
        return report
