"""
YOLO Trainer
============
Responsável pelo treinamento de modelos YOLO.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import time
from datetime import datetime
import json

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Aviso: ultralytics não instalado. Instale com: pip install ultralytics")

from .config import Config


class YOLOTrainer:
    """Gerencia treinamento de modelos YOLO."""
    
    def __init__(self, config: Config):
        """
        Inicializa o trainer.
        
        Args:
            config: Configuração do sistema
        """
        self.config = config
        self.model = None
        self.results = None
        self.training_history = []
        
        if YOLO is None:
            raise ImportError("ultralytics não está instalado. Instale com: pip install ultralytics")
    
    def load_model(self, model_path: Optional[str] = None) -> YOLO:
        """
        Carrega modelo YOLO.
        
        Args:
            model_path: Caminho do modelo (usa config se None)
            
        Returns:
            Modelo YOLO carregado
        """
        if model_path is None:
            model_path = self.config.model_name
        
        print(f"Carregando modelo: {model_path}")
        self.model = YOLO(model_path)
        return self.model
    
    def train(
        self,
        data: str,
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Treina o modelo YOLO.
        
        Args:
            data: Caminho para data.yaml do dataset
            name: Nome do experimento
            **kwargs: Parâmetros adicionais de treinamento
            
        Returns:
            Dicionário com resultados do treinamento
        """
        if self.model is None:
            self.load_model()
        
        # Prepara parâmetros
        train_params = self.config.get_yolo_params()
        train_params.update(kwargs)
        train_params['data'] = data
        
        if name:
            train_params['name'] = name
        else:
            train_params['name'] = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Define projeto
        train_params['project'] = str(self.config.results_path / 'training')
        
        print(f"\n{'='*60}")
        print(f"Iniciando treinamento: {train_params['name']}")
        print(f"{'='*60}")
        print(f"Dataset: {data}")
        print(f"Epochs: {train_params.get('epochs', 'N/A')}")
        print(f"Batch size: {train_params.get('batch', 'N/A')}")
        print(f"Image size: {train_params.get('imgsz', 'N/A')}")
        print(f"Device: {train_params.get('device', 'N/A')}")
        print(f"{'='*60}\n")
        
        # Treina
        start_time = time.time()
        try:
            self.results = self.model.train(**train_params)
            training_time = time.time() - start_time
            
            # Extrai métricas
            metrics = self._extract_metrics(self.results, training_time)
            
            # Salva histórico
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'name': train_params['name'],
                'params': train_params,
                'metrics': metrics,
                'training_time': training_time
            }
            self.training_history.append(history_entry)
            
            # Salva resultados
            self._save_training_results(history_entry)
            
            print(f"\n{'='*60}")
            print(f"Treinamento concluído!")
            print(f"Tempo: {training_time:.2f}s ({training_time/60:.2f}min)")
            print(f"{'='*60}\n")
            
            return metrics
            
        except Exception as e:
            print(f"Erro durante treinamento: {str(e)}")
            raise
    
    def validate(
        self,
        data: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Valida o modelo.
        
        Args:
            data: Caminho para data.yaml (usa último se None)
            **kwargs: Parâmetros adicionais
            
        Returns:
            Métricas de validação
        """
        if self.model is None:
            raise ValueError("Modelo não carregado. Use load_model() primeiro.")
        
        val_params = {
            'data': data,
            'device': self.config.device,
            'imgsz': self.config.imgsz,
            'batch': self.config.batch_size,
            'verbose': self.config.verbose,
        }
        val_params.update(kwargs)
        
        print(f"\n{'='*60}")
        print(f"Executando validação...")
        print(f"{'='*60}\n")
        
        results = self.model.val(**val_params)
        metrics = self._extract_val_metrics(results)
        
        print(f"\n{'='*60}")
        print(f"Validação concluída!")
        print(f"{'='*60}\n")
        
        return metrics
    
    def predict(
        self,
        source: str,
        save: bool = True,
        **kwargs
    ):
        """
        Realiza predições com o modelo.
        
        Args:
            source: Fonte das imagens
            save: Se deve salvar resultados
            **kwargs: Parâmetros adicionais
            
        Returns:
            Resultados das predições
        """
        if self.model is None:
            raise ValueError("Modelo não carregado. Use load_model() primeiro.")
        
        pred_params = {
            'source': source,
            'save': save,
            'device': self.config.device,
            'imgsz': self.config.imgsz,
        }
        pred_params.update(kwargs)
        
        return self.model.predict(**pred_params)
    
    def export_model(
        self,
        format: str = 'onnx',
        **kwargs
    ) -> Path:
        """
        Exporta modelo para formato específico.
        
        Args:
            format: Formato de exportação (onnx, torchscript, etc.)
            **kwargs: Parâmetros adicionais
            
        Returns:
            Caminho do modelo exportado
        """
        if self.model is None:
            raise ValueError("Modelo não carregado. Use load_model() primeiro.")
        
        export_path = self.model.export(format=format, **kwargs)
        print(f"Modelo exportado para: {export_path}")
        return Path(export_path)
    
    def _extract_metrics(self, results, training_time: float) -> Dict[str, Any]:
        """Extrai métricas dos resultados de treinamento."""
        try:
            # Métricas principais
            metrics = {
                'training_time': training_time,
                'box_loss': None,
                'cls_loss': None,
                'dfl_loss': None,
                'precision': None,
                'recall': None,
                'mAP50': None,
                'mAP50-95': None,
            }
            
            # Tenta extrair métricas do objeto results
            if hasattr(results, 'results_dict'):
                metrics.update(results.results_dict)
            elif hasattr(results, 'box'):
                if hasattr(results.box, 'map'):
                    metrics['mAP50-95'] = float(results.box.map)
                if hasattr(results.box, 'map50'):
                    metrics['mAP50'] = float(results.box.map50)
                if hasattr(results.box, 'p'):
                    metrics['precision'] = float(results.box.p)
                if hasattr(results.box, 'r'):
                    metrics['recall'] = float(results.box.r)
            
            return metrics
            
        except Exception as e:
            print(f"Aviso: Não foi possível extrair todas as métricas: {e}")
            return {'training_time': training_time}
    
    def _extract_val_metrics(self, results) -> Dict[str, Any]:
        """Extrai métricas de validação."""
        try:
            metrics = {}
            
            if hasattr(results, 'box'):
                if hasattr(results.box, 'map'):
                    metrics['mAP50-95'] = float(results.box.map)
                if hasattr(results.box, 'map50'):
                    metrics['mAP50'] = float(results.box.map50)
                if hasattr(results.box, 'p'):
                    metrics['precision'] = float(results.box.p)
                if hasattr(results.box, 'r'):
                    metrics['recall'] = float(results.box.r)
            
            return metrics
            
        except Exception as e:
            print(f"Aviso: Não foi possível extrair métricas de validação: {e}")
            return {}
    
    def _save_training_results(self, history_entry: Dict[str, Any]):
        """Salva resultados do treinamento em arquivo JSON."""
        results_dir = self.config.results_path / 'training_history'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Salva entrada individual
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = results_dir / f"training_{timestamp}.json"
        
        with open(result_file, 'w') as f:
            json.dump(history_entry, f, indent=4)
        
        # Atualiza histórico completo
        history_file = results_dir / 'full_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                full_history = json.load(f)
        else:
            full_history = []
        
        full_history.append(history_entry)
        
        with open(history_file, 'w') as f:
            json.dump(full_history, f, indent=4)
        
        print(f"Resultados salvos em: {result_file}")
    
    def get_training_history(self) -> list:
        """Retorna histórico de treinamentos."""
        return self.training_history
    
    def load_trained_model(self, model_path: Path) -> YOLO:
        """
        Carrega modelo já treinado.
        
        Args:
            model_path: Caminho para o modelo .pt
            
        Returns:
            Modelo carregado
        """
        print(f"Carregando modelo treinado: {model_path}")
        self.model = YOLO(str(model_path))
        return self.model
