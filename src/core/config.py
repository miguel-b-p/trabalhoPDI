"""
Configuração do Sistema
=======================
Gerencia todas as configurações do sistema de benchmark.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import yaml


@dataclass
class Config:
    """Configuração principal do sistema."""
    
    # Caminhos
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    models_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models")
    results_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results")
    
    # Configurações de treinamento
    model_name: str = "yolo11n.pt"
    device: str = "cuda"  # cuda, cpu, mps
    workers: int = 8
    batch_size: int = 16
    epochs: int = 100
    imgsz: int = 640
    
    # Dataset
    dataset_path: Optional[str] = None
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Otimização
    optimizer: str = "Adam"  # SGD, Adam, AdamW
    lr0: float = 0.01  # Learning rate inicial
    lrf: float = 0.01  # Learning rate final
    momentum: float = 0.937
    weight_decay: float = 0.0005
    
    # Augmentação
    augment: bool = True
    hsv_h: float = 0.015  # HSV-Hue augmentation
    hsv_s: float = 0.7    # HSV-Saturation augmentation
    hsv_v: float = 0.4    # HSV-Value augmentation
    degrees: float = 0.0  # Rotação
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    
    # Outros parâmetros
    patience: int = 50  # Early stopping
    save_period: int = -1
    cache: bool = False
    pretrained: bool = True
    verbose: bool = True
    seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        """Inicializa e valida os caminhos."""
        self.project_root = Path(self.project_root)
        self.data_path = Path(self.data_path)
        self.models_path = Path(self.models_path)
        self.results_path = Path(self.results_path)
        
        # Cria diretórios se não existirem
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte configuração para dicionário."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict
    
    def save(self, path: Path):
        """Salva configuração em arquivo JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Carrega configuração de arquivo JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def get_yolo_params(self) -> Dict[str, Any]:
        """Retorna apenas os parâmetros relevantes para o YOLO."""
        return {
            'epochs': self.epochs,
            'imgsz': self.imgsz,
            'batch': self.batch_size,
            'device': self.device,
            'workers': self.workers,
            'optimizer': self.optimizer,
            'lr0': self.lr0,
            'lrf': self.lrf,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'augment': self.augment,
            'hsv_h': self.hsv_h,
            'hsv_s': self.hsv_s,
            'hsv_v': self.hsv_v,
            'degrees': self.degrees,
            'translate': self.translate,
            'scale': self.scale,
            'shear': self.shear,
            'perspective': self.perspective,
            'flipud': self.flipud,
            'fliplr': self.fliplr,
            'mosaic': self.mosaic,
            'mixup': self.mixup,
            'copy_paste': self.copy_paste,
            'patience': self.patience,
            'save_period': self.save_period,
            'cache': self.cache,
            'pretrained': self.pretrained,
            'verbose': self.verbose,
            'seed': self.seed,
            'deterministic': self.deterministic,
        }


@dataclass
class BenchmarkConfig:
    """Configuração para benchmark de parâmetros."""
    
    # Parâmetros para benchmark
    benchmark_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Número de divisões para cada parâmetro (1/5, 2/5, 3/5, 4/5, 5/5)
    num_divisions: int = 5
    
    # Parâmetros fixos durante o benchmark
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    
    # Nome do benchmark
    benchmark_name: str = "yolo_benchmark"
    
    # Salvar checkpoints
    save_checkpoints: bool = True
    
    # Executar validação após cada treino
    run_validation: bool = True
    
    def __post_init__(self):
        """Inicializa configurações padrão se não fornecidas."""
        if not self.benchmark_params:
            self.benchmark_params = {
                'epochs': {'min': 10, 'max': 100, 'type': 'int'},
                'batch_size': {'min': 4, 'max': 32, 'type': 'int'},
                'imgsz': {'min': 320, 'max': 1280, 'type': 'int'},
                'lr0': {'min': 0.001, 'max': 0.1, 'type': 'float'},
                'augment': {'values': [True, False], 'type': 'bool'},
                'optimizer': {'values': ['SGD', 'Adam', 'AdamW'], 'type': 'categorical'},
            }
    
    def get_benchmark_values(self, param_name: str) -> List[Any]:
        """
        Gera valores de benchmark para um parâmetro específico.
        
        Args:
            param_name: Nome do parâmetro
            
        Returns:
            Lista de valores para testar
        """
        if param_name not in self.benchmark_params:
            raise ValueError(f"Parâmetro '{param_name}' não encontrado na configuração de benchmark")
        
        param_config = self.benchmark_params[param_name]
        param_type = param_config.get('type', 'float')
        
        if param_type == 'categorical' or param_type == 'bool':
            return param_config['values']
        
        # Para tipos numéricos, gera divisões
        min_val = param_config['min']
        max_val = param_config['max']
        
        values = []
        for i in range(1, self.num_divisions + 1):
            # Calcula valor proporcional (1/5, 2/5, 3/5, 4/5, 5/5)
            ratio = i / self.num_divisions
            value = min_val + (max_val - min_val) * ratio
            
            if param_type == 'int':
                value = int(value)
            
            values.append(value)
        
        return values
    
    def get_all_benchmark_combinations(self) -> List[Dict[str, Any]]:
        """
        Gera todas as combinações de parâmetros para benchmark.
        
        Returns:
            Lista de dicionários com combinações de parâmetros
        """
        combinations = []
        
        for param_name in self.benchmark_params:
            param_values = self.get_benchmark_values(param_name)
            
            for value in param_values:
                config = self.fixed_params.copy()
                config[param_name] = value
                config['_benchmark_param'] = param_name
                config['_benchmark_value'] = value
                combinations.append(config)
        
        return combinations
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'benchmark_params': self.benchmark_params,
            'num_divisions': self.num_divisions,
            'fixed_params': self.fixed_params,
            'benchmark_name': self.benchmark_name,
            'save_checkpoints': self.save_checkpoints,
            'run_validation': self.run_validation,
        }
    
    def save(self, path: Path):
        """Salva configuração em arquivo JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, path: Path) -> 'BenchmarkConfig':
        """Carrega configuração de arquivo JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
