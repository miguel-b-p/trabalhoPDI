"""
Gerenciador de Dados
====================
Gerencia datasets, preparação de dados e divisões.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
import yaml
from collections import defaultdict
import random


class DataManager:
    """Gerencia operações com datasets YOLO."""
    
    def __init__(self, data_path: Path):
        """
        Inicializa o gerenciador de dados.
        
        Args:
            data_path: Caminho para a pasta de dados
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def create_dataset_yaml(
        self,
        dataset_path: Path,
        train_path: str,
        val_path: str,
        test_path: Optional[str] = None,
        classes: List[str] = None,
        nc: Optional[int] = None
    ) -> Path:
        """
        Cria arquivo YAML de configuração do dataset YOLO.
        
        Args:
            dataset_path: Caminho base do dataset
            train_path: Caminho relativo para treino
            val_path: Caminho relativo para validação
            test_path: Caminho relativo para teste
            classes: Lista de nomes das classes
            nc: Número de classes
            
        Returns:
            Caminho do arquivo YAML criado
        """
        dataset_path = Path(dataset_path)
        
        if classes is None and nc is None:
            raise ValueError("Forneça 'classes' ou 'nc'")
        
        if nc is None:
            nc = len(classes)
        
        if classes is None:
            classes = [f"class_{i}" for i in range(nc)]
        
        yaml_config = {
            'path': str(dataset_path.absolute()),
            'train': train_path,
            'val': val_path,
            'nc': nc,
            'names': classes
        }
        
        if test_path:
            yaml_config['test'] = test_path
        
        yaml_path = dataset_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        return yaml_path
    
    def split_dataset(
        self,
        source_path: Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[Path, Path, Path]:
        """
        Divide dataset em treino, validação e teste.
        
        Args:
            source_path: Caminho com imagens e labels
            train_ratio: Proporção para treino
            val_ratio: Proporção para validação
            test_ratio: Proporção para teste
            seed: Seed para reprodutibilidade
            
        Returns:
            Tupla com caminhos (train, val, test)
        """
        source_path = Path(source_path)
        
        # Valida proporções
        total = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Soma das proporções deve ser 1.0, obtido {total}")
        
        # Encontra todas as imagens
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        images = []
        for ext in image_extensions:
            images.extend(source_path.glob(f'**/*{ext}'))
        
        if not images:
            raise ValueError(f"Nenhuma imagem encontrada em {source_path}")
        
        # Embaralha com seed
        random.seed(seed)
        random.shuffle(images)
        
        # Calcula divisões
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Cria estrutura de diretórios
        output_path = self.data_path / 'split_dataset'
        train_path = output_path / 'train'
        val_path = output_path / 'val'
        test_path = output_path / 'test'
        
        for split_path in [train_path, val_path, test_path]:
            (split_path / 'images').mkdir(parents=True, exist_ok=True)
            (split_path / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Copia arquivos
        def copy_files(image_list: List[Path], dest_path: Path):
            for img_path in image_list:
                # Copia imagem
                dest_img = dest_path / 'images' / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Procura label correspondente
                label_path = img_path.parent.parent / 'labels' / f"{img_path.stem}.txt"
                if not label_path.exists():
                    label_path = img_path.with_suffix('.txt')
                
                if label_path.exists():
                    dest_label = dest_path / 'labels' / f"{img_path.stem}.txt"
                    shutil.copy2(label_path, dest_label)
        
        copy_files(train_images, train_path)
        copy_files(val_images, val_path)
        copy_files(test_images, test_path)
        
        return train_path, val_path, test_path
    
    def get_dataset_stats(self, dataset_path: Path) -> Dict:
        """
        Obtém estatísticas do dataset.
        
        Args:
            dataset_path: Caminho do dataset
            
        Returns:
            Dicionário com estatísticas
        """
        dataset_path = Path(dataset_path)
        
        stats = {
            'total_images': 0,
            'total_labels': 0,
            'classes': defaultdict(int),
            'splits': {}
        }
        
        # Analisa cada split
        for split in ['train', 'val', 'test']:
            split_path = dataset_path / split
            if not split_path.exists():
                continue
            
            images_path = split_path / 'images'
            labels_path = split_path / 'labels'
            
            n_images = len(list(images_path.glob('*'))) if images_path.exists() else 0
            n_labels = len(list(labels_path.glob('*.txt'))) if labels_path.exists() else 0
            
            stats['splits'][split] = {
                'images': n_images,
                'labels': n_labels
            }
            
            stats['total_images'] += n_images
            stats['total_labels'] += n_labels
            
            # Conta classes
            if labels_path.exists():
                for label_file in labels_path.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                stats['classes'][class_id] += 1
        
        return stats
    
    def subsample_dataset(
        self,
        source_path: Path,
        output_path: Path,
        sample_ratio: float = 1.0,
        seed: int = 42
    ) -> Path:
        """
        Cria uma subamostra do dataset.
        
        Args:
            source_path: Caminho do dataset original
            output_path: Caminho para salvar subamostra
            sample_ratio: Proporção de dados a manter (0.0 a 1.0)
            seed: Seed para reprodutibilidade
            
        Returns:
            Caminho do dataset amostrado
        """
        source_path = Path(source_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        random.seed(seed)
        
        # Para cada split
        for split in ['train', 'val', 'test']:
            split_source = source_path / split
            if not split_source.exists():
                continue
            
            split_output = output_path / split
            (split_output / 'images').mkdir(parents=True, exist_ok=True)
            (split_output / 'labels').mkdir(parents=True, exist_ok=True)
            
            # Lista imagens
            images = list((split_source / 'images').glob('*'))
            
            # Amostra
            n_sample = int(len(images) * sample_ratio)
            sampled_images = random.sample(images, n_sample)
            
            # Copia
            for img_path in sampled_images:
                # Imagem
                dest_img = split_output / 'images' / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Label
                label_path = split_source / 'labels' / f"{img_path.stem}.txt"
                if label_path.exists():
                    dest_label = split_output / 'labels' / f"{img_path.stem}.txt"
                    shutil.copy2(label_path, dest_label)
        
        # Copia data.yaml se existir
        yaml_source = source_path / 'data.yaml'
        if yaml_source.exists():
            yaml_dest = output_path / 'data.yaml'
            
            # Atualiza path no yaml
            with open(yaml_source, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            yaml_config['path'] = str(output_path.absolute())
            
            with open(yaml_dest, 'w') as f:
                yaml.dump(yaml_config, f, default_flow_style=False)
        
        return output_path
    
    def validate_dataset(self, dataset_path: Path) -> Tuple[bool, List[str]]:
        """
        Valida estrutura do dataset YOLO.
        
        Args:
            dataset_path: Caminho do dataset
            
        Returns:
            Tupla (é_válido, lista_de_erros)
        """
        dataset_path = Path(dataset_path)
        errors = []
        
        # Verifica data.yaml
        yaml_path = dataset_path / 'data.yaml'
        if not yaml_path.exists():
            errors.append("Arquivo data.yaml não encontrado")
        
        # Verifica splits
        required_splits = ['train', 'val']
        for split in required_splits:
            split_path = dataset_path / split
            if not split_path.exists():
                errors.append(f"Split '{split}' não encontrado")
                continue
            
            images_path = split_path / 'images'
            labels_path = split_path / 'labels'
            
            if not images_path.exists():
                errors.append(f"Pasta 'images' não encontrada em {split}")
            
            if not labels_path.exists():
                errors.append(f"Pasta 'labels' não encontrada em {split}")
            
            # Verifica se há imagens
            if images_path.exists():
                images = list(images_path.glob('*'))
                if not images:
                    errors.append(f"Nenhuma imagem encontrada em {split}/images")
        
        return len(errors) == 0, errors
