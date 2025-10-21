import os

# Defaults for training UI (básico)
DEFAULT_MODEL = os.getenv('YOLO_TRAIN_MODEL', 'yolov8s.pt')
DEFAULT_DATA = os.getenv('YOLO_TRAIN_DATA', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'no_resize', 'data.yaml'))
DEFAULT_EPOCHS = int(os.getenv('YOLO_TRAIN_EPOCHS', '50'))
DEFAULT_IMGSZ = int(os.getenv('YOLO_TRAIN_IMGSZ', '640'))
DEFAULT_BATCH = os.getenv('YOLO_TRAIN_BATCH', 'auto')  # 'auto' or int
DEFAULT_DEVICE = os.getenv('YOLO_TRAIN_DEVICE', 'cuda')  # 'cpu' or 'cuda' or '0,1'
DEFAULT_PROJECT = os.getenv('YOLO_TRAIN_PROJECT', 'runs/detect')
DEFAULT_NAME = os.getenv('YOLO_TRAIN_NAME', 'train_ui')

# Catálogos de parâmetros YOLO com defaults usados pela lib
# Observação: estes valores refletem os padrões do Ultralytics YOLO para treinamento de detecção.
YOLO_OPTIM_PARAMS = {
    'lr0': {'type': 'float', 'default': 0.01, 'min': 1e-5, 'max': 1.0, 'decimals': 5, 'step': 1e-4, 'label': 'lr0'},
    'lrf': {'type': 'float', 'default': 0.01, 'min': 1e-5, 'max': 1.0, 'decimals': 5, 'step': 1e-4, 'label': 'lrf'},
    'momentum': {'type': 'float', 'default': 0.937, 'min': 0.0, 'max': 0.999, 'decimals': 4, 'step': 1e-4, 'label': 'momentum'},
    'weight_decay': {'type': 'float', 'default': 0.0005, 'min': 0.0, 'max': 1.0, 'decimals': 6, 'step': 1e-6, 'label': 'weight_decay'},
    'warmup_epochs': {'type': 'float', 'default': 3.0, 'min': 0.0, 'max': 20.0, 'decimals': 2, 'step': 0.1, 'label': 'warmup_epochs'},
    'optimizer': {'type': 'choice', 'default': 'SGD', 'choices': ['SGD', 'Adam', 'AdamW', 'RMSProp'], 'label': 'optimizer'},
    'cos_lr': {'type': 'bool', 'default': False, 'label': 'cos_lr'},
    'label_smoothing': {'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 0.5, 'decimals': 3, 'step': 0.01, 'label': 'label_smoothing'},
    'patience': {'type': 'int', 'default': 50, 'min': -1, 'max': 1000, 'label': 'patience'},
    'seed': {'type': 'int', 'default': 0, 'min': 0, 'max': 999999, 'label': 'seed'},
}

YOLO_AUG_PARAMS = {
    'degrees': {'type': 'float', 'default': 0.0, 'min': -180.0, 'max': 180.0, 'decimals': 1, 'step': 1.0, 'label': 'degrees'},
    'translate': {'type': 'float', 'default': 0.1, 'min': 0.0, 'max': 1.0, 'decimals': 2, 'step': 0.01, 'label': 'translate'},
    'scale': {'type': 'float', 'default': 0.5, 'min': 0.0, 'max': 3.0, 'decimals': 2, 'step': 0.01, 'label': 'scale'},
    'shear': {'type': 'float', 'default': 0.0, 'min': -180.0, 'max': 180.0, 'decimals': 1, 'step': 1.0, 'label': 'shear'},
    'perspective': {'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 1.0, 'decimals': 3, 'step': 0.01, 'label': 'perspective'},
    'flipud': {'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 1.0, 'decimals': 2, 'step': 0.01, 'label': 'flipud'},
    'fliplr': {'type': 'float', 'default': 0.5, 'min': 0.0, 'max': 1.0, 'decimals': 2, 'step': 0.01, 'label': 'fliplr'},
    'mosaic': {'type': 'float', 'default': 1.0, 'min': 0.0, 'max': 1.0, 'decimals': 2, 'step': 0.01, 'label': 'mosaic'},
    'mixup': {'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 1.0, 'decimals': 2, 'step': 0.01, 'label': 'mixup'},
    'copy_paste': {'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 1.0, 'decimals': 2, 'step': 0.01, 'label': 'copy_paste'},
    'hsv_h': {'type': 'float', 'default': 0.015, 'min': 0.0, 'max': 1.0, 'decimals': 3, 'step': 0.001, 'label': 'hsv_h'},
    'hsv_s': {'type': 'float', 'default': 0.7, 'min': 0.0, 'max': 1.0, 'decimals': 2, 'step': 0.01, 'label': 'hsv_s'},
    'hsv_v': {'type': 'float', 'default': 0.4, 'min': 0.0, 'max': 1.0, 'decimals': 2, 'step': 0.01, 'label': 'hsv_v'},
}

# Parâmetros adicionais expostos como "avançado"
YOLO_ADV_PARAMS = {
    'max_det': {'type': 'int', 'default': 300, 'min': 1, 'max': 10000, 'label': 'max_det'},
    'save_period': {'type': 'int', 'default': -1, 'min': -1, 'max': 1000, 'label': 'save_period'},
    'workers': {'type': 'int', 'default': 8, 'min': 0, 'max': 64, 'label': 'workers'},
    'cache': {'type': 'bool', 'default': False, 'label': 'cache'},
    'freeze': {'type': 'int', 'default': 0, 'min': 0, 'max': 50, 'label': 'freeze'},
    'rect': {'type': 'bool', 'default': False, 'label': 'rect'},
    'close_mosaic': {'type': 'int', 'default': 10, 'min': 0, 'max': 50, 'label': 'close_mosaic'},
    'amp': {'type': 'bool', 'default': True, 'label': 'amp'},
    'exist_ok': {'type': 'bool', 'default': False, 'label': 'exist_ok'},
    'resume': {'type': 'bool', 'default': False, 'label': 'resume'},
}
