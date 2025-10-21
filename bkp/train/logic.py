import os
import threading
from typing import Dict, Any, Optional

from PyQt5.QtCore import QThread, pyqtSignal

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    _import_err = e


class TrainWorker(QThread):
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.params = params
        self._stop = threading.Event()

    def run(self):
        if YOLO is None:
            self.error.emit(f"Ultralytics não disponível: {_import_err}")
            return
        try:
            model = YOLO(self.params.get('model', 'yolov8s.pt'))
            self.log.emit("Modelo carregado.")

            # Monta kwargs conforme UI - apenas o essencial + o que o usuário escolheu,
            # deixando YOLO aplicar defaults para o resto
            train_kwargs = {
                'data': self.params['data'],
                'epochs': int(self.params['epochs']),
                'imgsz': int(self.params['imgsz']),
                'device': self.params.get('device', 'cuda'),
                'project': self.params.get('project'),
                'name': self.params.get('name'),
                'batch': self.params.get('batch'),
                'plots': True,
            }

            # tudo mais: copiar diretamente do params; eles já refletem apenas o que o usuário adicionou
            for k, v in self.params.items():
                if k in train_kwargs or k in ['data', 'epochs', 'imgsz', 'device', 'project', 'name', 'batch']:
                    continue
                train_kwargs[k] = v

            self.log.emit(f"Iniciando treino com args: {train_kwargs}")
            results = model.train(**train_kwargs)
            self.finished.emit({'ok': True, 'results': str(results)})
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self._stop.set()
