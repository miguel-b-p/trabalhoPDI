import os
import json
from typing import Any, Dict

from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox,
    QDoubleSpinBox, QCheckBox, QTextEdit, QTabWidget, QMessageBox, QComboBox,
    QMenu
)
from PyQt5.QtCore import Qt


# Suporte a execução direta (python src/train/ui.py) e como módulo (python -m src.train.ui)
try:
    from . import config as cfg
    from .logic import TrainWorker
except ImportError:
    import sys as _sys, os as _os
    _sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from train import config as cfg  # type: ignore
    from train.logic import TrainWorker  # type: ignore


class TrainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO - Treinamento")
        self.setMinimumSize(700, 600)

        root = QWidget()
        self.setCentralWidget(root)
        self.main_layout = QVBoxLayout(root)

        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self._build_basic_tab()
        self._build_optim_tab()
        self._build_aug_tab()
        self._build_adv_tab()
        self._build_bottom_panel()

    # ----- Tabs -----
    def _build_basic_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        # Dataset
        g_data = QGroupBox("Dataset")
        l_data = QHBoxLayout(g_data)
        self.data_edit = QLineEdit(cfg.DEFAULT_DATA)
        btn_data = QPushButton("Selecionar...")
        btn_data.clicked.connect(self._choose_data)
        l_data.addWidget(QLabel("data.yaml:")); l_data.addWidget(self.data_edit); l_data.addWidget(btn_data)
        layout.addWidget(g_data)

        # Modelo
        g_model = QGroupBox("Modelo")
        l_model = QHBoxLayout(g_model)
        self.model_edit = QLineEdit(cfg.DEFAULT_MODEL)
        btn_model = QPushButton("Selecionar...")
        btn_model.clicked.connect(self._choose_model)
        l_model.addWidget(QLabel("Modelo (.pt):")); l_model.addWidget(self.model_edit); l_model.addWidget(btn_model)
        layout.addWidget(g_model)

        # Treino básico
        g_train = QGroupBox("Treinamento")
        l_train = QHBoxLayout(g_train)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 1000); self.epochs_spin.setValue(cfg.DEFAULT_EPOCHS)
        self.imgsz_spin = QSpinBox(); self.imgsz_spin.setRange(64, 4096); self.imgsz_spin.setValue(cfg.DEFAULT_IMGSZ)
        self.batch_edit = QLineEdit(str(cfg.DEFAULT_BATCH))
        self.device_edit = QLineEdit(cfg.DEFAULT_DEVICE)
        l_train.addWidget(QLabel("epochs:")); l_train.addWidget(self.epochs_spin)
        l_train.addWidget(QLabel("imgsz:")); l_train.addWidget(self.imgsz_spin)
        l_train.addWidget(QLabel("batch:")); l_train.addWidget(self.batch_edit)
        l_train.addWidget(QLabel("device:")); l_train.addWidget(self.device_edit)
        layout.addWidget(g_train)

        # Output
        g_out = QGroupBox("Saída")
        l_out = QHBoxLayout(g_out)
        self.project_edit = QLineEdit(cfg.DEFAULT_PROJECT)
        self.name_edit = QLineEdit(cfg.DEFAULT_NAME)
        l_out.addWidget(QLabel("project:")); l_out.addWidget(self.project_edit)
        l_out.addWidget(QLabel("name:")); l_out.addWidget(self.name_edit)
        layout.addWidget(g_out)

        layout.addStretch(1)
        self.tabs.addTab(w, "Básico")

    def _build_optim_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        self.optim_group = QGroupBox("Otimização")
        g_layout = QVBoxLayout(self.optim_group)
        # container onde linhas de parâmetros serão inseridas dinamicamente
        self.optim_params_container = QVBoxLayout()
        g_layout.addLayout(self.optim_params_container)
        # botão + para adicionar parâmetros disponíveis
        add_btn = QPushButton("+ Adicionar parâmetro")
        add_btn.clicked.connect(lambda: self._show_add_menu(add_btn, cfg.YOLO_OPTIM_PARAMS, self.optim_params_widgets, self.optim_params_container))
        g_layout.addWidget(add_btn, alignment=Qt.AlignLeft)
        layout.addWidget(self.optim_group)
        layout.addStretch(1)
        # dicionário para rastrear widgets adicionados
        self.optim_params_widgets: Dict[str, Any] = {}
        self.tabs.addTab(w, "Otimização")

    def _build_aug_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        self.aug_group = QGroupBox("Aumentação")
        g_layout = QVBoxLayout(self.aug_group)
        self.aug_params_container = QVBoxLayout()
        g_layout.addLayout(self.aug_params_container)
        add_btn = QPushButton("+ Adicionar parâmetro")
        add_btn.clicked.connect(lambda: self._show_add_menu(add_btn, cfg.YOLO_AUG_PARAMS, self.aug_params_widgets, self.aug_params_container))
        g_layout.addWidget(add_btn, alignment=Qt.AlignLeft)
        layout.addWidget(self.aug_group)
        layout.addStretch(1)
        self.aug_params_widgets: Dict[str, Any] = {}
        self.tabs.addTab(w, "Aumentação")

    def _build_adv_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        self.adv_group = QGroupBox("Avançado")
        g_layout = QVBoxLayout(self.adv_group)
        self.adv_params_container = QVBoxLayout()
        g_layout.addLayout(self.adv_params_container)
        add_btn = QPushButton("+ Adicionar parâmetro")
        add_btn.clicked.connect(lambda: self._show_add_menu(add_btn, cfg.YOLO_ADV_PARAMS, self.adv_params_widgets, self.adv_params_container))
        g_layout.addWidget(add_btn, alignment=Qt.AlignLeft)
        layout.addWidget(self.adv_group)
        layout.addStretch(1)
        self.adv_params_widgets: Dict[str, Any] = {}
        self.tabs.addTab(w, "Avançado")

    def _build_bottom_panel(self):
        # rodapé embutido no layout principal (evita DockWidgets)
        footer = QGroupBox("Execução")
        l = QVBoxLayout(footer)

        # botões
        btn_bar = QHBoxLayout()
        self.btn_start = QPushButton("Iniciar Treino")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton("Parar")
        self.btn_stop.clicked.connect(self._stop)
        btn_bar.addWidget(self.btn_start)
        btn_bar.addWidget(self.btn_stop)
        l.addLayout(btn_bar)

        # logs
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(180)
        l.addWidget(self.log)

        self.main_layout.addWidget(footer)

    # ----- Helpers para parâmetros dinâmicos -----
    def _show_add_menu(self, anchor_btn: QPushButton, catalog: Dict[str, Dict[str, Any]], registry: Dict[str, Any], container_layout: QVBoxLayout):
        menu = QMenu(self)
        # lista somente parâmetros ainda não adicionados
        available = [k for k in catalog.keys() if k not in registry]
        if not available:
            act = menu.addAction("Nenhum parâmetro disponível")
            act.setEnabled(False)
        else:
            for pname in available:
                menu.addAction(pname, lambda pn=pname: self._add_param(pn, catalog[pn], registry, container_layout))
        menu.exec_(anchor_btn.mapToGlobal(anchor_btn.rect().bottomLeft()))

    def _add_param(self, name: str, spec: Dict[str, Any], registry: Dict[str, Any], container_layout: QVBoxLayout):
        if name in registry:
            return
        row = QWidget()
        h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel(spec.get('label', name)))

        wtype = spec.get('type', 'float')
        input_widget: Any
        if wtype == 'bool':
            input_widget = QCheckBox()
            input_widget.setChecked(bool(spec.get('default', False)))
            h.addWidget(input_widget)
        elif wtype == 'choice':
            input_widget = QComboBox()
            for c in spec.get('choices', []):
                input_widget.addItem(str(c))
            default_choice = str(spec.get('default', ''))
            idx = max(0, input_widget.findText(default_choice))
            input_widget.setCurrentIndex(idx)
            h.addWidget(input_widget)
        elif wtype == 'int':
            input_widget = QSpinBox()
            input_widget.setRange(int(spec.get('min', -10**9)), int(spec.get('max', 10**9)))
            input_widget.setSingleStep(int(max(1, spec.get('step', 1))))
            input_widget.setValue(int(spec.get('default', 0)))
            h.addWidget(input_widget)
        else:  # float
            input_widget = QDoubleSpinBox()
            input_widget.setRange(float(spec.get('min', -1e12)), float(spec.get('max', 1e12)))
            input_widget.setDecimals(int(spec.get('decimals', 3)))
            step = float(spec.get('step', 0.1))
            if step > 0:
                input_widget.setSingleStep(step)
            input_widget.setValue(float(spec.get('default', 0.0)))
            h.addWidget(input_widget)

        btn_rm = QPushButton("x")
        btn_rm.setFixedWidth(24)
        def _remove():
            # remove do layout e do registro
            for i in range(container_layout.count()):
                item = container_layout.itemAt(i)
                if item and item.widget() is row:
                    container_layout.takeAt(i)
                    break
            row.deleteLater()
            registry.pop(name, None)
        btn_rm.clicked.connect(_remove)
        h.addWidget(btn_rm)

        container_layout.addWidget(row)
        registry[name] = {'widget': input_widget, 'type': wtype, 'row': row}

    def _collect_dynamic_params(self, registry: Dict[str, Any], catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, meta in registry.items():
            w = meta['widget']; t = meta['type']
            if t == 'bool':
                out[name] = bool(w.isChecked())
            elif t == 'choice':
                out[name] = str(w.currentText())
            elif t == 'int':
                out[name] = int(w.value())
            else:  # float
                out[name] = float(w.value())
        return out

    # ----- Actions -----
    def _choose_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar data.yaml", os.path.dirname(self.data_edit.text()), "YAML (*.yaml)")
        if path:
            self.data_edit.setText(path)

    def _choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar modelo", os.path.dirname(self.model_edit.text()), "Modelos (*.pt *.onnx *.engine *.xml *.bin)")
        if path:
            self.model_edit.setText(path)

    def _params(self) -> Dict[str, Any]:
        batch_val = self.batch_edit.text().strip()
        try:
            batch = int(batch_val)
        except Exception:
            batch = batch_val if batch_val else 'auto'

        params: Dict[str, Any] = {
            'data': self.data_edit.text(),
            'model': self.model_edit.text(),
            'epochs': self.epochs_spin.value(),
            'imgsz': self.imgsz_spin.value(),
            'batch': batch,
            'device': self.device_edit.text(),
            'project': self.project_edit.text(),
            'name': self.name_edit.text(),
        }

        # Coletar parâmetros adicionados dinamicamente nas abas
        params.update(self._collect_dynamic_params(self.optim_params_widgets, cfg.YOLO_OPTIM_PARAMS))
        params.update(self._collect_dynamic_params(self.aug_params_widgets, cfg.YOLO_AUG_PARAMS))
        params.update(self._collect_dynamic_params(self.adv_params_widgets, cfg.YOLO_ADV_PARAMS))
        return params

    def _start(self):
        params = self._params()
        self.worker = TrainWorker(params)
        self.worker.log.connect(self._on_log)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self._on_log("Treinamento iniciado.")

    def _stop(self):
        try:
            if hasattr(self, 'worker') and self.worker.isRunning():
                self.worker.terminate()
                self._on_log("Solicitação de parada enviada.")
        except Exception:
            pass

    def _on_log(self, msg: str):
        self.log.append(msg)

    def _on_error(self, msg: str):
        self.log.append(f"❌ {msg}")

    def _on_finished(self, info: Dict[str, Any]):
        self.log.append("✅ Treinamento finalizado.")


def main():
    import sys
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    w = TrainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
