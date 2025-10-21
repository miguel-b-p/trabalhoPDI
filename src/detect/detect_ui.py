import sys
import os
import json
import time
from typing import List

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout,
    QWidget, QPushButton, QTextEdit, QSlider, QHBoxLayout,
    QSpinBox, QDoubleSpinBox, QGroupBox, QComboBox, QMessageBox, QCheckBox,
    QTabWidget, QAction, QMenu, QFileDialog, QLineEdit
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QRect, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QTextCursor
from PyQt5.QtGui import QPainter, QColor, QPen

# --- Módulos de Lógica (simulados para o exemplo ser executável) ---
# Em seu projeto, use seus imports reais
try:
    import config as cfg
    from detect_logic import (
        WindowSelector,
        FastCaptureThread,
        FFMpegCaptureThread,
        DetectionThread,
        HAS_EWMH,
        HAS_UINPUT,
    )
except ImportError:
    # --- Início: Simulação de dependências para o código rodar ---
    # Remova ou substitua esta seção por seus módulos reais
    class MockConfig:
        OPENCV_THREADS = 4
        USE_OPENCL = True
        MODEL_PATH = "yolov8n.pt" # Caminho padrão
        DEFAULT_FPS_LIMIT = 60
        DEFAULT_CONFIDENCE = 0.35
        PIXEL_STEP_DEFAULT = 1
        INFER_FPS_LIMIT_DEFAULT = 15
        DETECT_RESOLUTION = (640, 640)
        ONLY_CURRENT_DESKTOP_DEFAULT = True
        CAPTURE_ENGINE_DEFAULT = 'mss'
        DRAW_FOV_DEFAULT = False
        FOV_RADIUS_DEFAULT = 150
        REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

    cfg = MockConfig()
    HAS_EWMH = sys.platform == "linux" # Simulação
    HAS_UINPUT = sys.platform == "linux" # Simulação

    class WindowSelector:
        @staticmethod
        def get_available_windows(only_current_desktop):
            print(f"Buscando janelas (apenas desktop atual: {only_current_desktop})")
            return [
                {"type": "monitor", "name": "Monitor 1 (1920x1080)", "monitor_index": 0},
                {"type": "window", "name": "Exemplo de Janela 1", "window_id": 123},
                {"type": "window", "name": "Exemplo de Janela 2", "window_id": 456},
            ]
        
        @staticmethod
        def get_window_region(info):
             if not info: return None
             if info.get("type") == "monitor":
                 return {"left": 0, "top": 0, "width": 1920, "height": 1080}
             return {"left": 100, "top": 100, "width": 800, "height": 600}

    class MockThread(QThread):
        """Classe base para simular as threads."""
        error_occurred = pyqtSignal(str)
        def __init__(self):
            super().__init__()
        def stop(self):
            self.quit()
            self.wait()

    class FastCaptureThread(MockThread):
        frame_ready = pyqtSignal(np.ndarray)
        region_changed = pyqtSignal(dict)
        def __init__(self, res, fps): super().__init__()
        def set_target_size(self, res): pass
        def set_fps_limit(self, fps): pass
        def set_region(self, region): self.region_changed.emit(region)
    
    class FFMpegCaptureThread(FastCaptureThread): pass

    class DetectionThread(MockThread):
        detection_ready = pyqtSignal(list, float)
        def __init__(self, model_path):
            super().__init__()
            self.model_path = model_path
        def set_frame(self, frame): pass
        def set_confidence(self, conf): pass
        def set_capture_region(self, region): pass
        def set_mouse_control_enabled(self, enabled): pass
        def set_mouse_gain(self, gain): pass
        def set_mouse_max_step(self, step): pass
        def set_axis_enabled(self, x=None, y=None): pass
        def set_color_filter(self, enabled, hsv, tol_h, min_s, min_v): pass
    # --- Fim: Simulação de dependências ---


class OverlayWindow(QWidget):
    """Janela transparente em tela inteira que desenha apenas os contornos detectados."""
    def __init__(self):
        super().__init__()
        flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        try:
            if hasattr(Qt, 'WindowTransparentForInput'):
                flags |= Qt.WindowTransparentForInput
        except Exception:
            pass
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        try:
            self.setWindowFlag(Qt.WindowDoesNotAcceptFocus, True)
        except Exception:
            pass
        self.setFocusPolicy(Qt.NoFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self._boxes = []
        self._offset_x = 0
        self._offset_y = 0
        self._draw_fov = bool(getattr(cfg, 'DRAW_FOV_DEFAULT', False))
        self._fov_radius = int(getattr(cfg, 'FOV_RADIUS_DEFAULT', 150))
        self._fov_center = (0, 0)
        self._fov_offset_x = 0
        self._fov_offset_y = 0

    def show_all_screens(self):
        try:
            screens = QApplication.screens()
            if not screens:
                self.showFullScreen()
                return
            min_x = min(s.geometry().x() for s in screens)
            min_y = min(s.geometry().y() for s in screens)
            max_r = max(s.geometry().x() + s.geometry().width() for s in screens)
            max_b = max(s.geometry().y() + s.geometry().height() for s in screens)
            self.setGeometry(min_x, min_y, max_r - min_x, max_b - min_y)
            self.show()
        except Exception:
            self.showFullScreen()

    def clear_boxes(self):
        self._boxes = []
        self.update()

    def set_boxes(self, boxes: list):
        self._boxes = boxes or []
        self.update()

    def set_offset(self, dx: int, dy: int):
        self._offset_x = int(dx)
        self._offset_y = int(dy)
        self.update()

    def paintEvent(self, event):
        if not self._boxes and not (self._draw_fov and self._fov_radius > 0):
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        box_pen = QPen(QColor(80, 220, 100, 255))
        box_pen.setWidth(3)
        painter.setPen(box_pen)
        for b in self._boxes:
            try:
                x1, y1, x2, y2 = int(b.get('x1', 0)), int(b.get('y1', 0)), int(b.get('x2', 0)), int(b.get('y2', 0))
                x1 += self._offset_x; x2 += self._offset_x
                y1 += self._offset_y; y2 += self._offset_y
                painter.drawRect(x1, y1, max(0, x2 - x1), max(0, y2 - y1))
                label = b.get('label')
                conf = b.get('conf')
                if label is not None and conf is not None:
                    text = f"{label} {conf:.2f}"
                    bg = QColor(0, 0, 0, 140)
                    fm = painter.fontMetrics()
                    tw = fm.horizontalAdvance(text) + 8
                    th = fm.height() + 4
                    painter.fillRect(x1, max(0, y1 - th), tw, th, bg)
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(x1 + 4, max(0, y1 - 4), text)
                    painter.setPen(box_pen)
            except Exception:
                continue

        if self._draw_fov and self._fov_radius > 0:
            try:
                cx = int(self._fov_center[0]) + int(self._fov_offset_x)
                cy = int(self._fov_center[1]) + int(self._fov_offset_y)
                fov_pen = QPen(QColor(30, 144, 255, 220))
                fov_pen.setWidth(2)
                painter.setPen(fov_pen)
                painter.drawEllipse(cx - self._fov_radius, cy - self._fov_radius, self._fov_radius * 2, self._fov_radius * 2)
            except Exception:
                pass

    def set_fov_draw_enabled(self, enabled: bool):
        self._draw_fov = bool(enabled)
        self.update()

    def set_fov_radius(self, radius_px: int):
        self._fov_radius = max(0, int(radius_px))
        self.update()

    def set_fov_center_from_region(self, region: dict):
        try:
            cx = int(region.get('left', 0)) + int(region.get('width', 0)) // 2
            cy = int(region.get('top', 0)) + int(region.get('height', 0)) // 2
            self._fov_center = (cx, cy)
            self.update()
        except Exception: pass

    def set_fov_offset(self, dx: int, dy: int):
        try:
            self._fov_offset_x = int(dx)
            self._fov_offset_y = int(dy)
            self.update()
        except Exception: pass


class RegionSelector(QWidget):
    """Widget para selecionar uma região da tela"""
    region_selected = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self.start_pos = None
        self.selection_rect = None
        
        # Cobrir todos os monitores
        desktop_rect = QApplication.desktop().virtualGeometry()
        self.setGeometry(desktop_rect)

    def paintEvent(self, event):
        if self.selection_rect:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(self.selection_rect, QColor(0, 0, 0, 0))
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            pen = QPen(QColor(0, 120, 215), 2)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)

    def mousePressEvent(self, event):
        self.start_pos = event.globalPos()
        self.selection_rect = QRect()

    def mouseMoveEvent(self, event):
        if self.start_pos:
            self.selection_rect = QRect(self.mapFromGlobal(self.start_pos), self.mapFromGlobal(event.globalPos())).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.start_pos:
            rect = QRect(self.start_pos, event.globalPos()).normalized()
            if rect.width() > 10 and rect.height() > 10:
                self.region_selected.emit({
                    "left": rect.x(), "top": rect.y(),
                    "width": rect.width(), "height": rect.height(), "type": "custom"
                })
        self.hide()
        self.deleteLater()


class ColorPickerOverlay(QWidget):
    """Overlay em tela cheia para escolher cor por clique."""
    color_picked = pyqtSignal(tuple)  # (h, s, v)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self._shot = None

    def start(self):
        try:
            screen = QApplication.primaryScreen()
            self._shot = screen.grabWindow(QApplication.desktop().winId())
            self.setGeometry(screen.geometry())
            self.showFullScreen()
        except Exception as e:
            print(f"Erro ao capturar tela para color picker: {e}")
            self.deleteLater()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._shot:
            pos = event.pos()
            c = self._shot.toImage().pixelColor(pos)
            r, g, b = c.red(), c.green(), c.blue()
            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
            self.color_picked.emit(tuple(int(v) for v in hsv))
        self.hide()
        self.deleteLater()


class ScreenDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Screen Detector")
        self.setGeometry(100, 100, 950, 850)

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        cv2.setNumThreads(cfg.OPENCV_THREADS)
        cv2.ocl.setUseOpenCL(cfg.USE_OPENCL)

        model_path = cfg.MODEL_PATH
        if not os.path.exists(model_path):
            print(f"Modelo não encontrado em: {model_path}")
            model_path = ""

        self.running = False
        self.fps_limit = cfg.DEFAULT_FPS_LIMIT
        self.confidence = cfg.DEFAULT_CONFIDENCE
        self.current_window_info = None
        self.only_current_desktop = cfg.ONLY_CURRENT_DESKTOP_DEFAULT
        self.overlay_enabled = True

        self.capture_thread = None # Será criado ao iniciar
        self.detection_thread = DetectionThread(model_path)
        self.detection_thread.detection_ready.connect(self.update_display_overlay)

        self.overlay = OverlayWindow()
        self.color_picker = None
        self._color_h, self._color_s, self._color_v = 0, 0, 0

        self._build_menubar()
        self.init_ui()
        self.statusBar().showMessage("Pronto")
        self.apply_theme("dark")

        self.refresh_windows_list()
        if self.window_combo.count() > 0:
            self.window_combo.setCurrentIndex(0)
            self.on_window_changed(0)
        
        try:
            default_engine = getattr(cfg, 'CAPTURE_ENGINE_DEFAULT', 'mss').lower()
            idx = self.engine_combo.findData(default_engine)
            if idx != -1: self.engine_combo.setCurrentIndex(idx)
        except Exception: pass

    # =========================================================================
    # == Construção da UI
    # =========================================================================
    def init_ui(self):
        """Inicializa a UI com um layout de abas para melhor organização."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        tabs = QTabWidget()
        tabs.addTab(self._create_main_tab(), "🚀 Principal")
        tabs.addTab(self._create_settings_tab(), "⚙️ Ajustes")
        tabs.addTab(self._create_advanced_tab(), "🔬 Avançado")
        main_layout.addWidget(tabs)

        display_layout = QHBoxLayout()
        self.image_label = QLabel("O preview da detecção aparecerá aqui.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid #2b2f3a; border-radius: 8px; background-color: #0d111c;")
        display_layout.addWidget(self.image_label, 2)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Monospace", 9))
        display_layout.addWidget(self.log_output, 1)

        main_layout.addLayout(display_layout)

    def _create_main_tab(self):
        """Cria a aba com os controles principais."""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setAlignment(Qt.AlignTop)

        capture_group = QGroupBox("🎯 Fonte de Captura")
        capture_layout = QVBoxLayout()
        source_layout = QHBoxLayout()
        self.window_combo = QComboBox()
        self.window_combo.currentIndexChanged.connect(self.on_window_changed)
        source_layout.addWidget(self.window_combo)
        self.refresh_button = QPushButton("Atualizar")
        self.refresh_button.clicked.connect(self.refresh_windows_list)
        source_layout.addWidget(self.refresh_button)
        self.checkbox_current_desktop = QCheckBox("Apenas desktop atual")
        if HAS_EWMH:
            self.checkbox_current_desktop.setChecked(self.only_current_desktop)
            self.checkbox_current_desktop.toggled.connect(self.on_toggle_current_desktop)
        else:
            self.checkbox_current_desktop.setEnabled(False)
            self.checkbox_current_desktop.setToolTip("Requer 'ewmh' para funcionar.")
        capture_layout.addLayout(source_layout)
        capture_layout.addWidget(self.checkbox_current_desktop)
        self.select_region_button = QPushButton("Selecionar Região Manualmente")
        self.select_region_button.clicked.connect(self.select_custom_region)
        capture_layout.addWidget(self.select_region_button)
        capture_group.setLayout(capture_layout)
        layout.addWidget(capture_group)

        model_group = QGroupBox("🧠 Modelo IA")
        model_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit(getattr(self.detection_thread, 'model_path', ''))
        self.browse_model_button = QPushButton("Procurar...")
        self.browse_model_button.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(self.browse_model_button)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        controls_group = QGroupBox("⏯️ Controles")
        controls_layout = QHBoxLayout()
        self.start_button = QPushButton("Iniciar Detecção")
        self.start_button.clicked.connect(self.toggle_detection)
        self.start_button.setStyleSheet("padding: 12px; font-weight: bold;")
        controls_layout.addWidget(self.start_button)
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        return tab_widget

    def _create_settings_tab(self):
        """Cria a aba de ajustes de desempenho e visualização."""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setAlignment(Qt.AlignTop)

        perf_group = QGroupBox("⚡ Desempenho e Precisão")
        perf_layout = QVBoxLayout()
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confiança:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 99)
        self.confidence_slider.setValue(int(self.confidence * 100))
        conf_layout.addWidget(self.confidence_slider)
        self.confidence_spinbox = QSpinBox()
        self.confidence_spinbox.setRange(1, 99)
        self.confidence_spinbox.setValue(int(self.confidence * 100))
        self.confidence_spinbox.setSuffix("%")
        conf_layout.addWidget(self.confidence_spinbox)
        self.confidence_slider.valueChanged.connect(self.confidence_spinbox.setValue)
        self.confidence_spinbox.valueChanged.connect(self.confidence_slider.setValue)
        self.confidence_spinbox.valueChanged.connect(self.on_confidence_changed)
        perf_layout.addLayout(conf_layout)

        fps_res_layout = QHBoxLayout()
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Rápido (mss)", "mss")
        self.engine_combo.addItem("Compatível (ffmpeg)", "ffmpeg")
        fps_res_layout.addWidget(QLabel("Motor:"))
        fps_res_layout.addWidget(self.engine_combo)
        fps_res_layout.addWidget(QLabel("FPS Captura:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(self.fps_limit)
        self.fps_spin.valueChanged.connect(self.on_fps_limit_changed)
        fps_res_layout.addWidget(self.fps_spin)
        fps_res_layout.addWidget(QLabel("Resolução:"))
        self.resize_w_spin = QSpinBox()
        self.resize_w_spin.setRange(320, 3840)
        self.resize_w_spin.setStep(16)
        self.resize_w_spin.setValue(cfg.DETECT_RESOLUTION[0])
        fps_res_layout.addWidget(self.resize_w_spin)
        self.resize_h_spin = QSpinBox()
        self.resize_h_spin.setRange(240, 2160)
        self.resize_h_spin.setStep(16)
        self.resize_h_spin.setValue(cfg.DETECT_RESOLUTION[1])
        fps_res_layout.addWidget(self.resize_h_spin)
        self.resize_w_spin.valueChanged.connect(self.on_resize_changed)
        self.resize_h_spin.valueChanged.connect(self.on_resize_changed)
        perf_layout.addLayout(fps_res_layout)
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)

        overlay_group = QGroupBox("🎨 Sobreposição (Overlay)")
        overlay_layout = QHBoxLayout()
        overlay_layout.addWidget(QLabel("Offset X:"))
        self.offset_x_spin = QSpinBox()
        self.offset_x_spin.setRange(-200, 200)
        self.offset_x_spin.setValue(0)
        self.offset_x_spin.valueChanged.connect(lambda v: self.overlay.set_offset(v, self.offset_y_spin.value()))
        overlay_layout.addWidget(self.offset_x_spin)
        overlay_layout.addWidget(QLabel("Offset Y:"))
        self.offset_y_spin = QSpinBox()
        self.offset_y_spin.setRange(-200, 200)
        self.offset_y_spin.setValue(0)
        self.offset_y_spin.valueChanged.connect(lambda v: self.overlay.set_offset(self.offset_x_spin.value(), v))
        overlay_layout.addWidget(self.offset_y_spin)
        overlay_group.setLayout(overlay_layout)
        layout.addWidget(overlay_group)
        
        fov_group = QGroupBox("⭕ Campo de Visão (FOV)")
        fov_layout = QVBoxLayout()
        self.draw_fov_checkbox = QCheckBox("Desenhar círculo de FOV")
        self.draw_fov_checkbox.setChecked(getattr(cfg, 'DRAW_FOV_DEFAULT', False))
        self.draw_fov_checkbox.toggled.connect(self.overlay.set_fov_draw_enabled)
        fov_layout.addWidget(self.draw_fov_checkbox)
        fov_controls_layout = QHBoxLayout()
        fov_controls_layout.addWidget(QLabel("Raio (px):"))
        self.fov_radius_spin = QSpinBox()
        self.fov_radius_spin.setRange(10, 1000)
        self.fov_radius_spin.setValue(getattr(cfg, 'FOV_RADIUS_DEFAULT', 150))
        self.fov_radius_spin.valueChanged.connect(self.overlay.set_fov_radius)
        fov_controls_layout.addWidget(self.fov_radius_spin)
        fov_controls_layout.addWidget(QLabel("Offset X:"))
        self.fov_offset_x_spin = QSpinBox()
        self.fov_offset_x_spin.setRange(-500, 500)
        self.fov_offset_x_spin.valueChanged.connect(lambda v: self.overlay.set_fov_offset(v, self.fov_offset_y_spin.value()))
        fov_controls_layout.addWidget(self.fov_offset_x_spin)
        fov_controls_layout.addWidget(QLabel("Offset Y:"))
        self.fov_offset_y_spin = QSpinBox()
        self.fov_offset_y_spin.setRange(-500, 500)
        self.fov_offset_y_spin.valueChanged.connect(lambda v: self.overlay.set_fov_offset(self.fov_offset_x_spin.value(), v))
        fov_controls_layout.addWidget(self.fov_offset_y_spin)
        fov_layout.addLayout(fov_controls_layout)
        fov_group.setLayout(fov_layout)
        layout.addWidget(fov_group)

        return tab_widget
    
    def _create_advanced_tab(self):
        """Cria a aba com configurações avançadas (mira, filtros)."""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setAlignment(Qt.AlignTop)

        mouse_group = QGroupBox("🖱️ Mira (Mouse Aiming)")
        if not HAS_UINPUT:
            mouse_group.setToolTip("Funcionalidade de mira desativada. Instale 'python-uinput' e execute como root.")
            mouse_group.setEnabled(False)
        mouse_layout = QVBoxLayout()
        self.mouse_checkbox = QCheckBox("Habilitar mira automática")
        self.mouse_checkbox.toggled.connect(self.on_mouse_control_toggled)
        mouse_layout.addWidget(self.mouse_checkbox)
        aim_controls = QHBoxLayout()
        aim_controls.addWidget(QLabel("Ganho:"))
        self.mouse_gain_spin = QDoubleSpinBox()
        self.mouse_gain_spin.setRange(0.1, 5.0)
        self.mouse_gain_spin.setSingleStep(0.1)
        self.mouse_gain_spin.setValue(0.6)
        self.mouse_gain_spin.valueChanged.connect(lambda v: self.detection_thread.set_mouse_gain(v))
        aim_controls.addWidget(self.mouse_gain_spin)
        aim_controls.addWidget(QLabel("Passo Máx:"))
        self.mouse_step_spin = QSpinBox()
        self.mouse_step_spin.setRange(1, 100)
        self.mouse_step_spin.setValue(15)
        self.mouse_step_spin.valueChanged.connect(lambda v: self.detection_thread.set_mouse_max_step(v))
        aim_controls.addWidget(self.mouse_step_spin)
        self.axis_x_checkbox = QCheckBox("Eixo X"); self.axis_x_checkbox.setChecked(True)
        self.axis_x_checkbox.toggled.connect(lambda c: self.detection_thread.set_axis_enabled(x=c))
        aim_controls.addWidget(self.axis_x_checkbox)
        self.axis_y_checkbox = QCheckBox("Eixo Y"); self.axis_y_checkbox.setChecked(True)
        self.axis_y_checkbox.toggled.connect(lambda c: self.detection_thread.set_axis_enabled(y=c))
        aim_controls.addWidget(self.axis_y_checkbox)
        mouse_layout.addLayout(aim_controls)
        mouse_group.setLayout(mouse_layout)
        layout.addWidget(mouse_group)

        color_group = QGroupBox("🌈 Filtro de Cor")
        color_layout = QVBoxLayout()
        self.color_filter_checkbox = QCheckBox("Habilitar filtro de cor por alvo")
        self.color_filter_checkbox.toggled.connect(self.on_color_filter_toggled)
        color_layout.addWidget(self.color_filter_checkbox)
        color_controls = QHBoxLayout()
        self.pick_color_button = QPushButton("Selecionar Cor na Tela")
        self.pick_color_button.clicked.connect(self.on_pick_color)
        color_controls.addWidget(self.pick_color_button)
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(24, 24)
        self.color_preview.setStyleSheet("border: 1px solid #555; border-radius: 4px; background: #000;")
        color_controls.addWidget(self.color_preview)
        color_layout.addLayout(color_controls)
        
        tolerances_layout = QHBoxLayout()
        tolerances_layout.addWidget(QLabel("Tol. Matiz(H):"))
        self.color_tol_h_spin = QSpinBox(); self.color_tol_h_spin.setRange(1, 45); self.color_tol_h_spin.setValue(15)
        self.color_tol_h_spin.valueChanged.connect(self.on_color_filter_toggled) # Atualiza filtro
        tolerances_layout.addWidget(self.color_tol_h_spin)
        tolerances_layout.addWidget(QLabel("Saturação(S) Mín:"))
        self.color_min_s_spin = QSpinBox(); self.color_min_s_spin.setRange(0, 255); self.color_min_s_spin.setValue(30)
        self.color_min_s_spin.valueChanged.connect(self.on_color_filter_toggled) # Atualiza filtro
        tolerances_layout.addWidget(self.color_min_s_spin)
        tolerances_layout.addWidget(QLabel("Valor(V) Mín:"))
        self.color_min_v_spin = QSpinBox(); self.color_min_v_spin.setRange(0, 255); self.color_min_v_spin.setValue(30)
        self.color_min_v_spin.valueChanged.connect(self.on_color_filter_toggled) # Atualiza filtro
        tolerances_layout.addWidget(self.color_min_v_spin)
        
        color_layout.addLayout(tolerances_layout)
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)

        return tab_widget

    def _build_menubar(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("📁 Arquivo")
        act_open_model = QAction("📂 Selecionar Modelo…", self, shortcut="Ctrl+O", triggered=self.browse_model)
        file_menu.addAction(act_open_model)
        act_save_cfg = QAction("💾 Salvar Configuração…", self, shortcut="Ctrl+S", triggered=self.save_config_to_json)
        file_menu.addAction(act_save_cfg)
        act_load_cfg = QAction("📥 Carregar Configuração…", self, shortcut="Ctrl+L", triggered=self.load_config_from_json)
        file_menu.addAction(act_load_cfg)
        file_menu.addSeparator()
        act_exit = QAction("🚪 Sair", self, shortcut="Ctrl+Q", triggered=self.close)
        file_menu.addAction(act_exit)

        view_menu = mb.addMenu("👁️ Exibir")
        self.act_dark = QAction("🌙 Tema escuro", self, checkable=True, checked=True)
        self.act_dark.triggered.connect(lambda c: self.apply_theme("dark" if c else "light"))
        view_menu.addAction(self.act_dark)
        self.act_overlay = QAction("🟩 Mostrar overlay", self, checkable=True, checked=True)
        self.act_overlay.toggled.connect(self.on_toggle_overlay_action)
        view_menu.addAction(self.act_overlay)

    # =========================================================================
    # == Lógica e Slots
    # =========================================================================
    def log(self, message):
        self.log_output.append(message)
        self.log_output.moveCursor(QTextCursor.End)

    def log_error(self, message):
        self.log_output.append(f'<span style="color: #ff4757;">{message}</span>')
        self.log_output.moveCursor(QTextCursor.End)

    def on_toggle_overlay_action(self, checked):
        self.overlay_enabled = bool(checked)
        if self.running:
            if self.overlay_enabled: self.overlay.show_all_screens()
            else: self.overlay.hide()

    def on_confidence_changed(self, value):
        self.confidence = value / 100.0
        self.detection_thread.set_confidence(self.confidence)
        self.log(f"Confiança ajustada para {self.confidence:.2f}")

    def on_fps_limit_changed(self, value):
        self.fps_limit = value
        if self.capture_thread:
            self.capture_thread.set_fps_limit(self.fps_limit)
        self.log(f"Limite de FPS de captura ajustado para {self.fps_limit}")

    def on_resize_changed(self):
        size = (self.resize_w_spin.value(), self.resize_h_spin.value())
        if self.capture_thread:
            self.capture_thread.set_target_size(size)
        self.log(f"Resolução de detecção alterada para {size[0]}x{size[1]}")

    def on_window_changed(self, index):
        if index < 0: return
        self.current_window_info = self.window_combo.itemData(index)
        self.log(f"Fonte selecionada: {self.window_combo.itemText(index)}")
        if self.running: # Se estiver rodando, atualiza a região dinamicamente
            region = WindowSelector.get_window_region(self.current_window_info)
            if region and self.capture_thread:
                self.capture_thread.set_region(region)

    def refresh_windows_list(self):
        self.window_combo.clear()
        try:
            windows = WindowSelector.get_available_windows(self.only_current_desktop)
            for w in windows:
                self.window_combo.addItem(w.get("name", "Desconhecido"), w)
            self.log(f"Lista de janelas atualizada ({len(windows)} encontradas).")
        except Exception as e:
            self.log_error(f"Erro ao listar janelas: {e}")

    def on_toggle_current_desktop(self, checked):
        self.only_current_desktop = checked
        self.refresh_windows_list()

    def select_custom_region(self):
        self.log("Selecione a região com o mouse...")
        self.region_selector = RegionSelector()
        self.region_selector.region_selected.connect(self.on_region_selected)
        self.region_selector.show()

    def on_region_selected(self, region):
        self.log(f"Região selecionada: {region['width']}x{region['height']} at ({region['left']},{region['top']})")
        region_info = {"custom_region": region, "type": "custom", "name": f"Região ({region['width']}x{region['height']})"}
        self.window_combo.addItem(region_info["name"], region_info)
        self.window_combo.setCurrentIndex(self.window_combo.count() - 1)
        self.current_window_info = region_info

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Modelo", "", "Modelos ONNX (*.onnx);;Todos os Arquivos (*)")
        if path:
            self.model_path_edit.setText(path)
            self.on_model_path_selected(path)

    def on_model_path_selected(self, path):
        if self.detection_thread:
            self.detection_thread.stop()
        self.detection_thread = DetectionThread(path)
        self.detection_thread.detection_ready.connect(self.update_display_overlay)
        # Re-aplica configurações atuais
        self.on_confidence_changed(self.confidence_spinbox.value())
        self.on_mouse_control_toggled(self.mouse_checkbox.isChecked())
        self.on_color_filter_toggled() # Atualiza com os valores atuais
        self.log(f"Novo modelo carregado: {os.path.basename(path)}")
        if self.running: # Se estiver rodando, reinicia a detecção
            self.detection_thread.start()
        
    def _wire_capture_signals(self, cap_thread):
        cap_thread.frame_ready.connect(self.detection_thread.set_frame)
        cap_thread.error_occurred.connect(self.on_capture_error)
        cap_thread.region_changed.connect(self.detection_thread.set_capture_region)
        cap_thread.region_changed.connect(self.overlay.set_fov_center_from_region)

    def _unwire_capture_signals(self, cap_thread):
        try: cap_thread.frame_ready.disconnect(self.detection_thread.set_frame)
        except Exception: pass
        try: cap_thread.error_occurred.disconnect(self.on_capture_error)
        except Exception: pass
        try: cap_thread.region_changed.disconnect(self.detection_thread.set_capture_region)
        except Exception: pass
        try: cap_thread.region_changed.disconnect(self.overlay.set_fov_center_from_region)
        except Exception: pass

    def toggle_detection(self):
        if not self.running:
            model_path = self.model_path_edit.text()
            if not model_path or not os.path.exists(model_path):
                self.log_error("Erro: Caminho do modelo inválido ou não definido.")
                QMessageBox.critical(self, "Erro", "Por favor, selecione um arquivo de modelo válido.")
                return
            if not self.current_window_info:
                self.log_error("Erro: Nenhuma fonte de captura selecionada.")
                QMessageBox.critical(self, "Erro", "Por favor, selecione uma janela, monitor ou região para capturar.")
                return

            region = WindowSelector.get_window_region(self.current_window_info)
            if not region:
                self.log_error("Não foi possível obter a região da fonte selecionada.")
                return

            self.running = True
            self.start_button.setText("Parar Detecção")
            self.statusBar().showMessage("Iniciando...")
            
            engine_key = self.engine_combo.currentData()
            res = (self.resize_w_spin.value(), self.resize_h_spin.value())
            
            if self.capture_thread: # Limpa a thread anterior se houver
                 self._unwire_capture_signals(self.capture_thread)
                 self.capture_thread.stop()

            if engine_key == 'ffmpeg':
                self.capture_thread = FFMpegCaptureThread(res, self.fps_limit)
            else:
                self.capture_thread = FastCaptureThread(res, self.fps_limit)

            self._wire_capture_signals(self.capture_thread)
            self.capture_thread.set_region(region)
            
            self.detection_thread.start()
            self.capture_thread.start()
            
            if self.overlay_enabled: self.overlay.show_all_screens()
            self.log("✅ Detecção iniciada.")

        else:
            self.running = False
            self.start_button.setText("Iniciar Detecção")
            self.statusBar().showMessage("Pronto")
            
            if self.capture_thread: self.capture_thread.stop()
            self.detection_thread.stop() # A thread de detecção também para
            self.overlay.hide()
            self.overlay.clear_boxes()
            self.image_label.clear() # Limpa a imagem de preview
            self.log("⏹️ Detecção parada.")

    def on_capture_error(self, error_msg):
        self.log_error(f"Erro na captura: {error_msg}")
        if self.running:
            self.toggle_detection() # Para a detecção

    def update_display_overlay(self, boxes, frame):
        if not self.running: return

        # Atualiza overlay
        if self.overlay_enabled:
            self.overlay.set_boxes(boxes)

        # Atualiza preview na UI
        try:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            # self.log_error(f"Erro ao exibir preview: {e}")
            pass

    def on_mouse_control_toggled(self, checked):
        self.detection_thread.set_mouse_control_enabled(checked)
        self.log(f"Mira automática {'ativada' if checked else 'desativada'}.")

    def on_pick_color(self):
        self.color_picker = ColorPickerOverlay()
        self.color_picker.color_picked.connect(self.on_color_picked)
        self.color_picker.start()

    def on_color_picked(self, hsv):
        self._color_h, self._color_s, self._color_v = hsv
        self.log(f"Cor selecionada (HSV): {hsv}")
        rgb = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0]
        self.color_preview.setStyleSheet(f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border-radius: 4px;")
        self.on_color_filter_toggled() # Aplica/atualiza o filtro

    def on_color_filter_toggled(self):
        enabled = self.color_filter_checkbox.isChecked()
        hsv_base = (self._color_h, self._color_s, self._color_v)
        tol_h = self.color_tol_h_spin.value()
        min_s = self.color_min_s_spin.value()
        min_v = self.color_min_v_spin.value()
        self.detection_thread.set_color_filter(enabled, hsv_base, tol_h, min_s, min_v)
        self.log(f"Filtro de cor {'ativado' if enabled else 'desativado'}.")

    def closeEvent(self, event):
        self.running = False
        if hasattr(self, 'capture_thread') and self.capture_thread: self.capture_thread.stop()
        if hasattr(self, 'detection_thread'): self.detection_thread.stop()
        if hasattr(self, 'overlay'): self.overlay.close()
        self.log("Aplicação encerrada.")
        event.accept()

    def apply_theme(self, mode: str = "dark"):
        if mode == "dark":
            ss = """
            QWidget { background-color: #0f1320; color: #e6e6e6; font-size: 13px; }
            QGroupBox { border: 1px solid #2b2f3a; border-radius: 8px; margin-top: 14px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #a0aec0; }
            QLabel { color: #d1d5db; background-color: transparent; }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { background-color: #141a2a; border: 1px solid #2b2f3a; border-radius: 6px; padding: 4px 6px; color: #e6e6e6; }
            QComboBox QAbstractItemView { background-color: #141a2a; color: #e6e6e6; selection-background-color: #2d6cdf; }
            QCheckBox { spacing: 6px; }
            QPushButton { background-color: #2d6cdf; color: white; border: none; padding: 8px 12px; border-radius: 6px; }
            QPushButton:hover { background-color: #2558b5; }
            QPushButton:pressed { background-color: #1f4a96; }
            QTabWidget::pane { border: 1px solid #2b2f3a; border-radius: 8px; }
            QTabBar::tab { background: #141a2a; border: 1px solid #2b2f3a; padding: 8px 10px; margin: 2px; border-radius: 6px; }
            QTabBar::tab:selected { background: #1b2236; color: #ffffff; border-color: #3b82f6; }
            QSlider::groove:horizontal { background: #2b2f3a; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #3b82f6; width: 14px; margin: -5px 0; border-radius: 7px; }
            QStatusBar { background: #0d111c; color: #9aa5b1; }
            QMenuBar { background: #0d111c; color: #e5e7eb; border-bottom: 1px solid #1f2937; }
            QMenuBar::item { background: transparent; padding: 6px 10px; }
            QMenuBar::item:selected { background: #1b2236; border-radius: 4px; }
            QMenu { background: #101425; color: #e5e7eb; border: 1px solid #2b2f3a; }
            QMenu::item { padding: 6px 16px; }
            QMenu::item:selected { background: #1f2a44; }
            QToolTip { color: #e5e7eb; background-color: #1b2236; border: 1px solid #2b2f3a; }
            """
        else: # light theme
            ss = "" # Deixado em branco, pode ser preenchido se necessário
        self.setStyleSheet(ss)

    def save_config_to_json(self): self.log("Função 'Salvar Config' não implementada neste exemplo.")
    def load_config_from_json(self): self.log("Função 'Carregar Config' não implementada neste exemplo.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = ScreenDetector()
    main_window.show()
    sys.exit(app.exec_())