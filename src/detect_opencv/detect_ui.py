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
    QGridLayout,
    QTabWidget, QAction, QMenu, QActionGroup, QToolButton, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QTextCursor
from PyQt5.QtGui import QPainter, QColor, QPen

import config as cfg
from detect_logic import (
    WindowSelector,
    FastCaptureThread,
    DetectionThread,
    HAS_EWMH,
    HAS_UINPUT,
)


class OverlayWindow(QWidget):
    """Janela transparente em tela inteira que desenha apenas os contornos detectados."""
    def __init__(self):
        super().__init__()
        # Janela sem borda, sempre no topo, não aparece na barra de tarefas
        flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        # Tornar totalmente transparente para entrada (quando suportado)
        try:
            if hasattr(Qt, 'WindowTransparentForInput'):
                flags |= Qt.WindowTransparentForInput
        except Exception:
            pass
        self.setWindowFlags(flags)
        # Transparente e não captura eventos do mouse (clique passa para baixo)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        # Não aceitar foco/ativação
        try:
            self.setWindowFlag(Qt.WindowDoesNotAcceptFocus, True)
        except Exception:
            pass
        self.setFocusPolicy(Qt.NoFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self._boxes = []  # lista de dicts: {x1,y1,x2,y2,label,conf}
        self._offset_x = 0
        self._offset_y = 0
        # FOV state
        try:
            import config as _cfg
            self._draw_fov = bool(getattr(_cfg, 'DRAW_FOV_DEFAULT', False))
            self._fov_radius = int(getattr(_cfg, 'FOV_RADIUS_DEFAULT', 150))
        except Exception:
            self._draw_fov = False
            self._fov_radius = 150
        self._fov_center = (0, 0)  # coordenadas de tela absolutas
        self._fov_offset_x = 0
        self._fov_offset_y = 0

    def show_all_screens(self):
        # Cobrir a área virtual combinada de todos os monitores
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
            # Fallback simples
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
        # Não desenhar fundo: manter totalmente transparente
        box_pen = QPen(QColor(80, 220, 100, 255))  # verde
        box_pen.setWidth(3)
        painter.setPen(box_pen)
        # Desenhar caixas
        for b in self._boxes:
            try:
                x1, y1, x2, y2 = int(b.get('x1', 0)), int(b.get('y1', 0)), int(b.get('x2', 0)), int(b.get('y2', 0))
                # aplicar offset
                x1 += self._offset_x
                x2 += self._offset_x
                y1 += self._offset_y
                y2 += self._offset_y
                painter.drawRect(x1, y1, max(0, x2 - x1), max(0, y2 - y1))
                # rótulo opcional
                label = b.get('label')
                conf = b.get('conf')
                if label is not None and conf is not None:
                    text = f"{label} {conf:.2f}"
                    # fundo sutil para leitura
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

        # Desenhar FOV circular
        if self._draw_fov and self._fov_radius > 0:
            try:
                # FOV não deve sofrer os offsets de desenho do overlay
                cx = int(self._fov_center[0]) + int(self._fov_offset_x)
                cy = int(self._fov_center[1]) + int(self._fov_offset_y)
                fov_pen = QPen(QColor(30, 144, 255, 220))  # azul
                fov_pen.setWidth(2)
                painter.setPen(fov_pen)
                painter.drawEllipse(cx - self._fov_radius, cy - self._fov_radius,
                                    self._fov_radius * 2, self._fov_radius * 2)
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
            left = int(region.get('left', 0))
            top = int(region.get('top', 0))
            width = int(region.get('width', 0))
            height = int(region.get('height', 0))
            cx = left + width // 2
            cy = top + height // 2
            self._fov_center = (cx, cy)
            self.update()
        except Exception:
            pass

    def set_fov_offset(self, dx: int, dy: int):
        try:
            self._fov_offset_x = int(dx)
            self._fov_offset_y = int(dy)
            self.update()
        except Exception:
            pass


class RegionSelector(QWidget):
    """Widget para selecionar uma região da tela"""
    region_selected = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self.start_pos = None
        self.end_pos = None
        self.selection_rect = None

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QColor, QPen
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
            end_pos = event.globalPos()
            self.selection_rect = QRect(
                self.mapFromGlobal(self.start_pos),
                self.mapFromGlobal(end_pos)
            ).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.start_pos:
            end_pos = event.globalPos()
            rect = QRect(self.start_pos, end_pos).normalized()
            if rect.width() > 10 and rect.height() > 10:
                region = {
                    "left": rect.x(),
                    "top": rect.y(),
                    "width": rect.width(),
                    "height": rect.height(),
                    "type": "custom"
                }
                self.region_selected.emit(region)
        self.hide()
        self.start_pos = None
        self.selection_rect = None


class ColorPickerOverlay(QWidget):
    """Overlay em tela cheia para escolher cor por clique.
    Captura screenshot do desktop virtual e lê a cor sob o cursor.
    """
    color_picked = pyqtSignal(tuple)  # (h, s, v)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self._shot = None  # ndarray BGR
        self._origin_left = 0
        self._origin_top = 0

    def start(self):
        try:
            import mss
            with mss.mss() as sct:
                m = sct.monitors[0]  # desktop virtual inteiro
                self._origin_left = int(m.get('left', 0))
                self._origin_top = int(m.get('top', 0))
                w = int(m.get('width', 1920))
                h = int(m.get('height', 1080))
                shot = sct.grab(m)
                arr = np.asarray(shot)
                if arr.ndim == 3 and arr.shape[2] == 4:
                    self._shot = arr[:, :, :3].copy()
                elif arr.ndim == 3 and arr.shape[2] == 3:
                    self._shot = arr.copy()
                else:
                    self._shot = None
                self.setGeometry(self._origin_left, self._origin_top, w, h)
                self.showFullScreen()
        except Exception:
            self._shot = None
            screen = QApplication.primaryScreen().geometry()
            self.setGeometry(screen)
            self.showFullScreen()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            gp = event.globalPos()
            x = gp.x() - self._origin_left
            y = gp.y() - self._origin_top
            try:
                if self._shot is not None and 0 <= y < self._shot.shape[0] and 0 <= x < self._shot.shape[1]:
                    b, g, r = [int(v) for v in self._shot[y, x]]
                    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
                    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
                    self.color_picked.emit((h, s, v))
            except Exception:
                pass
            self.hide()
        elif event.button() == Qt.RightButton:
            self.hide()

class ScreenDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV Screen Detector - Linux")
        self.setGeometry(100, 100, 900, 800)

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        cv2.setNumThreads(cfg.OPENCV_THREADS)
        cv2.ocl.setUseOpenCL(cfg.USE_OPENCL)

        # Nenhum modelo externo é necessário; detecção usa apenas OpenCV/NumPy
        model_path = ""

        self.running = False
        self.fps_limit = cfg.DEFAULT_FPS_LIMIT
        self.confidence = cfg.DEFAULT_CONFIDENCE
        self.pixel_step = getattr(cfg, 'PIXEL_STEP_DEFAULT', 1)
        self.infer_fps_limit = getattr(cfg, 'INFER_FPS_LIMIT_DEFAULT', 15)
        self.current_window_info = None
        self.only_current_desktop = cfg.ONLY_CURRENT_DESKTOP_DEFAULT
        self.overlay_enabled = True  # permite esconder/mostrar overlay sem parar a detecção

        # motor inicial será ajustado após UI; criar com MSS por padrão
        self.capture_thread = FastCaptureThread(cfg.DETECT_RESOLUTION, self.fps_limit)
        self.detection_thread = DetectionThread(model_path)
        # aplicar config inicial
        self.capture_thread.set_target_size(cfg.DETECT_RESOLUTION)
        self.capture_thread.set_fps_limit(self.fps_limit)

        self._wire_capture_signals(self.capture_thread)
        self.detection_thread.detection_ready.connect(self.update_display_overlay)

        # Parâmetros atuais do filtro humanoide (usados para preencher a UI)
        self.filter_min_area = float(getattr(self.detection_thread, 'filter_min_area', 900.0))
        self.filter_max_area = float(getattr(self.detection_thread, 'filter_max_area', 45000.0))
        self.filter_min_aspect = float(getattr(self.detection_thread, 'filter_min_aspect', 1.2))
        self.filter_max_aspect = float(getattr(self.detection_thread, 'filter_max_aspect', 6.2))
        self.filter_min_extent = float(getattr(self.detection_thread, 'filter_min_extent', 0.22))
        self.filter_min_solidity = float(getattr(self.detection_thread, 'filter_min_solidity', 0.28))
        self.filter_min_circularity = float(getattr(self.detection_thread, 'filter_min_circularity', 0.05))
        self.filter_max_circularity = float(getattr(self.detection_thread, 'filter_max_circularity', 0.55))
        self.filter_max_sym_diff = float(getattr(self.detection_thread, 'filter_max_sym_diff', 0.45))
        self.filter_min_top_ratio = float(getattr(self.detection_thread, 'filter_min_top_ratio', 0.18))
        self.filter_min_bottom_ratio = float(getattr(self.detection_thread, 'filter_min_bottom_ratio', 0.28))
        self.filter_conf_blend = float(getattr(self.detection_thread, 'filter_conf_blend', 0.6))
        self.filter_mask_mode = bool(getattr(self.detection_thread, 'use_preview_mask_detection', True))
        self._filter_param_widgets: List[QWidget] = []
        self._filter_last_debug: List[dict] = []

        self.region_selector = RegionSelector()
        self.region_selector.region_selected.connect(self.on_region_selected)

        # Overlay para escolha de cor
        self.color_picker = ColorPickerOverlay()
        self.color_picker.color_picked.connect(self.on_color_picked)
        self._color_h, self._color_s, self._color_v = 0, 0, 0

        # Menubar + Status bar + UI
        self._build_menubar()
        self.init_ui()
        self.statusBar().showMessage("Pronto")
        # Tema inicial
        try:
            self.apply_theme("dark")
        except Exception:
            pass

        # Aplica estado responsivo inicial
        try:
            self._apply_responsive_layout()
        except Exception:
            pass

        self.refresh_windows_list()
        # Selecionar a primeira opção por padrão para já definir a região de captura
        try:
            if self.window_combo.count() > 0:
                self.window_combo.setCurrentIndex(0)
        except Exception:
            pass
        # Aplicar motor default do config
        try:
            default_engine = getattr(cfg, 'CAPTURE_ENGINE_DEFAULT', 'mss').lower()
            for i in range(self.engine_combo.count()):
                if self.engine_combo.itemData(i) == default_engine:
                    self.engine_combo.setCurrentIndex(i)
                    break
        except Exception:
            pass

        # Estado do modo Train
        self._train_last_frame = None  # ndarray BGR
        self._train_last_region = None
        self._train_last_mask = None
        self._train_last_overlay = None

    def _wire_capture_signals(self, cap_thread):
        try:
            cap_thread.frame_ready.connect(self.detection_thread.set_frame)
            cap_thread.frame_ready.connect(self._on_preview_frame)
            cap_thread.error_occurred.connect(self.on_capture_error)
            cap_thread.region_changed.connect(self.detection_thread.set_capture_region)
            # Atualiza o centro do FOV na sobreposição quando região muda
            cap_thread.region_changed.connect(self.overlay.set_fov_center_from_region)
        except Exception:
            pass

    def _unwire_capture_signals(self, cap_thread):
        try:
            cap_thread.frame_ready.disconnect(self.detection_thread.set_frame)
        except Exception:
            pass
        try:
            cap_thread.error_occurred.disconnect(self.on_capture_error)
        except Exception:
            pass
        try:
            cap_thread.region_changed.disconnect(self.detection_thread.set_capture_region)
        except Exception:
            pass
        try:
            cap_thread.region_changed.disconnect(self.overlay.set_fov_center_from_region)
        except Exception:
            pass

    def apply_theme(self, mode: str = "dark"):
        """Aplica um tema global simples (dark/light) via stylesheet."""
        if mode == "dark":
            ss = """
            QWidget { background-color: #0f1320; color: #e6e6e6; font-size: 13px; }
            QGroupBox { border: 1px solid #2b2f3a; border-radius: 8px; margin-top: 14px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #a0aec0; }
            QLabel { color: #d1d5db; }
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
        else:
            ss = """
            QWidget { background: #f7f7fb; color: #1f2937; font-size: 13px; }
            QGroupBox { border: 1px solid #d7dbe7; border-radius: 8px; margin-top: 14px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #4b5563; }
            QLabel { color: #111827; }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { background-color: #ffffff; border: 1px solid #d7dbe7; border-radius: 6px; padding: 4px 6px; color: #111827; }
            QComboBox QAbstractItemView { background-color: #ffffff; color: #111827; selection-background-color: #e0e7ff; }
            QCheckBox { spacing: 6px; }
            QPushButton { background-color: #2563eb; color: white; border: none; padding: 8px 12px; border-radius: 6px; }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:pressed { background-color: #1e40af; }
            QTabWidget::pane { border: 1px solid #d7dbe7; border-radius: 8px; }
            QTabBar::tab { background: #ffffff; border: 1px solid #d7dbe7; padding: 8px 10px; margin: 2px; border-radius: 6px; }
            QTabBar::tab:selected { background: #eef2ff; color: #1f2937; border-color: #a5b4fc; }
            QSlider::groove:horizontal { background: #e5e7eb; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #2563eb; width: 14px; margin: -5px 0; border-radius: 7px; }
            QStatusBar { background: #f3f4f6; color: #4b5563; }
            QMenuBar { background: #ffffff; color: #1f2937; border-bottom: 1px solid #e5e7eb; }
            QMenuBar::item { background: transparent; padding: 6px 10px; }
            QMenuBar::item:selected { background: #eef2ff; border-radius: 4px; }
            QMenu { background: #ffffff; color: #111827; border: 1px solid #d7dbe7; }
            QMenu::item { padding: 6px 16px; }
            QMenu::item:selected { background: #e0e7ff; }
            QToolTip { color: #111827; background-color: #ffffff; border: 1px solid #d7dbe7; }
            """
        # Estiliza também o botão hambúrguer para o modo atual
        ss_extra_dark = """
            QToolButton#Hamburger { background-color: #111827; color: #e5e7eb; border: 1px solid #2b2f3a; padding: 8px 10px; border-radius: 8px; }
            QToolButton#Hamburger::menu-indicator { image: none; }
        """
        ss_extra_light = """
            QToolButton#Hamburger { background-color: #ffffff; color: #1f2937; border: 1px solid #d7dbe7; padding: 8px 10px; border-radius: 8px; }
            QToolButton#Hamburger::menu-indicator { image: none; }
        """
        self.setStyleSheet(ss + (ss_extra_dark if mode == "dark" else ss_extra_light))

    def _build_menubar(self):
        """Cria um menu reorganizado e agradável com modo compacto responsivo."""
        mb = self.menuBar()
        mb.clear()
        mb.setNativeMenuBar(False)

        # ===== Arquivo =====
        self.file_menu = mb.addMenu("Arquivo")
        self.act_save_cfg = QAction("Salvar configuração…", self)
        self.act_save_cfg.setShortcut("Ctrl+S")
        self.act_save_cfg.triggered.connect(self.save_config_to_json)
        self.file_menu.addAction(self.act_save_cfg)

        self.act_load_cfg = QAction("Carregar configuração…", self)
        self.act_load_cfg.setShortcut("Ctrl+L")
        self.act_load_cfg.triggered.connect(self.load_config_from_json)
        self.file_menu.addAction(self.act_load_cfg)

        self.file_menu.addSeparator()
        self.act_exit = QAction("Sair", self)
        self.act_exit.setShortcut("Ctrl+Q")
        self.act_exit.triggered.connect(self.close)
        self.file_menu.addAction(self.act_exit)

        # ===== Ferramentas =====
        self.tools_menu = mb.addMenu("Ferramentas")
        # Toggle de detecção também disponível no menu
        self.act_toggle_detection = QAction("Iniciar Detecção", self)
        self.act_toggle_detection.setShortcut("F5")
        self.act_toggle_detection.triggered.connect(self.toggle_detection)
        self.tools_menu.addAction(self.act_toggle_detection)

        self.act_bench = QAction("Benchmark", self)
        self.act_bench.setShortcut("Ctrl+B")
        self.act_bench.triggered.connect(self.on_benchmark_clicked)
        self.tools_menu.addAction(self.act_bench)

        # ===== Exibir =====
        self.view_menu = mb.addMenu("Exibir")
        theme_menu = self.view_menu.addMenu("Tema")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        self.act_theme_dark = QAction("Escuro", self, checkable=True)
        self.act_theme_light = QAction("Claro", self, checkable=True)
        theme_group.addAction(self.act_theme_dark)
        theme_group.addAction(self.act_theme_light)
        self.act_theme_dark.setChecked(True)
        self.act_theme_dark.triggered.connect(lambda: self.apply_theme("dark"))
        self.act_theme_light.triggered.connect(lambda: self.apply_theme("light"))
        theme_menu.addAction(self.act_theme_dark)
        theme_menu.addAction(self.act_theme_light)

        self.act_overlay = QAction("Mostrar overlay de detecção", self, checkable=True)
        self.act_overlay.setChecked(True)
        def _toggle_overlay_action(checked: bool):
            self.overlay_enabled = bool(checked)
            if self.running:
                if self.overlay_enabled:
                    try:
                        self.overlay.show_all_screens()
                    except Exception:
                        pass
                else:
                    try:
                        self.overlay.hide()
                    except Exception:
                        pass
        self.act_overlay.toggled.connect(_toggle_overlay_action)
        self.view_menu.addAction(self.act_overlay)

        # ===== Ajuda =====
        self.help_menu = mb.addMenu("Ajuda")
        self.act_about = QAction("Sobre…", self)
        def _about():
            QMessageBox.information(self, "Sobre", "OpenCV Screen Detector\nInterface redesenhada para melhor organização e estética.")
        self.act_about.triggered.connect(_about)
        self.help_menu.addAction(self.act_about)

        self.act_shortcuts = QAction("Atalhos…", self)
        def _shortcuts():
            QMessageBox.information(
                self,
                "Atalhos",
                "F5: Iniciar/Parar detecção\n"
                "Ctrl+B: Benchmark\n"
                "Ctrl+S: Salvar configuração\n"
                "Ctrl+L: Carregar configuração\n"
                "Ctrl+Q: Sair\n"
            )
        self.act_shortcuts.triggered.connect(_shortcuts)
        self.help_menu.addAction(self.act_shortcuts)

        # Menu compacto (hambúrguer) para telas estreitas
        self.compact_menu = QMenu("Menu", self)
        self._rebuild_compact_menu()

    def _update_status(self, text: str):
        try:
            self.statusBar().showMessage(text)
        except Exception:
            pass


    def _rebuild_compact_menu(self):
        """Reconstrói o menu hambúrguer agregando os principais itens de forma organizada."""
        if not hasattr(self, "compact_menu") or self.compact_menu is None:
            self.compact_menu = QMenu("Menu", self)
        m = self.compact_menu
        m.clear()
        m_file = m.addMenu("Arquivo")
        m_file.addAction(self.act_save_cfg)
        m_file.addAction(self.act_load_cfg)
        m_file.addSeparator()
        m_file.addAction(self.act_exit)

        m_tools = m.addMenu("Ferramentas")
        m_tools.addAction(self.act_toggle_detection)
        m_tools.addAction(self.act_bench)

        m_view = m.addMenu("Exibir")
        m_theme = m_view.addMenu("Tema")
        m_theme.addAction(self.act_theme_dark)
        m_theme.addAction(self.act_theme_light)
        m_view.addAction(self.act_overlay)

        m_help = m.addMenu("❓ Ajuda")
        m_help.addAction(self.act_shortcuts)
        m_help.addAction(self.act_about)

    def _apply_responsive_layout(self):
        """Mostra o menu clássico em telas largas e o hambúrguer quando a janela está estreita."""
        compact = self.width() < 900
        try:
            self.menuBar().setVisible(not compact)
        except Exception:
            pass
        if hasattr(self, "hamburger_btn") and self.hamburger_btn:
            self.hamburger_btn.setVisible(compact)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout()

    # ----- Config I/O -----
    def _collect_config_dict(self) -> dict:
        try:
            region = WindowSelector.get_window_region(self.current_window_info)
        except Exception:
            region = None
        try:
            engine = self.engine_combo.currentData()
        except Exception:
            engine = None
        cfg_dict = {
            "confidence": float(getattr(self, 'confidence', 0.25)),
            "fps_capture": int(getattr(self, 'fps_limit', cfg.DEFAULT_FPS_LIMIT)),
            "pixel_step": int(getattr(self, 'pixel_step', 1)),
            "infer_fps_limit": int(getattr(self, 'infer_fps_limit', 15)),
            "resize_w": int(self.resize_w_spin.value()) if hasattr(self, 'resize_w_spin') else cfg.DETECT_RESOLUTION[0],
            "resize_h": int(self.resize_h_spin.value()) if hasattr(self, 'resize_h_spin') else cfg.DETECT_RESOLUTION[1],
            "engine": engine,
            "only_current_desktop": bool(getattr(self, 'only_current_desktop', True)),
            "overlay_enabled": bool(getattr(self, 'overlay_enabled', True)),
            "overlay_offset_x": int(self.offset_x_spin.value()) if hasattr(self, 'offset_x_spin') else 0,
            "overlay_offset_y": int(self.offset_y_spin.value()) if hasattr(self, 'offset_y_spin') else 0,
            "fov_draw": bool(self.draw_fov_checkbox.isChecked()) if hasattr(self, 'draw_fov_checkbox') else False,
            "fov_radius": int(self.fov_radius_spin.value()) if hasattr(self, 'fov_radius_spin') else 150,
            "fov_offset_x": int(self.fov_offset_x_spin.value()) if hasattr(self, 'fov_offset_x_spin') else 0,
            "fov_offset_y": int(self.fov_offset_y_spin.value()) if hasattr(self, 'fov_offset_y_spin') else 0,
            "mouse_enabled": bool(self.mouse_checkbox.isChecked()) if hasattr(self, 'mouse_checkbox') else False,
            "aim_x_bias": int(self.aim_bias_x_spin.value()) if hasattr(self, 'aim_bias_x_spin') else 0,
            "aim_y_bias": int(self.aim_bias_y_spin.value()) if hasattr(self, 'aim_bias_y_spin') else 0,
            "axis_x_enabled": bool(self.axis_x_checkbox.isChecked()) if hasattr(self, 'axis_x_checkbox') else True,
            "axis_y_enabled": bool(self.axis_y_checkbox.isChecked()) if hasattr(self, 'axis_y_checkbox') else True,
            "mouse_gain": float(self.mouse_gain_spin.value()) if hasattr(self, 'mouse_gain_spin') else 0.6,
            "mouse_max_step": int(self.mouse_step_spin.value()) if hasattr(self, 'mouse_step_spin') else 15,

            "target_strategy_distance": bool(self.target_distance_checkbox.isChecked()) if hasattr(self, 'target_distance_checkbox') else True,
            "target_stick_radius_px": int(self.target_stick_radius_spin.value()) if hasattr(self, 'target_stick_radius_spin') else 120,

            # Filtro de cor (aba Geral)
            "color_filter_enabled": bool(self.color_filter_checkbox.isChecked()) if hasattr(self, 'color_filter_checkbox') else False,
            "color_h": int(getattr(self, '_color_h', 0)),
            "color_s": int(getattr(self, '_color_s', 0)),
            "color_v": int(getattr(self, '_color_v', 0)),
            "color_tol_h": int(self.color_tol_h_spin.value()) if hasattr(self, 'color_tol_h_spin') else 15,
            "color_min_s": int(self.color_min_s_spin.value()) if hasattr(self, 'color_min_s_spin') else 30,
            "color_min_v": int(self.color_min_v_spin.value()) if hasattr(self, 'color_min_v_spin') else 30,

            "selected_source": {
                "info": dict(self.current_window_info) if isinstance(self.current_window_info, dict) else None,
                "region": region,
            },
        }

        # ---- Train (HSV preview + morfologia) ----
        cfg_dict.update({
            # HSV de treino (independentes dos sliders da aba Geral)
            "train_h": int(self.train_h_spin.value()) if hasattr(self, 'train_h_spin') else int(getattr(self, '_color_h', 0)),
            "train_tol_h": int(self.train_tol_h_spin.value()) if hasattr(self, 'train_tol_h_spin') else int(getattr(cfg, 'COLOR_FILTER_TOL_H_DEFAULT', 15)),
            "train_smin": int(self.train_smin_spin.value()) if hasattr(self, 'train_smin_spin') else int(getattr(cfg, 'COLOR_FILTER_MIN_S_DEFAULT', 30)),
            "train_vmin": int(self.train_vmin_spin.value()) if hasattr(self, 'train_vmin_spin') else int(getattr(cfg, 'COLOR_FILTER_MIN_V_DEFAULT', 30)),

            # Morfologia
            "morph_edge_scale": float(self.train_edge_scale.value()) if hasattr(self, 'train_edge_scale') else float(getattr(cfg, 'MORPH_EDGE_SCALE_DEFAULT', 1.0)),
            "morph_kscale": float(self.train_kscale.value()) if hasattr(self, 'train_kscale') else float(getattr(cfg, 'MORPH_KSCALE_DEFAULT', 1.0)),
            "morph_dilate_iter": int(self.train_dil_iter.value()) if hasattr(self, 'train_dil_iter') else int(getattr(cfg, 'MORPH_DILATE_ITER_DEFAULT', 1)),
            "morph_close_iter": int(self.train_close_iter.value()) if hasattr(self, 'train_close_iter') else int(getattr(cfg, 'MORPH_CLOSE_ITER_DEFAULT', 1)),
            "morph_open_iter": int(self.train_open_iter.value()) if hasattr(self, 'train_open_iter') else int(getattr(cfg, 'MORPH_OPEN_ITER_DEFAULT', 1)),
        })

        # ---- Filtro humanoide ----
        cfg_dict.update({
            "filter_min_area": float(self.filter_min_area_spin.value()) if hasattr(self, 'filter_min_area_spin') else float(getattr(self, 'filter_min_area', 900.0)),
            "filter_max_area": float(self.filter_max_area_spin.value()) if hasattr(self, 'filter_max_area_spin') else float(getattr(self, 'filter_max_area', 45000.0)),
            "filter_min_aspect": float(self.filter_min_aspect_spin.value()) if hasattr(self, 'filter_min_aspect_spin') else float(getattr(self, 'filter_min_aspect', 1.2)),
            "filter_max_aspect": float(self.filter_max_aspect_spin.value()) if hasattr(self, 'filter_max_aspect_spin') else float(getattr(self, 'filter_max_aspect', 6.2)),
            "filter_min_extent": float(self.filter_min_extent_spin.value()) if hasattr(self, 'filter_min_extent_spin') else float(getattr(self, 'filter_min_extent', 0.22)),
            "filter_min_solidity": float(self.filter_min_solidity_spin.value()) if hasattr(self, 'filter_min_solidity_spin') else float(getattr(self, 'filter_min_solidity', 0.28)),
            "filter_min_circularity": float(self.filter_min_circularity_spin.value()) if hasattr(self, 'filter_min_circularity_spin') else float(getattr(self, 'filter_min_circularity', 0.05)),
            "filter_max_circularity": float(self.filter_max_circularity_spin.value()) if hasattr(self, 'filter_max_circularity_spin') else float(getattr(self, 'filter_max_circularity', 0.55)),
            "filter_max_sym_diff": float(self.filter_max_sym_spin.value()) if hasattr(self, 'filter_max_sym_spin') else float(getattr(self, 'filter_max_sym_diff', 0.45)),
            "filter_min_top_ratio": float(self.filter_min_top_spin.value()) if hasattr(self, 'filter_min_top_spin') else float(getattr(self, 'filter_min_top_ratio', 0.18)),
            "filter_min_bottom_ratio": float(self.filter_min_bottom_spin.value()) if hasattr(self, 'filter_min_bottom_spin') else float(getattr(self, 'filter_min_bottom_ratio', 0.28)),
            "filter_conf_blend": float(self.filter_conf_blend_spin.value()) if hasattr(self, 'filter_conf_blend_spin') else float(getattr(self, 'filter_conf_blend', 0.6)),
            "filter_mask_mode": bool(self.filter_mask_mode_checkbox.isChecked()) if hasattr(self, 'filter_mask_mode_checkbox') else bool(getattr(self, 'filter_mask_mode', True)),
        })

        return cfg_dict


    def save_config_to_json(self):
        from PyQt5.QtWidgets import QFileDialog
        try:
            default_dir = os.path.join(cfg.REPO_ROOT, 'runs', 'configs')
        except Exception:
            default_dir = os.path.join(os.getcwd(), 'runs', 'configs')
        os.makedirs(default_dir, exist_ok=True)
        default_name = time.strftime('detect_config_%Y%m%d-%H%M%S.json')
        path, _ = QFileDialog.getSaveFileName(self, "Salvar configuração", os.path.join(default_dir, default_name), "JSON (*.json)")
        if not path:
            return
        try:
            data = self._collect_config_dict()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._update_status(f"Configuração salva em: {path}")
            self.log(f"💾 Configuração salva: {path}")
        except Exception as e:
            self.log_error(f"Falha ao salvar configuração: {e}")

    def load_config_from_json(self):
        from PyQt5.QtWidgets import QFileDialog
        try:
            default_dir = os.path.join(cfg.REPO_ROOT, 'runs', 'configs')
        except Exception:
            default_dir = os.path.join(os.getcwd(), 'runs', 'configs')
        path, _ = QFileDialog.getOpenFileName(self, "Carregar configuração", default_dir, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.apply_config_dict(data)
            self._update_status(f"Configuração carregada de: {path}")
            self.log(f"📥 Configuração carregada: {path}")
        except Exception as e:
            self.log_error(f"Falha ao carregar configuração: {e}")

    def _set_engine(self, engine_key):
        if not engine_key:
            return
        try:
            for i in range(self.engine_combo.count()):
                if self.engine_combo.itemData(i) == engine_key:
                    self.engine_combo.setCurrentIndex(i)
                    break
        except Exception:
            pass

    def _select_window_from_info(self, info, region):
        try:
            # Garantir lista atualizada
            self.refresh_windows_list()
            if info is None:
                if self.window_combo.count() > 0:
                    self.window_combo.setCurrentIndex(0)
                return
            itype = info.get('type') if isinstance(info, dict) else None
            if itype == 'monitor':
                midx = int(info.get('monitor_index', -1))
                for i in range(self.window_combo.count()):
                    data = self.window_combo.itemData(i)
                    if isinstance(data, dict) and data.get('type') == 'monitor' and int(data.get('monitor_index', -2)) == midx:
                        self.window_combo.setCurrentIndex(i)
                        return
            elif itype == 'window':
                wid = int(info.get('window_id', -1))
                for i in range(self.window_combo.count()):
                    data = self.window_combo.itemData(i)
                    if isinstance(data, dict) and data.get('type') == 'window' and int(data.get('window_id', -2)) == wid:
                        self.window_combo.setCurrentIndex(i)
                        return
            if region and isinstance(region, dict):
                label = f"Região Carregada ({int(region.get('width', 0))}x{int(region.get('height', 0))})"
                region_info = {"custom_region": region, "type": "custom"}
                self.window_combo.addItem(label, region_info)
                self.window_combo.setCurrentIndex(self.window_combo.count() - 1)
        except Exception:
            pass

    def apply_config_dict(self, data: dict):
        # Confidence
        try:
            conf = float(data.get('confidence', self.confidence))
            conf_pct = int(max(1, min(99, round(conf * 100))))
            self.confidence_slider.setValue(conf_pct)
            self.confidence_spinbox.setValue(conf_pct)
        except Exception:
            pass

        # FPS captura
        try:
            fps = int(data.get('fps_capture', self.fps_limit))
            self.fps_spin.setValue(fps)
        except Exception:
            pass

        # Pixel step
        try:
            self.pixel_step = int(data.get('pixel_step', getattr(self, 'pixel_step', 1)))
            if hasattr(self, 'pixel_step_spin'):
                self.pixel_step_spin.setValue(self.pixel_step)
        except Exception:
            pass

        # FPS inferência
        try:
            self.infer_fps_limit = int(data.get('infer_fps_limit', getattr(self, 'infer_fps_limit', 15)))
            if hasattr(self, 'infer_fps_spin'):
                self.infer_fps_spin.setValue(self.infer_fps_limit)
        except Exception:
            pass

        # Tamanho de redimensionamento
        try:
            self.resize_w_spin.setValue(int(data.get('resize_w', self.resize_w_spin.value())))
        except Exception:
            pass
        try:
            self.resize_h_spin.setValue(int(data.get('resize_h', self.resize_h_spin.value())))
        except Exception:
            pass

        # Engine
        try:
            wanted = (data.get('engine') or '').lower()
            for i in range(self.engine_combo.count()):
                if (self.engine_combo.itemData(i) or '').lower() == wanted:
                    self.engine_combo.setCurrentIndex(i)
                    break
        except Exception:
            pass

        # Overlay on/off
        try:
            self.overlay_enabled = bool(data.get('overlay_enabled', self.overlay_enabled))
            self.overlay_checkbox.setChecked(self.overlay_enabled)
        except Exception:
            pass

        # Offset overlay
        try:
            self.offset_x_spin.setValue(int(data.get('overlay_offset_x', self.offset_x_spin.value())))
            self.offset_y_spin.setValue(int(data.get('overlay_offset_y', self.offset_y_spin.value())))
        except Exception:
            pass

        # FOV draw + params
        try:
            self.draw_fov_checkbox.setChecked(bool(data.get('fov_draw', self.draw_fov_checkbox.isChecked())))
        except Exception:
            pass
        try:
            self.fov_radius_spin.setValue(int(data.get('fov_radius', self.fov_radius_spin.value())))
        except Exception:
            pass
        try:
            self.fov_offset_x_spin.setValue(int(data.get('fov_offset_x', self.fov_offset_x_spin.value())))
            self.fov_offset_y_spin.setValue(int(data.get('fov_offset_y', self.fov_offset_y_spin.value())))
        except Exception:
            pass

        # Mouse settings
        try:
            self.mouse_checkbox.setChecked(bool(data.get('mouse_enabled', self.mouse_checkbox.isChecked())))
        except Exception:
            pass
        try:
            self.aim_bias_x_spin.setValue(int(data.get('aim_x_bias', self.aim_bias_x_spin.value())))
            self.aim_bias_y_spin.setValue(int(data.get('aim_y_bias', self.aim_bias_y_spin.value())))
        except Exception:
            pass
        try:
            self.axis_x_checkbox.setChecked(bool(data.get('axis_x_enabled', self.axis_x_checkbox.isChecked())))
            self.axis_y_checkbox.setChecked(bool(data.get('axis_y_enabled', self.axis_y_checkbox.isChecked())))
        except Exception:
            pass
        try:
            self.mouse_gain_spin.setValue(float(data.get('mouse_gain', self.mouse_gain_spin.value())))
            self.mouse_step_spin.setValue(int(data.get('mouse_max_step', self.mouse_step_spin.value())))
        except Exception:
            pass

        # Filtro de cor (aba Geral)
        try:
            self.color_filter_checkbox.setChecked(bool(data.get('color_filter_enabled', self.color_filter_checkbox.isChecked())))
            # sincroniza HSV “globais” também
            self._color_h = int(data.get('color_h', getattr(self, '_color_h', 0)))
            self._color_s = int(data.get('color_s', getattr(self, '_color_s', 0)))
            self._color_v = int(data.get('color_v', getattr(self, '_color_v', 0)))
            # atualizar prévia de cor
            try:
                rgb = cv2.cvtColor(np.uint8([[[self._color_h, self._color_s, self._color_v]]]), cv2.COLOR_HSV2BGR)[0, 0]
                b, g, r = int(rgb[0]), int(rgb[1]), int(rgb[2])
                self.color_preview.setStyleSheet(f"border: 1px solid #2b2f3a; background: rgb({r},{g},{b});")
            except Exception:
                pass
            # aplicar no detector ao vivo
            try:
                self.detection_thread.set_color_target_hsv(int(self._color_h), int(self._color_s), int(self._color_v))
            except Exception:
                pass
            # demais parâmetros de cor
            self.color_tol_h_spin.setValue(int(data.get('color_tol_h', self.color_tol_h_spin.value())))
            self.color_min_s_spin.setValue(int(data.get('color_min_s', self.color_min_s_spin.value())))
            self.color_min_v_spin.setValue(int(data.get('color_min_v', self.color_min_v_spin.value())))
            self.on_color_params_changed()
            self.on_color_filter_enabled_changed(self.color_filter_checkbox.isChecked())
        except Exception:
            pass

        # --- Filtro humanoide ---
        try:
            min_area = float(data.get('filter_min_area', getattr(self, 'filter_min_area', 900.0)))
            max_area = float(data.get('filter_max_area', getattr(self, 'filter_max_area', 45000.0)))
            min_aspect = float(data.get('filter_min_aspect', getattr(self, 'filter_min_aspect', 1.2)))
            max_aspect = float(data.get('filter_max_aspect', getattr(self, 'filter_max_aspect', 6.2)))
            min_extent = float(data.get('filter_min_extent', getattr(self, 'filter_min_extent', 0.22)))
            min_solidity = float(data.get('filter_min_solidity', getattr(self, 'filter_min_solidity', 0.28)))
            min_circularity = float(data.get('filter_min_circularity', getattr(self, 'filter_min_circularity', 0.05)))
            max_circularity = float(data.get('filter_max_circularity', getattr(self, 'filter_max_circularity', 0.55)))
            max_sym = float(data.get('filter_max_sym_diff', getattr(self, 'filter_max_sym_diff', 0.45)))
            min_top = float(data.get('filter_min_top_ratio', getattr(self, 'filter_min_top_ratio', 0.18)))
            min_bottom = float(data.get('filter_min_bottom_ratio', getattr(self, 'filter_min_bottom_ratio', 0.28)))
            conf_blend = float(data.get('filter_conf_blend', getattr(self, 'filter_conf_blend', 0.6)))
            mask_mode_val = bool(data.get('filter_mask_mode', getattr(self, 'filter_mask_mode', True)))

            self.filter_min_area = min_area
            self.filter_max_area = max_area
            self.filter_min_aspect = min_aspect
            self.filter_max_aspect = max_aspect
            self.filter_min_extent = min_extent
            self.filter_min_solidity = min_solidity
            self.filter_min_circularity = min_circularity
            self.filter_max_circularity = max_circularity
            self.filter_max_sym_diff = max_sym
            self.filter_min_top_ratio = min_top
            self.filter_min_bottom_ratio = min_bottom
            self.filter_conf_blend = conf_blend
            self.filter_mask_mode = mask_mode_val

            if hasattr(self, 'filter_min_area_spin'):
                self.filter_min_area_spin.blockSignals(True)
                self.filter_min_area_spin.setValue(int(min_area))
                self.filter_min_area_spin.blockSignals(False)
            if hasattr(self, 'filter_max_area_spin'):
                self.filter_max_area_spin.blockSignals(True)
                self.filter_max_area_spin.setValue(int(max_area))
                self.filter_max_area_spin.blockSignals(False)
                try:
                    self.filter_max_area_spin.setMinimum(int(self.filter_min_area_spin.value()) + 1)
                except Exception:
                    pass
            if hasattr(self, 'filter_min_aspect_spin'):
                self.filter_min_aspect_spin.blockSignals(True)
                self.filter_min_aspect_spin.setValue(float(min_aspect))
                self.filter_min_aspect_spin.blockSignals(False)
            if hasattr(self, 'filter_max_aspect_spin'):
                self.filter_max_aspect_spin.blockSignals(True)
                self.filter_max_aspect_spin.setValue(float(max_aspect))
                self.filter_max_aspect_spin.blockSignals(False)
            if hasattr(self, 'filter_min_extent_spin'):
                self.filter_min_extent_spin.blockSignals(True)
                self.filter_min_extent_spin.setValue(float(min_extent))
                self.filter_min_extent_spin.blockSignals(False)
            if hasattr(self, 'filter_min_solidity_spin'):
                self.filter_min_solidity_spin.blockSignals(True)
                self.filter_min_solidity_spin.setValue(float(min_solidity))
                self.filter_min_solidity_spin.blockSignals(False)
            if hasattr(self, 'filter_min_circularity_spin'):
                self.filter_min_circularity_spin.blockSignals(True)
                self.filter_min_circularity_spin.setValue(float(min_circularity))
                self.filter_min_circularity_spin.blockSignals(False)
            if hasattr(self, 'filter_max_circularity_spin'):
                self.filter_max_circularity_spin.blockSignals(True)
                self.filter_max_circularity_spin.setValue(float(max_circularity))
                self.filter_max_circularity_spin.blockSignals(False)
            if hasattr(self, 'filter_max_sym_spin'):
                self.filter_max_sym_spin.blockSignals(True)
                self.filter_max_sym_spin.setValue(float(max_sym))
                self.filter_max_sym_spin.blockSignals(False)
            if hasattr(self, 'filter_min_top_spin'):
                self.filter_min_top_spin.blockSignals(True)
                self.filter_min_top_spin.setValue(float(min_top))
                self.filter_min_top_spin.blockSignals(False)
            if hasattr(self, 'filter_min_bottom_spin'):
                self.filter_min_bottom_spin.blockSignals(True)
                self.filter_min_bottom_spin.setValue(float(min_bottom))
                self.filter_min_bottom_spin.blockSignals(False)
            if hasattr(self, 'filter_conf_blend_spin'):
                self.filter_conf_blend_spin.blockSignals(True)
                self.filter_conf_blend_spin.setValue(float(conf_blend))
                self.filter_conf_blend_spin.blockSignals(False)
            if hasattr(self, 'filter_mask_mode_checkbox'):
                self.filter_mask_mode_checkbox.blockSignals(True)
                self.filter_mask_mode_checkbox.setChecked(mask_mode_val)
                self.filter_mask_mode_checkbox.blockSignals(False)
            self.on_filter_mode_changed(mask_mode_val)
        except Exception:
            pass

        # --- Train (HSV/morfologia) ---
        try:
            # sliders/spins de treino (HSV)
            if hasattr(self, 'train_h_spin'):
                self.train_h_spin.setValue(int(data.get('train_h', self.train_h_spin.value())))
            if hasattr(self, 'train_tol_h_spin'):
                self.train_tol_h_spin.setValue(int(data.get('train_tol_h', self.train_tol_h_spin.value())))
            if hasattr(self, 'train_smin_spin'):
                self.train_smin_spin.setValue(int(data.get('train_smin', self.train_smin_spin.value())))
            if hasattr(self, 'train_vmin_spin'):
                self.train_vmin_spin.setValue(int(data.get('train_vmin', self.train_vmin_spin.value())))

            # morfologia (Train)
            if hasattr(self, 'train_edge_scale'):
                self.train_edge_scale.setValue(float(data.get('morph_edge_scale', self.train_edge_scale.value())))
            if hasattr(self, 'train_kscale'):
                self.train_kscale.setValue(float(data.get('morph_kscale', self.train_kscale.value())))
            if hasattr(self, 'train_dil_iter'):
                self.train_dil_iter.setValue(int(data.get('morph_dilate_iter', self.train_dil_iter.value())))
            if hasattr(self, 'train_close_iter'):
                self.train_close_iter.setValue(int(data.get('morph_close_iter', self.train_close_iter.value())))
            if hasattr(self, 'train_open_iter'):
                self.train_open_iter.setValue(int(data.get('morph_open_iter', self.train_open_iter.value())))

            # atualizar os internos que o preview usa

            # aplicar nos threads ao vivo
            try:
                self.detection_thread.set_morph_params(
                    data.get('morph_edge_scale'), data.get('morph_kscale'),
                    data.get('morph_dilate_iter'), data.get('morph_close_iter'),
                    data.get('morph_open_iter')
                )
            except Exception:
                pass
        except Exception:
            pass

        # Fonte selecionada (janela/região)
        try:
            sel = data.get('selected_source') or {}
            info = sel.get('info') if isinstance(sel, dict) else None
            region = sel.get('region') if isinstance(sel, dict) else None
            try:
                if region is None and isinstance(info, dict):
                    region = WindowSelector.get_window_region(info)
            except Exception:
                region = None
            self.current_window_info = info
            if isinstance(region, (list, tuple)) and len(region) == 4:
                self.region_selector.set_region(tuple(map(int, region)))
            else:
                self.region_selector.clear_region()
        except Exception:
            pass

        # Atualiza status
        try:
            self._update_status("Configuração aplicada.")
        except Exception:
            pass


    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        # ----- Header estilizado com ações principais -----
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 12, 12, 12)
        header.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #111827, stop:1 #0b1220);
                border: 1px solid #1f2937; border-radius: 10px;
            }
        """)

        title_box = QVBoxLayout()
        title = QLabel("OpenCV Screen Detector")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        subtitle = QLabel("Detecção em tempo real com visual refinado")
        subtitle.setStyleSheet("color: #9aa5b1;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        chips = QHBoxLayout()
        def _mk_chip(text: str, ok: bool):
            lb = QLabel(text)
            lb.setStyleSheet(
                "padding: 4px 8px; border-radius: 10px;" +
                ("background: rgba(16,185,129,0.18); color: #34d399; border: 1px solid #065f46;" if ok else
                 "background: rgba(239,68,68,0.18); color: #f87171; border: 1px solid #7f1d1d;")
            )
            return lb
        chips.addWidget(_mk_chip("EWMH", HAS_EWMH))
        chips.addWidget(_mk_chip("uinput", HAS_UINPUT))
        chips.addStretch(1)

        left_header = QVBoxLayout()
        left_header.addLayout(title_box)
        left_header.addLayout(chips)

        header_layout.addLayout(left_header, stretch=1)

        # Botões principais (migrados do rodapé)
        btns_box = QHBoxLayout()
        self.btn_toggle = QPushButton("Iniciar Detecção")
        self.btn_toggle.clicked.connect(self.toggle_detection)
        self.btn_toggle.setStyleSheet("""
            QPushButton { font-size: 14px; font-weight: bold; padding: 10px 14px; background-color: #10b981; color: white; border: none; border-radius: 8px; }
            QPushButton:hover { background-color: #059669; }
            QPushButton:pressed { background-color: #047857; }
        """)
        self.btn_benchmark = QPushButton("Benchmark")
        self.btn_benchmark.setToolTip("Executa um benchmark do detector com contagem regressiva de 5s e duração de 30s")
        self.btn_benchmark.clicked.connect(self.on_benchmark_clicked)
        self.btn_benchmark.setStyleSheet("""
            QPushButton { font-size: 14px; font-weight: bold; padding: 10px 14px; background-color: #2563eb; color: white; border: none; border-radius: 8px; }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:pressed { background-color: #1e40af; }
        """)
        btns_box.addWidget(self.btn_toggle)
        btns_box.addWidget(self.btn_benchmark)
        # Botão hambúrguer responsivo (menu compacto)
        self.hamburger_btn = QToolButton()
        self.hamburger_btn.setObjectName("Hamburger")
        self.hamburger_btn.setText("Menu")
        self.hamburger_btn.setToolTip("Menu compacto")
        self.hamburger_btn.setPopupMode(QToolButton.InstantPopup)
        try:
            self.hamburger_btn.setMenu(self.compact_menu)
        except Exception:
            pass
        self.hamburger_btn.setVisible(False)
        btns_box.addWidget(self.hamburger_btn)
        header_layout.addLayout(btns_box)

        layout.addWidget(header)

        # Tabs container
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.West)
        tabs.setMovable(False)

        # ----- Aba Captura -----
        captura_tab = QWidget()
        captura_tab_layout = QVBoxLayout()

        window_group = QGroupBox("Seleção de Captura")
        window_layout = QVBoxLayout()

        window_select_layout = QHBoxLayout()
        window_select_layout.addWidget(QLabel("🪟 Fonte:"))

        self.window_combo = QComboBox()
        self.window_combo.setMinimumWidth(300)
        self.window_combo.currentIndexChanged.connect(self.on_window_selected)
        window_select_layout.addWidget(self.window_combo)

        self.btn_refresh = QPushButton("🔄 Atualizar Lista")
        self.btn_refresh.clicked.connect(self.refresh_windows_list)
        self.btn_refresh.setMaximumWidth(160)
        window_select_layout.addWidget(self.btn_refresh)

        self.btn_select_region = QPushButton("📐 Selecionar Região")
        self.btn_select_region.clicked.connect(self.start_region_selection)
        self.btn_select_region.setMaximumWidth(160)
        window_select_layout.addWidget(self.btn_select_region)

        # Linha do motor de captura
        engine_layout = QHBoxLayout()
        engine_layout.addWidget(QLabel("⚙️ Motor:"))
        from PyQt5.QtWidgets import QComboBox as _QCB
        self.engine_combo = _QCB()
        self.engine_combo.addItem("MSS", "mss")
        self.engine_combo.addItem("Fastgrab", "fastgrab")
        # default do config
        try:
            default_engine = getattr(cfg, 'CAPTURE_ENGINE_DEFAULT', 'mss').lower()
        except Exception:
            default_engine = 'mss'
        idx = 0
        for i in range(self.engine_combo.count()):
            if self.engine_combo.itemData(i) == default_engine:
                idx = i
                break
        self.engine_combo.setCurrentIndex(idx)
        self.engine_combo.currentIndexChanged.connect(self.on_engine_changed)
        engine_layout.addWidget(self.engine_combo)

        window_layout.addLayout(window_select_layout)
        window_layout.addLayout(engine_layout)

        self.checkbox_current_desktop = QCheckBox("Mostrar apenas janelas do desktop atual")
        self.checkbox_current_desktop.setChecked(self.only_current_desktop)
        self.checkbox_current_desktop.stateChanged.connect(self.on_desktop_filter_changed)
        if not HAS_EWMH:
            self.checkbox_current_desktop.setEnabled(False)
            self.checkbox_current_desktop.setToolTip("Requer python-ewmh: pip install ewmh")
        window_layout.addWidget(self.checkbox_current_desktop)

        window_group.setLayout(window_layout)
        captura_tab_layout.addWidget(window_group)
        captura_tab.setLayout(captura_tab_layout)
        tabs.addTab(captura_tab, "🎛️ Captura")

        # Oculta a prévia de imagem; usaremos uma sobreposição transparente na tela inteira
        self.image_label = QLabel()
        self.image_label.setVisible(False)

        # ----- Aba Detecção (Configurações) -----
        configuracoes_tab = QWidget()
        configuracoes_layout = QVBoxLayout()

        settings_group = QGroupBox("Configurações")
        settings_layout = QVBoxLayout()
        # (Mouse moved to its own tab)

        # Linha: Confidence
        confidence_layout = QHBoxLayout()
        confidence_label = QLabel("Confidence:")
        confidence_label.setMinimumWidth(110)
        confidence_layout.addWidget(confidence_label)

        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(1)
        self.confidence_slider.setMaximum(99)
        self.confidence_slider.setValue(int(self.confidence * 100))
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        confidence_layout.addWidget(self.confidence_slider)

        self.confidence_spinbox = QSpinBox()
        self.confidence_spinbox.setMinimum(1)
        self.confidence_spinbox.setMaximum(99)
        self.confidence_spinbox.setValue(int(self.confidence * 100))
        self.confidence_spinbox.setSuffix("%")
        self.confidence_spinbox.valueChanged.connect(self.on_confidence_spinbox_changed)
        confidence_layout.addWidget(self.confidence_spinbox)

        settings_layout.addLayout(confidence_layout)

        self.confidence_value_label = QLabel(f"Valor atual: {self.confidence:.2f}")
        self.confidence_value_label.setAlignment(Qt.AlignCenter)
        self.confidence_value_label.setStyleSheet("font-weight: bold;")
        settings_layout.addWidget(self.confidence_value_label)

        # Linha: FPS
        fps_layout = QHBoxLayout()
        fps_label = QLabel("FPS (captura):")
        fps_label.setMinimumWidth(110)
        fps_layout.addWidget(fps_label)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(self.fps_limit)
        self.fps_spin.valueChanged.connect(self.on_fps_changed)
        fps_layout.addWidget(self.fps_spin)

        settings_layout.addLayout(fps_layout)

        # Linha: Passo de Pixels (amostragem)
        step_layout = QHBoxLayout()
        step_label = QLabel("Passo de pixels:")
        step_label.setMinimumWidth(110)
        step_layout.addWidget(step_label)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 16)
        self.step_spin.setValue(self.pixel_step)
        self.step_spin.setToolTip("Amostra 1 a cada N pixels para a inferência. 1 = todos.")
        self.step_spin.valueChanged.connect(self.on_pixel_step_changed)
        step_layout.addWidget(self.step_spin)

        settings_layout.addLayout(step_layout)

        # Linha: FPS de inferência (limite)
        infer_fps_layout = QHBoxLayout()
        infer_fps_label = QLabel("FPS (inferência):")
        infer_fps_label.setMinimumWidth(110)
        infer_fps_layout.addWidget(infer_fps_label)

        self.infer_fps_spin = QSpinBox()
        self.infer_fps_spin.setRange(1, 240)
        self.infer_fps_spin.setValue(self.infer_fps_limit)
        self.infer_fps_spin.setToolTip("Limita a taxa de inferência para reduzir uso de CPU.")
        self.infer_fps_spin.valueChanged.connect(self.on_infer_fps_changed)
        infer_fps_layout.addWidget(self.infer_fps_spin)

        settings_layout.addLayout(infer_fps_layout)

        # Linha: Resolução de redimensionamento
        resize_layout = QHBoxLayout()
        resize_label = QLabel("Resolução (WxH):")
        resize_label.setMinimumWidth(110)
        resize_layout.addWidget(resize_label)

        self.resize_w_spin = QSpinBox()
        self.resize_w_spin.setRange(16, 4096)
        self.resize_w_spin.setValue(cfg.DETECT_RESOLUTION[0])
        self.resize_w_spin.valueChanged.connect(self.on_resize_changed)
        resize_layout.addWidget(self.resize_w_spin)

        x_label = QLabel("x")
        resize_layout.addWidget(x_label)

        self.resize_h_spin = QSpinBox()
        self.resize_h_spin.setRange(16, 4096)
        self.resize_h_spin.setValue(cfg.DETECT_RESOLUTION[1])
        self.resize_h_spin.valueChanged.connect(self.on_resize_changed)
        resize_layout.addWidget(self.resize_h_spin)

        settings_layout.addLayout(resize_layout)

        # (Sem modelo externo — detecção usa apenas OpenCV/NumPy)

        # Filtro por Cor (HSV)
        color_group = QGroupBox("Filtro por Cor (HSV)")
        color_layout = QVBoxLayout()

        color_top = QHBoxLayout()
        self.color_filter_checkbox = QCheckBox("Habilitar filtro de cor")
        try:
            self.color_filter_checkbox.setChecked(bool(getattr(cfg, 'COLOR_FILTER_ENABLED_DEFAULT', False)))
        except Exception:
            self.color_filter_checkbox.setChecked(False)
        self.color_filter_checkbox.stateChanged.connect(lambda s: self.on_color_filter_enabled_changed(s == Qt.Checked))
        color_top.addWidget(self.color_filter_checkbox)

        self.color_preview = QLabel("  ")
        self.color_preview.setFixedSize(32, 18)
        self.color_preview.setStyleSheet("border: 1px solid #2b2f3a; background: #000;")
        color_top.addWidget(QLabel("Cor:"))
        color_top.addWidget(self.color_preview)

        self.btn_pick_color = QPushButton("Escolher Cor")
        self.btn_pick_color.clicked.connect(self.start_color_picker)
        color_top.addWidget(self.btn_pick_color)
        color_top.addStretch(1)
        color_layout.addLayout(color_top)

        color_params = QHBoxLayout()
        color_params.addWidget(QLabel("Tol. H:"))
        self.color_tol_h_spin = QSpinBox()
        self.color_tol_h_spin.setRange(0, 90)
        try:
            self.color_tol_h_spin.setValue(int(getattr(cfg, 'COLOR_FILTER_TOL_H_DEFAULT', 15)))
        except Exception:
            self.color_tol_h_spin.setValue(15)
        self.color_tol_h_spin.valueChanged.connect(lambda v: self.on_color_params_changed())
        color_params.addWidget(self.color_tol_h_spin)

        color_params.addWidget(QLabel("S mín:"))
        self.color_min_s_spin = QSpinBox()
        self.color_min_s_spin.setRange(0, 255)
        try:
            self.color_min_s_spin.setValue(int(getattr(cfg, 'COLOR_FILTER_MIN_S_DEFAULT', 30)))
        except Exception:
            self.color_min_s_spin.setValue(30)
        self.color_min_s_spin.valueChanged.connect(lambda v: self.on_color_params_changed())
        color_params.addWidget(self.color_min_s_spin)

        color_params.addWidget(QLabel("V mín:"))
        self.color_min_v_spin = QSpinBox()
        self.color_min_v_spin.setRange(0, 255)
        try:
            self.color_min_v_spin.setValue(int(getattr(cfg, 'COLOR_FILTER_MIN_V_DEFAULT', 30)))
        except Exception:
            self.color_min_v_spin.setValue(30)
        self.color_min_v_spin.valueChanged.connect(lambda v: self.on_color_params_changed())
        color_params.addWidget(self.color_min_v_spin)

        color_layout.addLayout(color_params)
        color_group.setLayout(color_layout)
        settings_layout.addWidget(color_group)

        settings_group.setLayout(settings_layout)
        configuracoes_layout.addWidget(settings_group)

        configuracoes_tab.setLayout(configuracoes_layout)
        tabs.addTab(configuracoes_tab, "🧠 Detecção")

        # ----- Aba Overlay -----
        overlay_tab = QWidget()
        overlay_tab_layout = QVBoxLayout()
        overlay_group = QGroupBox("Overlay")
        overlay_group_layout = QVBoxLayout()

        # Offsets de desenho
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Offset X:"))
        self.offset_x_spin = QSpinBox()
        self.offset_x_spin.setRange(-4096, 4096)
        self.offset_x_spin.setValue(0)
        self.offset_x_spin.valueChanged.connect(self.on_overlay_offset_changed)
        offset_layout.addWidget(self.offset_x_spin)

        offset_layout.addWidget(QLabel("Offset Y:"))
        self.offset_y_spin = QSpinBox()
        self.offset_y_spin.setRange(-4096, 4096)
        self.offset_y_spin.setValue(0)
        self.offset_y_spin.valueChanged.connect(self.on_overlay_offset_changed)
        offset_layout.addWidget(self.offset_y_spin)

        overlay_group_layout.addLayout(offset_layout)
        overlay_group.setLayout(overlay_group_layout)
        overlay_tab_layout.addWidget(overlay_group)
        overlay_tab.setLayout(overlay_tab_layout)
        tabs.addTab(overlay_tab, "🎯 Overlay")

        # ----- Aba Train -----
        train_tab = QWidget()
        train_layout = QVBoxLayout()

        # Label de orientação
        info_lbl = QLabel("Aba de configuração avançada. As alterações afetam a detecção em tempo real. Não há salvamento de arquivos.")
        train_layout.addWidget(info_lbl)

        # Pré-visualização
        preview_group = QGroupBox("Pré-visualização (como o OpenCV enxerga)")
        preview_layout = QHBoxLayout()
        self.train_mask_label = QLabel("Mask")
        self.train_mask_label.setMinimumSize(320, 180)
        self.train_mask_label.setStyleSheet("border: 1px solid #2b2f3a; background: #000;")
        preview_layout.addWidget(self.train_mask_label, 1)
        self.train_overlay_label = QLabel("Overlay")
        self.train_overlay_label.setMinimumSize(320, 180)
        self.train_overlay_label.setStyleSheet("border: 1px solid #2b2f3a; background: #000;")
        preview_layout.addWidget(self.train_overlay_label, 1)
        preview_group.setLayout(preview_layout)
        train_layout.addWidget(preview_group)

        # Ajustes de Segmentação (HSV + Morfologia)
        adjust_group = QGroupBox("Segmentação (HSV + Morfologia)")
        adjust_layout = QVBoxLayout()

        # Linha: HSV
        hsv_layout = QHBoxLayout()
        hsv_layout.addWidget(QLabel("H:"))
        self.train_h_spin = QSpinBox(); self.train_h_spin.setRange(0, 1000)
        self.train_h_spin.setValue(getattr(self, '_color_h', 0))
        self.train_h_spin.valueChanged.connect(self.on_train_params_changed)
        hsv_layout.addWidget(self.train_h_spin)

        hsv_layout.addWidget(QLabel("Tol H:"))
        self.train_tol_h_spin = QSpinBox(); self.train_tol_h_spin.setRange(0, 90)
        self.train_tol_h_spin.setValue(int(getattr(cfg, 'COLOR_FILTER_TOL_H_DEFAULT', 15)))
        self.train_tol_h_spin.valueChanged.connect(self.on_train_params_changed)
        hsv_layout.addWidget(self.train_tol_h_spin)

        hsv_layout.addWidget(QLabel("S mín:"))
        self.train_smin_spin = QSpinBox(); self.train_smin_spin.setRange(0, 255)
        self.train_smin_spin.setValue(int(getattr(cfg, 'COLOR_FILTER_MIN_S_DEFAULT', 30)))
        self.train_smin_spin.valueChanged.connect(self.on_train_params_changed)
        hsv_layout.addWidget(self.train_smin_spin)

        hsv_layout.addWidget(QLabel("V mín:"))
        self.train_vmin_spin = QSpinBox(); self.train_vmin_spin.setRange(0, 255)
        self.train_vmin_spin.setValue(int(getattr(cfg, 'COLOR_FILTER_MIN_V_DEFAULT', 30)))
        self.train_vmin_spin.valueChanged.connect(self.on_train_params_changed)
        hsv_layout.addWidget(self.train_vmin_spin)
        adjust_layout.addLayout(hsv_layout)

        # (sem modo distância — apenas hue ±Tol)

        # Linha: Morfologia
        morph_layout = QHBoxLayout()
        morph_layout.addWidget(QLabel("Dilatação (iter):"))
        self.train_dil_iter = QSpinBox(); self.train_dil_iter.setRange(0, 5)
        self.train_dil_iter.setValue(int(getattr(cfg, 'MORPH_DILATE_ITER_DEFAULT', 1)))
        self.train_dil_iter.valueChanged.connect(self.on_train_params_changed)
        morph_layout.addWidget(self.train_dil_iter)

        morph_layout.addWidget(QLabel("Fechamento (iter):"))
        self.train_close_iter = QSpinBox(); self.train_close_iter.setRange(0, 5)
        self.train_close_iter.setValue(int(getattr(cfg, 'MORPH_CLOSE_ITER_DEFAULT', 1)))
        self.train_close_iter.valueChanged.connect(self.on_train_params_changed)
        morph_layout.addWidget(self.train_close_iter)

        morph_layout.addWidget(QLabel("Abertura (iter):"))
        self.train_open_iter = QSpinBox(); self.train_open_iter.setRange(0, 5)
        self.train_open_iter.setValue(int(getattr(cfg, 'MORPH_OPEN_ITER_DEFAULT', 1)))
        self.train_open_iter.valueChanged.connect(self.on_train_params_changed)
        morph_layout.addWidget(self.train_open_iter)

        morph_layout.addWidget(QLabel("Edge scale:"))
        self.train_edge_scale = QDoubleSpinBox(); self.train_edge_scale.setRange(0.2, 3.0); self.train_edge_scale.setSingleStep(0.1)
        self.train_edge_scale.setValue(float(getattr(cfg, 'MORPH_EDGE_SCALE_DEFAULT', 1.0)))
        self.train_edge_scale.valueChanged.connect(self.on_train_params_changed)
        morph_layout.addWidget(self.train_edge_scale)

        morph_layout.addWidget(QLabel("Kernel scale:"))
        self.train_kscale = QDoubleSpinBox(); self.train_kscale.setRange(0.2, 3.0); self.train_kscale.setSingleStep(0.1)
        self.train_kscale.setValue(float(getattr(cfg, 'MORPH_KSCALE_DEFAULT', 1.0)))
        self.train_kscale.valueChanged.connect(self.on_train_params_changed)
        morph_layout.addWidget(self.train_kscale)

        adjust_layout.addLayout(morph_layout)
        adjust_group.setLayout(adjust_layout)
        train_layout.addWidget(adjust_group)

        train_tab.setLayout(train_layout)
        tabs.addTab(train_tab, "🧪 Train")

        # ----- Aba Filtro -----
        filtro_tab = QWidget()
        filtro_layout = QVBoxLayout()

        # Modo de operação (máscara simples x filtro avançado)
        filtro_mode_layout = QHBoxLayout()
        self.filter_mask_mode_checkbox = QCheckBox("Usar detecção simplificada (máscara integral)")
        self.filter_mask_mode_checkbox.setToolTip("Quando habilitado, a detecção usa apenas a máscara HSV sem os filtros humanoides.")
        self.filter_mask_mode_checkbox.setChecked(self.filter_mask_mode)
        self.filter_mask_mode_checkbox.stateChanged.connect(lambda s: self.on_filter_mode_changed(s == Qt.Checked))
        filtro_mode_layout.addWidget(self.filter_mask_mode_checkbox)
        filtro_mode_layout.addStretch(1)
        filtro_layout.addLayout(filtro_mode_layout)

        filtro_preview_group = QGroupBox("Pré-visualização do filtro humanoide")
        filtro_preview_layout = QHBoxLayout()
        self.filter_mask_label = QLabel("Mask")
        self.filter_mask_label.setMinimumSize(320, 180)
        self.filter_mask_label.setStyleSheet("border: 1px solid #2b2f3a; background: #000;")
        filtro_preview_layout.addWidget(self.filter_mask_label, 1)
        self.filter_overlay_label = QLabel("Overlay")
        self.filter_overlay_label.setMinimumSize(320, 180)
        self.filter_overlay_label.setStyleSheet("border: 1px solid #2b2f3a; background: #000;")
        filtro_preview_layout.addWidget(self.filter_overlay_label, 1)
        filtro_preview_group.setLayout(filtro_preview_layout)
        filtro_layout.addWidget(filtro_preview_group)

        filtro_params_group = QGroupBox("Parâmetros do filtro")
        filtro_params_grid = QGridLayout()
        row = 0

        def _add_spin(label_text: str, widget):
            nonlocal row
            filtro_params_grid.addWidget(QLabel(label_text), row, 0)
            filtro_params_grid.addWidget(widget, row, 1)
            self._filter_param_widgets.append(widget)
            row += 1

        self.filter_min_area_spin = QSpinBox()
        self.filter_min_area_spin.setRange(10, 200000)
        self.filter_min_area_spin.setValue(int(self.filter_min_area))
        self.filter_min_area_spin.valueChanged.connect(lambda _v: self.on_filter_params_changed())
        _add_spin("Área mín (px²):", self.filter_min_area_spin)

        self.filter_max_area_spin = QSpinBox()
        self.filter_max_area_spin.setRange(100, 400000)
        self.filter_max_area_spin.setValue(int(self.filter_max_area))
        self.filter_max_area_spin.valueChanged.connect(lambda _v: self.on_filter_params_changed())
        _add_spin("Área máx (px²):", self.filter_max_area_spin)

        self.filter_min_aspect_spin = QDoubleSpinBox()
        self.filter_min_aspect_spin.setDecimals(2)
        self.filter_min_aspect_spin.setRange(0.2, 10.0)
        self.filter_min_aspect_spin.setSingleStep(0.05)
        self.filter_min_aspect_spin.setValue(float(self.filter_min_aspect))
        self.filter_min_aspect_spin.valueChanged.connect(lambda _v: self.on_filter_params_changed())
        _add_spin("Aspecto mín (h/l):", self.filter_min_aspect_spin)

        self.filter_max_aspect_spin = QDoubleSpinBox()
        self.filter_max_aspect_spin.setDecimals(2)
        self.filter_max_aspect_spin.setRange(0.3, 12.0)
        self.filter_max_aspect_spin.setSingleStep(0.05)
        self.filter_max_aspect_spin.setValue(float(self.filter_max_aspect))
        self.filter_max_aspect_spin.valueChanged.connect(lambda _v: self.on_filter_params_changed())
        _add_spin("Aspecto máx (h/l):", self.filter_max_aspect_spin)

        self.filter_min_extent_spin = QDoubleSpinBox()
        self.filter_min_extent_spin.setDecimals(2)
        self.filter_min_extent_spin.setRange(0.0, 1.0)
        self.filter_min_extent_spin.setSingleStep(0.01)
        self.filter_min_extent_spin.setValue(float(self.filter_min_extent))
        self.filter_min_extent_spin.valueChanged.connect(lambda _v: self.on_filter_params_changed())
        _add_spin("Extensão mín:", self.filter_min_extent_spin)

        self.filter_min_solidity_spin = QDoubleSpinBox()
        self.filter_min_solidity_spin.setDecimals(2)
        self.filter_min_solidity_spin.setRange(0.0, 1.0)
        self.filter_min_solidity_spin.setSingleStep(0.01)
        self.filter_min_solidity_spin.setValue(float(self.filter_min_solidity))
        self.filter_min_solidity_spin.valueChanged.connect(lambda _v: self.on_filter_params_changed())
        _add_spin("Solidez mín:", self.filter_min_solidity_spin)

        self.filter_min_circularity_spin = QDoubleSpinBox()
        self.filter_min_circularity_spin.setDecimals(2)
        self.filter_min_circularity_spin.setRange(0.0, 1.0)
        self.filter_min_circularity_spin.setSingleStep(0.01)
        self.filter_min_circularity_spin.setValue(float(self.filter_min_circularity))
        self.filter_min_circularity_spin.valueChanged.connect(lambda _v: self.on_filter_params_changed())
        _add_spin("Circularidade mín:", self.filter_min_circularity_spin)

        self.filter_max_circularity_spin = QDoubleSpinBox()
        self.filter_max_circularity_spin.setDecimals(2)
        self.filter_max_circularity_spin.setRange(0.0, 1.0)
        self.filter_max_circularity_spin.setSingleStep(0.01)
        self.filter_max_circularity_spin.setValue(float(self.filter_max_circularity))
        self.filter_max_circularity_spin.valueChanged.connect(lambda _v: self.on_filter_params_changed())
        _add_spin("Circularidade máx:", self.filter_max_circularity_spin)

        self.filter_max_sym_spin = QDoubleSpinBox()
        self.filter_max_sym_spin.setDecimals(2)
        self.filter_max_sym_spin.setRange(0.0, 1.0)
        self.filter_max_sym_spin.setSingleStep(0.01)
        self.filter_max_sym_spin.setValue(float(self.filter_max_sym_diff))
        self.filter_max_sym_spin.valueChanged.connect(lambda _v: self.on_filter_params_changed())
        _add_spin("Dif. simetria máx:", self.filter_max_sym_spin)

        self.filter_min_top_spin = QDoubleSpinBox()
        self.filter_min_top_spin.setDecimals(2)
        self.filter_min_top_spin.setRange(0.0, 1.0)
        self.filter_min_top_spin.setSingleStep(0.01)
        self.filter_min_top_spin.setValue(float(self.filter_min_top_ratio))
        self.filter_min_top_spin.valueChanged.connect(lambda _v: self.on_filter_params_changed())
        _add_spin("Proporção topo mín:", self.filter_min_top_spin)

        self.filter_min_bottom_spin = QDoubleSpinBox()
        self.filter_min_bottom_spin.setDecimals(2)
        self.filter_min_bottom_spin.setRange(0.0, 1.0)
        self.filter_min_bottom_spin.setSingleStep(0.01)
        self.filter_min_bottom_spin.setValue(float(self.filter_min_bottom_ratio))
        self.filter_min_bottom_spin.valueChanged.connect(lambda _v: self.on_filter_params_changed())
        _add_spin("Proporção base mín:", self.filter_min_bottom_spin)

        self.filter_conf_blend_spin = QDoubleSpinBox()
        self.filter_conf_blend_spin.setDecimals(2)
        self.filter_conf_blend_spin.setRange(0.0, 1.0)
        self.filter_conf_blend_spin.setSingleStep(0.05)
        self.filter_conf_blend_spin.setValue(float(self.filter_conf_blend))
        self.filter_conf_blend_spin.valueChanged.connect(lambda _v: self.on_filter_params_changed())
        _add_spin("Mistura confiança:", self.filter_conf_blend_spin)

        filtro_params_group.setLayout(filtro_params_grid)
        filtro_layout.addWidget(filtro_params_group)

        filtro_debug_group = QGroupBox("Diagnóstico do último frame")
        filtro_debug_layout = QVBoxLayout()
        self.filter_debug_text = QTextEdit()
        self.filter_debug_text.setReadOnly(True)
        self.filter_debug_text.setMaximumHeight(140)
        filtro_debug_layout.addWidget(self.filter_debug_text)
        filtro_debug_group.setLayout(filtro_debug_layout)
        filtro_layout.addWidget(filtro_debug_group)

        filtro_tab.setLayout(filtro_layout)
        tabs.addTab(filtro_tab, "🧰 Filtro")

        # Ajustar estado inicial do filtro (habilita/desabilita widgets e aplica parâmetros)
        self.on_filter_mode_changed(self.filter_mask_mode)
        if not self.filter_mask_mode:
            self.on_filter_params_changed()
        self._update_filter_debug_text()

        # Timer para atualizar a prévia periodicamente
        self.train_preview_timer = QTimer(self)
        self.train_preview_timer.setInterval(200)
        self.train_preview_timer.timeout.connect(self.on_train_preview_tick)
        self.train_preview_timer.start()

        # ----- Aba Mouse -----
        mouse_tab = QWidget()
        mouse_tab_layout = QVBoxLayout()

        mouse_group = QGroupBox("Mouse")
        mouse_group_layout = QVBoxLayout()

        # Seleção de alvo
        target_group = QGroupBox("Seleção de Alvo")
        target_layout = QVBoxLayout()

        self.target_distance_checkbox = QCheckBox("Calcular Distância (mais próximo do centro)")
        try:
            self.target_distance_checkbox.setChecked(bool(getattr(cfg, 'TARGET_STRATEGY_DISTANCE_DEFAULT', True)))
        except Exception:
            self.target_distance_checkbox.setChecked(True)
        def _on_target_strategy_changed():
            try:
                self.detection_thread.set_target_strategy_distance(self.target_distance_checkbox.isChecked())
            except Exception:
                pass
        self.target_distance_checkbox.stateChanged.connect(lambda _s: _on_target_strategy_changed())
        target_layout.addWidget(self.target_distance_checkbox)

        stick_layout = QHBoxLayout()
        stick_layout.addWidget(QLabel("Estabilização (px):"))
        self.target_stick_radius_spin = QSpinBox()
        self.target_stick_radius_spin.setRange(0, 4000)
        try:
            self.target_stick_radius_spin.setValue(int(getattr(cfg, 'TARGET_STICK_RADIUS_DEFAULT', 120)))
        except Exception:
            self.target_stick_radius_spin.setValue(120)
        def _on_stick_changed(v: int):
            try:
                self.detection_thread.set_target_stick_radius(int(v))
            except Exception:
                pass
        self.target_stick_radius_spin.valueChanged.connect(_on_stick_changed)
        stick_layout.addWidget(self.target_stick_radius_spin)
        target_layout.addLayout(stick_layout)

        target_group.setLayout(target_layout)
        mouse_group_layout.addWidget(target_group)

        # (Sem grupo extra de estabilidade nesta versão)

        # Habilitar mouse (Kernel/uinput)
        mouse_enable_layout = QHBoxLayout()
        self.mouse_checkbox = QCheckBox("Habilitar Mouse (Kernel)")
        self.mouse_checkbox.setToolTip(
            "Mover mouse automaticamente para o alvo detectado usando o kernel Linux (uinput).\n"
            "Requer python-uinput e acesso de escrita ao dispositivo /dev/uinput."
        )
        self.mouse_checkbox.stateChanged.connect(self.on_mouse_checkbox_changed)
        self.mouse_checkbox.setEnabled(HAS_UINPUT)
        if not HAS_UINPUT:
            self.mouse_checkbox.setToolTip(self.mouse_checkbox.toolTip() + "\n⚠️ python-uinput não disponível.")
        mouse_enable_layout.addWidget(self.mouse_checkbox)
        mouse_group_layout.addLayout(mouse_enable_layout)

        # FOV controls
        fov_group = QGroupBox("FOV (Campo de Visão)")
        fov_layout = QVBoxLayout()

        draw_fov_layout = QHBoxLayout()
        self.draw_fov_checkbox = QCheckBox("Desenhar FOV na tela")
        try:
            self.draw_fov_checkbox.setChecked(bool(getattr(cfg, 'DRAW_FOV_DEFAULT', False)))
        except Exception:
            self.draw_fov_checkbox.setChecked(False)
        self.draw_fov_checkbox.stateChanged.connect(lambda s: self.overlay.set_fov_draw_enabled(s == Qt.Checked))
        draw_fov_layout.addWidget(self.draw_fov_checkbox)
        fov_layout.addLayout(draw_fov_layout)

        fov_size_layout = QHBoxLayout()
        fov_size_layout.addWidget(QLabel("Raio (px):"))
        self.fov_radius_spin = QSpinBox()
        self.fov_radius_spin.setRange(10, 2000)
        try:
            self.fov_radius_spin.setValue(int(getattr(cfg, 'FOV_RADIUS_DEFAULT', 150)))
        except Exception:
            self.fov_radius_spin.setValue(150)
        # Atualiza overlay e lógica
        def _on_fov_radius_changed(v: int):
            try:
                self.overlay.set_fov_radius(int(v))
            except Exception:
                pass
            try:
                self.detection_thread.set_fov_radius(int(v))
            except Exception:
                pass
        self.fov_radius_spin.valueChanged.connect(_on_fov_radius_changed)
        fov_size_layout.addWidget(self.fov_radius_spin)
        fov_layout.addLayout(fov_size_layout)

        # Ajuste de posição do FOV (offset relativo ao centro da região)
        fov_offset_layout = QHBoxLayout()
        fov_offset_layout.addWidget(QLabel("Ajuste X:"))
        self.fov_offset_x_spin = QSpinBox()
        self.fov_offset_x_spin.setRange(-4000, 4000)
        self.fov_offset_x_spin.setValue(0)
        fov_offset_layout.addWidget(self.fov_offset_x_spin)

        fov_offset_layout.addWidget(QLabel("Ajuste Y:"))
        self.fov_offset_y_spin = QSpinBox()
        self.fov_offset_y_spin.setRange(-4000, 4000)
        self.fov_offset_y_spin.setValue(0)
        fov_offset_layout.addWidget(self.fov_offset_y_spin)

        def _on_fov_offset_changed():
            dx = int(self.fov_offset_x_spin.value())
            dy = int(self.fov_offset_y_spin.value())
            try:
                self.overlay.set_fov_offset(dx, dy)
            except Exception:
                pass
            try:
                self.detection_thread.set_fov_offset(dx, dy)
            except Exception:
                pass
        self.fov_offset_x_spin.valueChanged.connect(lambda _v: _on_fov_offset_changed())
        self.fov_offset_y_spin.valueChanged.connect(lambda _v: _on_fov_offset_changed())

        fov_layout.addLayout(fov_offset_layout)

        fov_group.setLayout(fov_layout)
        mouse_group_layout.addWidget(fov_group)

        # Ajuste de mira (bias X/Y)
        aim_group = QGroupBox("Mira")
        aim_layout = QHBoxLayout()
        aim_layout.addWidget(QLabel("Correção X (px):"))
        self.aim_bias_x_spin = QSpinBox()
        self.aim_bias_x_spin.setRange(-2000, 2000)
        try:
            self.aim_bias_x_spin.setValue(int(getattr(cfg, 'AIM_X_BIAS_DEFAULT', 0)))
        except Exception:
            self.aim_bias_x_spin.setValue(0)
        aim_layout.addWidget(self.aim_bias_x_spin)

        aim_layout.addWidget(QLabel("Correção Y (px):"))
        self.aim_bias_y_spin = QSpinBox()
        self.aim_bias_y_spin.setRange(-2000, 2000)
        try:
            self.aim_bias_y_spin.setValue(int(getattr(cfg, 'AIM_Y_BIAS_DEFAULT', 0)))
        except Exception:
            self.aim_bias_y_spin.setValue(0)
        aim_layout.addWidget(self.aim_bias_y_spin)

        def _on_aim_bias_changed():
            bx = int(self.aim_bias_x_spin.value())
            by = int(self.aim_bias_y_spin.value())
            try:
                self.detection_thread.set_aim_bias(bx, by)
            except Exception:
                pass
        self.aim_bias_x_spin.valueChanged.connect(lambda _v: _on_aim_bias_changed())
        self.aim_bias_y_spin.valueChanged.connect(lambda _v: _on_aim_bias_changed())

        aim_group.setLayout(aim_layout)
        mouse_group_layout.addWidget(aim_group)

        # Eixos habilitados
        axis_group = QGroupBox("Eixos")
        axis_layout = QHBoxLayout()
        self.axis_x_checkbox = QCheckBox("Eixo X")
        self.axis_x_checkbox.setChecked(True)
        self.axis_y_checkbox = QCheckBox("Eixo Y")
        self.axis_y_checkbox.setChecked(True)

        def _on_axis_changed():
            try:
                self.detection_thread.set_axis_x_enabled(self.axis_x_checkbox.isChecked())
                self.detection_thread.set_axis_y_enabled(self.axis_y_checkbox.isChecked())
            except Exception:
                pass
        self.axis_x_checkbox.stateChanged.connect(lambda _s: _on_axis_changed())
        self.axis_y_checkbox.stateChanged.connect(lambda _s: _on_axis_changed())

        axis_layout.addWidget(self.axis_x_checkbox)
        axis_layout.addWidget(self.axis_y_checkbox)

        axis_group.setLayout(axis_layout)
        mouse_group_layout.addWidget(axis_group)

        # Velocidade do mouse
        speed_group = QGroupBox("Velocidade")
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Ganho:"))
        self.mouse_gain_spin = QDoubleSpinBox()
        self.mouse_gain_spin.setRange(0.05, 10.0)
        self.mouse_gain_spin.setDecimals(2)
        self.mouse_gain_spin.setSingleStep(0.05)
        try:
            self.mouse_gain_spin.setValue(float(getattr(cfg, 'MOUSE_GAIN', 0.6)))
        except Exception:
            self.mouse_gain_spin.setValue(0.6)
        speed_layout.addWidget(self.mouse_gain_spin)

        speed_layout.addWidget(QLabel("Passo máx.:"))
        self.mouse_step_spin = QSpinBox()
        self.mouse_step_spin.setRange(1, 200)
        try:
            self.mouse_step_spin.setValue(int(getattr(cfg, 'MOUSE_MAX_STEP', 15)))
        except Exception:
            self.mouse_step_spin.setValue(15)
        speed_layout.addWidget(self.mouse_step_spin)


        def _on_speed_changed():
            try:
                self.detection_thread.set_mouse_gain(float(self.mouse_gain_spin.value()))
            except Exception:
                pass
            try:
                self.detection_thread.set_mouse_max_step(int(self.mouse_step_spin.value()))
            except Exception:
                pass
        self.mouse_gain_spin.valueChanged.connect(lambda _v: _on_speed_changed())
        self.mouse_step_spin.valueChanged.connect(lambda _v: _on_speed_changed())

        speed_group.setLayout(speed_layout)
        mouse_group_layout.addWidget(speed_group)

        mouse_group.setLayout(mouse_group_layout)
        mouse_tab_layout.addWidget(mouse_group)
        mouse_tab.setLayout(mouse_tab_layout)
        tabs.addTab(mouse_tab, "🖱️ Mouse")

        # Adiciona tabs ao layout principal
        layout.addWidget(tabs)
        # Adiciona imagem oculta (compatibilidade)
        layout.addWidget(self.image_label)

        log_label = QLabel("📋 Log de Detecções:")
        log_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.setFont(QFont("Monospace", 8))
        self.log_text.setReadOnly(True)
        self.max_log_lines = 15
        layout.addWidget(self.log_text)

        central_widget.setLayout(layout)

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_fps_counter)
        self.ui_timer.start(100)

        self.frame_count = 0
        self.last_time = time.time()

        # Janela de sobreposição para desenhar contornos na tela inteira
        self.overlay = OverlayWindow()
        self.overlay.hide()
        # Aplicar estado inicial do FOV no overlay
        try:
            self.overlay.set_fov_radius(int(self.fov_radius_spin.value()))
            self.overlay.set_fov_draw_enabled(self.draw_fov_checkbox.isChecked())
            self.overlay.set_fov_offset(int(self.fov_offset_x_spin.value()), int(self.fov_offset_y_spin.value()))
        except Exception:
            pass
        # Aplicar FOV inicial na lógica de detecção
        try:
            self.detection_thread.set_fov_radius(int(self.fov_radius_spin.value()))
            self.detection_thread.set_fov_offset(int(self.fov_offset_x_spin.value()), int(self.fov_offset_y_spin.value()))
            self.detection_thread.set_aim_bias(int(self.aim_bias_x_spin.value()), int(self.aim_bias_y_spin.value()))
            self.detection_thread.set_axis_x_enabled(self.axis_x_checkbox.isChecked())
            self.detection_thread.set_axis_y_enabled(self.axis_y_checkbox.isChecked())
            self.detection_thread.set_mouse_gain(float(self.mouse_gain_spin.value()))
            self.detection_thread.set_mouse_max_step(int(self.mouse_step_spin.value()))
            self.detection_thread.set_target_strategy_distance(self.target_distance_checkbox.isChecked())
            self.detection_thread.set_target_stick_radius(int(self.target_stick_radius_spin.value()))
            # Filtro de cor inicial
            try:
                self._color_h, self._color_s, self._color_v = getattr(self, '_color_h', 0), getattr(self, '_color_s', 0), getattr(self, '_color_v', 0)
            except Exception:
                self._color_h, self._color_s, self._color_v = 0, 0, 0
            self.detection_thread.set_color_filter_enabled(self.color_filter_checkbox.isChecked())
            self.detection_thread.set_color_target_hsv(int(self._color_h), int(self._color_s), int(self._color_v))
            self.detection_thread.set_color_params(int(self.color_tol_h_spin.value()), int(self.color_min_s_spin.value()), int(self.color_min_v_spin.value()))
        except Exception:
            pass

    # (sem seleção de modelo; pipeline apenas OpenCV)

    # ----- Color filter handlers -----
    def on_color_filter_enabled_changed(self, enabled: bool):
        try:
            self.detection_thread.set_color_filter_enabled(bool(enabled))
            self.log("🎨 Filtro de cor: " + ("ativado" if enabled else "desativado"))
        except Exception:
            pass

    def on_color_params_changed(self):
        try:
            tol = int(self.color_tol_h_spin.value())
            smin = int(self.color_min_s_spin.value())
            vmin = int(self.color_min_v_spin.value())
            self.detection_thread.set_color_params(tol, smin, vmin)
        except Exception:
            pass

    def start_color_picker(self):
        # Minimiza a janela e mostra overlay de captura de cor
        try:
            self.showMinimized()
            QApplication.processEvents()
            time.sleep(0.3)
            self.color_picker.start()
        except Exception:
            self.showNormal()

    def on_color_picked(self, hsv: tuple):
        # hsv: (h,s,v)
        try:
            self.showNormal()
        except Exception:
            pass
        try:
            h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        except Exception:
            return
        # Atualiza preview e envia para a thread
        try:
            # Converter para RGB para o preview
            rgb = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
            b, g, r = int(rgb[0]), int(rgb[1]), int(rgb[2])
            self.color_preview.setStyleSheet(f"border: 1px solid #2b2f3a; background: rgb({r},{g},{b});")
        except Exception:
            pass
        try:
            self._color_h, self._color_s, self._color_v = h, s, v
            # Atualiza o detector ao vivo
            self.detection_thread.set_color_target_hsv(h, s, v)
            # Reflete também nos controles da aba Train (se existirem)
            try:
                if hasattr(self, 'train_h_spin'):
                    self.train_h_spin.blockSignals(True)
                    self.train_h_spin.setValue(int(h))
                    self.train_h_spin.blockSignals(False)
            except Exception:
                pass
            # Garantir filtro ativo e reprocessar
            self.color_filter_checkbox.setChecked(True)
            self.on_color_filter_enabled_changed(True)
            # Atualiza prévia imediatamente
            try:
                self.on_train_params_changed()
            except Exception:
                pass
            self.log(f"🎯 Cor escolhida (HSV): H={h} S={s} V={v}")
        except Exception:
            pass

    # ----- Filtro humanoide -----
    def on_filter_params_changed(self):
        if not hasattr(self, 'filter_min_area_spin'):
            return
        try:
            self.filter_max_area_spin.setMinimum(int(self.filter_min_area_spin.value()) + 1)
        except Exception:
            pass
        params = {
            'min_area': float(self.filter_min_area_spin.value()),
            'max_area': float(self.filter_max_area_spin.value()),
            'min_aspect': float(self.filter_min_aspect_spin.value()),
            'max_aspect': float(self.filter_max_aspect_spin.value()),
            'min_extent': float(self.filter_min_extent_spin.value()),
            'min_solidity': float(self.filter_min_solidity_spin.value()),
            'min_circularity': float(self.filter_min_circularity_spin.value()),
            'max_circularity': float(self.filter_max_circularity_spin.value()),
            'max_sym_diff': float(self.filter_max_sym_spin.value()),
            'min_top_ratio': float(self.filter_min_top_spin.value()),
            'min_bottom_ratio': float(self.filter_min_bottom_spin.value()),
            'conf_blend': float(self.filter_conf_blend_spin.value()),
        }
        self.filter_min_area = params['min_area']
        self.filter_max_area = params['max_area']
        self.filter_min_aspect = params['min_aspect']
        self.filter_max_aspect = params['max_aspect']
        self.filter_min_extent = params['min_extent']
        self.filter_min_solidity = params['min_solidity']
        self.filter_min_circularity = params['min_circularity']
        self.filter_max_circularity = params['max_circularity']
        self.filter_max_sym_diff = params['max_sym_diff']
        self.filter_min_top_ratio = params['min_top_ratio']
        self.filter_min_bottom_ratio = params['min_bottom_ratio']
        self.filter_conf_blend = params['conf_blend']
        try:
            self.detection_thread.set_humanoid_filter_params(**params)
        except Exception:
            pass
        try:
            self._update_status(
                f"Filtro: área {int(params['min_area'])}-{int(params['max_area'])} | aspecto {params['min_aspect']:.2f}-{params['max_aspect']:.2f}"
            )
        except Exception:
            pass

    def on_filter_mode_changed(self, enabled: bool):
        self.filter_mask_mode = bool(enabled)
        try:
            self.detection_thread.set_preview_mask_detection(self.filter_mask_mode)
        except Exception:
            pass
        for widget in getattr(self, '_filter_param_widgets', []):
            try:
                widget.setEnabled(not self.filter_mask_mode)
            except Exception:
                pass
        try:
            msg = "Detecção simplificada (sem filtro humanoide)" if enabled else "Filtro humanoide ativo"
            self._update_status(msg)
        except Exception:
            pass
        if not enabled:
            self.on_filter_params_changed()

    def _update_filter_debug_text(self):
        if not hasattr(self, 'filter_debug_text'):
            return
        entries = getattr(self, '_filter_last_debug', []) or []
        if not entries:
            self.filter_debug_text.setPlainText("Sem dados suficientes para diagnóstico. Aguarde capturas.")
            return
        lines = []
        for idx, entry in enumerate(entries[:40], start=1):
            area = float(entry.get('area', 0.0))
            aspect = float(entry.get('aspect', 0.0))
            extent = float(entry.get('extent', 0.0))
            conf = float(entry.get('conf', 0.0)) if entry.get('conf') is not None else 0.0
            reason = entry.get('reason', '')
            if entry.get('accepted'):
                lines.append(
                    f"{idx}. OK conf={conf:.2f} area={area:.0f} asp={aspect:.2f} ext={extent:.2f}"
                )
            else:
                lines.append(f"{idx}. X {reason or 'rejeitado'} area={area:.0f} asp={aspect:.2f}")
        self.filter_debug_text.setPlainText("\n".join(lines))

    # ---------- Train tab handlers ----------
    def _to_pixmap(self, img: np.ndarray) -> QPixmap:
        try:
            if img is None:
                return QPixmap()
            if img.ndim == 2:
                h, w = img.shape
                qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
                return QPixmap.fromImage(qimg.copy())
            elif img.ndim == 3 and img.shape[2] == 3:
                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
                return QPixmap.fromImage(qimg.copy())
        except Exception:
            pass
        return QPixmap()

    # Todos os handlers de Train aqui não salvam arquivos e não treinam; apenas ajustam ao vivo

    def on_train_params_changed(self):
        # Atualiza parâmetros no detector e reprocessa a prévia atual
        try:
            h = int(self.train_h_spin.value())
            tol = int(self.train_tol_h_spin.value())
            smin = int(self.train_smin_spin.value())
            vmin = int(self.train_vmin_spin.value())
            # Atualize só o H do alvo (a cor escolhida define S/V)
            self._color_h = h
            # Mantenha S/V do alvo conforme a última cor escolhida no picker
            s_alvo = int(getattr(self, "_color_s", 255))
            v_alvo = int(getattr(self, "_color_v", 255))
            # Ajusta parâmetros AO VIVO
            self.detection_thread.set_color_target_hsv(self._color_h, s_alvo, v_alvo)
            self.detection_thread.set_color_params(tol, smin, vmin)
            # (sem modo distância; apenas HSV clássico)
            edge_scale = float(self.train_edge_scale.value())
            kscale = float(self.train_kscale.value())
            dil = int(self.train_dil_iter.value())
            clo = int(self.train_close_iter.value())
            opn = int(self.train_open_iter.value())
            self.detection_thread.set_morph_params(edge_scale=edge_scale, kscale=kscale, dilate_iter=dil, close_iter=clo, open_iter=opn)
        except Exception:
            pass
        # A prévia será atualizada por timer com o último frame da captura

    def _on_preview_frame(self, frame: np.ndarray):
        try:
            self._train_last_frame = frame.copy()
        except Exception:
            pass

    def on_train_preview_tick(self):
        try:
            frame = self._train_last_frame
            if frame is None or frame.size == 0:
                return
            h, w = frame.shape[:2]
            region = {"left": 0, "top": 0, "width": w, "height": h}
            prev = self.detection_thread.debug_preview(frame, region)
            mask = prev.get('mask'); overlay = prev.get('overlay')
            if mask is not None:
                self.train_mask_label.setPixmap(self._to_pixmap(mask).scaled(self.train_mask_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if overlay is not None:
                self.train_overlay_label.setPixmap(self._to_pixmap(overlay).scaled(self.train_overlay_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            filtro_prev = self.detection_thread.debug_filter_preview(frame, region)
            f_mask = filtro_prev.get('mask') if isinstance(filtro_prev, dict) else None
            f_overlay = filtro_prev.get('overlay') if isinstance(filtro_prev, dict) else None
            if f_mask is not None and hasattr(self, 'filter_mask_label'):
                self.filter_mask_label.setPixmap(self._to_pixmap(f_mask).scaled(self.filter_mask_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if f_overlay is not None and hasattr(self, 'filter_overlay_label'):
                self.filter_overlay_label.setPixmap(self._to_pixmap(f_overlay).scaled(self.filter_overlay_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            try:
                self._filter_last_debug = filtro_prev.get('debug', []) if isinstance(filtro_prev, dict) else []
            except Exception:
                self._filter_last_debug = []
            self._update_filter_debug_text()
        except Exception:
            pass

    def on_overlay_offset_changed(self, _value: int):
        try:
            dx = int(self.offset_x_spin.value())
            dy = int(self.offset_y_spin.value())
        except Exception:
            dx, dy = 0, 0
        try:
            self.overlay.set_offset(dx, dy)
        except Exception:
            pass

    def on_desktop_filter_changed(self, state):
        self.only_current_desktop = state == Qt.Checked
        self.refresh_windows_list()

    def refresh_windows_list(self):
        self.window_combo.clear()
        try:
            windows = WindowSelector.get_windows_list(self.only_current_desktop)
            for window_name, window_info in windows:
                self.window_combo.addItem(window_name, window_info)
            if HAS_EWMH:
                self.log(f"✅ Lista atualizada: {len(windows)} opções encontradas")
            else:
                self.log("⚠️ python-ewmh não instalado. Apenas monitores disponíveis.")
                self.log("💡 Instale com: pip install ewmh")
            try:
                self._update_status(f"{len(windows)} fontes disponíveis")
            except Exception:
                pass
        except Exception as e:
            self.log_error(f"Erro ao atualizar lista: {str(e)}")
            self.window_combo.addItem("Tela Completa", None)

    def on_window_selected(self, index):
        if index >= 0:
            window_name = self.window_combo.currentText()
            self.current_window_info = self.window_combo.currentData()
            region = WindowSelector.get_window_region(self.current_window_info)
            self.capture_thread.set_capture_region(region, self.current_window_info)

            if self.current_window_info is None:
                info = "Tela Completa"
            elif self.current_window_info.get("type") == "monitor":
                info = f"Monitor {self.current_window_info['monitor_index']}"
            else:
                info = window_name

            self._update_status(f"Capturando: {info}")
            self.log(f"📷 Fonte alterada: {info}")

    def on_capture_error(self, error_msg: str):
        self.log_error(error_msg)
        try:
            self._update_status(error_msg)
        except Exception:
            pass
        if "Voltando para tela completa" in error_msg:
            self.window_combo.setCurrentIndex(0)

    def on_engine_changed(self, _index: int):
        # Só permite trocar com detecção parada
        if self.running:
            self.log("⏳ Pare a detecção para trocar o motor.")
            return
        new_engine = self.engine_combo.currentData()
        old_thread = self.capture_thread
        # Desconectar e parar antiga
        try:
            if old_thread.isRunning():
                try:
                    old_thread.stop()
                except Exception:
                    pass
                if not old_thread.wait(1500):
                    try:
                        old_thread.terminate()
                    except Exception:
                        pass
                    old_thread.wait(300)
        except Exception:
            pass
        self._unwire_capture_signals(old_thread)

        # Criar nova thread
        if new_engine == 'fastgrab':
            from detect_logic import FastgrabCaptureThread as _FGCT
            self.capture_thread = _FGCT(cfg.DETECT_RESOLUTION, self.fps_limit)
        else:
            self.capture_thread = FastCaptureThread(cfg.DETECT_RESOLUTION, self.fps_limit)

        self._wire_capture_signals(self.capture_thread)

        # Aplicar configs atuais
        try:
            self.capture_thread.set_target_size((self.resize_w_spin.value(), self.resize_h_spin.value()))
        except Exception:
            pass
        try:
            self.capture_thread.set_fps_limit(self.fps_limit)
        except Exception:
            pass
        # Repor região atual
        try:
            region = WindowSelector.get_window_region(self.current_window_info)
            self.capture_thread.set_capture_region(region, self.current_window_info)
        except Exception:
            pass
        self.log(f"🔧 Motor de captura: {self.engine_combo.currentText()}")

    def start_region_selection(self):
        reply = QMessageBox.information(
            self,
            "Seleção de Região",
            "A janela será minimizada.\n\nClique e arraste com o mouse para selecionar a região desejada.",
            QMessageBox.Ok | QMessageBox.Cancel
        )
        if reply == QMessageBox.Ok:
            self.showMinimized()
            QApplication.processEvents()
            time.sleep(0.5)
            screen = QApplication.primaryScreen().geometry()
            self.region_selector.setGeometry(screen)
            self.region_selector.showFullScreen()

    def on_region_selected(self, region: dict):
        self.showNormal()
        region_info = {"custom_region": region, "type": "custom"}
        self.capture_thread.set_capture_region(region, region_info)
        info = f"Região Customizada ({region['width']}x{region['height']})"
        self._update_status(f"Capturando: {info}")
        self.window_combo.addItem(info, region_info)
        self.window_combo.setCurrentIndex(self.window_combo.count() - 1)
        self.log(f"✂️ Região selecionada: {region['width']}x{region['height']} em ({region['left']}, {region['top']})")

    def on_confidence_changed(self, value: int):
        self.confidence = value / 100.0
        self.confidence_spinbox.setValue(value)
        self.confidence_value_label.setText(f"Valor atual: {self.confidence:.2f}")
        self.detection_thread.set_confidence(self.confidence)

    def on_confidence_spinbox_changed(self, value: int):
        self.confidence_slider.setValue(value)

    def on_fps_changed(self, value: int):
        self.fps_limit = max(1, int(value))
        self.capture_thread.set_fps_limit(self.fps_limit)
        self.log(f"⚙️ FPS de captura ajustado para {self.fps_limit}")

    def on_resize_changed(self):
        w = int(self.resize_w_spin.value())
        h = int(self.resize_h_spin.value())
        self.capture_thread.set_target_size((w, h))
        self.log(f"🖼️ Resolução de redimensionamento ajustada para {w}x{h}")

    # (sem modelo externo)

    def on_mouse_checkbox_changed(self, state: int):
        enabled = state == Qt.Checked
        ok, msg = self.detection_thread.set_mouse_enabled(enabled)
        if ok:
            if enabled:
                self.log(f"🖱️ Mouse (kernel): ativado — {msg}")
            else:
                self.log("🖱️ Mouse: desativado")
        else:
            self.log_error(msg)
            # Reverter checkbox
            self.mouse_checkbox.blockSignals(True)
            self.mouse_checkbox.setChecked(False)
            self.mouse_checkbox.blockSignals(False)

    def on_benchmark_clicked(self):
        # Passo 1: validações básicas — sem dependência de modelo

        # Aquecimento: garantir que a detecção esteja rodando ANTES da contagem
        started_here = False
        if not self.running:
            self.log("⚙️ Iniciando detecção para aquecimento...")
            self.toggle_detection()
            started_here = True

        # Passo 2: contagem regressiva
        self.log("⏳ Iniciando benchmark em 5 segundos...")
        for i in range(5, 0, -1):
            self.log(f"{i}...")
            QApplication.processEvents()
            time.sleep(1.0)

        # Passo 3: iniciar benchmark por 30s
        duration_s = 30
        self.log(f"🚀 Benchmark iniciado por {duration_s}s. Coletando métricas...")
        self.detection_thread.start_benchmark()

        t_start = time.time()
        while time.time() - t_start < duration_s:
            QApplication.processEvents()
            time.sleep(0.05)

        # Parar benchmark e, se necessário, parar detecção
        data = self.detection_thread.stop_benchmark()
        if started_here:
            self.toggle_detection()

        # Passo 4: consolidar métricas
        if not data:
            self.log_error("Nenhum dado coletado durante o benchmark. Verifique se há frames/detecções.")
            return

        import statistics
        # Suporta formatos antigos e novos de coleta
        def _get_time_ms(d):
            if "infer_time_ms" in d:
                return float(d["infer_time_ms"])
            if "infer_ms" in d:
                return float(d["infer_ms"])
            # fallback: 0
            return 0.0

        times_ms = [_get_time_ms(d) for d in data if _get_time_ms(d) > 0.0]
        if not times_ms:
            self.log_error("Dados de benchmark inválidos (sem tempos de inferência).")
            return

        def _get_fps(d):
            if "fps" in d:
                try:
                    return float(d["fps"])
                except Exception:
                    return 0.0
            t = _get_time_ms(d)
            return (1000.0 / t) if t > 0 else 0.0

        fps_list = [_get_fps(d) for d in data]
        n_boxes_list = [int(d.get("n_boxes", d.get("boxes", 0))) for d in data]
        avg_box_conf_list = [float(d["avg_box_conf"]) for d in data if d.get("avg_box_conf") is not None]

        avg_ms = float(statistics.mean(times_ms))
        p50_ms = float(np.percentile(times_ms, 50))
        p90_ms = float(np.percentile(times_ms, 90))
        p95_ms = float(np.percentile(times_ms, 95))
        p99_ms = float(np.percentile(times_ms, 99))
        min_ms = float(min(times_ms))
        max_ms = float(max(times_ms))
        avg_fps = float(statistics.mean(fps_list))
        med_fps = float(np.median(fps_list))
        std_ms = float(statistics.pstdev(times_ms)) if len(times_ms) > 1 else 0.0
        avg_boxes = float(statistics.mean(n_boxes_list))
        avg_box_conf = float(statistics.mean(avg_box_conf_list)) if avg_box_conf_list else float('nan')

        detector_name = data[0].get("detector", "opencv_contour")
        img_w = data[-1].get("img_w", getattr(self, 'resize_w_spin', None).value() if hasattr(self, 'resize_w_spin') else None)
        img_h = data[-1].get("img_h", getattr(self, 'resize_h_spin', None).value() if hasattr(self, 'resize_h_spin') else None)
        conf_used = float(data[-1].get("conf_threshold", getattr(self, 'confidence', 0.25)))
        total_samples = len(times_ms)

        self.log("✅ Benchmark concluído. Resumo:")
        self.log(f"Detector: {detector_name}")
        self.log(f"Resolução de detecção: {img_w}x{img_h}")
        self.log(f"Confidence: {conf_used:.2f}")
        self.log(f"Amostras: {total_samples}")
        self.log(f"Inferência [ms]: média={avg_ms:.2f}, p50={p50_ms:.2f}, p90={p90_ms:.2f}, p95={p95_ms:.2f}, p99={p99_ms:.2f}, min={min_ms:.2f}, max={max_ms:.2f}, std={std_ms:.2f}")
        self.log(f"FPS: média={avg_fps:.2f}, mediana={med_fps:.2f}")
        self.log(f"Média de boxes por frame: {avg_boxes:.2f}")
        if not np.isnan(avg_box_conf):
            self.log(f"Confiança média das detecções: {avg_box_conf:.3f}")

        # Passo 5: salvar CSV detalhado e claro
        import csv
        os.makedirs(os.path.join(cfg.REPO_ROOT, "runs", "benchmarks"), exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        csv_path = os.path.join(cfg.REPO_ROOT, "runs", "benchmarks", f"benchmark_{timestamp}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Cabeçalho mais claro
            writer.writerow(["frame_ts", "infer_ms", "inst_fps", "boxes", "avg_box_conf", "conf_threshold", "img_w", "img_h", "detector"])
            for row in data:
                ts = float(row.get('timestamp', row.get('ts', time.time())))
                t_ms = _get_time_ms(row)
                fps_v = _get_fps(row)
                nboxes = int(row.get('n_boxes', row.get('boxes', 0)))
                avgc = row.get('avg_box_conf')
                iw = row.get('img_w', img_w)
                ih = row.get('img_h', img_h)
                confv = float(row.get('conf_threshold', conf_used))
                det = row.get('detector', detector_name)
                writer.writerow([
                    f"{ts:.3f}", f"{t_ms:.4f}", f"{fps_v:.4f}", nboxes,
                    (f"{float(avgc):.4f}" if avgc is not None else ""),
                    f"{confv:.2f}", iw, ih, det
                ])

        # Resumo como arquivo separado (menos confuso)
        summary_path = os.path.join(cfg.REPO_ROOT, "runs", "benchmarks", f"benchmark_{timestamp}_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["detector", detector_name])
            writer.writerow(["resolution", f"{img_w}x{img_h}"])
            writer.writerow(["conf_threshold", f"{conf_used:.2f}"])
            writer.writerow(["samples", total_samples])
            writer.writerow(["infer_ms_avg", f"{avg_ms:.4f}"])
            writer.writerow(["infer_ms_p50", f"{p50_ms:.4f}"])
            writer.writerow(["infer_ms_p90", f"{p90_ms:.4f}"])
            writer.writerow(["infer_ms_p95", f"{p95_ms:.4f}"])
            writer.writerow(["infer_ms_p99", f"{p99_ms:.4f}"])
            writer.writerow(["infer_ms_min", f"{min_ms:.4f}"])
            writer.writerow(["infer_ms_max", f"{max_ms:.4f}"])
            writer.writerow(["infer_ms_std", f"{std_ms:.4f}"])
            writer.writerow(["fps_avg", f"{avg_fps:.4f}"])
            writer.writerow(["fps_median", f"{med_fps:.4f}"])
            writer.writerow(["boxes_avg", f"{avg_boxes:.4f}"])
            writer.writerow(["avg_box_conf", (f"{avg_box_conf:.4f}" if not np.isnan(avg_box_conf) else "")])

        self.log(f"CSV salvo em: {csv_path}")
        self.log(f"Resumo salvo em: {summary_path}")

    def toggle_detection(self):
        if not self.running:
            self.running = True
            self.capture_thread.running = True
            self.detection_thread.running = True
            self.detection_thread.set_confidence(self.confidence)
            try:
                self.detection_thread.set_pixel_step(self.pixel_step)
            except Exception:
                pass
            try:
                self.detection_thread.set_infer_fps_limit(self.infer_fps_limit)
            except Exception:
                pass
            # Sincronizar FPS de captura com o de inferência para evitar carga desnecessária
            try:
                new_cap_fps = min(int(self.fps_limit), int(self.infer_fps_limit))
                if new_cap_fps != self.fps_limit:
                    self.fps_limit = new_cap_fps
                    self.capture_thread.set_fps_limit(self.fps_limit)
                    try:
                        self.fps_spin.setValue(self.fps_limit)
                    except Exception:
                        pass
                    self.log(f"⚙️ Ajustado FPS de captura para {self.fps_limit} (alinhado à inferência)")
            except Exception:
                pass
            self.capture_thread.start()
            self.detection_thread.start()
            # Mostrar sobreposição transparente cobrindo todas as telas
            if self.overlay_enabled:
                self.overlay.show_all_screens()
            else:
                try:
                    self.overlay.hide()
                except Exception:
                    pass
            

            self.btn_toggle.setText("Parar Detecção")
            self.btn_toggle.setStyleSheet("""
                QPushButton { font-size: 14px; font-weight: bold; padding: 10px 14px; background-color: #ef4444; color: white; border: none; border-radius: 8px; }
                QPushButton:hover { background-color: #dc2626; }
                QPushButton:pressed { background-color: #b91c1c; }
            """)

            self.window_combo.setEnabled(False)
            self.btn_refresh.setEnabled(False)
            self.btn_select_region.setEnabled(False)
            self.checkbox_current_desktop.setEnabled(False)

            self._update_status("Detecção ativa")
            self.log(f"▶️ Detecção iniciada (confidence: {self.confidence:.2f})")
        else:
            self.running = False
            # Solicitar parada graciosa
            try:
                self.capture_thread.stop()
            except Exception:
                pass
            try:
                self.detection_thread.stop()
            except Exception:
                pass
            # Aguardar com timeouts para não travar a UI
            try:
                if not self.capture_thread.wait(2000):
                    # Em caso extremo, matar thread de captura
                    try:
                        self.capture_thread.terminate()
                    except Exception:
                        pass
                    self.capture_thread.wait(500)
            except Exception:
                pass
            try:
                if not self.detection_thread.wait(5000):
                    try:
                        self.detection_thread.terminate()
                    except Exception:
                        pass
                    self.detection_thread.wait(500)
            except Exception:
                pass

            self.btn_toggle.setText("▶ Iniciar Detecção")
            self.btn_toggle.setStyleSheet("""
                QPushButton { font-size: 14px; font-weight: bold; padding: 10px 14px; background-color: #10b981; color: white; border: none; border-radius: 8px; }
                QPushButton:hover { background-color: #059669; }
                QPushButton:pressed { background-color: #047857; }
            """)

            # Esconder sobreposição e limpar boxes
            self.overlay.clear_boxes()
            self.overlay.hide()

            self.window_combo.setEnabled(True)
            self.btn_refresh.setEnabled(True)
            self.btn_select_region.setEnabled(True)
            if HAS_EWMH:
                self.checkbox_current_desktop.setEnabled(True)

            self._update_status("Detecção pausada")
            self.log("⏸ Detecção pausada")

    def update_display_overlay(self, frame: np.ndarray, detections: List[str], boxes: list):
        # Atualiza apenas a sobreposição com caixas em coordenadas de tela
        try:
            self.overlay.set_boxes(boxes)
        except Exception:
            pass
        self.frame_count += 1
        if detections:
            self.log(f"🎯 Detectado: {', '.join(detections)}")

    def update_fps_counter(self):
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed > 1.0:
            fps = self.frame_count / elapsed
            title = (
                f"OpenCV Screen Detector - FPS: {fps:.1f} | Confidence: {self.confidence:.2f} "
                f"| Detecção: {self.resize_w_spin.value()}x{self.resize_h_spin.value()}"
            )
            if not HAS_EWMH:
                title += " | ⚠️ EWMH não disponível"
            self.setWindowTitle(title)
            self.frame_count = 0
            self.last_time = current_time

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        doc = self.log_text.document()
        if doc.blockCount() > self.max_log_lines:
            cursor = QTextCursor(doc.firstBlock())
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
        self.log_text.moveCursor(QTextCursor.End)

    def log_error(self, message: str):
        self.log(f"❌ {message}")

    def on_pixel_step_changed(self, value: int):
        self.pixel_step = max(1, int(value))
        try:
            self.detection_thread.set_pixel_step(self.pixel_step)
        except Exception:
            pass
        self.log(f"⚙️ Passo de pixels ajustado para N={self.pixel_step}")

    def on_infer_fps_changed(self, value: int):
        self.infer_fps_limit = max(1, int(value))
        try:
            self.detection_thread.set_infer_fps_limit(self.infer_fps_limit)
        except Exception:
            pass
        self.log(f"⚙️ FPS (inferência) ajustado para {self.infer_fps_limit}")

    def closeEvent(self, event):
        self.running = False
        if self.capture_thread.isRunning():
            try:
                self.capture_thread.stop()
                if not self.capture_thread.wait(1500):
                    try:
                        self.capture_thread.terminate()
                    except Exception:
                        pass
                    self.capture_thread.wait(300)
            except Exception:
                pass
        if self.detection_thread.isRunning():
            try:
                self.detection_thread.stop()
                if not self.detection_thread.wait(3000):
                    try:
                        self.detection_thread.terminate()
                    except Exception:
                        pass
                    self.detection_thread.wait(300)
            except Exception:
                pass
        event.accept()


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    detector = ScreenDetector()
    detector.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
