import sys
import os
import time
import math
import threading
from collections import deque
from typing import Optional, Dict, Tuple, List

import cv2
import numpy as np
import mss
from PyQt5.QtCore import QThread, pyqtSignal

import config as cfg

# Dependências opcionais (Linux): usar apenas EWMH
try:
    from ewmh import EWMH
    try:
        ewmh = EWMH()
        HAS_EWMH = True
    except Exception:
        HAS_EWMH = False
        ewmh = None
        print(cfg.MSG_MISSING_EWMH)
except ImportError:
    HAS_EWMH = False
    ewmh = None
    print(cfg.MSG_MISSING_EWMH)
    
# Xlib para mover o mouse (Xorg)
try:
    from Xlib import display as _x_display_mod, X as _X
    from Xlib.ext import xtest as _xtest
    HAS_XLIB_MOUSE = True
except Exception:
    HAS_XLIB_MOUSE = False

# uinput (opcional, não usado em Xorg-only)
try:
    import uinput
    HAS_UINPUT = True
except Exception:
    HAS_UINPUT = False
    # Não requerido em modo Xorg-only

# Detecção 100% OpenCV/NumPy (sem redes neurais)

# (sem dependências de deep learning)

# Fastgrab (opcional, captura muito rápida)
try:
    from fastgrab import screenshot as _fg_screenshot
    HAS_FASTGRAB = True
except Exception:
    HAS_FASTGRAB = False
    print("fastgrab não disponível. Instale com: pip install fastgrab")


class WindowSelector:
    """Gerencia a seleção de janelas/monitores para captura (Linux + monitores via mss)."""

    @staticmethod
    def get_windows_list(only_current_desktop: bool = True) -> List[Tuple[str, Optional[Dict]]]:
        windows: List[Tuple[str, Optional[Dict]]] = [("Tela Completa", None)]

        # Monitores disponíveis via mss
        try:
            with mss.mss() as sct:
                for i, monitor in enumerate(sct.monitors[1:], 1):
                    windows.append((
                        f"Monitor {i} ({monitor['width']}x{monitor['height']})",
                        {"monitor_index": i, "bbox": monitor, "type": "monitor"}
                    ))
        except Exception:
            pass

        # Janelas no Linux via EWMH
        if HAS_EWMH:
            windows.extend(WindowSelector._get_linux_windows(only_current_desktop))

        return windows

    @staticmethod
    def _get_linux_windows(only_current_desktop: bool = True) -> List[Tuple[str, Dict]]:
        windows: List[Tuple[str, Dict]] = []
        if not HAS_EWMH or ewmh is None:
            return windows
        try:
            # Obter lista de clientes (pilha se disponível)
            try:
                clients = ewmh.getClientListStacking() or []
            except Exception:
                clients = ewmh.getClientList() or []

            # Desktop atual
            current_desktop = None
            if only_current_desktop:
                try:
                    current_desktop = ewmh.getCurrentDesktop()
                except Exception:
                    current_desktop = None

            # Root para coordenadas absolutas
            root = None
            try:
                root = ewmh.display.screen().root  # type: ignore[attr-defined]
            except Exception:
                root = None

            # Geometria da tela
            screen_width = None
            screen_height = None
            try:
                if root is not None:
                    g = root.get_geometry()
                    screen_width = int(getattr(g, 'width', 0))
                    screen_height = int(getattr(g, 'height', 0))
            except Exception:
                pass

            seen_ids = set()
            for w in clients:
                try:
                    wid = getattr(w, 'id', None)
                    if wid is None or wid in seen_ids:
                        continue

                    # filtro por desktop
                    if current_desktop is not None:
                        try:
                            wdesk = ewmh.getWmDesktop(w)
                            # wdesk pode ser None ou -1 (sticky)
                            if wdesk not in (None, -1) and wdesk != current_desktop:
                                continue
                        except Exception:
                            pass

                    # estado (ignorar janelas escondidas/minimizadas)
                    try:
                        states = ewmh.getWmState(w, str=True) or []
                        if isinstance(states, (list, tuple)):
                            if any('HIDDEN' in s for s in states):
                                continue
                    except Exception:
                        pass

                    # nome
                    name = None
                    try:
                        name = ewmh.getWmName(w, str=True)
                    except Exception:
                        name = None
                    if not name:
                        # fallback leve via propriedade WM_CLASS
                        try:
                            cls = w.get_wm_class()
                            if cls:
                                name = cls[0] if isinstance(cls, tuple) else str(cls)
                        except Exception:
                            name = None
                    if not name:
                        continue

                    # geometria
                    try:
                        geom = w.get_geometry()
                        if hasattr(w, 'translate_coords') and root is not None:
                            coords = w.translate_coords(root, 0, 0)
                            if isinstance(coords, tuple) and len(coords) >= 3:
                                # (child, x, y)
                                x = int(coords[1])
                                y = int(coords[2])
                            else:
                                x = int(getattr(coords, 'x', 0))
                                y = int(getattr(coords, 'y', 0))
                        else:
                            x = int(getattr(geom, 'x', 0))
                            y = int(getattr(geom, 'y', 0))
                        width = int(getattr(geom, 'width', 0))
                        height = int(getattr(geom, 'height', 0))
                    except Exception:
                        continue

                    if width <= 50 or height <= 50:
                        continue

                    if screen_width is not None and screen_height is not None:
                        if not (x < screen_width and y < screen_height and x + width > 0 and y + height > 0):
                            continue

                    info = {
                        "window_id": wid,
                        "x": max(0, x),
                        "y": max(0, y),
                        "width": width if (screen_width is None) else min(width, max(0, screen_width - x)),
                        "height": height if (screen_height is None) else min(height, max(0, screen_height - y)),
                        "type": "window",
                    }
                    disp_name = f"Janela: {str(name)[:40]}"
                    if len(str(name)) > 40:
                        disp_name += "..."
                    disp_name += f" ({width}x{height})"
                    windows.append((disp_name, info))
                    seen_ids.add(wid)
                except Exception:
                    continue

            windows.sort(key=lambda x: x[0])
            return windows
        except Exception as e:
            print(f"Erro ao listar janelas via EWMH: {e}")
            return []

    @staticmethod
    def verify_window_exists(window_info: Optional[Dict]) -> bool:
        if not window_info:
            return True
        if window_info.get("type") == "monitor":
            return True
        if window_info.get("type") == "window":
            if not HAS_EWMH or ewmh is None:
                # Sem EWMH, não temos como verificar de forma confiável; assumir True para evitar interrupções
                return True
            try:
                clients = ewmh.getClientList() or []
                wid = int(window_info.get("window_id", -1))
                for w in clients:
                    try:
                        if getattr(w, 'id', None) == wid:
                            return True
                    except Exception:
                        continue
                return False
            except Exception:
                return True
        return True

    @staticmethod
    def get_window_region(window_info: Optional[Dict]) -> Dict:
        if window_info is None:
            try:
                with mss.mss() as sct:
                    return sct.monitors[1]
            except Exception:
                return {"left": 0, "top": 0, "width": 1920, "height": 1080}
        if window_info.get("type") == "monitor":
            try:
                with mss.mss() as sct:
                    return sct.monitors[window_info["monitor_index"]]
            except Exception:
                return {"left": 0, "top": 0, "width": 1920, "height": 1080}
        if window_info.get("type") == "window":
            if WindowSelector.verify_window_exists(window_info):
                return {
                    "left": window_info["x"],
                    "top": window_info["y"],
                    "width": window_info["width"],
                    "height": window_info["height"]
                }
            else:
                try:
                    with mss.mss() as sct:
                        return sct.monitors[1]
                except Exception:
                    return {"left": 0, "top": 0, "width": 1920, "height": 1080}
        if "custom_region" in window_info:
            return window_info["custom_region"]
        try:
            with mss.mss() as sct:
                return sct.monitors[1]
        except Exception:
            return {"left": 0, "top": 0, "width": 1920, "height": 1080}


class FastCaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    region_changed = pyqtSignal(dict)

    def __init__(self, target_size: Tuple[int, int] = cfg.DETECT_RESOLUTION, fps_limit: int = cfg.DEFAULT_FPS_LIMIT):
        super().__init__()
        self.target_size = target_size
        self.fps_limit = fps_limit
        self.frame_interval = 1.0 / max(1, int(self.fps_limit))
        self.running = False
        self.sct = None
        self.capture_region: Optional[Dict] = None
        self.window_info: Optional[Dict] = None
        self.mutex = threading.Lock()
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3

    def set_capture_region(self, region: Dict, window_info: Optional[Dict] = None):
        with self.mutex:
            self.capture_region = region
            self.window_info = window_info
            self.consecutive_errors = 0
        # emitir sinal com a região atual (para o DetectionThread saber posicionar mouse)
        try:
            self.region_changed.emit(region)
        except Exception:
            pass

    def set_fps_limit(self, fps: int):
        with self.mutex:
            self.fps_limit = max(1, int(fps))
            self.frame_interval = 1.0 / self.fps_limit

    def set_target_size(self, size: Tuple[int, int]):
        with self.mutex:
            w, h = size
            self.target_size = (max(16, int(w)), max(16, int(h)))

    def run(self):
        self.sct = mss.mss()
        last_time = time.perf_counter()
        last_error_check = time.time()

        try:
            while self.running:
                now = time.perf_counter()
                elapsed = now - last_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                last_time = time.perf_counter()

                with self.mutex:
                    region = self.capture_region if self.capture_region else self.sct.monitors[1]
                    window_info = self.window_info

                try:
                    current_time = time.time()
                    if window_info and current_time - last_error_check > 2.0:
                        last_error_check = current_time
                        if not WindowSelector.verify_window_exists(window_info):
                            raise Exception("Janela não está mais disponível")

                    screenshot = self.sct.grab(region)
                    # Converter para numpy sem cópia extra: usar buffer BGRA e descartar alpha (ficando em BGR)
                    try:
                        buf = np.frombuffer(screenshot.bgra, dtype=np.uint8)
                        frame = buf.reshape(screenshot.height, screenshot.width, 4)[:, :, :3]
                    except Exception:
                        frame = np.array(screenshot, dtype=np.uint8)[:, :, :3]

                    if frame.size == 0:
                        raise ValueError("Frame vazio capturado")

                    # Verificação leve para janela minimizada: amostra 1/16 dos pixels
                    if np.mean(frame[::16, ::16]) < 1:
                        raise ValueError("Janela pode estar minimizada ou em outro workspace")

                    # Redimensionar apenas se necessário; INTER_AREA para downscale, LINEAR para upscaling
                    ih, iw = frame.shape[0], frame.shape[1]
                    tw, th = int(self.target_size[0]), int(self.target_size[1])
                    if iw != tw or ih != th:
                        if tw < iw or th < ih:
                            interp = cv2.INTER_AREA
                        else:
                            interp = cv2.INTER_LINEAR
                        frame = cv2.resize(frame, (tw, th), interpolation=interp)
                    self.frame_ready.emit(frame)

                    self.consecutive_errors = 0

                except Exception as e:
                    self.consecutive_errors += 1
                    if "XGetImage" in str(e) or "não está mais disponível" in str(e):
                        if self.consecutive_errors >= self.max_consecutive_errors:
                            self.error_occurred.emit("Janela não acessível. Voltando para tela completa.")
                            with self.mutex:
                                self.capture_region = None
                                self.window_info = None
                            self.consecutive_errors = 0
                        else:
                            time.sleep(0.1)
                    elif self.consecutive_errors >= self.max_consecutive_errors * 2:
                        self.error_occurred.emit(f"Muitos erros consecutivos ({str(e)}). Voltando para tela completa.")
                        with self.mutex:
                            self.capture_region = None
                            self.window_info = None
                        self.consecutive_errors = 0
                    time.sleep(0.05)
        finally:
            try:
                if self.sct is not None:
                    self.sct.close()
            except Exception:
                pass

    def stop(self):
        self.running = False
        try:
            if self.sct is not None:
                self.sct.close()
        except Exception:
            pass


class DetectionThread(QThread):
    def _iou(self, a: dict, b: dict) -> float:
        try:
            ax1, ay1, ax2, ay2 = float(a['x1']), float(a['y1']), float(a['x2']), float(a['y2'])
            bx1, by1, bx2, by2 = float(b['x1']), float(b['y1']), float(b['x2']), float(b['y2'])
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            aarea = max(0.0, (ax2 - ax1) * (ay2 - ay1))
            barea = max(0.0, (bx2 - bx1) * (by2 - by1))
            union = aarea + barea - inter
            if union <= 0:
                return 0.0
            return inter / union
        except Exception:
            return 0.0

    def _merge_boxes_iou(self, boxes: List[dict], thr: float = 0.3) -> List[dict]:
        if not boxes:
            return []
        boxes = list(boxes)
        merged = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            base = dict(boxes[i])
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if self._iou(base, boxes[j]) > thr:
                    # merge: take min/max coords and max conf
                    base['x1'] = int(min(base['x1'], boxes[j]['x1']))
                    base['y1'] = int(min(base['y1'], boxes[j]['y1']))
                    base['x2'] = int(max(base['x2'], boxes[j]['x2']))
                    base['y2'] = int(max(base['y2'], boxes[j]['y2']))
                    base['conf'] = float(max(base.get('conf', 0.0), boxes[j].get('conf', 0.0)))
                    used[j] = True
            merged.append(base)
        return merged
    detection_ready = pyqtSignal(np.ndarray, list, list)

    def __init__(self, model_path: str = ""):
        super().__init__()
        # Nenhum modelo externo utilizado; parâmetro mantido por compatibilidade.
        self.running = False
        self.frame: Optional[np.ndarray] = None
        self._frame_q = deque(maxlen=1)  # ring buffer tamanho 1
        self.mutex = threading.Lock()
        self.confidence = cfg.DEFAULT_CONFIDENCE
        self.pixel_step = getattr(cfg, 'PIXEL_STEP_DEFAULT', 1)
        # Limite de FPS de inferência para reduzir CPU sem afetar confiança por frame
        self.infer_fps_limit = getattr(cfg, 'INFER_FPS_LIMIT_DEFAULT', 15)
        self._infer_interval = 1.0 / max(1, int(self.infer_fps_limit))
        self._last_infer_time = 0.0
        # Sinalização para evitar busy-loop: aguarda novo frame
        self.frame_event = threading.Event()
        # Imagem dummy mínima: UI de overlay não usa a imagem do sinal
        self._dummy_frame = np.zeros((1, 1, 3), dtype=np.uint8)
        # Controle de mouse
        self.mouse_enabled = False
        self.capture_region: Optional[Dict] = None  # região atual da captura (left, top, width, height)
        # Kernel (uinput)
        self._mouse_device = None
        self.mouse_gain = float(getattr(cfg, 'MOUSE_GAIN', 0.6))
        self.mouse_max_step = int(getattr(cfg, 'MOUSE_MAX_STEP', 15))
        self.mouse_invert_y = bool(getattr(cfg, 'MOUSE_INVERT_Y', False))
        
        # FOV (raio em pixels, centrado na região de captura)
        self.fov_radius = int(getattr(cfg, 'FOV_RADIUS_DEFAULT', 150))
        self.fov_offset_x = 0
        self.fov_offset_y = 0
        # Ajuste de mira (bias fixo em pixels de tela)
        self.aim_x_bias = int(getattr(cfg, 'AIM_X_BIAS_DEFAULT', 0))
        self.aim_y_bias = int(getattr(cfg, 'AIM_Y_BIAS_DEFAULT', 0))
        # Eixos habilitados
        self.axis_x_enabled = True
        self.axis_y_enabled = True
        # Filtro por cor (HSV)
        self.color_filter_enabled = bool(getattr(cfg, 'COLOR_FILTER_ENABLED_DEFAULT', False))
        self.color_hsv = (0, 0, 0)  # (h:0..179, s:0..255, v:0..255)
        self.color_tol_h = int(getattr(cfg, 'COLOR_FILTER_TOL_H_DEFAULT', 15))
        self.color_min_s = int(getattr(cfg, 'COLOR_FILTER_MIN_S_DEFAULT', 30))
        self.color_min_v = int(getattr(cfg, 'COLOR_FILTER_MIN_V_DEFAULT', 30))
        # Suporte a múltiplas faixas HSV (OR de máscaras)
        self.color_multi = []  # lista de dicts: {h,tol_h,s_min,v_min}
        # Preprocessamento opcional (color constancy)
        self.enable_gray_world = bool(int(os.getenv('DETECT_GRAY_WORLD', '0')))
        # Auto calibração de S/V (por cena)
        self.auto_sv = bool(int(os.getenv('DETECT_AUTO_SV', '0')))
        self.auto_sv_interval = float(os.getenv('DETECT_AUTO_SV_INTERVAL', '5'))
        self._last_auto_sv = 0.0
        # Debounce temporal simples (K de M) + tracker por IoU
        self.debounce_k = int(os.getenv('DETECT_DEBOUNCE_K', '3'))
        self.debounce_m = int(os.getenv('DETECT_DEBOUNCE_M', '5'))
        self._tracks = []  # [{id, box, hist: deque, misses_consec, first_ts, last_ts}]
        self._next_track_id = 1
        # FOV adaptativo simples
        self.adaptive_fov = bool(int(os.getenv('DETECT_ADAPTIVE_FOV', '1')))
        self.fov_radius_min = int(os.getenv('DETECT_FOV_MIN', str(max(20, int(getattr(cfg, 'FOV_RADIUS_DEFAULT', 150) * 0.6)))))
        self.fov_radius_max = int(os.getenv('DETECT_FOV_MAX', str(int(getattr(cfg, 'FOV_RADIUS_DEFAULT', 150)))))
        self.fov_lock_frames = int(os.getenv('DETECT_FOV_LOCK_FRAMES', '10'))
        self.fov_release_frames = int(os.getenv('DETECT_FOV_RELEASE_FRAMES', '8'))
        self._frames_with_target = 0
        self._frames_without_target = 0
        # Debug de filtros (reprovações)
        self.debug_filters = bool(int(os.getenv('DETECT_DEBUG_FILTERS', '0')))
        self._last_debug_rejects = []
        # Morfologia (ajustável via aba Train)
        try:
            self.morph_edge_scale = float(getattr(cfg, 'MORPH_EDGE_SCALE_DEFAULT', 1.0))
            self.morph_kscale = float(getattr(cfg, 'MORPH_KSCALE_DEFAULT', 1.0))
            self.morph_dilate_iter = int(getattr(cfg, 'MORPH_DILATE_ITER_DEFAULT', 1))
            self.morph_close_iter = int(getattr(cfg, 'MORPH_CLOSE_ITER_DEFAULT', 1))
            self.morph_open_iter = int(getattr(cfg, 'MORPH_OPEN_ITER_DEFAULT', 1))
        except Exception:
            self.morph_edge_scale = 1.0
            self.morph_kscale = 1.0
            self.morph_dilate_iter = 1
            self.morph_close_iter = 1
            self.morph_open_iter = 1
        # Parâmetros do filtro humanoide (contornos verticais, ajustáveis pela UI)
        try:
            self.filter_min_area = float(getattr(cfg, 'HUMANOID_MIN_AREA_DEFAULT', 900.0))
            self.filter_max_area = float(getattr(cfg, 'HUMANOID_MAX_AREA_DEFAULT', 45000.0))
            self.filter_min_aspect = float(getattr(cfg, 'HUMANOID_MIN_ASPECT_DEFAULT', 1.2))
            self.filter_max_aspect = float(getattr(cfg, 'HUMANOID_MAX_ASPECT_DEFAULT', 6.2))
            self.filter_min_extent = float(getattr(cfg, 'HUMANOID_MIN_EXTENT_DEFAULT', 0.22))
            self.filter_min_solidity = float(getattr(cfg, 'HUMANOID_MIN_SOLIDITY_DEFAULT', 0.28))
            self.filter_min_circularity = float(getattr(cfg, 'HUMANOID_MIN_CIRCULARITY_DEFAULT', 0.05))
            self.filter_max_circularity = float(getattr(cfg, 'HUMANOID_MAX_CIRCULARITY_DEFAULT', 0.55))
            self.filter_max_sym_diff = float(getattr(cfg, 'HUMANOID_MAX_SYM_DIFF_DEFAULT', 0.45))
            self.filter_min_top_ratio = float(getattr(cfg, 'HUMANOID_MIN_TOP_RATIO_DEFAULT', 0.18))
            self.filter_min_bottom_ratio = float(getattr(cfg, 'HUMANOID_MIN_BOTTOM_RATIO_DEFAULT', 0.28))
            self.filter_conf_blend = float(getattr(cfg, 'HUMANOID_CONFIDENCE_BLEND_DEFAULT', 0.6))
        except Exception:
            self.filter_min_area = 900.0
            self.filter_max_area = 45000.0
            self.filter_min_aspect = 1.2
            self.filter_max_aspect = 6.2
            self.filter_min_extent = 0.22
            self.filter_min_solidity = 0.28
            self.filter_min_circularity = 0.05
            self.filter_max_circularity = 0.55
            self.filter_max_sym_diff = 0.45
            self.filter_min_top_ratio = 0.18
            self.filter_min_bottom_ratio = 0.28
            self.filter_conf_blend = 0.6
        self.filter_collect_debug = False
        self._filter_debug_data: List[dict] = []
        # Parâmetros de treino (somente para pré-visualização na aba Train)
        self.train_params = {
            'h': int(self.color_hsv[0]),
            'tol_h': int(self.color_tol_h),
            's_min': int(self.color_min_s),
            'v_min': int(self.color_min_v),
            'edge_scale': float(self.morph_edge_scale),
            'kscale': float(self.morph_kscale),
            'dilate_iter': int(self.morph_dilate_iter),
            'close_iter': int(self.morph_close_iter),
            'open_iter': int(self.morph_open_iter),
        }
        # Dados de benchmark
        self._benchmark_enabled = False
        self._benchmark_data = []  # lista de dicts
        self._benchmark_start_time = None
        self._bench_mutex = threading.Lock()
        # Debounce/estabilidade de caixas para overlay
        self._stable_boxes: List[dict] = []
        # Geometria do desktop virtual (para cálculo de centro)
        self._union_left = 0
        self._union_top = 0
        self._union_w = None
        self._union_h = None
        # Estratégia de seleção de alvo e histerese
        self.target_strategy_distance = bool(getattr(cfg, 'TARGET_STRATEGY_DISTANCE_DEFAULT', True))
        self.target_stick_radius_px = int(getattr(cfg, 'TARGET_STICK_RADIUS_DEFAULT', 120))
        self._last_target_center = None  # (x,y) em coordenadas de tela
        # Aprendizado simples (amostra HSV em positivos/negativos)
        self._learn_pos_hsv: List[np.ndarray] = []
        self._learn_neg_hsv: List[np.ndarray] = []

        # Modo: detectar diretamente das máscaras (igual à pré-visualização)
        # Quando ativo, a lógica usa a máscara de cor integral do frame e
        # promove todos os contornos para caixas, sem filtros heurísticos
        # adicionais. Útil quando a prévia está correta e a detecção não.
        try:
            self.use_preview_mask_detection = bool(int(os.getenv('DETECT_USE_PREVIEW_MASK', '1')))
        except Exception:
            self.use_preview_mask_detection = True
        

    def _maybe_move_mouse(self, results, frame: np.ndarray):
        # Envia eventos relativos via uinput, baseados no delta até o centro da região
        if not self.mouse_enabled or not HAS_UINPUT or self._mouse_device is None:
            return
        with self.mutex:
            region = self.capture_region
            gain = float(self.mouse_gain)
            step_cap = max(1, int(self.mouse_max_step))
            invert_y = bool(self.mouse_invert_y)
            fov_r = int(self.fov_radius)
            ax_en = bool(self.axis_x_enabled)
            ay_en = bool(self.axis_y_enabled)
            use_dist = bool(self.target_strategy_distance)
            stick_r = int(self.target_stick_radius_px)
            last_center = self._last_target_center
        if not region:
            return
        if not ax_en and not ay_en:
            return
        try:
            boxes = getattr(results, 'boxes', None)
            if boxes is None or len(boxes) == 0:
                with self.mutex:
                    self._last_target_center = None
                return
            # Arrays
            xyxy_all = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy().reshape(-1)

            # Escalas frame->tela
            fh, fw = int(frame.shape[0]), int(frame.shape[1])
            sx = float(region['width']) / float(fw)
            sy = float(region['height']) / float(fh)

            # Centro do FOV (região capturada + offset do FOV)
            center_x = float(region['left']) + float(region['width']) / 2.0 + float(self.fov_offset_x)
            center_y = float(region['top']) + float(region['height']) / 2.0 + float(self.fov_offset_y)

            # Preparar candidatos (centros e filtros pelo FOV)
            rr = float(fov_r * fov_r)
            candidates = []  # (idx, center_x_abs, center_y_abs, dist2_to_fov_center, conf)
            for i in range(xyxy_all.shape[0]):
                x1, y1, x2, y2 = xyxy_all[i]
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                abs_cx = float(region['left']) + cx * sx
                abs_cy = float(region['top']) + cy * sy
                ddx = abs_cx - center_x
                ddy = abs_cy - center_y
                dist2 = ddx * ddx + ddy * ddy
                if fov_r > 0 and dist2 > rr:
                    continue
                candidates.append((i, abs_cx, abs_cy, dist2, float(confs[i])))

            if not candidates:
                with self.mutex:
                    self._last_target_center = None
                return

            # Estratégia: distância ao centro do FOV (opção de UI) ou métrica antiga (dist - conf*1e-3)
            if use_dist:
                candidates.sort(key=lambda t: t[3])  # menor dist2 primeiro
            else:
                candidates.sort(key=lambda t: (t[3] - t[4] * 1e-3))

            best_idx, best_abs_cx, best_abs_cy, best_dist2, _ = candidates[0]

            # Histerese: se alvo anterior existir e ainda próximo de alguma detecção, manter
            if last_center is not None and stick_r > 0:
                sxr = float(stick_r)
                stick_r2 = sxr * sxr
                closest_to_last = None  # (i, abs_cx, abs_cy, d2_last)
                lx, ly = float(last_center[0]), float(last_center[1])
                for i, abs_cx, abs_cy, dist2, conf in candidates:
                    dxl = abs_cx - lx
                    dyl = abs_cy - ly
                    d2_last = dxl * dxl + dyl * dyl
                    if d2_last <= stick_r2:
                        if (closest_to_last is None) or (d2_last < closest_to_last[3]):
                            closest_to_last = (i, abs_cx, abs_cy, d2_last)
                if closest_to_last is not None:
                    best_idx = closest_to_last[0]
                    best_abs_cx = closest_to_last[1]
                    best_abs_cy = closest_to_last[2]

            target_x = best_abs_cx
            target_y = best_abs_cy
            # Aplicar ajuste de mira (corrige tendência sistemática)
            target_x += float(self.aim_x_bias)
            target_y += float(self.aim_y_bias)

            dx = (target_x - center_x) * gain
            dy = (target_y - center_y) * gain
            if invert_y:
                dy = -dy

            

            # Desabilitar eixos conforme configuração
            if not ax_en:
                dx = 0.0
            if not ay_en:
                dy = 0.0

            # Enviar em passos pequenos para coerência com aceleração do jogo
            rem_x = dx
            rem_y = dy
            while abs(rem_x) >= 1.0 or abs(rem_y) >= 1.0:
                stepx = int(max(-step_cap, min(step_cap, rem_x))) if ax_en else 0
                stepy = int(max(-step_cap, min(step_cap, rem_y))) if ay_en else 0
                # garantir pelo menos 1 quando ainda resta delta
                if ax_en and stepx == 0 and abs(rem_x) >= 1.0:
                    stepx = 1 if rem_x > 0 else -1
                if ay_en and stepy == 0 and abs(rem_y) >= 1.0:
                    stepy = 1 if rem_y > 0 else -1
                try:
                    if stepx != 0 and stepy != 0:
                        # combinar em um único SYN
                        self._mouse_device.emit(uinput.REL_X, stepx, syn=False)
                        self._mouse_device.emit(uinput.REL_Y, stepy)
                    elif stepx != 0:
                        self._mouse_device.emit(uinput.REL_X, stepx)
                    elif stepy != 0:
                        self._mouse_device.emit(uinput.REL_Y, stepy)
                except Exception:
                    break
                rem_x -= stepx
                rem_y -= stepy
            with self.mutex:
                self._last_target_center = (best_abs_cx, best_abs_cy)
        except Exception as e:
            print(f"Falha ao mover mouse: {e}")
            with self.mutex:
                self._last_target_center = None
                
    def _maybe_move_mouse_from_boxes(self, det_boxes: List[dict], frame: np.ndarray):
        """Movimenta o mouse em direção ao melhor alvo com base nas caixas absolutas fornecidas."""
        if not self.mouse_enabled or not HAS_UINPUT or self._mouse_device is None:
            return
        with self.mutex:
            region = self.capture_region
            gain = float(self.mouse_gain)
            step_cap = max(1, int(self.mouse_max_step))
            invert_y = bool(self.mouse_invert_y)
            fov_r = int(self.fov_radius)
            ax_en = bool(self.axis_x_enabled)
            ay_en = bool(self.axis_y_enabled)
            use_dist = bool(self.target_strategy_distance)
            stick_r = int(self.target_stick_radius_px)
            last_center = self._last_target_center
        if not region or not det_boxes:
            return
        if not ax_en and not ay_en:
            return

        try:
            # Centro do FOV (região capturada + offset do FOV)
            center_x = float(region['left']) + float(region['width']) / 2.0 + float(self.fov_offset_x)
            center_y = float(region['top']) + float(region['height']) / 2.0 + float(self.fov_offset_y)
            rr = float(fov_r) * float(fov_r)

            candidates = []  # (abs_cx, abs_cy, dist2, conf)
            for b in det_boxes:
                x1 = float(b.get('x1', 0))
                y1 = float(b.get('y1', 0))
                x2 = float(b.get('x2', 0))
                y2 = float(b.get('y2', 0))
                conf = float(b.get('conf', 0))
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                ddx = cx - center_x
                ddy = cy - center_y
                dist2 = ddx * ddx + ddy * ddy
                if fov_r > 0 and dist2 > rr:
                    continue
                candidates.append((cx, cy, dist2, conf))

            if not candidates:
                with self.mutex:
                    self._last_target_center = None
                return

            if use_dist:
                candidates.sort(key=lambda t: t[2])  # menor distância ao centro
            else:
                candidates.sort(key=lambda t: (t[2] - t[3] * 1e-3))

            best_abs_cx, best_abs_cy, _, _ = candidates[0]

            # Histerese: se alvo anterior existir e ainda próximo de alguma detecção, manter
            if last_center is not None and stick_r > 0:
                sxr = float(stick_r)
                stick_r2 = sxr * sxr
                lx, ly = float(last_center[0]), float(last_center[1])
                closest_to_last = None  # (abs_cx, abs_cy, d2_last)
                for cx, cy, _, _ in candidates:
                    dxl = cx - lx
                    dyl = cy - ly
                    d2_last = dxl * dxl + dyl * dyl
                    if d2_last <= stick_r2:
                        if (closest_to_last is None) or (d2_last < closest_to_last[2]):
                            closest_to_last = (cx, cy, d2_last)
                if closest_to_last is not None:
                    best_abs_cx = closest_to_last[0]
                    best_abs_cy = closest_to_last[1]

            target_x = best_abs_cx + float(self.aim_x_bias)
            target_y = best_abs_cy + float(self.aim_y_bias)
            dx = (target_x - center_x) * gain
            dy = (target_y - center_y) * gain
            if invert_y:
                dy = -dy

            if not ax_en:
                dx = 0.0
            if not ay_en:
                dy = 0.0

            rem_x = dx
            rem_y = dy
            while abs(rem_x) >= 1.0 or abs(rem_y) >= 1.0:
                stepx = int(max(-step_cap, min(step_cap, rem_x))) if ax_en else 0
                stepy = int(max(-step_cap, min(step_cap, rem_y))) if ay_en else 0
                if ax_en and stepx == 0 and abs(rem_x) >= 1.0:
                    stepx = 1 if rem_x > 0 else -1
                if ay_en and stepy == 0 and abs(rem_y) >= 1.0:
                    stepy = 1 if rem_y > 0 else -1
                try:
                    if stepx != 0 and stepy != 0:
                        self._mouse_device.emit(uinput.REL_X, stepx, syn=False)
                        self._mouse_device.emit(uinput.REL_Y, stepy)
                    elif stepx != 0:
                        self._mouse_device.emit(uinput.REL_X, stepx)
                    elif stepy != 0:
                        self._mouse_device.emit(uinput.REL_Y, stepy)
                except Exception:
                    break
                rem_x -= stepx
                rem_y -= stepy
            with self.mutex:
                self._last_target_center = (best_abs_cx, best_abs_cy)

        except Exception:
            with self.mutex:
                self._last_target_center = None

    def _apply_gray_world(self, frame_bgr: np.ndarray) -> np.ndarray:
        try:
            b, g, r = cv2.split(frame_bgr)
            mb = float(np.mean(b)) + 1e-6
            mg = float(np.mean(g)) + 1e-6
            mr = float(np.mean(r)) + 1e-6
            m = (mb + mg + mr) / 3.0
            kb, kg, kr = m / mb, m / mg, m / mr
            b = np.clip(b.astype(np.float32) * kb, 0, 255).astype(np.uint8)
            g = np.clip(g.astype(np.float32) * kg, 0, 255).astype(np.uint8)
            r = np.clip(r.astype(np.float32) * kr, 0, 255).astype(np.uint8)
            return cv2.merge((b, g, r))
        except Exception:
            return frame_bgr

    def _make_color_mask(self, hsv: np.ndarray, params: Optional[Dict] = None, kref: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Cria máscara de cor em HSV com morfologia.
        Se `params` for fornecido, usa esses valores (modo Train);
        caso contrário, usa os parâmetros ao vivo.
        Suporta múltiplas faixas HSV (OR) via self.color_multi.
        """
        fh, fw = hsv.shape[:2]
        if params is None:
            with self.mutex:
                h0, s0, v0 = self.color_hsv
                tol = int(self.color_tol_h)
                s_min = int(self.color_min_s)
                v_min = int(self.color_min_v)
                edge_scale = float(self.morph_edge_scale)
                kscale = float(self.morph_kscale)
                dil_it = int(self.morph_dilate_iter)
                close_it = int(self.morph_close_iter)
                open_it = int(self.morph_open_iter)
                color_multi = list(self.color_multi) if hasattr(self, 'color_multi') else []
                color_filter_enabled = bool(getattr(self, 'color_filter_enabled', True))
        else:
            h0 = int(params.get('h', 0))
            tol = int(params.get('tol_h', 15))
            s_min = int(params.get('s_min', 30))
            v_min = int(params.get('v_min', 30))
            edge_scale = float(params.get('edge_scale', 1.0))
            kscale = float(params.get('kscale', 1.0))
            dil_it = int(params.get('dilate_iter', 1))
            close_it = int(params.get('close_iter', 1))
            open_it = int(params.get('open_iter', 1))
            # No modo Train não usamos color_multi
            color_multi = []
            color_filter_enabled = True

        # Se filtro de cor estiver desativado, retorna máscara cheia
        if not color_filter_enabled:
            return np.full((fh, fw), 255, dtype=np.uint8)

        def _range_mask(hsv_img, h_center, tol_h, smin, vmin):
            h1 = (int(h_center) - int(tol_h)) % 180
            h2 = (int(h_center) + int(tol_h)) % 180
            if h1 <= h2:
                lower = np.array([h1, max(0, int(smin)), max(0, int(vmin))], dtype=np.uint8)
                upper = np.array([h2, 255, 255], dtype=np.uint8)
                return cv2.inRange(hsv_img, lower, upper)
            else:
                lower1 = np.array([0, max(0, int(smin)), max(0, int(vmin))], dtype=np.uint8)
                upper1 = np.array([h2, 255, 255], dtype=np.uint8)
                lower2 = np.array([h1, max(0, int(smin)), max(0, int(vmin))], dtype=np.uint8)
                upper2 = np.array([179, 255, 255], dtype=np.uint8)
                return cv2.inRange(hsv_img, lower1, upper1) | cv2.inRange(hsv_img, lower2, upper2)

        # Máscara base (faixa principal)
        mask = _range_mask(hsv, h0, tol, s_min, v_min)
        # OR com faixas adicionais se existirem
        for rng in (color_multi or []):
            try:
                hh = int(rng.get('h', h0))
                tt = int(rng.get('tol_h', tol))
                ss = int(rng.get('s_min', s_min))
                vv = int(rng.get('v_min', v_min))
                mask |= _range_mask(hsv, hh, tt, ss, vv)
            except Exception:
                continue

        # Morfologia configurável
        # Para manter a coerência entre a pré-visualização (frame inteiro)
        # e a etapa fina por ROI, permitimos informar um tamanho de referência
        # (kref) para o cálculo de kernels. Assim, ao processar ROIs, usamos
        # o tamanho do frame completo como referência para os kernels.
        ref_w, ref_h = (fw, fh) if (kref is None) else (int(kref[0]), int(kref[1]))
        ref_min = max(1, int(min(ref_w, ref_h)))

        k_edge_base = max(1, int(ref_min / 300))
        k_edge = max(1, int(round(k_edge_base * edge_scale)))
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_edge*2 + 1, k_edge*2 + 1))
        if dil_it > 0:
            mask = cv2.dilate(mask, kernel_edge, iterations=dil_it)

        k_base = max(2, int(ref_min / 120))
        k = max(2, int(round(k_base * kscale)))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        if close_it > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_it)
        if open_it > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_it)
        return mask

    def _detect_humanoids_opencv(self, frame_bgr: np.ndarray, region: Dict, conf_thresh: float):
        """Detecção baseada em OpenCV (coarse->fine + merges + filtros).
        Retorna (det_boxes, timings) onde det_boxes são caixas em coordenadas absolutas
        de tela: x1,y1,x2,y2,label,conf.
        """
        det_boxes: List[dict] = []
        timings = {}
        if frame_bgr is None or frame_bgr.size == 0:
            return det_boxes, timings

        t0 = time.perf_counter()
        try:
            fh, fw = frame_bgr.shape[:2]
            if self.enable_gray_world:
                frame_bgr = self._apply_gray_world(frame_bgr)
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            t_hsv = time.perf_counter()

            # Auto calibração de S/V por cena (medidas leves)
            if self.auto_sv:
                if (time.time() - getattr(self, '_last_auto_sv', 0.0)) > max(0.5, float(getattr(self, 'auto_sv_interval', 5.0))):
                    try:
                        samp = hsv[::8, ::8]
                        v_mean = float(np.mean(samp[:, :, 2]))
                        s_median = float(np.median(samp[:, :, 1]))
                        with self.mutex:
                            vmin = int(self.color_min_v)
                            smin = int(self.color_min_s)
                        if v_mean < 80:
                            vmin = max(0, vmin - 10)
                        elif v_mean > 180:
                            vmin = min(255, vmin + 10)
                        if s_median < 120:
                            smin = max(0, smin - 10)
                        elif s_median > 200:
                            smin = min(255, smin + 10)
                        with self.mutex:
                            self.color_min_v = int(vmin)
                            self.color_min_s = int(smin)
                        self._last_auto_sv = time.time()
                    except Exception:
                        pass

            sx = float(region['width']) / float(fw)
            sy = float(region['height']) / float(fh)
            left = int(region['left'])
            top = int(region['top'])

            # Coarse step: downscale and find ROIs
            coarse_scale = float(os.getenv('DETECT_COARSE_SCALE', '0.5'))
            use_coarse = (0.1 <= coarse_scale < 1.0)
            rois = []  # list of (x1,y1,x2,y2) in frame coords
            if use_coarse:
                cw = max(16, int(fw * coarse_scale))
                ch = max(16, int(fh * coarse_scale))
                hsv_small = cv2.resize(hsv, (cw, ch), interpolation=cv2.INTER_AREA)
                mask_small = self._make_color_mask(hsv_small)
                contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    scale_inv = 1.0 / coarse_scale
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # Expand ROI by 1.6x around center and map to full-res
                        cx = x + w / 2.0
                        cy = y + h / 2.0
                        ex = w * 1.6
                        ey = h * 1.6
                        rx1 = int(max(0, (cx - ex / 2.0) * scale_inv))
                        ry1 = int(max(0, (cy - ey / 2.0) * scale_inv))
                        rx2 = int(min(fw, (cx + ex / 2.0) * scale_inv))
                        ry2 = int(min(fh, (cy + ey / 2.0) * scale_inv))
                        if rx2 - rx1 > 4 and ry2 - ry1 > 4:
                            rois.append((rx1, ry1, rx2, ry2))
                t_coarse = time.perf_counter()
                timings['coarse_ms'] = (t_coarse - t_hsv) * 1000.0
            else:
                t_coarse = t_hsv

            if not rois or len(rois) > 32:
                # Fallback: process whole frame
                mask = self._make_color_mask(hsv)
                rois = [(0, 0, fw, fh)]
            else:
                mask = None  # ROI-specific masks will be computed
            t_mask = time.perf_counter()
            timings['mask_ms'] = (t_mask - t_coarse) * 1000.0

            rejects = []
            filter_debug: List[dict] = []
            with self.mutex:
                min_area = float(self.filter_min_area)
                max_area = float(self.filter_max_area)
                min_aspect = float(self.filter_min_aspect)
                max_aspect = float(self.filter_max_aspect)
                min_extent = float(self.filter_min_extent)
                min_solidity = float(self.filter_min_solidity)
                min_circ = float(self.filter_min_circularity)
                max_circ = float(self.filter_max_circularity)
                max_sym = float(self.filter_max_sym_diff)
                min_top = float(self.filter_min_top_ratio)
                min_bottom = float(self.filter_min_bottom_ratio)
                blend_bias = float(self.filter_conf_blend)

            def _norm(val: float, vmin: float, vmax: float) -> float:
                if vmax <= vmin:
                    return 0.0
                return max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))

            # Refine within each ROI
            for (rx1, ry1, rx2, ry2) in rois:
                if rx2 <= rx1 or ry2 <= ry1:
                    continue
                if mask is None:
                    roi_hsv = hsv[ry1:ry2, rx1:rx2]
                    # Use o frame completo como referência de kernel
                    roi_mask = self._make_color_mask(roi_hsv, kref=(fw, fh))
                else:
                    roi_mask = mask[ry1:ry2, rx1:rx2]
                    roi_hsv = hsv[ry1:ry2, rx1:rx2]
                if roi_mask is None or roi_mask.size == 0:
                    continue
                contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours or []:
                    area = float(cv2.contourArea(cnt))
                    if area <= 1:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w <= 0 or h <= 0:
                        continue
                    rect_area = float(w * h)
                    extent = area / rect_area if rect_area > 1.0 else 0.0
                    try:
                        hull = cv2.convexHull(cnt)
                        hull_area = float(cv2.contourArea(hull))
                        solidity = (area / hull_area) if hull_area > 1.0 else 0.0
                    except Exception:
                        solidity = 0.0

                    aspect = float(h) / float(max(1, w))

                    cnt_mask = roi_mask[y:y + h, x:x + w]
                    roi_mask_bool = (cnt_mask > 0)
                    pixels_on = float(np.count_nonzero(roi_mask_bool))
                    debug_entry = {
                        'roi_box': (rx1 + x, ry1 + y, rx1 + x + w, ry1 + y + h),
                        'area': area,
                        'aspect': aspect,
                        'extent': extent,
                        'solidity': solidity,
                        'accepted': False,
                    }

                    if area < min_area or area > max_area:
                        debug_entry['reason'] = 'area'
                        if self.filter_collect_debug:
                            filter_debug.append(debug_entry)
                        continue
                    if aspect < min_aspect or aspect > max_aspect:
                        debug_entry['reason'] = 'aspect'
                        if self.filter_collect_debug:
                            filter_debug.append(debug_entry)
                        continue
                    if extent < min_extent:
                        debug_entry['reason'] = 'extent'
                        if self.filter_collect_debug:
                            filter_debug.append(debug_entry)
                        continue
                    if solidity < min_solidity:
                        debug_entry['reason'] = 'solidity'
                        if self.filter_collect_debug:
                            filter_debug.append(debug_entry)
                        continue

                    perimeter = float(cv2.arcLength(cnt, True))
                    circularity = 0.0
                    if perimeter > 1.0:
                        circularity = float((4.0 * math.pi * area) / (perimeter * perimeter))
                    debug_entry['circularity'] = circularity
                    if (min_circ > 0.0 and circularity < min_circ) or (max_circ > 0.0 and circularity > max_circ):
                        debug_entry['reason'] = 'circularity'
                        if self.filter_collect_debug:
                            filter_debug.append(debug_entry)
                        continue

                    if pixels_on < 5:
                        debug_entry['reason'] = 'pixels'
                        if self.filter_collect_debug:
                            filter_debug.append(debug_entry)
                        continue

                    cols_sum = roi_mask_bool.sum(axis=0).astype(np.float32)
                    half = max(1, int(cols_sum.size / 2))
                    left_mass = float(np.sum(cols_sum[:half]))
                    right_mass = float(np.sum(cols_sum[-half:]))
                    total_lr = left_mass + right_mass
                    sym_diff = float(abs(left_mass - right_mass) / total_lr) if total_lr > 0 else 0.0
                    debug_entry['sym_diff'] = sym_diff
                    if max_sym > 0.0 and sym_diff > max_sym:
                        debug_entry['reason'] = 'symmetry'
                        if self.filter_collect_debug:
                            filter_debug.append(debug_entry)
                        continue

                    rows_sum = roi_mask_bool.sum(axis=1).astype(np.float32)
                    total_rows = float(np.sum(rows_sum)) if np.sum(rows_sum) > 0 else 1.0
                    third = max(1, int(rows_sum.size / 3))
                    top_ratio = float(np.sum(rows_sum[:third]) / total_rows)
                    bottom_ratio = float(np.sum(rows_sum[-third:]) / total_rows)
                    debug_entry['top_ratio'] = top_ratio
                    debug_entry['bottom_ratio'] = bottom_ratio
                    if top_ratio < min_top:
                        debug_entry['reason'] = 'top'
                        if self.filter_collect_debug:
                            filter_debug.append(debug_entry)
                        continue
                    if bottom_ratio < min_bottom:
                        debug_entry['reason'] = 'bottom'
                        if self.filter_collect_debug:
                            filter_debug.append(debug_entry)
                        continue

                    sat_mean = 0.0
                    try:
                        sat_vals = roi_hsv[ry1 + y:ry1 + y + h, rx1 + x:rx1 + x + w, 1]
                        sat_sel = sat_vals[roi_mask_bool]
                        if sat_sel.size > 0:
                            sat_mean = float(np.mean(sat_sel)) / 255.0
                    except Exception:
                        pass

                    area_score = _norm(area, min_area, max_area)
                    solidity_score = _norm(solidity, min_solidity, 1.0)
                    extent_score = _norm(extent, min_extent, 1.0)
                    aspect_mid = (min_aspect + max_aspect) / 2.0
                    aspect_half = max(1e-3, (max_aspect - min_aspect) / 2.0)
                    aspect_score = max(0.0, 1.0 - abs(aspect - aspect_mid) / aspect_half)
                    circ_score = _norm(circularity, min_circ, max_circ if max_circ > min_circ else min_circ + 1.0)
                    sym_score = max(0.0, 1.0 - (sym_diff / max_sym if max_sym > 0 else 0.0))
                    top_score = _norm(top_ratio, min_top, min(0.6, min_top + 0.4))
                    bottom_score = _norm(bottom_ratio, min_bottom, min(0.75, min_bottom + 0.35))
                    sat_score = max(0.0, min(1.0, sat_mean))

                    conf_geom = (
                        0.18 * area_score + 0.15 * solidity_score + 0.12 * extent_score +
                        0.12 * aspect_score + 0.07 * circ_score + 0.12 * sym_score +
                        0.07 * top_score + 0.07 * bottom_score + 0.10 * sat_score
                    )
                    conf = float(blend_bias * conf_geom + (1.0 - blend_bias) * sat_score)

                    if conf < conf_thresh:
                        debug_entry['reason'] = 'confidence'
                        debug_entry['conf'] = conf
                        if self.debug_filters:
                            rejects.append({'reason': 'conf', 'val': conf, 'thr': conf_thresh})
                        if self.filter_collect_debug:
                            filter_debug.append(debug_entry)
                        continue

                    ax1 = int(left + (rx1 + x) * sx)
                    ay1 = int(top + (ry1 + y) * sy)
                    ax2 = int(left + (rx1 + x + w) * sx)
                    ay2 = int(top + (ry1 + y + h) * sy)

                    box = {
                        'x1': ax1,
                        'y1': ay1,
                        'x2': ax2,
                        'y2': ay2,
                        'label': 'humanoid_red',
                        'conf': float(conf),
                        'meta': {
                            'area': area,
                            'aspect': aspect,
                            'extent': extent,
                            'solidity': solidity,
                            'circularity': circularity,
                            'sym_diff': sym_diff,
                            'top_ratio': top_ratio,
                            'bottom_ratio': bottom_ratio,
                            'sat_mean': sat_mean,
                        }
                    }
                    det_boxes.append(box)
                    if self.filter_collect_debug:
                        dbg_box = dict(debug_entry)
                        dbg_box['conf'] = conf
                        dbg_box['accepted'] = True
                        dbg_box['abs_box'] = (ax1, ay1, ax2, ay2)
                        filter_debug.append(dbg_box)

            if self.debug_filters:
                try:
                    self._last_debug_rejects = rejects[-200:]
                except Exception:
                    pass

            if self.filter_collect_debug:
                try:
                    for entry in filter_debug:
                        if 'abs_box' not in entry:
                            rb = entry.get('roi_box')
                            if rb is not None:
                                ax1 = int(left + rb[0] * sx)
                                ay1 = int(top + rb[1] * sy)
                                ax2 = int(left + rb[2] * sx)
                                ay2 = int(top + rb[3] * sy)
                                entry['abs_box'] = (ax1, ay1, ax2, ay2)
                    self._filter_debug_data = filter_debug
                except Exception:
                    self._filter_debug_data = []

            # Merge overlapping boxes (IoU)
            det_boxes = self._merge_boxes_iou(det_boxes, thr=0.3)

            # Ordenar por confiança desc
            det_boxes.sort(key=lambda b: b.get('conf', 0.0), reverse=True)
            timings['total_ms'] = (time.perf_counter() - t0) * 1000.0
            return det_boxes, timings
        except Exception:
            timings['total_ms'] = (time.perf_counter() - t0) * 1000.0
            return det_boxes, timings

    def _detect_from_mask_fullframe(self, frame_bgr: np.ndarray, region: Dict) -> Tuple[List[dict], dict]:
        """Detecção direta a partir da máscara no frame completo, para igualar a prévia.
        Promove todos os contornos a caixas (apenas filtro mínimo de tamanho).
        """
        det_boxes: List[dict] = []
        timings: Dict[str, float] = {}
        if frame_bgr is None or frame_bgr.size == 0:
            return det_boxes, timings
        t0 = time.perf_counter()
        try:
            fh, fw = frame_bgr.shape[:2]
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            mask = self._make_color_mask(hsv)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sx = float(region['width']) / float(fw)
            sy = float(region['height']) / float(fh)
            left = int(region['left'])
            top = int(region['top'])
            for cnt in contours or []:
                x, y, w, h = cv2.boundingRect(cnt)
                if w <= 1 or h <= 1:
                    continue
                ax1 = int(left + x * sx)
                ay1 = int(top + y * sy)
                ax2 = int(left + (x + w) * sx)
                ay2 = int(top + (y + h) * sy)
                # Confiança fixa elevada para ignorar threshold; rótulo igual ao restante
                det_boxes.append({'x1': ax1, 'y1': ay1, 'x2': ax2, 'y2': ay2,
                                  'label': 'humanoid_red', 'conf': 0.99})
            # Merge caixas sobrepostas para reduzir ruído
            det_boxes = self._merge_boxes_iou(det_boxes, thr=0.3)
            det_boxes.sort(key=lambda b: b.get('conf', 0.0), reverse=True)
            timings['total_ms'] = (time.perf_counter() - t0) * 1000.0
            return det_boxes, timings
        except Exception:
            timings['total_ms'] = (time.perf_counter() - t0) * 1000.0
            return det_boxes, timings


    def set_fov_radius(self, radius_px: int):
        with self.mutex:
            self.fov_radius = max(0, int(radius_px))

    def set_fov_offset(self, dx: int, dy: int):
        with self.mutex:
            self.fov_offset_x = int(dx)
            self.fov_offset_y = int(dy)

    def set_aim_bias(self, bx: int, by: int):
        with self.mutex:
            self.aim_x_bias = int(bx)
            self.aim_y_bias = int(by)

    def set_target_strategy_distance(self, enabled: bool):
        with self.mutex:
            self.target_strategy_distance = bool(enabled)

    def set_target_stick_radius(self, radius_px: int):
        with self.mutex:
            try:
                r = int(radius_px)
            except Exception:
                r = self.target_stick_radius_px
            self.target_stick_radius_px = max(0, min(10000, r))

    

    def set_axis_x_enabled(self, enabled: bool):
        with self.mutex:
            self.axis_x_enabled = bool(enabled)

    def set_axis_y_enabled(self, enabled: bool):
        with self.mutex:
            self.axis_y_enabled = bool(enabled)

    def set_mouse_gain(self, gain: float):
        with self.mutex:
            try:
                g = float(gain)
            except Exception:
                g = self.mouse_gain
            # Limitar a uma faixa razoável
            self.mouse_gain = max(0.01, min(10.0, g))

    def set_mouse_max_step(self, step: int):
        with self.mutex:
            try:
                s = int(step)
            except Exception:
                s = self.mouse_max_step
            self.mouse_max_step = max(1, min(200, s))

    # ----- Filtro por cor (HSV) -----
    def set_color_filter_enabled(self, enabled: bool):
        with self.mutex:
            self.color_filter_enabled = bool(enabled)

    def set_color_target_hsv(self, h: int, s: int, v: int):
        with self.mutex:
            self.color_hsv = (
                max(0, min(179, int(h))),
                max(0, min(255, int(s))),
                max(0, min(255, int(v)))
            )

    def set_color_params(self, tol_h: int, min_s: int, min_v: int):
        with self.mutex:
            self.color_tol_h = max(0, min(90, int(tol_h)))
            self.color_min_s = max(0, min(255, int(min_s)))
            self.color_min_v = max(0, min(255, int(min_v)))

    # ----- Filtro humanoide -----
    def set_humanoid_filter_params(
        self,
        min_area: Optional[float] = None,
        max_area: Optional[float] = None,
        min_aspect: Optional[float] = None,
        max_aspect: Optional[float] = None,
        min_extent: Optional[float] = None,
        min_solidity: Optional[float] = None,
        min_circularity: Optional[float] = None,
        max_circularity: Optional[float] = None,
        max_sym_diff: Optional[float] = None,
        min_top_ratio: Optional[float] = None,
        min_bottom_ratio: Optional[float] = None,
        conf_blend: Optional[float] = None,
    ):
        with self.mutex:
            if min_area is not None:
                try:
                    self.filter_min_area = max(1.0, float(min_area))
                except Exception:
                    pass
            if max_area is not None:
                try:
                    self.filter_max_area = max(self.filter_min_area + 1.0, float(max_area))
                except Exception:
                    pass
            if min_aspect is not None:
                try:
                    self.filter_min_aspect = max(0.1, float(min_aspect))
                except Exception:
                    pass
            if max_aspect is not None:
                try:
                    self.filter_max_aspect = max(self.filter_min_aspect + 0.1, float(max_aspect))
                except Exception:
                    pass
            if min_extent is not None:
                try:
                    self.filter_min_extent = max(0.0, min(1.0, float(min_extent)))
                except Exception:
                    pass
            if min_solidity is not None:
                try:
                    self.filter_min_solidity = max(0.0, min(1.0, float(min_solidity)))
                except Exception:
                    pass
            if min_circularity is not None:
                try:
                    self.filter_min_circularity = max(0.0, min(1.0, float(min_circularity)))
                except Exception:
                    pass
            if max_circularity is not None:
                try:
                    self.filter_max_circularity = max(self.filter_min_circularity + 1e-3, float(max_circularity))
                except Exception:
                    pass
            if max_sym_diff is not None:
                try:
                    self.filter_max_sym_diff = max(0.0, min(1.0, float(max_sym_diff)))
                except Exception:
                    pass
            if min_top_ratio is not None:
                try:
                    self.filter_min_top_ratio = max(0.0, min(0.8, float(min_top_ratio)))
                except Exception:
                    pass
            if min_bottom_ratio is not None:
                try:
                    self.filter_min_bottom_ratio = max(0.0, min(0.9, float(min_bottom_ratio)))
                except Exception:
                    pass
            if conf_blend is not None:
                try:
                    self.filter_conf_blend = max(0.0, min(1.0, float(conf_blend)))
                except Exception:
                    pass

    def set_preview_mask_detection(self, enabled: bool):
        with self.mutex:
            self.use_preview_mask_detection = bool(enabled)

    def get_last_filter_debug(self) -> List[dict]:
        try:
            return [dict(item) for item in getattr(self, '_filter_debug_data', [])]
        except Exception:
            return []

    def debug_filter_preview(self, frame_bgr: np.ndarray, region: Dict) -> dict:
        if frame_bgr is None or frame_bgr.size == 0:
            return {}
        try:
            h, w = frame_bgr.shape[:2]
            if region is None:
                region = {'left': 0, 'top': 0, 'width': w, 'height': h}
            else:
                region = {
                    'left': int(region.get('left', 0)),
                    'top': int(region.get('top', 0)),
                    'width': int(region.get('width', w)),
                    'height': int(region.get('height', h)),
                }
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            mask = self._make_color_mask(hsv)
            overlay = frame_bgr.copy()
            try:
                self.filter_collect_debug = True
                det_boxes, _ = self._detect_humanoids_opencv(frame_bgr, region, float(self.confidence))
            finally:
                self.filter_collect_debug = False
            debug_entries = self.get_last_filter_debug()
            off_x = int(region.get('left', 0))
            off_y = int(region.get('top', 0))
            for entry in debug_entries:
                abs_box = entry.get('abs_box')
                if abs_box is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in abs_box]
                rel_box = (x1 - off_x, y1 - off_y, x2 - off_x, y2 - off_y)
                color = (0, 200, 0) if entry.get('accepted') else (0, 0, 200)
                cv2.rectangle(overlay, (rel_box[0], rel_box[1]), (rel_box[2], rel_box[3]), color, 2)
                try:
                    if entry.get('accepted'):
                        conf_val = entry.get('conf', 0.0)
                        label = f"{conf_val:.2f}"
                    else:
                        label = str(entry.get('reason', ''))
                    if label:
                        cv2.putText(overlay, label, (rel_box[0] + 2, rel_box[1] + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                except Exception:
                    pass
            return {
                'mask': mask,
                'overlay': overlay,
                'boxes': det_boxes,
                'debug': debug_entries,
            }
        except Exception as e:
            return {'error': str(e)}

    # ----- Extensões de cor e calibração -----
    def set_multi_color_ranges(self, ranges: List[dict]):
        """Define múltiplas faixas HSV adicionais (OR) além da faixa principal.
        Cada item do array deve conter chaves: h, tol_h, s_min, v_min.
        Ex.: [{'h':30,'tol_h':10,'s_min':40,'v_min':40}, ...]
        """
        try:
            with self.mutex:
                self.color_multi = [
                    {
                        'h': int(r.get('h', 0)),
                        'tol_h': int(r.get('tol_h', 15)),
                        's_min': int(r.get('s_min', 30)),
                        'v_min': int(r.get('v_min', 30)),
                    }
                    for r in (ranges or []) if isinstance(r, dict)
                ]
        except Exception:
            pass

    def set_auto_sv_enabled(self, enabled: bool, interval_s: Optional[float] = None):
        try:
            with self.mutex:
                self.auto_sv = bool(enabled)
                if interval_s is not None:
                    self.auto_sv_interval = float(interval_s)
        except Exception:
            pass

    def get_last_filter_rejects(self) -> List[dict]:
        try:
            return list(getattr(self, '_last_debug_rejects', []))
        except Exception:
            return []

    def _update_tracks(self, det_boxes: List[dict]) -> List[dict]:
        """Atualiza trilhas via associação por IoU e aplica debounce K de M.
        Retorna apenas caixas consideradas estáveis.
        """
        iou_thr = float(os.getenv('DETECT_TRACK_IOU', '0.3'))
        max_misses = int(os.getenv('DETECT_TRACK_MAX_MISSES', '2'))
        K = int(getattr(self, 'debounce_k', 3))
        M = int(getattr(self, 'debounce_m', 5))
        now_ts = time.time()

        tracks = getattr(self, '_tracks', [])
        used_tracks = set()
        used_dets = set()

        # Pré-calcular IoU matrix (lista esparsa)
        ious = []  # list of (iou, ti, di)
        for ti, tr in enumerate(tracks):
            tb = tr.get('box') or {}
            for di, db in enumerate(det_boxes or []):
                try:
                    iou = self._iou(tb, db)
                except Exception:
                    iou = 0.0
                if iou >= iou_thr:
                    ious.append((iou, ti, di))
        ious.sort(key=lambda t: t[0], reverse=True)

        # Inicialização do histórico
        for tr in tracks:
            if 'hist' not in tr or not isinstance(tr['hist'], deque):
                tr['hist'] = deque(maxlen=M)

        # Associação greedy por maior IoU
        for iou, ti, di in ious:
            if ti in used_tracks or di in used_dets:
                continue
            tr = tracks[ti]
            det = det_boxes[di]
            tr['box'] = det
            tr['misses_consec'] = 0
            tr['last_ts'] = now_ts
            tr.setdefault('first_ts', now_ts)
            tr['hist'].append(1)
            prev_conf = float(tr.get('conf', 0.0))
            tr['conf'] = 0.7 * prev_conf + 0.3 * float(det.get('conf', 0.0))
            tr['label'] = det.get('label', 'humanoid_red')
            used_tracks.add(ti)
            used_dets.add(di)

        # Criar trilhas para detecções novas
        for di, det in enumerate(det_boxes or []):
            if di in used_dets:
                continue
            tr = {
                'id': int(getattr(self, '_next_track_id', 1)),
                'box': det,
                'misses_consec': 0,
                'first_ts': now_ts,
                'last_ts': now_ts,
                'hist': deque([1], maxlen=M),
                'conf': float(det.get('conf', 0.0)),
                'label': det.get('label', 'humanoid_red'),
            }
            tracks.append(tr)
            self._next_track_id = int(tr['id']) + 1

        # Atualizar trilhas não associadas
        survivors = []
        for idx, tr in enumerate(tracks):
            if idx in used_tracks:
                survivors.append(tr)
                continue
            tr['misses_consec'] = int(tr.get('misses_consec', 0)) + 1
            tr['hist'].append(0)
            if tr['misses_consec'] <= max_misses:
                survivors.append(tr)
        self._tracks = survivors

        # Selecionar caixas estáveis (K de M)
        stable = []
        for tr in self._tracks:
            try:
                hits = sum(list(tr.get('hist', [])))
            except Exception:
                hits = 0
            if hits >= K:
                b = dict(tr.get('box', {}))
                if b:
                    b['conf'] = float(tr.get('conf', b.get('conf', 0.0)))
                    b['label'] = tr.get('label', b.get('label', 'humanoid_red'))
                    stable.append(b)
        return stable

    # ----- Parâmetros de treino (não alteram o detector ao vivo) -----
    def set_train_color_target_hsv(self, h: int, s: int, v: int):
        with self.mutex:
            self.train_params['h'] = max(0, min(179, int(h)))

    def set_train_color_params(self, tol_h: int, min_s: int, min_v: int):
        with self.mutex:
            self.train_params['tol_h'] = max(0, min(90, int(tol_h)))
            self.train_params['s_min'] = max(0, min(255, int(min_s)))
            self.train_params['v_min'] = max(0, min(255, int(min_v)))

    # ----- Morfologia (ajustes para o modo Train) -----
    def set_morph_params(self,
                         edge_scale: Optional[float] = None,
                         kscale: Optional[float] = None,
                         dilate_iter: Optional[int] = None,
                         close_iter: Optional[int] = None,
                         open_iter: Optional[int] = None):
        with self.mutex:
            if edge_scale is not None:
                try:
                    self.morph_edge_scale = float(edge_scale)
                except Exception:
                    pass
            if kscale is not None:
                try:
                    self.morph_kscale = float(kscale)
                except Exception:
                    pass
            if dilate_iter is not None:
                try:
                    self.morph_dilate_iter = max(0, int(dilate_iter))
                except Exception:
                    pass
            if close_iter is not None:
                try:
                    self.morph_close_iter = max(0, int(close_iter))
                except Exception:
                    pass
            if open_iter is not None:
                try:
                    self.morph_open_iter = max(0, int(open_iter))
                except Exception:
                    pass

    def set_train_morph_params(self,
                               edge_scale: Optional[float] = None,
                               kscale: Optional[float] = None,
                               dilate_iter: Optional[int] = None,
                               close_iter: Optional[int] = None,
                               open_iter: Optional[int] = None):
        with self.mutex:
            if edge_scale is not None:
                try:
                    self.train_params['edge_scale'] = float(edge_scale)
                except Exception:
                    pass
            if kscale is not None:
                try:
                    self.train_params['kscale'] = float(kscale)
                except Exception:
                    pass
            if dilate_iter is not None:
                try:
                    self.train_params['dilate_iter'] = max(0, int(dilate_iter))
                except Exception:
                    pass
            if close_iter is not None:
                try:
                    self.train_params['close_iter'] = max(0, int(close_iter))
                except Exception:
                    pass
            if open_iter is not None:
                try:
                    self.train_params['open_iter'] = max(0, int(open_iter))
                except Exception:
                    pass

    # ----- Aprendizado simples (HSV) -----
    def _sample_hsv_from_mask(self, hsv_img: np.ndarray, mask: np.ndarray, max_samples: int = 5000) -> np.ndarray:
        idx = np.where(mask > 0)
        if idx[0].size == 0:
            return np.empty((0, 3), dtype=np.uint8)
        n = idx[0].size
        if n > max_samples:
            sel = np.random.choice(n, size=max_samples, replace=False)
            return hsv_img[idx[0][sel], idx[1][sel]].astype(np.uint8)
        return hsv_img[idx].astype(np.uint8)

    def learn_from_image(self, frame_bgr: np.ndarray, region: Dict, approved: bool) -> dict:
        try:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            with self.mutex:
                params = dict(self.train_params)
            mask = self._make_color_mask(hsv, params=params)
            samples = self._sample_hsv_from_mask(hsv, mask)
            if samples.size == 0:
                return {"ok": False, "reason": "sem pixels na máscara"}
            with self.mutex:
                if approved:
                    self._learn_pos_hsv.append(samples)
                else:
                    self._learn_neg_hsv.append(samples)
            return {"ok": True, "n": int(samples.shape[0])}
        except Exception as e:
            return {"ok": False, "reason": str(e)}

    def recommend_params(self) -> dict:
        with self.mutex:
            pos_list = list(self._learn_pos_hsv)
        if not pos_list:
            return {}
        arr = np.concatenate(pos_list, axis=0)
        if arr.size == 0:
            return {}
        # média circular do hue (0..179)
        h = arr[:, 0].astype(np.float32)
        ang = h / 180.0 * (2.0 * np.pi)
        mean_cos = float(np.mean(np.cos(ang)))
        mean_sin = float(np.mean(np.sin(ang)))
        mean_ang = np.arctan2(mean_sin, mean_cos)
        if mean_ang < 0:
            mean_ang += 2.0 * np.pi
        h_center = int(round((mean_ang / (2.0 * np.pi)) * 180.0)) % 180
        R = np.sqrt(mean_cos**2 + mean_sin**2)
        circ_std = np.sqrt(max(0.0, -2.0 * np.log(max(1e-6, R))))  # radianos
        tol_h = int(round(min(90, max(3, (circ_std * 180.0 / np.pi)))))
        s_min = int(np.percentile(arr[:, 1], 20))
        v_min = int(np.percentile(arr[:, 2], 20))
        return {"h": h_center, "tol_h": tol_h, "s_min": s_min, "v_min": v_min}

    def apply_learned_params(self) -> dict:
        rec = self.recommend_params()
        if not rec:
            return {"ok": False}
        self.set_color_target_hsv(rec["h"], max(0, rec["s_min"]), max(0, rec["v_min"]))
        self.set_color_params(rec["tol_h"], rec["s_min"], rec["v_min"])
        return {"ok": True, **rec}

    def reset_learning(self):
        with self.mutex:
            self._learn_pos_hsv = []
            self._learn_neg_hsv = []
    

    def set_frame(self, frame: np.ndarray):
        try:
            self._frame_q.clear()
            self._frame_q.append(frame.copy())
        except Exception:
            pass
        try:
            self.frame_event.set()
        except Exception:
            pass

    def run(self):
        """Main inference loop.
        Waits for frames from the capture thread, throttles inference to
        `infer_fps_limit`, runs OpenCV-based detection, and emits screen-space boxes.
        All exceptions are caught to avoid propagating through Qt.
        """
        self.running = True
        self._last_infer_time = time.perf_counter()

        while self.running:
            try:
                # Wait for a new frame (wake-up signal) with a short timeout
                # so we can also react to stop() quickly.
                self.frame_event.wait(0.1)
                if not self.running:
                    break

                # Throttle inference FPS
                now = time.perf_counter()
                if (now - self._last_infer_time) < self._infer_interval:
                    continue
                self._last_infer_time = now

                # Grab the most recent frame and current params
                try:
                    frame = self._frame_q[-1]
                except Exception:
                    frame = None
                with self.mutex:
                    confidence = float(self.confidence)
                    region = self.capture_region
                # Reset the event so we wait for the next frame
                try:
                    self.frame_event.clear()
                except Exception:
                    pass

                if frame is None or frame.size == 0:
                    # Nothing to do yet; emit empty result to keep UI responsive
                    try:
                        self.detection_ready.emit(self._dummy_frame, [], [])
                    except Exception:
                        pass
                    continue

                # Ensure we know the capture region (for screen-space mapping)
                if region is None:
                    try:
                        with mss.mss() as sct:
                            region = sct.monitors[1]
                    except Exception:
                        h, w = frame.shape[:2]
                        region = {"left": 0, "top": 0, "width": w, "height": h}

                # Detecção: modo compatível com a prévia (máscara integral) ou pipeline heurístico
                if getattr(self, 'use_preview_mask_detection', True):
                    det_boxes, timings = self._detect_from_mask_fullframe(frame, region)
                else:
                    det_boxes, timings = self._detect_humanoids_opencv(frame, region, confidence)
                # Atualiza tracker e aplica debounce temporal (K de M)
                stable_boxes = self._update_tracks(det_boxes)

                # Para fins de visualização/log na UI, se ainda não houver
                # caixas estáveis, emita as caixas brutas deste frame.
                # Isso evita a sensação de "não detecta" quando a máscara
                # já enxerga os alvos mas o debounce temporal ainda não confirmou.
                emit_boxes = stable_boxes if stable_boxes else det_boxes
                det_labels: List[str] = [b.get('label', 'humanoid_red') for b in emit_boxes]

                # FOV adaptativo: estreita quando mantém alvo por N frames; alarga após M perdas
                if self.adaptive_fov:
                    if stable_boxes:
                        self._frames_with_target += 1
                        self._frames_without_target = 0
                        if self._frames_with_target >= int(self.fov_lock_frames):
                            try:
                                with self.mutex:
                                    cur = int(self.fov_radius)
                                    mn = int(self.fov_radius_min)
                                if cur != mn:
                                    self.set_fov_radius(mn)
                            except Exception:
                                pass
                    else:
                        self._frames_with_target = 0
                        self._frames_without_target += 1
                        if self._frames_without_target >= int(self.fov_release_frames):
                            try:
                                with self.mutex:
                                    cur = int(self.fov_radius)
                                    mx = int(self.fov_radius_max)
                                if cur != mx:
                                    self.set_fov_radius(mx)
                            except Exception:
                                pass

                # Movimento do mouse baseado nas detecções estáveis
                try:
                    self._maybe_move_mouse_from_boxes(stable_boxes, frame)
                except Exception:
                    pass

                # Benchmark collection (com estágios)
                if self._benchmark_enabled:
                    try:
                        infer_ms = float(timings.get('total_ms', (time.perf_counter() - now) * 1000.0))
                        coarse_ms = float(timings.get('coarse_ms', 0.0))
                        mask_ms = float(timings.get('mask_ms', 0.0))
                        avg_conf = None
                        try:
                            if stable_boxes:
                                avg_conf = float(np.mean([float(b.get('conf', 0.0)) for b in stable_boxes]))
                        except Exception:
                            avg_conf = None
                        with self._bench_mutex:
                            self._benchmark_data.append({
                                'ts': time.time(),
                                'img_w': int(frame.shape[1]),
                                'img_h': int(frame.shape[0]),
                                'conf_threshold': confidence,
                                'boxes': len(det_boxes),
                                'stable_boxes': len(stable_boxes),
                                'avg_box_conf': avg_conf,
                                'infer_ms': infer_ms,
                                'coarse_ms': coarse_ms,
                                'mask_ms': mask_ms,
                                'detector': 'opencv_contour'
                            })
                    except Exception:
                        pass

                # Emitir resultados (imagem dummy não é usada pelo overlay)
                try:
                    self.detection_ready.emit(self._dummy_frame, det_labels, emit_boxes)
                except Exception:
                    pass

            except Exception:
                # Never let exceptions bubble to Qt
                time.sleep(0.01)

    # ---- Debug/Preview para a aba Train ----
    def debug_preview(self, frame_bgr: np.ndarray, region: Dict) -> dict:
        try:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            mask = self._make_color_mask(hsv)
            overlay = frame_bgr.copy()
            # Extrair caixas diretamente da máscara atual (como sempre foi na prévia)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for cnt in contours or []:
                x, y, w, h = cv2.boundingRect(cnt)
                if w <= 1 or h <= 1:
                    continue
                boxes.append({
                    'x1': region['left'] + x,
                    'y1': region['top'] + y,
                    'x2': region['left'] + x + w,
                    'y2': region['top'] + y + h,
                    'label': 'mask_box',
                    'conf': 1.0
                })
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return {"mask": mask, "overlay": overlay, "boxes": boxes}
        except Exception as e:
            return {"error": str(e)}

    def set_confidence(self, confidence: float):
        with self.mutex:
            self.confidence = confidence

    # (sem método de modelo – pipeline OpenCV)

    def set_capture_region(self, region: Dict):
        with self.mutex:
            self.capture_region = region
            self._last_target_center = None

    def set_pixel_step(self, step: int):
        with self.mutex:
            self.pixel_step = max(1, int(step))

    def set_infer_fps_limit(self, fps: int):
        with self.mutex:
            self.infer_fps_limit = max(1, int(fps))
            self._infer_interval = 1.0 / float(self.infer_fps_limit)

    def set_mouse_enabled(self, enabled: bool):
        """Habilita/desabilita o controle de mouse via kernel Linux (uinput).
        Retorna uma tupla (ok: bool, msg: str). Em caso de falha, ok=False e msg explica o problema.
        """
        with self.mutex:
            if not enabled:
                self.mouse_enabled = False
                self._mouse_device = None
                return True, "Mouse desativado"

            if not HAS_UINPUT:
                self.mouse_enabled = False
                return False, "python-uinput não está disponível. Instale com: pip install python-uinput"

            # Inicializa uinput device (REL)
            try:
                import stat
                uinput_path = "/dev/uinput"
                if not os.path.exists(uinput_path):
                    return False, "/dev/uinput não encontrado. Carregue o módulo: sudo modprobe uinput"
                if not os.access(uinput_path, os.W_OK):
                    try:
                        st = os.stat(uinput_path)
                        mode = stat.S_IMODE(st.st_mode)
                        return False, (
                            f"Sem permissão para escrever em {uinput_path}. "
                            f"Modo atual: 0{oct(mode)[2:]}. "
                            "Execute como root ou ajuste permissões/grupos (ex.: adduser $USER input)."
                        )
                    except Exception:
                        return False, "Sem permissão para acessar /dev/uinput. Execute como root ou ajuste udev/permissões."

                if self._mouse_device is None:
                    self._mouse_device = uinput.Device(
                        [
                            uinput.REL_X,
                            uinput.REL_Y,
                            uinput.BTN_LEFT,
                            uinput.BTN_RIGHT,
                            uinput.BTN_MIDDLE,
                        ],
                        name="OpenCV Pointer",
                    )
                # Determinar união dos monitores (para centro correto)
                try:
                    with mss.mss() as sct:
                        mons = [m for m in sct.monitors[1:]] or [sct.monitors[1]]
                        min_left = min(int(m.get('left', 0)) for m in mons)
                        min_top = min(int(m.get('top', 0)) for m in mons)
                        max_right = max(int(m.get('left', 0)) + int(m.get('width', 0)) for m in mons)
                        max_bottom = max(int(m.get('top', 0)) + int(m.get('height', 0)) for m in mons)
                        self._union_left = int(min_left)
                        self._union_top = int(min_top)
                        self._union_w = int(max_right - min_left)
                        self._union_h = int(max_bottom - min_top)
                except Exception:
                    self._union_left, self._union_top = 0, 0
                    self._union_w, self._union_h = None, None

                self.mouse_enabled = True
                return True, "Mouse via kernel (REL) habilitado"
            except Exception as e:
                self._mouse_device = None
                self.mouse_enabled = False
                return False, f"Erro ao habilitar mouse via kernel: {e}"

    # ----- Benchmark API -----
    def start_benchmark(self):
        with self._bench_mutex:
            self._benchmark_enabled = True
            self._benchmark_data = []
            self._benchmark_start_time = time.time()

    def stop_benchmark(self):
        with self._bench_mutex:
            self._benchmark_enabled = False
            data = list(self._benchmark_data)
            self._benchmark_data = []
            self._benchmark_start_time = None
            return data

    def stop(self):
        self.running = False
        try:
            self.frame_event.set()
        except Exception:
            pass


class FFMpegCaptureThread(QThread):
    """Captura via FFmpeg x11grab para Xorg.
    Emite frames BGR redimensionados para target_size.
    """
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    region_changed = pyqtSignal(dict)

    def __init__(self, target_size: Tuple[int, int] = cfg.DETECT_RESOLUTION, fps_limit: int = cfg.DEFAULT_FPS_LIMIT):
        super().__init__()
        self.target_size = target_size
        self.fps_limit = fps_limit
        self.running = False
        self.capture_region: Optional[Dict] = None
        self.mutex = threading.Lock()
        self.proc = None
        self.stdout = None
        # Permite forçar DISPLAY via env var OPENCV_FFMPEG_DISPLAY (útil em setups não padrão)
        self.display = os.environ.get('OPENCV_FFMPEG_DISPLAY', os.environ.get('DISPLAY', ':0.0'))
        self.frame_bytes = 0

    def _stop_proc(self):
        try:
            if self.proc is not None:
                self.proc.kill()
        except Exception:
            pass
        self.proc = None
        self.stdout = None

    def _clamp_region(self, region: Dict) -> Dict:
        """Clamp region to non-negative coordinates within the primary screen.
        Avoids x11grab failures with negative offsets or oversize areas.
        """
        try:
            with mss.mss() as sct:
                root = sct.monitors[1]
            root_left = int(root.get('left', 0))
            root_top = int(root.get('top', 0))
            root_w = int(root.get('width', 1920))
            root_h = int(root.get('height', 1080))
        except Exception:
            root_left = 0
            root_top = 0
            root_w = max(16, int(self.target_size[0]))
            root_h = max(16, int(self.target_size[1]))

        left = int(region.get('left', 0))
        top = int(region.get('top', 0))
        width = int(region.get('width', self.target_size[0]))
        height = int(region.get('height', self.target_size[1]))

        # Clamp to be within [root_left, root_left+root_w) etc.
        if left < root_left:
            width -= (root_left - left)
            left = root_left
        if top < root_top:
            height -= (root_top - top)
            top = root_top
        max_right = root_left + root_w
        max_bottom = root_top + root_h
        if left + width > max_right:
            width = max(1, max_right - left)
        if top + height > max_bottom:
            height = max(1, max_bottom - top)

        width = max(16, width)
        height = max(16, height)
        return {"left": left, "top": top, "width": width, "height": height}

    def _start_proc(self, region: Dict):
        self._stop_proc()
        try:
            import shutil
            if not shutil.which('ffmpeg'):
                raise FileNotFoundError('ffmpeg não encontrado no PATH')
            # Wayland is not supported by x11grab
            if os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland':
                raise RuntimeError('Wayland detectado: FFmpeg x11grab indisponível (use motor MSS).')
        except Exception as e:
            raise e

        # Clamp region to avoid negative offsets/overshoot
        region = self._clamp_region(region or {})
        left = int(region.get('left', 0))
        top = int(region.get('top', 0))
        width = int(region.get('width', self.target_size[0]))
        height = int(region.get('height', self.target_size[1]))
        tw, th = int(self.target_size[0]), int(self.target_size[1])

        # Saída BGR24 em tamanho target (ffmpeg faz o scale)
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-nostats',
            '-f', 'x11grab',
            '-video_size', f'{width}x{height}',
            '-framerate', str(max(1, int(self.fps_limit))),
            '-i', f'{self.display}+{left},{top}',
            '-vf', f'scale={tw}:{th}:flags=bilinear',
            '-pix_fmt', 'bgr24',
            '-f', 'rawvideo', '-'
        ]

        import subprocess
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.stdout = self.proc.stdout
        self.frame_bytes = tw * th * 3

    def set_capture_region(self, region: Dict, window_info: Optional[Dict] = None):
        with self.mutex:
            self.capture_region = region
        try:
            self.region_changed.emit(region)
        except Exception:
            pass
        # reiniciar o processo se já estiver rodando
        if self.isRunning():
            try:
                with self.mutex:
                    self._start_proc(self.capture_region)
            except Exception as e:
                self.error_occurred.emit(f'FFmpeg init falhou: {e}')

    def set_fps_limit(self, fps: int):
        with self.mutex:
            self.fps_limit = max(1, int(fps))
        # reiniciar proc para aplicar framerate
        if self.isRunning() and self.capture_region:
            try:
                with self.mutex:
                    self._start_proc(self.capture_region)
            except Exception as e:
                self.error_occurred.emit(f'FFmpeg reinit (fps) falhou: {e}')

    def set_target_size(self, size: Tuple[int, int]):
        with self.mutex:
            tw, th = size
            self.target_size = (max(16, int(tw)), max(16, int(th)))
        # reiniciar proc para aplicar scale
        if self.isRunning() and self.capture_region:
            try:
                with self.mutex:
                    self._start_proc(self.capture_region)
            except Exception as e:
                self.error_occurred.emit(f'FFmpeg reinit (size) falhou: {e}')

    def run(self):
        # garantia de região
        with self.mutex:
            region = self.capture_region
        if region is None:
            # fallback para tela cheia primária via mss
            try:
                with mss.mss() as sct:
                    region = sct.monitors[1]
            except Exception:
                region = {'left': 0, 'top': 0, 'width': self.target_size[0], 'height': self.target_size[1]}
            with self.mutex:
                self.capture_region = region

        try:
            self._start_proc(region)
        except Exception as e:
            self.error_occurred.emit(f'FFmpeg não disponível ou erro: {e}')
            self.running = False
            return

        import os as _os
        import select as _select
        self.running = True
        partial = bytearray()
        fd = self.stdout.fileno() if self.stdout is not None else None
        last_frame_ts = time.time()
        while self.running:
            try:
                if fd is None:
                    # Try to restart the process once if it vanished
                    with self.mutex:
                        region = self.capture_region
                    try:
                        self._start_proc(region or {"left": 0, "top": 0, "width": self.target_size[0], "height": self.target_size[1]})
                        fd = self.stdout.fileno() if self.stdout is not None else None
                        partial = bytearray()
                        last_frame_ts = time.time()
                        if fd is None:
                            raise RuntimeError('FFmpeg stdout indisponível')
                    except Exception as e:
                        self.error_occurred.emit(f'FFmpeg não disponível ou erro: {e}')
                        time.sleep(0.2)
                        continue
                # Espera até 50ms por dados ou até stop() matar o proc
                r, _, _ = _select.select([fd], [], [], 0.05)
                if not r:
                    # No data; if we've had no frame for a while, try restart
                    if (time.time() - last_frame_ts) > 2.0:
                        try:
                            with self.mutex:
                                region = self.capture_region
                            self._start_proc(region or {"left": 0, "top": 0, "width": self.target_size[0], "height": self.target_size[1]})
                            fd = self.stdout.fileno() if self.stdout is not None else None
                            partial = bytearray()
                            last_frame_ts = time.time()
                            if fd is None:
                                raise RuntimeError('FFmpeg stdout indisponível')
                        except Exception:
                            self.error_occurred.emit('FFmpeg sem frames. Voltando para tela completa.')
                            time.sleep(0.2)
                    continue
                needed = self.frame_bytes - len(partial)
                # ler o que tiver disponível, até 'needed'
                chunk = _os.read(fd, needed)
                if not chunk:
                    # EOF: try to restart and notify UI
                    try:
                        with self.mutex:
                            region = self.capture_region
                        self._start_proc(region or {"left": 0, "top": 0, "width": self.target_size[0], "height": self.target_size[1]})
                        fd = self.stdout.fileno() if self.stdout is not None else None
                        partial = bytearray()
                        last_frame_ts = time.time()
                        if fd is None:
                            raise RuntimeError('FFmpeg stdout indisponível')
                        self.error_occurred.emit('FFmpeg reiniciado após EOF.')
                        continue
                    except Exception:
                        self.error_occurred.emit('FFmpeg finalizado/sem frames. Voltando para tela completa.')
                        time.sleep(0.2)
                        continue
                partial.extend(chunk)
                if len(partial) < self.frame_bytes:
                    continue
                buf = memoryview(partial)[:self.frame_bytes]
                arr = np.frombuffer(buf, dtype=np.uint8)
                with self.mutex:
                    tw, th = self.target_size
                try:
                    frame = arr.reshape((th, tw, 3))
                except Exception:
                    # desalinhado: descarta e segue
                    partial = partial[self.frame_bytes:]
                    continue
                # Detecta frames pretos (ex.: Wayland/x11grab sem permissão)
                try:
                    if frame.size == 0 or float(np.mean(frame[::16, ::16])) < 1.0:
                        self.error_occurred.emit('FFmpeg gerou frame vazio/preto. Voltando para tela completa.')
                        time.sleep(0.05)
                        continue
                except Exception:
                    pass
                # emitir frame (cópia mínima: np.frombuffer usa view)
                self.frame_ready.emit(frame.copy())
                last_frame_ts = time.time()
                # remover o frame consumido
                partial = bytearray(partial[self.frame_bytes:])
            except Exception:
                time.sleep(0.002)

        self._stop_proc()
        
    def stop(self):
        self.running = False
        try:
            self._stop_proc()
        finally:
            try:
                if self.stdout:
                    self.stdout.close()
            except Exception:
                pass


class FastgrabCaptureThread(QThread):
    """Captura via fastgrab.screenshot.Screenshot().capture().
    Emite frames BGR redimensionados para target_size.
    """
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    region_changed = pyqtSignal(dict)

    def __init__(self, target_size: Tuple[int, int] = cfg.DETECT_RESOLUTION, fps_limit: int = cfg.DEFAULT_FPS_LIMIT):
        super().__init__()
        self.target_size = (max(16, int(target_size[0])), max(16, int(target_size[1])))
        self.fps_limit = max(1, int(fps_limit))
        self.frame_interval = 1.0 / float(self.fps_limit)
        self.running = False
        self.capture_region: Optional[Dict] = None
        self.window_info: Optional[Dict] = None
        self.mutex = threading.Lock()
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self._grabber = None

    def set_capture_region(self, region: Dict, window_info: Optional[Dict] = None):
        with self.mutex:
            self.capture_region = region
            self.window_info = window_info
            self.consecutive_errors = 0
        try:
            self.region_changed.emit(region)
        except Exception:
            pass

    def set_fps_limit(self, fps: int):
        with self.mutex:
            self.fps_limit = max(1, int(fps))
            self.frame_interval = 1.0 / float(self.fps_limit)

    def set_target_size(self, size: Tuple[int, int]):
        with self.mutex:
            w, h = size
            self.target_size = (max(16, int(w)), max(16, int(h)))

    def _default_region(self) -> Dict:
        try:
            with mss.mss() as sct:
                return sct.monitors[1]
        except Exception:
            return {"left": 0, "top": 0, "width": self.target_size[0], "height": self.target_size[1]}

    def _union_origin(self) -> Tuple[int, int]:
        """Return the virtual desktop origin (left, top) using mss union monitor."""
        try:
            with mss.mss() as sct:
                m0 = sct.monitors[0]
                return int(m0.get('left', 0)), int(m0.get('top', 0))
        except Exception:
            return 0, 0

    def run(self):
        if not HAS_FASTGRAB:
            self.error_occurred.emit('fastgrab não disponível. Instale com: pip install fastgrab')
            return
        self.running = True
        try:
            self._grabber = _fg_screenshot.Screenshot()
        except Exception as e:
            self.error_occurred.emit(f'Falha ao inicializar fastgrab: {e}')
            return

        last_time = time.perf_counter()
        while self.running:
            try:
                now = time.perf_counter()
                elapsed = now - last_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                last_time = time.perf_counter()

                with self.mutex:
                    region = self.capture_region
                    tw, th = self.target_size
                if region is None:
                    region = self._default_region()

                # Capture full virtual screen, then crop to region accounting for union origin
                try:
                    img = self._grabber.capture()  # ndarray, likely RGBA
                except Exception as e:
                    self.consecutive_errors += 1
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        self.error_occurred.emit(f'fastgrab falhou ao capturar: {e}')
                        self.consecutive_errors = 0
                    time.sleep(0.02)
                    continue

                arr = np.asarray(img)
                if arr is None or arr.size == 0:
                    self.consecutive_errors += 1
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        self.error_occurred.emit('fastgrab retornou frame vazio.')
                        self.consecutive_errors = 0
                    time.sleep(0.02)
                    continue

                # Determine crop box in arr coords by shifting by union origin
                origin_left, origin_top = self._union_origin()
                x0 = int(region['left'] - origin_left)
                y0 = int(region['top'] - origin_top)
                w = int(region['width'])
                h = int(region['height'])
                H, W = int(arr.shape[0]), int(arr.shape[1])
                # Clamp crop
                x0c = max(0, min(W - 1, x0))
                y0c = max(0, min(H - 1, y0))
                x1c = max(0, min(W, x0 + w))
                y1c = max(0, min(H, y0 + h))
                if x1c <= x0c or y1c <= y0c:
                    # fallback to center crop of available area
                    x0c, y0c = 0, 0
                    x1c, y1c = W, H
                crop = arr[y0c:y1c, x0c:x1c]

                # Convert to BGR (fastgrab likely returns RGBA)
                try:
                    if crop.ndim == 3 and crop.shape[2] == 4:
                        frame = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR)
                    elif crop.ndim == 3 and crop.shape[2] == 3:
                        frame = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    elif crop.ndim == 2:
                        frame = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                    else:
                        frame = crop[:, :, :3].copy()
                except Exception:
                    frame = crop[:, :, :3].copy()

                if frame.size == 0:
                    self.consecutive_errors += 1
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        self.error_occurred.emit('fastgrab produziu recorte vazio.')
                        self.consecutive_errors = 0
                    time.sleep(0.01)
                    continue

                # Resize to target
                ih, iw = frame.shape[:2]
                if iw != tw or ih != th:
                    interp = cv2.INTER_AREA if (tw < iw or th < ih) else cv2.INTER_LINEAR
                    frame = cv2.resize(frame, (tw, th), interpolation=interp)

                self.frame_ready.emit(frame)
                self.consecutive_errors = 0
            except Exception:
                time.sleep(0.01)

    def stop(self):
        self.running = False


__all__ = [
    "WindowSelector",
    "FastCaptureThread",
    "FFMpegCaptureThread",
    "FastgrabCaptureThread",
    "DetectionThread",
    "HAS_EWMH",
    "HAS_UINPUT",
    "HAS_XLIB_MOUSE",
]
