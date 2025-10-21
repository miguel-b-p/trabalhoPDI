import sys
import os
import time
import threading
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

try:
    from ultralytics import YOLO
except ImportError:
    print(cfg.MSG_MISSING_YOLO)
    sys.exit(1)

# PyTorch (opcional, para garantir contexto de inferência sem gradiente)
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

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
    detection_ready = pyqtSignal(np.ndarray, list, list)

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.model = None
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
            else:
                if self.model_path:
                    print(f"Modelo não encontrado em: {self.model_path}")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
        self.running = False
        self.frame: Optional[np.ndarray] = None
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
        # Dados de benchmark
        self._benchmark_enabled = False
        self._benchmark_data = []  # lista de dicts
        self._benchmark_start_time = None
        self._bench_mutex = threading.Lock()
        # Geometria do desktop virtual (para cálculo de centro)
        self._union_left = 0
        self._union_top = 0
        self._union_w = None
        self._union_h = None
        # Estratégia de seleção de alvo e histerese
        self.target_strategy_distance = bool(getattr(cfg, 'TARGET_STRATEGY_DISTANCE_DEFAULT', True))
        self.target_stick_radius_px = int(getattr(cfg, 'TARGET_STICK_RADIUS_DEFAULT', 120))
        self._last_target_center = None  # (x,y) em coordenadas de tela
        

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

    

    def set_frame(self, frame: np.ndarray):
        with self.mutex:
            self.frame = frame.copy()
        try:
            self.frame_event.set()
        except Exception:
            pass

    def run(self):
        """Main inference loop.
        Waits for frames from the capture thread, throttles inference to
        `infer_fps_limit`, runs YOLO, and emits screen-space boxes.
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
                with self.mutex:
                    frame = None if self.frame is None else self.frame.copy()
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

                det_labels: List[str] = []
                det_boxes: List[dict] = []

                # Pré-processamento: filtro por cor (se habilitado)
                if self.color_filter_enabled:
                    try:
                        bgr = frame
                        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                        with self.mutex:
                            h0, s0, v0 = self.color_hsv
                            tol = int(self.color_tol_h)
                            s_min = int(self.color_min_s)
                            v_min = int(self.color_min_v)
                        # Faixas de hue com wrap (0..179)
                        h1 = (int(h0) - tol) % 180
                        h2 = (int(h0) + tol) % 180
                        if h1 <= h2:
                            lower = np.array([h1, max(0, s_min), max(0, v_min)], dtype=np.uint8)
                            upper = np.array([h2, 255, 255], dtype=np.uint8)
                            mask = cv2.inRange(hsv, lower, upper)
                        else:
                            lower1 = np.array([0, max(0, s_min), max(0, v_min)], dtype=np.uint8)
                            upper1 = np.array([h2, 255, 255], dtype=np.uint8)
                            lower2 = np.array([h1, max(0, s_min), max(0, v_min)], dtype=np.uint8)
                            upper2 = np.array([179, 255, 255], dtype=np.uint8)
                            mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
                        frame = cv2.bitwise_and(bgr, bgr, mask=mask)
                    except Exception:
                        pass

                # Run YOLO if available
                if self.model is not None:
                    try:
                        # Disable gradients if torch is available
                        if HAS_TORCH:
                            import torch as _torch
                            with _torch.no_grad():
                                results_list = self.model.predict(
                                    source=frame, conf=confidence, verbose=False
                                )
                        else:
                            results_list = self.model.predict(
                                source=frame, conf=confidence, verbose=False
                            )

                        if results_list:
                            results = results_list[0]
                            boxes = getattr(results, 'boxes', None)
                            names = getattr(self.model, 'names', None)

                            if boxes is not None and len(boxes) > 0:
                                # Convert to numpy arrays once
                                xyxy = boxes.xyxy.detach().cpu().numpy()
                                confs = boxes.conf.detach().cpu().numpy().reshape(-1)
                                cls_ids = (
                                    boxes.cls.detach().cpu().numpy().astype(int)
                                    if hasattr(boxes, 'cls') and boxes.cls is not None else
                                    np.zeros(len(confs), dtype=int)
                                )

                                # Frame is already at DETECT_RESOLUTION; map to screen coords
                                fh, fw = int(frame.shape[0]), int(frame.shape[1])
                                sx = float(region['width']) / float(fw)
                                sy = float(region['height']) / float(fh)
                                left = int(region['left'])
                                top = int(region['top'])

                                for i in range(xyxy.shape[0]):
                                    x1, y1, x2, y2 = xyxy[i]
                                    # Map to absolute screen coordinates
                                    ax1 = int(left + x1 * sx)
                                    ay1 = int(top + y1 * sy)
                                    ax2 = int(left + x2 * sx)
                                    ay2 = int(top + y2 * sy)
                                    label = None
                                    if names is not None:
                                        try:
                                            label = names[int(cls_ids[i])]
                                        except Exception:
                                            label = str(int(cls_ids[i]))
                                    det_labels.append(label if label is not None else str(int(cls_ids[i])))
                                    det_boxes.append({
                                        'x1': ax1, 'y1': ay1, 'x2': ax2, 'y2': ay2,
                                        'label': label, 'conf': float(confs[i])
                                    })

                                # Optional: move mouse towards the best target
                                try:
                                    self._maybe_move_mouse(results, frame)
                                except Exception:
                                    pass

                        # Benchmark collection
                        if self._benchmark_enabled:
                            try:
                                with self._bench_mutex:
                                    self._benchmark_data.append({
                                        'ts': time.time(),
                                        'img_w': int(frame.shape[1]),
                                        'img_h': int(frame.shape[0]),
                                        'conf_threshold': confidence,
                                        'boxes': len(det_boxes),
                                        # crude per-iteration timing using last interval
                                        'infer_ms': (time.perf_counter() - now) * 1000.0,
                                        'model_path': self.model_path or ''
                                    })
                            except Exception:
                                pass
                    except Exception as e:
                        # Keep UI alive even on inference errors
                        print(f"Erro na inferência: {e}")

                # Emit results (frame not used by overlay; send dummy)
                try:
                    self.detection_ready.emit(self._dummy_frame, det_labels, det_boxes)
                except Exception:
                    pass

            except Exception:
                # Never let exceptions bubble to Qt
                time.sleep(0.01)

    def set_confidence(self, confidence: float):
        with self.mutex:
            self.confidence = confidence

    def set_model_path(self, model_path: str):
        """Troca o caminho do modelo e recarrega o YOLO."""
        with self.mutex:
            if model_path and model_path != self.model_path:
                self.model_path = model_path
                try:
                    self.model = YOLO(self.model_path)
                except Exception as e:
                    print(f"Erro ao carregar modelo: {e}")

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
                        name="YOLO Pointer",
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
        # Permite forçar DISPLAY via env var YOLO_FFMPEG_DISPLAY (útil em setups não padrão)
        self.display = os.environ.get('YOLO_FFMPEG_DISPLAY', os.environ.get('DISPLAY', ':0.0'))
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
