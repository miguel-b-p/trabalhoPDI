import os

# Configurações principais centralizadas
# Tamanho da imagem usada para detecção (largura, altura)
DETECT_RESOLUTION = (800, 800)

# Limites padrão
DEFAULT_FPS_LIMIT = 30
DEFAULT_CONFIDENCE = 0.5
PIXEL_STEP_DEFAULT = 1  # 1 = todos os pixels; 2 = 1 a cada 2; etc.
INFER_FPS_LIMIT_DEFAULT = 15  # Limite de FPS para a thread de inferência (reduz CPU sem afetar a confiança por frame)

# Motor de captura: 'mss' (padrão) ou 'ffmpeg'
CAPTURE_ENGINE_DEFAULT = os.getenv('YOLO_CAPTURE_ENGINE', 'mss')

# Filtro padrão para exibir somente janelas da área de trabalho atual (Linux + EWMH)
ONLY_CURRENT_DESKTOP_DEFAULT = True

# Caminho do modelo: por padrão, usa o best.pt treinado em runs/detect/train2/weights/best.pt
# O caminho é resolvido relativo à raiz do projeto.
# REPO_ROOT deve apontar para a raiz do repositório (um nível acima de 'src')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.getenv('YOLO_MODEL_PATH', os.path.join(REPO_ROOT, 'runs', 'detect', 'train2', 'weights', 'best.pt'))

# OpenCV options
OPENCV_THREADS = 1
USE_OPENCL = True

# Mensagens
MSG_MISSING_YOLO = "YOLO não encontrado. Instale com: pip install ultralytics"
MSG_MISSING_XLIB = "python-xlib não encontrado. Instale com: pip install python-xlib"
MSG_MISSING_EWMH = "python-ewmh não encontrado. Instale com: pip install ewmh"
MSG_MISSING_UINPUT = "python-uinput não encontrado. Instale com: pip install python-uinput (requer /dev/uinput)"

# Controle de mouse (uinput)
MOUSE_GAIN = float(os.getenv('YOLO_MOUSE_GAIN', 0.6))  # ganho aplicado ao delta em pixels
MOUSE_MAX_STEP = int(os.getenv('YOLO_MOUSE_MAX_STEP', 15))  # limite de passo por evento
MOUSE_INVERT_Y = bool(int(os.getenv('YOLO_MOUSE_INVERT_Y', '0')))  # 1 para inverter eixo Y

# FOV (Field of View) circular para movimento do mouse
# O alvo só movimenta o mouse se o centro da detecção estiver dentro deste círculo,
# centralizado na região de captura atual.
FOV_RADIUS_DEFAULT = int(os.getenv('YOLO_FOV_RADIUS', '100'))  # raio em pixels
DRAW_FOV_DEFAULT = bool(int(os.getenv('YOLO_DRAW_FOV', '0')))   # 1 para desenhar círculo no overlay

# Ajuste fino de mira (bias em pixels, aplicado ao ponto alvo)
AIM_X_BIAS_DEFAULT = int(os.getenv('YOLO_AIM_X_BIAS', '0'))
AIM_Y_BIAS_DEFAULT = int(os.getenv('YOLO_AIM_Y_BIAS', '0'))

# Seleção de alvo
TARGET_STRATEGY_DISTANCE_DEFAULT = bool(int(os.getenv('YOLO_TARGET_DISTANCE', '1')))  # 1 = escolher alvo mais próximo do centro do FOV
TARGET_STICK_RADIUS_DEFAULT = int(os.getenv('YOLO_TARGET_STICK_RADIUS', '120'))       # px para manter o mesmo alvo (histerese)

# Filtro por cor (HSV)
COLOR_FILTER_ENABLED_DEFAULT = bool(int(os.getenv('YOLO_COLOR_FILTER', '0')))
COLOR_FILTER_TOL_H_DEFAULT = int(os.getenv('YOLO_COLOR_TOL_H', '15'))      # 0..90 (graus HSV ~ OpenCV 0..179)
COLOR_FILTER_MIN_S_DEFAULT = int(os.getenv('YOLO_COLOR_MIN_S', '30'))      # 0..255 (saturação mínima)
COLOR_FILTER_MIN_V_DEFAULT = int(os.getenv('YOLO_COLOR_MIN_V', '30'))      # 0..255 (valor mínimo)
