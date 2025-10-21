

import os
import time
import io
from datetime import datetime

# Observação importante:
# - uinput é uma interface para CRIAR dispositivos de entrada virtuais (emular eventos).
# - Para ESCUTAR cliques reais do mouse, a biblioteca indicada é "evdev" (leitura de /dev/input/event*).
# Abaixo, usamos evdev para ouvir o botão esquerdo e disparamos a captura quando ele é pressionado.
# Se você precisar, também posso adicionar uso de uinput para emitir eventos virtuais.

try:
    from PIL import ImageGrab, Image  # Pillow para captura/filtro e salvar JPEG
except Exception:  # PIL pode não estar disponível em alguns ambientes headless
    ImageGrab = None
    Image = None

# Nova abordagem: usar pynput para escutar cliques globais do mouse
try:
    from pynput import mouse
except Exception:
    mouse = None

# Captura de tela preferencial: mss (rápido e multiplataforma)
try:
    import mss
    from mss import tools as mss_tools
except Exception:
    mss = None
    mss_tools = None


def _timestamp_name(prefix: str = "screenshot", ext: str = "png") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}.{ext}"


def _save_image_from_bytes(data: bytes, caminho: str, formato: str, qualidade: int):
    if formato.upper() == "JPEG":
        # Converter bytes (BGRA/RGBA) para JPEG via Pillow se possível
        if Image is None:
            # salvar como PNG se Pillow não disponível
            with open(caminho, "wb") as f:
                f.write(data)
            return caminho
        img = Image.open(io.BytesIO(data))
        rgb = img.convert("RGB")
        rgb.save(caminho, format="JPEG", quality=qualidade)
    else:
        with open(caminho, "wb") as f:
            f.write(data)
    return caminho


def _capturar_com_mss(area_especifica=None):
    with mss.mss() as sct:
        if area_especifica:
            left, top, right, bottom = area_especifica
            mon = {"left": left, "top": top, "width": right - left, "height": bottom - top}
            shot = sct.grab(mon)
        else:
            shot = sct.grab(sct.monitors[0])
        # mss retorna RGB no atributo .rgb; converter para PNG bytes
        return mss_tools.to_png(shot.rgb, shot.size)


def _capturar_com_pillow(area_especifica=None):
    if area_especifica:
        img = ImageGrab.grab(bbox=area_especifica)
    else:
        img = ImageGrab.grab()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def capturar_tela(
    pasta_destino: str = "screenshots",
    formato: str = "PNG",
    qualidade: int = 100,
    area_especifica=None,
):
    """Captura uma screenshot e salva no disco.

    Estratégia:
    - Tenta mss primeiro (rápido e compatível com Wayland/X11/Windows/macOS).
    - Se não, cai para Pillow (ImageGrab).
    """
    os.makedirs(pasta_destino, exist_ok=True)

    ext = formato.lower()
    nome = _timestamp_name(ext=ext)
    caminho = os.path.join(pasta_destino, nome)

    img_bytes = None
    if mss is not None:
        try:
            img_bytes = _capturar_com_mss(area_especifica)
            origem = "MSS"
        except Exception:
            img_bytes = None
    if img_bytes is None and ImageGrab is not None:
        img_bytes = _capturar_com_pillow(area_especifica)
        origem = "PIL"
    if img_bytes is None:
        raise RuntimeError(
            "Nenhuma biblioteca de captura disponível. Instale: pip install mss pillow pynput"
        )

    _save_image_from_bytes(img_bytes, caminho, formato, qualidade)
    print(f"[+] Salvo: {caminho} ({origem})")
    return caminho


# Removido: detecção via evdev substituída por pynput (global, sem permissões em /dev/input)


def _loop_teclado(pasta_destino: str, formato: str, qualidade: int):
    """Fallback simples: captura ao pressionar Enter no terminal."""
    print("Nenhum dispositivo de mouse detectado via evdev.")
    print("Fallback: pressione Enter para capturar, ou 'q' + Enter para sair.")
    contador = 0
    try:
        while True:
            cmd = input("[Enter] para capturar | q + Enter para sair: ").strip().lower()
            if cmd == "q":
                break
            caminho = capturar_tela(
                pasta_destino=pasta_destino,
                formato=formato,
                qualidade=qualidade,
            )
            contador += 1
            print(f"[{contador}] Capturado: {caminho}")
    except KeyboardInterrupt:
        pass
    print(f"\n✅ Finalizado. Total de prints: {contador}")


def aguardar_clique_esquerdo(
    pasta_destino: str = "screenshots",
    formato: str = "PNG",
    qualidade: int = 100,
):
    """Escuta cliques globais do mouse (botão esquerdo) com pynput e captura a tela.

    Dependências: pip install pynput mss pillow
    """
    if mouse is None:
        print("pynput não está disponível. Instalando: pip install pynput")
        raise RuntimeError("pynput não está disponível. Instale com: pip install pynput")

    print("Pronto! Clique com o botão esquerdo do mouse para capturar a tela. Ctrl+C para sair.")

    contador = 0

    def on_click(x, y, button, pressed):
        nonlocal contador
        try:
            if pressed and str(button) == "Button.left":
                caminho = capturar_tela(
                    pasta_destino=pasta_destino,
                    formato=formato,
                    qualidade=qualidade,
                )
                contador += 1
                print(f"[{contador}] Capturado ao clicar: {caminho}")
        except Exception as e:
            print(f"Erro ao capturar: {e}")

    try:
        with mouse.Listener(on_click=on_click) as listener:
            listener.join()
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n✅ Finalizado. Total de prints: {contador}")


if __name__ == "__main__":
    # Interface simples via input() para pasta e formato, sem laço de FPS.
    try:
        pasta = input("Pasta de destino (default 'screenshots'): ").strip() or "screenshots"
        formato = input("Formato (PNG/JPEG) [default PNG]: ").strip().upper() or "PNG"
        if formato not in {"PNG", "JPEG"}:
            raise ValueError("Formato inválido. Use PNG ou JPEG.")
        qualidade = 100
        if formato == "JPEG":
            try:
                qualidade = int(input("Qualidade JPEG (1-100) [default 100]: ") or 100)
                if not (1 <= qualidade <= 100):
                    raise ValueError
            except ValueError:
                raise ValueError("Qualidade JPEG deve ser um inteiro entre 1 e 100.")

        aguardar_clique_esquerdo(pasta_destino=pasta, formato=formato, qualidade=qualidade)
    except Exception as e:
        print(f"Erro: {e}")
