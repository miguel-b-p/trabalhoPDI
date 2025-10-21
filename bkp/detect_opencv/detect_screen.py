# Arquivo legado mantido para compatibilidade.
# Agora a interface está em detect_ui.ScreenDetector, a lógica em detect_logic,
# e as configurações centralizadas em config.py.
# Execute este arquivo ou use `python src/detect_ui.py`.

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from detect_ui import ScreenDetector


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
