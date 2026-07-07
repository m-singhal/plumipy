#!/usr/bin/env python3
"""
PLUMIPY Desktop App
Launch: python app/main.py
"""
import sys
import os

# Allow importing plumipy from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("QtAgg")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont

from app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PLUMIPY")
    app.setOrganizationName("Plumipy")

    # Load stylesheet
    qss_path = os.path.join(os.path.dirname(__file__), "styles", "theme.qss")
    with open(qss_path, "r") as f:
        app.setStyleSheet(f.read())

    app.setFont(QFont("Helvetica Neue", 13))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
