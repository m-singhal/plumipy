import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QComboBox, QLabel, QSlider, QSizePolicy
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


DARK = {
    "bg":       "#1e1e2e",
    "axes_bg":  "#181825",
    "text":     "#cdd6f4",
    "grid":     "#313244",
    "spine":    "#45475a",
    "purple":   "#cba6f7",
    "blue":     "#89b4fa",
    "green":    "#a6e3a1",
    "red":      "#f38ba8",
    "yellow":   "#f9e2af",
    "teal":     "#94e2d5",
    "peach":    "#fab387",
}

def apply_dark_style(fig, axes):
    """Apply the Catppuccin-dark theme to a figure and its axes."""
    fig.patch.set_facecolor(DARK["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(DARK["axes_bg"])
        ax.tick_params(colors=DARK["text"], labelsize=9)
        ax.xaxis.label.set_color(DARK["text"])
        ax.yaxis.label.set_color(DARK["text"])
        if ax.get_title():
            ax.title.set_color(DARK["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK["spine"])
        ax.grid(True, color=DARK["grid"], linewidth=0.5, alpha=0.8)
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(DARK["axes_bg"])
            leg.get_frame().set_edgecolor(DARK["spine"])
            for t in leg.get_texts():
                t.set_color(DARK["text"])


class PlotCanvas(QWidget):
    """
    A self-contained widget: matplotlib figure + toolbar + save button.
    Usage:
        canvas = PlotCanvas(nrows=1, ncols=1, figsize=(8, 4))
        canvas.ax  → the main Axes (if 1×1)
        canvas.axes → list of all Axes
        canvas.draw()  → refresh
    """

    def __init__(self, nrows=1, ncols=1, figsize=(9, 4.5), parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.fig.patch.set_facecolor(DARK["bg"])

        self._axes_list = []
        for i in range(nrows * ncols):
            ax = self.fig.add_subplot(nrows, ncols, i + 1)
            self._axes_list.append(ax)

        self.axes = self._axes_list
        self.ax = self._axes_list[0] if self._axes_list else None

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        toolbar = NavigationToolbar2QT(self.canvas, self)
        toolbar.setStyleSheet(
            "background-color:#181825; color:#cdd6f4; border:none;"
            "QToolButton { color:#cdd6f4; }"
        )

        save_btn = QPushButton("Save figure…")
        save_btn.setObjectName("secondary_btn")
        save_btn.clicked.connect(self._save_dialog)

        top = QHBoxLayout()
        top.addWidget(toolbar)
        top.addStretch()
        top.addWidget(save_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self.canvas)

    def draw(self):
        apply_dark_style(self.fig, self.axes)
        self.canvas.draw_idle()

    def clear_axes(self):
        for ax in self.axes:
            ax.cla()

    def _save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save figure",
            "figure",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;SVG (*.svg);;PDF (*.pdf)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return

        # DPI dialog via simple input
        from PyQt6.QtWidgets import QInputDialog
        dpi, ok = QInputDialog.getInt(
            self, "Resolution", "DPI (72=screen, 150=good, 300=publication, 600=high):",
            300, 72, 1200, 50
        )
        if not ok:
            dpi = 300

        self.fig.savefig(path, dpi=dpi, bbox_inches="tight",
                         facecolor=self.fig.get_facecolor())
