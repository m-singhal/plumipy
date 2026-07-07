import io
from math import factorial

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage


# ── Catppuccin palette (mirrors plot_canvas.py) ───────────────────────────────
_BG     = "#181825"
_PURPLE = "#cba6f7"
_BLUE   = "#89b4fa"
_GREEN  = "#a6e3a1"
_RED    = "#f38ba8"
_YELLOW = "#f9e2af"
_TEAL   = "#94e2d5"
_PEACH  = "#fab387"
_TEXT   = "#cdd6f4"
_MUTED  = "#a6adc8"
_GRID   = "#313244"


# ── Mini-plot generators (Agg, no display needed) ────────────────────────────

def _make_plot_px(plot_fn, w=230, h=140) -> QPixmap:
    fig = Figure(figsize=(w / 100, h / 100), facecolor=_BG)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(_BG)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    plot_fn(ax)
    canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=_BG,
                edgecolor="none", bbox_inches="tight", pad_inches=0.05)
    buf.seek(0)
    return QPixmap.fromImage(QImage.fromData(buf.read()))


def _pl_spectrum(ax):
    E = np.linspace(820, 1200, 800)
    zpl = 1020
    spec = 0.55 * np.exp(-(E - zpl) ** 2 / (2 * 5 ** 2))
    for dE, amp in [(-48, 0.28), (-95, 0.14), (-140, 0.07), (-185, 0.035),
                    (48, 0.20), (95, 0.10), (140, 0.05)]:
        spec += amp * np.exp(-(E - (zpl + dE)) ** 2 / (2 * 8 ** 2))
    spec_abs = 0.55 * np.exp(-(E - zpl) ** 2 / (2 * 5 ** 2))
    for dE, amp in [(-48, 0.20), (-95, 0.10), (48, 0.28), (95, 0.14), (140, 0.07)]:
        spec_abs += amp * np.exp(-(E - (zpl - dE)) ** 2 / (2 * 8 ** 2))
    ax.fill_between(E, spec * 0.9, alpha=0.28, color=_BLUE)
    ax.plot(E, spec * 0.9, color=_BLUE, lw=1.8)
    ax.fill_between(E, spec_abs * 0.75, alpha=0.18, color=_GREEN)
    ax.plot(E, spec_abs * 0.75, color=_GREEN, lw=1.5, ls="--")
    ax.axvline(zpl, color=_PURPLE, lw=0.9, ls=":", alpha=0.7)


def _feynman_eph(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    vx, vy = 0.50, 0.70

    # Incoming electron → vertex
    ax.annotate("", xy=(vx - 0.03, vy), xytext=(0.06, vy),
                arrowprops=dict(arrowstyle="-|>", color=_BLUE, lw=2.5,
                                mutation_scale=14, shrinkA=0, shrinkB=0))
    # Outgoing electron vertex →
    ax.annotate("", xy=(0.94, vy), xytext=(vx + 0.03, vy),
                arrowprops=dict(arrowstyle="-|>", color=_BLUE, lw=2.5,
                                mutation_scale=14, shrinkA=0, shrinkB=0))
    # Phonon propagator (wavy line downward)
    t = np.linspace(0, 1, 320)
    x_ph = vx + 0.055 * np.sin(t * np.pi * 5)
    y_ph = vy - t * 0.52
    ax.plot(x_ph, y_ph, color=_YELLOW, lw=2.3, zorder=3)
    # Arrowhead at phonon end
    n = len(x_ph)
    ax.annotate("", xy=(x_ph[-1], y_ph[-1] - 0.022),
                xytext=(x_ph[int(n * 0.90)], y_ph[int(n * 0.90)]),
                arrowprops=dict(arrowstyle="-|>", color=_YELLOW, lw=1.6,
                                mutation_scale=12, shrinkA=0, shrinkB=0))
    # Vertex dot
    ax.plot(vx, vy, 'o', color=_PURPLE, markersize=10, zorder=6)
    # Labels
    ax.text(0.27, vy + 0.11, "e⁻", color=_BLUE, fontsize=13,
            fontweight='bold', ha='center', va='center')
    ax.text(0.73, vy + 0.11, "e⁻", color=_BLUE, fontsize=13,
            fontweight='bold', ha='center', va='center')
    ax.text(0.70, vy - 0.25, "phonon\nω, q", color=_YELLOW, fontsize=9,
            ha='center', va='center')
    ax.text(0.50, 0.97, "Electron–Phonon Coupling", color=_MUTED, fontsize=8,
            ha='center', va='top')


def _spring_lattice(ax):
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(0.05, 0.95)
    n = 7
    xs = np.linspace(0.08, 0.92, n)
    y0 = 0.50
    # First-mode sinusoidal displacement
    disp = 0.10 * np.sin(np.pi * np.arange(n) / (n - 1))
    # Springs
    for i in range(n - 1):
        x1, y1 = xs[i], y0 + disp[i]
        x2, y2 = xs[i + 1], y0 + disp[i + 1]
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx ** 2 + dy ** 2)
        px, py = -dy / dist, dx / dist
        t = np.linspace(0, 1, 80)
        n_coils = 5
        amp = 0.030
        sx = x1 + dx * t + px * amp * np.sin(t * np.pi * n_coils * 2)
        sy = y1 + dy * t + py * amp * np.sin(t * np.pi * n_coils * 2)
        ax.plot(sx, sy, color=_PEACH, lw=1.5, alpha=0.9, zorder=2)
    # Atoms
    atom_colors = [_BLUE, _PURPLE, _TEAL, _BLUE, _PURPLE, _TEAL, _BLUE]
    for i, (xi, di) in enumerate(zip(xs, disp)):
        yi = y0 + di
        ax.plot(xi, yi, 'o', color=atom_colors[i], markersize=15,
                markeredgecolor=_BG, markeredgewidth=1.5, zorder=5)
        # Displacement arrows
        if abs(di) > 0.01:
            sign = 1 if di > 0 else -1
            ax.annotate("", xy=(xi, yi + sign * 0.075),
                        xytext=(xi, yi),
                        arrowprops=dict(arrowstyle="-|>", color=_GREEN, lw=1.1,
                                        mutation_scale=7, shrinkA=3, shrinkB=0))
    ax.text(0.50, 0.90, "Phonon Normal Mode", color=_MUTED, fontsize=8,
            ha='center', va='center')


def _temp_spectra(ax):
    E = np.linspace(-4, 4, 300)
    for sigma, col, lw, alpha in [
        (0.22, _PURPLE, 2.4, 0.9),
        (0.55, _BLUE,   1.7, 0.8),
        (1.05, _GREEN,  1.1, 0.7),
    ]:
        s = np.exp(-E ** 2 / (2 * sigma ** 2))
        ax.fill_between(E, s, alpha=0.10, color=col)
        ax.plot(E, s, color=col, lw=lw, alpha=alpha)


def _mc_histogram(ax):
    rng = np.random.default_rng(42)
    s = 2.8
    samples = rng.poisson(s, 4000)
    bins = np.arange(0, 13)
    counts, _ = np.histogram(samples, bins=bins)
    x = bins[:-1]
    ax.bar(x, counts / counts.max(), color=_GREEN, alpha=0.68, width=0.82)
    pmf = np.array([np.exp(-s) * s ** k / factorial(int(k)) for k in x], dtype=float)
    ax.plot(x, pmf / pmf.max(), color=_PURPLE, lw=2.1,
            marker="o", ms=3.0, markerfacecolor=_PURPLE, markeredgewidth=0)


def _stacked_wells(ax):
    """Configuration coordinate diagram — displaced-squeezed oscillator."""
    k_gs = 0.82   # GS curvature (steep)
    k_es = 0.34   # ES curvature (shallow — squeezing effect)
    x0   = 0.95   # ES horizontal displacement
    dE   = 3.0    # ES minimum energy above GS

    x = np.linspace(-1.75, 2.75, 600)
    V_gs = k_gs * x ** 2
    V_es = k_es * (x - x0) ** 2 + dE

    clip_gs = 2.55
    clip_es = dE + 2.0

    m_gs = V_gs <= clip_gs
    m_es = V_es <= clip_es

    # Parabolas
    ax.plot(x[m_gs], V_gs[m_gs], color=_BLUE, lw=2.2, zorder=3)
    ax.plot(x[m_es], V_es[m_es], color=_TEAL, lw=2.2, zorder=3)

    # Vibrational energy levels — GS
    for lv in [0.38, 0.92, 1.50, 2.12]:
        xr = np.sqrt(lv / k_gs)
        ax.hlines(lv, -xr, xr, color=_RED, lw=1.1, alpha=0.90, zorder=4)

    # Vibrational energy levels — ES
    for lv_rel in [0.33, 0.82, 1.38, 1.95]:
        lv = dE + lv_rel
        xr = np.sqrt(lv_rel / k_es)
        ax.hlines(lv, x0 - xr, x0 + xr, color=_RED, lw=1.1, alpha=0.90, zorder=4)

    # Absorption arrow (left, upward, purple)
    ax.annotate("", xy=(-0.20, dE + 1.38), xytext=(-0.20, 0.38),
                arrowprops=dict(arrowstyle="-|>", color=_PURPLE, lw=2.0,
                               mutation_scale=11, shrinkA=0, shrinkB=0))
    ax.text(-0.44, dE * 0.5 + 0.1, "Absorption", color=_PURPLE, fontsize=6.0,
            fontweight="bold", ha="center", va="center", rotation=90)

    # Emission arrow (right, downward, green)
    ax.annotate("", xy=(x0 + 0.26, 1.50), xytext=(x0 + 0.26, dE + 0.33),
                arrowprops=dict(arrowstyle="-|>", color=_GREEN, lw=2.0,
                               mutation_scale=11, shrinkA=0, shrinkB=0))
    ax.text(x0 + 0.48, (dE + 0.33 + 1.50) * 0.5, "Emission", color=_GREEN,
            fontsize=6.0, fontweight="bold", ha="center", va="center", rotation=90)

    # ZPL dashed vertical line
    ax.vlines(x0, 0, dE, color=_TEXT, lw=0.7, ls="--", alpha=0.45, zorder=2)

    # ZPL double-headed arrow + label on the right
    ax.annotate("", xy=(2.42, dE), xytext=(2.42, 0.0),
                arrowprops=dict(arrowstyle="<->", color=_YELLOW, lw=1.2,
                               shrinkA=0, shrinkB=0))
    ax.text(2.58, dE * 0.5, "ZPL", color=_YELLOW, fontsize=6.5, ha="left", va="center")

    # Displacement q arrow at bottom
    ax.annotate("", xy=(x0, -0.37), xytext=(0.0, -0.37),
                arrowprops=dict(arrowstyle="<->", color=_MUTED, lw=0.9,
                               shrinkA=1, shrinkB=1))
    ax.text(x0 * 0.5, -0.50, "q", color=_MUTED, fontsize=7.5,
            ha="center", va="top", style="italic")

    # ω_gs, ω_es labels inside parabola arms
    ax.text(-1.40, 2.30, "ω_gs", color=_BLUE, fontsize=6.5,
            ha="center", va="center", style="italic")
    ax.text(2.00, dE + 1.75, "ω_es", color=_TEAL, fontsize=6.5,
            ha="center", va="center", style="italic")

    # Squeezing label at top
    ax.text(0.48, clip_es + 0.15, "Squeezing  (r)", color=_PEACH, fontsize=6.0,
            ha="center", va="bottom", style="italic")

    # GS / ES labels
    ax.text(-1.40, 0.20, "GS", color=_BLUE, fontsize=8.5, fontweight="bold")
    ax.text(1.90, dE - 0.35, "ES", color=_TEAL, fontsize=8.5, fontweight="bold")

    # Y-axis arrow
    ax.annotate("", xy=(-1.72, clip_es + 0.30), xytext=(-1.72, -0.68),
                arrowprops=dict(arrowstyle="-|>", color=_MUTED, lw=1.0,
                               mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.text(-1.90, (clip_es - 0.68) * 0.45, "Energy", color=_MUTED, fontsize=6.0,
            ha="right", va="center", rotation=90)

    # X-axis arrow
    ax.annotate("", xy=(2.72, -0.66), xytext=(-1.68, -0.66),
                arrowprops=dict(arrowstyle="-|>", color=_MUTED, lw=1.0,
                               mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.text(0.48, -0.88, "Configuration coordinate", color=_MUTED, fontsize=5.8,
            ha="center", va="top")

    ax.set_xlim(-2.05, 2.80)
    ax.set_ylim(-1.0, clip_es + 0.60)


# ── Feature card data ─────────────────────────────────────────────────────────

FEATURES = [
    {
        "plot_fn": _pl_spectrum,
        "title": "Emission & Absorption Spectra",
        "body": (
            "Full photoluminescence (PL) emission and optical absorption spectra "
            "including the zero-phonon line (ZPL) and all phonon sidebands, computed "
            "via the generating function approach within the harmonic approximation."
        ),
        "accent": _BLUE,
    },
    {
        "plot_fn": _feynman_eph,
        "title": "Huang–Rhys Factors",
        "body": (
            "Mode-resolved S<sub>k</sub> and total S = Σ<sub>k</sub> S<sub>k</sub>. "
            "Debye–Waller factor e<sup>−S</sup>. Mass-weighted displacements q<sub>k</sub>. "
            "Inverse participation ratio (IPR) for mode localization."
        ),
        "accent": _PURPLE,
    },
    {
        "plot_fn": _spring_lattice,
        "title": "Phonon Analysis",
        "body": (
            "Reads phonon normal modes and frequencies directly from VASP OUTCAR "
            "or Phonopy band.yaml. Full phonon spectrum, mode energies in meV / THz / cm⁻¹, "
            "and localization analysis."
        ),
        "accent": _TEAL,
    },
    {
        "plot_fn": _temp_spectra,
        "title": "Temperature Dependence",
        "body": (
            "Thermal Bose–Einstein phonon occupation n<sub>k</sub>(T) is included in the "
            "generating function. Spectra at any temperature T, from T = 0 K "
            "to finite temperature."
        ),
        "accent": _GREEN,
    },
    {
        "plot_fn": _mc_histogram,
        "title": "Monte Carlo Sampling",
        "body": (
            "Samples phonon emission numbers from a Poisson distribution with mean S<sub>k</sub> "
            "per mode. Numerically stable for any Huang–Rhys factor. Provides full "
            "statistical moments: mean, std, skewness, kurtosis."
        ),
        "accent": _YELLOW,
    },
    {
        "plot_fn": _stacked_wells,
        "title": "Displaced–Squeezed Oscillator",
        "body": (
            "Beyond standard Huang–Rhys: accounts for frequency changes between "
            "GS and ES potential energy surfaces. Squeezing parameter "
            "r<sub>k</sub> = ½ ln(ω<sub>ES,k</sub>/ω<sub>GS,k</sub>) "
            "captures mode-frequency ratios."
        ),
        "accent": _PEACH,
    },
]


# ── Feature card widget ───────────────────────────────────────────────────────

class FeatureCard(QFrame):
    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 16)

        # Mini-plot
        px = _make_plot_px(data["plot_fn"])
        img_lbl = QLabel()
        img_lbl.setPixmap(px)
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl.setStyleSheet(
            f"border: 1px solid {data['accent']}55; border-radius: 6px; padding: 2px;"
        )
        lay.addWidget(img_lbl)

        # Title
        title = QLabel(data["title"])
        title.setFont(QFont("Helvetica Neue", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {data['accent']};")
        title.setWordWrap(True)
        lay.addWidget(title)

        # Body
        body = QLabel(data["body"])
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setObjectName("hint_label")
        body.setStyleSheet("color: #a6adc8; font-size: 13px; line-height: 1.5;")
        lay.addWidget(body)


# ── Workflow overview row ─────────────────────────────────────────────────────

class WorkflowRow(QFrame):
    def __init__(self, num, title, body, parent=None):
        super().__init__(parent)
        self.setObjectName("card")

        lay = QHBoxLayout(self)
        lay.setSpacing(16)
        lay.setContentsMargins(14, 12, 14, 12)

        badge = QLabel(num)
        badge.setObjectName("step_inactive")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(28, 28)
        lay.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)

        text = QLabel(
            f"<b style='color:#cba6f7'>{title}</b>"
            f"<br><span style='color:#a6adc8'>{body}</span>"
        )
        text.setWordWrap(True)
        text.setTextFormat(Qt.TextFormat.RichText)
        text.setStyleSheet("font-size: 14px; line-height: 1.5;")
        lay.addWidget(text, 1)


# ── Home Page ─────────────────────────────────────────────────────────────────

class HomePage(QWidget):
    go_to_inputs     = pyqtSignal()
    new_calculation  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(48, 40, 48, 48)
        lay.setSpacing(30)

        # ── Hero ─────────────────────────────────────────────
        hero_title = QLabel("PLUMIPY")
        hero_title.setObjectName("hero_title")
        hero_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(hero_title)

        subtitle = QLabel("Vibronic Spectra from First-Principles DFT")
        subtitle.setFont(QFont("Helvetica Neue", 16))
        subtitle.setStyleSheet("color: #a6adc8; letter-spacing: 1px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(subtitle)

        # ── Buttons ──────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(14)
        btn_row.addStretch()

        start_btn = QPushButton("  Get Started  ")
        start_btn.setObjectName("primary_btn")
        start_btn.setFixedSize(200, 44)
        start_btn.setFont(QFont("Helvetica Neue", 14, QFont.Weight.Bold))
        start_btn.clicked.connect(self.go_to_inputs)
        btn_row.addWidget(start_btn)

        refresh_btn = QPushButton("  ↺  New Calculation")
        refresh_btn.setObjectName("secondary_btn")
        refresh_btn.setFixedSize(200, 44)
        refresh_btn.setFont(QFont("Helvetica Neue", 13))
        refresh_btn.clicked.connect(self.new_calculation)
        btn_row.addWidget(refresh_btn)

        btn_row.addStretch()
        lay.addLayout(btn_row)

        # ── About blurb ───────────────────────────────────────
        about = QFrame()
        about.setObjectName("info_card")
        about_lay = QVBoxLayout(about)
        about_text = QLabel(
            "<b>PLUMIPY</b> computes <b>photoluminescence (emission)</b> and "
            "<b>optical absorption spectra</b> within the harmonic approximation "
            "using the generating function method, based on outputs from "
            "<b>VASP</b> or <b>Phonopy</b>. It supports standard Huang–Rhys "
            "theory, the displaced–squeezed oscillator model, temperature-dependent "
            "spectra, and Monte Carlo sampling for large Huang–Rhys factors. "
            "Structures can also be provided from <b>Gaussian</b>, <b>CP2K</b>, "
            "<b>ORCA</b>, <b>Quantum ESPRESSO</b>, or any code via NumPy arrays."
        )
        about_text.setWordWrap(True)
        about_text.setTextFormat(Qt.TextFormat.RichText)
        about_text.setStyleSheet("color: #cdd6f4; font-size: 15px; line-height: 1.6;")
        about_lay.addWidget(about_text)
        lay.addWidget(about)

        # ── Feature cards grid ────────────────────────────────
        feat_lbl = QLabel("What PLUMIPY computes")
        feat_lbl.setObjectName("section_title")
        feat_lbl.setFont(QFont("Helvetica Neue", 16, QFont.Weight.Bold))
        lay.addWidget(feat_lbl)

        for row_start in (0, 3):
            row_lay = QHBoxLayout()
            row_lay.setSpacing(14)
            for feat in FEATURES[row_start:row_start + 3]:
                row_lay.addWidget(FeatureCard(feat))
            lay.addLayout(row_lay)

        # ── Workflow overview ─────────────────────────────────
        wf_lbl = QLabel("Three ways to use PLUMIPY")
        wf_lbl.setObjectName("section_title")
        wf_lbl.setFont(QFont("Helvetica Neue", 16, QFont.Weight.Bold))
        lay.addWidget(wf_lbl)

        for num, title, body in [
            ("1",
             "Adiabatic Approximation  (structure-based)",
             "Upload relaxed GS and ES POSCAR/CONTCAR files plus phonon data "
             "(OUTCAR or band.yaml). The structural displacement ΔR = "
             "R<sub>ES</sub> − R<sub>GS</sub> is projected onto normal "
             "modes to obtain S<sub>k</sub>. Most accurate approach."),
            ("2",
             "Vertical Gradient Approximation  (force-based)",
             "Provide forces from GS and ES single-point calculations at the same "
             "reference geometry. The coupling α<sub>k</sub> = "
             "ΔF·e<sub>k</sub> is obtained from the force difference "
             "ΔF = F<sub>ES</sub> − F<sub>GS</sub>. "
             "Cheaper than a full ES relaxation."),
            ("3",
             "External Vibrational Data",
             "Provide frequencies and normal modes as NumPy arrays — works with "
             "Gaussian, CP2K, ORCA, Quantum ESPRESSO, or any other code. Accepts "
             ".npy, .npz, or whitespace-delimited .dat / .txt files."),
        ]:
            lay.addWidget(WorkflowRow(num, title, body))

        lay.addStretch()

        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
