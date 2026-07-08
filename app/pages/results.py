import numpy as np
import mplcursors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QTabWidget, QScrollArea, QComboBox, QPushButton,
    QCheckBox, QSizePolicy, QGridLayout, QDoubleSpinBox, QMessageBox, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from app.widgets.plot_canvas import PlotCanvas, DARK
from app.pages.phonon_viewer import PhononViewerWidget

# ─────────────────────────────────────────────────────────────────────────────
# Energy unit helpers
# ─────────────────────────────────────────────────────────────────────────────

UNIT_FACTORS = {
    "meV": 1.0,
    "eV":  1e-3,
    "cm⁻¹": 8.0655,
    "nm":  None,
}

UNIT_LABELS = {
    "meV": "meV",
    "eV":  "eV",
    "cm⁻¹": "cm⁻¹",
    "nm": "nm",
}


def convert_energy(E_meV, unit):
    if unit == "meV":
        return E_meV
    elif unit == "eV":
        return E_meV * 1e-3
    elif unit == "cm⁻¹":
        return E_meV * 8.0655
    elif unit == "nm":
        return np.where(E_meV > 0, 1239.8 / E_meV, np.nan)
    return E_meV


def _draw_zpl(ax, zpl_meV, unit):
    if zpl_meV is None or zpl_meV <= 0:
        return
    zpl_u = float(convert_energy(np.array([float(zpl_meV)]), unit)[0])
    ax.axvline(zpl_u, color=DARK["yellow"], lw=1.2, ls="--", alpha=0.85,
               label="ZPL", zorder=5)


def _to_meV(E_arr, unit):
    """Convert energy array from given unit to meV."""
    E_arr = np.asarray(E_arr, dtype=float)
    if unit == "meV":
        return E_arr
    elif unit == "eV":
        return E_arr * 1000.0
    elif unit == "cm⁻¹":
        return E_arr / 8.0655
    elif unit == "nm":
        return np.where(E_arr > 0, 1239.8 / E_arr, np.nan)
    return E_arr


def _apply_exp(exp_dict, data_unit, display_unit, y_scale, x_shift_meV):
    """Return (E_plot, I_plot) with unit conversion, shift, and scale applied."""
    E_meV = _to_meV(exp_dict["E"], data_unit) + float(x_shift_meV)
    return convert_energy(E_meV, display_unit), np.asarray(exp_dict["I"], float) * float(y_scale)


def _card(parent=None):
    f = QFrame(parent)
    f.setObjectName("card")
    return f


def _metric_widget(value_str, label_str, color="#a6e3a1"):
    w = QFrame()
    w.setObjectName("card")
    lay = QVBoxLayout(w)
    lay.setSpacing(4)
    v = QLabel(value_str)
    v.setObjectName("metric_value")
    v.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")
    v.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(v)
    l = QLabel(label_str)
    l.setObjectName("metric_label")
    l.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(l)
    return w, v


def _sep():
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color: #313244; margin: 4px 0;")
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 – Overview / all computed quantities
# ─────────────────────────────────────────────────────────────────────────────
class OverviewTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._inner = QWidget()
        self._lay = QVBoxLayout(self._inner)
        self._lay.setContentsMargins(24, 20, 24, 24)
        self._lay.setSpacing(20)

        scroll.setWidget(self._inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _clear(self):
        while self._lay.count():
            item = self._lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _section(self, title):
        lbl = QLabel(title)
        lbl.setObjectName("section_title")
        lbl.setFont(QFont("Helvetica Neue", 15, QFont.Weight.Bold))
        return lbl

    def _grid_card(self, rows):
        """rows: list of (label_html, value_str, unit_str) tuples"""
        f = QFrame()
        f.setObjectName("card")
        g = QGridLayout(f)
        g.setHorizontalSpacing(24)
        g.setVerticalSpacing(10)
        g.setContentsMargins(16, 14, 16, 14)
        for i, (lbl, val, unit) in enumerate(rows):
            l = QLabel(lbl)
            l.setObjectName("hint_label")
            l.setTextFormat(Qt.TextFormat.RichText)
            l.setStyleSheet("color: #6c7086; font-size: 13px;")
            g.addWidget(l, i, 0)
            v = QLabel(f"<b>{val}</b>" + (f"<span style='color:#6c7086; font-size:12px'>  {unit}</span>" if unit else ""))
            v.setObjectName("field_label")
            v.setTextFormat(Qt.TextFormat.RichText)
            v.setStyleSheet("font-size: 14px;")
            g.addWidget(v, i, 1)
        g.setColumnStretch(2, 1)
        return f

    def populate(self, results: dict, config: dict = None):
        self._clear()
        config = config or {}

        HR  = results.get("HR", None)
        DWF = np.exp(-HR) if HR is not None else None
        Sk  = results.get("Sk", None)
        Ek  = results.get("Ek_gs", None)

        # ── Primary metrics row ──────────────────────────────
        metric_row = QHBoxLayout()
        metric_row.setSpacing(14)
        if HR is not None:
            w, _ = _metric_widget(f"{HR:.4f}", "Total Huang–Rhys Factor  S", "#cba6f7")
            metric_row.addWidget(w)
        if DWF is not None:
            w, _ = _metric_widget(f"{DWF:.5f}", "Debye–Waller Factor  e⁻ˢ", "#89b4fa")
            metric_row.addWidget(w)
        if Ek is not None:
            w, _ = _metric_widget(str(len(Ek)), "Phonon Modes", "#a6adc8")
            metric_row.addWidget(w)
        if "masses" in results:
            w, _ = _metric_widget(str(len(results["masses"])), "Atoms", "#94e2d5")
            metric_row.addWidget(w)
        metric_row.addStretch()
        metric_w = QWidget()
        metric_w.setLayout(metric_row)
        self._lay.addWidget(metric_w)

        # ── Spectral outputs ──────────────────────────────────
        self._lay.addWidget(self._section("Spectral Outputs"))
        spectral_rows = []

        if "standard_hr" in results:
            std = results["standard_hr"]
            E_em  = std["E_photon_emission"]
            I_em  = np.real(std["I_emission"])
            E_abs = std["E_photon_absorption"]
            I_abs = np.real(std["I_absorption"])
            pk_em  = E_em[np.argmax(I_em)]
            pk_abs = E_abs[np.argmax(I_abs)]
            stokes = pk_abs - pk_em
            spectral_rows += [
                ("Emission peak energy (analytical)", f"{pk_em:.2f}", "meV"),
                ("Absorption peak energy (analytical)", f"{pk_abs:.2f}", "meV"),
                ("Stokes shift  (E<sub>abs</sub> − E<sub>em</sub>)", f"{stokes:.2f}", "meV"),
            ]

        if HR is not None and Ek is not None:
            spectral_rows += [
                ("Total Huang–Rhys factor  S = Σ<sub>k</sub> S<sub>k</sub>", f"{HR:.6f}", ""),
                ("Debye–Waller factor  e<sup>−S</sup>", f"{DWF:.6f}", ""),
                ("ZPL linewidth parameter  γ", f"{config.get('gamma', '—')}", "meV"),
            ]

        if Ek is not None:
            spectral_rows += [
                ("GS phonon energy range", f"{Ek.min():.2f} – {Ek.max():.2f}", "meV"),
                ("GS phonon energy range",
                 f"{Ek.min()*8.0655:.1f} – {Ek.max()*8.0655:.1f}", "cm⁻¹"),
                ("GS phonon energy range",
                 f"{Ek.min()/4.13566:.3f} – {Ek.max()/4.13566:.3f}", "THz"),
            ]
            # deduplicate: show meV, cm⁻¹, THz as one row
            spectral_rows = [r for r in spectral_rows if r[0] != "GS phonon energy range"]
            spectral_rows.append((
                "GS phonon energy range",
                f"{Ek.min():.2f}–{Ek.max():.2f} meV  ·  "
                f"{Ek.min()*8.0655:.0f}–{Ek.max()*8.0655:.0f} cm⁻¹  ·  "
                f"{Ek.min()/4.13566:.2f}–{Ek.max()/4.13566:.2f} THz",
                ""
            ))

        if spectral_rows:
            self._lay.addWidget(self._grid_card(spectral_rows))

        # ── Mode analysis ─────────────────────────────────────
        if Sk is not None and Ek is not None:
            self._lay.addWidget(self._section("Mode Analysis"))
            top_n = min(5, len(Sk))
            sort_idx = np.argsort(Sk)[::-1]
            mode_rows = []
            for rank, idx in enumerate(sort_idx[:top_n]):
                mode_rows.append((
                    f"Mode #{idx+1} rank {rank+1}",
                    f"S<sub>k</sub> = {Sk[idx]:.5f}  ·  "
                    f"{Ek[idx]:.2f} meV  ({Ek[idx]*8.0655:.1f} cm⁻¹)",
                    ""
                ))
            cum5 = np.sum(Sk[sort_idx[:5]])
            mode_rows.append((
                "Cumulative HR (top 5 modes)",
                f"{cum5:.5f}  ({100*cum5/HR:.1f}% of total S)" if HR else f"{cum5:.5f}",
                ""
            ))
            if "IPR_gs" in results:
                ipr = results["IPR_gs"]
                mode_rows.append((
                    "Most localized mode  (lowest IPR)",
                    f"Mode #{np.argmin(ipr)+1}  ·  IPR = {ipr.min():.4f}",
                    ""
                ))
                mode_rows.append((
                    "Least localized mode  (highest IPR)",
                    f"Mode #{np.argmax(ipr)+1}  ·  IPR = {ipr.max():.4f}",
                    ""
                ))
            self._lay.addWidget(self._grid_card(mode_rows))

        # ── Monte Carlo statistics ─────────────────────────────
        if "monte_carlo_emission" in results:
            self._lay.addWidget(self._section("Monte Carlo Emission Statistics"))
            mc = results["monte_carlo_emission"]
            mc_rows = [
                ("Mean photon energy  ⟨E⟩",        f"{mc['mean']:.3f}",     "meV"),
                ("Median photon energy",             f"{mc['median']:.3f}",   "meV"),
                ("Mode (most probable energy)",      f"{mc['mode']:.3f}",     "meV"),
                ("Standard deviation  σ<sub>MC</sub>", f"{mc['std']:.3f}",   "meV"),
                ("Skewness  γ₁",                    f"{mc['skewness']:.4f}", ""),
                ("Excess kurtosis  γ₂",             f"{mc['kurtosis']:.4f}", ""),
                ("FWHM (2.355·σ)",                  f"{2.355*mc['std']:.3f}", "meV"),
            ]
            self._lay.addWidget(self._grid_card(mc_rows))

        # ── Squeezed model ─────────────────────────────────────
        if results.get("squeezed") and "Ek_es" in results:
            self._lay.addWidget(self._section("Displaced–Squeezed Oscillator"))
            sq = results["squeezed"]
            Ek_es = results["Ek_es"]
            rk = sq.get("rk", np.zeros(1))
            sq_rows = [
                ("ES phonon energy range",
                 f"{Ek_es.min():.2f}–{Ek_es.max():.2f}", "meV"),
                ("Max squeezing parameter  |r<sub>k</sub>|<sub>max</sub>",
                 f"{np.abs(rk).max():.4f}", ""),
                ("RMS squeezing  √⟨r<sub>k</sub>²⟩",
                 f"{np.sqrt(np.mean(rk**2)):.4f}", ""),
            ]
            if Sk is not None:
                n_em  = float(np.sum(Sk + np.sinh(rk) ** 2))
                n_abs = float(np.sum(Sk * np.exp(2 * rk) + np.sinh(rk) ** 2))
                sq_rows += [
                    ("Mean phonon number  ⟨n⟩<sub>em</sub>"
                     " = Σ<sub>k</sub>(S<sub>k</sub> + sinh²r<sub>k</sub>)",
                     f"{n_em:.4f}", ""),
                    ("Mean phonon number  ⟨n⟩<sub>abs</sub>"
                     " = Σ<sub>k</sub>(S<sub>k</sub>e<sup>2r<sub>k</sub></sup>"
                     " + sinh²r<sub>k</sub>)",
                     f"{n_abs:.4f}", ""),
                ]
            self._lay.addWidget(self._grid_card(sq_rows))

        # ── Atom composition ─────────────────────────────────
        if "atoms" in results:
            self._lay.addWidget(self._section("System Composition"))
            atoms = results["atoms"]
            comp = "  ".join(f"{k}×{v}" for k, v in atoms.items())
            self._lay.addWidget(self._grid_card([("Composition", comp, "")]))

        # ── Parameters used ──────────────────────────────────
        self._lay.addWidget(self._section("Calculation Parameters"))
        param_rows = [
            ("Zero-phonon line  E<sub>ZPL</sub>",
             f"{config.get('zpl', '—')}", "meV"),
            ("Phonon sideband broadening  σ<sub>1</sub>",
             f"{config.get('sigma_init', '—')}", "meV"),
            ("Phonon sideband broadening  σ<sub>2</sub>",
             f"{config.get('sigma_final', '—')}", "meV"),
            ("Overall spectral broadening  γ",
             f"{config.get('gamma', '—')}", "meV"),
            ("Temperature  T",
             f"{config.get('temperature', 0.0)}", "K"),
            ("Modes subtracted (low-freq.)",
             f"{config.get('subtract_modes', 0)}", ""),
            ("Sideband line shape",
             "Lorentzian" if config.get("sidebands_broadening_lorentzian") else "Gaussian", ""),
            ("Monte Carlo sampling",
             "Yes" if config.get("monte_carlo_emission") else "No", ""),
            ("Squeezed oscillator model",
             "Yes" if config.get("enable_squeezing") else "No", ""),
            ("Workflow",
             config.get("workflow", "—").replace("_", " ").title(), ""),
        ]
        self._lay.addWidget(self._grid_card(param_rows))
        self._lay.addStretch()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 – Spectral function S(E) + Sk bars  (interactive)
# ─────────────────────────────────────────────────────────────────────────────
class SpectralFunctionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = None
        self._cursor = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Energy unit:"))
        self._unit = QComboBox()
        self._unit.addItems(["meV", "eV", "cm⁻¹", "nm"])
        self._unit.setFixedWidth(90)
        self._unit.currentTextChanged.connect(self._replot)
        ctrl.addWidget(self._unit)
        ctrl.addStretch()
        info = QLabel("Hover over a bar to inspect that phonon mode")
        info.setObjectName("hint_label")
        ctrl.addWidget(info)
        lay.addLayout(ctrl)

        self._canvas = PlotCanvas(nrows=1, ncols=1, figsize=(9, 4.5))
        lay.addWidget(self._canvas, 1)

        self._hover_card = QFrame()
        self._hover_card.setObjectName("info_card")
        self._hover_card.setVisible(False)
        hc_lay = QHBoxLayout(self._hover_card)
        hc_lay.setSpacing(28)
        self._hc_labels = {}
        for key in ["Mode", "Energy (meV)", "Frequency", "Sₖ", "IPR", "Cumul. HR"]:
            col = QVBoxLayout()
            col.setSpacing(2)
            v = QLabel("—")
            v.setObjectName("field_label")
            v.setStyleSheet("color:#cba6f7; font-size:15px; font-weight:bold;")
            l = QLabel(key)
            l.setObjectName("hint_label")
            col.addWidget(v)
            col.addWidget(l)
            hc_lay.addLayout(col)
            self._hc_labels[key] = v
        hc_lay.addStretch()
        lay.addWidget(self._hover_card)

    def populate(self, results: dict):
        self._results = results
        self._replot()

    def _replot(self):
        if self._results is None:
            return
        results = self._results
        unit = self._unit.currentText()

        if "standard_hr" not in results or "Sk" not in results:
            return

        std  = results["standard_hr"]
        Sk   = results["Sk"]
        Ek_gs = results["Ek_gs"]
        HR   = results["HR"]

        E_ph  = convert_energy(std["E_phonons"], unit)
        S_E   = std["S_E"]
        Ek_u  = convert_energy(Ek_gs, unit)

        ax = self._canvas.ax
        ax.cla()
        ax2 = ax.twinx()
        ax2.set_facecolor(DARK["axes_bg"])
        ax2.tick_params(colors=DARK["text"], labelsize=9)
        for spine in ax2.spines.values():
            spine.set_edgecolor(DARK["spine"])

        ax.plot(E_ph, S_E, color=DARK["blue"], linewidth=1.9,
                label=f"S(E)  [{UNIT_LABELS[unit]}⁻¹]", zorder=3)
        ax.set_xlabel(f"Phonon Energy  ({UNIT_LABELS[unit]})", color=DARK["text"])
        ax.set_ylabel(f"Spectral function  S(E)  [{UNIT_LABELS[unit]}⁻¹]", color=DARK["blue"])
        ax.tick_params(axis="y", colors=DARK["blue"])

        bar_width = (Ek_u.max() - Ek_u.min()) * 0.012 if len(Ek_u) > 1 else 1.0
        bars = ax2.bar(Ek_u, Sk, width=bar_width,
                       color=DARK["red"], alpha=0.7, label="Sₖ  (per mode)", zorder=2)
        ax2.set_ylabel("Huang–Rhys factor  Sₖ  (per mode)", color=DARK["red"])
        ax2.tick_params(axis="y", colors=DARK["red"])

        ax.set_title(
            f"Spectral Function S(E)   |   S = {HR:.4f}   |   e⁻ˢ = {np.exp(-HR):.5f}",
            color=DARK["text"]
        )

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  facecolor=DARK["axes_bg"], edgecolor=DARK["spine"],
                  labelcolor=DARK["text"], fontsize=10, loc="upper right")

        self._canvas.fig.tight_layout()
        apply_dark_to_twin(self._canvas.fig, [ax, ax2])
        self._canvas.canvas.draw_idle()

        if self._cursor:
            try:
                self._cursor.remove()
            except Exception:
                pass

        freqs_thz = Ek_gs / 4.13566
        freqs_cm  = Ek_gs * 8.0655
        cum_hr    = np.cumsum(Sk)
        ipr       = results.get("IPR_gs", np.full_like(Sk, np.nan))
        card      = self._hover_card
        hc        = self._hc_labels

        self._cursor = mplcursors.cursor(bars, hover=True)

        @self._cursor.connect("add")
        def on_add(sel, _u=unit):
            idx = sel.index
            sel.annotation.set_text(
                f"Mode #{idx + 1}\n"
                f"Energy: {Ek_u[idx]:.3f} {_u}\n"
                f"Sk: {Sk[idx]:.6f}"
            )
            sel.annotation.get_bbox_patch().set(
                fc=DARK["axes_bg"], ec=DARK["spine"], alpha=0.92
            )
            sel.annotation.set_color(DARK["text"])
            sel.annotation.set_fontsize(11)
            hc["Mode"].setText(f"#{idx + 1}")
            hc["Energy (meV)"].setText(f"{Ek_gs[idx]:.3f}")
            hc["Frequency"].setText(
                f"{freqs_thz[idx]:.3f} THz  /  {freqs_cm[idx]:.1f} cm⁻¹"
            )
            hc["Sₖ"].setText(f"{Sk[idx]:.6f}")
            hc["IPR"].setText(f"{ipr[idx]:.2f}" if not np.isnan(ipr[idx]) else "—")
            hc["Cumul. HR"].setText(f"{cum_hr[idx]:.5f}")
            card.setVisible(True)

        @self._cursor.connect("remove")
        def on_remove(sel):
            card.setVisible(False)


def apply_dark_to_twin(fig, axes):
    fig.patch.set_facecolor(DARK["bg"])
    for ax in axes:
        ax.set_facecolor(DARK["axes_bg"])
        ax.grid(True, color=DARK["grid"], linewidth=0.5, alpha=0.8)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 – Emission spectra  (interactive)
# ─────────────────────────────────────────────────────────────────────────────
class EmissionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = None
        self._cursors = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Energy unit:"))
        self._unit = QComboBox()
        self._unit.addItems(["meV", "eV", "cm⁻¹", "nm"])
        self._unit.setFixedWidth(90)
        self._unit.currentTextChanged.connect(self._replot)
        ctrl.addWidget(self._unit)
        ctrl.addSpacing(20)
        self._show_analytical = QCheckBox("Analytical")
        self._show_analytical.setChecked(True)
        self._show_analytical.toggled.connect(self._replot)
        self._show_mc = QCheckBox("Monte Carlo")
        self._show_mc.setChecked(True)
        self._show_mc.toggled.connect(self._replot)
        self._show_exp_em = QCheckBox("Experiment")
        self._show_exp_em.setChecked(True)
        self._show_exp_em.toggled.connect(self._replot)
        ctrl.addWidget(self._show_analytical)
        ctrl.addSpacing(16)
        ctrl.addWidget(self._show_mc)
        ctrl.addSpacing(16)
        ctrl.addWidget(self._show_exp_em)
        ctrl.addStretch()
        lay.addLayout(ctrl)

        # ── Experimental data controls (shown only when exp data is loaded) ────
        self._exp_ctrl_row_em = QFrame()
        self._exp_ctrl_row_em.setObjectName("info_card")
        self._exp_ctrl_row_em.setVisible(False)
        _ec = QHBoxLayout(self._exp_ctrl_row_em)
        _ec.setContentsMargins(8, 4, 8, 4)
        _ec.setSpacing(10)
        _ec.addWidget(QLabel("Exp. unit:"))
        self._exp_unit_em = QComboBox()
        self._exp_unit_em.addItems(["meV", "eV", "cm⁻¹", "nm"])
        self._exp_unit_em.setCurrentText("eV")
        self._exp_unit_em.setFixedWidth(80)
        self._exp_unit_em.currentTextChanged.connect(self._replot)
        _ec.addWidget(self._exp_unit_em)
        _ec.addSpacing(6)
        _ec.addWidget(QLabel("Y-scale:"))
        self._exp_yscale_em = QDoubleSpinBox()
        self._exp_yscale_em.setRange(0.0, 1e9)
        self._exp_yscale_em.setValue(1.0)
        self._exp_yscale_em.setDecimals(4)
        self._exp_yscale_em.setSingleStep(0.1)
        self._exp_yscale_em.setFixedWidth(100)
        self._exp_yscale_em.valueChanged.connect(self._replot)
        _ec.addWidget(self._exp_yscale_em)
        _ec.addSpacing(6)
        _ec.addWidget(QLabel("X-shift (meV):"))
        self._exp_xshift_em = QDoubleSpinBox()
        self._exp_xshift_em.setRange(-1e5, 1e5)
        self._exp_xshift_em.setValue(0.0)
        self._exp_xshift_em.setDecimals(2)
        self._exp_xshift_em.setSingleStep(1.0)
        self._exp_xshift_em.setFixedWidth(110)
        self._exp_xshift_em.valueChanged.connect(self._replot)
        _ec.addWidget(self._exp_xshift_em)
        _ec.addSpacing(12)
        self._exp_export_em_btn = QPushButton("Export scaled")
        self._exp_export_em_btn.clicked.connect(self._do_export_em)
        _ec.addWidget(self._exp_export_em_btn)
        _ec.addStretch()
        lay.addWidget(self._exp_ctrl_row_em)
        self._export_cb = None

        self._canvas = PlotCanvas(figsize=(9, 4.5))
        lay.addWidget(self._canvas, 1)

        self._stats_frame = QFrame()
        self._stats_frame.setObjectName("card")
        self._stats_frame.setVisible(False)
        sf_lay = QHBoxLayout(self._stats_frame)
        sf_lay.setSpacing(24)
        self._stat_labels = {}
        for key in ["Mean", "Median", "Mode", "Std Dev", "FWHM", "Skewness", "Kurtosis"]:
            col = QVBoxLayout(); col.setSpacing(2)
            v = QLabel("—"); v.setObjectName("field_label")
            v.setStyleSheet("color:#89b4fa; font-size:15px; font-weight:bold;")
            l = QLabel(key); l.setObjectName("hint_label")
            col.addWidget(v); col.addWidget(l)
            sf_lay.addLayout(col)
            self._stat_labels[key] = v
        sf_lay.addStretch()
        lay.addWidget(self._stats_frame)

    def _remove_cursors(self):
        for c in self._cursors:
            try:
                c.remove()
            except Exception:
                pass
        self._cursors.clear()

    def populate(self, results: dict):
        self._results = results
        has_mc  = "monte_carlo_emission" in results
        has_exp = "exp_emission" in results
        self._show_mc.setEnabled(has_mc)
        self._show_mc.setChecked(has_mc)
        self._show_exp_em.setEnabled(has_exp)
        self._show_exp_em.setChecked(has_exp)
        self._exp_ctrl_row_em.setVisible(has_exp)
        self._replot()

    def _replot(self):
        if self._results is None:
            return
        self._remove_cursors()
        results = self._results
        unit = self._unit.currentText()
        u_lbl = UNIT_LABELS[unit]

        ax = self._canvas.ax
        ax.cla()

        has_std = "standard_hr" in results
        has_mc  = "monte_carlo_emission" in results

        zpl = results.get("_zpl_meV")

        if has_std and self._show_analytical.isChecked():
            std = results["standard_hr"]
            E = convert_energy(std["E_photon_emission"], unit)
            I = np.real(std["I_emission"])
            line, = ax.plot(E, I, color=DARK["blue"], linewidth=2,
                            label="Analytical (generating function)", zorder=3)

            c = mplcursors.cursor(line, hover=True)
            @c.connect("add")
            def on_add_line(sel, _u=u_lbl):
                x, y = sel.target
                sel.annotation.set_text(
                    f"Photon Energy: {x:.3f} {_u}\nL(E): {y:.4e}"
                )
                sel.annotation.get_bbox_patch().set(fc=DARK["axes_bg"], ec=DARK["spine"], alpha=0.92)
                sel.annotation.set_color(DARK["text"])
                sel.annotation.set_fontsize(11)
            self._cursors.append(c)

        if has_mc and self._show_mc.isChecked():
            mc = results["monte_carlo_emission"]
            E_mc = convert_energy(mc["E_photon_emission"], unit)
            I_mc = mc["I_emission"]
            w = np.mean(np.diff(E_mc)) * 0.95
            bars = ax.bar(E_mc, I_mc, width=w,
                          color=DARK["red"], alpha=0.55,
                          label="Monte Carlo", zorder=2)

            c = mplcursors.cursor(bars, hover=True)
            @c.connect("add")
            def on_add_mc(sel, _u=u_lbl):
                x, y = sel.target
                sel.annotation.set_text(
                    f"Photon Energy: {x:.3f} {_u}\nMC L(E): {y:.4e}"
                )
                sel.annotation.get_bbox_patch().set(fc=DARK["axes_bg"], ec=DARK["spine"], alpha=0.92)
                sel.annotation.set_color(DARK["text"])
                sel.annotation.set_fontsize(11)
            self._cursors.append(c)

            fwhm = 2.355 * mc["std"]
            sl = self._stat_labels
            sl["Mean"].setText(f"{mc['mean']:.2f} {u_lbl}")
            sl["Median"].setText(f"{mc['median']:.2f} {u_lbl}")
            sl["Mode"].setText(f"{mc['mode']:.2f} {u_lbl}")
            sl["Std Dev"].setText(f"{mc['std']:.2f} {u_lbl}")
            sl["FWHM"].setText(f"{fwhm:.2f} {u_lbl}")
            sl["Skewness"].setText(f"{mc['skewness']:.3f}")
            sl["Kurtosis"].setText(f"{mc['kurtosis']:.3f}")
            self._stats_frame.setVisible(True)
        else:
            self._stats_frame.setVisible(False)

        exp = results.get("exp_emission")
        if exp and self._show_exp_em.isChecked():
            E_exp, I_exp = _apply_exp(
                exp,
                self._exp_unit_em.currentText(),
                unit,
                self._exp_yscale_em.value(),
                self._exp_xshift_em.value(),
            )
            line_exp, = ax.plot(E_exp, I_exp, color=DARK["peach"], lw=1.8,
                                ls="-", label="Experiment", zorder=4)
            c_exp = mplcursors.cursor(line_exp, hover=True)
            @c_exp.connect("add")
            def on_add_exp(sel, _u=u_lbl):
                x, y = sel.target
                sel.annotation.set_text(
                    f"Photon Energy: {x:.3f} {_u}\nExp. Intensity: {y:.4e}"
                )
                sel.annotation.get_bbox_patch().set(fc=DARK["axes_bg"], ec=DARK["spine"], alpha=0.92)
                sel.annotation.set_color(DARK["text"])
                sel.annotation.set_fontsize(11)
            self._cursors.append(c_exp)

        _draw_zpl(ax, zpl, unit)
        ax.set_xlabel(f"Photon Energy  ({u_lbl})", color=DARK["text"])
        ax.set_ylabel("L(E)  [arb. units]", color=DARK["text"])
        ax.set_title("Photoluminescence Emission Spectrum", color=DARK["text"])
        if ax.get_lines() or ax.patches:
            ax.legend(facecolor=DARK["axes_bg"], edgecolor=DARK["spine"],
                      labelcolor=DARK["text"], fontsize=10)
        self._canvas.draw()

    def _do_export_em(self):
        if not (self._results and self._export_cb):
            return
        exp = self._results.get("exp_emission")
        if not exp:
            return
        E_meV = _to_meV(exp["E"], self._exp_unit_em.currentText())
        E_out = E_meV + self._exp_xshift_em.value()
        I_out = np.asarray(exp["I"], float) * self._exp_yscale_em.value()
        self._export_cb("exp_emission_scaled", E_out, I_out)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 – Absorption spectra  (interactive)
# ─────────────────────────────────────────────────────────────────────────────
class AbsorptionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = None
        self._cursors = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Energy unit:"))
        self._unit = QComboBox()
        self._unit.addItems(["meV", "eV", "cm⁻¹", "nm"])
        self._unit.setFixedWidth(90)
        self._unit.currentTextChanged.connect(self._replot)
        ctrl.addWidget(self._unit)
        ctrl.addSpacing(20)
        self._overlay = QCheckBox("Overlay emission")
        self._overlay.toggled.connect(self._replot)
        self._show_exp_abs = QCheckBox("Experiment")
        self._show_exp_abs.setChecked(True)
        self._show_exp_abs.toggled.connect(self._replot)
        ctrl.addWidget(self._overlay)
        ctrl.addSpacing(16)
        ctrl.addWidget(self._show_exp_abs)
        ctrl.addStretch()
        lay.addLayout(ctrl)

        # ── Experimental data controls (shown only when exp data is loaded) ────
        self._exp_ctrl_row_abs = QFrame()
        self._exp_ctrl_row_abs.setObjectName("info_card")
        self._exp_ctrl_row_abs.setVisible(False)
        _ec = QHBoxLayout(self._exp_ctrl_row_abs)
        _ec.setContentsMargins(8, 4, 8, 4)
        _ec.setSpacing(10)
        _ec.addWidget(QLabel("Exp. unit:"))
        self._exp_unit_abs = QComboBox()
        self._exp_unit_abs.addItems(["meV", "eV", "cm⁻¹", "nm"])
        self._exp_unit_abs.setCurrentText("eV")
        self._exp_unit_abs.setFixedWidth(80)
        self._exp_unit_abs.currentTextChanged.connect(self._replot)
        _ec.addWidget(self._exp_unit_abs)
        _ec.addSpacing(6)
        _ec.addWidget(QLabel("Y-scale:"))
        self._exp_yscale_abs = QDoubleSpinBox()
        self._exp_yscale_abs.setRange(0.0, 1e9)
        self._exp_yscale_abs.setValue(1.0)
        self._exp_yscale_abs.setDecimals(4)
        self._exp_yscale_abs.setSingleStep(0.1)
        self._exp_yscale_abs.setFixedWidth(100)
        self._exp_yscale_abs.valueChanged.connect(self._replot)
        _ec.addWidget(self._exp_yscale_abs)
        _ec.addSpacing(6)
        _ec.addWidget(QLabel("X-shift (meV):"))
        self._exp_xshift_abs = QDoubleSpinBox()
        self._exp_xshift_abs.setRange(-1e5, 1e5)
        self._exp_xshift_abs.setValue(0.0)
        self._exp_xshift_abs.setDecimals(2)
        self._exp_xshift_abs.setSingleStep(1.0)
        self._exp_xshift_abs.setFixedWidth(110)
        self._exp_xshift_abs.valueChanged.connect(self._replot)
        _ec.addWidget(self._exp_xshift_abs)
        _ec.addSpacing(12)
        self._exp_export_abs_btn = QPushButton("Export scaled")
        self._exp_export_abs_btn.clicked.connect(self._do_export_abs)
        _ec.addWidget(self._exp_export_abs_btn)
        _ec.addStretch()
        lay.addWidget(self._exp_ctrl_row_abs)
        self._export_cb = None

        self._canvas = PlotCanvas(figsize=(9, 4.5))
        lay.addWidget(self._canvas, 1)

    def _remove_cursors(self):
        for c in self._cursors:
            try:
                c.remove()
            except Exception:
                pass
        self._cursors.clear()

    def populate(self, results: dict):
        self._results = results
        has_exp = "exp_absorption" in results
        self._show_exp_abs.setEnabled(has_exp)
        self._show_exp_abs.setChecked(has_exp)
        self._exp_ctrl_row_abs.setVisible(has_exp)
        self._replot()

    def _replot(self):
        if self._results is None:
            return
        self._remove_cursors()
        results = self._results
        unit = self._unit.currentText()
        u_lbl = UNIT_LABELS[unit]
        ax = self._canvas.ax
        ax.cla()

        zpl = results.get("_zpl_meV")

        if "standard_hr" in results:
            std = results["standard_hr"]
            E_abs = convert_energy(std["E_photon_absorption"], unit)
            I_abs = np.real(std["I_absorption"])
            line_a, = ax.plot(E_abs, I_abs, color=DARK["green"], linewidth=2,
                              label="Absorption", zorder=3)

            c_a = mplcursors.cursor(line_a, hover=True)
            @c_a.connect("add")
            def on_add_abs(sel, _u=u_lbl):
                x, y = sel.target
                sel.annotation.set_text(
                    f"Photon Energy: {x:.3f} {_u}\nL(E): {y:.4e}"
                )
                sel.annotation.get_bbox_patch().set(fc=DARK["axes_bg"], ec=DARK["spine"], alpha=0.92)
                sel.annotation.set_color(DARK["text"])
                sel.annotation.set_fontsize(11)
            self._cursors.append(c_a)

            if self._overlay.isChecked():
                E_em = convert_energy(std["E_photon_emission"], unit)
                I_em = np.real(std["I_emission"])
                line_e, = ax.plot(E_em, I_em, color=DARK["blue"], linewidth=2,
                                  linestyle="--", label="Emission", zorder=2, alpha=0.8)

                c_e = mplcursors.cursor(line_e, hover=True)
                @c_e.connect("add")
                def on_add_em(sel, _u=u_lbl):
                    x, y = sel.target
                    sel.annotation.set_text(
                        f"Photon Energy: {x:.3f} {_u}\nL(E): {y:.4e}"
                    )
                    sel.annotation.get_bbox_patch().set(fc=DARK["axes_bg"], ec=DARK["spine"], alpha=0.92)
                    sel.annotation.set_color(DARK["text"])
                    sel.annotation.set_fontsize(11)
                self._cursors.append(c_e)

        exp = results.get("exp_absorption")
        if exp and self._show_exp_abs.isChecked():
            E_exp, I_exp = _apply_exp(
                exp,
                self._exp_unit_abs.currentText(),
                unit,
                self._exp_yscale_abs.value(),
                self._exp_xshift_abs.value(),
            )
            line_exp, = ax.plot(E_exp, I_exp, color=DARK["peach"], lw=1.8,
                                label="Experiment", zorder=4)
            c_exp = mplcursors.cursor(line_exp, hover=True)
            @c_exp.connect("add")
            def on_add_exp(sel, _u=u_lbl):
                x, y = sel.target
                sel.annotation.set_text(
                    f"Photon Energy: {x:.3f} {_u}\nExp. Intensity: {y:.4e}"
                )
                sel.annotation.get_bbox_patch().set(fc=DARK["axes_bg"], ec=DARK["spine"], alpha=0.92)
                sel.annotation.set_color(DARK["text"])
                sel.annotation.set_fontsize(11)
            self._cursors.append(c_exp)

        _draw_zpl(ax, zpl, unit)
        ax.set_xlabel(f"Photon Energy  ({u_lbl})", color=DARK["text"])
        ax.set_ylabel("L(E)  [arb. units]", color=DARK["text"])
        ax.set_title("Optical Absorption Spectrum", color=DARK["text"])
        if ax.get_lines():
            ax.legend(facecolor=DARK["axes_bg"], edgecolor=DARK["spine"],
                      labelcolor=DARK["text"], fontsize=10)
        self._canvas.draw()

    def _do_export_abs(self):
        if not (self._results and self._export_cb):
            return
        exp = self._results.get("exp_absorption")
        if not exp:
            return
        E_meV = _to_meV(exp["E"], self._exp_unit_abs.currentText())
        E_out = E_meV + self._exp_xshift_abs.value()
        I_out = np.asarray(exp["I"], float) * self._exp_yscale_abs.value()
        self._export_cb("exp_absorption_scaled", E_out, I_out)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 – Mode analysis  (interactive)
# ─────────────────────────────────────────────────────────────────────────────
class _SkScatterTab(QWidget):
    """Sk vs phonon energy scatter plot (inner sub-tab of ModeAnalysisTab)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = None
        self._cursor  = None
        self._cbar    = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        info = QLabel("Hover over a point — mode details appear as a floating tooltip")
        info.setObjectName("hint_label")
        lay.addWidget(info)

        self._canvas = PlotCanvas(nrows=1, ncols=1, figsize=(9, 5))
        lay.addWidget(self._canvas, 1)

        self._hover_card = QFrame()
        self._hover_card.setObjectName("info_card")
        self._hover_card.setVisible(False)
        hc_lay = QHBoxLayout(self._hover_card)
        hc_lay.setSpacing(24)
        self._hc = {}
        for key in ["Mode #", "Energy (meV)", "Freq (THz)", "Freq (cm⁻¹)", "Sₖ", "IPR"]:
            col = QVBoxLayout(); col.setSpacing(2)
            v = QLabel("—"); v.setObjectName("field_label")
            v.setStyleSheet("color:#cba6f7; font-size:14px; font-weight:bold;")
            l = QLabel(key); l.setObjectName("hint_label")
            col.addWidget(v); col.addWidget(l)
            hc_lay.addLayout(col)
            self._hc[key] = v
        hc_lay.addStretch()
        lay.addWidget(self._hover_card)

    def populate(self, results: dict):
        self._results = results
        self._replot()

    def _replot(self):
        if self._results is None:
            return
        results = self._results
        if "Sk" not in results:
            return

        Sk        = results["Sk"]
        Ek        = results["Ek_gs"]
        ipr       = results.get("IPR_gs", np.ones_like(Sk))
        freqs_thz = Ek / 4.13566
        freqs_cm  = Ek * 8.0655

        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        ax = self._canvas.ax
        ax.cla()

        ipr_norm = (ipr - ipr.min()) / (ipr.max() - ipr.min() + 1e-12)

        sc = ax.scatter(Ek, Sk, c=ipr_norm, cmap="plasma",
                        s=np.clip(Sk * 400, 10, 800),
                        alpha=0.75, edgecolors=DARK["spine"], linewidths=0.5)
        ax.set_xlabel("Phonon Energy  (meV)", color=DARK["text"])
        ax.set_ylabel("Huang–Rhys factor  Sk", color=DARK["text"])
        ax.set_title(
            r"$S_k$ vs Phonon Energy  (size $\propto S_k$,  colour = IPR)",
            color=DARK["text"]
        )

        self._cbar = self._canvas.fig.colorbar(sc, ax=ax, pad=0.01)
        self._cbar.set_label("IPR (normalised)", color=DARK["text"])
        self._cbar.ax.tick_params(colors=DARK["text"])

        self._canvas.draw()

        hc   = self._hc
        card = self._hover_card

        if self._cursor:
            try:
                self._cursor.remove()
            except Exception:
                pass

        self._cursor = mplcursors.cursor(sc, hover=True)

        @self._cursor.connect("add")
        def on_add(sel):
            i = sel.index
            sel.annotation.set_text(
                f"Mode #{i + 1}\n"
                f"Energy: {Ek[i]:.3f} meV\n"
                f"Freq: {freqs_thz[i]:.3f} THz  /  {freqs_cm[i]:.1f} cm⁻¹\n"
                f"Sk: {Sk[i]:.6f}\n"
                f"IPR: {ipr[i]:.4f}"
            )
            sel.annotation.get_bbox_patch().set(
                fc=DARK["axes_bg"], ec=DARK["spine"], alpha=0.92
            )
            sel.annotation.set_color(DARK["text"])
            sel.annotation.set_fontsize(10)
            hc["Mode #"].setText(f"#{i + 1}")
            hc["Energy (meV)"].setText(f"{Ek[i]:.4f}")
            hc["Freq (THz)"].setText(f"{freqs_thz[i]:.4f}")
            hc["Freq (cm⁻¹)"].setText(f"{freqs_cm[i]:.2f}")
            hc["Sₖ"].setText(f"{Sk[i]:.6f}")
            hc["IPR"].setText(f"{ipr[i]:.4f}")
            card.setVisible(True)

        @self._cursor.connect("remove")
        def on_remove(sel):
            card.setVisible(False)


class ModeAnalysisTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        lay.addWidget(self._tabs, 1)

        self._sk_tab      = _SkScatterTab()
        self._phonon_tab  = PhononViewerWidget()

        self._tabs.addTab(self._sk_tab,     "Sk and IPR")
        self._tabs.addTab(self._phonon_tab, "Normal Mode Vectors")

    def populate(self, results: dict):
        self._sk_tab.populate(results)
        if "modes_gs" in results and "R_gs" in results:
            self._phonon_tab.set_results(results)
        else:
            self._phonon_tab.clear()


def plt_colormap(norm_values):
    return cm.plasma(norm_values)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6 – Advanced / generating function / squeezed
# ─────────────────────────────────────────────────────────────────────────────
class AdvancedTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = None
        self._cursors = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        lay.addWidget(self._tabs)

        self._gt_canvas = PlotCanvas(figsize=(9, 4))
        self._tabs.addTab(self._gt_canvas, "Generating Function  G(t)")

        self._sq_canvas = PlotCanvas(nrows=2, ncols=2, figsize=(10, 7))
        self._tabs.addTab(self._sq_canvas, "Squeezed Oscillator")

        # Overlay tab — 2x2: top row always shown, bottom row for squeezed
        self._ov_widget = QWidget()
        ov_lay = QVBoxLayout(self._ov_widget)
        ov_lay.setContentsMargins(8, 8, 8, 8)
        ov_lay.setSpacing(8)
        ov_ctrl = QHBoxLayout()
        ov_ctrl.addWidget(QLabel("Energy unit:"))
        self._ov_unit = QComboBox()
        self._ov_unit.addItems(["meV", "eV", "cm⁻¹", "nm"])
        self._ov_unit.setFixedWidth(90)
        self._ov_unit.currentTextChanged.connect(self._replot_overlay)
        ov_ctrl.addWidget(self._ov_unit)
        ov_ctrl.addStretch()
        self._ov_hint = QLabel("")
        self._ov_hint.setObjectName("hint_label")
        ov_ctrl.addWidget(self._ov_hint)
        ov_lay.addLayout(ov_ctrl)

        # Exp emission controls for overlay
        self._ov_exp_em_row = QFrame()
        self._ov_exp_em_row.setObjectName("info_card")
        self._ov_exp_em_row.setVisible(False)
        _oem = QHBoxLayout(self._ov_exp_em_row)
        _oem.setContentsMargins(8, 3, 8, 3)
        _oem.setSpacing(10)
        _oem.addWidget(QLabel("Em. exp. unit:"))
        self._ov_exp_unit_em = QComboBox()
        self._ov_exp_unit_em.addItems(["meV", "eV", "cm⁻¹", "nm"])
        self._ov_exp_unit_em.setCurrentText("eV")
        self._ov_exp_unit_em.setFixedWidth(75)
        self._ov_exp_unit_em.currentTextChanged.connect(self._replot_overlay)
        _oem.addWidget(self._ov_exp_unit_em)
        _oem.addWidget(QLabel("Y-scale:"))
        self._ov_exp_yscale_em = QDoubleSpinBox()
        self._ov_exp_yscale_em.setRange(0.0, 1e9)
        self._ov_exp_yscale_em.setValue(1.0)
        self._ov_exp_yscale_em.setDecimals(4)
        self._ov_exp_yscale_em.setSingleStep(0.1)
        self._ov_exp_yscale_em.setFixedWidth(95)
        self._ov_exp_yscale_em.valueChanged.connect(self._replot_overlay)
        _oem.addWidget(self._ov_exp_yscale_em)
        _oem.addWidget(QLabel("X-shift (meV):"))
        self._ov_exp_xshift_em = QDoubleSpinBox()
        self._ov_exp_xshift_em.setRange(-1e5, 1e5)
        self._ov_exp_xshift_em.setValue(0.0)
        self._ov_exp_xshift_em.setDecimals(2)
        self._ov_exp_xshift_em.setSingleStep(1.0)
        self._ov_exp_xshift_em.setFixedWidth(105)
        self._ov_exp_xshift_em.valueChanged.connect(self._replot_overlay)
        _oem.addWidget(self._ov_exp_xshift_em)
        _oem.addStretch()
        ov_lay.addWidget(self._ov_exp_em_row)

        # Exp absorption controls for overlay
        self._ov_exp_abs_row = QFrame()
        self._ov_exp_abs_row.setObjectName("info_card")
        self._ov_exp_abs_row.setVisible(False)
        _oabs = QHBoxLayout(self._ov_exp_abs_row)
        _oabs.setContentsMargins(8, 3, 8, 3)
        _oabs.setSpacing(10)
        _oabs.addWidget(QLabel("Abs. exp. unit:"))
        self._ov_exp_unit_abs = QComboBox()
        self._ov_exp_unit_abs.addItems(["meV", "eV", "cm⁻¹", "nm"])
        self._ov_exp_unit_abs.setCurrentText("eV")
        self._ov_exp_unit_abs.setFixedWidth(75)
        self._ov_exp_unit_abs.currentTextChanged.connect(self._replot_overlay)
        _oabs.addWidget(self._ov_exp_unit_abs)
        _oabs.addWidget(QLabel("Y-scale:"))
        self._ov_exp_yscale_abs = QDoubleSpinBox()
        self._ov_exp_yscale_abs.setRange(0.0, 1e9)
        self._ov_exp_yscale_abs.setValue(1.0)
        self._ov_exp_yscale_abs.setDecimals(4)
        self._ov_exp_yscale_abs.setSingleStep(0.1)
        self._ov_exp_yscale_abs.setFixedWidth(95)
        self._ov_exp_yscale_abs.valueChanged.connect(self._replot_overlay)
        _oabs.addWidget(self._ov_exp_yscale_abs)
        _oabs.addWidget(QLabel("X-shift (meV):"))
        self._ov_exp_xshift_abs = QDoubleSpinBox()
        self._ov_exp_xshift_abs.setRange(-1e5, 1e5)
        self._ov_exp_xshift_abs.setValue(0.0)
        self._ov_exp_xshift_abs.setDecimals(2)
        self._ov_exp_xshift_abs.setSingleStep(1.0)
        self._ov_exp_xshift_abs.setFixedWidth(105)
        self._ov_exp_xshift_abs.valueChanged.connect(self._replot_overlay)
        _oabs.addWidget(self._ov_exp_xshift_abs)
        _oabs.addStretch()
        ov_lay.addWidget(self._ov_exp_abs_row)

        self._ov_canvas = PlotCanvas(nrows=2, ncols=2, figsize=(10, 7))
        ov_lay.addWidget(self._ov_canvas, 1)
        self._tabs.addTab(self._ov_widget, "Overlay")

    def _remove_cursors(self):
        for c in self._cursors:
            try:
                c.remove()
            except Exception:
                pass
        self._cursors.clear()

    def populate(self, results: dict):
        self._results = results
        self._remove_cursors()
        has_sq = bool(results.get("squeezed"))
        self._ov_hint.setText(
            "4 panels: Standard HR (top row) + Squeezed (bottom row)"
            if has_sq else
            "2 panels: Standard HR  —  A(E) and L(E)"
        )
        self._ov_exp_em_row.setVisible(bool(results.get("exp_emission")))
        self._ov_exp_abs_row.setVisible(bool(results.get("exp_absorption")))
        self._plot_gt(results)
        if has_sq:
            self._plot_squeezed(results)
        self._replot_overlay()

    def _plot_gt(self, results):
        if "standard_hr" not in results:
            return
        std = results["standard_hr"]
        G_t = std["G_t"]
        t   = std["t_fs"]

        ax = self._gt_canvas.ax
        ax.cla()
        line_re, = ax.plot(t, np.real(G_t), color=DARK["blue"],
                           label="Re[G(t)]", lw=1.9)
        line_im, = ax.plot(t, np.imag(G_t), color=DARK["yellow"],
                           label="Im[G(t)]", lw=1.5, ls="--")
        ax.set_xlabel("Time  (fs)", color=DARK["text"])
        ax.set_ylabel("Generating function  G(t)", color=DARK["text"])
        ax.set_title("Phonon Generating Function  G(t) = exp(S(e^{iωt} - 1))", color=DARK["text"])
        ax.legend(facecolor=DARK["axes_bg"], edgecolor=DARK["spine"],
                  labelcolor=DARK["text"], fontsize=10)

        for line, comp in [(line_re, "Re"), (line_im, "Im")]:
            c = mplcursors.cursor(line, hover=True)
            @c.connect("add")
            def on_add(sel, _comp=comp):
                x, y = sel.target
                sel.annotation.set_text(f"t = {x:.2f} fs\n{_comp}[G(t)] = {y:.4f}")
                sel.annotation.get_bbox_patch().set(fc=DARK["axes_bg"], ec=DARK["spine"], alpha=0.92)
                sel.annotation.set_color(DARK["text"])
                sel.annotation.set_fontsize(11)
            self._cursors.append(c)

        self._gt_canvas.draw()

    def _plot_squeezed(self, results):
        sq    = results["squeezed"]
        Ek_gs = results["Ek_gs"]
        Ek_es = results.get("Ek_es", Ek_gs)
        zpl   = results.get("_zpl_meV")
        axes  = self._sq_canvas.axes
        for ax in axes:
            ax.cla()

        ax1, ax2, ax3, ax4 = axes

        rk = sq.get("rk", np.zeros(len(Ek_gs)))
        ax1.bar(np.arange(1, len(rk) + 1), rk, color=DARK["purple"], alpha=0.8)
        ax1.set_xlabel("Mode index", color=DARK["text"])
        ax1.set_ylabel("Squeezing parameter  rₖ", color=DARK["text"])
        ax1.set_title(
            r"Squeezing parameters  "
            r"$r_k = \frac{1}{2}\ln\!\left(\frac{\omega_{ES,k}}{\omega_{GS,k}}\right)$",
            color=DARK["text"]
        )
        ax1.axhline(0, color=DARK["spine"], lw=0.8)

        ax2.plot(sq["E_phonons"], sq["S_E_emission"],
                 color=DARK["blue"], label="Emission S(E)")
        ax2.plot(sq["E_phonons"], sq["S_E_absorption"],
                 color=DARK["green"], ls="--", label="Absorption S(E)")
        ax2.set_xlabel("Phonon Energy  (meV)", color=DARK["text"])
        ax2.set_ylabel("S(E)  [meV⁻¹]", color=DARK["text"])
        ax2.set_title("Squeezed Spectral Functions", color=DARK["text"])
        ax2.legend(facecolor=DARK["axes_bg"], edgecolor=DARK["spine"],
                   labelcolor=DARK["text"], fontsize=9)

        E_em = convert_energy(sq["E_photon_emission"], "meV")
        ax3.plot(E_em, np.real(sq["I_emission"]), color=DARK["blue"], lw=2)
        _draw_zpl(ax3, zpl, "meV")
        ax3.set_xlabel("Photon Energy  (meV)", color=DARK["text"])
        ax3.set_ylabel("L(E)  [arb. units]", color=DARK["text"])
        ax3.set_title("Squeezed Emission Spectrum", color=DARK["text"])
        if ax3.get_lines():
            ax3.legend(facecolor=DARK["axes_bg"], edgecolor=DARK["spine"],
                       labelcolor=DARK["text"], fontsize=9)

        E_abs = convert_energy(sq["E_photon_absorption"], "meV")
        ax4.plot(E_abs, np.real(sq["I_absorption"]), color=DARK["green"], lw=2)
        _draw_zpl(ax4, zpl, "meV")
        ax4.set_xlabel("Photon Energy  (meV)", color=DARK["text"])
        ax4.set_ylabel("L(E)  [arb. units]", color=DARK["text"])
        ax4.set_title("Squeezed Absorption Spectrum", color=DARK["text"])
        if ax4.get_lines():
            ax4.legend(facecolor=DARK["axes_bg"], edgecolor=DARK["spine"],
                       labelcolor=DARK["text"], fontsize=9)

        self._sq_canvas.draw()

    def _replot_overlay(self):
        if self._results is None:
            return
        results  = self._results
        unit     = self._ov_unit.currentText()
        u_lbl    = UNIT_LABELS[unit]
        has_std  = "standard_hr" in results
        has_sq   = bool(results.get("squeezed"))
        zpl      = results.get("_zpl_meV")
        exp_em   = results.get("exp_emission")
        exp_abs  = results.get("exp_absorption")

        axes = self._ov_canvas.axes
        for ax in axes:
            ax.cla()
            ax.set_visible(False)

        if not has_std:
            self._ov_canvas.draw()
            return

        std = results["standard_hr"]
        E_em = convert_energy(std["E_photon_emission"],   unit)
        E_ab = convert_energy(std["E_photon_absorption"], unit)

        def _style(ax, title, ylabel, xlabel=True):
            ax.set_title(title, color=DARK["text"])
            if xlabel:
                ax.set_xlabel(f"Photon Energy  ({u_lbl})", color=DARK["text"])
            ax.set_ylabel(ylabel, color=DARK["text"])
            ax.legend(facecolor=DARK["axes_bg"], edgecolor=DARK["spine"],
                      labelcolor=DARK["text"], fontsize=9)

        # Panel 0: Standard HR  A(E)
        ax0 = axes[0]
        ax0.set_visible(True)
        ax0.plot(E_em, np.real(std["A_E_emission"]),
                 color=DARK["blue"],  lw=2,      label="Emission")
        ax0.plot(E_ab, np.real(std["A_E_absorption"]),
                 color=DARK["green"], lw=2, ls="--", label="Absorption")
        _draw_zpl(ax0, zpl, unit)
        _style(ax0, "Standard HR  —  A(E)", "A(E)",
               xlabel=not has_sq)

        # Panel 1: Standard HR  L(E)
        ax1 = axes[1]
        ax1.set_visible(True)
        ax1.plot(E_em, np.real(std["I_emission"]),
                 color=DARK["blue"],  lw=2,      label="Emission")
        ax1.plot(E_ab, np.real(std["I_absorption"]),
                 color=DARK["green"], lw=2, ls="--", label="Absorption")
        if exp_em:
            _E, _I = _apply_exp(exp_em, self._ov_exp_unit_em.currentText(), unit,
                                self._ov_exp_yscale_em.value(), self._ov_exp_xshift_em.value())
            ax1.plot(_E, _I, color=DARK["peach"], lw=1.8, label="Exp. Emission")
        if exp_abs:
            _E, _I = _apply_exp(exp_abs, self._ov_exp_unit_abs.currentText(), unit,
                                self._ov_exp_yscale_abs.value(), self._ov_exp_xshift_abs.value())
            ax1.plot(_E, _I, color=DARK["peach"], lw=1.8, ls="--", label="Exp. Absorption")
        _draw_zpl(ax1, zpl, unit)
        _style(ax1, "Standard HR  —  L(E)", "L(E)  [arb. units]",
               xlabel=not has_sq)

        if has_sq:
            sq   = results["squeezed"]
            E_sq_em = convert_energy(sq["E_photon_emission"],   unit)
            E_sq_ab = convert_energy(sq["E_photon_absorption"], unit)

            # Panel 2: Squeezed  A(E)
            ax2 = axes[2]
            ax2.set_visible(True)
            ax2.plot(E_sq_em, np.real(sq["A_E_emission"]),
                     color=DARK["purple"], lw=2,      label="Emission")
            ax2.plot(E_sq_ab, np.real(sq["A_E_absorption"]),
                     color=DARK["teal"],   lw=2, ls="--", label="Absorption")
            _draw_zpl(ax2, zpl, unit)
            _style(ax2, "Squeezed Oscillator  —  A(E)", "A(E)")

            # Panel 3: Squeezed  L(E)
            ax3 = axes[3]
            ax3.set_visible(True)
            ax3.plot(E_sq_em, np.real(sq["I_emission"]),
                     color=DARK["purple"], lw=2,      label="Emission")
            ax3.plot(E_sq_ab, np.real(sq["I_absorption"]),
                     color=DARK["teal"],   lw=2, ls="--", label="Absorption")
            if exp_em:
                _E, _I = _apply_exp(exp_em, self._ov_exp_unit_em.currentText(), unit,
                                    self._ov_exp_yscale_em.value(), self._ov_exp_xshift_em.value())
                ax3.plot(_E, _I, color=DARK["peach"], lw=1.8, label="Exp. Emission")
            if exp_abs:
                _E, _I = _apply_exp(exp_abs, self._ov_exp_unit_abs.currentText(), unit,
                                    self._ov_exp_yscale_abs.value(), self._ov_exp_xshift_abs.value())
                ax3.plot(_E, _I, color=DARK["peach"], lw=1.8, ls="--", label="Exp. Absorption")
            _draw_zpl(ax3, zpl, unit)
            _style(ax3, "Squeezed Oscillator  —  L(E)", "L(E)  [arb. units]")

        self._ov_canvas.fig.tight_layout()
        self._ov_canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Top-level ResultsPage
# ─────────────────────────────────────────────────────────────────────────────
class ResultsPage(QWidget):
    save_requested       = pyqtSignal(str, object, object)  # (label, results, config)
    run_another_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results    = None
        self._config     = None
        self._hdf5_path  = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(8)

        header = QLabel("Results")
        header.setObjectName("section_title")
        header.setFont(QFont("Helvetica Neue", 17, QFont.Weight.Bold))
        lay.addWidget(header)

        # ── Comparison action bar ─────────────────────────────────────────────
        self._action_bar = QFrame()
        self._action_bar.setObjectName("info_card")
        self._action_bar.setVisible(False)
        ab_outer = QVBoxLayout(self._action_bar)
        ab_outer.setContentsMargins(12, 8, 12, 8)
        ab_outer.setSpacing(6)

        ab_row = QHBoxLayout()
        ab_row.setSpacing(10)
        self._comp_count_lbl = QLabel("Not added to comparison")
        self._comp_count_lbl.setStyleSheet("color: #6c7086; font-size: 12px;")
        ab_row.addWidget(self._comp_count_lbl)
        ab_row.addStretch()

        self._add_comp_btn = QPushButton("+ Add to Comparison")
        self._add_comp_btn.setObjectName("secondary_btn")
        self._add_comp_btn.clicked.connect(self._toggle_label_editor)
        ab_row.addWidget(self._add_comp_btn)

        run_another_btn = QPushButton("↺  Run Another Calculation")
        run_another_btn.setObjectName("secondary_btn")
        run_another_btn.clicked.connect(self.run_another_requested)
        ab_row.addWidget(run_another_btn)
        ab_outer.addLayout(ab_row)

        # Inline label editor (hidden until user clicks Add)
        self._label_editor = QFrame()
        self._label_editor.setStyleSheet(
            "QFrame { border: none; background: transparent; }"
        )
        le_lay = QHBoxLayout(self._label_editor)
        le_lay.setContentsMargins(0, 0, 0, 0)
        le_lay.setSpacing(8)
        lbl_lbl = QLabel("Label:")
        lbl_lbl.setStyleSheet("color: #a6adc8; font-size: 12px;")
        le_lay.addWidget(lbl_lbl)
        self._label_edit = QLineEdit("Calculation 1")
        self._label_edit.setFixedWidth(220)
        self._label_edit.setPlaceholderText("e.g. VASP PBE, Gaussian B3LYP, …")
        le_lay.addWidget(self._label_edit)
        confirm_btn = QPushButton("Save")
        confirm_btn.setObjectName("primary_btn")
        confirm_btn.setFixedWidth(70)
        confirm_btn.clicked.connect(self._on_save_confirm)
        le_lay.addWidget(confirm_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondary_btn")
        cancel_btn.setFixedWidth(70)
        cancel_btn.clicked.connect(lambda: self._label_editor.setVisible(False))
        le_lay.addWidget(cancel_btn)
        le_lay.addStretch()
        self._label_editor.setVisible(False)
        ab_outer.addWidget(self._label_editor)

        lay.addWidget(self._action_bar)
        # ─────────────────────────────────────────────────────────────────────

        self._placeholder = QLabel(
            "No results yet — configure inputs and click  ▶ Run Calculation."
        )
        self._placeholder.setObjectName("hint_label")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("font-size:15px; padding:80px;")
        lay.addWidget(self._placeholder)

        self._tabs = QTabWidget()
        self._tabs.setVisible(False)
        lay.addWidget(self._tabs, 1)

        self._tab_overview = OverviewTab()
        self._tab_spectral = SpectralFunctionTab()
        self._tab_emission = EmissionTab()
        self._tab_absorb   = AbsorptionTab()
        self._tab_modes    = ModeAnalysisTab()
        self._tab_advanced = AdvancedTab()

        self._tabs.addTab(self._tab_overview, "ℹ  Overview")
        self._tabs.addTab(self._tab_spectral, "S(E) / Sk")
        self._tabs.addTab(self._tab_emission, "Emission")
        self._tabs.addTab(self._tab_absorb,   "Absorption")
        self._tabs.addTab(self._tab_modes,    "Mode Analysis")
        self._tabs.addTab(self._tab_advanced, "Advanced")

        # Wire export callbacks so tabs can write back to results + HDF5
        self._tab_emission._export_cb = self._export_exp
        self._tab_absorb._export_cb   = self._export_exp

    def show_results(self, results: dict, config: dict = None):
        self._results   = results
        self._config    = config or {}
        self._hdf5_path = (
            self._config.get("hdf5_path")
            if self._config.get("save_hdf5")
            else None
        )
        self._placeholder.setVisible(False)
        self._action_bar.setVisible(True)
        self._label_editor.setVisible(False)
        self._tabs.setVisible(True)

        if config:
            results["_zpl_meV"] = config.get("zpl") or None

        self._tab_overview.populate(results, config)
        self._tab_spectral.populate(results)
        self._tab_emission.populate(results)
        self._tab_absorb.populate(results)
        self._tab_modes.populate(results)
        self._tab_advanced.populate(results)

        self._tabs.setCurrentIndex(0)

    # ── Comparison helpers ────────────────────────────────────────────────────
    def set_comparison_count(self, n: int):
        """Called by MainWindow whenever the comparison store changes."""
        if n == 0:
            self._comp_count_lbl.setText("Not added to comparison")
            self._add_comp_btn.setEnabled(True)
            self._label_edit.setText("Calculation 1")
        elif n >= 3:
            self._comp_count_lbl.setText(f"● {n} / 3 saved  —  comparison full")
            self._add_comp_btn.setEnabled(False)
            self._label_editor.setVisible(False)
        else:
            self._comp_count_lbl.setText(f"● {n} / 3 saved")
            self._add_comp_btn.setEnabled(True)
            self._label_edit.setText(f"Calculation {n + 1}")

    def _toggle_label_editor(self):
        self._label_editor.setVisible(not self._label_editor.isVisible())

    def _on_save_confirm(self):
        if self._results is None:
            return
        label = self._label_edit.text().strip() or "Calculation"
        self.save_requested.emit(label, self._results, self._config)
        self._label_editor.setVisible(False)

    def _export_exp(self, key: str, E_meV, I):
        """Store scaled experimental data in results and append to HDF5 if configured."""
        if self._results is None:
            return
        self._results[key] = {"E": E_meV, "I": I}
        if self._hdf5_path:
            try:
                import h5py
                with h5py.File(self._hdf5_path, "a") as hf:
                    if key in hf:
                        del hf[key]
                    g = hf.create_group(key)
                    g.create_dataset("E", data=np.asarray(E_meV))
                    g.create_dataset("I", data=np.asarray(I))
            except Exception as exc:
                dlg = QMessageBox(self)
                dlg.setWindowTitle("HDF5 write warning")
                dlg.setIcon(QMessageBox.Icon.Warning)
                dlg.setText(f"Scaled data saved to results dict but could not update HDF5:\n{exc}")
                dlg.exec()
