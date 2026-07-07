import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QLineEdit, QCheckBox, QComboBox, QScrollArea,
    QSizePolicy, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from app.widgets.plot_canvas import PlotCanvas, DARK, apply_dark_style
from app.pages.results import convert_energy, _to_meV

# One color per comparison slot — distinct, from the app palette
SLOT_COLORS = [DARK["purple"], DARK["blue"], DARK["teal"]]
# Experimental data is always this color regardless of which calc it came from
EXP_COLOR = DARK["peach"]   # #fab387 — warm orange, distinct from all slot colors


class ComparePage(QWidget):
    """Side-by-side spectral comparison for up to 3 saved calculations."""

    remove_requested = pyqtSignal(int)   # index of entry to remove

    def __init__(self, parent=None):
        super().__init__(parent)
        self._store = []   # list of {"label": str, "results": dict, "config": dict}

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_left_panel())

        div = QFrame()
        div.setFrameShape(QFrame.Shape.VLine)
        div.setStyleSheet("background: #313244; max-width: 1px;")
        outer.addWidget(div)

        outer.addWidget(self._build_right_panel(), 1)

    # ── Left panel ────────────────────────────────────────────────────────────
    def _build_left_panel(self):
        panel = QWidget()
        panel.setFixedWidth(240)
        panel.setStyleSheet("background: #181825;")

        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 18, 14, 18)
        lay.setSpacing(0)

        lbl = QLabel("Saved Calculations")
        lbl.setStyleSheet(
            "color: #cba6f7; font-size: 13px; font-weight: bold; padding-bottom: 10px;"
        )
        lay.addWidget(lbl)

        # Scrollable list of saved-calc entries
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")

        self._entries_container = QWidget()
        self._entries_container.setStyleSheet("background: transparent;")
        self._entries_lay = QVBoxLayout(self._entries_container)
        self._entries_lay.setContentsMargins(0, 0, 0, 0)
        self._entries_lay.setSpacing(8)
        self._entries_lay.addStretch()

        scroll.setWidget(self._entries_container)
        lay.addWidget(scroll, 1)

        lay.addSpacing(10)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: #313244; max-height: 1px; margin: 4px 0;")
        lay.addWidget(sep)
        lay.addSpacing(10)

        # Series checkboxes
        series_lbl = QLabel("Show series:")
        series_lbl.setStyleSheet("color: #6c7086; font-size: 12px; padding-bottom: 4px;")
        lay.addWidget(series_lbl)

        self._cb_standard = QCheckBox("Standard HR  (solid)")
        self._cb_mc       = QCheckBox("Monte Carlo  (dashed)")
        self._cb_squeezed = QCheckBox("Squeezed  (dotted)")
        self._cb_exp      = QCheckBox("Experimental  (markers)")

        for cb in [self._cb_standard, self._cb_mc, self._cb_squeezed, self._cb_exp]:
            cb.setChecked(True)
            cb.toggled.connect(self._replot)
            cb.setStyleSheet("font-size: 12px; color: #cdd6f4; padding: 2px 0;")
            lay.addWidget(cb)

        # ── Experimental data controls (hidden until exp data is present) ─────
        lay.addSpacing(8)
        self._exp_section = QFrame()
        self._exp_section.setStyleSheet(
            f"QFrame {{ border-left: 3px solid {EXP_COLOR}; "
            "background: #1e1e2e; border-radius: 4px; padding: 2px; }}"
        )
        self._exp_section.setVisible(False)
        exp_lay = QVBoxLayout(self._exp_section)
        exp_lay.setContentsMargins(8, 6, 6, 6)
        exp_lay.setSpacing(5)

        exp_hdr = QLabel("Experimental alignment:")
        exp_hdr.setStyleSheet(f"color: {EXP_COLOR}; font-size: 11px; font-weight: bold;")
        exp_lay.addWidget(exp_hdr)

        def _spin_row(label, widget):
            r = QHBoxLayout()
            r.setSpacing(6)
            l = QLabel(label)
            l.setStyleSheet("color: #6c7086; font-size: 11px;")
            l.setFixedWidth(90)
            r.addWidget(l)
            r.addWidget(widget)
            r.addStretch()
            return r

        self._exp_unit_combo = QComboBox()
        self._exp_unit_combo.addItems(["meV", "eV", "cm⁻¹", "nm"])
        self._exp_unit_combo.setCurrentText("eV")
        self._exp_unit_combo.setFixedWidth(72)
        self._exp_unit_combo.currentTextChanged.connect(self._replot)
        exp_lay.addLayout(_spin_row("Data unit:", self._exp_unit_combo))

        self._exp_yscale = QDoubleSpinBox()
        self._exp_yscale.setRange(1e-9, 1e9)
        self._exp_yscale.setDecimals(4)
        self._exp_yscale.setValue(1.0)
        self._exp_yscale.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        self._exp_yscale.setFixedWidth(90)
        self._exp_yscale.valueChanged.connect(self._replot)
        exp_lay.addLayout(_spin_row("Y-scale:", self._exp_yscale))

        self._exp_xshift = QDoubleSpinBox()
        self._exp_xshift.setRange(-1e5, 1e5)
        self._exp_xshift.setDecimals(1)
        self._exp_xshift.setValue(0.0)
        self._exp_xshift.setSingleStep(10.0)
        self._exp_xshift.setFixedWidth(90)
        self._exp_xshift.valueChanged.connect(self._replot)
        exp_lay.addLayout(_spin_row("X-shift (meV):", self._exp_xshift))

        lay.addWidget(self._exp_section)
        # ─────────────────────────────────────────────────────────────────────

        lay.addSpacing(12)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("background: #313244; max-height: 1px; margin: 4px 0;")
        lay.addWidget(sep2)
        lay.addSpacing(10)

        # Energy unit for calculated spectra
        unit_row = QHBoxLayout()
        unit_row.setSpacing(8)
        unit_lbl = QLabel("Energy unit:")
        unit_lbl.setStyleSheet("color: #6c7086; font-size: 12px;")
        unit_row.addWidget(unit_lbl)
        self._unit_combo = QComboBox()
        self._unit_combo.addItems(["meV", "eV", "cm⁻¹", "nm"])
        self._unit_combo.setFixedWidth(80)
        self._unit_combo.currentTextChanged.connect(self._replot)
        unit_row.addWidget(self._unit_combo)
        unit_row.addStretch()
        lay.addLayout(unit_row)

        lay.addSpacing(14)

        clear_btn = QPushButton("Clear All")
        clear_btn.setObjectName("secondary_btn")
        clear_btn.clicked.connect(self._on_clear_all)
        lay.addWidget(clear_btn)

        return panel

    # ── Right panel ───────────────────────────────────────────────────────────
    def _build_right_panel(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(12)

        title = QLabel("Spectral Comparison")
        title.setObjectName("section_title")
        title.setFont(QFont("Helvetica Neue", 17, QFont.Weight.Bold))
        lay.addWidget(title)

        hint = QLabel(
            "Spectra are normalised to their individual peak for visual comparison.\n"
            "Solid = Standard HR · Dashed = Monte Carlo · Dotted = Squeezed · "
            "Markers = Experimental"
        )
        hint.setObjectName("hint_label")
        hint.setStyleSheet("color: #6c7086; font-size: 12px; padding-bottom: 2px;")
        lay.addWidget(hint)

        # Placeholder shown until ≥ 2 calcs are saved
        self._placeholder = QLabel(
            "Save at least 2 calculations to compare their spectra.\n\n"
            "After running a calculation, click  + Add to Comparison  in the Results page."
        )
        self._placeholder.setObjectName("hint_label")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            "font-size: 14px; padding: 60px; color: #6c7086;"
        )
        lay.addWidget(self._placeholder, 1)

        # Two-subplot canvas (emission left, absorption right)
        self._canvas = PlotCanvas(nrows=1, ncols=2, figsize=(12, 4.5))
        self._canvas.setVisible(False)
        lay.addWidget(self._canvas, 1)

        return panel

    # ── Public API ────────────────────────────────────────────────────────────
    def update_store(self, store: list):
        self._store = store
        self._rebuild_entries()
        has_enough = len(store) >= 2
        self._placeholder.setVisible(not has_enough)
        self._canvas.setVisible(has_enough)
        # Show experimental controls only when at least one calc has exp data
        has_exp = any(
            e["results"].get("exp_emission") or e["results"].get("exp_absorption")
            for e in store
        )
        self._exp_section.setVisible(has_exp)
        if len(store) >= 1:
            self._replot()

    # ── Entries list ──────────────────────────────────────────────────────────
    def _rebuild_entries(self):
        # Remove all widgets from the entries container
        while self._entries_lay.count() > 1:   # keep the trailing stretch
            item = self._entries_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, entry in enumerate(self._store):
            color = SLOT_COLORS[i % len(SLOT_COLORS)]
            row = self._make_entry_row(i, entry["label"], color)
            self._entries_lay.insertWidget(i, row)

    def _make_entry_row(self, idx: int, label: str, color: str):
        f = QFrame()
        f.setStyleSheet(
            f"QFrame {{ border-left: 3px solid {color}; background: #1e1e2e; "
            f"border-radius: 4px; margin: 0; }}"
        )
        row_lay = QHBoxLayout(f)
        row_lay.setContentsMargins(8, 5, 5, 5)
        row_lay.setSpacing(6)

        le = QLineEdit(label)
        le.setStyleSheet(
            "QLineEdit { background: transparent; border: none; color: #cdd6f4; "
            "font-size: 13px; padding: 0; }"
        )
        le.setPlaceholderText(f"Calculation {idx + 1}")
        le.editingFinished.connect(
            lambda _le=le, _i=idx: self._rename_entry(_i, _le.text())
        )
        row_lay.addWidget(le, 1)

        rm = QPushButton("✕")
        rm.setFixedSize(20, 20)
        rm.setStyleSheet(
            "QPushButton { background: transparent; color: #6c7086; border: none; "
            "font-size: 10px; }"
            "QPushButton:hover { color: #f38ba8; }"
        )
        rm.clicked.connect(lambda _, i=idx: self.remove_requested.emit(i))
        row_lay.addWidget(rm)
        return f

    def _rename_entry(self, idx: int, text: str):
        if 0 <= idx < len(self._store):
            self._store[idx]["label"] = text.strip() or f"Calculation {idx + 1}"
        self._replot()

    def _on_clear_all(self):
        # Emit remove for each slot in reverse order so indices stay valid
        for i in range(len(self._store) - 1, -1, -1):
            self.remove_requested.emit(i)

    # ── Plotting ──────────────────────────────────────────────────────────────
    def _replot(self):
        if not self._store:
            return

        unit     = self._unit_combo.currentText()
        show_std = self._cb_standard.isChecked()
        show_mc  = self._cb_mc.isChecked()
        show_sq  = self._cb_squeezed.isChecked()
        show_exp = self._cb_exp.isChecked()

        ax_em, ax_abs = self._canvas.axes
        ax_em.cla()
        ax_abs.cla()

        for i, entry in enumerate(self._store):
            results = entry["results"]
            label   = entry["label"] or f"Calculation {i + 1}"
            color   = SLOT_COLORS[i % len(SLOT_COLORS)]

            # Standard HR analytical
            std = results.get("standard_hr")
            if show_std and std:
                E_em  = convert_energy(std["E_photon_emission"],  unit)
                I_em  = np.real(std["I_emission"])
                if I_em.max() > 0:
                    ax_em.plot(E_em, I_em / I_em.max(), color=color,
                               lw=1.8, ls="-", label=f"{label} (analytical)", zorder=3)

                E_abs = convert_energy(std["E_photon_absorption"], unit)
                I_abs = np.real(std["I_absorption"])
                if I_abs.max() > 0:
                    ax_abs.plot(E_abs, I_abs / I_abs.max(), color=color,
                                lw=1.8, ls="-", label=f"{label} (analytical)", zorder=3)

            # Monte Carlo emission
            mc = results.get("monte_carlo_emission")
            if show_mc and mc:
                E_mc = convert_energy(mc["E_photon_emission"], unit)
                I_mc = np.asarray(mc["I_emission"], float)
                if I_mc.max() > 0:
                    ax_em.plot(E_mc, I_mc / I_mc.max(), color=color,
                               lw=1.4, ls="--", label=f"{label} (MC)", zorder=3)

            # Squeezed oscillator
            sq = results.get("squeezed")
            if show_sq and sq:
                E_sq_em = convert_energy(sq["E_photon_emission"],  unit)
                I_sq_em = np.real(sq["I_emission"])
                if I_sq_em.max() > 0:
                    ax_em.plot(E_sq_em, I_sq_em / I_sq_em.max(), color=color,
                               lw=1.4, ls=":", label=f"{label} (squeezed)", zorder=3)
                E_sq_abs = convert_energy(sq["E_photon_absorption"], unit)
                I_sq_abs = np.real(sq["I_absorption"])
                if I_sq_abs.max() > 0:
                    ax_abs.plot(E_sq_abs, I_sq_abs / I_sq_abs.max(), color=color,
                                lw=1.4, ls=":", label=f"{label} (squeezed)", zorder=3)

        # Experimental data — plotted once from the first calc that has it,
        # with its own distinctive color and the user's alignment controls.
        if show_exp:
            exp_data_unit = self._exp_unit_combo.currentText()
            y_scale       = self._exp_yscale.value()
            x_shift_meV   = self._exp_xshift.value()

            def _plot_exp_once(ax, key):
                for entry in self._store:
                    raw = entry["results"].get(key)
                    if not raw:
                        continue
                    E_meV = _to_meV(np.asarray(raw["E"], float), exp_data_unit) + x_shift_meV
                    E_plt = convert_energy(E_meV, unit)
                    I_plt = np.asarray(raw["I"], float) * y_scale
                    if I_plt.max() > 0:
                        ax.plot(E_plt, I_plt / I_plt.max(),
                                color=EXP_COLOR, lw=1.6, ls="-",
                                marker=".", markersize=3, markevery=5,
                                label="Experiment", zorder=5)
                    break   # only the first one found

            _plot_exp_once(ax_em,  "exp_emission")
            _plot_exp_once(ax_abs, "exp_absorption")

        unit_lbl = unit
        for ax, title in [(ax_em, "Emission"), (ax_abs, "Absorption")]:
            ax.set_title(title, fontsize=12, pad=6)
            ax.set_xlabel(unit_lbl, fontsize=10)
            ax.set_ylabel("Normalised Intensity", fontsize=10)
            if ax.has_data():
                leg = ax.legend(
                    fontsize=9, loc="best", framealpha=0.85,
                    facecolor=DARK["axes_bg"], edgecolor=DARK["spine"],
                )
                for t in leg.get_texts():
                    t.set_color(DARK["text"])
            else:
                ax.text(0.5, 0.5, "No data available",
                        transform=ax.transAxes, ha="center", va="center",
                        color=DARK["spine"], fontsize=11)

        self._canvas.fig.tight_layout(pad=2.0)
        self._canvas.draw()
