from __future__ import annotations

import os

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QFileDialog, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from plumipy.photoluminescence import Photoluminescence


class InverseHessianPage(QWidget):
    """Standalone tool, two sub-tabs: build H^-1 from a phonon calculation,
    and apply a Newton step R = R + H^-1 F to a structure."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        lay.addWidget(tabs)

        tabs.addTab(_BuildHessianTab(), "Build Inverse Hessian")
        tabs.addTab(_ApplyNewtonStepTab(), "Apply Newton Step")


class _BuildHessianTab(QWidget):
    """
    Build H^-1 from a phonon calculation, with full manual control over
    which modes are excluded before inversion (no auto-threshold -- the
    user sees every mode's signed energy and picks by number and/or energy
    range; the union of both is removed).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pl = Photoluminescence()
        self._masses = None
        self._freqs = None
        self._modes = None
        self._energies = None
        self._H_inv = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(12)

        header = QLabel("Inverse Hessian Builder")
        header.setObjectName("section_title")
        header.setFont(QFont("Helvetica Neue", 16, QFont.Weight.Bold))
        root.addWidget(header)

        hint = QLabel(
            "Builds H⁻¹ from a Gamma-point phonon calculation such that "
            "H⁻¹·F (F in eV/Å) gives a displacement in Å directly. "
            "Modes are listed with SIGNED energy — negative means imaginary/unstable "
            "(VASP's 'f/i' tag, or a native negative frequency in band.yaml). Choose which "
            "modes to exclude by number and/or energy range below; the union of both is "
            "removed before inversion."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hint_label")
        hint.setStyleSheet("color: #a6adc8; font-size: 13px;")
        root.addWidget(hint)

        # ── File picker ──────────────────────────────────────────────────
        file_frame = QFrame()
        file_frame.setObjectName("info_card")
        file_lay = QHBoxLayout(file_frame)
        file_lay.setContentsMargins(10, 8, 10, 8)
        file_lay.addWidget(QLabel("Phonon file (OUTCAR or band.yaml):"))
        self._path_edit = QLineEdit()
        self._path_edit.setReadOnly(True)
        self._path_edit.setPlaceholderText("Not selected")
        file_lay.addWidget(self._path_edit, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        file_lay.addWidget(browse_btn)
        self._analyze_btn = QPushButton("Analyze")
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.clicked.connect(self._analyze)
        file_lay.addWidget(self._analyze_btn)
        root.addWidget(file_frame)

        self._summary_lbl = QLabel("")
        self._summary_lbl.setObjectName("hint_label")
        self._summary_lbl.setStyleSheet("color: gray; font-size: 12px;")
        root.addWidget(self._summary_lbl)

        # ── Mode table ───────────────────────────────────────────────────
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Mode #", "Energy (meV)", "Freq (THz)", "Status"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setMinimumHeight(320)
        root.addWidget(self._table, 1)

        # ── Exclusion controls ───────────────────────────────────────────
        excl_frame = QFrame()
        excl_frame.setObjectName("info_card")
        excl_lay = QVBoxLayout(excl_frame)
        excl_lay.setContentsMargins(10, 8, 10, 8)
        excl_lay.setSpacing(8)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Remove by mode number:"))
        self._mode_range_edit = QLineEdit()
        self._mode_range_edit.setPlaceholderText(
            "e.g. 1-5, 9-11  (1-based, blank = none)")
        row1.addWidget(self._mode_range_edit)
        excl_lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Remove by energy (meV):"))
        self._energy_range_edit = QLineEdit()
        self._energy_range_edit.setPlaceholderText(
            "e.g. 0-24, 100-110  (blank = none; negative bounds allowed)")
        row2.addWidget(self._energy_range_edit)
        excl_lay.addLayout(row2)

        self._mode_range_edit.textChanged.connect(self._update_preview)
        self._energy_range_edit.textChanged.connect(self._update_preview)

        self._preview_lbl = QLabel("")
        self._preview_lbl.setObjectName("hint_label")
        self._preview_lbl.setStyleSheet("color: #cba6f7; font-size: 12px;")
        excl_lay.addWidget(self._preview_lbl)

        root.addWidget(excl_frame)

        # ── Build / Save ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._build_btn = QPushButton("▶  Construct Inverse Hessian")
        self._build_btn.setObjectName("primary_btn")
        self._build_btn.setEnabled(False)
        self._build_btn.clicked.connect(self._build)
        btn_row.addWidget(self._build_btn)

        self._save_btn = QPushButton("Save as .npy")
        self._save_btn.setObjectName("secondary_btn")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save)
        btn_row.addWidget(self._save_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self._result_lbl = QLabel("")
        self._result_lbl.setObjectName("hint_label")
        self._result_lbl.setWordWrap(True)
        root.addWidget(self._result_lbl)

    # ── File handling ────────────────────────────────────────────────────

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Phonon file", "",
            "Phonon files (*.yaml OUTCAR* outcar* *.txt);;All files (*)"
        )
        if path:
            self._path_edit.setText(path)
            self._analyze_btn.setEnabled(True)

    def _analyze(self):
        path = self._path_edit.text()
        if not path:
            return
        try:
            self._masses, self._freqs, self._modes, self._energies = \
                self._pl.ReadPhononsFlagged(path)
        except Exception as e:
            self._summary_lbl.setText(f"Error reading file: {e}")
            return

        n_imag = int((self._freqs < 0).sum())
        n_total = len(self._freqs)
        self._summary_lbl.setText(
            f"{len(self._masses)} atoms, {n_total} modes total  —  "
            f"{n_imag} flagged imaginary/unstable, {n_total - n_imag} stable"
        )

        self._table.setRowCount(n_total)
        red = QColor("#f38ba8")
        for i in range(n_total):
            is_imag = self._freqs[i] < 0
            status = "Imaginary / unstable" if is_imag else "stable"
            vals = [str(i + 1), f"{self._energies[i]:.5f}",
                    f"{self._freqs[i]:.5f}", status]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if is_imag:
                    item.setForeground(red)
                self._table.setItem(i, col, item)

        self._H_inv = None
        self._build_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._result_lbl.setText("")
        self._update_preview()

    # ── Exclusion preview ────────────────────────────────────────────────

    def _update_preview(self):
        if self._energies is None:
            return
        try:
            mask = self._pl.build_exclusion_mask(
                self._energies,
                self._mode_range_edit.text(),
                self._energy_range_edit.text(),
            )
        except Exception as e:
            self._preview_lbl.setText(f"Error parsing exclusion: {e}")
            return
        n_excl = int(mask.sum())
        n_kept = len(mask) - n_excl
        self._preview_lbl.setText(
            f"{n_excl} of {len(mask)} modes will be removed (union)  —  "
            f"{n_kept} kept"
        )

    # ── Build / Save ─────────────────────────────────────────────────────

    def _build(self):
        if self._energies is None:
            return
        try:
            mask = self._pl.build_exclusion_mask(
                self._energies,
                self._mode_range_edit.text(),
                self._energy_range_edit.text(),
            )
        except Exception as e:
            self._result_lbl.setText(f"Error: {e}")
            return

        keep = ~mask
        if keep.sum() == 0:
            self._result_lbl.setText("Error: all modes excluded — nothing to invert.")
            return

        try:
            self._H_inv = self._pl.InverseHessian(
                self._masses, self._energies[keep], self._modes[keep]
            )
        except Exception as e:
            self._result_lbl.setText(f"Error: {e}")
            self._H_inv = None
            self._save_btn.setEnabled(False)
            return

        N3 = self._H_inv.shape[0]
        self._result_lbl.setText(
            f"H⁻¹ built: shape ({N3}, {N3}) from {int(keep.sum())} kept modes  —  "
            f"units Å²/eV, so H⁻¹·F(eV/Å) → Å"
        )
        self._save_btn.setEnabled(True)

    def _save(self):
        if self._H_inv is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save inverse Hessian", "inverse_hessian.npy",
            "NumPy array (*.npy)"
        )
        if not path:
            return
        np.save(path, self._H_inv)
        self._result_lbl.setText(
            self._result_lbl.text().split("  —  Saved")[0] +
            f"  —  Saved to {path}"
        )


class _ApplyNewtonStepTab(QWidget):
    """
    R_new = R + H^-1 F, saved back out in the SAME format as the input
    structure file. Forces from an OUTCAR are read from its LAST TOTAL-FORCE
    block (ReadForces already scans to the last, not the first, occurrence).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pl = Photoluminescence()
        self._positions = None
        self._format_info = None
        self._forces = None
        self._H_inv = None
        self._R_new = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(12)

        header = QLabel("Apply Newton Step")
        header.setObjectName("section_title")
        header.setFont(QFont("Helvetica Neue", 16, QFont.Weight.Bold))
        root.addWidget(header)

        hint = QLabel(
            "Computes R = R + H⁻¹F and saves the result in the SAME format as the "
            "input structure file (e.g. a POSCAR in stays a POSCAR out, with the "
            "same lattice, species, and atom ordering). Forces read from an OUTCAR "
            "use its LAST ionic step, not the first."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hint_label")
        hint.setStyleSheet("color: #a6adc8; font-size: 13px;")
        root.addWidget(hint)

        picker_frame = QFrame()
        picker_frame.setObjectName("info_card")
        picker_lay = QVBoxLayout(picker_frame)
        picker_lay.setContentsMargins(10, 8, 10, 8)
        picker_lay.setSpacing(8)

        self._struct_edit = self._file_row(
            picker_lay, "Structure file:", self._browse_struct,
            "POSCAR / CONTCAR (VASP), .xyz (standard or extended), "
            ".npy / .npz / .dat / .txt array (N_atoms, 3) in Å"
        )
        self._force_edit = self._file_row(
            picker_lay, "Force file:", self._browse_force,
            "OUTCAR (last ionic step used) · .npy / .npz array (N_atoms, 3) [eV/Å] · "
            ".dat / .txt whitespace-delimited"
        )
        self._hinv_edit = self._file_row(
            picker_lay, "Inverse Hessian (.npy):", self._browse_hinv,
            "As built in the 'Build Inverse Hessian' tab -- shape (3N,3N), units Å²/eV"
        )
        root.addWidget(picker_frame)

        self._info_lbl = QLabel("")
        self._info_lbl.setObjectName("hint_label")
        self._info_lbl.setWordWrap(True)
        self._info_lbl.setStyleSheet("color: gray; font-size: 12px;")
        root.addWidget(self._info_lbl)

        btn_row = QHBoxLayout()
        self._apply_btn = QPushButton("▶  Apply Newton Step")
        self._apply_btn.setObjectName("primary_btn")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(self._apply_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("secondary_btn")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save)
        btn_row.addWidget(self._save_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self._result_lbl = QLabel("")
        self._result_lbl.setObjectName("hint_label")
        self._result_lbl.setWordWrap(True)
        root.addWidget(self._result_lbl)

        root.addStretch()

    # ── UI helper ────────────────────────────────────────────────────────

    def _file_row(self, parent_lay, label, callback, hint):
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(140)
        row.addWidget(lbl)
        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setPlaceholderText(hint)
        row.addWidget(edit, 1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: callback(edit))
        row.addWidget(btn)
        parent_lay.addLayout(row)
        return edit

    def _check_ready(self):
        ready = bool(
            self._struct_edit.text() and self._force_edit.text() and self._hinv_edit.text()
        )
        self._apply_btn.setEnabled(ready)

    # ── Browse callbacks ─────────────────────────────────────────────────

    def _browse_struct(self, edit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Structure file", "",
            "All files (*);;Structure files (*.npy *.npz *.txt *.dat *.vasp *.xyz)"
        )
        if path:
            edit.setText(path)
            self._check_ready()

    def _browse_force(self, edit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Force file", "",
            "All files (*);;Force files (*.npy *.npz *.txt *.dat)"
        )
        if path:
            edit.setText(path)
            self._check_ready()

    def _browse_hinv(self, edit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Inverse Hessian", "", "NumPy array (*.npy);;All files (*)"
        )
        if path:
            edit.setText(path)
            self._check_ready()

    # ── Apply / Save ─────────────────────────────────────────────────────

    def _apply(self):
        self._R_new = None
        self._save_btn.setEnabled(False)
        try:
            self._positions, self._format_info = self._pl.ReadStructureFile(
                self._struct_edit.text())
            self._forces = self._pl.ReadForceFile(self._force_edit.text())
            self._H_inv = np.load(self._hinv_edit.text())
        except Exception as e:
            self._result_lbl.setText(f"Error reading inputs: {e}")
            return

        self._info_lbl.setText(
            f"structure: {self._positions.shape[0]} atoms  ({self._format_info['kind']})   "
            f"forces: {self._forces.shape}   H⁻¹: {self._H_inv.shape}"
        )

        try:
            self._R_new = self._pl.ApplyNewtonStep(
                self._positions, self._forces, self._H_inv)
        except Exception as e:
            self._result_lbl.setText(f"Error: {e}")
            return

        disp = self._R_new - self._positions
        max_d = float(np.abs(disp).max())
        mean_d = float(np.abs(disp).mean())
        self._result_lbl.setText(
            f"Newton step applied  —  max|Δ| = {max_d:.5f} Å,  mean|Δ| = {mean_d:.5f} Å"
        )
        self._save_btn.setEnabled(True)

    def _save(self):
        if self._R_new is None:
            return
        kind = self._format_info["kind"]
        src = self._struct_edit.text()
        base, ext = os.path.splitext(src)
        default_name = os.path.basename(base) + "_newton" + (ext or "")
        filt = {
            "poscar": "All files (*)",
            "xyz": "XYZ files (*.xyz)",
            "array": f"Array files (*{self._format_info.get('ext','.npy')})",
        }.get(kind, "All files (*)")

        path, _ = QFileDialog.getSaveFileName(self, "Save structure", default_name, filt)
        if not path:
            return
        try:
            self._pl.WriteStructureFile(path, self._R_new, self._format_info)
        except Exception as e:
            self._result_lbl.setText(f"Error saving: {e}")
            return
        self._result_lbl.setText(
            self._result_lbl.text().split("  —  Saved")[0] + f"  —  Saved to {path}"
        )
