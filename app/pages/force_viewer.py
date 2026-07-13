from __future__ import annotations

import json
import os
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QDoubleSpinBox, QComboBox,
    QFileDialog, QSizePolicy,
)
from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

from plumipy.photoluminescence import Photoluminescence

_HTML = os.path.join(os.path.dirname(__file__), "..", "html", "phonon_viewer.html")
_HTML = os.path.abspath(_HTML)

_VCOL = {
    "H":(240,240,240),"C":(102,102,102),"N":(84,108,220),
    "O":(224, 68, 68),"B":(224,148, 72),"Si":(170,170,220),
    "S":(220,200, 48),"F":( 68,210, 68),"Cl":( 48,200, 48),
    "P":(224,128,  0),"Fe":(200, 96, 48),"Cu":(180,120, 48),
}
_VCOL_DEF = (160,160,160)
_VRAD = {
    "H":0.31,"C":0.77,"N":0.74,"O":0.73,"B":0.84,
    "Si":1.11,"S":1.04,"F":0.72,"Cl":0.99,"P":1.06,
}
_VRAD_DEF = 0.80
_COV = {
    "H":0.31,"C":0.76,"N":0.71,"O":0.66,"B":0.84,
    "Si":1.11,"S":1.05,"F":0.57,"Cl":1.02,"P":1.07,
}
_COV_DEF = 0.80

_FORCE_LABELS = ["GS Forces", "ES Forces", "ΔF = F_ES − F_GS"]


class ForceViewerWidget(QWidget):
    """
    Sub-tab for the Mode Analysis panel (VG workflow).
    Visualises GS/ES/difference force vectors on an uploaded structure
    using the same WebGL renderer as PhononViewerWidget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results      = None
        self._struct_path  = None   # path to user-uploaded structure file
        self._positions    = None   # (N,3) Cartesian, loaded from structure
        self._species      = None   # list[str] per-atom
        self._lattice      = None   # (3,3) or None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── control bar ───────────────────────────────────────────────────────
        bar = QFrame()
        bar.setObjectName("info_card")
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(10, 6, 10, 6)
        bar_lay.setSpacing(12)

        self._struct_btn = QPushButton("Browse structure…")
        self._struct_btn.setToolTip(
            "Upload a POSCAR / CONTCAR / .vasp / .xyz file with atom positions"
        )
        self._struct_btn.clicked.connect(self._browse_structure)
        bar_lay.addWidget(self._struct_btn)

        self._struct_label = QLabel("No file loaded")
        self._struct_label.setStyleSheet("color: gray;")
        bar_lay.addWidget(self._struct_label)

        bar_lay.addSpacing(8)
        bar_lay.addWidget(QLabel("Force:"))
        self._force_combo = QComboBox()
        self._force_combo.addItems(_FORCE_LABELS)
        self._force_combo.setFixedWidth(170)
        self._force_combo.currentIndexChanged.connect(self._on_force_changed)
        bar_lay.addWidget(self._force_combo)

        bar_lay.addWidget(QLabel("Scale:"))
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.1, 200.0)
        self._scale_spin.setSingleStep(0.5)
        self._scale_spin.setValue(1.0)
        self._scale_spin.setFixedWidth(72)
        self._scale_spin.setToolTip("Arrow length multiplier")
        bar_lay.addWidget(self._scale_spin)

        bar_lay.addWidget(QLabel("Threshold:"))
        self._thresh_spin = QDoubleSpinBox()
        self._thresh_spin.setRange(0.0, 10.0)
        self._thresh_spin.setSingleStep(0.01)
        self._thresh_spin.setDecimals(4)
        self._thresh_spin.setValue(0.01)
        self._thresh_spin.setFixedWidth(80)
        self._thresh_spin.setToolTip(
            "Minimum |force| in eV/Å to draw an arrow"
        )
        bar_lay.addWidget(self._thresh_spin)

        bar_lay.addStretch()

        self._render_btn = QPushButton("▶  Render")
        self._render_btn.setEnabled(False)
        self._render_btn.clicked.connect(self._render)
        bar_lay.addWidget(self._render_btn)

        self._vesta_btn = QPushButton("Save .vesta")
        self._vesta_btn.setEnabled(False)
        self._vesta_btn.setToolTip("Export force vectors for VESTA")
        self._vesta_btn.clicked.connect(self._save_vesta)
        bar_lay.addWidget(self._vesta_btn)

        root.addWidget(bar)

        # ── placeholder label shown before a structure is loaded ──────────────
        self._hint = QLabel(
            "Load a POSCAR / CONTCAR / .xyz structure file to visualise force vectors."
        )
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setStyleSheet("color: gray; font-style: italic; padding: 24px;")
        root.addWidget(self._hint)

        # ── WebGL view ────────────────────────────────────────────────────────
        self._view = QWebEngineView()
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._view.setVisible(False)

        settings = self._view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)

        self._page_ready = False
        self._pending_js = None
        self._view.loadFinished.connect(self._on_load_finished)
        self._view.load(QUrl.fromLocalFile(_HTML))
        root.addWidget(self._view, 1)

    # ── load callback ─────────────────────────────────────────────────────────
    def _on_load_finished(self, ok: bool):
        self._page_ready = True
        if self._pending_js:
            self._view.page().runJavaScript(self._pending_js)
            self._pending_js = None

    def _run_js(self, js: str):
        if self._page_ready:
            self._view.page().runJavaScript(js)
        else:
            self._pending_js = js

    # ── public API ────────────────────────────────────────────────────────────
    def set_results(self, results: dict):
        self._results = results
        has_gs = "F_gs" in results
        has_es = "F_es" in results

        # Enable/disable combo items based on available forces
        for i, label in enumerate(_FORCE_LABELS):
            item_enabled = (
                (i == 0 and has_gs) or
                (i == 1 and has_es) or
                (i == 2 and has_gs and has_es)
            )
            # Grey out unavailable items via flags
            flags = self._force_combo.model().item(i).flags()
            from PyQt6.QtCore import Qt as _Qt
            if item_enabled:
                self._force_combo.model().item(i).setFlags(
                    flags | _Qt.ItemFlag.ItemIsEnabled
                )
            else:
                self._force_combo.model().item(i).setFlags(
                    flags & ~_Qt.ItemFlag.ItemIsEnabled
                )

        # Select first available
        if has_gs:
            self._force_combo.setCurrentIndex(0)
        elif has_es:
            self._force_combo.setCurrentIndex(1)

        self._update_render_btn()
        self._suggest_scale()

    def clear(self):
        self._results     = None
        self._positions   = None
        self._species     = None
        self._lattice     = None
        self._struct_path = None
        self._struct_label.setText("No file loaded")
        self._struct_label.setStyleSheet("color: gray;")
        self._render_btn.setEnabled(False)
        self._vesta_btn.setEnabled(False)
        self._hint.setVisible(True)
        self._view.setVisible(False)
        self._run_js("if(window.clearScene) clearScene();")

    # ── structure file browser ────────────────────────────────────────────────
    def _browse_structure(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open structure file",
            "",
            "Structure files (*.vasp *.xyz POSCAR CONTCAR CONTCAR_*);;All Files (*)",
        )
        if not path:
            return
        try:
            pl = Photoluminescence()
            pos, atoms, lat = pl.ReadStructure(path)
            self._positions   = pos
            self._species     = self._expand_atoms(atoms, len(pos))
            self._lattice     = lat
            self._struct_path = path
            name = os.path.basename(path)
            self._struct_label.setText(name)
            self._struct_label.setStyleSheet("")
        except Exception as exc:
            self._struct_label.setText(f"Error: {exc}")
            self._struct_label.setStyleSheet("color: red;")
            return
        self._update_render_btn()
        self._suggest_scale()

    @staticmethod
    def _expand_atoms(atoms, n: int) -> list[str]:
        if isinstance(atoms, list):
            return [str(el) for el in atoms]
        if isinstance(atoms, dict):
            lst = []
            for el, count in atoms.items():
                lst.extend([str(el)] * int(count))
            return lst
        return ["X"] * n

    # ── force selection ───────────────────────────────────────────────────────
    def _get_force_array(self) -> np.ndarray | None:
        if self._results is None:
            return None
        idx = self._force_combo.currentIndex()
        if idx == 0:
            return self._results.get("F_gs")
        if idx == 1:
            return self._results.get("F_es")
        # ΔF
        F_gs = self._results.get("F_gs")
        F_es = self._results.get("F_es")
        if F_gs is not None and F_es is not None:
            return F_es - F_gs
        return None

    def _on_force_changed(self):
        self._suggest_scale()

    def _suggest_scale(self):
        forces = self._get_force_array()
        if forces is None:
            return
        max_norm = float(np.max(np.linalg.norm(forces, axis=1)))
        if max_norm > 1e-6:
            suggested = round(2.0 / max_norm, 2)
            self._scale_spin.setValue(float(np.clip(suggested, 0.1, 200.0)))

    def _update_render_btn(self):
        has_forces   = self._results is not None and self._get_force_array() is not None
        has_struct   = self._positions is not None
        self._render_btn.setEnabled(has_forces and has_struct)
        self._vesta_btn.setEnabled(has_forces and has_struct and self._lattice is not None)

    # ── render ────────────────────────────────────────────────────────────────
    def _render(self):
        forces = self._get_force_array()
        if forces is None or self._positions is None:
            return

        scale  = self._scale_spin.value()
        thresh = self._thresh_spin.value()

        vectors = []
        for v in forces:
            norm = float(np.linalg.norm(v))
            vectors.append((v * scale).tolist() if norm >= thresh else [0.0, 0.0, 0.0])

        payload = {
            "positions": self._positions.tolist(),
            "species":   self._species,
            "vectors":   vectors,
            "cell":      self._lattice.tolist() if self._lattice is not None else None,
            "mode_info": {
                "index":      self._force_combo.currentIndex(),
                "energy_meV": 0.0,
                "Sk":         0.0,
            },
        }

        self._hint.setVisible(False)
        self._view.setVisible(True)
        self._run_js(f"updateScene({json.dumps(payload)});")

    # ── VESTA export ──────────────────────────────────────────────────────────
    def _save_vesta(self):
        forces = self._get_force_array()
        if forces is None or self._positions is None or self._lattice is None:
            return

        label = _FORCE_LABELS[self._force_combo.currentIndex()].replace(" ", "_").replace("=","").replace("−","-")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save VESTA file",
            f"forces_{label}.vesta",
            "VESTA files (*.vesta)"
        )
        if not path:
            return

        content = self._build_vesta(forces, self._scale_spin.value(), self._thresh_spin.value(), label)
        with open(path, "w") as f:
            f.write(content)

    def _build_vesta(self, forces: np.ndarray, scale: float, thresh: float, label: str) -> str:
        lat     = self._lattice
        R_cart  = self._positions
        species = self._species

        a_vec, b_vec, c_vec = lat[0], lat[1], lat[2]
        a = np.linalg.norm(a_vec)
        b = np.linalg.norm(b_vec)
        c = np.linalg.norm(c_vec)
        alpha = np.degrees(np.arccos(np.clip(np.dot(b_vec,c_vec)/(b*c), -1,1)))
        beta  = np.degrees(np.arccos(np.clip(np.dot(a_vec,c_vec)/(a*c), -1,1)))
        gamma = np.degrees(np.arccos(np.clip(np.dot(a_vec,b_vec)/(a*b), -1,1)))

        frac = np.linalg.solve(lat.T, R_cart.T).T
        unique_els = list(dict.fromkeys(species))

        lines = [
            "#VESTA_FORMAT_VERSION 3.5.0", "",
            "CRYSTAL", "",
            "TITLE",
            f" Forces  —  {label}  —  PLUMIPY", "",
            "GROUP",
            " 1 1 P 1", "",
            "CELLP",
            f"  {a:.6f} {b:.6f} {c:.6f}  {alpha:.4f} {beta:.4f} {gamma:.4f}",
            "  0.000000 0.000000 0.000000 0.000000 0.000000 0.000000", "",
            "STRUC",
        ]

        for i, (el, fr) in enumerate(zip(species, frac)):
            label_at = f"{el}{i+1}"
            lines.append(
                f"  {i+1:3d}  {el:2s}  {label_at:8s}  1.000000"
                f"  {fr[0]:.6f}  {fr[1]:.6f}  {fr[2]:.6f}    1a    1"
            )
            lines.append("                            0   0   0   0   0")
        lines += ["  0 0 0 0 0 0", ""]

        lines += [
            "BOUND",
            "       0        1         0        1         0        1",
            "  0   0   0   0  0",
        ]

        lines.append("SBOND")
        bond_id = 1
        seen_pairs: set = set()
        for el1 in unique_els:
            for el2 in unique_els:
                pair = tuple(sorted([el1, el2]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                max_bond = (_COV.get(el1, _COV_DEF) + _COV.get(el2, _COV_DEF)) * 1.15
                lines.append(
                    f"  {bond_id}  {el1:2s}  {el2:2s}  0.00000  {max_bond:.5f}"
                    f"  0  1  1  0  1  0.250  2.000 127 127 127"
                )
                bond_id += 1
        lines += ["  0 0 0 0", ""]

        lines.append("SITET")
        for i, el in enumerate(species):
            label_at = f"{el}{i+1}"
            rad = _VRAD.get(el, _VRAD_DEF)
            rc, gc, bc = _VCOL.get(el, _VCOL_DEF)
            lines.append(
                f"  {i+1:3d}  {label_at:8s}  {rad:.4f}"
                f"  {rc} {gc} {bc}  {rc} {gc} {bc}  204  0"
            )
        lines += ["  0 0 0 0 0 0", ""]

        vec_entries = []
        for i, v in enumerate(forces):
            if np.linalg.norm(v) >= thresh:
                vec_entries.append((i, v * scale))

        lines.append("VECTR")
        for vid, (atom_i, sv) in enumerate(vec_entries, start=1):
            lines.append(f" {vid:4d}  {sv[0]:10.5f}  {sv[1]:10.5f}  {sv[2]:10.5f}  0")
            lines.append(f" {atom_i+1:4d}  0  0  0  0")
            lines.append("  0  0  0  0  0")
        lines += [" 0 0 0 0 0", ""]

        lines.append("VECTT")
        for vid in range(1, len(vec_entries) + 1):
            lines.append(f" {vid:4d}   0.300  255   0   0   0")
        lines += [" 0 0 0 0 0", ""]

        lines += ["SPLAN", "  0   0   0   0", ""]

        lines.append("ATOMT")
        for i, el in enumerate(unique_els, start=1):
            rad = _VRAD.get(el, _VRAD_DEF)
            rc, gc, bc = _VCOL.get(el, _VCOL_DEF)
            lines.append(
                f"  {i}  {el:2s}  {rad:.4f}"
                f"  {rc} {gc} {bc}  {rc} {gc} {bc}  204"
            )
        lines += ["  0 0 0 0 0 0", ""]

        return "\n".join(lines)
