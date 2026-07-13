from __future__ import annotations

import json
import os
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QDoubleSpinBox, QSpinBox,
    QFileDialog, QSizePolicy,
)
from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

from plumipy.photoluminescence import Photoluminescence

# ── path to the bundled HTML ───────────────────────────────────────────────────
_HTML = os.path.join(os.path.dirname(__file__), "..", "html", "phonon_viewer.html")
_HTML = os.path.abspath(_HTML)

# ── CPK element colours (for VESTA VECTT — not used in Python rendering) ──────
_CPK_HEX = {
    "H": (255,255,255), "C": (90,90,90),   "N": (50,50,255),
    "O": (255,55,55),   "B": (255,140,65),  "Si":(158,158,230),
    "S": (230,205,25),  "F": (50,230,50),   "Cl":(50,217,50),
    "P": (255,140,0),
}
_DEFAULT_CPK = (180,180,180)


class PhononViewerWidget(QWidget):
    """
    Sub-tab for the Mode Analysis panel.
    Displays phonon displacement vectors on the crystal structure
    using an embedded WebGL renderer (no internet required).

    For VG-type calculations (no structure in results), an optional
    "Browse structure…" row is shown so the user can supply one.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results        = None
        self._ext_positions  = None   # (N,3) loaded from file when R_gs absent
        self._ext_species    = None   # list[str]
        self._ext_lattice    = None   # (3,3) or None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── main control bar ──────────────────────────────────────────────────
        bar = QFrame()
        bar.setObjectName("info_card")
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(10, 6, 10, 6)
        bar_lay.setSpacing(16)

        bar_lay.addWidget(QLabel("Mode:"))
        self._mode_spin = QSpinBox()
        self._mode_spin.setRange(1, 1)
        self._mode_spin.setFixedWidth(72)
        self._mode_spin.setToolTip("Normal mode index (1-based)")
        bar_lay.addWidget(self._mode_spin)

        bar_lay.addWidget(QLabel("Scale:"))
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.1, 200.0)
        self._scale_spin.setSingleStep(0.5)
        self._scale_spin.setValue(3.0)
        self._scale_spin.setFixedWidth(72)
        self._scale_spin.setToolTip("Arrow length multiplier")
        bar_lay.addWidget(self._scale_spin)

        bar_lay.addWidget(QLabel("Threshold:"))
        self._thresh_spin = QDoubleSpinBox()
        self._thresh_spin.setRange(0.0, 1.0)
        self._thresh_spin.setSingleStep(0.005)
        self._thresh_spin.setDecimals(4)
        self._thresh_spin.setValue(0.01)
        self._thresh_spin.setFixedWidth(80)
        self._thresh_spin.setToolTip("Minimum |displacement| to show arrow")
        bar_lay.addWidget(self._thresh_spin)

        bar_lay.addStretch()

        self._render_btn = QPushButton("▶  Render")
        self._render_btn.setEnabled(False)
        self._render_btn.clicked.connect(self._render)
        bar_lay.addWidget(self._render_btn)

        self._vesta_btn = QPushButton("Save .vesta")
        self._vesta_btn.setEnabled(False)
        self._vesta_btn.setToolTip("Export displacement vectors for VESTA")
        self._vesta_btn.clicked.connect(self._save_vesta)
        bar_lay.addWidget(self._vesta_btn)

        root.addWidget(bar)

        # ── optional structure row (VG workflow only, hidden by default) ───────
        self._struct_bar = QFrame()
        self._struct_bar.setObjectName("info_card")
        struct_lay = QHBoxLayout(self._struct_bar)
        struct_lay.setContentsMargins(10, 4, 10, 4)
        struct_lay.setSpacing(10)

        lbl = QLabel(
            "No structure in results (VG workflow) — upload a structure file to visualise modes:"
        )
        lbl.setStyleSheet("color: gray; font-style: italic;")
        struct_lay.addWidget(lbl)

        self._struct_btn = QPushButton("Browse structure…")
        self._struct_btn.setToolTip(
            "Load a POSCAR / CONTCAR / .vasp / .xyz file with atom positions"
        )
        self._struct_btn.clicked.connect(self._browse_structure)
        struct_lay.addWidget(self._struct_btn)

        self._struct_label = QLabel("No file loaded")
        self._struct_label.setStyleSheet("color: gray;")
        struct_lay.addWidget(self._struct_label)

        struct_lay.addStretch()
        self._struct_bar.setVisible(False)
        root.addWidget(self._struct_bar)

        # ── WebGL view ────────────────────────────────────────────────────────
        self._view = QWebEngineView()
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Enable WebGL (off by default in QWebEngineView)
        settings = self._view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)

        self._page_ready    = False
        self._pending_js    = None          # JS to run once page finishes loading
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
            self._pending_js = js   # will fire once loadFinished

    # ── public API ────────────────────────────────────────────────────────────
    def set_results(self, results: dict):
        self._results = results
        n_modes = len(results.get("modes_gs", []))
        self._mode_spin.setRange(1, max(n_modes, 1))
        # Default to mode with highest Sk
        if "Sk" in results and n_modes:
            best = int(np.argmax(results["Sk"])) + 1
            self._mode_spin.setValue(best)

        # Show structure picker when R_gs not provided (VG workflow)
        needs_struct = "R_gs" not in results
        self._struct_bar.setVisible(needs_struct)
        if not needs_struct:
            # Reset any previously loaded external structure
            self._ext_positions = None
            self._ext_species   = None
            self._ext_lattice   = None

        # Auto-scale using highest-Sk mode
        if n_modes > 0:
            best_idx = (int(np.argmax(results["Sk"])) if "Sk" in results else 0)
            max_norm = float(np.max(np.linalg.norm(results["modes_gs"][best_idx], axis=1)))
            if max_norm > 1e-6:
                suggested = round(2.0 / max_norm, 1)
                self._scale_spin.setValue(float(np.clip(suggested, 0.1, 200.0)))

        self._update_buttons()

    def clear(self):
        self._results        = None
        self._ext_positions  = None
        self._ext_species    = None
        self._ext_lattice    = None
        self._struct_bar.setVisible(False)
        self._struct_label.setText("No file loaded")
        self._struct_label.setStyleSheet("color: gray;")
        self._render_btn.setEnabled(False)
        self._vesta_btn.setEnabled(False)
        self._run_js("if(window.clearScene) clearScene();")

    # ── structure file browser (VG workflow) ──────────────────────────────────
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
            self._ext_positions = pos
            self._ext_species   = self._expand_atoms(atoms, len(pos))
            self._ext_lattice   = lat
            name = os.path.basename(path)
            self._struct_label.setText(name)
            self._struct_label.setStyleSheet("")
        except Exception as exc:
            self._struct_label.setText(f"Error: {exc}")
            self._struct_label.setStyleSheet("color: red;")
            return
        self._update_buttons()

    # ── button state ──────────────────────────────────────────────────────────
    def _update_buttons(self):
        r = self._results
        if r is None:
            self._render_btn.setEnabled(False)
            self._vesta_btn.setEnabled(False)
            return
        n_modes  = len(r.get("modes_gs", []))
        has_pos  = "R_gs" in r or self._ext_positions is not None
        _lat     = r.get("lattice")
        lat      = _lat if _lat is not None else self._ext_lattice
        has_spec = "atoms" in r or self._ext_species is not None
        self._render_btn.setEnabled(n_modes > 0 and has_pos)
        self._vesta_btn.setEnabled(n_modes > 0 and has_pos and lat is not None and has_spec)

    # ── private helpers ───────────────────────────────────────────────────────
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

    def _species_list(self) -> list[str]:
        """Per-atom element list — from results dict or externally loaded file."""
        atoms = self._results.get("atoms") if self._results else None
        if atoms is not None:
            return self._expand_atoms(atoms, len(self._results.get("R_gs", [])))
        if self._ext_species is not None:
            return self._ext_species
        n = len(self._ext_positions) if self._ext_positions is not None else 0
        return ["X"] * n

    def _get_positions(self) -> np.ndarray:
        r = self._results
        if r is not None and "R_gs" in r:
            return r["R_gs"]
        return self._ext_positions

    def _get_lattice(self):
        r = self._results
        lat = r.get("lattice") if r else None
        return lat if lat is not None else self._ext_lattice

    def _render(self):
        r = self._results
        if r is None or "modes_gs" not in r:
            return

        positions = self._get_positions()
        if positions is None:
            return

        idx    = self._mode_spin.value() - 1
        scale  = self._scale_spin.value()
        thresh = self._thresh_spin.value()

        species  = self._species_list()
        mode_vec = r["modes_gs"][idx]          # (N_atoms, 3)

        # Scale vectors; zero out those below threshold
        vectors = []
        for v in mode_vec:
            norm = float(np.linalg.norm(v))
            vectors.append((v * scale).tolist() if norm >= thresh else [0.0, 0.0, 0.0])

        lat  = self._get_lattice()
        cell = lat.tolist() if lat is not None else None

        Ek = r.get("Ek_gs", np.zeros(len(mode_vec)))
        Sk = r.get("Sk",    np.zeros(len(mode_vec)))

        payload = {
            "positions": positions.tolist(),
            "species":   species,
            "vectors":   vectors,
            "cell":      cell,
            "mode_info": {
                "index":      idx,
                "energy_meV": float(Ek[idx]),
                "Sk":         float(Sk[idx]),
            },
        }

        js = f"updateScene({json.dumps(payload)});"
        self._run_js(js)

    def _save_vesta(self):
        r = self._results
        if r is None:
            return
        if self._get_positions() is None or self._get_lattice() is None:
            return

        idx    = self._mode_spin.value() - 1
        scale  = self._scale_spin.value()
        thresh = self._thresh_spin.value()

        path, _ = QFileDialog.getSaveFileName(
            self, "Save VESTA file",
            f"mode_{idx+1:04d}.vesta",
            "VESTA files (*.vesta)"
        )
        if not path:
            return

        content = self._build_vesta(idx, scale, thresh)
        with open(path, "w") as f:
            f.write(content)

    def _build_vesta(self, mode_idx: int, scale: float, thresh: float) -> str:
        r        = self._results
        lat      = self._get_lattice()
        R_cart   = self._get_positions()
        species  = self._species_list()
        mode_vec = r["modes_gs"][mode_idx]

        # VESTA element colors (RGB 0-255) and radii
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
        # Covalent radii for SBOND cutoffs
        _COV = {
            "H":0.31,"C":0.76,"N":0.71,"O":0.66,"B":0.84,
            "Si":1.11,"S":1.05,"F":0.57,"Cl":1.02,"P":1.07,
        }
        _COV_DEF = 0.80

        # Cell parameters
        a_vec, b_vec, c_vec = lat[0], lat[1], lat[2]
        a = np.linalg.norm(a_vec)
        b = np.linalg.norm(b_vec)
        c = np.linalg.norm(c_vec)
        alpha = np.degrees(np.arccos(np.clip(np.dot(b_vec,c_vec)/(b*c), -1,1)))
        beta  = np.degrees(np.arccos(np.clip(np.dot(a_vec,c_vec)/(a*c), -1,1)))
        gamma = np.degrees(np.arccos(np.clip(np.dot(a_vec,b_vec)/(a*b), -1,1)))

        frac = np.linalg.solve(lat.T, R_cart.T).T  # (N,3) fractional coords

        # Unique elements in appearance order
        unique_els = list(dict.fromkeys(species))

        lines = [
            "#VESTA_FORMAT_VERSION 3.5.0", "",
            "CRYSTAL", "",
            "TITLE",
            f" Mode {mode_idx+1}  —  PLUMIPY phonon vectors", "",
            "GROUP",
            " 1 1 P 1", "",
            "CELLP",
            f"  {a:.6f} {b:.6f} {c:.6f}  {alpha:.4f} {beta:.4f} {gamma:.4f}",
            "  0.000000 0.000000 0.000000 0.000000 0.000000 0.000000", "",
            "STRUC",
        ]

        for i, (el, fr) in enumerate(zip(species, frac)):
            label = f"{el}{i+1}"
            lines.append(
                f"  {i+1:3d}  {el:2s}  {label:8s}  1.000000"
                f"  {fr[0]:.6f}  {fr[1]:.6f}  {fr[2]:.6f}    1a    1"
            )
            lines.append("                            0   0   0   0   0")
        lines += ["  0 0 0 0 0 0", ""]

        # BOUND — full unit cell
        lines += [
            "BOUND",
            "       0        1         0        1         0        1",
            "  0   0   0   0  0",
        ]

        # SBOND — one entry per unique element pair with covalent cutoff
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

        # SITET — per-atom color/radius (same order as STRUC)
        lines.append("SITET")
        for i, el in enumerate(species):
            label = f"{el}{i+1}"
            rad = _VRAD.get(el, _VRAD_DEF)
            rc, gc, bc = _VCOL.get(el, _VCOL_DEF)
            lines.append(
                f"  {i+1:3d}  {label:8s}  {rad:.4f}"
                f"  {rc} {gc} {bc}  {rc} {gc} {bc}  204  0"
            )
        lines += ["  0 0 0 0 0 0", ""]

        # VECTR / VECTT
        vec_entries = []
        for i, v in enumerate(mode_vec):
            if np.linalg.norm(v) >= thresh:
                vec_entries.append((i, v * scale))

        lines.append("VECTR")
        for vid, (atom_i, sv) in enumerate(vec_entries, start=1):
            lines.append(f" {vid:4d}  {sv[0]:10.5f}  {sv[1]:10.5f}  {sv[2]:10.5f}  0")
            lines.append(f" {atom_i+1:4d}  0  0  0  0")
            lines.append("  0  0  0  0  0")
        lines += [" 0 0 0 0 0", ""]

        lines.append("VECTT")
        for vid, _ in enumerate(vec_entries, start=1):
            lines.append(f" {vid:4d}   0.300  255   0   0   0")
        lines += [" 0 0 0 0 0", ""]

        lines += ["SPLAN", "  0   0   0   0", ""]

        # ATOMT — unique element types
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
