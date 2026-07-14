from __future__ import annotations

import re
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QFileDialog, QSpinBox,
    QSizePolicy,
)
from PyQt6.QtCore import Qt

import mplcursors

from app.widgets.plot_canvas import PlotCanvas, DARK

_NUMERIC_EXTS = {'.npy', '.npz', '.txt', '.dat'}


# ── Standalone parsers ────────────────────────────────────────────────────────

def _floats(line: str) -> list[float]:
    return [float(x) for x in re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', line)]


def parse_yaml(path: str) -> dict:
    """Parse Phonopy band.yaml → masses, freqs (THz), modes, species, positions (Å)."""
    with open(path) as f:
        lines = f.readlines()

    lattice, species, coords_frac, masses_list = [], [], [], []
    for i, l in enumerate(lines):
        if l.strip().startswith('phonon:'):
            break
        if l.strip().startswith('lattice:'):
            for j in range(1, 4):
                lattice.append(_floats(lines[i + j])[:3])
        if re.match(r'^-\s+symbol:', l):
            species.append(l.split()[2])
        if re.match(r'^\s+coordinates:', l):
            coords_frac.append(_floats(l)[:3])
        if re.match(r'^\s+mass:', l):
            masses_list.append(float(l.split()[1]))

    lattice    = np.array(lattice)
    positions  = np.array(coords_frac) @ lattice
    masses     = np.array(masses_list)
    N          = len(masses)

    # Re-read frequencies and eigenvectors (only first 3N = Gamma-point modes)
    lines_s = [l.strip() for l in lines]
    freqs, modes = [], []
    with open(path) as f:
        for ln, l in enumerate(f):
            if 'frequency:' in l:
                freqs.append(float(l.split()[1]))
                ev = []
                for i in range(ln + 3, ln + 4 * N + 2, 4):
                    xyz = [float(lines_s[i + j].split()[2].strip(',')) for j in range(3)]
                    ev.append(xyz)
                modes.append(ev)
                if len(modes) == 3 * N:
                    break

    freqs = np.clip(np.array(freqs, dtype=float), 0, None)
    modes = np.array(modes, dtype=float)
    return dict(masses=masses, freqs=freqs, modes=modes,
                species=species, positions=positions, source='yaml')


def parse_outcar(path: str) -> dict:
    """Parse VASP OUTCAR → masses, freqs (THz), modes, species, positions (Å)."""
    with open(path) as f:
        lines = [l.strip() for l in f]

    # Species
    titel_species: list[str] = []
    for l in lines:
        if 'TITEL' in l:
            m = re.search(r'PAW_PBE\s+(\w+)', l)
            if m and m.group(1) not in titel_species:
                titel_species.append(m.group(1))

    ions_line = next(l for l in lines if 'ions per type' in l)
    counts    = list(map(int, ions_line.split('=')[1].split()))
    species   = [sp for sp, c in zip(titel_species, counts) for _ in range(c)]
    N         = len(species)

    # Masses
    mass_idx = lines.index("Mass of Ions in am")
    raw_masses = np.array(lines[mass_idx + 1].split()[2:], dtype=float)
    masses = np.repeat(raw_masses, counts)

    # Phonon block bounds
    idx = next(i for i, l in enumerate(lines)
               if 'Eigenvectors and eigenvalues of the dynamical matrix' in l)
    end_idx = next(
        i for i, l in enumerate(lines)
        if i > idx and ('Finite differences POTIM=' in l
                        or 'ELASTIC MODULI CONTR FROM IONIC RELAXATION' in l)
    )

    freqs, modes, positions = [], [], None
    for i in range(idx, end_idx + 1):
        if 'THz' in lines[i]:
            toks = lines[i].split()
            freqs.append(float(toks[toks.index('THz') - 1]))
            block = [lines[j].split() for j in range(i + 2, i + 2 + N)]
            if positions is None:
                positions = np.array([[float(r[0]), float(r[1]), float(r[2])]
                                      for r in block])
            modes.append([[float(r[3]), float(r[4]), float(r[5])] for r in block])

    freqs = np.array(freqs, dtype=float)
    modes = np.array(modes, dtype=float)
    srt   = np.argsort(freqs)
    return dict(masses=masses, freqs=freqs[srt], modes=modes[srt],
                species=species, positions=positions, source='outcar')


def parse_numeric(modes_path: str, energies_path: str) -> dict:
    """Parse .npy/.npz/.txt/.dat: modes array + separate energies (meV)."""
    def _load(p):
        ext = Path(p).suffix.lower()
        if ext in ('.npy', '.npz'):
            d = np.load(p, allow_pickle=False)
            if hasattr(d, 'files'):
                d = d[d.files[0]]
            return d
        return np.loadtxt(p)

    modes = _load(modes_path)
    freqs = _load(energies_path)   # caller supplies meV directly

    if modes.ndim == 2:
        modes = modes.reshape(modes.shape[0], modes.shape[1] // 3, 3)

    return dict(masses=None, freqs=freqs, modes=modes,
                species=None, positions=None, source='numeric')


# ── Atom mapping ──────────────────────────────────────────────────────────────

def map_atoms(pos_p: np.ndarray, pos_d: np.ndarray,
              threshold: float = 0.5) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Nearest-neighbour matching: defect atom i → pristine atom d2p[i].
    d2p[i] = -1  when atom i is an interstitial (no pristine atom within threshold).
    vacancies: pristine indices unmatched by any defect atom.
    """
    dists = np.linalg.norm(
        pos_d[:, None, :] - pos_p[None, :, :], axis=2)   # (N_D, N_P)

    d2p       = np.full(len(pos_d), -1, dtype=int)
    matched_p : set[int] = set()

    for i in range(len(pos_d)):
        j = int(np.argmin(dists[i]))
        if dists[i, j] < threshold:
            d2p[i] = j
            matched_p.add(j)

    vacancies     = [j for j in range(len(pos_p)) if j not in matched_p]
    interstitials = [i for i in range(len(pos_d)) if d2p[i] < 0]
    return d2p, vacancies, interstitials


# ── Projection ────────────────────────────────────────────────────────────────

def compute_projection(data_d: dict, data_p: dict,
                       d2p: np.ndarray) -> np.ndarray:
    """
    c_sq[k, k'] = |c_{k,k'}|²

    c_{k,k'} = Σ_i  √(m^P_i / m^D_i)  ê^D_{k,i} · ê^P_{k', σ(i)}

    Vacancy: ê^D_{k,j} = 0 by construction (skipped via d2p).
    Interstitial: ê^P_{k', j_new} = 0 (pristine has no atom there).
    """
    modes_d  = data_d['modes']   # (N_k_d, N_D, 3)
    modes_p  = data_p['modes']   # (N_k_p, N_P, 3)
    masses_d = data_d['masses']
    masses_p = data_p['masses']
    N_D      = modes_d.shape[1]

    # Mass correction factor per defect atom
    if masses_d is not None and masses_p is not None:
        mf = np.zeros(N_D)
        for i in range(N_D):
            j = d2p[i]
            if j >= 0:
                mf[i] = np.sqrt(masses_p[j] / masses_d[i])
    else:
        mf = np.where(d2p >= 0, 1.0, 0.0).astype(float)

    # Weighted defect modes: (N_k_d, N_D*3)
    e_d = (modes_d * mf[None, :, None]).reshape(modes_d.shape[0], -1)

    # Re-ordered pristine modes aligned to defect atom indices: (N_k_p, N_D*3)
    N_k_p = modes_p.shape[0]
    e_p   = np.zeros((N_k_p, N_D, 3))
    for i in range(N_D):
        j = d2p[i]
        if j >= 0:
            e_p[:, i, :] = modes_p[:, j, :]
    e_p = e_p.reshape(N_k_p, -1)

    c = e_d @ e_p.T   # (N_k_d, N_k_p)
    return c ** 2


# ── Widget ────────────────────────────────────────────────────────────────────

class PhononProjectionWidget(QWidget):
    """
    Standalone sub-tab: project defect normal modes onto the pristine phonon basis.
    User loads two phonon files independently; no connection to the main results dict.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._data_p : dict | None = None
        self._data_d : dict | None = None
        self._d2p    : np.ndarray | None = None
        self._c_sq   : np.ndarray | None = None   # (N_k_d, N_k_p)
        self._cursor : object | None = None

        # paths for energy files (numeric mode)
        self._path_pe : str | None = None   # pristine energies
        self._path_de : str | None = None   # defect   energies

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── File input frame ──────────────────────────────────────────────
        file_frame = QFrame()
        file_frame.setObjectName("info_card")
        fl = QVBoxLayout(file_frame)
        fl.setContentsMargins(10, 8, 10, 8)
        fl.setSpacing(6)

        self._pristine_row, self._btn_p, self._lbl_p = self._file_row(
            "Pristine phonons:", fl, self._browse_pristine)
        self._defect_row, self._btn_d, self._lbl_d = self._file_row(
            "Defect phonons:", fl, self._browse_defect)

        # Extra rows for numeric files (hidden by default)
        self._pe_row, self._btn_pe, self._lbl_pe = self._file_row(
            "Pristine energies (meV):", fl, self._browse_pe)
        self._de_row, self._btn_de, self._lbl_de = self._file_row(
            "Defect energies (meV):", fl, self._browse_de)
        self._pe_row.setVisible(False)
        self._de_row.setVisible(False)

        root.addWidget(file_frame)

        # ── Control bar ───────────────────────────────────────────────────
        ctrl = QFrame()
        ctrl.setObjectName("info_card")
        cl = QHBoxLayout(ctrl)
        cl.setContentsMargins(10, 6, 10, 6)
        cl.setSpacing(12)

        cl.addWidget(QLabel("Defect mode k:"))
        self._mode_spin = QSpinBox()
        self._mode_spin.setRange(1, 1)
        self._mode_spin.setFixedWidth(68)
        self._mode_spin.setEnabled(False)
        self._mode_spin.valueChanged.connect(self._on_mode_changed)
        cl.addWidget(self._mode_spin)

        self._project_btn = QPushButton("▶  Project")
        self._project_btn.setEnabled(False)
        self._project_btn.clicked.connect(self._run_projection)
        cl.addWidget(self._project_btn)

        cl.addSpacing(8)
        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("hint_label")
        self._status_lbl.setStyleSheet("color: gray; font-size: 11px;")
        cl.addWidget(self._status_lbl, 1)

        root.addWidget(ctrl)

        # ── Hint label ────────────────────────────────────────────────────
        self._hint = QLabel(
            "Browse a pristine phonon file and a defect phonon file, "
            "then click  ▶ Project."
        )
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setObjectName("hint_label")
        self._hint.setStyleSheet("color: gray; font-style: italic; padding: 32px;")
        root.addWidget(self._hint, 1)

        # ── Plot ──────────────────────────────────────────────────────────
        self._canvas = PlotCanvas(nrows=1, ncols=1, figsize=(9, 5))
        self._canvas.setVisible(False)
        root.addWidget(self._canvas, 1)

        # ── Hover card ────────────────────────────────────────────────────
        self._hover_card = QFrame()
        self._hover_card.setObjectName("info_card")
        self._hover_card.setVisible(False)
        hc_lay = QHBoxLayout(self._hover_card)
        hc_lay.setSpacing(24)
        self._hc: dict[str, QLabel] = {}
        for key in ["Pristine mode k′", "E_k′ (meV)", "Freq (THz)", "|c|²", "Σ|c|² (all k′)"]:
            col = QVBoxLayout(); col.setSpacing(2)
            v = QLabel("—"); v.setObjectName("field_label")
            v.setStyleSheet("color:#cba6f7; font-size:14px; font-weight:bold;")
            lbl = QLabel(key); lbl.setObjectName("hint_label")
            col.addWidget(v); col.addWidget(lbl)
            hc_lay.addLayout(col)
            self._hc[key] = v
        hc_lay.addStretch()
        root.addWidget(self._hover_card)

    # ── File-row helper ───────────────────────────────────────────────────

    def _file_row(self, label_text: str, parent_layout,
                  callback) -> tuple[QFrame, QPushButton, QLabel]:
        row = QFrame()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        lbl = QLabel(label_text)
        lbl.setFixedWidth(168)
        rl.addWidget(lbl)
        btn = QPushButton("Browse…")
        btn.setFixedWidth(80)
        btn.clicked.connect(callback)
        rl.addWidget(btn)
        path_lbl = QLabel("—")
        path_lbl.setObjectName("hint_label")
        path_lbl.setStyleSheet("color: gray; font-size: 11px;")
        path_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        rl.addWidget(path_lbl, 1)
        parent_layout.addWidget(row)
        return row, btn, path_lbl

    # ── Browse callbacks ──────────────────────────────────────────────────

    def _browse_pristine(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pristine phonon file", "",
            "Phonon files (*.yaml OUTCAR* outcar* *.npy *.npz *.txt *.dat);;All files (*)")
        if path:
            self._load_file(path, 'pristine')

    def _browse_defect(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Defect phonon file", "",
            "Phonon files (*.yaml OUTCAR* outcar* *.npy *.npz *.txt *.dat);;All files (*)")
        if path:
            self._load_file(path, 'defect')

    def _browse_pe(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pristine mode energies (meV)", "",
            "Numeric files (*.npy *.npz *.txt *.dat);;All files (*)")
        if path:
            self._path_pe = path
            self._lbl_pe.setText(Path(path).name)
            self._try_load_numeric('pristine')

    def _browse_de(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Defect mode energies (meV)", "",
            "Numeric files (*.npy *.npz *.txt *.dat);;All files (*)")
        if path:
            self._path_de = path
            self._lbl_de.setText(Path(path).name)
            self._try_load_numeric('defect')

    # ── File loading ──────────────────────────────────────────────────────

    def _is_numeric(self, path: str) -> bool:
        return Path(path).suffix.lower() in _NUMERIC_EXTS

    def _load_file(self, path: str, side: str):
        lbl  = self._lbl_p  if side == 'pristine' else self._lbl_d
        pe_r = self._pe_row if side == 'pristine' else self._de_row
        lbl.setText(Path(path).name)

        if self._is_numeric(path):
            # Store path; wait for energy file too
            if side == 'pristine':
                self._path_p_modes = path
                self._data_p = None
            else:
                self._path_d_modes = path
                self._data_d = None
            pe_r.setVisible(True)
            self._update_project_btn()
            return

        pe_r.setVisible(False)
        try:
            ext = Path(path).suffix.lower()
            data = parse_yaml(path) if ext == '.yaml' else parse_outcar(path)
        except Exception as e:
            lbl.setText(f"Error: {e}")
            return

        if side == 'pristine':
            self._data_p = data
        else:
            self._data_d = data
            N_modes = data['modes'].shape[0]
            self._mode_spin.setRange(1, N_modes)
            self._mode_spin.setValue(1)

        self._update_project_btn()

    def _try_load_numeric(self, side: str):
        if side == 'pristine':
            mp = getattr(self, '_path_p_modes', None)
            ep = self._path_pe
        else:
            mp = getattr(self, '_path_d_modes', None)
            ep = self._path_de

        if mp is None or ep is None:
            return
        try:
            data = parse_numeric(mp, ep)
        except Exception as e:
            lbl = self._lbl_p if side == 'pristine' else self._lbl_d
            lbl.setText(f"Error: {e}")
            return

        if side == 'pristine':
            self._data_p = data
        else:
            self._data_d = data
            N_modes = data['modes'].shape[0]
            self._mode_spin.setRange(1, N_modes)
            self._mode_spin.setValue(1)

        self._update_project_btn()

    def _update_project_btn(self):
        ready = self._data_p is not None and self._data_d is not None
        self._project_btn.setEnabled(ready)
        self._mode_spin.setEnabled(ready and self._c_sq is not None)

    # ── Projection ────────────────────────────────────────────────────────

    def _run_projection(self):
        dp = self._data_p
        dd = self._data_d

        # Atom mapping
        if dp['positions'] is not None and dd['positions'] is not None:
            d2p, vac, inter = map_atoms(dp['positions'], dd['positions'])
            N_D = len(dd['positions'])
            matched = int(np.sum(d2p >= 0))
            self._status_lbl.setText(
                f"N_P={len(dp['positions'])}  N_D={N_D}  "
                f"matched={matched}  vacancies={len(vac)}  interstitials={len(inter)}"
            )
        else:
            # Numeric files: index-based mapping
            N_D = dd['modes'].shape[1]
            N_P = dp['modes'].shape[1]
            n   = min(N_D, N_P)
            d2p = np.full(N_D, -1, dtype=int)
            d2p[:n] = np.arange(n)
            self._status_lbl.setText(
                f"N_P={N_P}  N_D={N_D}  index-based mapping  "
                f"(no positions available)"
            )

        self._d2p  = d2p
        self._c_sq = compute_projection(dd, dp, d2p)   # (N_k_d, N_k_p)

        self._mode_spin.setEnabled(True)
        self._hint.setVisible(False)
        self._canvas.setVisible(True)
        self._replot()

    def _on_mode_changed(self):
        if self._c_sq is not None:
            self._replot()

    # ── Plot ──────────────────────────────────────────────────────────────

    def _replot(self):
        if self._c_sq is None or self._data_p is None or self._data_d is None:
            return

        k = self._mode_spin.value() - 1   # 0-indexed
        row = self._c_sq[k]               # (N_k_p,)

        dp = self._data_p
        dd = self._data_d

        # Pristine energies in meV
        if dp['source'] in ('yaml', 'outcar'):
            Ep = dp['freqs'] * 4.13566   # THz → meV
        else:
            Ep = dp['freqs']             # already meV

        # Defect energy for title
        if dd['source'] in ('yaml', 'outcar'):
            Ek_d = dd['freqs'][k] * 4.13566
            Ek_d_str = f"{Ek_d:.2f} meV"
        else:
            Ek_d_str = f"mode {k + 1}"

        sum_csq = float(row.sum())

        # Remove cursor before clearing
        if self._cursor is not None:
            try:
                self._cursor.remove()
            except Exception:
                pass
            self._cursor = None

        ax = self._canvas.ax
        ax.cla()

        freqs_thz = dp['freqs'] if dp['source'] in ('yaml', 'outcar') else Ep / 4.13566
        freqs_cm  = Ep * 8.0655

        # Lollipop: stem colour scaled by |c|²
        max_val = row.max() if row.max() > 0 else 1.0
        colors = [
            DARK["blue"] if v >= 0.3 * max_val else
            (DARK["purple"] if v >= 0.05 * max_val else DARK["spine"])
            for v in row
        ]

        ax.vlines(Ep, 0, row, colors=colors, linewidth=0.9, alpha=0.80, zorder=2)
        ax.axhline(0, color=DARK["spine"], lw=0.7, zorder=1)

        ax.set_xlabel(r"Pristine phonon energy  $E_{k'}^P$  (meV)", color=DARK["text"])
        ax.set_ylabel(r"$|c_{k,k'}|^2$", color=DARK["text"])
        ax.set_title(
            f"Defect mode k={k + 1}  ({Ek_d_str})  →  pristine basis"
            f"       Σ|c|² = {sum_csq:.4f}",
            color=DARK["text"], fontsize=9,
        )
        ax.set_ylim(bottom=0)

        self._canvas.fig.tight_layout()
        self._canvas.draw()

        # Invisible scatter for hover
        sc = ax.scatter(Ep, row, s=14, alpha=0, zorder=3)
        self._cursor = mplcursors.cursor(sc, hover=True)
        hc   = self._hc
        card = self._hover_card

        @self._cursor.connect("add")
        def on_add(sel):
            i = sel.index
            sel.annotation.set_text(
                f"Pristine mode {i + 1}\n"
                f"Energy: {Ep[i]:.3f} meV\n"
                f"|c|²: {row[i]:.6f}"
            )
            sel.annotation.get_bbox_patch().set(
                fc=DARK["axes_bg"], ec=DARK["spine"], alpha=0.92)
            sel.annotation.set_color(DARK["text"])
            sel.annotation.set_fontsize(10)
            hc["Pristine mode k′"].setText(f"#{i + 1}")
            hc["E_k′ (meV)"].setText(f"{Ep[i]:.4f}")
            hc["Freq (THz)"].setText(f"{freqs_thz[i]:.4f}")
            hc["|c|²"].setText(f"{row[i]:.6f}")
            hc["Σ|c|² (all k′)"].setText(f"{sum_csq:.4f}")
            card.setVisible(True)

        @self._cursor.connect("remove")
        def on_remove(sel):
            card.setVisible(False)

    # ── Public API ────────────────────────────────────────────────────────

    def clear(self):
        """Called when the main calculation changes; this tab is standalone so just reset."""
        pass   # user's loaded files are preserved across main-calc changes

    def populate(self, results: dict):
        pass   # standalone tab; ignores main results
