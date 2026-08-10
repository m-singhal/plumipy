from __future__ import annotations

import re
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QFileDialog, QSpinBox, QDoubleSpinBox,
    QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea,
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
    """Parse VASP OUTCAR → masses, freqs (THz), modes, species, positions (Å), lattice."""
    with open(path) as f:
        lines = [l.strip() for l in f]

    # Species
    titel_species: list[str] = []
    for l in lines:
        if 'TITEL' in l:
            m = re.search(r'PAW_PBE\s+(\w+)', l)
            if m and m.group(1) not in titel_species:
                titel_species.append(m.group(1))

    ions_line_idx = next(i for i, l in enumerate(lines) if 'ions per type' in l)
    ions_line     = lines[ions_line_idx]
    counts        = list(map(int, ions_line.split('=')[1].split()))
    species       = [sp for sp, c in zip(titel_species, counts) for _ in range(c)]
    N             = len(species)

    # Masses
    mass_idx   = lines.index("Mass of Ions in am")
    raw_masses = np.array(lines[mass_idx + 1].split()[2:], dtype=float)
    masses     = np.repeat(raw_masses, counts)

    # Supercell lattice — first occurrence of "direct lattice vectors" AFTER "ions per type"
    lat_idx = next(
        i for i, l in enumerate(lines)
        if i > ions_line_idx and 'direct lattice vectors' in l
    )
    lattice = np.array([_floats(lines[lat_idx + j + 1])[:3] for j in range(3)])

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
                species=species, positions=positions, lattice=lattice, source='outcar')


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
              lat_p: np.ndarray | None = None,
              lat_d: np.ndarray | None = None,
              threshold: float = 0.5) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Nearest-neighbour matching: defect atom i → pristine atom d2p[i].
    d2p[i] = -1  when atom i is an interstitial (no pristine atom within threshold).

    When lat_p and lat_d are provided (3×3 Å matrices) the matching uses the
    minimum-image convention in fractional coordinates so that PBC and
    different cell origins are handled correctly.
    """
    N_D = len(pos_d)
    N_P = len(pos_p)
    d2p       = np.full(N_D, -1, dtype=int)
    matched_p : set[int] = set()

    if lat_p is not None and lat_d is not None:
        inv_lat_p = np.linalg.inv(lat_p)
        inv_lat_d = np.linalg.inv(lat_d)
        frac_p = (pos_p @ inv_lat_p) % 1.0   # (N_P, 3)
        frac_d = (pos_d @ inv_lat_d) % 1.0   # (N_D, 3)

        # Auto-detect a rigid fractional shift between the two supercells.
        # Use the first defect atom to find the nearest pristine (without any
        # threshold), then derive shift = frac_d[0] - frac_p[nearest].
        df0   = frac_d[0] - frac_p            # (N_P, 3)
        df0  -= np.round(df0)
        j0    = int(np.argmin(np.linalg.norm(df0 @ lat_d, axis=1)))
        shift = frac_d[0] - frac_p[j0]
        shift -= np.round(shift)              # fractional, in (−0.5, 0.5]

        frac_d_aligned = (frac_d - shift) % 1.0

        for i in range(N_D):
            df = frac_d_aligned[i] - frac_p  # (N_P, 3)
            df -= np.round(df)
            dists_i = np.linalg.norm(df @ lat_d, axis=1)
            j = int(np.argmin(dists_i))
            if dists_i[j] < threshold:
                d2p[i] = j
                matched_p.add(j)
    else:
        dists = np.linalg.norm(
            pos_d[:, None, :] - pos_p[None, :, :], axis=2)   # (N_D, N_P)
        for i in range(N_D):
            j = int(np.argmin(dists[i]))
            if dists[i, j] < threshold:
                d2p[i] = j
                matched_p.add(j)

    vacancies     = [j for j in range(N_P) if j not in matched_p]
    interstitials = [i for i in range(N_D) if d2p[i] < 0]
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

    c = e_d @ e_p.T   # (N_k_d, N_k_p), signed
    return c, c ** 2


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
        self._c      : np.ndarray | None = None   # (N_k_d, N_k_p), signed
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

        cl.addWidget(QLabel("σ (meV):"))
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.1, 50.0)
        self._sigma_spin.setSingleStep(0.5)
        self._sigma_spin.setValue(2.0)
        self._sigma_spin.setFixedWidth(68)
        self._sigma_spin.setEnabled(False)
        self._sigma_spin.valueChanged.connect(self._on_sigma_changed)
        cl.addWidget(self._sigma_spin)

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

        # ── Scrollable area: plot then table ──────────────────────────────
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll_area.setVisible(False)

        _inner = QWidget()
        _inner_layout = QVBoxLayout(_inner)
        _inner_layout.setContentsMargins(0, 0, 4, 8)
        _inner_layout.setSpacing(8)

        # Mirrored stick spectrum: pristine up, defect down
        self._spec_canvas = PlotCanvas(nrows=1, ncols=1, figsize=(14, 3))
        self._spec_canvas.setMinimumHeight(230)
        self._spec_canvas.setMaximumHeight(300)
        self._spec_canvas.canvas.wheelEvent = lambda e: self._scroll_area.wheelEvent(e)
        _inner_layout.addWidget(self._spec_canvas)

        # Canvas — tall enough to fill the viewport so table starts below the fold
        self._canvas = PlotCanvas(nrows=1, ncols=1, figsize=(14, 8))
        self._canvas.setMinimumHeight(520)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # matplotlib consumes wheel events; forward them to the scroll area
        self._canvas.wheelEvent = lambda e: self._scroll_area.wheelEvent(e)
        _inner_layout.addWidget(self._canvas)

        # Table label + table (visible only after scrolling down)
        self._table_lbl = QLabel("Contributing pristine modes  (|c|² > 1×10⁻⁸, sorted by |c|²):")
        self._table_lbl.setObjectName("hint_label")
        self._table_lbl.setStyleSheet("color: gray; font-size: 11px; padding-top: 4px;")
        self._table_lbl.setVisible(False)
        _inner_layout.addWidget(self._table_lbl)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Pristine mode k′", "Energy (meV)", "Freq (THz)", "c", "|c|²"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        # No internal scrollbar — outer scroll area handles it
        self._table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._table.setVisible(False)
        _inner_layout.addWidget(self._table)
        _inner_layout.addStretch(1)

        self._scroll_area.setWidget(_inner)
        root.addWidget(self._scroll_area, 1)

        # ── Hover card — pinned outside the scroll area ───────────────────
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
            lat_p = dp.get('lattice')
            lat_d = dd.get('lattice')
            d2p, vac, inter = map_atoms(dp['positions'], dd['positions'], lat_p, lat_d)
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
        self._c, self._c_sq = compute_projection(dd, dp, d2p)   # (N_k_d, N_k_p)

        self._mode_spin.setEnabled(True)
        self._sigma_spin.setEnabled(True)
        self._hint.setVisible(False)
        self._scroll_area.setVisible(True)
        self._plot_spectrum()
        self._replot()

    def _on_mode_changed(self):
        if self._c_sq is not None:
            self._replot()

    def _on_sigma_changed(self):
        if self._c_sq is not None:
            self._replot()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _plot_spectrum(self):
        """Mirrored stick spectrum — pristine modes up, defect modes down."""
        Ep, Ed = self._energies()

        fig = self._spec_canvas.fig
        fig.set_layout_engine("none")          # keep room for the side labels
        ax = self._spec_canvas.ax
        ax.cla()

        ax.vlines(Ep, 0,  1, color=DARK["blue"],   lw=0.8, alpha=0.85)
        ax.vlines(Ed, 0, -1, color=DARK["purple"], lw=0.8, alpha=0.85)
        ax.axhline(0, color=DARK["text"], lw=2.0, zorder=5)

        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.set_xlim(0, max(float(Ep.max()), float(Ed.max())) * 1.03)

        ax.text(-0.055, 0.75, "Pristine", transform=ax.transAxes,
                fontsize=13, fontweight="bold", color=DARK["blue"],
                ha="center", va="center", clip_on=False)
        ax.text(-0.055, 0.25, "Defect", transform=ax.transAxes,
                fontsize=13, fontweight="bold", color=DARK["purple"],
                ha="center", va="center", clip_on=False)

        ax.set_xlabel("Phonon Energy  (meV)", color=DARK["text"],
                      fontsize=12, fontweight="bold")
        ax.set_title(
            f"Mode spectra   —   pristine: {len(Ep)} modes,  defect: {len(Ed)} modes",
            color=DARK["text"], fontsize=9,
        )
        ax.grid(alpha=0.3, axis="x")

        fig.subplots_adjust(left=0.11, right=0.99, top=0.86, bottom=0.26)
        self._spec_canvas.draw()

    def _energies(self):
        dp = self._data_p
        dd = self._data_d
        Ep = dp['freqs'] * 4.13566 if dp['source'] in ('yaml', 'outcar') else dp['freqs']
        Ed = dd['freqs'] * 4.13566 if dd['source'] in ('yaml', 'outcar') else dd['freqs']
        return Ep, Ed

    def _clear_cursor(self):
        if self._cursor is not None:
            try:
                self._cursor.remove()
            except Exception:
                pass
            self._cursor = None

    # ── Projection table ──────────────────────────────────────────────────

    def _update_table(self):
        k    = self._mode_spin.value() - 1
        row  = self._c_sq[k]
        c_signed = self._c[k]
        Ep, _ = self._energies()
        freqs_thz_p = self._data_p['freqs']

        # All modes above numerical noise floor, sorted descending by |c|²
        THRESH = 1e-8
        idx    = np.where(row > THRESH)[0]
        srt    = idx[np.argsort(row[idx])[::-1]]

        self._table.setRowCount(len(srt))
        for r, ki in enumerate(srt):
            c2 = float(row[ki])
            c  = float(c_signed[ki])
            vals = [f"{ki + 1}", f"{Ep[ki]:.4f}", f"{freqs_thz_p[ki]:.4f}",
                    f"{c:.6f}", f"{c2:.6f}"]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(r, col, item)

        # Size the table to show all rows so the outer scroll area is the only scroller
        row_h    = self._table.verticalHeader().defaultSectionSize()
        header_h = self._table.horizontalHeader().height()
        total_h  = header_h + len(srt) * row_h + 4
        self._table.setMinimumHeight(total_h)
        self._table.setMaximumHeight(total_h)

        self._table.setVisible(True)
        self._table_lbl.setVisible(True)

    # ── Plot ──────────────────────────────────────────────────────────────

    def _replot(self):
        if self._c_sq is None or self._data_p is None or self._data_d is None:
            return

        k      = self._mode_spin.value() - 1
        row    = self._c_sq[k]
        sigma  = self._sigma_spin.value()
        Ep, Ed = self._energies()
        sum_csq = float(row.sum())

        self._clear_cursor()
        self._hover_card.setVisible(False)

        ax = self._canvas.ax
        ax.cla()

        # ── Gaussian spectral function ────────────────────────────────────
        # G(E) = Σ_{k'} |c|² exp(-(E-E_k')²/(2σ²))
        # Peak at each mode = |c|², so y-axis is shared with scatter.
        E_lo = max(0.0, Ep.min() - 6 * sigma)
        E_hi = Ep.max() + 6 * sigma
        E_fine = np.linspace(E_lo, E_hi, 4000)
        gauss = np.zeros_like(E_fine)
        # Only modes above noise floor contribute meaningfully
        max_val = float(row.max()) if row.max() > 0 else 1.0
        active  = row > max_val * 1e-5
        for e_k, c2 in zip(Ep[active], row[active]):
            gauss += c2 * np.exp(-0.5 * ((E_fine - e_k) / sigma) ** 2)

        ax.fill_between(E_fine, gauss,
                        color=DARK["blue"], alpha=0.18, zorder=1)
        ax.plot(E_fine, gauss,
                color=DARK["blue"], lw=1.1, alpha=0.7, zorder=2)

        # ── Scatter: only points above 0.5 % of max ──────────────────────
        thresh = max_val * 0.005
        mask   = row > thresh
        Ep_m   = Ep[mask]
        row_m  = row[mask]
        orig_i = np.where(mask)[0]           # indices back into full Ep / row

        # Marker area ∝ |c|² (linear), max size = 250 pt²
        sizes  = row_m / max_val * 250

        # Three-tier colour by relative magnitude
        colors_m = np.where(
            row_m >= 0.3 * max_val, 0,
            np.where(row_m >= 0.05 * max_val, 1, 2)
        )
        palette = [DARK["blue"], DARK["purple"], DARK["spine"]]
        c_rgba  = [palette[ci] for ci in colors_m]

        sc = ax.scatter(Ep_m, row_m,
                        s=sizes, c=c_rgba, alpha=0.85,
                        linewidths=0.4, edgecolors='white',
                        zorder=3)

        ax.axhline(0, color=DARK["spine"], lw=0.5, zorder=0)

        freqs_thz_p = self._data_p['freqs']   # THz, aligned with Ep

        Ek_str = f"{Ed[k]:.2f} meV" if self._data_d['source'] in ('yaml', 'outcar') \
                 else f"mode {k + 1}"
        ax.set_xlabel(r"Pristine phonon energy  $E_{k'}^P$  (meV)", color=DARK["text"])
        ax.set_ylabel(r"$|c_{k,k'}|^2$", color=DARK["text"])
        ax.set_title(
            f"Defect mode k={k + 1}  ({Ek_str})  →  pristine basis"
            f"       Σ|c|² = {sum_csq:.4f}   σ = {sigma:.1f} meV",
            color=DARK["text"], fontsize=9,
        )
        ax.set_ylim(bottom=-0.008 * max_val)
        ax.set_xlim(E_lo, E_hi)

        self._canvas.fig.tight_layout()
        self._canvas.draw()

        # ── Projection table ───────────────────────────────────────────────
        self._update_table()

        # ── Hover ─────────────────────────────────────────────────────────
        self._cursor = mplcursors.cursor(sc, hover=True)
        hc   = self._hc
        card = self._hover_card

        @self._cursor.connect("add")
        def on_add(sel):
            i    = orig_i[sel.index]
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
            hc["Freq (THz)"].setText(f"{freqs_thz_p[i]:.4f}")
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
