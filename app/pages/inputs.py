from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QScrollArea, QButtonGroup, QRadioButton,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QStackedWidget,
    QSizePolicy, QFileDialog, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from app.widgets.file_picker import FilePicker


# ─────────────────────────────────────────────────────────────────────────────
# Step indicator row
# ─────────────────────────────────────────────────────────────────────────────
class StepIndicator(QWidget):
    TITLES = ["Workflow", "Input Files", "Parameters", "Advanced"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = 0
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        self._circles = []
        self._titles = []
        self._build()

    def _build(self):
        n = len(self.TITLES)
        for i, title in enumerate(self.TITLES):
            col = QVBoxLayout()
            col.setSpacing(4)
            col.setAlignment(Qt.AlignmentFlag.AlignHCenter)

            circle = QLabel(str(i + 1))
            circle.setObjectName("step_inactive")
            circle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(circle, 0, Qt.AlignmentFlag.AlignHCenter)

            lbl = QLabel(title)
            lbl.setObjectName("hint_label")
            lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            col.addWidget(lbl, 0, Qt.AlignmentFlag.AlignHCenter)

            self._lay.addLayout(col)
            self._circles.append(circle)
            self._titles.append(lbl)

            if i < n - 1:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setFixedHeight(2)
                line.setStyleSheet("background:#313244; margin-top:10px;")
                self._lay.addWidget(line, 1)

        self._refresh()

    def set_step(self, step: int):
        self._current = step
        self._refresh()

    def _refresh(self):
        for i, (c, t) in enumerate(zip(self._circles, self._titles)):
            if i < self._current:
                c.setObjectName("step_done")
                c.setText("✓")
                t.setStyleSheet("color: #a6e3a1;")
            elif i == self._current:
                c.setObjectName("step_active")
                c.setText(str(i + 1))
                t.setStyleSheet("color: #cba6f7; font-weight: bold;")
            else:
                c.setObjectName("step_inactive")
                c.setText(str(i + 1))
                t.setStyleSheet("color: #6c7086;")
            c.style().unpolish(c)
            c.style().polish(c)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Workflow choice
# ─────────────────────────────────────────────────────────────────────────────
WORKFLOWS = [
    {
        "key": "adiabatic",
        "icon": "⚛",
        "title": "Adiabatic Approximation  (structure-based)",
        "body": (
            "The most accurate approach. You provide the fully relaxed geometries of both "
            "the ground state (GS) and excited state (ES). The structural displacement "
            "ΔR = R<sub>ES</sub> − R<sub>GS</sub> is projected onto the phonon "
            "normal modes to obtain mass-weighted displacements q<sub>k</sub> and "
            "Huang–Rhys factors S<sub>k</sub> = ω<sub>k</sub> q<sub>k</sub><sup>2</sup>/2ℏ."
            "<br><br>"
            "<b>Requires:</b> GS structure + ES structure + GS phonons<br>"
            "<i>Formats:</i> POSCAR/CONTCAR (VASP), OUTCAR or band.yaml (phonons)"
        ),
    },
    {
        "key": "gradient",
        "icon": "⚡",
        "title": "Vertical Gradient Approximation  (force-based)",
        "body": (
            "Use this when a full ES relaxation is too expensive. Provide forces from "
            "GS and ES single-point calculations at a reference geometry. The electron–phonon "
            "coupling is estimated from the force difference "
            "ΔF = F<sub>ES</sub> − F<sub>GS</sub> projected onto phonon modes.<br><br>"
            "<b>Two variants:</b><br>"
            "• <b>GS geometry</b>: both calculations at the GS minimum (standard vertical gradient)<br>"
            "• <b>ES geometry</b>: both calculations at the ES minimum (adiabatic correction)<br><br>"
            "<b>Requires:</b> GS forces + ES forces + GS phonons<br>"
            "<i>Formats:</i> OUTCAR (VASP), .npy, .npz, or whitespace-delimited .dat/.txt"
        ),
    },
    {
        "key": "vibrational",
        "icon": "♦",
        "title": "External Vibrational Data",
        "body": (
            "You have frequencies and normal modes from an external code — "
            "Gaussian, CP2K, ORCA, Quantum ESPRESSO, or any other program. "
            "Choose adiabatic (structure-based) or vertical gradient (force-based) "
            "to compute Huang–Rhys factors.<br><br>"
            "<b>Requires:</b> Frequencies + Normal modes + Atomic masses + "
            "GS/ES structures <i>or</i> GS/ES forces<br>"
            "<i>Formats:</i> .npy · .npz · .dat · .txt"
        ),
    },
]


class WorkflowCard(QFrame):
    selected = pyqtSignal(str)

    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.key = data["key"]
        self._active = False
        self.setObjectName("card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style()

        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        header = QLabel(f"{data['icon']}  {data['title']}")
        header.setFont(QFont("Helvetica Neue", 13, QFont.Weight.Bold))
        header.setStyleSheet("color: #cdd6f4;")
        lay.addWidget(header)

        body = QLabel(data["body"])
        body.setWordWrap(True)
        body.setObjectName("hint_label")
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setStyleSheet("color: #a6adc8; font-size: 13px;")
        lay.addWidget(body)

    def _apply_style(self):
        if self._active:
            self.setStyleSheet(
                "QFrame#card { border: 2px solid #cba6f7; background-color: #2a2a3e; border-radius:10px; padding:12px; }"
            )
        else:
            self.setStyleSheet(
                "QFrame#card { border: 2px solid #313244; background-color: #181825; border-radius:10px; padding:12px; }"
            )

    def set_active(self, v: bool):
        self._active = v
        self._apply_style()

    def mousePressEvent(self, e):
        self.selected.emit(self.key)
        super().mousePressEvent(e)


class Step1Workflow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected = "adiabatic"

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)

        title = QLabel("Which inputs do you have from your DFT calculation?")
        title.setObjectName("section_title")
        lay.addWidget(title)

        hint = QLabel(
            "Choose the workflow that matches the data you have available. "
            "You can always come back and change this."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hint_label")
        lay.addWidget(hint)

        self._cards = {}
        for w in WORKFLOWS:
            card = WorkflowCard(w)
            card.selected.connect(self._on_select)
            self._cards[w["key"]] = card
            lay.addWidget(card)

        self._cards["adiabatic"].set_active(True)
        lay.addStretch()

    def _on_select(self, key):
        for k, c in self._cards.items():
            c.set_active(k == key)
        self._selected = key

    def get_workflow(self) -> str:
        return self._selected


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Input files (adapts to workflow)
# ─────────────────────────────────────────────────────────────────────────────

def _section(title, hint=""):
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    t = QLabel(title)
    t.setObjectName("field_label")
    lay.addWidget(t)
    if hint:
        h = QLabel(hint)
        h.setObjectName("hint_label")
        h.setWordWrap(True)
        lay.addWidget(h)
    return w, lay


STRUCT_FILTER = "All Files (*);;Structure files (*.npy *.npz *.txt *.dat *.vasp)"
PHONON_FILTER = "All Files (*);;Phonon files (*.yaml *.npy *.npz *.txt *.dat)"
FORCE_FILTER  = "All Files (*);;Force files (*.npy *.npz *.txt *.dat)"
ARRAY_FILTER  = "All Files (*);;Array files (*.npy *.npz *.txt *.dat)"


class Step2Files(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._workflow = "adiabatic"

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(0)

        # Each panel gets its OWN picker dict so keys never collide
        self._pickers_adiabatic   = {}
        self._pickers_gradient    = {}
        self._pickers_vibrational = {}
        self._pickers_vib_struct  = {}   # structure_gs / structure_es (adiabatic)
        self._pickers_vib_forces  = {}   # forces_gs / forces_es (vertical gradient)
        self._vib_approx          = "r"  # "r" = adiabatic, "f" = gradient

        self._stack = QStackedWidget()
        self._outer.addWidget(self._stack)

        self._build_adiabatic()
        self._build_gradient()
        self._build_vibrational()

    def _make_picker(self, store: dict, key, label, hint, ffilter, optional=False):
        row = QWidget()
        lay = QVBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        suffix = "  <span style='color:#6c7086; font-size:12px'>(optional)</span>" if optional else ""
        lbl = QLabel(label + suffix)
        lbl.setObjectName("field_label")
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(lbl)
        h_lbl = QLabel(hint)
        h_lbl.setObjectName("hint_label")
        h_lbl.setTextFormat(Qt.TextFormat.RichText)
        h_lbl.setWordWrap(True)
        h_lbl.setStyleSheet("color: #585b70; font-size: 12px; padding-bottom: 2px;")
        lay.addWidget(h_lbl)
        picker = FilePicker(hint="", file_filter=ffilter, optional=False)
        lay.addWidget(picker)
        store[key] = picker
        return row

    def _build_adiabatic(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        p = self._pickers_adiabatic

        lay.addWidget(self._info(
            "Provide the fully relaxed GS and ES geometries plus GS phonons. "
            "The displacement ΔR = R<sub>ES</sub> − R<sub>GS</sub> is projected "
            "onto phonon normal modes to compute S<sub>k</sub>. "
            "ES phonons are optional — needed only for the squeezed oscillator model (Step 4)."
        ))

        lay.addWidget(self._section_header("Structures"))
        lay.addWidget(self._make_picker(p, "structure_gs", "Ground state structure",
            "Relaxed GS geometry. Accepted: POSCAR / CONTCAR (VASP, any extension), "
            ".npy array shape (Natoms, 3) in Å", STRUCT_FILTER))
        lay.addWidget(self._make_picker(p, "structure_es", "Excited state structure",
            "Relaxed ES geometry — same formats as above", STRUCT_FILTER))

        lay.addWidget(self._section_header("Phonons"))
        lay.addWidget(self._make_picker(p, "phonons_gs", "Ground state phonons",
            "OUTCAR (VASP, contains vibrational analysis) · band.yaml / qpoints.yaml (Phonopy)",
            PHONON_FILTER))
        lay.addWidget(self._make_picker(p, "phonons_es",
            "Excited state phonons  <span style='color:#6c7086'>(optional)</span>",
            "Required only for 'Displaced–Squeezed Oscillator' in Step 4. Same formats as GS.",
            PHONON_FILTER, optional=True))

        lay.addStretch()
        self._stack.addWidget(w)

    def _build_gradient(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        p = self._pickers_gradient

        lay.addWidget(self._info(
            "Provide GS and ES single-point forces at the same reference geometry, "
            "plus GS phonons. No relaxed ES structure needed. "
            "The coupling α<sub>k</sub> is obtained from "
            "ΔF = F<sub>ES</sub> − F<sub>GS</sub> projected onto phonon eigenvectors.<br><br>"
            "<b>Tip:</b> You can use the GS minimum geometry (standard vertical gradient) "
            "<i>or</i> the ES minimum geometry (adiabatic correction via ES single-point)."
        ))

        lay.addWidget(self._section_header("Forces"))
        lay.addWidget(self._make_picker(p, "forces_gs", "Ground state forces",
            "GS forces at reference geometry. "
            "Accepted: OUTCAR (VASP) · .npy / .npz array shape (Natoms, 3) [eV/Å] · "
            ".dat / .txt whitespace-delimited", FORCE_FILTER))
        lay.addWidget(self._make_picker(p, "forces_es", "Excited state forces",
            "ES forces at the same reference geometry — same formats as GS forces",
            FORCE_FILTER))

        lay.addWidget(self._section_header("Phonons"))
        lay.addWidget(self._make_picker(p, "phonons_gs", "Ground state phonons",
            "OUTCAR (VASP, contains vibrational analysis) · band.yaml / qpoints.yaml (Phonopy)",
            PHONON_FILTER))

        lay.addStretch()
        self._stack.addWidget(w)

    def _build_vibrational(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        p = self._pickers_vibrational

        lay.addWidget(self._info(
            "Provide vibrational frequencies, normal modes, and coupling data from any "
            "external code — Gaussian, CP2K, ORCA, Quantum ESPRESSO, MOLPRO, etc. "
            "Select how the Huang–Rhys factors should be computed below.<br><br>"
            "<b>Array formats accepted:</b> "
            ".npy (NumPy binary) · .npz (NumPy compressed) · "
            ".dat or .txt (whitespace-delimited text)"
        ))

        # ── HR Calculation Method ──────────────────────────────────────
        lay.addWidget(self._section_header("Huang–Rhys Calculation Method"))

        method_frame = QFrame()
        method_frame.setObjectName("info_card")
        mf_lay = QVBoxLayout(method_frame)
        mf_lay.setSpacing(10)

        self._vib_rb_adiabatic = QRadioButton(
            "Adiabatic Approximation  (structure-based, recommended)"
        )
        self._vib_rb_gradient = QRadioButton(
            "Vertical Gradient Approximation  (force-based)"
        )
        self._vib_rb_adiabatic.setChecked(True)
        self._vib_rb_adiabatic.toggled.connect(self._toggle_vib_method)

        mf_lay.addWidget(self._vib_rb_adiabatic)
        lbl_ad = QLabel(
            "Provide relaxed GS and ES structures. Displacement "
            "ΔR = R<sub>ES</sub> − R<sub>GS</sub> is projected onto normal modes."
        )
        lbl_ad.setObjectName("hint_label")
        lbl_ad.setTextFormat(Qt.TextFormat.RichText)
        lbl_ad.setStyleSheet("margin-left: 24px; color: #a6adc8; font-size: 12px;")
        mf_lay.addWidget(lbl_ad)

        mf_lay.addWidget(self._vib_rb_gradient)
        lbl_gr = QLabel(
            "Provide GS and ES forces at the same reference geometry. "
            "Coupling estimated from ΔF = F<sub>ES</sub> − F<sub>GS</sub> "
            "projected onto normal modes."
        )
        lbl_gr.setObjectName("hint_label")
        lbl_gr.setTextFormat(Qt.TextFormat.RichText)
        lbl_gr.setStyleSheet("margin-left: 24px; color: #a6adc8; font-size: 12px;")
        mf_lay.addWidget(lbl_gr)
        lay.addWidget(method_frame)

        # Stacked widget: structures (idx=0) vs forces (idx=1)
        self._vib_coupling_stack = QStackedWidget()

        struct_page = QWidget()
        sp_lay = QVBoxLayout(struct_page)
        sp_lay.setContentsMargins(0, 0, 0, 0)
        sp_lay.setSpacing(12)
        sp_lay.addWidget(self._section_header("Structures"))
        sp_lay.addWidget(self._make_picker(
            self._pickers_vib_struct, "structure_gs", "Ground state structure",
            "Relaxed GS geometry. Accepted: POSCAR/CONTCAR (VASP), "
            ".npy array shape (N_atoms, 3) in Å, .dat/.txt whitespace-delimited",
            STRUCT_FILTER))
        sp_lay.addWidget(self._make_picker(
            self._pickers_vib_struct, "structure_es", "Excited state structure",
            "Relaxed ES geometry — same formats as GS", STRUCT_FILTER))
        sp_lay.addStretch()
        self._vib_coupling_stack.addWidget(struct_page)

        force_page = QWidget()
        fp_lay = QVBoxLayout(force_page)
        fp_lay.setContentsMargins(0, 0, 0, 0)
        fp_lay.setSpacing(12)
        fp_lay.addWidget(self._section_header("Forces"))
        fp_lay.addWidget(self._make_picker(
            self._pickers_vib_forces, "forces_gs", "Ground state forces",
            "GS forces at reference geometry. "
            "Accepted: .npy/.npz array shape (N_atoms, 3) [eV/Å] · "
            ".dat/.txt whitespace-delimited",
            FORCE_FILTER))
        fp_lay.addWidget(self._make_picker(
            self._pickers_vib_forces, "forces_es", "Excited state forces",
            "ES forces at the same reference geometry — same formats as GS",
            FORCE_FILTER))
        fp_lay.addStretch()
        self._vib_coupling_stack.addWidget(force_page)

        lay.addWidget(self._vib_coupling_stack)

        # ── Frequency unit ─────────────────────────────────────────────
        unit_row = QHBoxLayout()
        unit_row.setSpacing(12)
        lbl_u = QLabel("Frequency units:")
        lbl_u.setObjectName("field_label")
        unit_row.addWidget(lbl_u)
        self._unit_combo = QComboBox()
        self._unit_combo.addItems(["cm⁻¹", "THz"])
        self._unit_combo.setFixedWidth(110)
        unit_row.addWidget(self._unit_combo)
        unit_row.addStretch()
        lay.addLayout(unit_row)

        # ── Ground state vibrational ────────────────────────────────────
        lay.addWidget(self._section_header("Ground State Vibrational Properties"))
        lay.addWidget(self._make_picker(p, "vib_freqs_gs",
            "Vibrational frequencies  — shape (N<sub>modes</sub>,)",
            "1D array of mode frequencies. N_modes = 3·N_atoms (include all modes; "
            "low-frequency modes can be subtracted in Step 3).", ARRAY_FILTER))
        lay.addWidget(self._make_picker(p, "vib_modes_gs",
            "Normal mode eigenvectors  — shape (N<sub>modes</sub>, N<sub>atoms</sub>, 3)",
            "3D array: each row is a mode, each atom has 3 Cartesian components. "
            "Mass-weighted (amu½·Å) or unweighted (Å) — specify in file or README.",
            ARRAY_FILTER))

        # ── Excited state vibrational (optional) ───────────────────────
        lay.addWidget(self._section_header(
            "Excited State Vibrational Properties  "
            "<span style='color:#6c7086; font-weight:normal'>(optional — needed for squeezing)</span>"
        ))
        lay.addWidget(self._make_picker(p, "vib_freqs_es", "Excited state frequencies",
            "Leave empty to reuse GS frequencies (no squeezing)", ARRAY_FILTER, optional=True))
        lay.addWidget(self._make_picker(p, "vib_modes_es", "Excited state normal modes",
            "Leave empty to reuse GS modes", ARRAY_FILTER, optional=True))

        # ── Atomic masses ──────────────────────────────────────────────
        lay.addWidget(self._section_header("Atomic Masses"))
        lay.addWidget(self._make_picker(p, "masses",
            "Atomic masses  — shape (N<sub>atoms</sub>,)  [amu]",
            "1D array in atomic mass units (e.g. H=1.008, C=12.011, N=14.007, Si=28.085)",
            ARRAY_FILTER))

        lay.addStretch()
        self._stack.addWidget(w)

    @staticmethod
    def _info(text):
        f = QFrame()
        f.setObjectName("info_card")
        lay = QVBoxLayout(f)
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setObjectName("hint_label")
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setStyleSheet("color: #a6adc8; font-size: 13px;")
        lay.addWidget(lbl)
        return f

    @staticmethod
    def _section_header(text):
        lbl = QLabel(f"<b>{text}</b>")
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setStyleSheet("color: #cdd6f4; font-size: 14px; padding-top: 4px;")
        return lbl

    def _toggle_vib_method(self, adiabatic_checked: bool):
        self._vib_coupling_stack.setCurrentIndex(0 if adiabatic_checked else 1)
        self._vib_approx = "r" if adiabatic_checked else "f"

    def get_vib_approx(self) -> str:
        return self._vib_approx

    def set_workflow(self, wf: str):
        self._workflow = wf
        idx = {"adiabatic": 0, "gradient": 1, "vibrational": 2}[wf]
        self._stack.setCurrentIndex(idx)

    def get_freq_unit(self) -> str:
        text = self._unit_combo.currentText()
        return "cm^-1" if text == "cm⁻¹" else "THz"

    def get_paths(self) -> dict:
        store = {
            "adiabatic":   self._pickers_adiabatic,
            "gradient":    self._pickers_gradient,
            "vibrational": self._pickers_vibrational,
        }[self._workflow]
        paths = {k: (p.path() or None) for k, p in store.items()}
        if self._workflow == "vibrational":
            coupling = (
                self._pickers_vib_struct if self._vib_approx == "r"
                else self._pickers_vib_forces
            )
            paths.update({k: (p.path() or None) for k, p in coupling.items()})
        return paths


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Physical parameters
# ─────────────────────────────────────────────────────────────────────────────
class LabeledSpin(QWidget):
    def __init__(self, label, default, min_=0.0, max_=1e6, decimals=1,
                 suffix="", hint="", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setObjectName("field_label")
        row.addWidget(lbl)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(min_, max_)
        self.spin.setDecimals(decimals)
        self.spin.setValue(default)
        self.spin.setSuffix(f"  {suffix}" if suffix else "")
        self.spin.setFixedWidth(120)
        row.addWidget(self.spin)
        row.addStretch()
        lay.addLayout(row)

        if hint:
            h = QLabel(hint)
            h.setObjectName("hint_label")
            h.setWordWrap(True)
            h.setTextFormat(Qt.TextFormat.RichText)
            h.setStyleSheet("color: #6c7086; font-size: 13px;")
            lay.addWidget(h)

    def value(self):
        return self.spin.value()


class Step3Parameters(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(18)

        # ZPL
        self.zpl = LabeledSpin(
            "Zero-Phonon Line (ZPL)  E₀₋₀", 1000.0, 0, 1e7, 1, "meV",
            hint=(
                "The purely electronic 0→0 transition energy with zero phonons involved. "
                "Sets the spectral centre — emission and absorption are symmetric about "
                "E<sub>ZPL</sub>. In defect physics this is the defect transition energy; "
                "in molecular spectroscopy it is the 0–0 line."
            )
        )
        lay.addWidget(self.zpl)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("color:#313244;"); lay.addWidget(sep1)

        # Broadening
        brd_title = QLabel("Phonon Sideband Broadening  (σ)")
        brd_title.setObjectName("section_title")
        lay.addWidget(brd_title)

        brd_hint = QLabel(
            "<b>σ controls the broadening applied to each individual phonon sideband.</b> "
            "Each mode's contribution is convolved with a Gaussian (or Lorentzian) of "
            "width σ. You can vary σ linearly from σ<sub>1</sub> at the lowest-frequency "
            "mode to σ<sub>2</sub> at the highest — set both equal for homogeneous broadening. "
            "This does <i>not</i> affect the ZPL width; see γ below for that."
        )
        brd_hint.setWordWrap(True)
        brd_hint.setObjectName("hint_label")
        brd_hint.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(brd_hint)

        type_row = QHBoxLayout()
        type_row.setSpacing(20)
        type_row.addWidget(QLabel("Sideband line shape:"))
        self._rb_gauss = QRadioButton("Gaussian")
        self._rb_gauss.setChecked(True)
        self._rb_lor = QRadioButton("Lorentzian")
        type_row.addWidget(self._rb_gauss)
        type_row.addWidget(self._rb_lor)
        type_row.addStretch()
        lay.addLayout(type_row)

        sigma_row = QHBoxLayout()
        sigma_row.setSpacing(20)
        self.sigma1 = LabeledSpin(
            "σ₁  (lowest mode)", 3.0, 0.1, 500, 1, "meV",
            hint="Sideband width at the lowest phonon frequency"
        )
        self.sigma2 = LabeledSpin(
            "σ₂  (highest mode)", 3.0, 0.1, 500, 1, "meV",
            hint="Sideband width at the highest phonon frequency"
        )
        sigma_row.addWidget(self.sigma1)
        sigma_row.addWidget(self.sigma2)
        sigma_row.addStretch()
        lay.addLayout(sigma_row)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#313244;"); lay.addWidget(sep2)

        # Gamma
        self.gamma = LabeledSpin(
            "γ — Overall spectral envelope broadening", 2.0, 0.0, 500, 1, "meV",
            hint=(
                "<b>γ controls the overall lineshape of the full optical spectrum</b>, "
                "including the ZPL. Applied as e<sup>−γ|t|</sup> in the time domain "
                "(Lorentzian convolution). It sets the ZPL linewidth and the width of "
                "the entire spectral envelope — reflecting homogeneous dephasing, "
                "lifetime broadening, or instrumental resolution."
            )
        )
        self.gamma.findChild(QLabel, "field_label")  # for completeness
        lay.addWidget(self.gamma)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setStyleSheet("color:#313244;"); lay.addWidget(sep3)

        # Temperature
        self.temp = LabeledSpin(
            "Temperature  T", 0.0, 0.0, 5000.0, 1, "K",
            hint=(
                "At T = 0 K only the vibrational ground state is occupied (default). "
                "For T &gt; 0 K, the Bose–Einstein occupation "
                "n<sub>k</sub>(T) = 1/(e<sup>ℏω<sub>k</sub>/k<sub>B</sub>T</sup>−1) "
                "is included in the generating function. "
                "Applies to standard Huang–Rhys theory only (not the squeezed model)."
            )
        )
        lay.addWidget(self.temp)

        sep4 = QFrame(); sep4.setFrameShape(QFrame.Shape.HLine)
        sep4.setStyleSheet("color:#313244;"); lay.addWidget(sep4)

        # Subtract modes
        sub_row = QHBoxLayout()
        sub_lbl = QLabel("Subtract low-frequency modes")
        sub_lbl.setObjectName("field_label")
        sub_row.addWidget(sub_lbl)
        self.subtract = QSpinBox()
        self.subtract.setRange(0, 100)
        self.subtract.setValue(0)
        self.subtract.setFixedWidth(90)
        sub_row.addWidget(self.subtract)
        sub_row.addStretch()
        lay.addLayout(sub_row)

        sub_hint = QLabel(
            "Removes the N lowest-frequency modes before computing S<sub>k</sub>.<br>"
            "• <b>0</b> — no removal (default, recommended for periodic solids whose "
            "acoustic modes are already near zero at Γ)<br>"
            "• <b>3</b> — remove acoustic/translational modes in periodic supercells "
            "(3 modes with ω → 0 at Γ)<br>"
            "• <b>5</b> — remove translational + rotational in <i>linear</i> molecules "
            "(3 trans. + 2 rot.)<br>"
            "• <b>6</b> — remove translational + rotational in <i>non-linear</i> molecules "
            "(3 trans. + 3 rot.) — standard for cluster/molecular codes"
        )
        sub_hint.setWordWrap(True)
        sub_hint.setObjectName("hint_label")
        sub_hint.setTextFormat(Qt.TextFormat.RichText)
        sub_hint.setStyleSheet("color: #6c7086; font-size: 13px;")
        lay.addWidget(sub_hint)

        lay.addStretch()

    def lorentzian(self) -> bool:
        return self._rb_lor.isChecked()

    def values(self) -> dict:
        return {
            "zpl": self.zpl.value(),
            "sigma_init": self.sigma1.value(),
            "sigma_final": self.sigma2.value(),
            "gamma": self.gamma.value(),
            "temperature": self.temp.value(),
            "subtract_modes": self.subtract.value(),
            "sidebands_broadening_lorentzian": self.lorentzian(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Advanced options
# ─────────────────────────────────────────────────────────────────────────────
class Step4Advanced(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(18)

        # Monte Carlo
        mc_frame = QFrame(); mc_frame.setObjectName("card")
        mc_lay = QVBoxLayout(mc_frame)
        self.mc_check = QCheckBox("  Monte Carlo Sampling")
        self.mc_check.setFont(QFont("Helvetica Neue", 13, QFont.Weight.Bold))
        self.mc_check.setChecked(True)
        mc_lay.addWidget(self.mc_check)
        mc_body = QLabel(
            "Samples phonon numbers from a Poisson distribution with mean Sₖ per mode, "
            "giving the emission spectrum directly as a histogram of photon energies. "
            "This is <b>numerically stable for any Huang–Rhys factor</b>, including the "
            "large-S limit where the generating function approach becomes unreliable. "
            "Always recommended — it also provides statistical moments (mean, std, skewness, kurtosis)."
        )
        mc_body.setWordWrap(True)
        mc_body.setObjectName("hint_label")
        mc_body.setTextFormat(Qt.TextFormat.RichText)
        mc_lay.addWidget(mc_body)
        lay.addWidget(mc_frame)

        # Squeezing
        sq_frame = QFrame(); sq_frame.setObjectName("card")
        sq_lay = QVBoxLayout(sq_frame)
        self.sq_check = QCheckBox("  Displaced–Squeezed Oscillator Model")
        self.sq_check.setFont(QFont("Helvetica Neue", 13, QFont.Weight.Bold))
        self.sq_check.setChecked(False)
        self.sq_check.toggled.connect(self._toggle_sq)
        sq_lay.addWidget(self.sq_check)
        sq_body = QLabel(
            "Goes beyond standard Huang–Rhys theory by accounting for changes in "
            "curvature (frequency) of the PES between GS and ES, not just the displacement. "
            "Requires excited-state phonons (provided in Step 2). The squeezing parameter "
            "r<sub>k</sub> = ½ ln(ω<sub>ES,k</sub>/ω<sub>GS,k</sub>) captures the mode-frequency ratio."
        )
        sq_body.setWordWrap(True)
        sq_body.setObjectName("hint_label")
        sq_body.setTextFormat(Qt.TextFormat.RichText)
        sq_lay.addWidget(sq_body)

        self._sq_params = QWidget()
        sp_lay = QHBoxLayout(self._sq_params)
        sp_lay.setContentsMargins(0, 8, 0, 0)
        sp_lay.setSpacing(20)
        self.sigma_sq = LabeledSpin("σ_squeezed", 1.0, 0.1, 500, 1, "meV")
        self.gamma_sq = LabeledSpin("γ_squeezed", 2.0, 0.0, 500, 1, "meV")
        sp_lay.addWidget(self.sigma_sq)
        sp_lay.addWidget(self.gamma_sq)
        sp_lay.addStretch()
        self._sq_params.setVisible(False)
        sq_lay.addWidget(self._sq_params)

        self._sq_na_hint = QLabel(
            "Not available for the Vertical Gradient Approximation — only one set of "
            "normal modes is provided, so ω<sub>GS</sub> = ω<sub>ES</sub> and the "
            "squeezing parameter r<sub>k</sub> = ½ ln(ω<sub>ES,k</sub>/ω<sub>GS,k</sub>) "
            "= 0 identically. Use the Adiabatic or External Vibrational Data workflow "
            "with separate ES frequencies to enable this option."
        )
        self._sq_na_hint.setWordWrap(True)
        self._sq_na_hint.setTextFormat(Qt.TextFormat.RichText)
        self._sq_na_hint.setStyleSheet(
            "color: #f9e2af; font-size: 12px; padding: 4px 0 2px 0;"
        )
        self._sq_na_hint.setVisible(False)
        sq_lay.addWidget(self._sq_na_hint)

        lay.addWidget(sq_frame)

        # HDF5 save
        hdf_frame = QFrame(); hdf_frame.setObjectName("card")
        hdf_lay = QVBoxLayout(hdf_frame)
        self.hdf5_check = QCheckBox("  Save results as HDF5")
        self.hdf5_check.setFont(QFont("Helvetica Neue", 13, QFont.Weight.Bold))
        self.hdf5_check.setChecked(True)
        self.hdf5_check.toggled.connect(self._toggle_hdf5)
        hdf_lay.addWidget(self.hdf5_check)
        hdf_body = QLabel(
            "Saves all computed arrays (spectra, Sk, modes, etc.) to a structured HDF5 file "
            "for later analysis, plotting, or sharing."
        )
        hdf_body.setWordWrap(True)
        hdf_body.setObjectName("hint_label")
        hdf_lay.addWidget(hdf_body)

        self._hdf5_row = QWidget()
        hr_lay = QHBoxLayout(self._hdf5_row)
        hr_lay.setContentsMargins(0, 6, 0, 0)
        hr_lay.setSpacing(8)
        hr_lay.addWidget(QLabel("Save to:"))
        self._hdf5_path = QLineEdit("spectra_output.h5")
        hr_lay.addWidget(self._hdf5_path)
        browse_h = QPushButton("Browse")
        browse_h.setObjectName("browse_btn")
        browse_h.setFixedWidth(80)
        browse_h.clicked.connect(self._browse_hdf5)
        hr_lay.addWidget(browse_h)
        hdf_lay.addWidget(self._hdf5_row)
        lay.addWidget(hdf_frame)

        # Experimental data
        exp_frame = QFrame(); exp_frame.setObjectName("card")
        exp_lay = QVBoxLayout(exp_frame)
        exp_title = QLabel("  Experimental Data  (optional)")
        exp_title.setFont(QFont("Helvetica Neue", 13, QFont.Weight.Bold))
        exp_lay.addWidget(exp_title)
        exp_body = QLabel(
            "Load measured spectra to overlay on computed results. "
            "Provide a file containing two columns/rows: Energy and Intensity "
            "(2×N or N×2). Accepted formats: <b>.txt  .dat  .npy  .npz</b> or no extension."
        )
        exp_body.setWordWrap(True)
        exp_body.setObjectName("hint_label")
        exp_body.setTextFormat(Qt.TextFormat.RichText)
        exp_lay.addWidget(exp_body)

        _EXP_FILTER = "Data files (*.txt *.dat *.npy *.npz);;All files (*)"

        def _exp_row(label):
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 4, 0, 0)
            rl.setSpacing(8)
            rl.addWidget(QLabel(label))
            le = QLineEdit()
            le.setPlaceholderText("optional — leave blank to skip")
            rl.addWidget(le, 1)
            btn = QPushButton("Browse")
            btn.setObjectName("browse_btn")
            btn.setFixedWidth(80)
            btn.clicked.connect(lambda _, _le=le: self._browse_exp(_le, _EXP_FILTER))
            rl.addWidget(btn)
            return row, le

        em_row,  self._exp_em_path  = _exp_row("Emission:")
        abs_row, self._exp_abs_path = _exp_row("Absorption:")
        exp_lay.addWidget(em_row)
        exp_lay.addWidget(abs_row)
        lay.addWidget(exp_frame)

        lay.addStretch()

    def set_workflow(self, wf: str):
        """Disable/enable squeezing based on the selected workflow."""
        is_gradient = (wf == "gradient")
        self.sq_check.setEnabled(not is_gradient)
        if is_gradient:
            self.sq_check.setChecked(False)   # also hides _sq_params via toggled
        self._sq_na_hint.setVisible(is_gradient)

    def _toggle_sq(self, v):
        self._sq_params.setVisible(v)

    def _toggle_hdf5(self, v):
        self._hdf5_row.setVisible(v)

    def _browse_hdf5(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HDF5", "spectra_output.h5", "HDF5 (*.h5 *.hdf5)",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if path:
            self._hdf5_path.setText(path)

    def _browse_exp(self, line_edit, ffilter):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open experimental data", "", ffilter,
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if path:
            line_edit.setText(path)

    def values(self) -> dict:
        return {
            "monte_carlo_emission":  self.mc_check.isChecked(),
            "enable_squeezing":      self.sq_check.isChecked(),
            "sigma_squeezed":        self.sigma_sq.value() if self.sq_check.isChecked() else None,
            "gamma_squeezed":        self.gamma_sq.value() if self.sq_check.isChecked() else None,
            "save_hdf5":             self.hdf5_check.isChecked(),
            "hdf5_path":             self._hdf5_path.text() if self.hdf5_check.isChecked() else None,
            "exp_emission_path":     self._exp_em_path.text()  or None,
            "exp_absorption_path":   self._exp_abs_path.text() or None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Inputs Page (wizard container)
# ─────────────────────────────────────────────────────────────────────────────
class InputsPage(QWidget):
    """
    Emits `run_requested` with a dict of all parameters when the user
    clicks 'Run Calculation' on the last step.
    """
    run_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._step = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 24, 32, 24)
        outer.setSpacing(16)

        # Header
        title = QLabel("Configure Calculation")
        title.setObjectName("section_title")
        title.setFont(QFont("Helvetica Neue", 16, QFont.Weight.Bold))
        outer.addWidget(title)

        # Step indicator
        self._indicator = StepIndicator()
        outer.addWidget(self._indicator)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#313244; margin:4px 0;")
        outer.addWidget(sep)

        # Scrollable step content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._content = QStackedWidget()

        self._step1 = Step1Workflow()
        self._step2 = Step2Files()
        self._step3 = Step3Parameters()
        self._step4 = Step4Advanced()

        self._content.addWidget(self._step1)
        self._content.addWidget(self._step2)
        self._content.addWidget(self._step3)
        self._content.addWidget(self._step4)

        scroll.setWidget(self._content)
        outer.addWidget(scroll, 1)

        # Navigation buttons
        nav = QHBoxLayout()
        self._back_btn = QPushButton("← Back")
        self._back_btn.setObjectName("secondary_btn")
        self._back_btn.setFixedWidth(110)
        self._back_btn.clicked.connect(self._go_back)
        self._back_btn.setEnabled(False)

        self._next_btn = QPushButton("Next →")
        self._next_btn.setObjectName("primary_btn")
        self._next_btn.setFixedWidth(180)
        self._next_btn.clicked.connect(self._go_next)

        nav.addWidget(self._back_btn)
        nav.addStretch()
        nav.addWidget(self._next_btn)
        outer.addLayout(nav)

        self._update_ui()

    def _update_ui(self):
        self._content.setCurrentIndex(self._step)
        self._indicator.set_step(self._step)
        self._back_btn.setEnabled(self._step > 0)
        is_last = self._step == 3
        self._next_btn.setText("▶  Run Calculation" if is_last else "Next →")
        # propagate workflow choice to step 2 (file pickers) and step 4 (squeezing lock)
        if self._step == 1:
            self._step2.set_workflow(self._step1.get_workflow())
        if self._step == 3:
            self._step4.set_workflow(self._step1.get_workflow())

    def _go_back(self):
        if self._step > 0:
            self._step -= 1
            self._update_ui()

    def _go_next(self):
        if self._step < 3:
            self._step += 1
            self._update_ui()
        else:
            self._emit_run()

    def _emit_run(self):
        wf = self._step1.get_workflow()
        paths = self._step2.get_paths()
        params = self._step3.values()
        advanced = self._step4.values()
        freq_unit = self._step2.get_freq_unit()

        if wf == "vibrational":
            qk_type = self._step2.get_vib_approx()
        else:
            qk_type = {"adiabatic": "r", "gradient": "f"}[wf]

        config = {
            "workflow": wf,
            "qk_calculation_type": qk_type,
            "vibrational_freqs_unit": freq_unit,
            **paths,
            **params,
            **advanced,
        }
        self.run_requested.emit(config)

    def reset(self):
        self._step = 0
        self._update_ui()

    def reset_to_step2(self):
        """Return to the Input Files step keeping the current workflow selection."""
        self._step = 1
        self._update_ui()
