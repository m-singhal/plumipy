import traceback
import numpy as np
import h5py
import io

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QStackedWidget, QFrame,
    QProgressBar, QMessageBox, QFileDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont

from app.pages.home    import HomePage
from app.pages.inputs  import InputsPage
from app.pages.results import ResultsPage
from app.pages.compare import ComparePage

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plumipy import calculate_spectra_analytical


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread  (keeps GUI responsive during calculation)
# ─────────────────────────────────────────────────────────────────────────────
class CalcWorker(QObject):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self):
        try:
            cfg = self.config
            wf  = cfg.get("workflow", "adiabatic")

            kwargs = dict(
                structure_gs            = cfg.get("structure_gs") or None,
                structure_es            = cfg.get("structure_es") or None,
                forces_gs               = cfg.get("forces_gs") or None,
                forces_es               = cfg.get("forces_es") or None,
                phonons_gs              = cfg.get("phonons_gs") or None,
                phonons_es              = cfg.get("phonons_es") or None,
                vibrational_freqs_gs    = cfg.get("vib_freqs_gs") or None,
                vibrational_freqs_es    = cfg.get("vib_freqs_es") or None,
                vibrational_modes_gs    = cfg.get("vib_modes_gs") or None,
                vibrational_modes_es    = cfg.get("vib_modes_es") or None,
                masses                  = cfg.get("masses") or None,
                qk_calculation_type     = cfg.get("qk_calculation_type"),
                zpl                     = cfg.get("zpl", 1000.0),
                sigma_init              = cfg.get("sigma_init", 3.0),
                sigma_final             = cfg.get("sigma_final", 3.0),
                gamma                   = cfg.get("gamma", 2.0),
                sidebands_broadening_lorentzian = cfg.get("sidebands_broadening_lorentzian", False),
                vibrational_freqs_unit  = cfg.get("vibrational_freqs_unit", "cm^-1"),
                subtract_modes          = int(cfg.get("subtract_modes", 0)),
                temperature             = cfg.get("temperature", 0.0),
                enable_squeezing        = cfg.get("enable_squeezing", False),
                sigma_squeezed          = cfg.get("sigma_squeezed"),
                gamma_squeezed          = cfg.get("gamma_squeezed"),
                monte_carlo_emission    = cfg.get("monte_carlo_emission", True),
                save_to_hdf5            = False,   # handled separately below
            )

            results = calculate_spectra_analytical(**kwargs)

            # Save HDF5 if requested
            if cfg.get("save_hdf5") and cfg.get("hdf5_path"):
                _save_hdf5(results, cfg["hdf5_path"])

            self.finished.emit(results)

        except Exception:
            self.error.emit(traceback.format_exc())


def _load_exp_data(path: str):
    """Load (energy, intensity) from a 2×N or N×2 file. Returns (E, I) or raises."""
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        npz = np.load(path)
        arr = npz[list(npz.keys())[0]]
    else:
        arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {arr.shape}")
    # 2×N: more columns than rows and exactly 2 rows
    if arr.shape[0] == 2 and arr.shape[1] > 2:
        return arr[0], arr[1]
    # N×2: 2 columns (handles N=2 as well, defaulting to column-wise)
    if arr.shape[1] == 2:
        return arr[:, 0], arr[:, 1]
    # Fallback: 2×N when shape is exactly (2, 2)
    if arr.shape[0] == 2:
        return arr[0], arr[1]
    raise ValueError(f"Cannot parse energy/intensity from shape {arr.shape}")


def _save_hdf5(results: dict, path: str):
    def _write(group, d):
        for k, v in d.items():
            if isinstance(v, dict):
                _write(group.create_group(k), v)
            elif v is None:
                pass
            else:
                try:
                    group.create_dataset(k, data=np.array(v))
                except Exception:
                    pass
    with h5py.File(path, "w") as f:
        _write(f, results)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar nav button
# ─────────────────────────────────────────────────────────────────────────────
class NavButton(QPushButton):
    def __init__(self, icon, label, parent=None):
        super().__init__(f"  {icon}  {label}", parent)
        self.setObjectName("nav_btn")
        self.setCheckable(False)
        self._active = False

    def set_active(self, v: bool):
        self._active = v
        self.setProperty("active", "true" if v else "false")
        self.style().unpolish(self)
        self.style().polish(self)


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PLUMIPY — Vibronic Spectra Toolkit")
        self.resize(1400, 860)
        self._results          = None
        self._thread           = None
        self._worker           = None
        self._comparison_store = []   # list of {"label", "results", "config"}

        self._build_ui()
        self._go_to("home")

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._make_sidebar())
        root.addWidget(self._make_content(), 1)

    def _make_sidebar(self):
        sb = QWidget()
        sb.setObjectName("sidebar")
        sb.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(sb)
        lay.setContentsMargins(0, 0, 0, 16)
        lay.setSpacing(2)

        title = QLabel("PLUMIPY")
        title.setObjectName("sidebar_title")
        lay.addWidget(title)

        sep = QFrame()
        sep.setObjectName("sidebar_sep")
        sep.setFrameShape(QFrame.Shape.HLine)
        lay.addWidget(sep)
        lay.addSpacing(4)

        self._nav_btns = {}
        for key, icon, label in [
            ("home",    "🏠", "Home"),
            ("inputs",  "📁", "Inputs"),
            ("results", "📊", "Results"),
            ("compare", "⚖",  "Compare"),
        ]:
            btn = NavButton(icon, label)
            btn.clicked.connect(lambda _, k=key: self._go_to(k))
            self._nav_btns[key] = btn
            lay.addWidget(btn)

        lay.addSpacing(4)
        sep2 = QFrame()
        sep2.setObjectName("sidebar_sep")
        sep2.setFrameShape(QFrame.Shape.HLine)
        lay.addWidget(sep2)
        lay.addSpacing(4)

        self._run_btn = QPushButton("  ▶  Run")
        self._run_btn.setObjectName("run_btn")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run_clicked)
        lay.addWidget(self._run_btn)

        lay.addStretch()

        # Status
        status_sep = QFrame()
        status_sep.setObjectName("sidebar_sep")
        status_sep.setFrameShape(QFrame.Shape.HLine)
        lay.addWidget(status_sep)

        self._status_lbl = QLabel("Status")
        self._status_lbl.setObjectName("status_label")
        lay.addWidget(self._status_lbl)

        self._status_val = QLabel("● Ready")
        self._status_val.setObjectName("status_value")
        self._status_val.setStyleSheet("color: #a6e3a1; padding: 0 16px 14px 16px; font-size:11px;")
        lay.addWidget(self._status_val)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedHeight(4)
        self._progress.setVisible(False)
        self._progress.setStyleSheet("margin: 0 12px;")
        lay.addWidget(self._progress)

        return sb

    def _make_content(self):
        self._stack = QStackedWidget()
        self._stack.setObjectName("content_stack")

        self._home_page    = HomePage()
        self._inputs_page  = InputsPage()
        self._results_page = ResultsPage()
        self._compare_page = ComparePage()

        self._stack.addWidget(self._home_page)      # index 0
        self._stack.addWidget(self._inputs_page)    # index 1
        self._stack.addWidget(self._results_page)   # index 2
        self._stack.addWidget(self._compare_page)   # index 3

        # Navigation / workflow signals
        self._home_page.go_to_inputs.connect(lambda: self._go_to("inputs"))
        self._home_page.new_calculation.connect(self._new_calculation)
        self._inputs_page.run_requested.connect(self._start_run)
        self._inputs_page.run_requested.connect(lambda _: self._run_btn.setEnabled(False))

        # Comparison signals
        self._results_page.save_requested.connect(self._on_save_to_comparison)
        self._results_page.run_another_requested.connect(self._on_run_another)
        self._compare_page.remove_requested.connect(self._on_remove_comparison_entry)

        return self._stack

    # ── Navigation ────────────────────────────────────────────────────────────
    _PAGE_IDX = {"home": 0, "inputs": 1, "results": 2, "compare": 3}

    def _go_to(self, key: str):
        self._stack.setCurrentIndex(self._PAGE_IDX[key])
        for k, btn in self._nav_btns.items():
            btn.set_active(k == key)

    def _new_calculation(self):
        self._inputs_page.reset()
        self._go_to("inputs")

    # ── Comparison store management ───────────────────────────────────────────
    def _on_save_to_comparison(self, label: str, results, config):
        if len(self._comparison_store) >= 3:
            return
        self._comparison_store.append({"label": label, "results": results, "config": config or {}})
        self._sync_comparison()

    def _on_run_another(self):
        self._inputs_page.reset()   # returns to Step 1 — Workflow selection
        self._go_to("inputs")

    def _on_remove_comparison_entry(self, idx: int):
        if 0 <= idx < len(self._comparison_store):
            self._comparison_store.pop(idx)
            self._sync_comparison()

    def _sync_comparison(self):
        """Push the current store to the compare page and update the results bar."""
        n = len(self._comparison_store)
        self._compare_page.update_store(self._comparison_store)
        self._results_page.set_comparison_count(n)
        # Update Compare nav button label to show count when calcs are saved
        compare_label = f"⚖  Compare  ({n})" if n else "⚖  Compare"
        self._nav_btns["compare"].setText(f"  {compare_label}")

    # ── Run control ───────────────────────────────────────────────────────────
    def _on_run_clicked(self):
        """Run button in sidebar re-triggers the last config."""
        if self._last_config:
            self._start_run(self._last_config)

    _last_config: dict = {}

    def _start_run(self, config: dict):
        self._last_config = config
        self._run_btn.setEnabled(False)
        self._set_status("⏳ Running…", "#f9e2af")
        self._progress.setVisible(True)

        self._thread = QThread()
        self._worker = CalcWorker(config)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_done(self, results: dict):
        # Load experimental data files if provided
        for cfg_key, res_key, label in [
            ("exp_emission_path",    "exp_emission",    "emission"),
            ("exp_absorption_path",  "exp_absorption",  "absorption"),
        ]:
            path = (self._last_config or {}).get(cfg_key)
            if path:
                try:
                    E, I = _load_exp_data(path)
                    results[res_key] = {"E": E, "I": I}
                except Exception as exc:
                    dlg = QMessageBox(self)
                    dlg.setWindowTitle("Experimental data warning")
                    dlg.setIcon(QMessageBox.Icon.Warning)
                    dlg.setText(
                        f"Could not load experimental {label} file:\n{path}\n\n{exc}"
                    )
                    dlg.exec()

        self._results = results
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        HR = results.get("HR", None)
        status = f"● Done  |  S = {HR:.3f}" if HR is not None else "● Done"
        self._set_status(status, "#a6e3a1")

        self._results_page.show_results(results, self._last_config)
        self._go_to("results")

    def _on_error(self, tb: str):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._set_status("● Error", "#f38ba8")

        dlg = QMessageBox(self)
        dlg.setWindowTitle("Calculation failed")
        dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.setText("An error occurred during the calculation.")
        dlg.setDetailedText(tb)
        dlg.exec()

    def _set_status(self, text: str, color: str):
        self._status_val.setText(text)
        self._status_val.setStyleSheet(
            f"color: {color}; padding: 0 16px 14px 16px; font-size:11px;"
        )

    # Called by InputsPage after step 4 is configured, enable Run button
    def _enable_run(self):
        self._run_btn.setEnabled(True)
