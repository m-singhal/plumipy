"""
PLUMIPY: Vibronic spectra toolkit based on Huang–Rhys theory.

This package provides tools to compute photoluminescence and absorption
spectra from first-principles inputs, including support for:

- Standard Huang–Rhys theory
- Displaced–squeezed oscillator model
- VASP / Phonopy outputs
- HDF5-based workflows

Main entry points:
    - calculate_spectra_analytical
    - load_hdf5_results
"""

from .api import calculate_spectra_analytical
from .io import load_hdf5_results
from .photoluminescence import Photoluminescence

__all__ = [
    "calculate_spectra_analytical",
    "load_hdf5_results",
    "Photoluminescence",
]

__version__ = "0.1.0"