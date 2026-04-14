# PLUMIPY

Vibronic spectra toolkit based on Huang–Rhys theory.

## Installation

```bash
git clone https://github.com/<your-username>/plumipy.git
cd plumipy
python -m venv plumipy-env
source venv/bin/activate   # mac/linux
pip install -r requirements.txt
```

## Running the app

```bash
cd plumipy
streamlit run app.py
```

## Overview

**PLUMIPY** is a tool for calculating photoluminescence and absorption spectra within the harmonic approximation from first-principles based on the generating function approach.

### Features

- For solids and molecules  
- Supports standard Huang–Rhys theory  
- Beyond Huang–Rhys: Independent-mode displaced–squeezed harmonic oscillator approximation  
- Integrates seamlessly with:
  - VASP (CONTCAR, OUTCAR)  
  - Phonopy (band.yaml)  
- Compatible with other DFT codes via external parsers and NumPy  
- Temperature-dependent spectra  
- Force-based (vertical gradient) and structure-based (adiabatic PES) methods  
- Interactive GUI for visualization  
- Option to save results in HDF5 format  

---

## What this tool does

### Spectra and spectral functions
- Photoluminescence (emission) and absorption spectra  
- Spectral function S(E) and generating functions  
- Energy-resolved phonon contributions  

### Electron–phonon coupling analysis
- Mode-resolved Huang–Rhys factors S_k  
- Total Huang–Rhys factor  
- Mass-weighted displacements in normal mode basis  

### Phonon and vibrational properties
- Phonon frequencies, normal modes, and mode energies  
- Inverse participation ratio (IPR)  
- Ground and excited state vibrational information  

### Advanced physical models
- Standard Huang–Rhys theory  
- Temperature-dependent spectra  
- Displaced–squeezed oscillator model  

### Flexible input support
- Structures (POSCAR/CONTCAR, numpy/text formats)  
- Forces or geometries  
- Phonons (Phonopy band.yaml, VASP OUTCAR)  

### Data export
- All computed quantities saved in HDF5 format  
- Suitable for post-processing and analysis  

---

## How to use

PLUMIPY supports multiple workflows depending on available inputs:

### 1. Adiabatic PES approach (structure-based)
- Upload ground and excited state structures  
- Provide phonon data (VASP / Phonopy)  

### 2. Vertical Gradient approximation (force-based)
- Upload ground state structure  
- Provide forces (ground and excited state)  
- Provide phonon data  

### 3. Vibrational inputs (alternative route)
- Provide frequencies, modes, and masses directly  
- Useful for non-VASP or custom workflows  

### 4. Set physical parameters
- Zero-phonon line (ZPL)  
- Broadening parameters (σ, γ)  
- Temperature and optional settings  

---
