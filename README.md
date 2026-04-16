# PLUMIPY

Vibronic spectra toolkit based on Huang–Rhys theory.

## Installation

```bash
git clone https://github.com/m-singhal/plumipy.git
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

# 📦 Outputs and Data Structure

PLUMIPY returns all computed quantities as a structured dictionary. When exporting to **HDF5**, this structure is preserved and can be accessed directly for post-processing.

---

## ⚙️ Output Availability (Important)

Output availability depends on the selected inputs and calculation flags:

* **Structures (GS + ES)** → Geometry + adiabatic PES
* **Forces** → Vertical gradient approximation
* **Phonons / Vibrations (GS)** → Mode-resolved quantities
* **Phonons / Vibrations (ES)** → Required for squeezing
* **ZPL provided** → Enables spectra (standard HR)
* **Enable Squeezing = True** → Enables squeezed outputs (requires GS + ES phonons)

---

## 🔬 Core Quantity

* `data["hbar"]`
    * Reduced Planck constant
    * **Units:** $\sqrt{meV \cdot amu} \cdot \text{Å}$

---

## 🧱 Geometry & Structure

* `data["R_gs"]` → (N_atoms, 3), Å
* `data["R_es"]` → (N_atoms, 3), Å
* `data["atoms"]` → Atomic species

---

## ⚡ Forces (if provided)

* `data["F_gs"]` → (N_atoms, 3), eV/Å
* `data["F_es"]` → (N_atoms, 3), eV/Å

---

## ⚖️ Atomic Masses

* `data["masses"]` → (N_atoms,), amu

---

## 🎵 Phonons (Ground State)

* `data["freqs_gs"]` → (N_modes,), **THz or cm⁻¹**  
* `data["modes_gs"]` → (N_modes, N_atoms, 3), **dimensionless**  
* `data["Ek_gs"]` → (N_modes,), **meV**  
* `data["wk_gs"]` → (N_modes,), **√(meV/amu)/Å**  
* `data["IPR_gs"]` → (N_modes,), **dimensionless**  

---

## 🎵 Phonons (Excited State)

* `data["freqs_es"]` → (N_modes,), **THz or cm⁻¹**  
* `data["modes_es"]` → (N_modes, N_atoms, 3), **dimensionless**  
* `data["Ek_es"]` → (N_modes,), **meV**  
* `data["wk_es"]` → (N_modes,), **√(meV/amu)/Å**  
* `data["IPR_es"]` → (N_modes,), **dimensionless**  

> [!NOTE]
> If ES phonons are not provided, ground-state values are reused by default.

---

## 🔗 Electron–Phonon Coupling

* `data["qk"]` → (N_modes,), **√(amu)·Å**  
* `data["Sk"]` → (N_modes,), **dimensionless**  
* `data["HR"]` → scalar, **dimensionless**  

---

## 🌈 Standard Huang–Rhys Spectra
*(Available when ZPL is provided)*

```python
std = data["standard_hr"]

S_E = std["S_E"]              # (N_E,), 1/meV
E_ph = std["E_phonons"]      # (N_E,), meV

G_t = std["G_t"]             # (N_t,), dimensionless
t = std["t_fs"]              # (N_t,), fs

E_em = std["E_photon_emission"]   # (N_photon,), meV
I_em = std["I_emission"]          # arbitrary units

E_abs = std["E_photon_absorption"]  # (N_photon,), meV
I_abs = std["I_absorption"]         # arbitrary units
```
---
## 🌀 Displaced–Squeezed Model  
*(Enable Squeezing = True + ES phonons required)*

```python
sq = data["squeezed"]

rk = sq["rk"]                     # (N_modes,), dimensionless

G_t_em = sq["G_t_emission"]      # (N_t,)
G_t_abs = sq["G_t_absorption"]   # (N_t,)
t = sq["t_fs"]                   # (N_t,), fs

E_em = sq["E_photon_emission"]   # (N_photon,), meV
E_abs = sq["E_photon_absorption"]# (N_photon,), meV

I_em = sq["I_emission"]          # arbitrary units
I_abs = sq["I_absorption"]       # arbitrary units

nk_em = sq["nk_mean_emission"]   # (N_E,)
nk_abs = sq["nk_mean_absorption"]# (N_E,)

E_ph = sq["E_phonons"]           # (N_E,), meV

S_E_em = sq["S_E_emission"]      # (N_E,), 1/meV
S_E_abs = sq["S_E_absorption"]   # (N_E,), 1/meV
```
---

## 💾 Using HDF5 Output (for analysis)

When exporting results, the HDF5 file preserves the internal dictionary structure, making it easy to reload for custom post-processing.

### 📥 Step 1: Load file
```python
data = load_hdf5_results("spectra_output.h5")
```

### 📊 Step 2: Access data
```python
Sk = data["Sk"]
Ek = data["Ek_gs"]

std = data["standard_hr"]
I_em = std["I_emission"]
```

### 🌀 Step 3: Access squeezed (if enabled)
```python
if data["squeezed"] is not None:
    sq = data["squeezed"]
    I_sq = sq["I_emission"]
```
---

### 📌 Notes
    - Structure mirrors the results dictionary  
    - Not all keys are always present  
    - Energies are in **meV**  
    - Arrays are NumPy-compatible  

    👉 This enables full flexibility for custom analysis and plotting.


