import streamlit as st
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
import io
import h5py
from api import calculate_spectra_analytical

st.set_page_config(layout="wide", page_title="Plumipy")
st.title("PLUMIPY")

# =========================
# HELPER
# =========================
def save_file(uploaded):
    if uploaded is None:
        return None

    name = uploaded.name

    if "." in name:
        suffix = "." + name.split(".")[-1]
    else:
        suffix = ""   # ← IMPORTANT: no fake extension

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.close()
    return tmp.name

# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.title("Inputs")

# -------- STRUCTURES --------
with st.sidebar.expander("Structures", expanded=True):
    st.caption("Required for Adiabatic PES Approximation")
    structure_gs = st.file_uploader("Ground state structure")
    structure_es = st.file_uploader("Excited state structure")
    st.caption("Supported formats: .npy, .npz, .txt, .dat, POSCAR/CONTCAR (VASP)")

# -------- PHONONS --------
with st.sidebar.expander("Phonons", expanded=True):
    st.caption("Use this for periodic crystals like solids/molecular crystals")
    phonons_gs = st.file_uploader("Ground state phonons")
    phonons_es = st.file_uploader("Excited state phonons (optional: Use this with 'Enable squeezing')")
    st.caption("Supported formats: .yaml (Phonopy), OUTCAR (VASP)")

# -------- FORCES --------
with st.sidebar.expander("Forces (optional)"):
    st.caption("Only required for the 'Vertical Gradient Approximation'")
    forces_gs = st.file_uploader("Ground state forces")
    forces_es = st.file_uploader("Excited state forces")
    st.caption("Supported formats: .npy, .npz, .txt, .dat, OUTCAR (VASP)")

# -------- VIBRATIONAL --------
with st.sidebar.expander("Vibrational Inputs (optional)"):
    st.caption("Can be used for vibrational and phonon frequencies/normal modes in molecules/solids parsed externally from codes other than VASP")
    units = st.selectbox("Units", ["cm⁻¹", "THz"])
    unit_map = {
    "cm⁻¹": "cm^-1",
    "THz": "THz"
    }
    units = unit_map.get(units, None)
    vib_freqs_gs = st.file_uploader("Ground state frequencies")
    vib_freqs_es = st.file_uploader("Excited state frequencies")
    st.caption("1D array: (Nmodes,)")
    st.caption("Supported formats: .npy, .npz, .txt, .dat")
    vib_modes_gs = st.file_uploader("Ground state normal modes")
    vib_modes_es = st.file_uploader("Excited state normal modes")
    st.caption("3D array: (Nmodes, Natoms, 3)")
    st.caption("Supported formats: .npy, .npz, .txt, .dat")

# -------- MASSES --------
with st.sidebar.expander("Atomic masses (amu) (optional)"):
    st.caption("Required with 'Vibrational Inputs' tag as masses are not provided.")
    masses = st.file_uploader("Masses file")
    st.caption("1D array: (Natoms,)")
    st.caption("Supported formats: .npy, .npz, .txt, .dat")

# -------- PARAMETERS --------
with st.sidebar.expander("Parameters", expanded=True):

    zpl = st.number_input("ZPL (meV)", value=1000.0)
    st.caption("Zero phonon line/0-0 transition energy in molecules")
    broadening_type = ["Gaussian", "Lorentzian"]
    sidebands_broadening_lorentzian = st.selectbox("Type of broadening", broadening_type)
    if sidebands_broadening_lorentzian == broadening_type[0]:
        sidebands_broadening_lorentzian = False
    else: 
        sidebands_broadening_lorentzian = True
    sigma = st.number_input("Sigma (meV)", value=3.0)
    st.caption("Homogenous broadening of the sidebands")
    gamma = st.number_input("Gamma (meV)", value=2.0)
    st.caption("Homogenous broadening of the entire spectra")
    calculation_type = ["Adiabatic PES Approximation ('Structures tag')", "Vertical Gradient Approximation ('Forces tag')"]
    qk_type = st.selectbox("Calculation type", calculation_type)
    if qk_type == calculation_type[0]:
        qk_type = "r"
    else:
        qk_type = "f"

    subtract_modes = st.number_input("Subtract modes", value=0)
    st.caption("Remove low-frequency modes. For e.g. a value of 3 removes first 3 modes that can be translational modes (important for force-based calculations)")

    temperature = st.number_input("Temperature (K)", value=0.0)
    st.caption("Works only within the standard Huang-Rhys theory and not while 'Enable squeezing is switched on'")

    enable_squeezing = st.checkbox("Enable squeezing")
    st.caption("Check this option for Displaced-squeezed approximation (Excited state phonons/Vibrational Inputs are required)")

    sigma_sq = st.number_input("Sigma squeezed", value=3.0)
    gamma_sq = st.number_input("Gamma squeezed", value=2.0)
    st.caption("Broadening values for the squeezed theory (use with 'Enable squeezing')")

    save_hdf5 = st.checkbox("Save HDF5", value=True)
    st.caption("Save the outputs in HDF5 format")

run = st.sidebar.button("Run")


# =========================
# HOMEPAGE / INTRO
# =========================
if not run:

    st.markdown("## A Vibronic Spectra Toolkit")

    st.markdown(
    """
    **PLUMIPY** is a tool for calculating photoluminescence and absorption spectra within the harmonic approximation from first-principles based on the generating function approach.

    Features:
    - For solids and molecules
    - Supports standard Huang–Rhys theory
    - Going beyond the Huang-Rhys theory: The Independent mode displaced-squeezed harmonic oscillator approximation
    - Integrates seamlessly with VASP output files - CONTCAR, OUTCAR and Phonopy - band.yaml
    - Integrates to other Density Functional theory codes via external parsers and NumPy
    - Temperature-dependent spectra
    - Force-based (vertical gradient) and structure-based (adiabatic potential energy surface) methods
    - Provides a GUI to interact with the plots
    - Option to save the data in HDF5 format 

    ---
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### What this tool does")

        st.markdown(
        """
        **Spectra and spectral functions**
        - Photoluminescence (emission) and absorption spectra  
        - Spectral function \( S(E) \) and generating functions  
        - Energy-resolved phonon contributions  

        **Electron–phonon coupling analysis**
        - Mode-resolved Huang–Rhys factors \( S_k \)  
        - Total Huang–Rhys factor  
        - Mass-weighted displacements in normal mode basis  

        **Phonon and vibrational properties**
        - Phonon frequencies, normal modes, and mode energies  
        - Inverse participation ratio (IPR) for mode localization  
        - Ground and excited state vibrational information  

        **Advanced physical models**
        - Standard Huang–Rhys theory  
        - Temperature-dependent spectra  
        - Displaced–squeezed oscillator model (squeezing effects)  

        **Flexible input support**
        - Structures (POSCAR/CONTCAR, numpy/text formats)  
        - Forces or geometries for coupling calculations  
        - Phonons from Phonopy (band.yaml) or VASP (OUTCAR)  

        **Data export**
        - All computed quantities saved in structured HDF5 format  
        - Ready for post-processing and analysis  
        """
        )

    with col2:
        st.markdown("### How to use")

        st.markdown(
        """
        **PLUMIPY** supports multiple workflows depending on the available inputs:

        **1. Adiabatic PES approach (structure-based)**  
        - Upload ground and excited state structures  
        - Provide phonon data (VASP / Phonopy)  

        **2. Vertical Gradient approximation (force-based)**  
        - Upload ground state structure  
        - Provide forces (ground and excited state)
        - Provide phonon data (VASP / Phonopy)  

        **3. Vibrational inputs (alternative route)**  
        - Directly provide frequencies, modes, and masses  
        - Useful for non-VASP or custom workflows  

        **4. Set physical parameters**  
        - Zero-phonon line (ZPL)  
        - Broadening parameters (σ, γ)  
        - Temperature and optional settings
        """
        )

    st.markdown("---")

    st.info("👉 Start by selecting inputs from the sidebar and click **Run**.")

# =========================
# RUN
# =========================
if run:

    with st.spinner("Running..."):

        results = calculate_spectra_analytical(
            structure_gs=save_file(structure_gs),
            structure_es=save_file(structure_es),
            forces_gs=save_file(forces_gs),
            forces_es=save_file(forces_es),
            phonons_gs=save_file(phonons_gs),
            phonons_es=save_file(phonons_es),
            vibrational_freqs_gs=save_file(vib_freqs_gs),
            vibrational_freqs_es=save_file(vib_freqs_es),
            vibrational_modes_gs=save_file(vib_modes_gs),
            vibrational_modes_es=save_file(vib_modes_es),
            masses=save_file(masses),
            qk_calculation_type=qk_type,
            zpl=zpl,
            sigma=sigma,
            gamma=gamma,
            sidebands_broadening_lorentzian=sidebands_broadening_lorentzian,
            vibrational_freqs_unit=units,
            subtract_modes=subtract_modes,
            temperature=temperature,
            enable_squeezing=enable_squeezing,
            sigma_squeezed=sigma_sq,
            gamma_squeezed=gamma_sq,
            save_to_hdf5=False
        )

    st.success("Done")

    # =========================
    # CREATE HDF5 IN MEMORY
    # =========================
    hdf5_buffer = None

    if save_hdf5:
        hdf5_buffer = io.BytesIO()

        with h5py.File(hdf5_buffer, "w") as f:
            for key, value in results.items():
                try:
                    f.create_dataset(key, data=value)
                except Exception:
                    pass

        hdf5_buffer.seek(0)

    # =========================
    # STANDARD HR
    # =========================
    if "standard_hr" in results:

        std = results["standard_hr"]

        Ek_gs = results["Ek_gs"]
        Sk = results["Sk"]

        # Plot 1
        fig1 = go.Figure()

        fig1.add_trace(go.Scatter(
            x=std["E_phonons"],
            y=std["S_E"],
            line=dict(color="royalblue"),
            name="S(E) (meV⁻¹)"
        ))

        fig1.add_trace(go.Bar(
            x=Ek_gs,
            y=Sk,
            marker=dict(color="red", line=dict(width=0)),
            name="Sₖ",
            yaxis="y2",
            opacity=0.6
        ))

        fig1.update_layout(
            title=f"Spectral Function | HR = {results['HR']: .3f}",
            
            xaxis=dict(
                title="Phonon Energies (meV)"
            ),

            yaxis=dict(
                title="S(E)  (meV⁻¹)",
                color="royalblue"
            ),

            yaxis2=dict(
                title="Sₖ",
                overlaying="y",
                side="right",
                color="red"
            )
        )

        st.plotly_chart(fig1, use_container_width=True)

        # Plot 2: Emission
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=std["E_photon_emission"],
            y=np.real(std["I_emission"]),
            line=dict(color="royalblue"),
            name="Emission"
        ))

        fig2.update_layout(
            title="Emission Spectra",
            xaxis=dict(title="Photon Energy (meV)"),
            yaxis=dict(title="PL (arb. units)")
        )

        st.plotly_chart(fig2, use_container_width=True)


        # Plot 3: Absorption
        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(
            x=std["E_photon_absorption"],
            y=np.real(std["I_absorption"]),
            line=dict(color="green"),
            name="Absorption"
        ))

        fig3.update_layout(
            title="Absorption Spectra",
            xaxis=dict(title="Photon Energy (meV)"),
            yaxis=dict(title="Absorption (arb. units)")
        )

        st.plotly_chart(fig3, use_container_width=True)

    # =========================
    # SQUEEZED
    # =========================
    if "squeezed" in results and results["squeezed"] is not None:

        sq = results["squeezed"]
        Ek_gs = results["Ek_gs"]
        Ek_es = results["Ek_es"]

        # -------- Plot 4 --------
        fig4 = go.Figure()

        fig4.add_trace(go.Scatter(
            x=sq["E_phonons"],
            y=sq["S_E_emission"],
            line=dict(color="royalblue"),
            name="S(E) (meV⁻¹)"
        ))

        fig4.add_trace(go.Bar(
            x=Ek_gs,
            y=sq["nk_mean_emission"],
            marker=dict(color="red", line=dict(width=0)),
            name="⟨nₖ⟩",
            yaxis="y2",
            opacity=0.6
        ))

        fig4.update_layout(
            title="Squeezed Emission: Spectral Function and Average Phonons Emitted/mode",
            xaxis=dict(title="Phonon Energy (meV)"),
            yaxis=dict(title="S(E) (meV⁻¹)", color="royalblue"),
            yaxis2=dict(
                title="⟨nₖ⟩",
                overlaying="y",
                side="right",
                color="red"
            )
        )

        st.plotly_chart(fig4, use_container_width=True)

        # -------- Plot 5 --------
        fig5 = go.Figure()

        fig5.add_trace(go.Scatter(
            x=sq["E_photon_emission"],
            y=np.real(sq["I_emission"]),
            line=dict(color="royalblue"),
            name="Emission"
        ))

        fig5.update_layout(
            title="Squeezed Emission Spectra",
            xaxis=dict(title="Photon Energy (meV)"),
            yaxis=dict(title="PL (arb. units)")
        )

        st.plotly_chart(fig5, use_container_width=True)

        # -------- Plot 6 --------
        fig6 = go.Figure()

        fig6.add_trace(go.Scatter(
            x=sq["E_phonons"],
            y=sq["S_E_absorption"],
            line=dict(color="green"),
            name="S(E) (meV⁻¹)"
        ))

        fig6.add_trace(go.Bar(
            x=Ek_es,
            y=sq["nk_mean_absorption"],
            marker=dict(color="red", line=dict(width=0)),
            name="⟨nₖ⟩",
            yaxis="y2",
            opacity=0.6
        ))

        fig6.update_layout(
            title="Squeezed Absorption: Spectral Function and Average Phonons Emitted/mode",
            xaxis=dict(title="Phonon Energy (meV)"),
            yaxis=dict(title="S(E) (meV⁻¹)", color="green"),
            yaxis2=dict(
                title="⟨nₖ⟩",
                overlaying="y",
                side="right",
                color="red"
            )
        )

        st.plotly_chart(fig6, use_container_width=True)

        # -------- Plot 7 --------
        fig7 = go.Figure()

        fig7.add_trace(go.Scatter(
            x=sq["E_photon_absorption"],
            y=np.real(sq["I_absorption"]),
            line=dict(color="green"),
            name="Absorption"
        ))

        fig7.update_layout(
            title="Squeezed Absorption Spectra",
            xaxis=dict(title="Photon Energy (meV)"),
            yaxis=dict(title="PL (arb. units)")
        )

        st.plotly_chart(fig7, use_container_width=True)

    # =========================
    # DOWNLOAD
    # =========================
    if save_hdf5 and hdf5_buffer is not None:

        st.download_button(
            label="Download HDF5",
            data=hdf5_buffer,
            file_name="spectra_output.h5",
            mime="application/octet-stream"
        )