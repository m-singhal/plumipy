import numpy as np
from pathlib import Path
from .photoluminescence import Photoluminescence

def calculate_spectra_analytical(
        structure_gs: Path | np.ndarray | None, # Ground state structure, either as a file path or a numpy array. If a file path is provided, the function will determine the file type based on the extension and load it accordingly. Supported file types include .npy, .npz, .txt, .dat, and common structure file formats like POSCAR/CONTCAR.
        structure_es: Path | np.ndarray | None, # Excited state structure, either as a file path or a numpy array. If a file path is provided, the function will determine the file type based on the extension and load it accordingly. Supported file types include .npy, .npz, .txt, .dat, and common structure file formats like POSCAR/CONTCAR.
        forces_gs: Path | np.ndarray | None, # Ground state forces, either as a file path or a numpy array. If a file path is provided, the function will determine the file type based on the extension and load it accordingly. Supported file types include .npy, .npz, .txt, and .dat.
        forces_es: Path | np.ndarray | None, # Excited state forces, either as a file path or a numpy array. If a file path is provided, the function will determine the file type based on the extension and load it accordingly. Supported file types include .npy, .npz, .txt, and .dat.
        phonons_gs: Path | np.ndarray | None, # Ground state phonons, either as a file path or a numpy array. If a file path is provided, the function will determine the file type based on the extension and load it accordingly. Supported file types include .npy, .npz, .txt, and .dat.
        phonons_es: Path | np.ndarray | None, # Excited state phonons, either as a file path or a numpy array. If a file path is provided, the function will determine the file type based on the extension and load it accordingly. Supported file types include .npy, .npz, .txt, and .dat.
        vibrational_freqs_gs: Path | np.ndarray | None, # Ground state vibrational frequencies, either as a file path or a numpy array. If a file path is provided, the function will determine the file type based on the extension and load it accordingly. Supported file types include .npy, .npz, .txt, and .dat. 
        vibrational_freqs_es: Path | np.ndarray | None, # Excited state vibrational frequencies, either as a file path or a numpy array. If a file path is provided, the function will determine the file type based on the extension and load it accordingly. Supported file types include .npy, .npz, .txt, and .dat. 
        vibrational_modes_gs: Path | np.ndarray | None, # Ground state vibrational modes, either as a file path or a numpy array. If a file path is provided, the function will determine the file type based on the extension and load it accordingly. Supported file types include .npy, .npz, .txt, and .dat.
        vibrational_modes_es: Path | np.ndarray | None, # Excited state vibrational modes, either as a file path or a numpy array. If a file path is provided, the function will determine the file type based on the extension and load it accordingly. Supported file types include .npy, .npz, .txt, and .dat.
        masses: Path | np.ndarray | None, # Atomic masses, either as a file path or a numpy array. If a file path is provided, the function will determine the file type based on the extension and load it accordingly. Supported file types include .npy, .npz, .txt, and .dat. Units: atomic mass units (amu). Dimesion: (N_atoms,)
        qk_calculation_type: str | None, # Method to calculate the displacement between the ground state and excited state adiabatic potential energy surface minima. 'r' for coordinate-based calculation using the difference in geometries, and 'f' for force-based calculation using the difference in forces. This will determine how the Huang-Rhys factors are calculated, which in turn affects the resulting spectra.
        zpl: float | None, # Zero-phonon line energy in meV or 0-0 transition in molecules. This is the energy of the electronic transition without any phonon/vibrational modes involvement and serves as a reference point for the spectra.
        sigma_init: float | None, # Gaussian/Lorentzian broadening of phonon sidebands in meV. sigma_init and sigma_final will make the broadening linearly dependent on phonon energies.
        sigma_final: float | None, # Gaussian/Lorentzian broadening of phonon sidebands in meV. sigma_init and sigma_final will make the broadening linearly dependent on phonon energies. If None is given then sigma_init is used as homogenous broadening.
        gamma: float | None, # Lorentzian broadening of emission lines in meV.
        sidebands_broadening_lorentzian: bool = False, # Whether to use Lorentzian broadening for phonon sidebands/vibrational progressions instead of Gaussian.
        vibrational_freqs_unit: str = "cm^-1", # Unit of the vibrational frequencies provided by the user. Supported units include "cm^-1" and "THz". The function will convert the frequencies to a consistent unit (meV) for internal calculations. This parameter is only relevant if vibrational frequencies are provided. 
        temperature: float | None = 0.0, # Temperature dependence is only included for standard Huang-Rhys theory. Units: Kelvin 
        subtract_modes: int | None = 0, # Number of low-energy modes to subtract from the phonon/vibrational spectrum. This can be useful to remove translational and rotational modes in molecules or acoustic modes in solids that do not contribute to the optical spectra.
        enable_squeezing: bool = False, # Option to use the "Displaced-Squeezed oscillator" model going beyond Huang-Rhys theory. It gives converges to Huang-Rhys if phonon_gs == phonon_es or vibrational_freqs_gs == vibrational_freqs_es. 
        squeezing_parameter: np.ndarray | None = None, # Squeezing parameter for the more general theory that takes into account the displacement and curvature changes between the ground state and excited state potential energy surfaces. This can result in same spectra as the standard Huang-Rhys theory if the vibrational/phonon frequencies of the ground state and excited state are the same. Units: dimensionless, an array of shape (N_modes,). But usuallt this is not required as it will be calculated automatically. This is just for testing the changes in spectra with user defined squeezing parameters.
        sigma_squeezed: float | None = None, # Gaussian/Lorentzian broadening for the more general formalism that takes into account the displacement and curvature changes between the ground state and excited state potential energy surfaces. This can result in same spectra as the standard Huang-Rhys theory if the vibrational/phonon frequencies of the ground state and excited state are the same.
        gamma_squeezed: float | None = None, # Lorentzian broadening for the more general formalism that takes into account the displacement and curvature changes between the ground state and excited state potential energy surfaces. This can result in same spectra as the standard Huang-Rhys theory if the vibrational/phonon frequencies of the ground state and excited state are the same.
        save_to_hdf5: bool = False, # Whether to save the results to an HDF5 file for easier loading and analysis in the future. The file will be saved in the current working directory with a name based on the input parameters and a timestamp to ensure uniqueness.
        ):
    
    
    """
    OUTPUTS:
    A dictionary containing all results from the calculations. The presence of keys depends on the inputs provided.

    Core quantities:
    - "hbar": Reduced Planck's constant in units of sqrt(meV*AMU)*Angstrom.

    Geometry & structure:
    - "R_gs": Ground state geometry, shape (N_atoms, 3), units: Angstrom.
    - "R_es": Excited state geometry, shape (N_atoms, 3), units: Angstrom.
    - "atoms": Atomic species list (only present if structure is read from file formats like POSCAR/CONTCAR).

    Forces:
    - "F_gs": Ground state forces, shape (N_atoms, 3), units: eV/Angstrom.
    - "F_es": Excited state forces, shape (N_atoms, 3), units: eV/Angstrom.

    Masses:
    - "masses": Atomic masses, shape (N_atoms,), units: amu.

    Phonon / vibrational properties (ground state):
    - "freqs_gs": Frequencies, shape (N_modes,), units: THz (phonons) or cm^-1 (vibrations).
    - "modes_gs": Normal modes, shape (N_modes, N_atoms, 3), dimensionless.
    - "Ek_gs": Mode energies, shape (N_modes,), units: meV.
    - "wk_gs": Angular frequencies, shape (N_modes,), units: sqrt(meV/AMU)/Angstrom.
    - "IPR_gs": Inverse participation ratio, shape (N_modes,), dimensionless.

    Phonon / vibrational properties (excited state):
    - "freqs_es": Frequencies, shape (N_modes,), units: THz or cm^-1.
    - "modes_es": Normal modes, shape (N_modes, N_atoms, 3), dimensionless.
    - "Ek_es": Mode energies, shape (N_modes,), units: meV.
    - "wk_es": Angular frequencies, shape (N_modes,), units: sqrt(meV/AMU)/Angstrom.
    - "IPR_es": Inverse participation ratio, shape (N_modes,), dimensionless.

    NOTE:
    If excited-state phonons/vibrations are not provided, ground-state values are reused.

    Electron-phonon coupling (only if qk_calculation_type is provided):
    - "qk": Mass-weighted displacement in normal mode basis, shape (N_modes,), units: sqrt(AMU)*Angstrom.
    - "Sk": Partial Huang-Rhys factors, shape (N_modes,), dimensionless.
    - "HR": Total Huang-Rhys factor (sum of Sk), scalar.

    Standard Huang-Rhys spectra (only if zpl is provided):
    - "standard_hr": Dictionary containing:
        - "S_E": Spectral function, shape (N_phonon_energies,), units: 1/meV.
        - "E_phonons": Phonon energy grid, shape (N_phonon_energies,), units: meV.
        - "G_t": Generating function, shape (N_time_points,), dimensionless.
        - "t_fs": Time grid, shape (N_time_points,), units: fs.
        - "E_photon_emission": Emission photon energies, shape (N_photon_energies,), units: meV.
        - "A_E_emission": Emission spectral function, arbitrary units.
        - "I_emission": Emission intensity, arbitrary units.
        - "E_photon_absorption": Absorption photon energies, shape (N_photon_energies,), units: meV.
        - "A_E_absorption": Absorption spectral function, arbitrary units.
        - "I_absorption": Absorption intensity (normalized), arbitrary units.

    Squeezed (displaced-squeezed oscillator model):
    - "squeezed": Either None or a dictionary (if enable_squeezing=True) containing:
        - "rk": Squeezing parameters, shape (N_modes,), dimensionless.
        - "G_t_emission": Emission generating function, shape (N_time_points,).
        - "G_t_absorption": Absorption generating function, shape (N_time_points,).
        - "t_fs": Time grid, shape (N_time_points,), units: fs.
        - "E_photon_emission": Emission photon energies, shape (N_photon_energies,), units: meV.
        - "A_E_emission": Emission spectral function.
        - "I_emission": Emission intensity.
        - "E_photon_absorption": Absorption photon energies.
        - "A_E_absorption": Absorption spectral function.
        - "I_absorption": Absorption intensity.
        - "nk_mean_emission": Mean phonon occupation (emission), shape (N_phonon_energies,).
        - "nk_mean_absorption": Mean phonon occupation (absorption), shape (N_phonon_energies,).
        - "E_phonons": Phonon energy grid, shape (N_phonon_energies,), units: meV.
        - "S_E_emission": Emission spectral density, shape (N_phonon_energies,), units: 1/meV.
        - "S_E_absorption": Absorption spectral density, shape (N_phonon_energies,), units: 1/meV.

    File output:
    - If save_to_hdf5=True, all results are also saved to "spectra_output.h5" in hierarchical format.

    General notes:
    - Not all keys are guaranteed to exist; they depend on which inputs are provided.
    - Units are consistent internally (energies in meV).
    """


    '''
    Initializing a dictionary to store all the results
    '''
    results = {}
    results["hbar"] = 0.6582*np.sqrt(9.646) ## Units: sqrt(meV*AMU)*Angstrom

    '''
    Creating an instance of the Photoluminescence class
    '''
    pl = Photoluminescence()

    '''
    Loading the ground state and excited state geometries. The function can handle both numpy arrays and file paths. If a file path is provided, it will determine the file type based on the extension and load it accordingly.
    These geometries are loaded independently, allowing for flexibility in how the data is provided. The results are stored in the results dictionary for later use in the calculations or as an output for the user.
    '''
    if structure_gs is not None:
        if isinstance(structure_gs, np.ndarray):
            R_gs = structure_gs
        elif isinstance(structure_gs, str):
            extension = Path(structure_gs).suffix.lower()
            if extension in [".npy", ".npz"]:
                R_gs = np.load(structure_gs)
            elif extension in [".txt", ".dat"]:
                R_gs = np.loadtxt(structure_gs)
            else:
                R_gs, atoms = pl.ReadStructure(structure_gs)
                results["atoms"] = atoms
        else:
            raise ValueError("Unsupported type for structure_gs. Must be a numpy array or a file path.")
        results["R_gs"] = R_gs  ## Units: Angstrom, an array of shape (N_atoms, 3)
        
    if structure_es is not None:
        if isinstance(structure_es, np.ndarray):
            R_es = structure_es
        elif isinstance(structure_es, str):
            extension = Path(structure_es).suffix.lower()
            if extension in [".npy", ".npz"]:
                R_es = np.load(structure_es)
            elif extension in [".txt", ".dat"]:
                R_es = np.loadtxt(structure_es)
            else:
                R_es, atoms = pl.ReadStructure(structure_es)
                if "atoms" in results:
                    if not np.array_equal(results["atoms"], atoms):
                        raise ValueError("GS and ES atoms mismatch.")
                else:
                    results["atoms"] = atoms
        else:
            raise ValueError("Unsupported type for structure_es. Must be a numpy array or a file path.")
        results["R_es"] = R_es  ## Units: Angstrom, an array of shape (N_atoms, 3)

    
    
    '''
    Loading the ground state and excited state forces. The function can handle both numpy arrays and file paths. If a file path is provided, it will determine the file type based on the extension and load it accordingly.
    These forces are loaded independently, allowing for flexibility in how the data is provided. The results are stored in the results dictionary for later use in the calculations or as an output for the user.
    '''
    if forces_gs is not None:
        if isinstance(forces_gs, np.ndarray):
            F_gs = forces_gs
        elif isinstance(forces_gs, str):
            extension = Path(forces_gs).suffix.lower()
            if extension in [".npy", ".npz"]:
                F_gs = np.load(forces_gs)
            elif extension in [".txt", ".dat"]:
                F_gs = np.loadtxt(forces_gs)
            else:
                F_gs = pl.ReadForces(forces_gs)
        else:
            raise ValueError("Unsupported type for forces_gs. Must be a numpy array or a file path.")
        results["F_gs"] = F_gs  ## Units: eV/Angstrom, an array of shape (N_atoms, 3)

    if forces_es is not None:
        if isinstance(forces_es, np.ndarray):
            F_es = forces_es
        elif isinstance(forces_es, str):
            extension = Path(forces_es).suffix.lower()
            if extension in [".npy", ".npz"]:
                F_es = np.load(forces_es)
            elif extension in [".txt", ".dat"]:
                F_es = np.loadtxt(forces_es)
            else:
                F_es = pl.ReadForces(forces_es)
        else:
            raise ValueError("Unsupported type for forces_es. Must be a numpy array or a file path.")
        results["F_es"] = F_es  ## Units: eV/Angstrom, an array of shape (N_atoms, 3)


    '''
    Loading phonon frequencies and modes for both ground and excited states. This part of the code only works for band.yaml or OUTCAR files containing information about the Gamm point phonons.
    This will only work for VASP calculations.
    '''
    if phonons_gs is not None:
        if isinstance(phonons_gs, str):
            extension = Path(phonons_gs).suffix.lower()
            if extension in [".yaml"]:
                masses, freqs_gs, modes_gs = pl.ReadPhononsPhonopy(phonons_gs)
                freqs_gs = freqs_gs[:int(freqs_gs.shape[0]/2)]
                modes_gs = modes_gs[:int(modes_gs.shape[0]/2),...]
            else:
                if "atoms" not in results:
                    raise ValueError("Atoms information is required to load phonons from a non-YAML file. It can be produced only by using POSCAR/CONTCAR files to load structures.")
                else:
                    masses, freqs_gs, modes_gs = pl.ReadPhononsVasp(phonons_gs, results["atoms"])
            freqs_gs[freqs_gs <= 0] = 1e-6  ## Replace zero or negative frequencies with a small positive value to avoid issues in calculations
            results["masses"] = masses  ## Units: atomic mass units (amu), an array of shape (N_atoms,)
            results["freqs_gs"] = freqs_gs[subtract_modes:]  ## Units: THz
            results["modes_gs"] = modes_gs[subtract_modes:,...]  ## Units: dimensionless, an array of shape (N_modes, N_atoms, 3)
            results["Ek_gs"] = pl.FreqToEnergy(results["freqs_gs"])  ## Units: meV
            results["wk_gs"] = results["Ek_gs"]/results["hbar"]  ## Units: sqrt(meV/AMU)/Angstrom
            results["IPR_gs"] = pl.InverseParticipationRatio(results["modes_gs"])  ## Units: dimensionless, an array of shape (N_modes,)

    if phonons_es is not None:
        if isinstance(phonons_es, str):
            extension = Path(phonons_es).suffix.lower()
            if extension in [".yaml"]:
                masses, freqs_es, modes_es = pl.ReadPhononsPhonopy(phonons_es)
                freqs_es = freqs_es[:int(freqs_es.shape[0]/2)]
                modes_es = modes_es[:int(modes_es.shape[0]/2),...]
            else:
                if "atoms" not in results:
                    raise ValueError("Atoms information is required to load phonons from a non-YAML file. It can be produced only by using POSCAR/CONTCAR files to load structures.")
                else:
                    masses, freqs_es, modes_es = pl.ReadPhononsVasp(phonons_es, results["atoms"])
            freqs_es[freqs_es <= 0] = 1e-6  ## Replace zero or negative frequencies with a small positive value to avoid issues in calculations
            results["masses"] = masses  ## Units: atomic mass units (amu), an array of shape (N_atoms,)
            results["freqs_es"] = freqs_es[subtract_modes:]  ## Units: THz
            results["modes_es"] = modes_es[subtract_modes:,...]  ## Units: dimensionless, an array of shape (N_modes, N_atoms, 3)
            results["Ek_es"] = pl.FreqToEnergy(results["freqs_es"])  ## Units: meV
            results["wk_es"] = results["Ek_es"]/results["hbar"]  ## Units: sqrt(meV/AMU)/Angstrom
            results["IPR_es"] = pl.InverseParticipationRatio(results["modes_es"])  ## Units: dimensionless, an array of shape (N_modes,)
    else:
        if phonons_gs is not None:
            results["freqs_es"] = results["freqs_gs"]  ## Units: THz
            results["modes_es"] = results["modes_gs"]  ## Units: dimensionless, an array of shape (N_modes, N_atoms, 3)
            results["Ek_es"] = results["Ek_gs"]  ## Units: meV
            results["wk_es"] = results["wk_gs"]  ## Units: sqrt(meV/AMU)/Angstrom
            results["IPR_es"] = pl.InverseParticipationRatio(results["modes_es"])  ## Units: dimensionless, an array of shape (N_modes,)

    
    '''
    Loading vibrational frequencies and normal modes for both ground state and excited state. This can load numpy arrays saved from external calculations like Gaussian, ORCA, etc. 
    It can also be used for VASP, Quantum ESPRESSO, or any other DFT code if the user has a way to extract the vibrational frequencies and modes and save them in a compatible format.
    '''
    if vibrational_freqs_gs is not None and vibrational_modes_gs is not None:
        if isinstance(vibrational_freqs_gs, np.ndarray) and isinstance(vibrational_modes_gs, np.ndarray):
            freqs_vib_gs = vibrational_freqs_gs
            modes_vib_gs = vibrational_modes_gs
        elif isinstance(vibrational_freqs_gs, str) and isinstance(vibrational_modes_gs, str):
            extension_freqs = Path(vibrational_freqs_gs).suffix.lower()
            extension_modes = Path(vibrational_modes_gs).suffix.lower()
            if extension_freqs in [".npy", ".npz"] and extension_modes in [".npy", ".npz"]:
                freqs_vib_gs = np.load(vibrational_freqs_gs)
                modes_vib_gs = np.load(vibrational_modes_gs)
            elif extension_freqs in [".txt", ".dat"] and extension_modes in [".txt", ".dat"]:
                freqs_vib_gs = np.loadtxt(vibrational_freqs_gs)
                modes_vib_gs = np.loadtxt(vibrational_modes_gs)
            else:
                raise ValueError("Unsupported file types for vibrational frequencies or modes. Both must be either .npy/.npz or .txt/.dat.")
        else:
            raise ValueError("Unsupported types for vibrational frequencies or modes. Both must be either numpy arrays or file paths.")
        freqs_vib_gs = freqs_vib_gs[subtract_modes:]
        modes_vib_gs = modes_vib_gs[subtract_modes:,...]
        results["freqs_gs"] = freqs_vib_gs  ## Units: cm^-1
        results["modes_gs"] = modes_vib_gs  ## Units: dimensionless, an array of shape (N_vib_modes, N_atoms, 3)
        if vibrational_freqs_unit == "cm^-1":
            results["Ek_gs"] = 0.12398*freqs_vib_gs  ## Units: meV
        elif vibrational_freqs_unit == "THz":
            results["Ek_gs"] = pl.FreqToEnergy(freqs_vib_gs)  ## Units: meV
        results["wk_gs"] = results["Ek_gs"]/results["hbar"]  ## Units: sqrt(meV/AMU)/Angstrom
        results["IPR_gs"] = pl.InverseParticipationRatio(results["modes_gs"])  ## Units: dimensionless, an array of shape (N_modes,)

    if vibrational_freqs_es is not None and vibrational_modes_es is not None:
        if isinstance(vibrational_freqs_es, np.ndarray) and isinstance(vibrational_modes_es, np.ndarray):
            freqs_vib_es = vibrational_freqs_es
            modes_vib_es = vibrational_modes_es
        elif isinstance(vibrational_freqs_es, str) and isinstance(vibrational_modes_es, str):
            extension_freqs = Path(vibrational_freqs_es).suffix.lower()
            extension_modes = Path(vibrational_modes_es).suffix.lower()
            if extension_freqs in [".npy", ".npz"] and extension_modes in [".npy", ".npz"]:
                freqs_vib_es = np.load(vibrational_freqs_es)
                modes_vib_es = np.load(vibrational_modes_es)
            elif extension_freqs in [".txt", ".dat"] and extension_modes in [".txt", ".dat"]:
                freqs_vib_es = np.loadtxt(vibrational_freqs_es)
                modes_vib_es = np.loadtxt(vibrational_modes_es)
            else:
                raise ValueError("Unsupported file types for vibrational frequencies or modes. Both must be either .npy/.npz or .txt/.dat.")
        else:
            raise ValueError("Unsupported types for vibrational frequencies or modes. Both must be either numpy arrays or file paths.")
        results["freqs_es"] = freqs_vib_es  ## Units: cm^-1
        results["modes_es"] = modes_vib_es  ## Units: dimensionless, an array of shape (N_vib_modes, N_atoms, 3)
        if vibrational_freqs_unit == "cm^-1":
            results["Ek_es"] = 0.12398*freqs_vib_es  ## Units: meV
        elif vibrational_freqs_unit == "THz":
            results["Ek_es"] = pl.FreqToEnergy(freqs_vib_es)  ## Units: meV
        results["wk_es"] = results["Ek_es"]/results["hbar"]  ## Units: sqrt(meV/AMU)/Angstrom
        results["IPR_es"] = pl.InverseParticipationRatio(results["modes_es"])  ## Units: dimensionless, an array of shape (N_modes,)
    else:
        if vibrational_freqs_gs is not None and vibrational_modes_gs is not None:
            results["freqs_es"] = results["freqs_gs"]  ## Units: cm^-1
            results["modes_es"] = results["modes_gs"]  ## Units: dimensionless, an array of shape (N_vib_modes, N_atoms, 3)
            results["Ek_es"] = results["Ek_gs"]  ## Units: meV
            results["wk_es"] = results["wk_gs"]  ## Units: sqrt(meV/AMU)/Angstrom
            results["IPR_es"] = results["IPR_gs"]  ## Units: dimensionless, an array of shape (N_modes,)

    
    '''
    Loading atomic masses. This can be loaded from a file or extracted from the phonon files if they contain the atomic masses. 
    It is useful for molecules specially with Gaussian or ORCA calculations where the user might not have phonon information but still has the atomic masses.
    '''
    if masses is not None:
        if isinstance(masses, np.ndarray):
            masses_array = masses
        elif isinstance(masses, str):
            extension = Path(masses).suffix.lower()
            if extension in [".npy", ".npz"]:
                masses_array = np.load(masses)
            elif extension in [".txt", ".dat"]:
                masses_array = np.loadtxt(masses)
            else:
                raise ValueError("Unsupported file type for masses.")
        else:
            raise ValueError("Unsupported type for masses. Must be a numpy array or a file path.")
        results["masses"] = masses_array  ## Units: atomic mass units (amu), an array of shape (N_atoms,)


    '''
    Displacement between the ground state and excited state adiabatic potential energy surface minima with ground state normal modes as the basis. 
    This is a crucial step in calculating the Huang-Rhys factors and the resulting spectra.
    '''
    if qk_calculation_type is not None:
        has_structures = (structure_gs is not None) and (structure_es is not None)
        has_forces = (forces_gs is not None) and (forces_es is not None)

        if (structure_gs is None) ^ (structure_es is None):
            raise ValueError("Provide both structure_gs and structure_es.")

        if (forces_gs is None) ^ (forces_es is None):
            raise ValueError("Provide both forces_gs and forces_es.")

        if not (has_structures or has_forces):
            raise ValueError("Provide either structures or forces.")
        if "masses" not in results:
            raise ValueError("Masses are required for qk calculation.")
        if qk_calculation_type == 'r':
            qk = pl.ConfigCoordinates(results["masses"], results["R_es"], results["R_gs"], results["modes_gs"])
        elif qk_calculation_type == 'f':
            qk = pl.ConfigCoordinatesF(results["masses"], results["F_es"], results["F_gs"], results["modes_gs"], results["Ek_gs"])
        else:
            raise ValueError("Unsupported qk_calculation_type. Must be 'r' for coordinate-based or 'f' for force-based calculation.")
        
        results["qk"] = qk  ## Units: sqrt(AMU)*Angstrom, an array of shape (N_modes,)
        results["Sk"] = results["wk_gs"]*(results["qk"]**2)/(2*results["hbar"]) ## Units: dimensionless, an array of shape (N_modes,)
        results["HR"] = results["Sk"].sum()  ## Units: dimensionless, a single value representing the total Huang-Rhys factor
        
    if zpl is not None:
        tmax = 2000
        if zpl != 0:
            Emax = 2.5*zpl
        else:
            Emax = 5000
        tmax_meV = pl.TimeScaling(tmax)
        
        '''
        Standard Huang-Rhys formalism.
        '''
        standard_hr = {}
        E_phonons = pl.IV(0, Emax, tmax_meV)
        S_E = pl.SpectralFunction(results["Sk"], results["Ek_gs"], E_phonons, sigma_init, sigma_final, Lorentz=sidebands_broadening_lorentzian)
        t_meV, S_t, _ = pl.FourierSpectralFunction(results["Sk"], results["Ek_gs"], S_E, E_phonons)
        G_t = pl.GeneratingFunction(results["Sk"], S_t, t_meV, results["Ek_gs"], E_phonons, temperature)
        
        # Emission
        E_photon_emission, A_E_emission = pl.OpticalSpectralFunction(G_t, t_meV, zpl, gamma)
        E_photon_emission, A_E_emission, I_emission = pl.LuminescenceIntensity(E_photon_emission, A_E_emission, zpl)

        # Absorption
        E_photon_absorption = 2*zpl - E_photon_emission
        idx = np.argsort(E_photon_absorption)
        E_photon_absorption = E_photon_absorption[idx]
        A_E_absorption = A_E_emission[idx]
        I_absorption = E_photon_absorption*np.real(A_E_absorption)
        I_absorption /= np.trapezoid(I_absorption, E_photon_absorption)


        t_fs = pl.TimeScaling(t_meV, reverse = True)
        standard_hr["S_E"] = S_E[E_phonons <= (max(results["Ek_gs"]) + 36)]  ## Units: 1/meV, an array of shape (N_phonon_energies,)
        standard_hr["E_phonons"] = E_phonons[E_phonons <= (max(results["Ek_gs"]) + 36)] ## Units: meV, an array of shape (N_phonon_energies,)
        standard_hr["G_t"] = G_t[(t_fs >= 0) & (t_fs <= 550)] ## Units: dimensionless, an array of shape (N_time_points,)
        standard_hr["t_fs"] = t_fs[(t_fs >= 0) & (t_fs <= 550)] ## Units: fs, an array of shape (N_time_points,)
        standard_hr["E_photon_emission"] = E_photon_emission ## Units: meV, an array of shape (N_photon_energies,)
        standard_hr["A_E_emission"] = A_E_emission ## Units: arb. units, an array of shape (N_photon_energies,)
        standard_hr["I_emission"] = I_emission ## Units: arb. units, an array of shape (N_photon_energies,)
        standard_hr["E_photon_absorption"] = E_photon_absorption ## Units: meV, an array of shape (N_photon_energies,)
        standard_hr["A_E_absorption"] = A_E_absorption ## Units: arb. units, an array of shape (N_photon_energies,)
        standard_hr["I_absorption"] = I_absorption ## Units: arb. units, an array of shape (N_photon_energies,)
        results["standard_hr"] = standard_hr


        '''
        Luminescence and absorption under a more general formalism that takes into account the displacement and curvature changes between the ground state and excited state potential energy surfaces. 
        This can result in same spectra as the standard Huang-Rhys theory if the vibrational/phonon frequencies of the ground state and excited state are the same.
        '''
        if enable_squeezing:
            if sigma_squeezed is None and gamma_squeezed is None:
                raise ValueError("Specify the broadening parameters for the squeezed spectra. gamma_squeezed can usually be the same as gamma but for sigma_squeezed a smaller value than sigma is recommended.")
            squeezed = {}

            rk, G_t_emission, G_t_absorption = pl.generating_function_distorted(results["Sk"], results["Ek_gs"], results["Ek_es"], t_meV, sigma_squeezed, squeezing_parameter)
            
            E_photon_emission, A_E_emission = pl.OpticalSpectralFunction(G_t_emission, t_meV, zpl, gamma_squeezed)
            E_photon_absorption, A_E_absorption = pl.OpticalSpectralFunction(G_t_absorption, t_meV, zpl, gamma_squeezed)

            E_photon_emission, A_E_emission, I_emission = pl.LuminescenceIntensity(E_photon_emission, A_E_emission, zpl)
            E_photon_absorption, A_E_absorption, I_absorption = pl.LuminescenceIntensity(E_photon_absorption, A_E_absorption, zpl, absorption = True)

            t_fs = pl.TimeScaling(t_meV, reverse = True)
            nk_mean_emission, nk_mean_absorption, E_phonons, S_E_emission, S_E_absorption = pl.spectral_function_distorted(results["Sk"], rk, results["Ek_gs"], results["Ek_es"], sigma_squeezed)

            squeezed["rk"] = rk ## Units: dimensionless, an array of shape (N_modes,)
            squeezed["G_t_emission"] = G_t_emission ## Units: dimensionless, an array of shape (N_time_points,)
            squeezed["G_t_absorption"] = G_t_absorption ## Units: dimensionless, an array of shape (N_time_points,)
            squeezed["t_fs"] = t_fs ## Units: fs, an array of shape (N_time_points,)
            squeezed["E_photon_emission"] = E_photon_emission ## Units: meV, an array of shape (N_photon_energies,)
            squeezed["A_E_emission"] = A_E_emission ## Units: arb. units, an array of shape (N_photon_energies,)
            squeezed["I_emission"] = I_emission ## Units: arb. units, an array of shape (N_photon_energies,)
            squeezed["E_photon_absorption"] = E_photon_absorption ## Units: meV, an array of shape (N_photon_energies,)
            squeezed["A_E_absorption"] = A_E_absorption ## Units: arb. units, an array of shape (N_photon_energies,)
            squeezed["I_absorption"] = I_absorption ## Units: arb. units, an array of shape (N_photon_energies,)
            squeezed["nk_mean_emission"] = nk_mean_emission ## Units: dimensionless, an array of shape (N_photon_energies,)
            squeezed["nk_mean_absorption"] = nk_mean_absorption ## Units: dimensionless, an array of shape (N_photon_energies,)
            squeezed["E_phonons"] = E_phonons ## Units: meV, an array of shape (N_phonon_energies,)
            squeezed["S_E_emission"] = S_E_emission ## Units: 1/meV, an array of shape (N_phonon_energies,)
            squeezed["S_E_absorption"] = S_E_absorption ## Units: 1/meV, an array of shape (N_phonon_energies,)
            results["squeezed"] = squeezed 
        else:
            results["squeezed"] = None
    
    if save_to_hdf5:
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 saving. Install with: pip install h5py"
            )
        filename = f"spectra_output.h5"

        def save_dict(group, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    subgroup = group.create_group(k)
                    save_dict(subgroup, v)
                else:
                    try:
                        group.create_dataset(k, data=v)
                    except TypeError:
                        pass  # skip unsupported types

        with h5py.File(filename, "w") as f:
            save_dict(f, results)

        print(f"Results saved to {filename}")
    
    return results



            




    





    


