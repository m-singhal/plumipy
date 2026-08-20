import re
import numpy as np

class ReadFiles:

  def __init__(self):
    pass

  def ReadStructureXYZ(self, path):
    """
    Reads atomic positions from a standard or extended .xyz file.

    Standard XYZ:
        Line 0 : number of atoms
        Line 1 : comment (ignored if no Lattice keyword)
        Line 2+: element  x  y  z  [Å, Cartesian]

    Extended XYZ (ASE / OVITO style):
        Line 1 comment carries key=value pairs; if a Lattice="…" field is
        present its 9 values are interpreted as a row-major 3×3 lattice
        matrix (a1 a2 a3  b1 b2 b3  c1 c2 c3).

    Returns:
        positions : (N, 3) float array, Cartesian Å
        species   : list of element symbols, one per atom, in file order
        lattice   : (3, 3) float array (row vectors), or None if not given
    """
    with open(path, 'r') as f:
        lines = [l.rstrip('\n') for l in f.readlines()]

    natoms = int(lines[0].strip())
    comment = lines[1] if len(lines) > 1 else ""

    # Parse extended-XYZ Lattice field if present
    lattice = None
    lat_match = re.search(r'Lattice\s*=\s*"([^"]+)"', comment, re.IGNORECASE)
    if lat_match:
        vals = [float(x) for x in lat_match.group(1).split()]
        if len(vals) == 9:
            lattice = np.array(vals, dtype=float).reshape(3, 3)

    species, positions = [], []
    for i in range(2, 2 + natoms):
        parts = lines[i].split()
        species.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return np.array(positions, dtype=float), species, lattice

  def ReadStructure(self, path):

    """
    Input:   1. path - Location of POSCAR/CONTCAR or .xyz file as a string.

    Outputs: 1. Position vectors of all the atoms as numpy array of shape (total number of atoms, 3), where 3 is
                the x,y and z space coordinates.
             2. Atomic species: dict {element: count} for POSCAR/CONTCAR,
                or list [el, el, ...] (per-atom, file order) for .xyz.
             3. Lattice vectors (3,3) array, or None for .xyz with no Lattice field.
    """
    import os
    if os.path.splitext(path)[1].lower() == '.xyz':
        return self.ReadStructureXYZ(path)

    with open(path,'r') as file:

      lines = file.readlines()

      scaling_factor = float(lines[1].strip())
      lattice_vectors = [lines[i].strip().split() for i in range(2,5)]
      lattice_vectors = scaling_factor*(np.array(lattice_vectors).astype(float))

      atomic_species = lines[5].strip().split()
      number_of_atoms = np.array(lines[6].strip().split()).astype(int)
      tot = sum(number_of_atoms)

      lattice_type = lines[7].strip()

      atomic_positions = [lines[i].strip().split() for i in range(8,8+tot)]
      atomic_positions = np.array(atomic_positions).astype(float)
      atoms = dict(zip(atomic_species, number_of_atoms))
      
      if lattice_type != "Direct":
        latticeInv = np.linalg.inv(lattice_vectors.T)
        Rd = np.array([np.dot(latticeInv,vec) for vec in atomic_positions])
        atomic_positions = Rd

      atomic_positions[atomic_positions > 0.99] -= 1
      atomic_positions = np.dot(atomic_positions, lattice_vectors)
      return (atomic_positions, atoms, lattice_vectors)
      
      

  def ReadPhononsPhonopy(self, path):

    """
    Input:   1. path: Location of band.yaml file as a string.

    Outputs: 1. Atomic_masses is a 1D array of masses (AMU) of each atom in the same sequence as
                that of Atomic positions in previous function.
             2. Phonon frequencies (THz) as a 1D at Gamma point. Length of the array = number of normal modes.
             3. Eigenvectors corresponding to the phonon frequencies as a 3D array of
                shape (number of normal mode, number of atoms, 3), where 3 is the x,y and z coordinates.
    """
    with open(path,'r') as file:
      lines = [ts.strip() for ts in file]

    atomic_masses = []
    freqs = []
    normal_modes = []
    with open(path,'r') as file:
      for line in file:
        if "mass:" in line:
          atomic_masses.append(line.split()[1])
    atomic_masses = np.array(atomic_masses).astype(float)
    total_atoms = len(atomic_masses)
    with open(path,'r') as file:
      line_number = -1
      for line in file:
        line_number += 1
        if "frequency:" in line:
          freqs.append(float(line.split()[1]))
          ev_internal = []
          for i in range(line_number+3,line_number + 4*total_atoms + 2,4):
            xyz = [lines[i+j].split()[2] for j in range(3)]
            ev_internal.append(xyz)
          normal_modes.append(ev_internal)
    freqs = np.array(freqs).astype(float)
    freqs[freqs<0] = 0
    normal_modes = np.array([[[float(x.strip(',')) for x in sublist] for sublist in outer] for outer in normal_modes])
    # band.yaml may contain multiple q-points (e.g. Γ→Γ path); only the first 3N modes belong to Γ.
    n_modes = 3 * total_atoms
    return atomic_masses, freqs[:n_modes], normal_modes[:n_modes]

  
  def ReadPhononsVasp(self, path):

    """
    From VASP OUTCAR.
    Input:   1. path: Location of band.yaml file as a string.

    Outputs:  1.Atomic_masses is a 1D array of masses (AMU) of each atom in the same sequence as
                that of Atomic positions in previous function.
              2. Phonon frequencies (THz) as a 1D at Gamma point. Length of the array = number of normal modes.
              3. Eigenvectors corresponding to the phonon frequencies as a 3D array of
                shape (number of normal mode, number of atoms, 3), where 3 is the x,y and z coordinates.
    """

    freqs = []
    normal_modes = []

    with open(path, 'r') as file:
        lines = [line.strip() for line in file]

    # --- get masses ---
    mass_idx = lines.index("Mass of Ions in am")
    atomic_masses = np.array(lines[mass_idx + 1].split()[2:], dtype=float)

    # --- get number of atoms per type ---
    ions_line = [l for l in lines if "ions per type" in l][0]
    number_of_atoms = np.array(ions_line.split('=')[1].split(), dtype=int)

    total_atoms = np.sum(number_of_atoms)

    # --- expand masses correctly ---
    atomic_masses_full = np.repeat(atomic_masses, number_of_atoms)

    # --- locate phonon block ---
    index_init = lines.index("Eigenvectors and eigenvalues of the dynamical matrix")

    index_final = next(
        i for i, line in enumerate(lines)
        if "Finite differences POTIM=" in line
        or "ELASTIC MODULI CONTR FROM IONIC RELAXATION" in line
    )

    # --- parse modes ---
    for i in range(index_init, index_final + 1):
        if "THz" in lines[i]:
            freq = float(lines[i].split()[lines[i].split().index("THz") - 1])
            freqs.append(freq)

            mode_block = [
                lines[j].split()
                for j in range(i + 2, i + 2 + total_atoms)
            ]
            normal_modes.append(mode_block)

    freqs = np.array(freqs, dtype=float)
    normal_modes = np.array(normal_modes, dtype=float)[..., 3:]

    # --- sort ---
    sort = np.argsort(freqs)
    freqs = freqs[sort]
    normal_modes = normal_modes[sort]

    return atomic_masses_full, freqs, normal_modes

  def ReadPhononsFlagged(self, path):

    """
    Reads OUTCAR or band.yaml WITHOUT discarding imaginary/unstable modes.

    ReadPhononsVasp only keeps the numeric THz value and drops VASP's 'f/i'
    (imaginary) tag; ReadPhononsPhonopy clips negative frequencies to 0. Both
    make it impossible to tell which modes were unstable after the fact. This
    unifies both formats into one signed convention: NEGATIVE energy/frequency
    means imaginary/unstable, regardless of which file it came from (VASP
    reports a positive magnitude with a separate 'f/i' text tag; Phonopy
    reports a literal negative number -- both become a negative value here).

    Input:  path - OUTCAR or band.yaml.

    Output: masses (N,) amu, freqs (N_modes,) THz [signed], modes
            (N_modes, N_atoms, 3), energies (N_modes,) meV [signed]
    """
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext == ".yaml":
        masses, freqs, modes = self._read_phonopy_signed(path)
    else:
        masses, freqs, modes = self._read_outcar_signed(path)
    energies = 4.13566 * freqs
    return masses, freqs, modes, energies

  def _read_phonopy_signed(self, path):
    with open(path, 'r') as file:
        lines = [ts.strip() for ts in file]

    atomic_masses = []
    for line in lines:
        if "mass:" in line:
            atomic_masses.append(line.split()[1])
    atomic_masses = np.array(atomic_masses, dtype=float)
    total_atoms = len(atomic_masses)

    freqs, normal_modes = [], []
    for line_number, line in enumerate(lines):
        if "frequency:" in line:
            freqs.append(float(line.split()[1]))
            ev_internal = []
            for i in range(line_number + 3, line_number + 4 * total_atoms + 2, 4):
                xyz = [lines[i + j].split()[2] for j in range(3)]
                ev_internal.append(xyz)
            normal_modes.append(ev_internal)

    freqs = np.array(freqs, dtype=float)   # sign preserved -- NOT clipped to 0
    normal_modes = np.array(
        [[[float(x.strip(',')) for x in sub] for sub in outer] for outer in normal_modes]
    )
    n_modes = 3 * total_atoms
    return atomic_masses, freqs[:n_modes], normal_modes[:n_modes]

  def _read_outcar_signed(self, path):
    with open(path, 'r') as file:
        lines = [line.strip() for line in file]

    mass_idx = lines.index("Mass of Ions in am")
    atomic_masses = np.array(lines[mass_idx + 1].split()[2:], dtype=float)

    ions_line = [l for l in lines if "ions per type" in l][0]
    number_of_atoms = np.array(ions_line.split('=')[1].split(), dtype=int)
    total_atoms = int(np.sum(number_of_atoms))
    atomic_masses_full = np.repeat(atomic_masses, number_of_atoms)

    index_init = lines.index("Eigenvectors and eigenvalues of the dynamical matrix")
    index_final = next(
        i for i, line in enumerate(lines)
        if i > index_init and (
            "Finite differences POTIM=" in line
            or "ELASTIC MODULI CONTR FROM IONIC RELAXATION" in line
        )
    )

    freqs, normal_modes = [], []
    for i in range(index_init, index_final + 1):
        if "THz" in lines[i]:
            toks = lines[i].split()
            freq = float(toks[toks.index("THz") - 1])
            if "f/i" in lines[i]:
                freq = -freq   # VASP prints a positive magnitude; unify sign convention
            freqs.append(freq)
            mode_block = [lines[j].split() for j in range(i + 2, i + 2 + total_atoms)]
            normal_modes.append(mode_block)

    freqs = np.array(freqs, dtype=float)
    normal_modes = np.array(normal_modes, dtype=float)[..., 3:]

    # Sort by |freq| (magnitude) so imaginary modes land alongside the other
    # near-zero modes rather than all being sorted to one end by their sign.
    sort = np.argsort(np.abs(freqs))
    return atomic_masses_full, freqs[sort], normal_modes[sort]

  def parse_mode_range(self, text):
    """'1-5, 9-11' (1-based, as shown to the user) -> set of 0-based indices."""
    text = (text or "").strip().strip("()")
    if not text:
        return set()
    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r'^(\d+)\s*-\s*(\d+)$', part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            out.update(range(lo - 1, hi))
        else:
            out.add(int(part) - 1)
    return out

  def parse_energy_ranges(self, text):
    """'0-24, 100-110' (meV) -> list of (lo, hi) tuples. Negative bounds allowed.

    Uses an anchored two-group match rather than scanning for all numbers:
    a bare findall on '-?\\d+' misreads the separating '-' in e.g. '0-24' as
    a sign on '24', giving (-24, ...) instead of splitting into (0, 24).
    """
    text = (text or "").strip().strip("()")
    if not text:
        return []
    ranges = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r'^(-?\d+\.?\d*)\s*-\s*(-?\d+\.?\d*)$', part)
        if not m:
            raise ValueError(f"Cannot parse energy range: '{part}'")
        lo, hi = float(m.group(1)), float(m.group(2))
        ranges.append((min(lo, hi), max(lo, hi)))
    return ranges

  def build_exclusion_mask(self, energies_meV, mode_range_text, energy_range_text):
    """Union of mode-number and energy-range exclusions. True = excluded."""
    N = len(energies_meV)
    mask = np.zeros(N, dtype=bool)
    for idx in self.parse_mode_range(mode_range_text):
        if 0 <= idx < N:
            mask[idx] = True
    for lo, hi in self.parse_energy_ranges(energy_range_text):
        mask |= (energies_meV >= lo) & (energies_meV <= hi)
    return mask

  def InverseHessian(self, masses, energies_meV_kept, modes_kept):
    """
    Constructs H^-1 (units Angstrom^2/eV) from the KEPT modes only, such that
    H_inv @ F.ravel() with F given directly in eV/Angstrom yields a
    displacement in Angstrom -- no separate unit conversion needed by the
    caller.

        H^-1 = 1000 * M^-1/2  eta  Omega_signed^-2  eta^T  M^-1/2

    Uses SIGNED Omega^2 = sign(E) * (E/hbar)^2, not |E|^2/hbar^2: if a mode
    the user chose to KEEP is flagged imaginary (negative energy), this
    preserves its true negative curvature in the saved matrix rather than
    silently treating it as an ordinary positive-curvature mode. No small-E
    threshold is applied -- a small-but-nonzero energy just gives a large
    (finite) contribution, which is the user's call to exclude or not. Only
    an EXACT zero (1/0, genuinely undefined rather than merely large) raises.
    """
    energies_meV_kept = np.asarray(energies_meV_kept, dtype=float)
    signed_w2 = np.sign(energies_meV_kept) * (energies_meV_kept / self.hbar) ** 2
    if np.any(signed_w2 == 0):
        raise ValueError(
            "One or more kept modes have exactly zero energy -- their inverse "
            "stiffness is undefined (1/0), not just large. Exclude them before "
            "constructing H^-1."
        )
    m3 = np.repeat(masses, 3)
    inv_sqrt_m = 1.0 / np.sqrt(m3)
    N_k = len(energies_meV_kept)
    eta = modes_kept.reshape(N_k, -1) * inv_sqrt_m[None, :]
    H_inv = 1000.0 * (eta.T / signed_w2[None, :]) @ eta
    return H_inv

  def ReadForces(self, path):

    """
    Reads and stores the Forces (eV/Angstrom) on each atom from the OUTCAR file and returns a 2D array.
    """
    with open(path, "r") as f:
            lines = f.readlines()
            start = end = None
            for index,line in enumerate(lines):
                if "TOTAL-FORCE" in line:
                    start = index + 2
                if "total drift" in line:
                    end = index - 1
            if start is None or end is None:
                raise ValueError(f"Force data not found in OUTCAR.")
            F = np.loadtxt(lines[start:end])
    return F[:,3:]

  def ReadForceFile(self, path):
    """
    Reads forces from any of: OUTCAR (last TOTAL-FORCE block -- see
    ReadForces, which already scans to the LAST occurrence, not the first),
    .npy, .npz, or whitespace-delimited .dat/.txt.
    """
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        d = np.load(path)
        return d[d.files[0]]
    if ext in (".txt", ".dat"):
        return np.loadtxt(path)
    return self.ReadForces(path)

  def ReadStructureFile(self, path):
    """
    Reads a structure from any supported format, returning both the
    Cartesian positions AND enough metadata to write the SAME format back
    out later (WriteStructureFile).

    Output: positions (N,3) Cartesian Angstrom, format_info dict:
        POSCAR/CONTCAR -> {"kind":"poscar", "atoms":{el:count,...},
                            "lattice":(3,3), "comment":str}
        OUTCAR          -> same "poscar" kind (geometry + lattice extracted
                            from the OUTCAR's own LAST ionic-step block, so
                            it can be saved straight back out as a POSCAR)
        .xyz            -> {"kind":"xyz", "species":[el,...], "lattice":(3,3)|None}
        .npy/.npz/.dat/.txt -> {"kind":"array", "ext":str}
    """
    import os
    ext = os.path.splitext(path)[1].lower()

    if ext in (".npy", ".npz", ".txt", ".dat"):
        if ext == ".npy":
            positions = np.load(path)
        elif ext == ".npz":
            d = np.load(path)
            positions = d[d.files[0]]
        else:
            positions = np.loadtxt(path)
        return positions, {"kind": "array", "ext": ext}

    if ext == ".xyz":
        positions, species, lattice = self.ReadStructureXYZ(path)
        return positions, {"kind": "xyz", "species": species, "lattice": lattice}

    # OUTCAR detection: a cheap check on the first line (every OUTCAR opens
    # with a "vasp.X.Y.Z ..." banner), before falling through to the
    # POSCAR/CONTCAR parser -- which would otherwise misparse an OUTCAR's
    # header as POSCAR fields.
    with open(path) as f:
        first_line = f.readline()
    if first_line.strip().lower().startswith("vasp."):
        positions, atoms, lattice = self._read_outcar_structure(path)
        return positions, {
            "kind": "poscar", "atoms": atoms, "lattice": lattice,
            "comment": "Generated by plumipy from OUTCAR (last ionic step)",
        }

    positions, atoms, lattice = self.ReadStructure(path)
    with open(path) as f:
        comment = f.readline().rstrip("\n")
    return positions, {"kind": "poscar", "atoms": atoms, "lattice": lattice,
                        "comment": comment}

  def _read_outcar_structure(self, path):
    """
    Reads geometry from an OUTCAR's LAST POSITION+TOTAL-FORCE block --
    first 3 columns are position, next 3 are force; only the position
    columns are kept here (see ReadForces / ReadForceFile for the force
    columns from that same block) -- plus the supercell lattice and
    per-species atom counts, so this can be written out as a POSCAR.
    """
    with open(path) as f:
        lines = [l.strip() for l in f]

    titel_species = []
    for l in lines:
        if "TITEL" in l:
            m = re.search(r'PAW_PBE\s+(\w+)', l)
            if m and m.group(1) not in titel_species:
                titel_species.append(m.group(1))

    ions_line_idx = next(i for i, l in enumerate(lines) if "ions per type" in l)
    counts = list(map(int, lines[ions_line_idx].split("=")[1].split()))
    N = int(sum(counts))
    atoms = dict(zip(titel_species, counts))

    # Supercell lattice -- first "direct lattice vectors" block AFTER
    # "ions per type" (the one printed earlier, from the POTCAR, is the
    # primitive cell and is the wrong one for a supercell calculation).
    lat_idx = next(
        i for i, l in enumerate(lines)
        if i > ions_line_idx and "direct lattice vectors" in l
    )
    lattice = np.array([
        [float(x) for x in lines[lat_idx + j + 1].split()[:3]]
        for j in range(3)
    ])

    force_idxs = [i for i, l in enumerate(lines) if "TOTAL-FORCE" in l]
    if not force_idxs:
        raise ValueError(f"No TOTAL-FORCE block found in {path}.")
    start = force_idxs[-1] + 2   # last occurrence = last ionic step
    block = np.array([
        [float(x) for x in lines[start + j].split()]
        for j in range(N)
    ])
    positions = block[:, :3]   # first three columns = geometry
    return positions, atoms, lattice

  def WriteStructureFile(self, path, positions, format_info):
    """Writes `positions` back out in the SAME format described by
    `format_info` (as returned by ReadStructureFile)."""
    kind = format_info["kind"]
    if kind == "array":
        ext = format_info["ext"]
        if ext == ".npy":
            np.save(path, positions)
        elif ext == ".npz":
            np.savez(path, positions=positions)
        else:
            np.savetxt(path, positions)
    elif kind == "xyz":
        self._write_xyz(path, positions, format_info["species"], format_info.get("lattice"))
    elif kind == "poscar":
        self._write_poscar(path, positions, format_info["atoms"], format_info["lattice"],
                           format_info.get("comment", "Generated by plumipy"))
    else:
        raise ValueError(f"Unknown structure format kind: {kind}")

  def _write_poscar(self, path, positions, atoms, lattice, comment):
    species = list(atoms.keys())
    counts = list(atoms.values())
    if sum(counts) != len(positions):
        raise ValueError(
            f"Atom count mismatch: POSCAR species/counts sum to {sum(counts)}, "
            f"positions has {len(positions)} rows."
        )
    lines = [comment or "Generated by plumipy", "   1.0"]
    for row in lattice:
        lines.append(f"   {row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}")
    lines.append("  " + "  ".join(str(s) for s in species))
    lines.append("  " + "  ".join(str(int(c)) for c in counts))
    lines.append("Cartesian")
    for p in positions:
        lines.append(f"  {p[0]:.10f}  {p[1]:.10f}  {p[2]:.10f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

  def _write_xyz(self, path, positions, species, lattice):
    if len(species) != len(positions):
        raise ValueError(
            f"Atom count mismatch: {len(species)} species vs {len(positions)} positions."
        )
    lines = [str(len(positions))]
    if lattice is not None:
        lat_str = " ".join(f"{x:.10f}" for row in lattice for x in row)
        lines.append(f'Lattice="{lat_str}" Properties=species:S:1:pos:R:3')
    else:
        lines.append("")
    for el, p in zip(species, positions):
        lines.append(f"{el}  {p[0]:.10f}  {p[1]:.10f}  {p[2]:.10f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

  def ApplyNewtonStep(self, positions, forces, H_inv):
    """
    R_new = R + H^-1 F.

    positions, forces: (N,3) Cartesian Angstrom / eV per Angstrom
    H_inv: (3N,3N), units Angstrom^2/eV (as built by InverseHessian), so the
    dot product with forces directly in eV/Angstrom yields Angstrom -- no
    separate unit conversion needed here.
    """
    N = positions.shape[0]
    if H_inv.shape != (3 * N, 3 * N):
        raise ValueError(
            f"H_inv shape {H_inv.shape} does not match 3*N_atoms={3*N} "
            f"implied by the structure file ({N} atoms)."
        )
    if forces.shape != positions.shape:
        raise ValueError(
            f"forces shape {forces.shape} does not match positions shape "
            f"{positions.shape}."
        )
    disp = (H_inv @ forces.ravel()).reshape(N, 3)
    return positions + disp



class Photoluminescence(ReadFiles):

  def __init__(self):

    """
    Define all the variables by reading the input files like POSCAR_GS/CONTCAR_GS, POSCAR_ES/CONTCAR_ES, and band.yaml.
    """
    self.hbar = 0.6582*np.sqrt(9.646) #sqrt(meV*AMU)*Angstrom
    super().__init__()

  def IV(self, iv_low, iv_high, rv_high):

    """
    This function can be used to obtain a 1D time array with equal intervals.

    iv: Independent Variable;
    rv: Reciprocal Variable.

    Inputs: Min max values of independent variable, and
    max value of rv required by the user.

    div: Minimum resolution of iv.

    Output: 1D array of independent variable (usually time in this case).
    """

    div = (2*np.pi)/(2*rv_high)
    return np.arange(iv_low, iv_high, div)

  def Fourier(self, independent_variable, function):
      iv = independent_variable
      div = iv[1] - iv[0]
      rv = 2*np.pi*np.fft.fftfreq(len(iv),div)
      sort = np.argsort(rv)
      reciprocal_variable = rv[sort]
      dft = np.fft.fft(function)[sort]
      fourier_transform = div*dft*np.exp(-1j*reciprocal_variable*iv[0])
      return reciprocal_variable, fourier_transform

  def InverseFourier(self, independent_variable, function):
      iv = independent_variable
      div = iv[1] - iv[0]
      rv = 2*np.pi*np.fft.fftfreq(len(iv),div)
      sort = np.argsort(rv)
      reciprocal_variable = rv[sort]
      idft = np.fft.ifft(function)[sort]
      inverse_fourier_transform = div*idft*np.exp(1j*reciprocal_variable*iv[0])*len(rv)/(2*np.pi)
      return reciprocal_variable, inverse_fourier_transform

  def Trapezoidal(self, integrand, iv, equally_spaced = True):

    """
    Calculates the integral using Trapezoidal Rule.

    Inputs: integrand and iv are arrays of same dimension. equally_spaced: determines whether the method should integrate using
    equally spaced or unequally spaced intervals.

    Output: Integration result.
    """
    div = iv[1] - iv[0]
    return (div/2)*(np.sum(integrand[1:-1]) + integrand[0] + integrand[-1]) if equally_spaced \
    else np.sum(np.array([((iv[i+1] - iv[i])/2)*(integrand[i+1] + integrand[i]) for i in range(len(iv)-1)]))

  def FreqToEnergy(self, freqs):

    """Coversion of frequencies (THz) to Energy (meV)."""

    return 4.13566*freqs


  def TimeScaling(self, t, reverse = False):

    """
    Changes time array t from femtoseconds to meV^-1. This is a necesaary step after initializing time through IV
    function in order to maintain consistency in units while performing Fourier Transform.
    """
    return t/658.2119 if reverse == False else t*658.2119

  def Lorentzian(self, x, x0, sigma):

    """
    Used to fit Dirac-Delta as Lorentzian function, where sigma = 6 has units of meV.
    The factor of 0.8 multiplying sigma is to make this function have similarities to
    Gaussian for same standard deviation, sigma.
    """
    return ((1/np.pi)*(sigma*0.8))/(((sigma*0.8)**2) + ((x - x0)**2))

  def Gaussian(self, x, x0, sigma):

    """
    Gaussian fit for Dirac-Delta with sigma = 6 (meV) as standard deviation.
    """
    return (1/np.sqrt(2*np.pi*(sigma**2)))*np.exp(-((x-x0)**2)/(2*(sigma**2)))

  def ConfigCoordinates(self, masses, R_es, R_gs, modes):

    """
    Calculates the qk factor (AMU^0.5-Angstrom) for different normal modes as a 1D array of
    length = total number of normal modes.
    """
    masses = np.sqrt(masses)
    R_diff = R_es - R_gs
    mR_diff = np.array([masses[i]*R_diff[i,:] for i in range(len(masses))])
    qk = np.array([np.sum(mR_diff*modes[i,:,:]) for i in range(modes.shape[0])])
    return qk

  def ConfigCoordinatesF(self, masses, F_es, F_gs, modes, Ek):

    """
    Calculates the qk factor (AMU^0.5-Angstrom) for different normal modes as a 1D array of
    length = total number of normal modes. This function uses forces on atoms rather than their position vectors
    as used in previous function.
    """
    masses = np.sqrt(masses)
    F_diff = (F_es - F_gs)*1000
    mF_diff = np.array([(1/masses[i])*F_diff[i,:] for i in range(len(masses))])
    qk = np.array([np.sum(mF_diff*modes[i,:,:]) for i in range(modes.shape[0])])
    qk = (self.hbar**2/Ek**2)*qk
    return qk

  def PartialHR(self, freqs, qk):

    """
    Calculates the Sk (unit less) as a 1D array of length equal to total number of normal modes.
    """
    return (2*np.pi*freqs*(qk**2))/(2*0.6582*9.646)

  def SpectralFunction(self, Sk, Ek, E_meV_positive, sigma_init, sigma_final=None, Lorentz = False):

    """
    Calculates S(hbar_omega) or S(E) (unit less) by using Gaussian or Lorentzian fit
    for Direc-Delta with sigma = 6 meV by default.

    Ek: Normal mode phonon energies.
    """
    self.sigma = sigma_init
    if sigma_final is not None:
      sigma_k = sigma_init + (sigma_final - sigma_init)*((Ek - Ek.min())/(Ek.max() - Ek.min()))
    else:
      sigma_k = sigma_init
    
    if Lorentz == False:
      S_E = np.array([np.dot(Sk,self.Gaussian(i,Ek,sigma_k)) for i in E_meV_positive])
    else:
      S_E = np.array([np.dot(Sk,self.Lorentzian(i,Ek,sigma_k)) for i in E_meV_positive])
    return S_E

  def FourierSpectralFunction(self, Sk, Ek, S_E, E_meV_positive):

    """
    Calculates the Fourier transform of S(E) which is equal to S(t).
    """
    t_meV, S_t = self.Fourier(E_meV_positive, S_E)
    S_t_exact = np.array([np.dot(Sk,np.exp(-1j*Ek*i)) for i in t_meV])
    return t_meV, S_t, S_t_exact

  def GeneratingFunction(self, Sk, S_t, t_meV, Ek, E_meV_positive, T):

    """
    Calculates the generating function G(t).
    """
    if T == 0.0:
      G_t = np.exp((S_t) - (np.sum(Sk)))
    else:
      Kb = 8.61733326e-2 # Boltzmann constant in meV/k
      nk = 1/((np.exp(Ek/(Kb*T))) - 1)
      C_E = np.array([np.dot(nk*Sk,self.Gaussian(i,Ek,self.sigma)) for i in E_meV_positive])
      C_t = self.Fourier(E_meV_positive, C_E)[1]
      C_t_inv = self.InverseFourier(E_meV_positive, C_E)[1]
      G_t = np.exp((S_t) - (np.sum(Sk)) + C_t + C_t_inv - 2*np.sum(nk*Sk))
    return G_t
  
  def generating_function_distorted(self, Sk, Ek_gs, Ek_es, t_meV, sigma, rk_init = None):
     
      if rk_init is not None:
         rk = rk_init
      else:
        rk = 0.5*np.log(Ek_es/Ek_gs)
        rk[np.isclose(rk, 0)] = 1e-8
      broadening = np.exp(-0.5*((t_meV**2)*(sigma**2)))

      # Emission
      rho_k_t = np.array([np.exp(-1j*t_meV*Ek_gs[k])*np.tanh(rk[k]) for k in range(len(Sk))])
      L_k_t = np.array([(1 + np.tanh(rk[k]))*((np.tanh(rk[k]) - rho_k_t[k])/((1 + rho_k_t[k])*np.tanh(rk[k]))) for k in range(len(Sk))])
      ln_G = np.array([np.log(np.cosh(rk[k])) + 0.5*np.log(1 - rho_k_t[k]**2) + Sk[k]*L_k_t[k] for k in range(len(Sk))])
      G_t_emission = broadening*np.exp(-np.sum(ln_G, axis=0))

      # Absorption
      rho_k_t = np.array([np.exp(1j*t_meV*Ek_es[k])*np.tanh(rk[k]) for k in range(len(Sk))])
      L_k_t = np.array([(1 - np.tanh(rk[k]))*((np.tanh(rk[k]) - rho_k_t[k])/((1 - rho_k_t[k])*np.tanh(rk[k]))) for k in range(len(Sk))])
      Sk_abs = Sk*np.exp(2*rk)
      ln_G = np.array([np.log(np.cosh(rk[k])) + 0.5*np.log(1 - rho_k_t[k]**2) + Sk_abs[k]*L_k_t[k] for k in range(len(Sk))])
      G_t_absorption = broadening*np.exp(-np.sum(ln_G, axis=0))
      
      return rk, G_t_emission, G_t_absorption
  
  def spectral_function_distorted(self, Sk, rk, Ek_gs, Ek_es, sigma):
     
     Emax = max(Ek_gs.max(), Ek_es.max())
     E_meV_positive = np.linspace(0, 1.5*Emax, num = 1500)
     
     # Emission
     nk_mean_emission = Sk + np.sinh(rk)**2
     S_E_emission = np.array([np.dot(nk_mean_emission,self.Gaussian(i,Ek_gs,sigma)) for i in E_meV_positive])

     # Absorption
     nk_mean_absorption = Sk*np.exp(2*rk) + np.sinh(rk)**2
     S_E_absorption = np.array([np.dot(nk_mean_absorption,self.Gaussian(i,Ek_es,sigma)) for i in E_meV_positive])

     return nk_mean_emission, nk_mean_absorption, E_meV_positive, S_E_emission, S_E_absorption

      

  def OpticalSpectralFunction(self, G_t, t_meV, zpl, gamma):
    
    E_meV = np.linspace(zpl - 1000, zpl + 1000, 2000)

    A_E = []

    for E in E_meV:
        integrand = (
            G_t
            * np.exp(-1j * (E - zpl) * t_meV)
            * np.exp(-gamma * np.abs(t_meV))
        )

        A_val = np.trapezoid(integrand, t_meV)
        A_E.append(A_val)

    return E_meV, np.array(A_E)

  # def OpticalSpectralFunction(self, G_t, t_meV, zpl, gamma, absorption = False):

  #   """
  #   Calculates the optical spectra A(E).
  #   """
  #   if absorption:
  #      E_meV, A_E =  self.InverseFourier(t_meV, (G_t*np.exp(-1j*zpl*t_meV))*np.exp(-(gamma*np.abs(t_meV))))
  #   else:
  #      E_meV, A_E =  self.Fourier(t_meV, (G_t*np.exp(1j*zpl*t_meV))*np.exp(-(gamma*np.abs(t_meV))))
  #   return E_meV, A_E

  def LuminescenceIntensity(self, E_meV, A_E, zpl, absorption = False):

    """
    Calculates the normalized photoluminescence (PL), L(E)
    """
    # A_E = A_E[(E_meV >= (zpl - 600)) & (E_meV <= (zpl + 600))]
    # E_meV = E_meV[(E_meV >= (zpl - 600)) & (E_meV <= (zpl + 600))]
    if absorption:
        L_E = (E_meV)*np.real(A_E)
        L_E /= np.trapezoid(L_E, E_meV)
    else:
       L_E = (E_meV**3)*np.real(A_E)
       L_E /= np.trapezoid(L_E, E_meV)
    return E_meV, A_E, L_E

  def InverseParticipationRatio(self, modes):

    """
    Calculates the IPR (1D array) for each mode.
    """
    p = np.einsum("ijk -> ij", modes**2)
    IPR = 1/np.einsum("ij -> i", p**2)
    return IPR
  
  def monte_carlo_sampling(self, zpl, Sk, Ek, sigma, n_samples=100000):
     Sk = np.array(Sk)
     Ek = np.array(Ek)

     # Sample phonon numbers: shape (n_samples, n_modes)
     nk = np.random.poisson(lam=Sk, size=(n_samples, len(Sk)))

     # Total emitted phonon energy per sample
     E_loss = np.dot(nk, Ek)

     # Photon energies
     E_photon = zpl - E_loss
     E_photon = E_photon + np.random.normal(0, sigma, len(E_photon))
     hist, bins = np.histogram(E_photon, bins=500, density=True)
     bin_centers = 0.5*(bins[:-1] + bins[1:])

     # --- central tendency ---
     mean = np.mean(E_photon)
     median = np.median(E_photon)
     mode = bin_centers[np.argmax(hist)]

     # --- spread ---
     var = np.var(E_photon)
     std = np.std(E_photon)

     # --- higher moments ---
     centered = E_photon - mean
     m2 = np.mean(centered**2)
     m3 = np.mean(centered**3)
     m4 = np.mean(centered**4)

     # skewness (Pearson moment coefficient)
     skewness = m3 / (m2**1.5)

     # kurtosis (Pearson, not excess)
     kurtosis = m4 / (m2**2)

     # excess kurtosis (more commonly reported)
     excess_kurtosis = kurtosis - 3

     return bin_centers, hist, mean, median, mode, var, std, skewness, excess_kurtosis
  
  @staticmethod
  def anharmonic_coefficients(F_es, F_gs, modes, masses, wk, qk):
     """
     Calculates the lamba_k from U = 1/2(wk**2)Q**2 + lambda_k(q**3) 
     """
     F_diff = (F_es - F_gs)*1000
     Fk = np.array([np.dot(1/np.sqrt(masses),np.sum(modes[k]*F_diff, axis = 1)) for k in range(len(wk))])
     lam_k = (Fk - (wk**2)*qk)/(3*(qk**2))
     return lam_k