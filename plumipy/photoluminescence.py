import numpy as np

class ReadFiles:

  def __init__(self):
    pass

  def ReadStructure(self, path):

    """
    Input:   1. path - Location of POSCAR or CONTCAR file as a string.

    Outputs: 1. Position vectors of all the atoms as numpy array of shape (total number of atoms, 3), where 3 is
                the x,y and z space coordinates.
             2. Dictionary of atomic species and there corresponding number of atoms.
    """
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
      return (atomic_positions, atoms)
      
      

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
    return atomic_masses, freqs, normal_modes

  def ReadPhononsVasp(self, path, atoms):

    """
    From VASP OUTCAR.
    Input:   1. path: Location of band.yaml file as a string.
             2. atoms: Dictionary of atomic species and there corresponding number of atoms.

    Outputs:  1.Atomic_masses is a 1D array of masses (AMU) of each atom in the same sequence as
                that of Atomic positions in previous function.
              2. Phonon frequencies (THz) as a 1D at Gamma point. Length of the array = number of normal modes.
              3. Eigenvectors corresponding to the phonon frequencies as a 3D array of
                shape (number of normal mode, number of atoms, 3), where 3 is the x,y and z coordinates.
    """

    freqs = []
    normal_modes = []
    number_of_atoms = np.array([i[1] for i in atoms.items()])
    total_atoms = np.sum(number_of_atoms)


    with open(path, 'r') as file:
          lines = [line.strip() for line in file]

          index = lines.index("Mass of Ions in am")
          atomic_masses = lines[index + 1].split()[2:]
          atomic_masses = np.array(atomic_masses).astype(float)

          index_init = lines.index("Eigenvectors and eigenvalues of the dynamical matrix")
          index_final = next(
    i for i, line in enumerate(lines)
    if "Finite differences POTIM=" in line or "ELASTIC MODULI CONTR FROM IONIC RELAXATION (kBar)" in line
)


          for i in range(index_init, index_final + 1):
            internal_modes = []
            if "THz" in lines[i]:
              freqs.append(lines[i].split()[lines[i].split().index("THz") - 1])
              internal_modes = [lines[j].split() for j in range(i + 2, i + 2 + total_atoms)]
              normal_modes.append(internal_modes)

    atomic_masses = np.repeat(atomic_masses, number_of_atoms)
    freqs = np.array(freqs).astype(float)
    sort = np.argsort(freqs)
    freqs = freqs[sort]
    normal_modes = np.array(normal_modes).astype(float)[...,3:]
    normal_modes = normal_modes[sort]
    return atomic_masses, freqs, normal_modes

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

  def SpectralFunction(self, Sk, Ek, E_meV_positive, sigma = 6, Lorentz = False):

    """
    Calculates S(hbar_omega) or S(E) (unit less) by using Gaussian or Lorentzian fit
    for Direc-Delta with sigma = 6 meV by default.

    Ek: Normal mode phonon energies.
    """
    self.sigma = sigma
    if Lorentz == False:
      S_E = np.array([np.dot(Sk,self.Gaussian(i,Ek,sigma)) for i in E_meV_positive])
    else:
      S_E = np.array([np.dot(Sk,self.Lorentzian(i,Ek,sigma)) for i in E_meV_positive])
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
  
  @staticmethod
  def anharmonic_coefficients(F_es, F_gs, modes, masses, wk, qk):
     """
     Calculates the lamba_k from U = 1/2(wk**2)Q**2 + lambda_k(q**3) 
     """
     F_diff = (F_es - F_gs)*1000
     Fk = np.array([np.dot(1/np.sqrt(masses),np.sum(modes[k]*F_diff, axis = 1)) for k in range(len(wk))])
     lam_k = (Fk - (wk**2)*qk)/(3*(qk**2))
     return lam_k