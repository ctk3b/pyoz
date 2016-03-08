#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this file is part of the pyOZ bundle
# pyOZ is a solver of the Ornstein-Zernike equation written in Python
# pyOZ is released under the BSD license
# see the file LICENSE for more information

"""
Module defining parameters and constants
revision of this file: 0.1.6

"""

from math import sqrt, pi
from scipy import inf

pyoz_version = "0.3"

class const:
  """
  class containing some frequently used constants
  constructor needs temperature as parameter
  
  Avogadro constant
  Boltzmann constant
  temperature
  kT
  1/kT
  conversion L -> A^3
  conversion A^3 -> L
  ft prefactor for the forward transform (dft implementation dependent) 2/pi
  ft prefactor for the inverse transform (dft implementation dependent) 2/pi
  
  """

  # system properties
  T  = 300.0                    # K
  # relative solvent permittivity
  epsilon_r = 78.3
  # coulomb interaction factor - Bjerrum length
  # V(coul) in kT is then calculated as V = b_l * z1 * z2 / r
  # with z in elementary charge units and r in A
  bjerrum_length = 0.0

  # some basic constants
  Na = 6.0221415                # x 10^{23} mol^{-1}
  k = 1.3806504                 # x 10^{-23} J K^{-1}
  kT = T * k                    # x 10^{-23} J
  beta = 1 / kT                 # x 10^{23} J^{-1}
  elm_chg = 1.602176487         # x 10^{-19} C
  epsilon_0 = 8.854187817       # x 10^{-12} C^2 J^{-1} m^{-1}

  # distance conversion factors
  conv_LtoA3 = 1.0e27 
  conv_A3toL = 1.0e-27 
  conv_mtoA = 1.0e10
  conv_nmtoA = 1.0e1
  conv_pmtoA = 1.0e-2
  conv_Atom = 1.0e-10
  conv_molLtoparA3 = Na * 1e-4  # mol/L -> particle density, particle / A^3
  conv_parA3tomolL = 1e4/Na     # particle density, particle / A^3 -> mol/L

  # energy conversion factors
  conv_kcalmoltokT = 4184 / (k * Na * T)
  conv_kJmoltokT = conv_kcalmoltokT / 4.184
  conv_eVtokT = Na * 1.60217646 * 1e1 * conv_kJmoltokT

  # other factors
  # number to be used instead of infinity in cases where inf makes problem
  # i.e., 
  #     matplotlib doesn't support display of inf
  #     diff() of [inf, inf, 0, 0] returns [nan,-inf,0] => nan and -inf are a problem
  #         in the evaluation!
  # let's put an arbitrary value, just big enough
  inf_value = 500000.0
  nan_value = 0.0
  
  def __init__(self, T = 300.0, epsilon_r = 78.3):
    # define constants connected with temperature
    self.T = T
    self.epsilon_r = epsilon_r
    self.kT = self.T * self.k
    self.beta = 1 / self.kT
    self.conv_kcalmoltokT = 4184 / (self.k * self.Na * self.T) 
    self.conv_kJmoltokT = self.conv_kcalmoltokT / 4.184
    #self.coul_factor = 1 / (4 * pi * self.T)
    self.bjerrum_length = (self.elm_chg)**2 * 1.0e7 / (4 * pi * self.epsilon_0 * self.epsilon_r * self.kT)

  def print_const(self):
    # for use with print(instance)
    print("\tconstants and factors")
    print("\ttemperature \t\t\t%f K" % self.T)
    print("\trelative permittivity \t\t%f" % self.epsilon_r)
    print("\tBjerrum length \t\t\t%f A" % self.bjerrum_length)
    print("")

    print("\tAvogadro constant \t\t%e 1/mol" % (self.Na * 1e23))
    print("\tBoltzmann constant \t\t%e J/K" % (self.k * 1e-23))
    print("\tkT \t\t\t\t%e J" % (self.kT * 1e-23))
    print("\t1/kT = beta \t\t\t%e 1/J" % (self.beta * 1e23))
    print("\telementary charge \t\t%e C" % (self.elm_chg * 1e-19))
    print("\tvacuum permittivity \t\t%e C^2/Jm" % (self.epsilon_0 * 1e-12))
    print("")
    
    print("\tconversion kcal/mol to kT \t%f" % self.conv_kcalmoltokT)
    print("\tconversion kJ/mol to kT \t%f" % self.conv_kJmoltokT)
    print("\tconversion eV to kT\t\t%f" % self.conv_eVtokT)
    #print("")

#    print("\tFT prefactor \t\t\t%f" % self.fft_prefactor)
#    print("\tiFT prefactor \t\t\t%f" % self.ift_prefactor)
    print("")
    
# ************************************************************************************************

class ctrl_defaults:
  # defaults for ctrl part of the input file
  
  # use graphics?
  do_graphics = False
  # plot minimum and maximum x value
  graphics_p_xmin = 0.0
  graphics_p_xmax = 10.0
  
  # number of discretization points
  npoints_exp = 12
  npoints = 2**npoints_exp - 1
  
  # step in real space
  deltar = 0.05
   
  # mixing parameter
  mix_param = 0.4
  # convergence criterion
  convergence_crit = 1e-9
  # maximum number of iterations
  max_iter = 1000
  
  # maximum dsqn
  max_dsqn = 100.0
  
  # Newton-Raphson/CG related stuff
  do_nr = False
  
  # fraction of B to be used as starting value of X in the NR procedure
  nr_mix_param = 0.0
  
  # relative convergence factor
  # the NR/CG cycle is taken as converged, when the difference in two successive X drops below
  # nr_convergence_factor * DSQN (of the main cycle)
  nr_convergence_factor = 0.1
  
  # max number of iterations in the inner NR/CG cycle
  nr_max_iter = 10

  # use non-converged dgam as increment if nr_max_iter has been reached? 
  # if false, the Picard iteration will be used instead
  nr_noconv_incr = False

# ************************************************************************************************

class syst_defaults:
  # defaults for syst part of the input file
  
  # temperature
  temp = 300.0
  # epsilon_r
  epsilon_r = 78.3
  # closure relation
  closure_name = "hnc"
  # concentration in moles/L
  conc_unit = "mol_l"
  # concentration in particles/A3
  #conc_unit = "part_a3"

# ************************************************************************************************

class parm_defaults:
  # defaults for individual potentials
  
  # hard spheres 
  # distance unit - a (angstrom), m, nm, pm
  hs_unit = 'A'
  hs_title = 'hs'

  # coulomb potential
  coul_title = 'coulomb'
  # alpha parameter for the error function (Ng formalism)
  #erf_alpha = None
  # minimal kappa value - empirically established
  erf_min_kappa = 0.05
  
  # charge-induced dipole interaction
  chg_inddip_title = 'chg-dip'
  
  # lennard-jones
  lj_title = 'lj'
  # sigma unit - a (angstrom), m, nm, pm
  lj_sigma_unit = 'A'
  # epsilon unit - kt, ev, kj_mol, kcal_mol
  lj_epsilon_unit = 'kT'
  # combination rules
  lj_sigma_rule = 'arit'
  lj_epsilon_rule = 'geom'

  # external pmf
  pmf_title = 'pmf'
  # pmf unit - kt, ev, kj_mol, kcal_mol
  pmf_unit = 'kT'
  # pmf distance unit - a (angstrom), m, nm, pm
  pmf_distance_unit = 'A'
  # interpolation scheme - linear or cosine
  pmf_interp_type = 'linear'
  # values to be added before read-in data and after them
  # for r->0
  pmf_before = inf
  # for r->infinity
  pmf_after = 0.0

# ************************************************************************************************

class outp_defaults:
  # defaults for outp part of the input file
  
  # default name
  name = "pyoz"
  gamma_suffix = '-gamma.dat'
  gr_suffix = '-gr.dat'
  ur_suffix = '-ur.dat'
  urtot_suffix = '-urtot.dat'
  cr_suffix = '-cr.dat'
  S_suffix = '-sk.dat'
  
  # Gamma related stuff
  # write output?
  G_ij_write = False
  # filename
  G_ij_name = name + gamma_suffix
  # save to file frequence (in iterations)
  G_ij_savefreq = 25
  # use binary data
  G_ij_binary = True
  
  # g(r) related stuff
  # write?
  g_ij_write = False
  # filename
  g_ij_name = name + gr_suffix
  # save to file frequence (in iterations)
  #g_ij_freq = 25

  # U(r) related stuff
  # write?
  U_ij_write = False
  # filename
  U_ij_name = name + ur_suffix
  # write U_ij + Gamma?
  Utot_ij_write = False
  # filename
  Utot_ij_name = name + urtot_suffix
  
  # c(r) related stuff
  # write?
  c_ij_write = False
  # filename
  c_ij_name = name + cr_suffix

  # S(k) related stuff
  # write?
  S_ij_write = False
  # filename
  S_ij_name = name + S_suffix

# ************************************************************************************************
  
if __name__ == "__main__":
  print __doc__
  inst = const()
  inst.print_const()

