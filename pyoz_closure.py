#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this file is part of the pyOZ bundle
# pyOZ is a solver of the Ornstein-Zernike equation written in Python
# pyOZ is released under the BSD license
# see the file LICENSE for more information

"""
Module defining closure relations
revision of this file: 0.1.1

currently supported:
  hnc - HyperNetted Chain
  py - Percus-Yevick
  dummy - dummy function
  
in the whole program (except where stated otherwise), the numpy/scipy arrays are used. as
a result, all mathematical operations are done element-wise. this applies also for multiplication
of vectors and arrays.
"""

# python modules
# mathematical functions
from scipy import exp
# other functions
from sys import exit

# **********************************************************************************************  

# all supported closures are defined below
# don't forget to modify 
#   supported_closures = {'hnc' : closure_hnc, 'dummy' : closure_dummy}
# at the end of file - need to do it in this way - declaration only is not supported (?)
# or at least i don't know how to do it...

# **********************************************************************************************  

def closure_hnc(syst, r, modMayerFunc, U_discontinuity, G_r_ij):
  """the HNC closure, calculates the direct correlation function from the pair potential and
     the gamma function; also the total and pair correlation functions
     
     according to HNC
     g_ij = modMayerFunc*exp(G_r_ij)
     h_ij = g_ij - 1
     c_ij = modMayerFunc*exp(G_r_ij) - G_r_ij - 1
     
     the discontinuities of the interaction potentials are taken care of here as well
     with help of the U_discontinuity list
  """

  #print("\tHNC closure")
  gamma_term = exp(G_r_ij)

  # calculate the pair correlation function
  g_r_ij = modMayerFunc['u_ij'] * modMayerFunc['erf'] * gamma_term

  # treat the discontinuities
  # index of the potential in the parm array (index_pot) is not used here
  for discontinuity in U_discontinuity:
    if (discontinuity != None):
      for i in range(syst['ncomponents']):
        for j in range(syst['ncomponents']):
          # get the parameters of the discontinuity
          dr, left_neigh, right_neigh = discontinuity[i][j]
          # and apply it
          g_r_ij[i, j, dr] *= 0.5 * (exp(-left_neigh) + exp(-right_neigh))
  # end for disc in U_discontinuity
  
  c_r_ij = g_r_ij - G_r_ij - 1

  return(c_r_ij, g_r_ij)

# **********************************************************************************************  

def closure_py(syst, r, modMayerFunc, U_discontinuity, G_r_ij):
  """the PY closure, calculates the direct correlation function from the pair potential and
     the gamma function; also the total and pair correlation functions
     
     according to PY
     g_ij = modMayerFunc*(1 + G_r_ij)
     h_ij = g_ij - 1
     c_ij = modMayerFunc*(1 + G_r_ij) - G_r_ij - 1
     
     the discontinuities of the interaction potentials are taken care of here as well
     with help of the U_discontinuity list
  """

  #print("\tPY closure")
  gamma_term = 1.0 + G_r_ij

  g_r_ij = modMayerFunc['u_ij'] * modMayerFunc['erf'] * gamma_term

  # treat the discontinuities
  # index of the potential in the parm array (index_pot) is not used here
  for discontinuity in U_discontinuity:
    if (discontinuity != None):
      for i in range(syst['ncomponents']):
        for j in range(syst['ncomponents']):
          # get the parameters of the discontinuity
          dr, left_neigh, right_neigh = discontinuity[i][j]
          # and apply it
          g_r_ij[i, j, dr] *= 0.5 * (exp(-left_neigh) + exp(-right_neigh))
  # end for disc in U_discontinuity
  
  c_r_ij = g_r_ij - G_r_ij - 1
  
  return(c_r_ij, g_r_ij)

# **********************************************************************************************  

def closure_dummy(syst, r, modMayerFunc, U_discontinuity, G_r_ij):
  """dummy closure function
     returns zero arrays
  """
  
  #print("\tdummy closure")
  from scipy import zeros_like
  g_r_ij = zeros_like(G_r_ij)
  c_r_ij = zeros_like(G_r_ij)
  
  return(c_r_ij, g_r_ij)
  
# **********************************************************************************************  

def calcGammaTerm(syst, G_r_ij):
  """calculate the Gamma contribution to g(r), c(r) according to the closure relation
  """
  
  if (syst['closure_name'] == 'hnc'):
    return(exp(G_r_ij))
  elif (syst['closure_name'] == 'py'):
    return(1.0 + G_r_ij)
  else:
    print('unknown closure!')
    exit(2)

# **********************************************************************************************  

supported_closures = {'hnc' : closure_hnc, 'py' : closure_py, 'dummy' : closure_dummy}

if __name__ == "__main__":
  print __doc__
  print("Usage as a standalone application/script is not supported at the moment")

