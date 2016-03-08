#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this file is part of the pyOZ bundle
# pyOZ is a solver of the Ornstein-Zernike equation written in Python
# pyOZ is released under the BSD license
# see the file LICENSE for more information

"""
Module defining miscellaneous functions for the program
revision of this file: 0.1.3

currently contained are:
  convergence_dsqn
    calculates the convergence criterion

  dotproduct
    calculates the dot product of two matrices

  interpolate_linear
  interpolate_cosine
    routines for linear and cosine interpolation of discrete data
"""

# python modules
# mathematical functions
from math import sqrt, cos, pi
# simpson numerical integration routine
# all possibilities (integration of discretized functions) are:
#   trapz         -- Use trapezoidal rule to compute integral from samples.
#   cumtrapz      -- Use trapezoidal rule to cumulatively compute integral.
#   simps         -- Use Simpson's rule to compute integral from samples.
#   romb          -- Use Romberg Integration to compute integral from
#                    (2**k + 1) evenly-spaced samples.
# some documentation is here http://www.scipy.org/doc/api_docs/SciPy.integrate.quadrature.html
from scipy.integrate import simps as int_simpson

# **********************************************************************************************  

def convergence_dsqn(ctrl, syst, G_r_ij, G_n_ij):
  """calculates the difference between old and new Gamma function
     
     it's calculated according to
     dsqn = sqrt(sum[(G_new - G_old)^2] / (npoints * ncomponents^2))
  """
  
  # calculate the square of all differences and convert to a 1D array
  norm_dsqn = ((G_r_ij - G_n_ij)**2).sum()
  # normalize to number of points * number of arrays
  norm_dsqn /=  ctrl['npoints'] * syst['ncomponents']**2
  # do sqrt()
  norm_dsqn = sqrt(norm_dsqn)

  return(norm_dsqn)

# **********************************************************************************************  

def dotproduct(ctrl, syst, r, X_ij, Y_ij):
  """calculates the dot product of two functions X, Y discretized on N points and represented
     as square matrices at every discretization point according to
  
     dotprod = sum_ij [rho_i rho_j \int X_ij(r) Y_ij(r) 4 \pi r^2 dr]
     
     with rho being constant factors stored in a square matrix of the same dimension as X(r) and Y(r)
  """
  dotprod = 0.0

  # calculate the integrand (array product)
  integrand = 4.0 * pi * X_ij * Y_ij
  
  for i in range(syst['ncomponents']):
    for j in range(syst['ncomponents']):
      integrand[i,j,:] *= syst['dens']['num'][i]*syst['dens']['num'][j] * r**2
      dotprod += int_simpson(integrand[i,j], r, dx=ctrl['deltar'], even='last')      
  
  return(dotprod)

# **********************************************************************************************  

def interpolate_linear(val1, val2, mu):
  """
    linear interpolation routine
    
    val1 and val2 are values at positions x1, x2
    target point is specified by mu
    mu=0 => x1, mu=1 => x2
  """
  return(val1 * (1.0 - mu) + val2 * mu)

def interpolate_cosine(val1, val2, mu):
  """
    cosine interpolation routine
    
    val1 and val2 are values at positions x1, x2
    target point is specified by mu
    mu=0 => x1, mu=1 => x2
  """
  mu2 = (1.0 - cos(mu * pi))/2.0;
  return(val1 * (1.0 - mu2) + val2 * mu2);

# **********************************************************************************************  

if __name__ == "__main__":
  print __doc__
  print("Usage as a standalone application/script is not supported at the moment")

