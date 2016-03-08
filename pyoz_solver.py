#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this file is part of the pyOZ bundle
# pyOZ is a solver of the Ornstein-Zernike equation written in Python
# pyOZ is released under the BSD license
# see the file LICENSE for more information

"""
Module defining solver-related functions
revision of this file: 0.1.2

currently contained are solvers for 1,2,3,n components:
  solver_1
    tested and used
  solver_2
    tested and used
  solver_3
    not tested, linalg.solve tested instead
  solver_n
    tested and used

we are solving the matrix problem in fourier space
note that the convolution theorem involves a constant factor ('ff') depending on the used forward fourier transform normalization constant

H = C + ff CH
H - ff CH = C
{E - ff C}H = C
H = {E - ff C}^-1 * C
h = H / dens_factor

inputs for the solver functions are
(E-aC) = a, C = b, np - number of discretization points and dens_factor - matrix with density factors
"""

# python modules
from sys import exit
# mathematical functions
# linalg from numpy is faster!
from numpy import linalg
from scipy import empty_like, mat, ones

# **********************************************************************************************

# one component - simple
def solver_1(a, b, np):
  """
    solver optimized for "1x1 matrices" and dr discretization points
  """
  # check if a is nonzero everywhere (to avoid dividing by zero)
  if (a == 0.).any():
    print("singular matrix, cannot invert: (1-C) contains zero term")
    exit(1)
  
  # do the calculation and return immediately
  # we do not need the np here
  #return((b/a)/dens_factor)
  return(b/a)

# two components
def solver_2(a, b, np):
  """
    solver optimized for 2x2 matrices and dr discretization points
  """
  # prepare the array for the resulting function
  h = empty_like(a)
  
  # calculate the determinant
  a_det = a[0,0]*a[1,1] - a[1,0]*a[0,1]
  if (a_det == 0.0).any():
    print("singular matrix, cannot invert: determinant of (1-C) matrix is zero for dr=%u" % dr)
    exit(1)    

  # perform the calculation for every discretization point
  for dr in range(np):
    # calculate the inverse
    a_inv = ones((2,2,)) / a_det[dr]
    a_inv[0,0] *=        a[1,1,dr]
    a_inv[0,1] *= -1.0 * a[0,1,dr]
    a_inv[1,0] *= -1.0 * a[1,0,dr]
    a_inv[1,1] *=        a[0,0,dr]
    
    # do the calculation
    # using matrix algebra from scipy/numpy
    #h[:,:,dr] = (mat(a_inv[dr]) * mat(b[:,:,dr])) / dens_factor
    # explicitly - is faster
    h[0,0,dr] = (a_inv[0,0]*b[0,0,dr] + a_inv[0,1]*b[1,0,dr])
    h[0,1,dr] = (a_inv[0,0]*b[0,1,dr] + a_inv[0,1]*b[1,1,dr])
    h[1,0,dr] = (a_inv[1,0]*b[0,0,dr] + a_inv[1,1]*b[1,0,dr])
    h[1,1,dr] = (a_inv[1,0]*b[0,1,dr] + a_inv[1,1]*b[1,1,dr])
    
  return(h)

# 3 components
def solver_3(a, b, np):
  """
    solver optimized for "3x3 matrices" and dr discretization points
  """
  # prepare the array for the resulting function
  h = empty_like(a)
  
  # calculate the determinant
  a_det = a[0,0]*(a[1,1]*a[2,2] - a[2,1]*a[1,2]) - a[0,1]*(a[1,0]*a[2,2] - a[2,0]*a[1,2]) + a[0,2]*(a[1,0]*a[2,1] - a[2,0]*a[1,1])
  if (a_det == 0.0).any():
    print("singular matrix, cannot invert: determinant of (1-C) matrix is zero for dr=%u" % dr)
    exit(1)

  # perform the calculation for every discretization point
  for dr in range(np):
    # calculate the inverse
    a_inv = ones((3,3)) / a_det[dr]
    # the indexes in a_inv already correspond to transposed matrix!
    a_inv[0,0] *=        (a[1,1,dr]*a[2,2,dr] - a[2,1,dr]*a[1,2,dr])
    a_inv[1,0] *= -1.0 * (a[1,0,dr]*a[2,2,dr] - a[2,0,dr]*a[1,2,dr])
    a_inv[2,0] *=        (a[1,0,dr]*a[2,1,dr] - a[2,0,dr]*a[1,1,dr])
    a_inv[0,1] *= -1.0 * (a[0,1,dr]*a[2,2,dr] - a[2,1,dr]*a[0,2,dr])
    a_inv[1,1] *=        (a[0,0,dr]*a[2,2,dr] - a[2,0,dr]*a[0,2,dr])
    a_inv[2,1] *= -1.0 * (a[0,0,dr]*a[2,1,dr] - a[2,0,dr]*a[0,1,dr])
    a_inv[0,2] *=        (a[0,1,dr]*a[1,2,dr] - a[1,1,dr]*a[0,2,dr])
    a_inv[1,2] *= -1.0 * (a[0,0,dr]*a[1,2,dr] - a[1,0,dr]*a[0,2,dr])
    a_inv[2,2] *=        (a[0,0,dr]*a[1,1,dr] - a[1,0,dr]*a[0,1,dr])
  
    # do the calculation
    # using matrix algebra from scipy/numpy
    h[:,:,dr] = (mat(a_inv) * mat(b[:,:,dr]))
    # explicitly - might be faster, but is not!
    #h[0,0,dr] = (a_inv[0,0]*b[0,0,dr] + a_inv[0,1]*b[1,0,dr] + a_inv[0,2]*b[2,0,dr]) / dens_factor[0,0]
    #h[0,1,dr] = (a_inv[0,0]*b[0,1,dr] + a_inv[0,1]*b[1,1,dr] + a_inv[0,2]*b[2,1,dr]) / dens_factor[0,1]
    #h[0,2,dr] = (a_inv[0,0]*b[0,2,dr] + a_inv[0,1]*b[1,2,dr] + a_inv[0,2]*b[2,2,dr]) / dens_factor[0,2]
    #h[1,0,dr] = (a_inv[1,0]*b[0,0,dr] + a_inv[1,1]*b[1,0,dr] + a_inv[1,2]*b[2,0,dr]) / dens_factor[1,0]
    #h[1,1,dr] = (a_inv[1,0]*b[0,1,dr] + a_inv[1,1]*b[1,1,dr] + a_inv[1,2]*b[2,1,dr]) / dens_factor[1,1]
    #h[1,2,dr] = (a_inv[1,0]*b[0,2,dr] + a_inv[1,1]*b[1,2,dr] + a_inv[1,2]*b[2,2,dr]) / dens_factor[1,2]
    #h[2,0,dr] = (a_inv[2,0]*b[0,0,dr] + a_inv[2,1]*b[1,0,dr] + a_inv[2,2]*b[2,0,dr]) / dens_factor[2,0]
    #h[2,1,dr] = (a_inv[2,0]*b[0,1,dr] + a_inv[2,1]*b[1,1,dr] + a_inv[2,2]*b[2,1,dr]) / dens_factor[2,1]
    #h[2,2,dr] = (a_inv[2,0]*b[0,2,dr] + a_inv[2,1]*b[1,2,dr] + a_inv[2,2]*b[2,2,dr]) / dens_factor[2,2]
    
  return(h)

# n components - let's use numpy's linalg.solve function
def solver_n(a, b, np):
  """
    solver optimized for nxn matrices and dr discretization points
  """
  
  # create the function that will be returned
  h = empty_like(a)
  
  for dr in range(np):
    # solve the matrix problem for all dr
    # and divide by the density prefactor
    # remember that the zero elements in syst['dens']['ij'] were replaced by 1.0
    # in order to avoid numerical problem in further division by this value
    h[:,:,dr] = linalg.solve(a[:,:,dr], b[:,:,dr])

  return(h)

# **********************************************************************************************  

if __name__ == "__main__":
  print __doc__
  print("Usage as a standalone application/script is not supported at the moment")
  from scipy import array
  a1 = array([[4,-2,0],[-1,5,-2],[3,1,0]])
  a = ones((3,3,5))
  b1 = array([[5,5,5], [-7,-7,-7], [3,3,3]])
  b = ones((3,3,5))
  
  
  for i in range(5):
    a[:,:,i] = a1
    b[:,:,i] = b1
  test = solver_3(a, b, 5,ones((3,3)))
  print(test)
  
