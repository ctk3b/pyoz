#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this file is part of the pyOZ bundle
# pyOZ is a solver of the Ornstein-Zernike equation written in Python
# pyOZ is released under the BSD license
# see the file LICENSE for more information

# for the details on the sine transform code itself, 
# please see the documentation of the respective routine (around line 240)

"""
  module for performing various types of discrete transforms
  revision of this file: 0.1.4
  
  for the 3D-Fourier transformation, there is a normalization factor of 1/(2*pi)^3 involved
  then, the forward FT can be normalized  
    ft_3d_forward = 1/(2pi)^3
    ft_3d_inverse = 1
  or both FTs can be normalized, yielding unitary transformation
    ft_3d_forward = sqrt(1/(2pi)^3)
    ft_3d_inverse = sqrt(1/(2pi)^3)
  or the inverse FT can be normalized
    ft_3d_forward = 1
    ft_3d_inverse = 1/(2pi)^3
  
  when moving from 3d-fourier transform to fourier-bessel transformation (FBT), there's a factor
  of 4pi involved for both forward and inverse transforms
  the factor can probably be distributed 'freely' as well
  both factors within forward transformation
    ft_fb_forward = 16pi^2
    ft_fb_inverse = 1  
  factor distributed among forward and inverse transform 
    ft_fb_forward = 4pi
    ft_fb_inverse = 4pi
  or both factors within inverse transformation
    ft_fb_inverse = 16pi^2  
    ft_fb_forward = 1
  
  it's also important to note that
    F = FBT(f) = ft_3d_forward * ft_fb_forward * 1/k int_0^\infty f(r) r sin(kr) dr
    f = iFBT(F) = ft_3d_inverse * ft_fb_inverse * 1/r int_0^\infty f(k) k sin(kr) dr
  this can be written in terms of a Fourier-sinus tranform (FST)
    G = FST(g) = ft_fs_forward * int_0^\infty g(r)  sin(kr) dr
    g = iFST(G) = ft_fs_inverse int_0^\infty g(k)  sin(kr) dk
  i.e., we define new functions g = r*f and G=k*F and perform the FST, then divide by the k (forward)
  or r (inverse) and we have the FBT/iFBT result. this approach is used because the FST is easily
  performed using the standard FFT. one has to take care of the different pre-factors, of course.
  
  the continuous sinus transformations (forward and inverse) are normalized to 2/pi
  then, the forward FST can be normalized  
    ft_fs_forward = 2/pi
    ft_3d_inverse = 1
  or both FTs can be normalized, yielding unitary transformation
    ft_3d_forward = sqrt(2/pi)
    ft_3d_inverse = sqrt(2/pi)
  or the inverse FT can be normalized
    ft_3d_forward = 1
    ft_3d_inverse = 2/pi
  
  the following text is DST implementation dependent! it's necessary to check the implementation
  in the respective scientific library! use test_dst.py
  
  employed discrete sinus transformations routines are themselves normalized to the number of points
  i.e., there aren't any factors necessary => iDST(DST(f)) = f
  
  however, DST(f) is not the complete fourier-sinus transformation. it's necessary to 
  multiply it with the step size in real space, i.e., 
    F' = FST'(f) = DST(f) * dr
  where F' or FST' is the fourier-sinus transformation of f using discretely sampled data
  in order to get an equivalent of the exact continuous sinus transformation, we need
  to multiply with the respective constant factor
    F = FST(f) = ft_fs_forward * DST(f) * dr
  the same applies to the inverse transformation - it's necessary to multiply
  it with the step size in reciprocal space, i.e.,
    f' = iFST'(F) = iDST(F) * dk
  again, in order to get an equivalent of the exact continuous inverse sinus transformation,
  we have to multiply with the constant prefactor
    f = iFST(F) = ft_fs_inverse * iDST(F) * dr
  at this moment, it's clear that
    x = iFST(FST(f)) != f
  because
    x = ft_fs_inverse * dk * iDST(ft_fs_forward * DST(f) * dr) =
      = ft_fs_inverse * ft_fs_forward * dr * dk * f
  so at this point, we still don't have the correct FST/iFST pair
  in this program, dk = pi / (N dr), and therefore
    x = ft_fs_inverse * ft_fs_forward * dr * pi / (N * dr) * f = 
      = ft_fs_inverse * ft_fs_forward * pi * f / N
  this means, that although the discrete sinus transformation routine is normalized, the introduction of
  the dr, dk factors brings the number of points back into play.
  when the DST and iDST routines are checked against the fourier-sinus transform of a function where the
  solution is analytically known, the following information emerges.
  the function is, for example
    g = exp(-a * r**2) * sin(b * r)
  with the analytical FST
    G = exp(-1.0 * (k**2 + b**2)/(4.0 * a)) * sinh(b * k / (2.0*a)) / sqrt(2.0*a)
  where the unitary FST was used (sqrt(2/pi) for both forward and inverse transform.
  the result of the numerical procedure (as described above) has to be multiplied by the same FST prefactor 
  in order to get the same function
    sqrt(2/pi) * F = sqrt(2/pi) * FST(f) = sqrt(2/pi) * DST(f) * dr = G
  in order to get the f=g back, we need to multiply by the iFST prefactor and then take care of the N factor. as
  described above, the function we get in this way is
    x = ft_fs_inverse * ft_fs_forward * pi * f / N = 2/pi * pi * f / N = 2f/N
  therefore, we have to further multiply the result with N/2.
  we can, therefore, conclude that
  F = DST(f) r
  f = iDST(F) k N/2
  the factor of N/2 stays the same whatever normalization combination we use
"""

# numpy contains the fft, that is used for the private dst and idst routines
import numpy as np
from numpy import array, zeros, zeros_like, imag, real, float64, complex128
from numpy import fft as np_fft

# constants/functions needed for the normalization factors
from math import pi, sqrt

# **********************************************************************************************  

# let's define the normalization constants as described above
# these are private

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ! do not change the definitions since the program relies on the selection made here !
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 3D Fourier transform
# normalization within the forward transformation
#_ft_3d_forward = 1/(2.0 * pi)**3
#_ft_3d_inverse = 1.0
# unitary
#_ft_3d_forward = sqrt(1/(2.0 * pi)**3)
#_ft_3d_inverse = sqrt(1/(2.0 * pi)**3)
# normalization within the inverse transformation
_ft_3d_forward = 1.0
_ft_3d_inverse = 1/(2.0 * pi)**3

# 3D FT -> Fourier-Bessel transform
# both 4pi factors within the forward transformation
#_ft_fb_forward = 16.0 * pi**2
#_ft_fb_inverse = 1.0
# unitary
_ft_fb_forward = 4.0 * pi
_ft_fb_inverse = 4.0 * pi
# both 4pi factors within the inverse transformation
#_ft_fb_forward = 1.0
#_ft_fb_inverse = 16.0 * pi**2

# Fourier-sinus transform
# normalization within the forward transformation
#_ft_fs_forward = 2.0/pi
#_ft_fs_inverse = 1.0
# unitary
#_ft_fs_forward = sqrt(2.0/pi)
#_ft_fs_inverse = sqrt(2.0/pi)
# normalization within the inverse transformation
#_ft_fs_forward = 1.0
#_ft_fs_inverse = 2.0/pi
# in this DST/iDST implementation, no factor is used!
_ft_fs_forward = 1.0
_ft_fs_inverse = 1.0

# **********************************************************************************************  

class dft:
  """
    class for discrete fourier transforms
    includes the constants involved together with the procedures

    r, k, deltar, deltak, npoints will have to be entered - self.npoints/pi
        should be then called Npointscorr or something like that
    at the moment, leave it as it is
  """
  npoints = 0
  dr = 0.0
  dk = 0.0
  r = None
  k = None
  fft_prefactor = 1.0
  ift_prefactor = 1.0
  # we are making a FT of a convolution f*g
  # it's known that there is an additional constant factor involved, equal to 
  # the reciprocal of the forward fourier transform normalization factor
  # we need to access this factor from the solver routine
  # due to the chosen parameters, this is equal 1.0 at the moment
  ft_convolution_factor = 1.0/_ft_3d_forward

  def __init__(self, npoints, dr, dk, r, k):
    self.npoints = npoints
    self.dr = dr
    self.dk = dk
    self.r = r
    self.k = k
    
    # when FST is used in order to replace the FBT, it comes with no prefactor. therefore,
    # any prefactor already present has to be canceled
    # since the prefactor in this case is 1.0, we can skip it
    # care is taken here of the dr*dk product and the normalization of the continuous FST
    #self.fft_prefactor = _ft_3d_forward * _ft_fb_forward / ft_fs_forward 
    self.fft_prefactor = _ft_3d_forward * _ft_fb_forward 
    #self.ift_prefactor = _ft_3d_inverse * _ft_fb_inverse * (npoints / 2.0) / ft_fs_inverse
    self.ift_prefactor = _ft_3d_inverse * _ft_fb_inverse * (npoints / 2.0)
    
  def print_status(self):
    """
      print out the status of the dft class - constants, ...
    """
    print("\tFT set up for\t\t\t%u points" % self.npoints)
    print("\tFT prefactor \t\t\t%f" % self.fft_prefactor)
    print("\tiFT prefactor \t\t\t%f" % self.ift_prefactor)
    print("\tfactor involved in convolution\t%f" % self.ft_convolution_factor)
  # end def print_status()
    
  def dfbt(self, f, norm=1.0, corr=0.0):
    """
      perform the discrete Fourier-Bessel transform of the input function f
      
      we replace the classical 3D Fourier transform by a Fourier-Bessel transform,
      which in turn is evaluated using the Fourier-sinus transform
      
      note that there are some extra factors/operations involved
      
      if corr (correction) is defined, it is added to the transformed function
      if norm (normalization) is defined, the resulting function is multiplied by it
      
      see the text above and the documentation for more information
    """
    # FT(f) can be calculated as DST(f*r)dr/k
    F = self.fft_prefactor * _dst(f * self.r) * self.dr / self.k
    
    # apply the normalization
    F *= norm

    # apply the correction (already includes sqrt(rho_i rho_j)
    Fc = F + corr

    # return both corrected and uncorrected functions
    return(F, Fc)
  # end def dft()
    
  def idfbt(self, F, norm=1.0, corr=0.0):
    """
      perform the inverse discrete Fourier-Bessel transform of the input function F
      
      we replace the classical 3D Fourier transform by a Fourier-Bessel transform,
      which in turn is evaluated using the Fourier-sinus transform
      
      note that there are some extra factors/operations involved

      if norm (normalization) is defined, the resulting function is divided by it
      in case of zero value, the resulting function is set to zero, or to corr (if defined)

      see the text above and the documentation for more information
    """
    if (norm == 0.0):
      # infinite dilution
      f = zeros_like(F) + corr
    else:
      f = (self.ift_prefactor * _idst(F * self.k) * self.dk / self.r) / norm 

    return(f)
  # end def idft()
    
# **********************************************************************************************  

# DST-I according to wiki
#
# code is coming from the scipy sandbox
# http://projects.scipy.org/scipy/scipy/browser/trunk/scipy/sandbox/image/transforms.py
#
# luc's version is also present (padding the function to twice as many samples
# with zeroes. the formal correctness of luc's code has to be checked

def _dst(x,axis=-1):
    """Discrete Sine Transform (DST-I)

    Implemented using 2(N+1)-point FFT
    xsym = r_[0,x,0,-x[::-1]]
    DST = (-imag(fft(xsym))/2)[1:(N+1)]

    adjusted to work over an arbitrary axis for entire n-dim array
    """
    n = len(x.shape)
    N = x.shape[axis]
    slices = [None]*3
    for k in range(3):
        slices[k] = []
        for j in range(n):
            slices[k].append(slice(None))
    newshape = list(x.shape)
    newshape[axis] = 2*(N+1)
    xtilde = zeros(newshape,float64)
    slices[0][axis] = slice(1,N+1)
    slices[1][axis] = slice(N+2,None)
    slices[2][axis] = slice(None,None,-1)
    for k in range(3):
        slices[k] = tuple(slices[k])
    xtilde[slices[0]] = x
    xtilde[slices[1]] = -x[slices[2]]
    Xt = np_fft.fft(xtilde,axis=axis)
    return (-imag(Xt)/2)[slices[0]]

def _idst(v,axis=-1):
    n = len(v.shape)
    N = v.shape[axis]
    slices = [None]*3
    for k in range(3):
        slices[k] = []
        for j in range(n):
            slices[k].append(slice(None))
    newshape = list(v.shape)
    newshape[axis] = 2*(N+1)
    Xt = zeros(newshape,complex128)
    slices[0][axis] = slice(1,N+1)
    slices[1][axis] = slice(N+2,None)
    slices[2][axis] = slice(None,None,-1)
    val = 2j*v
    for k in range(3):
        slices[k] = tuple(slices[k])
    Xt[slices[0]] = -val
    Xt[slices[1]] = val[slices[2]]
    xhat = real(np_fft.ifft(Xt,axis=axis))
    return xhat[slices[0]]

# **********************************************************************************************  

# DST-I according to Luc (fill with zeros)
def _dst_luc(x,axis=-1):
    """Discrete Sine Transform (DST-I)

    Implemented using 2(N+1)-point FFT
    xsym = r_[0,x,(N+1)*0]
    DST = (-imag(fft(xsym))/2)[1:(N+1)]

    adjusted to work over an arbitrary axis for entire n-dim array
    """
    n = len(x.shape)
    N = x.shape[axis]
    slices = [None]*3
    for k in range(3):
        slices[k] = []
        for j in range(n):
            slices[k].append(slice(None))
    newshape = list(x.shape)
    newshape[axis] = 2*(N+1)
    xtilde = zeros(newshape,float64)
    slices[0][axis] = slice(1,N+1)
    slices[1][axis] = slice(N+2,None)
    slices[2][axis] = slice(None,None,-1)
    for k in range(3):
        slices[k] = tuple(slices[k])
    xtilde[slices[0]] = x
    # the following is not used here
    # instead it's replaced by zeros
    #xtilde[slices[1]] = -x[slices[2]]
    Xt = np_fft.fft(xtilde,axis=axis)
    #return (-imag(Xt)/2)[slices[0]]
    return (-imag(Xt))[slices[0]]

def idst_luc(v,axis=-1):
    n = len(v.shape)
    N = v.shape[axis]
    slices = [None]*3
    for k in range(3):
        slices[k] = []
        for j in range(n):
            slices[k].append(slice(None))
    newshape = list(v.shape)
    newshape[axis] = 2*(N+1)
    Xt = zeros(newshape,complex128)
    slices[0][axis] = slice(1,N+1)
    slices[1][axis] = slice(N+2,None)
    slices[2][axis] = slice(None,None,-1)
    #val = 2j*v
    val = 4j*v
    for k in range(3):
        slices[k] = tuple(slices[k])
    Xt[slices[0]] = -val
    # the following is not used here
    # instead it's replaced by zeros
    #Xt[slices[1]] = val[slices[2]]
    xhat = real(np_fft.ifft(Xt,axis=axis))
    return xhat[slices[0]]

# **********************************************************************************************  

if __name__ == "__main__":
  print __doc__
  print("\nDST of an array [1,2,3,4,5,6,7,8,9]")
  
  a = array([1,2,3,4,5,6,7,8,9])
  b1 = dst(a)
  b2 = dst_luc(a)
  c1 = idst(b1)
  c2 = idst_luc(b2)
  
  print("version wiki")
  print(b1)
  print(c1)
  print(" ")
  
  print("version Luc")
  print(b2)
  print(c2)

