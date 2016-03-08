#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this file is part of the pyOZ bundle
# pyOZ is a solver of the Ornstein-Zernike equation written in Python
# pyOZ is released under the BSD license
# see the file LICENSE for more information

"""
Module defining plot-related functions for the program

first preliminary version, crude parameters, ...

revision of this file: 0.1.1
"""

# python modules
import pylab
from scipy import isnan, isinf, copy, zeros


# **********************************************************************************************  

# global variables
# placeholders for individual plots
p_U_ij = []; p_G_ij = []; p_c_ij = []; p_g_ij = [];

# **********************************************************************************************  

def plot_initialize(ctrl, syst, const, r):
  print("initializing graphic subsystem\n")
  # functions to plot
  # interaction potential U, resp. PMF + erf correction
  # Gamma(r)
  # c(r) + c_f(r)
  # g(r)
  
  pylab.ion()

  # prepare the plots in order to avoid re-scaling of axes

  # U_ij
  pylab.subplot(2,2,1, autoscale_on=False)
  pylab.xlim((ctrl['graphics_p_xmin'], ctrl['graphics_p_xmax']))
  pylab.ylim((-5,10))
  for i in range(syst['ncomponents']):
    for j in range(i,syst['ncomponents']):
      p_U_ij.append(pylab.plot(r, zeros(ctrl['npoints']))[0])
      p_U_ij[-1].set_label("U(%s,%s)" % (syst['name'][i], syst['name'][j]))
  pylab.legend()
  # erf correction
  for i in range(syst['ncomponents']):
    for j in range(i,syst['ncomponents']):
      p_U_ij.append(pylab.plot(r, zeros(ctrl['npoints']))[0])
      p_U_ij[-1].set_label("Uerf(%s,%s)" % (syst['name'][i], syst['name'][j]))
  pylab.legend()
  
  # Gamma function
  pylab.subplot(2,2,2, autoscale_on=False)
  pylab.xlim((ctrl['graphics_p_xmin'], ctrl['graphics_p_xmax']))
  pylab.ylim((-1,1))
  for i in range(syst['ncomponents']):
    for j in range(i,syst['ncomponents']):
      p_G_ij.append(pylab.plot(r, zeros(ctrl['npoints']))[0])
      p_G_ij[-1].set_label("Gams(%s,%s)" % (syst['name'][i], syst['name'][j]))
  pylab.legend()

  # direct correlation function c
  pylab.subplot(2,2,3, autoscale_on=False)
  pylab.xlim((ctrl['graphics_p_xmin'], ctrl['graphics_p_xmax']))
  pylab.ylim((-3,3))
  for i in range(syst['ncomponents']):
    for j in range(i,syst['ncomponents']):
      p_c_ij.append(pylab.plot(r, zeros(ctrl['npoints']))[0])
      p_c_ij[-1].set_label("cs(%s,%s)" % (syst['name'][i], syst['name'][j]))
  pylab.legend()
  for i in range(syst['ncomponents']):
    for j in range(i,syst['ncomponents']):
      p_c_ij.append(pylab.plot(r, zeros(ctrl['npoints']))[0])
      p_c_ij[-1].set_label("cf(%s,%s)" % (syst['name'][i], syst['name'][j]))
  pylab.legend()

  # pair correlation function g
  pylab.subplot(2,2,4, autoscale_on=False)
  pylab.xlim((ctrl['graphics_p_xmin'], ctrl['graphics_p_xmax']))
  pylab.ylim((-0.5,10))
  for i in range(syst['ncomponents']):
    for j in range(i,syst['ncomponents']):
      p_g_ij.append(pylab.plot(r, zeros(ctrl['npoints']))[0])
      p_g_ij[-1].set_label("g(%s,%s)" % (syst['name'][i], syst['name'][j]))
  pylab.legend()
  
  pylab.draw()

# **********************************************************************************************  
  
def plot_update(syst, const, U_r = None, U_erf = None, G_r = None, c_r = None, g_r = None, c_f = None):
  # update the plotted data
  # check for the minimum/maximum and update axes?

  # with U_r we can have a problem (it can contain Inf and NaN)
  # that doesn't work with pylab
  # replace it with some other value
  if (U_r != None):
    U_plot = copy(U_r)
    U_plot[isinf(U_plot)] = const.inf_value
    U_plot[isnan(U_plot)] = const.nan_value

  # it's necessary to create deep copies of objects that are subject to change
  # i.e., G_r, c_r, c_f - they are all multiplied/divided by some factors. this can
  # lead to problems with display, since pylab.draw() is redrawing all objects
  # if
  #     set_ydata(c_r_ij)
  #     draw()
  # is used, then pylab obtains reference to c_r_ij. when in the program c_r_ij is multiplied by r
  # then a later invocation of
  #     set_ydata(G_r_ij)
  #     draw()
  # will display updated G_r, but also updated c_r - and we don't want that!
  subplot_number = 0
  for i in range(syst['ncomponents']):
    for j in range(i,syst['ncomponents']):
      if (U_r != None): p_U_ij[subplot_number].set_ydata(U_plot[i,j])
      if (G_r != None): p_G_ij[subplot_number].set_ydata(copy(G_r[i,j]))
      if (c_r != None): p_c_ij[subplot_number].set_ydata(copy(c_r[i,j]))
      if (g_r != None): p_g_ij[subplot_number].set_ydata(g_r[i,j])
      subplot_number += 1
    # end for j in range(ncomponents)
  # end for i in range(ncomponents)
      
  for i in range(syst['ncomponents']):
    for j in range(i,syst['ncomponents']):
      if (c_f != None): p_c_ij[subplot_number].set_ydata(copy(c_f[i,j]))
      if (U_erf != None): p_U_ij[subplot_number].set_ydata(U_erf[i,j])
      subplot_number += 1
    # end for j in range(ncomponents)
  # end for i in range(ncomponents)

  pylab.draw()

# **********************************************************************************************  

if __name__ == "__main__":
  print __doc__
  print("Usage as a standalone application/script is not supported at the moment")

