#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this file is part of the pyOZ bundle
# pyOZ is a solver of the Ornstein-Zernike equation written in Python
# pyOZ is released under the BSD license
# see the file LICENSE for more information

"""
Module defining functions for interaction potential handling
revision of this file: 0.1.4

"""

# import external modules
# 
import sys

# scipy related stuff
from scipy import exp, zeros
# 'administrative' functions
from scipy import copy, isinf
# mathematical functions
from scipy import diff, gradient
# symbols, constants, ...
from scipy import inf, nan, isnan, pi
# error function
from scipy.special import erf

# **********************************************************************************************  

# calculate the U_ij potential
def def_potential(ctrl, syst, parm, const, dft, r, k):
  """calculate the total U_ij potential and initialize the array storing its discretized form
     store individual potentials of the total interaction
     store list with discontinuities of the individual potentials
     store dictionary with ng-renormalized coulomb potential (real and fourier space)
     store a list storing numerically differentiated potentials (where needed) 
  
     this routine takes the ng-renormalization procedure into account
     the unit of energy is now kT
     
     potential parameters are stored in the 'parm' dictionary (see pyoz_input for details)
     
     currently supported potentials:
     hs (hard spheres)
     coulomb (electrostatics)
     chg_inddip (1/r^4 dependent charage-induced dipole interaction; requires coulomb!)
     lj (van der Waals with sigma and epsilon parameters)
     pmf (potential of mean force calculated by some other method)
  """
  
  # pre-initialize array with discretized total potential to zero
  U_ij = zeros((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))

  # pre-initialize the list for discretized potential contributions (hs, lj, coulomb, ...) to zero
  U_ij_individual = [ None ] * len(parm)
  for i in range(len(parm)):
    U_ij_individual[i] = zeros((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
  # list to store numerically differentiated individual potentials (where no analytical derivative is available)
  # not used at the moment
  dU_ij_individual = [ None ] * len(parm)

  # list to store discontinuities for every potential type
  U_discontinuity = [ None ] * len(parm)

  # erf-corrected 1/r potential functions (according to Ng)
  # nonzero only if coulomb potential is used
  U_erf_ij = {  'real' : zeros((syst['ncomponents'], syst['ncomponents'], ctrl['npoints'])), 
                'fourier' : zeros((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))}

  # evaluate the potentials one by one  
  for pot_index in range(len(parm)):
    potential = parm[pot_index]
    pot_type = potential['pot_type']
    
    # ****************************************

    # hard spheres potential
    if (pot_type == 'hs'):
      #print("Hard spheres potential")
      # this potential has discontinuity - prepare the array
      U_discontinuity[pot_index] = []
      # evaluate the combinations one by one
      for i in range(syst['ncomponents']):
        # create the list over first index (i)
        U_discontinuity[pot_index].append([])
        for j in range(syst['ncomponents']):
          dr = potential['dis_index'][i,j]

          # for the values up to dr - 1 (slice [a:b] goes from a to b-1)
          # will be changed to infinity
          # the rest stays zero
          # discontinuity is recorded
          # to dr = 0 corresponds the distance of deltar!
          U_ij_individual[pot_index][i,j,:dr] = inf
          U_ij[i,j,:] += U_ij_individual[pot_index][i,j,:]
          # create the list over second index and store the information at U_disc[pot_index][i][j]
          # please note that this is ordinary N-dimensional list => the indices have to be
          # treated separately!
          U_discontinuity[pot_index][i].append([dr,inf,0.0])
          #print("shapes:", U_ij_individual[pot_index].shape, U_ij.shape, len(U_discontinuity))
        # end for j in range(ncomponents)
      # end for i in range(ncomponents)
    # end if (pot_type = 'hs')  
          
    # ****************************************

    # Coulomb potential
    elif (pot_type == 'coulomb'):
      for i in range(syst['ncomponents']):
        for j in range(syst['ncomponents']):
          U_ij_individual[pot_index][i,j,:] = potential['chg_ij'][i,j] / r
          U_ij[i,j,:] += U_ij_individual[pot_index][i,j,:] 
          # modified 1/r potential function according to Ng
          # this factor is constant throughout the calculation
          U_erf_ij['real'][i,j,:] = U_ij_individual[pot_index][i,j,:] * erf(potential['erf_alpha'][i,j] * r)
          # in Fourier space, the form of this function is also known
          #U_erf_ij['fourier'][i,j,:] = dft.fft_prefactor * potential['chg_ij'][i,j] * exp( -1.0 * k**2 / (4.0 * ctrl['erf_alpha']**2)) / k**2
          #U_erf_ij['fourier'][i,j,:] = dft.fft_prefactor * potential['chg_ij'][i,j] * exp( -1.0 * k**2 / (4.0 * potential['erf_alpha'][i,j]**2)) / k**2
          # we will use the U_erf_ij multiplied by the density prefactor sqrt(rho_i rho_j)
          U_erf_ij['fourier'][i,j,:] = syst['dens']['ij'][i,j] * dft.fft_prefactor * potential['chg_ij'][i,j] * exp( -1.0 * k**2 / (4.0 * potential['erf_alpha'][i,j]**2)) / k**2          
        # end for j in range(ncomponents)
      # end for i in range(ncomponents)            
    # end if (pot_type = 'coulomb')
        
    # ****************************************
    
    # 1/r^4 dependent interaction as described in Netz et al. Curr Opinion Colloid Interface Sci 2004, 9, 192-197
    # taking into account charge-induced dipole contributions
    # all constant factors are included in the chgdip_ij
    elif (pot_type == 'chg_inddip'):
      for i in range(syst['ncomponents']):
        for j in range(syst['ncomponents']):
          U_ij_individual[pot_index][i,j,:] = (potential['chgdip_ij'][i,j] + potential['chgdip_ij'][j,i]) / r**4
          U_ij[i,j,:] += U_ij_individual[pot_index][i,j,:] 
        # end for j in range(ncomponents)
      # end for i in range(ncomponents)
    # end if (pot_type == 'chg_inddip')

    # ****************************************

    # Lennard-Jones potential
    elif (pot_type == 'lj'):
      for i in range(syst['ncomponents']):
        for j in range(syst['ncomponents']):
          U_ij_individual[pot_index][i,j,:] = 4.0 * potential['eps_ij'][i,j] * ((potential['sigma_ij'][i,j]/r)**12 - (potential['sigma_ij'][i,j]/r)**6) 
          U_ij[i,j,:] += U_ij_individual[pot_index][i,j,:] 
        # end for j in range(ncomponents)
      # end for i in range(ncomponents)
    # end if (pot_type = 'lj')

    # ****************************************

    # potential of mean force from external file
    elif (pot_type == 'pmf'):
      # the raw data was stored during the input processing
      # only the unique combinations were stored
      print("processing pmf data")
      # let's copy the values from parm array - the code here will be quite
      # long and it is better to make these things simpler...
      # distances
      pmf_r = parm[pot_index]['pmf_r']
      # and the respective pmfs
      pmf_ij = parm[pot_index]['pmf_ij']

      # stuff for adding of missing values
      # currently only constant value is supported
      # in the future, it might be possible to have extrapolation, ...
      pmf_fill = parm[pot_index]['pmf_fill']
      # min/max distance values in pmf_r
      pmf_rmin = parm[pot_index]['pmf_border']['min']
      pmf_rmax = parm[pot_index]['pmf_border']['max']
      # interpolation scheme
      pmf_interp = parm[pot_index]['pmf_interp']
      print("\t%s interpolation will be used" % pmf_interp['type'])

      # prepare the data
      # set all values to nan - good to see if everything works correctly
      U_ij_individual[pot_index][:,:,:] = nan

      # firstly interpolate the values, adding of missing data will come afterwards
      # the interpolated data could be used for extrapolation in the 'outer' regions
      # someday in the future
      # we will now treat all combinations, pmfs are stored for unique combinations only
      # variable used for translation complete <-> unique
      ind_pmf = 0

      for i in range(syst['ncomponents']):
        for j in range(i, syst['ncomponents']):
          print("\tcombination pmf(%u,%u)" % (i+1, j+1))
          
          # statistics
          add_before = 0
          add_after = 0
          interpolated = 0
          
          # actual index in the original pmf_r
          ind_r = 0
          # the pmf hat at least 2 points

          for dr in range(ctrl['npoints']):
            r_actual = (dr + 1) * ctrl['deltar']
            #print("data act=%f (dr=%u,min=%f,max=%f)" % (r_actual, dr, pmf_rmin[i], pmf_rmax[i]))
            if ( r_actual < pmf_rmin[ind_pmf] ):
              # use the fill value for r -> 0
              U_ij_individual[pot_index][i,j,dr] = pmf_fill['before']
              U_ij_individual[pot_index][j,i,dr] = pmf_fill['before']
              add_before += 1
              #print("before %u %f" % (add_before, r_actual))
            elif ( r_actual > pmf_rmax[ind_pmf]):
              # use the fill value for r -> infinity
              U_ij_individual[pot_index][i,j,dr] = pmf_fill['after']
              U_ij_individual[pot_index][j,i,dr] = pmf_fill['after']
              add_after += 1
              #print("after %u %f" % (add_after, r_actual))
            else:
              # interpolation
              #print("interp (ind_r=%u,border=%u)" % (ind_r, len(pmf_r[ind_pmf]) - 1))
              while (ind_r < (len(pmf_r[ind_pmf]) - 1)):
                r1 = pmf_r[ind_pmf][ind_r]
                r2 = pmf_r[ind_pmf][ind_r + 1]
                val1 = pmf_ij[ind_pmf][ind_r] 
                val2 = pmf_ij[ind_pmf][ind_r + 1] 
                #print("\tsearching, r=%f r1=%f r2=%f" % (r_actual, r1, r2))
                if (r1 <= r_actual <= r2):
                  # the position of r between r1 and r2
                  mu = 1.0 - (r2 - r_actual)/(r2-r1)
                  U_ij_individual[pot_index][i,j,dr] = pmf_interp['func'](val1, val2, mu)
                  U_ij_individual[pot_index][j,i,dr] = pmf_interp['func'](val1, val2, mu)
                  interpolated += 1
                  #print("\t\tinterpolated %u" % (interpolated))
                  break
                else:
                  # move in the pmf_r/pmf_ij to the next set of values
                  ind_r += 1
                  #print("\t\tskipping")
              #end while (ind_r < (len(pmf_r[ind_pmf]) - 1)):
            #end if ( r_actual < pmf_rmin[i] )
          #end for dr in range(ctrl['npoints'])
          print("\t\tadded before pmf\t%u" % add_before)
          print("\t\tinterpolated\t\t%u" % interpolated)
          print("\t\tadded after pmf \t%u" % add_after)
          ind_pmf += 1
        #end for i in range(syst['ncomponents']):
      #end for i in range(syst['ncomponents']):

      # test if nan is somewhere in the U_ij_individual array - this would indicate problem
      if (isnan(U_ij_individual[pot_index][:,:,:]).any()):
        for i in range(syst['ncomponents']):
          for j in range(syst['ncomponents']):
            for dr in range(ctrl['npoints']):
              if (isnan(U_ij_individual[pot_index][i,j,dr])):
                print("nan at %u,%u,%u" % (i,j,dr))
                print("critical problem with interpolation algorithm! nan value present!")
                sys.exit(2)
      # end if (isnan(U_ij_individual[pot_index][:,:,:]).any):
      
      # differentiation of the PMF

      # we can have a problem with inf and nan values
      # diff() of [inf,inf,0,0,0] produces [nan,-inf,0,0] and this poses a problem for the osmotic
      # coefficient calculation!
      # make deep copy of the original array
      #U_ij_sanitized = copy(U_ij_individual[pot_index][:,:,:])
      # replace the inf by const.inf_value
      #U_ij_sanitized[isinf(U_ij_sanitized)] = const.inf_value

      # let's first pre-set the array to zero (it's None at the moment)
      # differentiation of N points leads to N-1 points => let's set leave the first to be zero
      #dU_ij_individual[pot_index] = zeros((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
      
      # the first and the last value are set artificially to zero
      # evaluate the derivative with the sanitized function
      #dU_ij_individual[pot_index][:,:,1:] = diff(U_ij_sanitized[:,:,:]) / ctrl['deltar']
      # function gradient returns centered (3-point) difference - better!
      # at the moment, the difference is not used anywhere!
      #for i in range(syst['ncomponents']):
      #  for j in range(syst['ncomponents']):
      #    dU_ij_individual[pot_index][i,j] = gradient(U_ij_sanitized[i,j], ctrl['deltar'])
      
      # add the individual potential to the total potential
      U_ij += U_ij_individual[pot_index]
      
      print("")
    # end if (pot_type = 'pmf')

    # ****************************************
          
    else:
      print("unknown potential! exiting")
      sys.exit(1)
    # end else (unknown potential type)
    
  # end for pot_type in parm

  return(U_ij, U_ij_individual, dU_ij_individual, U_discontinuity, U_erf_ij)

# **********************************************************************************************  

def def_modMayerFunc(syst, U_ij, U_ij_individual, U_discontinuity, U_erf_ij):
  """calculate exp(-beta U_ij) == Mayer function + 1 according to the discretized potential
     
     calculate also the contributions of individual potentials to exp(Uij)
     exp(-Uij)=exp(-U1ij-U2ij-U3ij-...)=exp(-U1ij)exp(-U2ij)exp(-U3ij)...
     the order of contributions is the same as the order of potentials in the parm list
     store the contributions in a list; they will be used later (osmotic coefficients)
  """
  modMayerFunc = {}
  
  # since the potential is in kT units, we just do the exp(-U_ij)
  modMayerFunc['u_ij'] = exp(-U_ij)
  # evaluate the correction according to Ng
  modMayerFunc['erf'] = exp(U_erf_ij)
  
  # now evaluate the contributions
  modMayerFunc['contrib'] = []
  for pot_index in range(len(U_ij_individual)):
    #print("calculating contribs", pot_index)
    modMayerFunc['contrib'].append(exp(-U_ij_individual[pot_index]))
    # and take care of the discontinuities, if present
    if (U_discontinuity[pot_index] != None):
      for i in range(syst['ncomponents']):
        for j in range(syst['ncomponents']):
          dr, left_neigh, right_neigh = U_discontinuity[pot_index][i][j]
          modMayerFunc['contrib'][pot_index][i, j, dr] *= 0.5 * (exp(-left_neigh) + exp(-right_neigh))
    # end if (U_discontinuity[pot_index] != None)

  return(modMayerFunc)

# **********************************************************************************************  

if __name__ == "__main__":
  print __doc__
  print("Usage as a standalone application/script is not supported at the moment")

