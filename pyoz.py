#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this file is part of the pyOZ bundle
# pyOZ is a solver of the Ornstein-Zernike equation written in Python
# pyOZ is released under the BSD license
# see the file LICENSE for more information

"""
Program for numerical solution of the Ornstein-Zernike equation
version of the program is defined in pyoz_const.py
revision of this file: 0.1.8

check the file _features.txt and website http://pyoz.vrbka.net for further information

in the whole program (except where stated otherwise), the numpy/scipy arrays are used. as
a result, all mathematical operations are done element-wise. this applies also for multiplication
of vectors and arrays.
"""

# python modules
import sys
# timing functions
from time import time
#import scipy
from scipy import array, mat, copy, eye, fromfile, zeros, isfinite, empty_like, empty

# own modules
import pyoz_input as inputdata
# interaction potential handling
import pyoz_potential as potential
# discrete transformations
import pyoz_dft as ft
# import the convergence-related routines
from pyoz_misc import convergence_dsqn, dotproduct
# functions for property evaluation
import pyoz_property as properties
# 
from pyoz_const import pyoz_version
from pyoz_closure import calcGammaTerm
# graphics, constants, closures and solvers are loaded later

# **********************************************************************************************  

def main(argv=None):
  print("")
  print("pyOZ - iterative solver of the Ornstein-Zernike equation")
  print("version %s, Lubos Vrbka, 2008-2009" % pyoz_version)
  print("")
  
  if (argv==None):
    argv = sys.argv
  
  # parse the input file
  cmdline = inputdata.parse_cmdline(argv)
  # if -o is specified on command line, then it is used as stdout
  # if -o is not specified, then console is used
  if (cmdline['output'] != None):
    try:
      sys.stdout = open(cmdline['output'], "wt")
    except IOError, msg:
      sys.stdout = sys.__stdout__
      print("error opening output file %s" % cmdline['output'])
      print msg
      sys.exit(2)
    sys.stderr.write("output redirected to " + cmdline['output'] + "\n")
    
  # parse the control file with settings and parameters
  # return 4 collections and class with constants
  ctrl, syst, parm, outp, const = inputdata.parse_input(cmdline)

  # allocate distance arrays
  # array of distance in real space
  r = array(map(lambda x: (x + 1) * ctrl['deltar'], range(ctrl['npoints']) ))
  # array of distances in reciprocal space
  k = array(map(lambda x: (x + 1) * ctrl['deltak'], range(ctrl['npoints']) ))

  # initialize the DFT class
  print("initializing DFT routines")
  dft = ft.dft(ctrl['npoints'], ctrl['deltar'], ctrl['deltak'], r, k)
  dft.print_status()
  print("")
  
  # initialize the plotting subsystem if requested
  if (ctrl['do_graphics']):
    import pyoz_plot
    pyoz_plot.plot_initialize(ctrl, syst, const, r)
  # end if(do_graphics):
  
  # calculate the total U_ij potential, contributions of individual potentials (hs, lj, coulomb, ...)
  # also numerical derivatives of the contributions (where no analytical form is available)
  # and get the information on discontinuities
  # calculate also the erf-corrected direct correlation functions in real and fourier space
  # according to Ng
  U_ij, U_ij_individual, dU_ij_individual, U_discontinuity, U_erf_ij = potential.def_potential(ctrl, syst, parm, const, dft, r, k)
  # write the pair potential to the file, if requested
  if (outp['U_ij_write']):
    print("writing pair potential\t(%s)" % outp['U_ij_name'])
    try:
      fw = open(outp['U_ij_name'], "wt")
      for dr in range(ctrl['npoints']):
        fw.write("%8.3f" % r[dr])
        for i in range(syst['ncomponents']):
          for j in range(i, syst['ncomponents']):
            fw.write("%20.5e" % U_ij[i, j, dr])
        # end for i,j in range ncomponents...
        fw.write("\n")
      fw.close()
    except IOError, msg:
      print("error while saving interaction potential")
      print(msg)
      sys.exit(1)
    print("")
  
  # calculate the exp(-beta U_ij) for total potential (Mayer function + 1)
  # calculate the exp(-beta U_ij) for all individual potentials (hs, lj, coulomb) and evaluate
  # discontinuities where necessary
  # calculate the erf-correction contribution exp(U_erf_ij)
  # store all in a dictionary modMayerFunc
  modMayerFunc = potential.def_modMayerFunc(syst, U_ij, U_ij_individual, U_discontinuity, U_erf_ij['real'])
  # store the mayer function itself for the purpose of CG procedure with PY closure
  M = modMayerFunc['u_ij'] - 1.0
  
  # allocate arrays with direct, total and pair corr. functions, Gamma function
  # some arrays will emerge from arithmetic operations, 
  # showing them here makes the code clearer
  # real space: _r_, Fourier space _f_
  # direct correlation function with Ng-correction (cs) applied
  # direct correlation function without (c) and with (C) density factor applied
  # w/o density correction in real space; w/ density correction in Fourier space
  #c_r_ij
  C_f_ij = zeros((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
  Cs_f_ij = zeros((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
  #cs_r_ij  
  # pair correlation function
  #g_r_ij
  # h in Fourier space without (h) and with (H) density factor applied
  #H_f_ij
  # matrix of partial structure factors
  #S
  # actual, old (o) and new (n) values for Gamma
  # these are short-ranged Gamma (see the code for more details)
  G_r_ij = zeros((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
  #G_o_ij
  G_n_ij = zeros((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
  G_f_ij = zeros((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))

  # identity matrix for the solver (dr copies!)
  e_ij = eye(syst['ncomponents'])
  #o_ij = ones((syst['ncomponents'],syst['ncomponents']))
  E_ij = zeros((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
  #O_ij = zeros((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
  for dr in range(ctrl['npoints']):
    E_ij[:,:,dr] = e_ij
    #O_ij[:,:,dr] = o_ij
  # zero array for the newton-raphson 
  Z = zeros((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
  # arrays for the Newton-Raphson procedure (allocated even if not used)
  CFXq = empty((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
  AX  = empty((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
  AXq = empty((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
  Rq  = empty((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
  SRS = empty((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
  SRSq = empty((syst['ncomponents'], syst['ncomponents'], ctrl['npoints']))
  
  # process initial Gamma for the first iteration
  # IMPORTANT: do not forget, that we are dealing with the short-range Gamma all the time
  # in case of noncharged systems, it is equal to normal Gamma
  if (cmdline['gamma'] != None):
    print("attempting load of G_ij from %s" % cmdline['gamma'])
    # attempt load of data from external file
    # short-ranged Gamma is loaded
    try:
      if (cmdline['binarygamma']):
        # binary file
        G_r_ij = fromfile(cmdline['gamma'])
      else:
        # text file
       G_r_ij = fromfile(cmdline['gamma'], float, -1, " ")
      # end if (cmdline['binarygamma']):
      G_r_ij.shape = (syst['ncomponents'], syst['ncomponents'], ctrl['npoints'])
    # end of the try block
    except:
      print("\tload failed, using zero Gamma function")
      # set G_r_ij to a zero Gamma and apply the Ng-correction
      # in reality the short-ranged Gamma function is then not zero, but the original Gamma is
      # when no long-ranged potential (coulomb) is present, the correction is zero
      G_r_ij = -U_erf_ij['real']
    else:
      print("\tsuccesfully loaded")
      # symmetrize the 'matrix' for all values of r
      # G(1,2) = G(2,1) - the pair potentials are symmetric
      for dr in range(ctrl['npoints']):
        G_r_ij[:,:,dr] = (G_r_ij[:,:,dr] + G_r_ij[:,:,dr].transpose()) / 2
    # end else of the try/except/else block  
  else:
    # the Gamma is not loaded - we have zero Gamma, but need to apply the Ng-correction
    # in order to have short-range Gamma
    print("using zero Gamma function")
    G_r_ij = -U_erf_ij['real']
  # end if (cmdline['gamma'] == None)
  print("")

  # update the plot if requested
  if (ctrl['do_graphics']):
    # Gamma will be updated after the closure is called - just to save some unnecessary calls
    pyoz_plot.plot_update(syst, const, U_r=U_ij, U_erf=U_erf_ij['real'], G_r=None, c_r=None, g_r=None, c_f=None)
  # end if(do_graphics):
  
  # the last thing to decide - which solver will be used?
  # linalg.solve is not very efficient on 1x1 and 2x2 matrices that are most frequently used
  # let's try with own functions for such cases
  if (syst['ncomponents'] == 1):
    from pyoz_solver import solver_1 as solver_function
    print("using optimized solver for 1 component")
  elif (syst['ncomponents'] == 2):
    # solver_2 works but is slower!
    from pyoz_solver import solver_2 as solver_function
    print("using optimized solver for 2 components")    
  elif (syst['ncomponents'] == 3):
    from pyoz_solver import solver_n as solver_function
    print("using numpy linalg solver for 3 components")
  elif (syst['ncomponents'] > 3):
    from pyoz_solver import solver_n as solver_function
    print("using numpy linalg solver")
  # the correct solver has been selected

  print("\nstarting iteration\n==================")
  
  total_iter = 0
  converged = 0
  niter = 0

  while (not converged and niter < ctrl['max_iter']):
    # timing purposes
    time_beg = time()
    
    niter += 1
    total_iter += 1
    
    print("main\t%4u     " % niter),

    # show "progress bar" when output is redirected
    if (cmdline['output'] != None):
      sys.stderr.write(".")
      sys.stderr.flush()
      if ((total_iter % 25) == 0):
        sys.stderr.write("\n")

    # create copy of original Gamma function
    G_o_ij = copy(G_r_ij)
    
    # call closure relation and get c_r_ij
    # the Ng-formalism is already applied, the erf-correction is taken care of 
    # in the definition of the modified Mayer function
    cs_r_ij, g_r_ij = syst['closure'](syst, r, modMayerFunc, U_discontinuity, G_r_ij)
    
    # update the plot if requested
    if (ctrl['do_graphics']):
      pyoz_plot.plot_update(syst, const, U_r=None, U_erf=None, G_r=G_r_ij, c_r=cs_r_ij, g_r=g_r_ij, c_f=None)
    # end if(do_graphics):

    # FT c_r_ij to c_f_ij 
    # we are using Fourier-sine transform; there are some steps involved in between FBT (Bessel) and FST
    # this will not be discussed here, check the documentation and pyoz_dft.py for further information
    # the whole program is using FTs normalized with the density prefactors
    # sqrt(rho_i * rho_j) in order to have dimensionless functions in k-space
    # i.e., the FTs are multiplied by this factor, iFTs are divided by this factor
    # it follows then, that infinite dilution is taken care of there as well
    for i in range(syst['ncomponents']):
      for j in range(syst['ncomponents']):
        # perform the Fourier-Bessel transform of the short-ranged direct correlation function
        # compensate for the ng correction, return short-ranged and full c in fourier space 
        (Cs_f_ij[i,j], C_f_ij[i,j]) = dft.dfbt(cs_r_ij[i,j], norm=syst['dens']['ij'][i,j], corr=-U_erf_ij['fourier'][i,j])
      # end for j in range(ncomponents)
    # end for i in range(ncomponents)
       
    # update the plot if requested
    if (ctrl['do_graphics']):
      pyoz_plot.plot_update(syst, const, U_r=None, U_erf=None, G_r=None, c_r=None, g_r=None, c_f=C_f_ij)
    # end if(do_graphics):

    # now we have to solve the matrix problem in the Fourier space
    # note that the convolution theorem involves a constant factor ('a')
    # depending on the used forward fourier transform normalization constant
    # H = C + aCH
    # H - aCH = C
    # (E - aC)H = C
    # H = {E - aC}^-1 * C
    # however, thanks to the normalization chosen so that for FT it is 1, we can write
    # H = {E - C}^-1 * C
    # E + H = {E - C}^-1 * (C + E - C)
    # S = {E - C}^-1 * E
    H_f_ij = solver_function((E_ij - dft.ft_convolution_factor * C_f_ij), C_f_ij, ctrl['npoints'])
    from math import pi
    S = E_ij + H_f_ij
    #S = solver_function((E_ij - C_f_ij), E_ij, ctrl['npoints'])
    
    # convert H to short ranged Gamma G(k) = H(k) - Cs(k)
    #G_f_ij = H_f_ij - Cs_f_ij
    # convert S to short ranged Gamma G(k) = S(k) - E - Cs(k)
    G_f_ij = S - E_ij - Cs_f_ij
    
    # FT G_f_ij to G_r_ij 
    for i in range(syst['ncomponents']):
      for j in range(syst['ncomponents']):
        # perform the inverse Fourier transform of the Gamma function
        G_n_ij[i,j] = dft.idfbt(G_f_ij[i,j], norm=syst['dens']['ij'][i,j], corr=-U_erf_ij['real'][i,j])
      # end for j in range(ncomponents)
    # end for i in range(ncomponents)

    # *********************************************************************************************

    # test for convergence and write the gamma if everything is OK
    norm_dsqn = convergence_dsqn(ctrl, syst, G_o_ij, G_n_ij)
    time_end = time()
    print("%f sec - DSQN %.3e -" % ((time_end - time_beg), norm_dsqn)),
    if (norm_dsqn > ctrl['max_dsqn'] or (not isfinite(norm_dsqn))):
      print("\nDSQN too large, calculation is probably diverging")
      print("check inputs and outputs and/or increase the value of max_dsqn (%e at the moment)" % ctrl['max_dsqn'])
      sys.exit(2)

    if (norm_dsqn <= ctrl['convergence_crit']):
      print("converged")
      converged = 1
    else:
      print("not converged")
      
      # test if we do picard or newton-raphson
      if (not ctrl['do_nr']):
        # perform the picard mixing
        # calculate the new Gamma
        G_r_ij = (1.0 - ctrl['mix_param']) * G_o_ij + ctrl['mix_param'] * G_n_ij
      else:
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Newton-Raphson/Conjugate gradients method
        
        # we are trying to solve the problem AX = B where
        # A is a linear operator, X is dgamma and B is difference between input and output gamma
        # we are using iterative method to get the solution (non-symmetric conjugate gradients)
        # more details are given in my notes or in
        # Zerah: J Comp Phys 61 1985, 280
        # Belloni: J Chem Phys 88 (8) 1988, 5143
        # the names of the respective valuables in the following section will be kept consistent
        # with the papers

        nr_converged = 0
        nr_niter = 0
        
        # calculate the convergence criterion
        # it's done relatively to the DSQN of the main cycle in order to avoid
        # unnecessary iterations in the beginning, where the linear
        # approximation is not exact
        nr_convergence_crit = norm_dsqn * ctrl['nr_convergence_factor']
        
        # calculate B, as defined in the paper
        B = G_n_ij - G_o_ij
        
        # now initialize the system for step n=0 - iteration 1 will provide X(1)
        # the values of X(n) and X(n-1) needed for the iterative algorithm
        # let's choose X(0)=X(-1)=const * B
        # X(n)=Xcur; X(n-1)=Xold; X(n+1)=Xnew
        Xold = B * ctrl['nr_mix_param']
        Xcur = copy(Xold)
        Xnew = copy(Xold)
        
        # R(0) definition
        Rcur = 0.0
        # alpha(0) definition - will use L instead!
        Lcur = 0.0

        # for the first iteration, set W(2) to 1.0
        Wnew = 1.0
        # we don't need Wcur here, but let's define it...
        Wcur = 1.0

        # we will also need matrices S (both HNC and PY) and H (HNC) or M (PY - Mayer function) for the operator A (defined in the nr-cycle)
        # we use functions from the main cycle
        # we do it with arrays here and convert to matrices where needed and appropriate
        # we have to take the H using the old Gamma! i.e., taking g(r) provided by the closure shortly after the main iteration
        # cycle is started and subtracting 1
        H = g_r_ij - 1.0
        # modMayerFunc is MayerFunc + 1.0
        # this has been done before, so commenting out
        #M = modMayerFunc['u_ij'] - 1.0

        # the arrays for the operations involving operators A and At were created during the initialization
        
        # the code will operate with the matrix CF (closure factor), which is set according to closure
        # to either H (total correlation function, for HNC) or to M (mayer function, for PY)
        # check below for the algorithm details
        if (syst['closure_name'] == 'hnc'):
          CF = H.copy()
        elif (syst['closure_name'] == 'py'):
          CF = M
        else:
          sys.stderr.write("unsupported closure! \n")
          sys.exit(1)
          
        # timing purposes
        nr_time_beg = time()

        # increase the counter for the next half-iteration
        total_iter += 1
        
        # we make one half-iteration (number 0) and then carry on with full cycles until convergence
        while (not nr_converged and nr_niter <= ctrl['nr_max_iter']):
          print("  nr/cg\t    :%-4u" % (nr_niter)),
          # show "progress bar" when output is redirected - this time with plus sign
          # normal iterations are done with "." as an indicator
          if (cmdline['output'] != None):
            sys.stderr.write("+")
            sys.stderr.flush()
            if ((total_iter % 25) == 0):
              sys.stderr.write("\n")
          # end if (cmdline['output'] != None)

          # 'shift' the respective functions/values) (except R, which will be done later)
          Xold = copy(Xcur)
          Xcur = copy(Xnew)
          # store R from previous iteration R(n) to R(n-1)
          Rold = copy(Rcur)
          # store alpha (number) from previus iteraation alpha(n) to alpha(n-1)
          Lold = Lcur
          # store W (number) from previous iteration W(n+1) to W(n) 
          Wcur = Wnew
          
          # for matricial relations (part of operators A and At) we need to do everything separately for each discretization step
          # perform the calculation of AX
          # this will be done in several steps since
          # !!!!!!!!!!!!!!!!!!! in HNC !!!!!!!!!!!!!!!!!!! 
          # AX = 1X - iFT ( S FT(HX) S - FT(HX))
          # !!!!!!!!!!!!!!!!!!! in PY !!!!!!!!!!!!!!!!!!!
          # AX = 1X - iFT ( S FT(MX) S - FT(MX))
          # where M is the Mayer function exp(-betaU) - 1

          # the code will operate with the matrix CF (closure factor), which is set according to closure
          # to either H (total correlation function, for HNC) or to M (mayer function, for PY)
          # this was done outside of this cycle
            
          # H.X is not matricial product!
          CFX = CF*Xcur
          
          for i in range(syst['ncomponents']):
            for j in range(syst['ncomponents']):
              CFXq[i,j] = dft.dfbt(CFX[i,j])[0]
              
          # matricial products here
          for dr in range(ctrl['npoints']):
            AXq[:,:,dr] = mat(S[:,:,dr])*mat(CFXq[:,:,dr])*mat(S[:,:,dr]) - mat(CFXq[:,:,dr])
            
          for i in range(syst['ncomponents']):
            for j in range(syst['ncomponents']):
              AX[i,j] = Xcur[i,j] - dft.idfbt(AXq[i,j])

          # calculate Rcur = R(n) = B - AX(n)
          # do the calculation
          Rcur = B - AX
          
          # check for convergence here - if converged, abandon the cycle!
          # we check how far is Rcur from zero (Z is zero array with the same dimensions as Rcur)
          nr_norm_dsqn = convergence_dsqn(ctrl, syst, Rcur, Z)
          #nr_norm_dsqn = convergence_dsqn(ctrl, syst, B, AX)

          nr_time_end = time()

          # convergence is tested relatively to the DSQN of the 'outer' cycle
          print("%f sec - rel. DSQN %.3e -" % ((nr_time_end - nr_time_beg), nr_norm_dsqn/norm_dsqn)),
          if (nr_norm_dsqn > ctrl['max_dsqn'] or (not isfinite(nr_norm_dsqn))):
            print("\n\tDSQN too large, calculation is probably diverging")
            break
          
          if (nr_norm_dsqn <= nr_convergence_crit):
            print("converged")
            nr_converged = 1
          else:
            # the conjugate gradients algorithm needed - NR has not converged
            print("not converged")
  
            nr_time_beg = time()

            nr_niter += 1
            total_iter += 1
        
            # perform the calculation of AtR
            # this will be done in several steps since and At is an adjoint of the operator A
            # !!!!! in HNC !!!!!
            # AT R = 1R - FT (S iFT(R) S - iFT(R))H
            # !!!!! in PY !!!!!
            # AT R = 1R - FT (S iFT(R) S - iFT(R))M
            
            # the operator works with the matrix CF, set according to the used closure
            for i in range(syst['ncomponents']):
              for j in range(syst['ncomponents']):
                # even though R is r-space function, we are using the inverse FT here
                # the definition of the adjoint requires its usage here!
                # the problem is also the normalization sqrt(rho_i rho_j) that would be applied
                # incorrectly in case normal FT would be used here
                # note that the sinus transform is the same in r- and k-spaces => the difference really
                # lies in the normalization
                #Rq[i,j] = dft.dfbt(Rcur[i,j])[0]
                Rq[i,j] = dft.idfbt(Rcur[i,j])
                
            # matricial product
            for dr in range(ctrl['npoints']):
              SRS[:,:,dr] = mat(S[:,:,dr])*mat(Rq[:,:,dr])*mat(S[:,:,dr])-mat(Rq[:,:,dr])
 
            for i in range(syst['ncomponents']):
              for j in range(syst['ncomponents']):
                # even though SRS is k-space function, we are using the forward FT here
                # the definition of the adjoint requires its usage here!
                # the problem is also the normalization sqrt(rho_i rho_j) that would be applied
                # incorrectly in case iFT would be used here
                # note that the sinus transform is the same in r- and k-spaces => the difference really
                # lies in the normalization
                # remember that FT returns 2 functions in this case
                #SRSq[i,j] = dft.idfbt(SRS[i,j])
                SRSq[i,j] = dft.dfbt(SRS[i,j])[0]
                
            # not a matricial product!
            AtR = Rcur - SRSq*CF

            # calculate Lcur = alpha(n) = (R(n),R(n))/(AtR(n),AtR(n))
            # where (Y,Z) is inner product \sum_ij rho_i rho_j \int Y_ij Z_ij 4 \pi r^2 dr
            Lcur = abs(dotproduct(ctrl, syst, r, Rcur, Rcur))/abs(dotproduct(ctrl, syst, r, AtR, AtR))

            # calculate Wnew = W(n+1) (except for first iteration)
            if (nr_niter != 1):
              # do the full calculation, in the first iteration the value is pre-set to 1.0
              Wpartial = 1.0 - Lcur * abs(dotproduct(ctrl, syst, r, Rcur, Rcur))/(Lold * Wcur * abs(dotproduct(ctrl, syst, r, Rold, Rold)))
              Wnew = 1.0 / Wpartial
            # end calculation of Wnew
            
            # calculate X(n+1) = X(n-1) + W(n+1)(alpha(n)ATR(n) + X(n) + X(n-1)
            Xnew = Xold + Wnew * (Lcur * AtR + Xcur - Xold)            
          # end if (nr_norm_dsqn <= nr_convergence_crit) - handling of the else-branch (not converged)
        # end while (not nr-converged and nr_niter < ctrl['nr_max_iter'])
        
        # in case convergence was not reached, do Picard
        if (not nr_converged):
          print("\tcouldn't converge NR/CG cycle,"),
          if (not ctrl['nr_noconv_incr']):
            print("using Picard iteration instead")
            G_r_ij = (1.0 - ctrl['mix_param']) * G_o_ij + ctrl['mix_param'] * G_n_ij
          else:
            print("using non-converged increment")
            G_r_ij = G_o_ij + Xnew
        else:
          #G_r_ij = G_o_ij + Xcur
          G_r_ij = G_o_ij + Xnew

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      # end else (Newton-Raphson method)
    # end else (calculation not converged)

    # output data
    # test whether some function of interest should be saved
    # if savefreq is 0, then skip
    # first element of the result of modf() call is a remainder after division
    # if it is zero, then save the file...
    # Gamma function
    if ((outp['G_ij_write']) and (outp['G_ij_savefreq'] != 0) and (not divmod(niter, outp['G_ij_savefreq'])[1])):
      # store the Gamma function
      print("\tGamma function stored")
      try:
        if (outp['G_ij_binary']):
          G_r_ij.tofile(outp['G_ij_name'])
        else:
          G_r_ij.tofile(outp['G_ij_name'], " ", "%e")
      except IOError, msg:
        print("error while saving Gamma function")
        print(msg)
        sys.exit(1)
    #end if

    # *********************************************************************************************

  # end while (not converged)

  print("\niteration process completed in iteration %u" % niter)
  if (converged):
    print("\tcalculation converged")
  else:
    print("\tcalculation not converged; maximum number of iterations reached\n")
  
  # do closure
  cs_r_ij, g_r_ij = syst['closure'](syst, r, modMayerFunc, U_discontinuity, G_r_ij)
  # and evaluate uncorrected c(r) as well
  c_r_ij = cs_r_ij - U_erf_ij['real']

  # update the plot if requested
  if (ctrl['do_graphics']):
    pyoz_plot.plot_update(syst, const, U_r=None, U_erf=None, G_r=G_r_ij, c_r=cs_r_ij, g_r=g_r_ij, c_f=None)
  # end if(do_graphics):
  
  print("\nsaving outputs")
  # some error checking should be added here! for both g, G  
  try:
    # save g_r_ij to file
    if (outp['g_ij_write']):
      print("\tpair correlation function\t(%s)" % outp['g_ij_name'])
      fw = open(outp['g_ij_name'], "wt")
      fw.write("%8.3f" % 0.0)
      for i in range(syst['ncomponents']):
        for j in range(i, syst['ncomponents']):
          fw.write("%10.5f" % 0.0)
      fw.write("\n")
      for dr in range(ctrl['npoints']):
        fw.write("%8.3f" % r[dr])
        for i in range(syst['ncomponents']):
          for j in range(i, syst['ncomponents']):
            fw.write("%10.5f" % g_r_ij[i, j, dr])
        # end for i,j in range ncomponents...
        fw.write("\n")
      fw.close()
    
    # save c_r_ij to file
    if (outp['c_ij_write']):
      print("\tdirect correlation function\t(%s)" % outp['c_ij_name'])
      fw = open(outp['c_ij_name'], "wt")
      fw.write("%8.3f" % 0.0)
      for i in range(syst['ncomponents']):
        for j in range(i, syst['ncomponents']):
          fw.write("%10.5f" % 0.0)
      fw.write("\n")
      for dr in range(ctrl['npoints']):
        fw.write("%8.3f" % r[dr])
        for i in range(syst['ncomponents']):
          for j in range(i, syst['ncomponents']):
            # careful, we have to write the complete c(r), i.e., we need to compensate for the 
            # Ng-correction!
            # cs(r) = c(r) + Ucorr(r) => c(r) = cs(r) - Ucorr(r)
            fw.write("%10.5f" % c_r_ij[i, j, dr])
        # end for i,j in range ncomponents...
        fw.write("\n")
      fw.close()
    
    # save S to file
    if (outp['S_ij_write']):
      print("\tpartial structure factors\t(%s)" % outp['S_ij_name'])
      fw = open(outp['S_ij_name'], "wt")
      fw.write("%8.3f" % 0.0)
      for i in range(syst['ncomponents']):
        for j in range(i, syst['ncomponents']):
          fw.write("%10.5f" % 0.0)
      fw.write("\n")
      for dr in range(ctrl['npoints']):
        fw.write("%8.3f" % k[dr])
        for i in range(syst['ncomponents']):
          for j in range(i, syst['ncomponents']):
            # careful, we have to write the complete c(r), i.e., we need to compensate for the 
            # Ng-correction!
            # cs(r) = c(r) + Ucorr(r) => c(r) = cs(r) - Ucorr(r)
            fw.write("%10.5f" % S[i, j, dr])
        # end for i,j in range ncomponents...
        fw.write("\n")
      fw.close()
    
    if (outp['G_ij_write']):
      print("\tGamma function\t\t\t(%s)" % outp['G_ij_name'])
      # store the Gamma function if required
      # short-ranged Gamma is saved!
      if (outp['G_ij_binary']):
        G_r_ij.tofile(outp['G_ij_name'])
      else:
        G_r_ij.tofile(outp['G_ij_name'], " ", "%e")
        
    # save total interaction (U+Gamma(long-ranged!)) to file
    if (outp['Utot_ij_write']):
      print("\ttotal potential (U+Gamma)\t(%s)" % outp['Utot_ij_name'])
      fw = open(outp['Utot_ij_name'], "wt")
      for dr in range(ctrl['npoints']):
        fw.write("%8.3f" % r[dr])
        for i in range(syst['ncomponents']):
          for j in range(i, syst['ncomponents']):
            fw.write("%20.5e" % (U_ij[i, j, dr] + G_r_ij[i,j,dr] + U_erf_ij['real'][i,j,dr]))
        # end for i,j in range ncomponents...
        fw.write("\n")
      fw.close()
  # end try block
  except IOError, msg:
    print("error while saving output")
    print(msg)
    sys.exit(1)
  # end try/except block
  
  # now 'remove' the Ng-renormalization
  G_r_ij += U_erf_ij['real']
  # we also have the short range G in the Fourier space
  # if needed, it can be of course used here!
  # from now on, the G_r_ij is the real Gamma function without the 
  # stuff making the convergence easier
  
  # calculate also the term involving Gamma in HNC (exp(Gamma)) and PY (1+Gamma)
  # and save it
  G_term_ij = calcGammaTerm(syst, G_r_ij)
  
  # calculation of thermodynamic properties
  print("\ncalculation of (thermodynamic) properties")
  if (converged):
    # only evaluate the properties for converged calculation!
    # all information is printed inside these functions

    # for excess chem potential and compressibility, we need short range version of c_ij
    # it's different short-range than the one coming from Ng (which is finite for r=0)
    # our is given just by c_{ij}^s = c_{ij} + \beta U_{ij}^{coulomb} since for coulomb, 
    # c(r) = -\beta U_ij^{coulomb} for r->\infty
    # we check here, if coulomb potential is used and if yes, we just subtract the coulomb interaction from it
    # step 1 - get the index in the parm array, where coulomb info is stored. the same index is used in the
    # U_ij_individual array
    print("\ttesting for long-ranged potentials")
    index = -1
    for i in range(len(parm)) :
      if ('coulomb' in parm[i].values()):
        index = i
    if (index >= 0):
      print("\t\tfound, using short-ranged c(r)\n")
      # ----- was done already! firstly, we get rid of ng, (subtract U_erf_ij) converting cs_r_ij to c_r_ij
      # then we add the coulomb (lr correction) as shown above, to get c_r_ij_sr
      #c_r_ij_sr = cs_r_ij -U_erf_ij['real'] + U_ij_individual[index]
      c_r_ij_sr = c_r_ij + U_ij_individual[index]
    else:
      print("\t\tnot found, using original c(r)\n")
      # in this case, ng-correction is zero and cs_r_ij is already the function we need
      c_r_ij_sr = c_r_ij

    # kirkwood-buff integrals
    properties.kirkwood_buff(ctrl, syst, r, g_r_ij)

    # osmotic coefficients
    properties.osmotic_coeff(ctrl, syst, parm, const, r, g_r_ij, G_r_ij, G_term_ij, U_ij_individual, dU_ij_individual, U_discontinuity, modMayerFunc['contrib'])
    
    # excess chemical potential/activity
    # only supported for hnc!
    if (syst['closure_name'] == 'hnc'):
      properties.excess_chempot(ctrl, syst, const, r, g_r_ij - 1.0, G_r_ij, c_r_ij_sr, index, parm)
    else:
      print("\texcess chemical potentials available only with HNC\n")
      
    # isothermal compressibility
    properties.compressibility(ctrl, syst, const, r, c_r_ij_sr)
    
  # end if (converged)
  else:
    print("\tnon-converged calculation, properties won't be evaluated\n")
  # end of calculation of thermodynamic properties
  
  print("calculation finished\n")
  if (cmdline['output'] != None):
    sys.stderr.write("\ncalculation finished\n")

  if(ctrl['do_graphics']):
    # stop in order not to destroy the window with the plotted functions
    sys.stderr.write("press enter to close the graphics window and exit\n")
    sys.stdin.readline()
    
if __name__ == "__main__":
  sys.exit(main())

