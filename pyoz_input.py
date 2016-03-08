#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this file is part of the pyOZ bundle
# pyOZ is a solver of the Ornstein-Zernike equation written in Python
# pyOZ is released under the BSD license
# see the file LICENSE for more information

"""
Module with functions for input file parsing
revision of this file: 0.1.9
"""

# import external modules
# file handling, ...
import sys
# command line parameter parsing
import getopt
# needed things from scipy 
from scipy import array, zeros, pi, int8, int32, inf, float_, copy, divide
# needed things from math
from math import sqrt, ceil, modf
# regular expressions
from re import match

# internal (own) modules
# constants, conversion factors
import pyoz_const
# closure relations
from pyoz_closure import supported_closures

# **********************************************************************************************  
# classes for exception handling - needed to disable deprecation warning for python > 2.4

class badValue(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

class unkKeyword(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

class pmfError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

# **********************************************************************************************  

def parse_cmdline(argv):
  """usage information
     pyoz.py [ parameters ] name.inp
     
     recognized parameters
     -h/--help      print this information
     -i/--input=    name of the input file (mandatory)
     -n/--name=     name of the system (default = pyoz)
     -o/--output=   output filename, (default = standard output)
     -g/--gamma=    input gamma function filename
     -t/--textgamma read gamma function in text format (default = no, use binary)
  """
  cmdline = {}
  
  # later - stdout redirect might be created!
  # the setting is currently ignored! everything goes to stdout
  cmdline['input'] = None
  cmdline['output'] = None
  cmdline['name'] = None
  cmdline['gamma'] = None
  cmdline['binarygamma'] = True
  
  try:
    options, args = getopt.getopt(argv[1:], "hi:n:o:g:t", ["help", "input=", "name=", "output=", "gamma=", "textgamma"])
  except getopt.GetoptError, msg:
    print msg
    print parse_cmdline.__doc__
    sys.exit(2)

  #first step: parsing options (if any)
  for o, a in options:
    if o == "-h" or o == "--help" :
      print parse_cmdline.__doc__
      sys.exit(0)
    if o == "-i" or o == "--input":
      cmdline['input'] = a
    if o == "-o" or o == "--output":
      cmdline['output'] = a
    if o == "-n" or o == "--name":
      cmdline['name'] = a
    if o == "-g" or o == "--gamma":
      cmdline['gamma'] = a
    if o == "-t" or o == "--textgamma":
      cmdline['binarygamma'] = False
  #second step: parsing remaining arguments - there shouldn't be any at the moment!
  if (len(args) > 0):
    print "no extra arguments are allowed on command line"
    print parse_cmdline.__doc__
    sys.exit(2)
  if (cmdline['input'] == None):
    print "input file name missing"
    print parse_cmdline.__doc__
    sys.exit(2)    

  #print("\tinput filename\t%s" % cmdline['input'])
  #print("\toutput filename\t%s" % cmdline['output'])
  #print("\tgamma filename\t%s" % cmdline['gamma'])
  #print("\tproject name\t%s" % cmdline['name'])
  
  #end of command line arguments parsing
  return(cmdline)

# **********************************************************************************************  

def parse_input(cmdline):
  """parse the input file
  
  return 4 collections and 1 class (see detailed description below)
    ctrl - dict with control variables 
    syst - dict with some system properties
    parm - list of dictionaries with parameters
    outp - list of dictionaries with outuput-related variables
    const - constants class
  """

  print("pyOZ version %s input parser\n" % pyoz_const.pyoz_version)
  
  # open the input file and read the data
  try:
    fr = open(cmdline['input'], "rt")
    lines = fr.readlines()
    fr.close()
  except IOError, msg:
    print("\topening/loading of input file failed!")
    print("\t" + str(msg))
    sys.exit(2)
  # end try opening/loading input files

  # prepare new array
  lines_orig = lines[:]
  # remove blank characters from beginning and end of lines
  # and convert input to lowercase (retain also original data for filenames, ...)
  for i in range(len(lines)):
    # strip the white characters
    lines_orig[i] = lines[i].strip()
    # and convert to lowercase
    lines[i] = lines_orig[i].lower()
  
  # find the keywords/blocks anc check for correctness
  # store the information in the dictionary and list
  # index of the respective section start
  start_name = {'ctrl': None, 'syst': None, 'parm': None, 'outp': None}
  start_order = 0
  # general section starts and ends
  start_pos = []
  end_pos = []
  # we are looking for the following pairs
  # these are always the first words on the line (both of them!)
  # %ctrl ... %end
  # %syst ... %end
  # %parm ... %end
  # %outp ... %end
  for i in range(len(lines)):
    # keyword is the first word on line
    # handle the case when the line is empty
    if (len(lines[i]) > 0):
      keyword = lines[i].split()[0]
    else:
      keyword = ""
    # now check whether a keyword is recognized
    if (match(r'%ctrl$', keyword)):
      if (start_name['ctrl'] != None):
        print("second ctrl section not allowed, line %u" % (i + 1))
      else:
        start_name['ctrl'] = start_order
        start_order += 1
        start_pos.append(i)
    # end if (match(r'%ctrl$', keyword)):
    elif (match(r'%syst$', keyword)):
      if (start_name['syst'] != None):
        print("second syst section not allowed, line %u" % (i + 1))
      else:
        start_name['syst'] = start_order
        start_order += 1
        start_pos.append(i)
    # end if (match(r'%syst$', keyword)):
    elif (match(r'%parm$', keyword)):
      if (start_name['parm'] != None):
        print("second parm section not allowed, line %u" % (i + 1))
      else:
        start_name['parm'] = start_order
        start_order += 1
        start_pos.append(i)
    # end if (match(r'%parm$', keyword)):
    elif (match(r'%outp$', keyword)):
      if (start_name['outp'] != None):
        print("second outp section not allowed, line %u" % (i + 1))
      else:
        start_name['outp'] = start_order
        start_order += 1
        start_pos.append(i)
    # end if (match(r'%outp$', keyword)):
    elif (match(r'%end$', lines[i])):
      end_pos.append(i)
    # end if (match(r'%end$', keyword)):
    #else
    #  pass
    #end else
  # end for i in range(len(lines)):
  
  # now check for correctness
  # syst and parm sections are required
  if (start_name['syst'] == None):
    print("\t%syst section required and not present")
    sys.exit(2)
  if (start_name['parm'] == None):
    print("\t%parm section required and not present")
    sys.exit(2)

  # check if the read-in data are ok
  # position of blocks, misplaced statements, ...
  # if not, the function prints out a message and quits
  check_validity(lines, 1, start_pos, end_pos)
  
  # dissect the data and parse the individual data blocks with respective functions
  # ctrl can be omitted
  print("control statements")
  if (start_name['ctrl'] == None):
    ctrl = parse_ctrl(cmdline, [], [], -1, -1)
  else:
    # we don't want the %ctrl keyword
    # using slicing [a:b] we get rid of the value at index b -> no change needed for end_index
    start_index = start_pos[start_name['ctrl']] + 1
    end_index = end_pos[start_name['ctrl']]
    ctrl = parse_ctrl(cmdline, lines[start_index:end_index], lines_orig[start_index:end_index], start_index + 1, end_index + 1)
  print("")

  # syst is always present
  print("system information")
  start_index = start_pos[start_name['syst']] + 1
  end_index = end_pos[start_name['syst']]
  syst, const = parse_syst(ctrl, lines[start_index:end_index], lines_orig[start_index:end_index], start_index + 1, end_index + 1)
  print("")

  # parm is always present
  print("potential parameters")
  start_index = start_pos[start_name['parm']] + 1
  end_index = end_pos[start_name['parm']]
  parm = parse_parm(ctrl, syst, const, lines[start_index:end_index], lines_orig[start_index:end_index], start_index + 1, end_index + 1)

  # outp can be omitted
  print("output controls")
  if (start_name['outp'] == None):
    outp = parse_outp(cmdline, [], [], -1, -1)
  else:
    start_index = start_pos[start_name['outp']] + 1
    end_index = end_pos[start_name['outp']]
    outp = parse_outp(cmdline, lines[start_index:end_index], lines_orig[start_index:end_index], start_index + 1, end_index + 1)
  print("")

  return(ctrl, syst, parm, outp, const)

# **********************************************************************************************  

def parse_ctrl(cmdline, lines, lines_orig, lineno_beg, lineno_end):
  """
    parse the control statements  
  """

  ctrl_def = pyoz_const.ctrl_defaults()
  ctrl = {}

  # pre-set the default values, description below or in the pyoz_const.py
  do_graphics = ctrl_def.do_graphics
  graphics_p_xmin = ctrl_def.graphics_p_xmin
  graphics_p_xmax = ctrl_def.graphics_p_xmax
  npoints_exp = ctrl_def.npoints_exp
  npoints = ctrl_def.npoints
  deltar = ctrl_def.deltar
  max_r = None
  deltak = None
  max_k = None
  mix_param = ctrl_def.mix_param
  max_dsqn = ctrl_def.max_dsqn
  convergence_crit = ctrl_def.convergence_crit
  max_iter = ctrl_def.max_iter
  do_nr = ctrl_def.do_nr
  nr_mix_param = ctrl_def.nr_mix_param
  nr_convergence_factor = ctrl_def.nr_convergence_factor
  nr_max_iter = ctrl_def.nr_max_iter
  nr_noconv_incr = ctrl_def.nr_noconv_incr
  
  # parse the inputs and (possibly) overwrite the input data
  try:
    for i in range(len(lines)):
      # let's simplify the commands
      # we don't do in in lines in order to have line numbers
      # lines contains data converted all to lowercase
      l = lines[i]
      # lines_orig contains data with original case - used for filenames, ...
      orig_l = lines_orig[i]
      
      # keyword is the first word on line
      # handle the case when the line is empty
      if (len(l) > 0):
        keyword = l.split()[0]
      else:
        keyword = ""
      
      # now check whether a keyword is recognized
      if (match(r'#', l) or (l == "")):
        # commentary
        pass
      elif (match(r'graph$', keyword)):
        do_graphics = True
      elif (match(r'nograph$', keyword)):
        do_graphics = False
      elif (match(r'grxmin$', keyword)):
        graphics_p_xmin = float((l.split())[1])
      elif (match(r'grxmax$', keyword)):
        graphics_p_xmax = float((l.split())[1])
      elif (match(r'npoints$', keyword)):
        npoints = int((l.split())[1])
        # we need to then subtract one because of the particular fourier transform used
        # the last point will be added by the transform itself
        npoints -= 1
        if (npoints <= 0):
          raise badValue((i, "cannot use zero/negative number of discretization points!"))
      elif (match(r'deltar$', keyword)):
        deltar = float((l.split())[1])
        if (deltar <= 0.0):
          raise badValue((i, "step size must be larger than 0"))
      elif (match(r'mix_param$', keyword)):
        mix_param = float((l.split())[1])
        if ((mix_param <= 0) or (mix_param > 1)):
          raise badValue((i, "mixing parameter not in allowed range 0 < mix_param ≤ 1"))
      elif (match(r'conv_crit$', keyword)):
        conv_crit = float((l.split())[1])
        if (conv_crit <= 0.0):
          raise badValue((i, "convergence criterion has to be larger than 0"))
        if (conv_crit >= 1.0):
          raise badValue((i, "convergence criterion has to be smaller than 1"))  
      elif (match(r'max_iter$', keyword)):
        max_iter = int((l.split())[1])
        if (max_iter <= 0):
          raise badValue((i, "minimum number of 1 iterations has to be performed"))
      elif (match(r'max_dsqn$', keyword)):
        max_dsqn = float((l.split())[1])
        if (max_dsqn <= 0):
          raise badValue((i, "only positive dsqn values are allowed"))
      elif (match(r'use_nr$', keyword)):
        do_nr = True
        #print("NR/CG method has been disabled in this version due to severe problems!")
        #sys.exit(1)
      elif (match(r'nr_mix_param$', keyword)):
        nr_mix_param = float((l.split())[1])
        if ((nr_mix_param < 0) or (nr_mix_param > 1)):
          raise badValue((i, "NR mixing parameter not in allowed range 0 ≤ nr_mix_param ≤ 1"))
      elif (match(r'nr_conv_crit$', keyword)):
        nr_convergence_factor = float((l.split())[1])
        if (nr_convergence_factor <= 0.0):
          raise badValue((i, "NR relative convergence criterion has to be larger than 0"))
      elif (match(r'nr_max_iter$', keyword)):
        nr_max_iter = int((l.split())[1])
        if (nr_max_iter <= 0):
          raise badValue((i, "minimum number of 1 NR iterations has to be performed"))
      elif (match(r'nr_noconv_incr$', keyword)):
        nr_noconv_incr = True
      else:
        raise unkKeyword((i, l))
    # end for i in range(len(lines)):
  # end try
  except ValueError, msg:
    print("error processing line %lu (%s)" % (i + lineno_beg, l))
    print(msg)
    sys.exit(2)
  except unkKeyword, msg:
    print("error processing line %lu (%s)" % (msg.value[0] + lineno_beg, l))
    print("unknown keyword")
    sys.exit(2)
  except badValue, msg:
    print ("error processing line %lu (%s)" % (msg.value[0] + lineno_beg, l))
    print (msg.value[1])
    sys.exit(2)
  except IndexError, msg:
    print("error processing line %lu (%s)" % (i + lineno_beg, l))
    print("not enough data on the line")
    sys.exit(2)
    
  # end try/except block

  # save the data into the ctrl dictionary and print out the settings
  # number of discretization points
  ctrl['npoints_exp'] = npoints_exp
  ctrl['npoints'] = npoints
  print("\tnumber of points\t\t%u" % (ctrl['npoints'] + 1))
  if (((npoints + 1) & npoints) != 0): 
    # logical AND
    # the value was already decremented!
    # the input should be a power of 2
    print("\t!!!!! you should use number of points which is a power of 2 !!!!!")

  # recalculate step in fourier space, and maximum distances in r and k
  max_r = deltar * npoints
  deltak = pi / max_r
  max_k = deltak * npoints

  # step in real space and maximum distance
  ctrl['deltar'] = deltar
  ctrl['max_r'] = max_r
  print("\tdeltar, maximum r\t\t%f, %f" % (ctrl['deltar'], ctrl['max_r']))
  
  # step in reciprocal space and maximum distance
  ctrl['deltak'] = deltak
  ctrl['max_k'] = max_k
  print("\tdeltak, maximum k\t\t%f, %f" % (ctrl['deltak'], ctrl['max_k']))
  
  # mixing parameter
  ctrl['mix_param'] = mix_param
  # convergence criterion
  ctrl['convergence_crit'] = conv_crit
  # maximum value of DSQN (convergence criterion)
  ctrl['max_dsqn'] = max_dsqn
  # maximum number of iterations
  ctrl['max_iter'] = max_iter
  print("\tmax iterations\t\t\t%lu" % ctrl['max_iter'])
  print("\tconvergence criterion\t\t%e" % ctrl['convergence_crit'])
  print("\tmaximal allowed DSQN\t\t%e" % ctrl['max_dsqn'])

  ctrl['do_nr'] = do_nr
  ctrl['nr_mix_param'] = nr_mix_param
  ctrl['nr_convergence_factor'] = nr_convergence_factor
  ctrl['nr_max_iter'] = nr_max_iter
  ctrl['nr_noconv_incr'] = nr_noconv_incr
  if (ctrl['do_nr']):
    print("\n\tNewton-Raphson iteration method will be used")
    print("\tsupport of Newton-Raphson is highly experimental!")
    print("\tNR max iterations\t\t%lu" % ctrl['nr_max_iter'])
    print("\tNR initial mix parameter\t%e" % ctrl['nr_mix_param'])
    print("\tNR relative convergence factor\t%e" % ctrl['nr_convergence_factor'])
    if (not nr_noconv_incr):
      print("\tPicard will be used if NR fails")
      print("\tfallback Picard mix parameter\t%e" % ctrl['mix_param'])
    else:
      print("\tnon-converged increment will be used if NR fails")
  else:
    print("\n\tPicard iteration method will be used")
    print("\tPicard mix parameter\t\t%e" % ctrl['mix_param'])

  # use graphics?
  ctrl['do_graphics'] = do_graphics
  if (ctrl['do_graphics']):
    print("\n\tgraphic output will be used (experimental!)")
    # plot minimum and maximum x value
    if (graphics_p_xmin >= graphics_p_xmax):
      print("\t\tinvalid x-range, setting to default")
      ctrl['graphics_p_xmin'] = ctrl_def.graphics_p_xmin
      ctrl['graphics_p_xmax'] = ctrl_def.graphics_p_xmax
    else:
      ctrl['graphics_p_xmin'] = graphics_p_xmin
      ctrl['graphics_p_xmax'] = graphics_p_xmax
    # end if (graphics_p_xmin >= graphics_p_xmax)
  else:
    print("\n\tgraphic output won't be used")

  return(ctrl)
  
# **********************************************************************************************  

def parse_syst(ctrl, lines, lines_orig, lineno_beg, lineno_end):
  """
    get the system information, whatever might that be...
  """

  syst = {}
  syst_def = pyoz_const.syst_defaults()
  
  # temperature, epsilon_r, closure_name, conc_unit have default values here
  # in syst, we have to first pre-parse the input and get number of components,
  # temp and epsilon_r
  # then, const class can be created and the remaining parameters can be passed
  temp = syst_def.temp 
  epsilon_r = syst_def.epsilon_r
  closure_name = syst_def.closure_name
  conc_unit = syst_def.conc_unit
  
  # number of components
  ncomponents = None
  # names of the constituents
  name = None
  # their concentrations
  conc = None
  
  # parse the inputs and (possibly) overwrite the input data
  try:
    for i in range(len(lines)):
      # make the commands simpler
      # lines contains data converted to lowercase
      l = lines[i]
      # unconverted data
      orig_l = lines_orig[i]

      # keyword is the first word on line
      # handle the case when the line is empty
      if (len(l) > 0):
        keyword = l.split()[0]
      else:
        keyword = ""

      if (match(r'#', l) or (l == "")):
        # commentary
        pass
      elif (match(r'temp$', keyword)):
        temp = float((l.split())[1])
        if (temp < 0.0):
          raise badValue((i, "negative temperature not allowed"))          
      elif (match(r'epsilon_r$', keyword)):
        epsilon_r = float((l.split())[1])
        if (epsilon_r <= 0.0):
          raise badValue((i, "epsilon_r has to be larger than zero"))
      elif (match(r'ncomp$', keyword)):
        ncomponents = int((l.split())[1])
        if (ncomponents <= 0):
          raise badValue((i, "number of components has to be larger than 0"))
      elif (match(r'closure$', keyword)):
        closure_name = (l.split())[1]
      elif (match(r'names$', keyword)):
        name = (orig_l.split())[1:]
      elif (match(r'conc$', keyword)):
        conc = (l.split())[1:]
      elif (match(r'conc_unit$', keyword)):
        conc_unit = (l.split())[1]
      else:
        raise unkKeyword((i, l))
    # end for i in range(len(lines)):
  # end try
  except ValueError, msg:
    print("error processing line %lu (%s)" % (i + lineno_beg, l))
    print(msg)
    sys.exit(2)
  except unkKeyword, msg:
    print("error processing line %lu (%s)" % (msg.value[0] + lineno_beg, l))
    print("unknown keyword")
    sys.exit(2)
  except badValue, msg:
    print ("error processing line %lu (%s)" % (msg.value[0] + lineno_beg, l))
    print (msg.value[1])
    sys.exit(2)
  # end try/except block
  
  # set the temperature
  syst['temp'] = temp
  # dielectric constant of the solvent
  syst['epsilon_r'] = epsilon_r
  # evaluate the constants
  # initialize the class with constants, ...
  const = pyoz_const.const(syst['temp'], syst['epsilon_r'])
  const.print_const()
  
  # ncomponents and concentrations are required
  if (ncomponents == None):
    print("number of components missing in input!")
    sys.exit(2)
  if (conc == None):
    print("concentrations missing in input!")
    sys.exit(2)

  # process the names
  name_value = []
  if (name == None):
    for i in range(ncomponents):
      # create the line A, B, C, D, ...
      name_value.append(chr(ord('A') + i))
  else:
    # try to get the data from input
    if (len(name) < ncomponents):
      print("not enough component names in input (got %u need %u)" % (len(name), ncomponents))
      sys.exit(2)
    else:
      name_value = name[0:ncomponents]
  # end if (name == None) - names processing
  
  syst['ncomponents'] = ncomponents
  #ncombinations = ncomponents**2
  print("\tnumber of components\t\t%lu" % syst['ncomponents'])

  if (closure_name in supported_closures):
    syst['closure_name'] = closure_name
    syst['closure'] = supported_closures[syst['closure_name']]
    print("\tclosure relation\t\t%s" % syst['closure_name'].upper())
  else:
    print("\tunsupported or unknown closure %s" % closure_name )
    sys.exit(1)
  # end if (closure_name in supported_closures)

  # names of the constituents
  # if not given, use A, B, ...
  syst['name'] = name_value
  print("\tconstituent names\t\t"),
  for i in range(syst['ncomponents']):
    print("%s " % syst['name'][i]),
  print("")
  
  # process the concentration data
  try:
    # we have to create numpy array and then enlarge it
    conc_value = zeros((ncomponents,))
    for i in range(ncomponents):
      # there has to be one concentration value for every component
      conc_value[i] = float(conc[i])
  # end try
  except ValueError, msg:
    print("error processing concentration of component %u (%s)" % (i + 1, conc[i]))
    print(msg)
    sys.exit(2)
  except IndexError, msg:
    print("error processing concentration of component %u - not enough values in the input" % (i + 1))
    print(msg)
    sys.exit(2)
  # end try/except construct

  # save concentration in mol/L
  # and particle density in particles/A3
  if (conc_unit == "part_3"):
    # we have the particle density, calculate the molarity
    dens_num = conc_value
    # concentration in mol/L
    syst['conc_mol'] = dens_num * const.conv_parA3tomolL
  elif (conc_unit == "mol_l"):
    # we have molarity, calculate the particle density
    syst['conc_mol'] = conc_value
    # particle density pro A^3
    dens_num = syst['conc_mol'] * const.conv_molLtoparA3
  else:
    print("unknown concentration unit %s" % conc_unit)
    sys.exit(2)
  # end if (conc_unit == "part_3")...
  
  # is there any component at infinite dilution?
  inf_dilute = False
  # calculate density prefactors sqrt(rho_i * rho_j)
  dens_ij = zeros((syst['ncomponents'], syst['ncomponents']))
  # evaluate contributions where infinite dilution is present
  dens_inf = zeros((syst['ncomponents'], syst['ncomponents']), dtype=int8)
  for i in range(syst['ncomponents']):
    for j in range(syst['ncomponents']):
      dens_ij[i,j] = sqrt(dens_num[i] * dens_num[j])
      if (syst['conc_mol'][i] == 0.0 or syst['conc_mol'][j] == 0.0):
        # set to 1 to avoid problems with division by zero
        # not needed any longer - treated in FT
        #dens_ij[i,j] = 1.0
        dens_inf[i,j] = 1
        inf_dilute = True
      # end if (syst['conc_mol'][i] == 0.0 or syst['conc_mol'][j] == 0.0):
    # end for i in range(syst['ncomponents']):
  # end for j in range(syst['ncomponents']):
    
  syst['dens'] = {}
  syst['dens']['num'] = dens_num
  # total particle number density
  syst['dens']['totnum'] = syst['dens']['num'].sum()
  syst['dens']['ij'] = dens_ij
  syst['dens']['inf_dilution'] = dens_inf
  # molar fractions xi = rho_i/rho_tot
  syst['dens']['xi'] = syst['dens']['num']/syst['dens']['totnum']
  # is there any component at infinite dilution?
  syst['inf_dilute'] = inf_dilute

  # number of unique combinations (11,12,13,...,22,23,...33,...)
  # how many unique combinations we have?
  # it's defined by combinations with repetition
  # we are looking for k-tuples from n items, carrying at most k same items at a time
  # it's then (n+k-1 over k) = (n+k-1)!/(k!(n-1)!)
  # k is always 2 (pair potential), so the number of combinations (U(12)=U(21)) is
  # (ncomp+1)*ncomp/2
  syst['ncomb'] = (syst['ncomponents']+1)*syst['ncomponents']/2
  
  print("\tconcentrations mol/L\t\t"),
  for i in range(syst['ncomponents']):
    print("%f " % syst['conc_mol'][i]),
  print("")
  print("\tparticle densities part/A3\t"),
  for i in range(syst['ncomponents']):
    print("%f " % syst['dens']['num'][i]),
  print("")
  print("\ttotal particle density\t\t%f" % syst['dens']['totnum'])
  if (syst['dens']['totnum'] != 0.0):
    print("\tmolar fractions\t\t\t"),
    for i in range(syst['ncomponents']):
      print("%f " % syst['dens']['xi'][i]),
    print("")
  if (dens_inf.any() == 1):
    print("\tat least one component at infinite dilution")

  return(syst, const)

# **********************************************************************************************  

def parse_parm(ctrl, syst, const, lines, lines_orig, lineno_beg, lineno_end):
  """
   parse the parameters in lines array and store them as a list of dictionaries called "parm"
   there is one dictionary for each defined interaction potential, the potentials
   are parsed one by one in a for cycle
   
   supported types are described below:
     
     keys = name of the property
     values = respective parameters or their combinations
     
     currently supported:
     hs (hard spheres)
       pot_type = hs
       hs_diameter = array with hard spheres diameters (ncomponents x 1)
       hs_ij = array with hs diameter combinations (ncomponents x ncomponents)
       dis_index = array with indices of the diameters in the distance (ncomponents x ncomponents)
                         
     coulomb (electrostatics)
       pot_type = coulomb
       erf_alpha = value for the ng-renormalization coefficient alpha
       charge = array with charges (elementary charge unit, (ncomponents))
       chg_ij = array with products bjerrum_length * qi * qj (ncomponents x ncomponents)
     chg_inddip (1/r^4 charge-ind. dipole potential, Netz, Curr Opinion Colloid Interface Sci 2004, 9, 192-197
            has to be given together with coulomb potential (shares parameters)
       pot_type = chg_inddip
       charge = array with charges (elementary charge unit (ncomponents))
       alpha = array with excess polarizabilities (A^3, ncomponents)
       chgdip_ij = array with products (-1/2) * bjerrum_length * qi^2 * alphaj (ncomponents x ncomponents)
       
     lj (van der Waals with sigma and epsilon parameters)
       pot_type = lj
       comb_rule = dictionary with combination rules ('sigma', 'eps'; 'geom' = geometric, 'arit' = arithmetic average)
       sigma = array with sigma values in A (ncomponents) 
       epsilon = array with epsilon values  in kT (ncomponents, 1)
       sigma_ij = array with sigma combinations (ncomponents, ncomponents)
       epsilon_ij = array with epsilon combinations (ncomponents, ncomponents)
     
     pmf (potential of mean force calculated by some other method)
        pot_type = pmf
        pmf_r = array with distances (pmf is discretized; number_of_combinations x len(pmf))
        pmf_ij = array with the pmfs themselves (number_of_combinations x len(pmf))
        pmf_fill = dict, values/operations to be used outside the provided data (pmf_before, pmf_after)
        pmf_border = dict, indexes of border values for interpolation (min, max)
        pmf_interp = dict, interpolation type and function (type, func)
  """  
  
  # storage structure for all potential data
  parm = []
  parm_def = pyoz_const.parm_defaults()

  # pre-parsing of the data - dissect the input into individual contributions
  # find the keywords/blocks and check for correctness
  start_pos = []
  end_pos = []
  
  # we are looking for the following pairs
  # these are always the first words on the line (both of them!)
  # %potential ... %end_potential
  for i in range(len(lines)):
    if (match(r'%potential', lines[i])):
      start_pos.append(i)
    # end if (match(r'%potential', lines[i])):
    elif (match(r'%end_potential', lines[i])):
      end_pos.append(i)
    # end if (match(r'%end', lines[i])):
    #else
    #  pass
    #end else
  # end for i in range(len(lines)):
  
  # check if the read-in data are ok
  # position of blocks, misplaced statements, ...
  # if not, the function prints out a message and quits
  check_validity(lines, lineno_beg, start_pos, end_pos)

  # now process the potential one-by-one
  for pot_index in range(len(start_pos)):
    # lowercase data
    pot_data = lines[start_pos[pot_index] : end_pos[pot_index] + 1]
    # original case data
    pot_data_orig = lines_orig[start_pos[pot_index] : end_pos[pot_index] + 1]
    
    # get the potential type - it has to follow the %potential statement
    try:
      potential = (pot_data[0].split())[1]
    except IndexError, msg:
      print("error processing line %lu (%s)" % (i + lineno_beg, l))
      print("missing potential type")
      sys.exit(2)

    # starting line number of the respective section
    lineno_pot = lineno_beg + start_pos[pot_index]

    print("\tpotential type\t\t\t"),
    
    # now process the potential itself
    # ****************************************
    
    if (potential == 'hs'):
      # hard spheres
      print("hard spheres (%s)" % potential)
      
      pot_title = parm_def.hs_title
      # is done later
      #hs_diameter = zeros(syst['ncomponents'])
      hs_diameter_loaded = False
      # should individual hs combinations be used?
      hs_diam_comb = False
      
      hs_unit_orig = parm_def.hs_unit
      hs_unit = hs_unit_orig.lower()
      
      # the only allowed parameters are 'hs_diameter' (required) and 'hs_unit'
      # hs_unit can be A, m, nm, pm - it will always be converted to A
      for i in range(1, len(pot_data) - 1):
        # first and last lines, containing %potential...%end_potential construct, are skipped
        # pot_data contains input data converted to lowercase
        l = pot_data[i]
        # pot_data_orig contains input data in original case
        orig_l = pot_data_orig[i]
        
        # keyword is the first word on line
        # handle the case when the line is empty
        if (len(l) > 0):
          keyword = l.split()[0]
        else:
          keyword = ""
        
        try:
          if (match(r'#', l) or (l == "")):
            # commentary
            pass
          elif (match(r'pot_title$', keyword)):
            pot_title = " ".join(map(str, (orig_l.split())[1:]))
          elif (match(r'hs_diameter$', keyword)):
            # skip the first value (keyword), get the next ncomp values (diameters) and skip the rest
            # (might be some comment, ...)
            if (hs_diameter_loaded):
              print("hs diameters already loaded")
              exit(1)
            hs_diameter = zeros(syst['ncomponents'])
            hs_diameter[:] = float_(array( (l.split())[ 1:syst['ncomponents']+1 ] ))
            hs_diameter_loaded = True
          elif (match(r'hs_diam_comb$', keyword)):
            # enter the diameter combinations directly
            # skip the first value (keyword), get the next ncomp values (diameters) and skip the rest
            # (might be some comment, ...)
            if (hs_diameter_loaded):
              print("hs diameters already loaded")
              exit(1)
            hs_diameter = zeros(syst['ncomb'])
            hs_diameter[:] = float_(array( (l.split())[ 1:(syst['ncomb'] + 1)] ))
            hs_diameter_loaded = True
            hs_diam_comb = True
            #print("\tusing ")
          elif (match(r'hs_unit$', keyword)):
            hs_unit = (l.split())[1]
            hs_unit_orig = (orig_l.split())[1]
          else:
            raise unkKeyword((i,l))
          # end if (match...)
        # end try
        except ValueError, msg:
          print("error converting hs radii on line %lu (%s)" % (i + lineno_pot, l))
          print(msg)
          sys.exit(2)
        except IndexError, msg:
          print("line %lu with input data too short: %s" % (i + lineno_pot, l))
          print(msg)
          sys.exit(2)
        except unkKeyword, msg:
          print("error processing line %lu (%s)" % (msg.value[0] + lineno_beg, l))
          print("unknown keyword")
          sys.exit(2)
        # end try/except block
      # end for i in range(1, len(pot_data) - 1):
      
      # title
      print("\ttitle\t\t\t\t%s" % pot_title)

      # test for correctness
      # all diameters have to be > 0
      if (hs_diameter_loaded):
        if (hs_diameter.any() < 0.0):
          print("invalid entry for hs diameters! all values must be larger than 0.0")
          sys.exit(2)
      else:
        print("hs diameters not given in input file!")
        sys.exit(2)
      
      # check the unit and convert the value to Angstroms
      print("\tinput distance units\t\t"),
      # if there is some problem, it will be treated in the function!
      (unit_name, unit_factor) = _check_distance_unit(const, hs_unit, hs_unit_orig)
      print(unit_name)
      hs_diameter *= unit_factor
      
      # calculate combinations of hs diameters and store them in an array
      # calculate also the indices of the hs diameters in the distance array
      hs_ij = zeros((syst['ncomponents'],syst['ncomponents']))
      dis_index = zeros((syst['ncomponents'],syst['ncomponents']), dtype=int32)
      # number of the combination
      nc = 0
      for i in range(syst['ncomponents']):
        for j in range(i, syst['ncomponents']):
          if (hs_diam_comb):
            # diameters were entered on the commandline
            hs_ij[i,j] = hs_diameter[nc]
            hs_ij[j,i] = hs_diameter[nc]
          else:
            # calculate the diameters
            hs_ij[i,j] = 0.5 * (hs_diameter[i] + hs_diameter[j])
            hs_ij[j,i] = 0.5 * (hs_diameter[i] + hs_diameter[j])
          #end if (hs_diam_comb):
          
          if (hs_ij[i,j] > ctrl['max_r']):
            print("hs contact distance (%f) beyond the discretization interval (max=%f)" % (hs_ij[i,j], ctrl['max_r']))
            exit(1)

          # calculate the index corresponding to contact distance
          # not 100% correct - how to treat non-integer multiples of step size?
          dis_index[i,j] = int(round(hs_ij[i,j] / ctrl['deltar'])) - 1
          dis_index[j,i] = int(round(hs_ij[i,j] / ctrl['deltar'])) - 1

          # debug output
          #rest,result = modf(hs_ij[i,j] / ctrl['deltar'])
          if (hs_ij[i,j] != ctrl['deltar'] * (dis_index[i,j] + 1)):
            print("\tWARNING: hard sphere diameter (combination %u,%u)\n\t\t\t%f not an integer multiple of step size\n\t\t\t%f used instead!" % ((i + 1), (j + 1), hs_ij[i,j], ctrl['deltar'] * (dis_index[i,j] + 1)))
            print("\tWARNING: detection not 100%, but functionality of program not affected!")
          #else:
          #  print("good")
          # increase the combination number
          nc += 1
        # end for j in range(ncomponents)
      # end for i in range(ncomponents)

      #print("hs-combinations")
      #print(hs_ij)
      # add the radii and combined hs radii into the parameters list
      parm.append({})
      parm[-1]['pot_type'] = 'hs'
      parm[-1]['pot_title'] = pot_title
      parm[-1]['hs_diameter'] = hs_diameter
      parm[-1]['hs_ij'] = hs_ij
      parm[-1]['dis_index'] = dis_index
    
      # print hard sphere diameters
      print("\ths radii (Angstroms, "),
      if (hs_diam_comb):
        print("direct)\t"),
      else:
        print("calc.)\t"),
      for i in range(syst['ncomponents']):
        for j in range(i, syst['ncomponents']):
          print("%f " % hs_ij[i,j]),
      print("\n")
      
    # end if (potential == 'hs')
    
    # ****************************************

    elif (potential == 'coulomb'):
      # coulomb potential
      print("Coulomb potential (%s)" % potential)

      # PY is not valid with long-ranged potentials!
      if (syst['closure_name'] == 'py'):
        print("\tuse of long-ranged Coulomb potential is not consistent")
        print("\t  with the Percus-Yevick approximation!")
        sys.exit(2)

      pot_title = parm_def.coul_title
      charge = zeros(syst['ncomponents'])
      charge_loaded = False
      
      # should individual alpha combinations be used?
      erf_alpha = zeros((syst['ncomponents'],syst['ncomponents']))
      erf_alpha_comb = False
      erf_alpha_loaded = False
      
      # polarizabilities for 1/r^4 charge-induced dipole potential
      chg_inddip = False
      dip_alpha = zeros(syst['ncomponents'])
      dip_alpha_loaded = False
      
      # allowed parameters 'charge' (required)
      #
      # 'chg_inddip' if charge-induced dipole interaction according to netz is to be used
      # then 'alpha' with values of excess polarizabilities
      
      for i in range(1, len(pot_data) - 1):
        # first and last lines, containing %potential...%end_potential construct, are skipped
        # pot_data contains input data converted to lowercase
        l = pot_data[i]
        # pot_data_orig contains input data in original case
        orig_l = pot_data_orig[i]

        # keyword is the first word on line
        # handle the case when the line is empty
        if (len(l) > 0):
          keyword = l.split()[0]
        else:
          keyword = ""
        
        try:
          if (match(r'#', l) or (l == "")):
            # commentary
            pass
          elif (match(r'pot_title$', keyword)):
            pot_title = " ".join(map(str, (orig_l.split())[1:]))
          elif (match(r'erf_alpha$', keyword)):
            # skip the first value (keyword), get the next value and skip the rest
            # (might be some comment, ...)
            if (erf_alpha_loaded):
              print("alpha value(s) already specified!")
              exit(1)
            erf_alpha_val = float((l.split())[1])
            erf_alpha_loaded = True
            erf_alpha_comb = False
          elif (match(r'erf_alpha_comb$', keyword)):
            # enter the erf combinations directly
            # skip the first value (keyword), get the next ncomb values and skip the rest
            # (might be some comment, ...)
            if (erf_alpha_loaded):
              print("alpha value(s) already specified!")
              exit(1)
            erf_alpha_val = zeros(syst['ncomb'])
            erf_alpha_val[:] = float_(array( (l.split())[ 1:(syst['ncomb'] + 1)] ))
            erf_alpha_loaded = True
            erf_alpha_comb = True
          elif (match(r'charge$', keyword)):
            # skip the first value (keyword), get the next ncomp values (charges) and skip the rest
            # (might be some comment, ...
            charge[:] = float_(array( (l.split())[ 1:syst['ncomponents']+1 ] ))
            charge_loaded = True
          elif (match(r'chg_inddip$', keyword)):
            chg_inddip = True
          elif (match(r'dip_alpha$', keyword)):
            # skip the first value (keyword), get the next ncomp values (excess polarizabilities) and skip the rest
            dip_alpha[:] = float_(array( (l.split())[ 1:syst['ncomponents']+1 ] ))
            dip_alpha_loaded = True
          else:
            raise unkKeyword((i,l))
          # end if (match...)
        # end try
        except ValueError, msg:
          print("error converting charges on line %lu (%s)" % (i + lineno_pot, l))
          print(msg)
          sys.exit(2)
        except IndexError, msg:
          print("line %lu with input data too short: %s" % (i + lineno_pot, l))
          print(msg)
          sys.exit(2)
        except unkKeyword, msg:
          print("error processing line %lu (%s)" % (msg.value[0] + lineno_beg, l))
          print("unknown keyword")
          sys.exit(2)
        # end try/except block
      # end for i in range(1, len(pot_data) - 1):
      
      # title
      print("\ttitle\t\t\t\t%s" %  pot_title)

      # test for correctness
      # are the charges zero?
      if (charge_loaded):
        if (charge.all() == 0.0):
          print("warning: all charges are zero!")
      else:
        print("charges not given in input file!")
        sys.exit(2)
      
      # system has to be electro-neutral
      # due to rounding errors, we have to put some finite number instead of 0.0
      total_charge = (charge * syst['conc_mol']).sum()
      #if (total_charge != 0.0):
      if (abs(total_charge) > 1e-10):
        print("system is not electrically neutral! charge*conc=", total_charge, "elm.chg mol/L")
        sys.exit(2)
      
      # charges in elementary charge units
      #charge = array([1.0, -1.0])
      # bjerrum_length * qi*qj products
      chg_ij = zeros((syst['ncomponents'],syst['ncomponents']))
      for i in range(syst['ncomponents']):
        for j in range(syst['ncomponents']):
          chg_ij[i,j] = const.bjerrum_length * charge[i] * charge[j]
          # end for j in range(ncomponents)
        # end for i in range(ncomponents)
      #print("charge combinations multiplied by Bjerrum length")
      #print(chg_ij)
      
      # add the charges and products into the parameters list
      parm.append({})
      parm[-1]['pot_type'] = 'coulomb'
      parm[-1]['pot_title'] = pot_title
      parm[-1]['charge'] = charge
      parm[-1]['chg_ij'] = chg_ij

      # print charges
      print("\tcharges (elementary)\t\t"),
      for i in range(syst['ncomponents']):
        print("%f " % charge[i]),
      print("")

      # process the Ng-renormalization alpha parameter for the error function

      # firstly prepare the arrat erf_alpha to be later stored to potential parameters
      if (erf_alpha_loaded):
        # the value(s) was/were loaded
        # just need to fill in the respective array
        if (erf_alpha_comb):
          ind_alpha = 0
          # individual combinations were entered
          for i in range(syst['ncomponents']):
            for j in range(i,syst['ncomponents']):
              erf_alpha[i,j] = erf_alpha_val[ind_alpha]
              erf_alpha[j,i] = erf_alpha_val[ind_alpha]
              ind_alpha += 1
            #end for j in range(i,syst['ncomponents']):
          #end for i in range(syst['ncomponents']):
        else:
          # just one constant value for all combinations
          erf_alpha[:] = erf_alpha_val
      # end if(erf_alpha_loaded), else follows
      else:
        # alpha was not loaded
        # we will calculate it from the debye screening length kappa
        # our alpha = kappa/2
        # got this from luc, will have to find justification
        # he is using analytical msa solution for hs+coulomb (valid only for hard potentials)
        # and setting alpha so that the short range Gamma vanishes at k=0
        # the kappa is more or less kappa/2
        # we have also soft potentials => we cannot use the msa way
        
        # we want to get kappa, defined as sqrt(8 pi lambda_B Na I)
        # lambda_B = Bjerrum length, I = ionic strength = 0.5 sum(c_i z_i^2)
        ionic_strength = 0.5
        erf_adjusted = False
        for i in range(syst['ncomponents']):
          ionic_strength *= charge[i]**2 * syst['conc_mol'][i]
        # calculate kappa. the factor 1e-4 comes from conversion - na [10^23] * I [10^-27 mol/a^3] * l_b [a]
        # the factors 10^x are removed from the constants in the const class
        kappa = sqrt(8.0 * pi * const.bjerrum_length * const.Na * ionic_strength * 1e-4)
        # kappa=0.050 is empirically shown to be the border where the ng-renormalization
        # still works with alpha=kappa/2
        if (kappa < parm_def.erf_min_kappa):
          kappa = parm_def.erf_min_kappa
          erf_adjusted = True
        # get the alpha as kappa/2
        erf_alpha[:] = kappa/2
      # end if (erf_alpha_loaded) else

      # store the values and provide some output
      parm[-1]['erf_alpha'] = erf_alpha
      print("\talpha for Ng "),
      # print erf setting
      if (not erf_alpha_loaded):
        if (erf_adjusted):
          print("(kappa too small)\t"),
        else:
          print("(from kappa)\t"),
      else:
        if (erf_alpha_comb):
          print("(combinations)\t"),
        else:
          print("(one value)\t"),
      # print the values themselves
      for i in range(syst['ncomponents']):
        for j in range(i, syst['ncomponents']):
          print("%.3f " % erf_alpha[i,j]),
      print("\n")

      # chg_inddip required?
      if (chg_inddip):
        pot_title = parm_def.chg_inddip_title
        print("\tpotential type\t\t\t1/r^4 charge-induced dipole (chg_inddip)")
        if (not dip_alpha_loaded):
          print("charge-induced dipole requested but no polarizabilities entered!")
          sys.exit(2)

        # excess polarizabilities are in A^3 units
        # let's now calculate the (-1/2) bjerrum_length * qi^2 * alphaj
        chgdip_ij = zeros((syst['ncomponents'],syst['ncomponents']))
        for i in range(syst['ncomponents']):
          for j in range(syst['ncomponents']):
            chgdip_ij[i,j] = -0.5 * const.bjerrum_length * charge[i]**2 * dip_alpha[j]
            # end for j in range(ncomponents)
          # end for i in range(ncomponents)
        #print("charge-induced dipole factors")
        #print(chgdip_ij)

        parm.append({})
        parm[-1]['pot_type'] = 'chg_inddip'
        parm[-1]['pot_title'] = pot_title
        parm[-1]['charge'] = charge
        parm[-1]['chgdip_ij'] = chgdip_ij
        parm[-1]['alpha'] = dip_alpha

        # print polarizabilities
        print("\texcess polarizability (A^3)\t"),
        for i in range(syst['ncomponents']):
          print("%f " % dip_alpha[i]),
        print("\n")
      # end if (chg_inddip):
    
    # end if (potential == 'coulomb')

    # ****************************************
  
    elif (potential == 'lj'):
      # van der Waals potential
      print("van der Waals potential (%s)" % potential)

      pot_title = parm_def.lj_title
      # sigma - distance where the potential is zero
      # epsilon - depth of the potential minimum
      sigma = zeros(syst['ncomponents'])
      epsilon = zeros(syst['ncomponents'])
      sigma_loaded = False
      epsilon_loaded = False
      sigma_unit_orig = parm_def.lj_sigma_unit
      sigma_unit = sigma_unit_orig.lower()
      epsilon_unit_orig = parm_def.lj_epsilon_unit
      epsilon_unit = epsilon_unit_orig.lower()
      # combination rules for mixed A-B interactions
      comb_rule = {'sigma':parm_def.lj_sigma_rule, 'epsilon':parm_def.lj_epsilon_rule}

      # the allowed parameters are 'sigma' amd 'epsilon' (required) 
      # 'sigma_unit' amd 'epsilon_unit'
      # 'sigma_rule', 'epsilon_rule'
      # sigma_unit can be A, m, nm, pm - it will always be converted to A
      # epsilon_unit can be eV, kT, kJ_mol, kcal_mol - it will always be converted to kT
      # rules can be either 'arit' or 'geom' for arithmetic and geometric averages
      for i in range(1, len(pot_data) - 1):
        # first and last lines, containing %potential...%end_potential construct, are skipped
        # pot_data contains input data converted to lowercase
        l = pot_data[i]
        # pot_data_orig contains input data in original case
        orig_l = pot_data_orig[i]

        # keyword is the first word on line
        # handle the case when the line is empty
        if (len(l) > 0):
          keyword = l.split()[0]
        else:
          keyword = ""
        
        try:
          if (match(r'#', l) or (l == "")):
            # commentary
            pass
          elif (match(r'pot_title$', keyword)):
            pot_title = " ".join(map(str, (orig_l.split())[1:]))
          elif (match(r'sigma$', keyword)):
            # skip the first value (keyword), get the next ncomp values (sigma) and skip the rest
            # (might be some comment, ...
            sigma[:] = float_(array( (l.split())[ 1:syst['ncomponents']+1 ] ))
            sigma_loaded = True
          elif (match(r'epsilon$', keyword)):
            # skip the first value (keyword), get the next ncomp values (epsilon) and skip the rest
            # (might be some comment, ...
            epsilon[:] = float_(array( (l.split())[ 1:syst['ncomponents']+1 ] ))
            epsilon_loaded = True
          elif (match(r'sigma_unit$', keyword)):
            sigma_unit = (l.split())[1]
            sigma_unit_orig = (orig_l.split())[1]
          elif (match(r'epsilon_unit$', keyword)):
            epsilon_unit = (l.split())[1]
            epsilon_unit_orig = (orig_l.split())[1]
          elif (match(r'sigma_rule$', keyword)):
            comb_rule['sigma'] = (l.split())[1]
            sigma_rule_orig = (orig_l.split())[1]
          elif (match(r'epsilon_rule$', keyword)):
            comb_rule['epsilon'] = (l.split())[1]
            epsilon_rule_orig = (orig_l.split())[1]
          else:
            raise unkKeyword((i,l))
          # end if (match...)
        # end try
        except ValueError, msg:
          print("error converting sigma/epsilon on line %lu (%s)" % (i + lineno_pot, l))
          print(msg)
          sys.exit(2)
        except IndexError, msg:
          print("line %lu with input values too short: %s" % (i + lineno_pot, l))
          print(msg)
          sys.exit(2)
        except unkKeyword, msg:
          print("error processing line %lu (%s)" % (msg.value[0] + lineno_beg, l))
          print("unknown keyword")
          sys.exit(2)
        # end try/except block
      # end for i in range(1, len(pot_data) - 1):

      # title
      print("\ttitle\t\t\t\t%s" % pot_title)

      # test for correctness
      # sigma values have to be given, have to be > 0
      if (sigma_loaded):
        if (sigma.any() < 0.0):
          print("invalid entry for sigma! all values must be larger than 0.0")
          sys.exit(2)
      else:
        print("sigma values not given in input file!")
        sys.exit(2)
      
      # epsilon values have to be given
      if (not epsilon_loaded):
        print("epsilon values not given in input file!")
        sys.exit(2)

      # check the units and convert the values to Angstroms and kT
      print("\tsigma units\t\t\t"),
      # if there is some problem, it will be treated in the function!
      (unit_name, unit_factor) = _check_distance_unit(const, sigma_unit, sigma_unit_orig)
      print(unit_name)
      sigma *= unit_factor

      print("\tepsilon units\t\t\t"),
      # if there is some problem, it will be treated in the function!
      (unit_name, unit_factor) = _check_energy_unit(const, epsilon_unit, epsilon_unit_orig)
      print(unit_name)
      epsilon *= unit_factor
      
      # combination rules handling
      print("\tcombination rules")
      # sigma
      print("\t\tsigma\t\t\t"),
      if ((comb_rule['sigma']) == 'arit'):
        print('arithmetic average')
      elif ((comb_rule['sigma']) == 'geom'):
        print('geometric average')
      else:
        # unknown combination rule
        print("unknown combination rule %s" % sigma_rule_orig)
        print("supported rules arit (arithmetic) and geom (geometric)")
        sys.exit(2)
      # epsilon
      print("\t\tepsilon\t\t\t"),
      if ((comb_rule['epsilon']) == 'arit'):
        print('arithmetic average')
      elif ((comb_rule['epsilon']) == 'geom'):
        print('geometric average')
      else:
        # unknown combination rule
        print("unknown combination rule %s" % epsilon_rule_orig)
        print("supported rules arit (arithmetic) and geom (geometric)")
        sys.exit(2)
      # end combination rules handling
      
      #sigma = array([4.0, 6.0])
      # epsilon parameter in kJ/mol
      #epsilon = array([0.1, 0.1])
      #print("epsilon before conv", epsilon)
      # several units supported? - kcal/mol, kJ/mol, eV, ...?
      # convert to kT
      #epsilon *= const.conv_kJmoltokT
      #epsilon *= const.conv_kcalmoltokT
      #print("epsilon after conv", epsilon)
      # combination rules for sigma and epsilon (arithmetic, geometric)
      #comb_rule = {'sigma':'arit', 'epsilon':'geom'}

      sigma_ij = zeros((syst['ncomponents'],syst['ncomponents']))
      eps_ij = zeros((syst['ncomponents'],syst['ncomponents']))
      for i in range(syst['ncomponents']):
        for j in range(syst['ncomponents']):
          # calculate sigma combinations
          if (comb_rule['sigma'] == 'arit'):
            sigma_ij[i,j] = 0.5 * (sigma[i] + sigma[j])
          else:
            sigma_ij[i,j] = sqrt(sigma[i] * sigma[j])

          # calculate epsilon combinations
          if (comb_rule['epsilon'] == 'arit'):
            eps_ij[i,j] = 0.5 * (epsilon[i] + epsilon[j])
          else:
            eps_ij[i,j] = sqrt(epsilon[i] * epsilon[j])
        # end for j in range(ncomponents)
      # end for i in range(ncomponents)

      #print("van der Waals combinations")
      #print(sigma_ij)
      #print("epsilon in kT")
      #print(eps_ij)

      # add the sigma, epsilon, and combination rules to the parameters list
      parm.append({})
      parm[-1]['pot_type'] = 'lj'
      parm[-1]['pot_title'] = pot_title
      parm[-1]['sigma'] = sigma
      parm[-1]['epsilon'] = epsilon
      parm[-1]['comb_rule'] = comb_rule
      parm[-1]['sigma_ij'] = sigma_ij
      parm[-1]['eps_ij'] = eps_ij
      
      # print lj parameters
      print("\tsigma (Angstroms)\t\t"),
      for i in range(syst['ncomponents']):
        print("%f " % sigma[i]),
      print("")
      print("\tepsilon (kT)\t\t\t"),
      for i in range(syst['ncomponents']):
        print("%f " % epsilon[i]),
      print("\n")      
 
    # end if (potential == 'lj')
    
    # ****************************************
    
    elif (potential == 'pmf'):
      # PMF from file
      print("PMF from external file(s) (%s)" % potential)

      pot_title = parm_def.pmf_title
      # single file?
      pmf_single = None
      # pmf filename - single file
      pmf_filename = None
      # pmf filename - multiple files
      pmf_filenames = None
      # energy unit
      pmf_unit_orig = parm_def.pmf_unit
      pmf_unit = pmf_unit_orig.lower()
      # distance unit
      pmf_distance_unit_orig = parm_def.pmf_distance_unit
      pmf_distance_unit = pmf_distance_unit_orig.lower()

      # interpolation scheme
      pmf_interp_type = parm_def.pmf_interp_type
      # maximum and minimum values to be added before and after read-in pmf
      # for r->0
      pmf_before = parm_def.pmf_before
      # for r->inf
      pmf_after = parm_def.pmf_after
      # number of lines to be skipped
      pmf_linetoskip = 0
      
      # the allowed parameters are 'pmf_filename' (required) and
      # 'pmf_unit', 'lineskip' (no of lines to skip), 'interp_type' (interpolation scheme)
      # 'pmf_before' and 'pmf_after' (values to be added before and after the read-in data)
      # these are also in pmf_unit!
      # pmf_unit can be eV, kT, kJ_mol, kcal_mol - it will always be converted to kT
      for i in range(1, len(pot_data) - 1):
        # first and last lines, containing %potential...%end_potential construct, are skipped
        # pot_data contains input data converted to lowercase
        l = pot_data[i]
        # pot_data_orig contains input data in original case
        orig_l = pot_data_orig[i]

        # keyword is the first word on line
        # handle the case when the line is empty
        if (len(l) > 0):
          keyword = l.split()[0]
        else:
          keyword = ""
        
        try:
          if (match(r'#', l) or (l == "")):
            # commentary
            pass
          elif (match(r'pot_title$', keyword)):
            pot_title = " ".join(map(str, (orig_l.split())[1:]))
          elif (match(r'pmf_filename$', keyword)):
            # skip the first value (keyword), get the next value and skip the rest
            pmf_filename = (l.split())[1]
            pmf_single = True
          elif (match(r'pmf_filenames$', keyword)):
            pmf_filenames = (l.split())[1:]
            pmf_single = False
          elif (match(r'pmf_unit$', keyword)):
            pmf_unit = (l.split())[1]
            pmf_unit_orig = (orig_l.split())[1]
          elif (match(r'pmf_distance_unit$', keyword)):
            pmf_distance_unit = (l.split())[1]
            pmf_distance_unit_orig = (orig_l.split())[1]
          elif (match(r'lineskip$', keyword)):
            pmf_linetoskip = int((l.split())[1])
          elif (match(r'interp_type$', keyword)):
            pmf_interp_type = (l.split())[1]
            pmf_interp_type_orig = (orig_l.split())[1]
          elif (match(r'pmf_before$', keyword)):
            pmf_before = float((l.split())[1])
          elif (match(r'pmf_after$', keyword)):
            pmf_after = float((l.split())[1])
          else:
            raise unkKeyword((i,l))
          # end if (match...)
        # end try
        except IndexError, msg:
          print("line %lu with input data too short: %s" % (i + lineno_pot, l))
          print(msg)
          sys.exit(2)
        except ValueError, msg:
          print("error converting numerical input data on line %lu (%s)" % (i + lineno_pot, l))
          print(msg)
          sys.exit(2)
        except unkKeyword, msg:
          print("error processing line %lu (%s)" % (msg.value[0] + lineno_beg, l))
          print("unknown keyword")
          sys.exit(2)
        # end try/except block
      # end for i in range(1, len(pot_data) - 1):

      # title
      print("\ttitle\t\t\t\t%s" %  pot_title)
      
      # test for correctness
      # filename with all pmf combinations has to be given
      # or list of filenames (one for every combination)
      if ((pmf_filename == None) and (pmf_filenames == None)):
        print("pmf filename(s) not given in input file!")
        sys.exit(2)
      if ((pmf_filename != None) and (pmf_filenames != None)):
        print("cannot have both pmf_filename and pmf_filenames in the same section!")
        sys.exit(2)
      
      # check the energy units and set the conversion factor to kT
      pmf_conversion = 1.0
      print("\tpmf units\t\t\t"),
      # if there is some problem, it will be treated in the function!
      (unit_name, unit_factor) = _check_energy_unit(const, pmf_unit, pmf_unit_orig)
      print(unit_name)
      pmf_conversion *= unit_factor

      # check the distamce units and convert the values to Angstroms and kT
      pmfdist_conversion = 1.0
      print("\tpmf distance units\t\t"),
      # if there is some problem, it will be treated in the function!
      (unit_name, unit_factor) = _check_distance_unit(const, pmf_distance_unit, pmf_distance_unit_orig)
      print(unit_name)
      pmfdist_conversion *= unit_factor

      # test interpolation scheme
      print("\tinterpolation scheme\t\t"),
      if (pmf_interp_type == 'linear'):
        print("linear")
        from pyoz_misc import interpolate_linear as interpolate
      elif (pmf_interp_type == 'cosine'):
        print("cosine")
        from pyoz_misc import interpolate_cosine as interpolate
      else:
        # unknown unit
        print("unknown interpolation scheme %s" % pmf_interp_type_orig)
        print("supported schemes are linear and cosine")
        sys.exit(2)
      # set the function
      pmf_interp_func = interpolate

      # added values
      print("\tvalues added (r->0)\t\t%f" % pmf_before)
      print("\tvalues added (r->inf)\t\t%f" % pmf_after)
      
      # the file(s) will be read now here and the basic validity check will be performed
      # in case the single file is requested, the format is (without commas)
      #     r,11,12,13,...1n,22,23,...2n,33,...(n-1)(n-1),(n-1)n,nn
      # in separate files it is
      #     r 11 (file 1), r 12 (file 2), ...

      # the following could be achieved by using something like
      #   open file
      #   a = fromfile(file, sep=" ", float)
      #   a.shape(ncom, ncomp, len(datasets))
      # which raises an exception when something fails, but unfortunately without line number
      # the used approach yields line number where the some problem is located
      # and is, therefore, more user-friendly (albeit slower, but it doesn't matter here...)
      # another problem with fromfile is that it would read only the combinations as shown above
      # but we actually need all ncomp x ncomp combinations at the end!

      # how many unique combinations we have?
      # in one file, this gives the number of columns (+1 for distance)
      # otherwise, this gives the number of files
      # distance, combinations * values
      num_comb = syst['ncomb']
      
      # in case separate files were requested, check for correctness
      if (pmf_single == False):
        if (len(pmf_filenames) != num_comb):
          print("not enough file names with individual pmf. needed %u got %u" % (num_comb, len(pmf_filenames)))
          sys.exit(2)

      print("\tattempting load of pmf")
      
      try:
        # pmf_r and pmf_ij are lists containing arrays to store pmf data
        # the list is used due to the fact, that in case of separate files,
        # the length of the data isn't necesarilly the same
        if (pmf_single == True):
          print("\t\tcomplete pmf (%s)" % pmf_filename)

          # try to open the file, exceptions are handled below
          fr = open(pmf_filename, "r")
          # read all lines
          datasets_orig = fr.readlines()
          # remove the first pmf_linetoskip values
          datasets_orig = datasets_orig[pmf_linetoskip:]
          datasets = []
          # strip, remove all empty lines
          for line in datasets_orig:
            nline = line.strip()
            if (nline != ''):
              datasets.append(nline)
          # end for line in datasets_orig
          
          if (len(datasets) < 2):
            # we need at least 2 points
            raise pmfError(("need at least 2 points, got %u " % len(datasets[0])))
          fr.close()
          # pre-allocate arrays
          # distance
          pmf_r = [ None ] * num_comb
          # and the respective pmf values
          pmf_ij = [ None ] * num_comb
          for i in range(num_comb):
            pmf_r[i] = zeros(len(datasets))
            pmf_ij[i] = zeros(len(datasets))
          #end for i in range(num_comb):            

        else:
          # pre-allocate arrays
          datasets = []
          pmf_r = [ None ] * num_comb
          pmf_ij = [ None ] * num_comb
          for i in range(num_comb): 
            print("\t\tindividual pmf %u (%s)" % (i, pmf_filenames[i]))
            # try to open the file, exceptions are handled below
            fr = open(pmf_filenames[i], "r")
            # read all lines
            datasets_orig = fr.readlines()
            # remove the first pmf_linetoskip values
            datasets_orig = datasets_orig[pmf_linetoskip:]
            datasets.append([])
            # strip, remove all empty lines
            for line in datasets_orig:
              nline = line.strip()
              if (nline != ''):
                datasets[i].append(nline)
            # end for line in datasets_orig
            
            if (len(datasets[i]) < 2):
              # we need at least 2 points
              raise pmfError(("need at least 2 points, got %u " % len(datasets[i])))
            fr.close()
            # modify the arrays to correspond to the length of the pmf
            # distance
            #print("pre-allocation: %u,%u" % (num_comb, len(datasets)))
            pmf_r[i] = zeros(len(datasets[i]))
            # and the respective pmf values
            pmf_ij[i] = zeros(len(datasets[i]))
          #end for i in range(num_comb): 
        # end if (pmf_single == True) - reading in the data
                
        # now process the data - spilt to distances and values
        print("\tdata processing")

        # min and max r values
        # has to be the first and last value, respectively
        pmf_rmin = [ 0.0 ] * num_comb
        pmf_rmax = [ 0.0 ] * num_comb        
        
        # name of the currently processed file
        c_name = ""
        # inded of column with pmf data in that file; r is always the first column with index 0
        c_column = 1
        # current dataset
        c_dataset = None
        
        # the following code is certainly unoptimal (at least when single file pmfs are used)
        # but it doesn't matter - this is not performance-critical part
        
        # required number of values per line
        if (pmf_single == True):
          # distance + pmf for combinations
          num_val = num_comb + 1
        else:
          # just distance - pmf pair
          num_val = 2
        
        for i in range(num_comb):
          if (pmf_single == True):
            c_name = pmf_filename
            c_column = i + 1
            c_dataset = datasets
            #print("c_column", c_column)
          else:
            c_name = pmf_filenames[i]
            c_column = 1
            c_dataset = datasets[i]
            
          print("\t\tcombination %u of %u (file %s, columns 1 and %u)" % (i + 1, num_comb, c_name, c_column + 1))
          # split to distance and values
          # index is (linenumber - pmf_linetoskip - 1)
          index = 0
          for line in c_dataset:
            sline = line.split()
            if (len(sline) < num_val):
              # something is not correct
              raise pmfError(("expected %u values, got %u: %s" % (num_val, len(sline), line)))
            # end if (len(sline) != num_val)
            elif (len(sline) > num_val):
              # skip extra data but issue a warning
              print("\t\tWARNING: extra data on line. expected %u values, got %u: %s" % (num_val, len(sline), line))
              sline = sline[:num_val]
            # convert r and the respective pmf value to float and kT/A units
            # store in the arrays
            #print("string\ti=%u\tindex=%u\tr=%s\t\tpmf=%s" % (i, index, sline[0], sline[c_column]))
            #print("problem %f" % pmf_r[i][index])
            pmf_r[i][index] = float(sline[0]) * pmfdist_conversion
            pmf_ij[i][index] = float(sline[c_column]) * pmf_conversion
            #print("val\t\t\t\tr=%f\tpmf=%f" % (pmf_r[i][index]/pmfdist_conversion, pmf_ij[i][index]/pmf_conversion))

            # test the r value for correctness
            if (index == 0):
              pmf_rmin[i] = pmf_r[i][index]
              pmf_rmax[i] = pmf_r[i][index]
            else:
              # the first r value has to be the smallest
              # and it must increase among the datasets
              if ((pmf_r[i][index] <= pmf_rmin[i]) or (pmf_r[i][index] <= pmf_rmax[i])):
                raise pmfError(("datasets have to be sorted in increasing order: around line %s" % line))
              else:
                pmf_rmax[i] = pmf_r[i][index]
            # end if (index == 0) - testing for correctness
   
            #print("val\t\t\tind=%u\tr=%f\tpmf=%f" % (i, pmf_r[i][index]/pmfdist_conversion, pmf_ij[i][index]/pmf_conversion))

            # increment the line counter and go to the next line
            index += 1
          # end for line in c_dataset
        # end for i in range(num_comb)
      # end try
      except IOError, errmsg:
          print("load of PMF failed - IO error: %s" % errmsg)
          sys.exit(2)
      except pmfError, msg:
          print("load of PMF failed: %s" % msg.value)
          sys.exit(2)
      # end except pmfError - PMF-related problems
      except ValueError, errmsg:
          print("load of PMF failed - conversion error: %s" % errmsg)
          print("around line %u" % (index + pmf_linetoskip + 1))
          sys.exit(2)
      # end except ValueError - conversion of strings to floats failed
      else:
        print("\t\tloaded pmf data succesfully")
        # print some statistics?
      # end try/except/else block 

      # store the values
      parm.append({})
      parm[-1]['pot_type'] = 'pmf'
      parm[-1]['pot_title'] = pot_title
      # potential of mean force distances
      parm[-1]['pmf_r'] = pmf_r
      # and the pmf itself
      parm[-1]['pmf_ij'] = pmf_ij
      # values to be used outside the region where the data is available
      parm[-1]['pmf_fill'] = {'before':pmf_before, 'after':pmf_after}
      # border index in the r array
      parm[-1]['pmf_border'] = {'min':pmf_rmin, 'max':pmf_rmax}
      # store the interpolation function to be used in the dictionary
      parm[-1]['pmf_interp'] = {'type':pmf_interp_type, 'func':pmf_interp_func}

      # add empty line to output
      print("")
  
    # end if (potential == 'pmf')
    
    # ****************************************
    
    else:
      print("unknown potential (%s)!" % potential)
      sys.exit(2)
    # end if potential is not known
  # end for pot_index in range(len(start_pos)):
  # i.e., for all potentials in the input file

  # return the parameter data
  return(parm)

# **********************************************************************************************  

def parse_outp(cmdline, lines, lines_orig, lineno_beg, lineno_end):
  """
    parse the output statements  
  """
  
  outp_def = pyoz_const.outp_defaults()
  outp = {}

  # pre-set the default values, description below or in the pyoz_const.py
  # Gamma related variables
  # always written at least in the end
  G_ij_write =    outp_def.G_ij_write
  G_ij_name =     outp_def.G_ij_name
  G_ij_savefreq = outp_def.G_ij_savefreq
  G_ij_binary =   outp_def.G_ij_binary
  # g(r) related variables
  # always written at least in the end
  g_ij_write =    outp_def.g_ij_write
  g_ij_name =     outp_def.g_ij_name
  # potential related variables
  # not written when not requested
  # pair potential
  U_ij_write =    outp_def.U_ij_write
  U_ij_name =     outp_def.U_ij_name
  # pair potential + Gamma
  Utot_ij_write =    outp_def.Utot_ij_write
  Utot_ij_name =     outp_def.Utot_ij_name
  # direct correlatio function
  c_ij_write =    outp_def.c_ij_write
  c_ij_name =     outp_def.c_ij_name
  # structure factors
  S_ij_write =    outp_def.S_ij_write
  S_ij_name =     outp_def.S_ij_name
  
  # apply correction according to the commandline
  if (cmdline['name'] != None):
    G_ij_name = cmdline['name'] + outp_def.gamma_suffix
    g_ij_name = cmdline['name'] + outp_def.gr_suffix
    c_ij_name = cmdline['name'] + outp_def.cr_suffix
    S_ij_name = cmdline['name'] + outp_def.S_suffix
    U_ij_name = cmdline['name'] + outp_def.ur_suffix
    Utot_ij_name = cmdline['name'] + outp_def.urtot_suffix
  
  # parse the inputs and (possibly) overwrite the input data
  try:
    for i in range(len(lines)):
      # let's simplify the commands
      # we don't do in in lines in order to have line numbers
      # lines contains data converted all to lowercase
      l = lines[i]
      # lines_orig contains data with original case - used for filenames, ...
      orig_l = lines_orig[i]
      
      # keyword is the first word on line
      # handle the case when the line is empty
      if (len(l) > 0):
        keyword = l.split()[0]
      else:
        keyword = ""
      
      # now check whether a keyword is recognized
      if (match(r'#', l) or (l == "")):
        # commentary
        pass

      # **************************************************************************
      # Gamma
      # any of the three following keywords switches the output of Gamma
      elif (match(r'gamma_write$', keyword)):
        G_ij_write = True
      elif (match(r'gamma_name$', keyword)):
        G_ij_write = True
        G_ij_name = (orig_l.split())[1]
      elif (match(r'gamma_freq$', keyword)):
        G_ij_freq = int((l.split())[1])
        G_ij_write = True
        if (G_ij_freq < 0):
          raise badValue((i, "the Gamma save frequency cannot be negative"))
      elif (match(r'gamma_binary$', keyword)):
        G_ij_binary = True
      elif (match(r'gamma_text$', keyword)):
        G_ij_binary = False

      # **************************************************************************
      # g(r)

      elif (match(r'gr_write$', keyword)):
        g_ij_write = True
      elif (match(r'gr_name$', keyword)):
        g_ij_write = True
        g_ij_name = (orig_l.split())[1]
      
      # **************************************************************************
      # u(r) - total potential

      elif (match(r'ur_write$', keyword)):
        U_ij_write = True
      elif (match(r'ur_name$', keyword)):
        U_ij_write = True
        U_ij_name = (orig_l.split())[1]
      elif (match(r'urtot_write$', keyword)):
        Utot_ij_write = True
      elif (match(r'urtot_name$', keyword)):
        Utot_ij_write = True
        Utot_ij_name = (orig_l.split())[1]
      
      # **************************************************************************
      # c(r)

      elif (match(r'cr_write$', keyword)):
        c_ij_write = True
      elif (match(r'cr_name$', keyword)):
        c_ij_write = True
        c_ij_name = (orig_l.split())[1]
      
      # **************************************************************************
      # S(k)

      elif (match(r'sk_write$', keyword)):
        S_ij_write = True
      elif (match(r'sk_name$', keyword)):
        S_ij_write = True
        S_ij_name = (orig_l.split())[1]
      
      # **************************************************************************

      else:
        raise unkKeyword((i,l))
    # end for i in range(len(lines)):
  # end try
  except ValueError, msg:
    print("error processing line %lu (%s)" % (i + lineno_beg, l))
    print(msg)
    sys.exit(2)
  except unkKeyword, msg:
    print("error processing line %lu (%s)" % (msg.value[0] + lineno_beg, l))
    print("unknown keyword")
    sys.exit(2)
  except badValue, msg:
    print ("error processing line %lu (%s)" % (msg.value[0] + lineno_beg, l))
    print (msg.value[1])
    sys.exit(2)
  except IndexError, msg:
    print("error processing line %lu (%s)" % (i + lineno_beg, l))
    print("not enough data on the line")
    sys.exit(2)
    
  # end try/except block
    
  # save the data into the outp dictionary and print out the settings

  # Gamma-related stuff
  outp['G_ij_write'] = G_ij_write
  # output file with gamma function
  outp['G_ij_name'] = G_ij_name
  # Gamma save to file frequence (in iterations)
  outp['G_ij_savefreq'] = G_ij_savefreq
  # use binary data
  outp['G_ij_binary'] = G_ij_binary
  if (outp['G_ij_write']):
    if (outp['G_ij_savefreq'] == 0):
      print("\tGamma function will be saved at the end to %s" % (outp['G_ij_name']))
    else:
      print("\tGamma function will be saved every %lu iterations to %s" % (outp['G_ij_savefreq'], outp['G_ij_name']))
    if (outp['G_ij_binary']):
      print("\t\tbinary format will be used")
    else:
      print("\t\ttext format will be used")
  #else:
  #  print("\tGamma function will not be written during the calculation!")
  
  # g(r) related stuff
  outp['g_ij_write'] = g_ij_write
  outp['g_ij_name'] = g_ij_name
  if (outp['g_ij_write']):
    print("\tg(r) will be saved to file %s" % (outp['g_ij_name']))
  else:
    print("\tg(r) will not be written!")

  # c(r) related stuff
  outp['c_ij_write'] = c_ij_write
  outp['c_ij_name'] = c_ij_name
  if (outp['c_ij_write']):
    print("\tc(r) will be saved to file %s" % (outp['c_ij_name']))
  #else:
  #  print("\tc(r) will not be written!")

  # S(k) related stuff
  outp['S_ij_write'] = S_ij_write
  outp['S_ij_name'] = S_ij_name
  if (outp['S_ij_write']):
    print("\tS(k) will be saved to file %s" % (outp['S_ij_name']))
  #else:
  #  print("\tS(k) will not be written!")

  # U(r) related stuff
  outp['U_ij_write'] = U_ij_write 
  outp['U_ij_name'] = U_ij_name
  if (outp['U_ij_write']):
    print("\tpair potential will be saved to file %s" % (outp['U_ij_name']))
  outp['Utot_ij_write'] = Utot_ij_write 
  outp['Utot_ij_name'] = Utot_ij_name
  if (outp['U_ij_write']):
    print("\tpair potential with Gamma will be saved to file %s" % (outp['Utot_ij_name']))
  
  return(outp)

# **********************************************************************************************  

def check_validity(lines, lineno_beg, start_pos, end_pos):
  """
     function for checking the validity of input data
     
     lines = list of input lines
     lineno_beg = number of the first line in the 'lines' list
     start_pos = list of indexes of section beginnings
     end_pos = list of indexes of section ends
     
     checks sections in the input file (correct order, not nested, ...)
     checks for misplaced lines (keywords out of declaration blocks, ...)
     function used by parse_input and parse_parm
  """

  # the number of section openings and closings has to match
  if (len(end_pos) != len(start_pos)):
    print("\tnumber of section openings (%u) doesn't agree with the number of %%end statements (%u)" % (len(start_pos), len(end_pos)))
    sys.exit(2)

  # the sections are not allowed to be nested, to overlap, ...
  for i in range(len(start_pos)):
    if (start_pos[i] > end_pos[i]):
      print("\tend of unopened section (line %u)" % (lineno_beg + end_pos[i]))
      sys.exit(2)
  #end for i in range(len(start_pos)):
  for i in range(len(start_pos) - 1):
    if (end_pos[i] > start_pos[i+1]):
      print("\toverlapping/nested sections not supported (line %u)" % (lineno_beg + end_pos[i]))
      sys.exit(2)
  #end for i in range(len(start_pos) - 1):
  
  # check for the reamining lines outside of the %xxxx blocks - if not blank/comments -> error
  # beginning of file
  for i in range(start_pos[0] - 1):
    #print(i, lines[i])
    if (not (match("#", lines[i]) or lines[i] == "")):
      print("\tunknown identifier (%s) outside of declaration block on line %u" %  (lines[i], lineno_beg + i))
      sys.exit(2)
  # end for i in range(start_pos[0] - 1):
  # now process the data in between of the blocks
  for j in range(len(start_pos) - 1):
    for i in range(end_pos[j] + 1, start_pos[j+1]):
      if (not (match("#", lines[i]) or lines[i] == "")):
        print("\tunknown identifier (%s) outside of declaration block on line %u" %  (lines[i], lineno_beg + i))
        sys.exit(2)
    # end for i in range(len(start_pos) - 1):
  # end for j in range(end_pos[j] + 1, start_pos[j+1]-1):
  # and now the remaining data
  for i in range(end_pos[len(start_pos) - 1] + 1, len(lines)):
    if (not (match("#", lines[i]) or lines[i] == "")):
      print("\tunknown identifier (%s) outside of declaration block on line %u" %  (lines[i], lineno_beg + i))
      sys.exit(2)
  # end for i in range(start_pos[len(start_pos) - 1] - 1, len(lines)):
  
# **********************************************************************************************  

# 2 functions to be used to check whether supported energy and distance unit
# was used for input

def _check_energy_unit(const, user_unit, user_unit_orig):
  # name of the unit to be printed out outside the function
  unit_name = ""
  # multiplication factor for conversion to kT
  unit_factor = 0.0
  
  if (user_unit == "kt"):
    unit_name = "kT"
    unit_factor = 1.0
  elif (user_unit == "ev"):
    unit_name = "eV"
    unit_factor = const.conv_eVtokT
  elif (user_unit == "kj_mol"):
    unit_name = "kJ/mol"
    unit_factor = const.conv_kJmoltokT
  elif (user_unit == "kcal_mol"):
    unit_name = "kcal/mol"
    unit_factor = const.conv_kcalmoltokT
  else:
    # unknown unit
    print("unknown unit %s" % user_unit_orig)
    print("supported units kT, eV, kJ_mol, kcal_mol")
    sys.exit(2)
  # end handling energy units
  return((unit_name, unit_factor))
  
def _check_distance_unit(const, user_unit, user_unit_orig):
  # name of the unit to be printed out outside the function
  unit_name = ""
  # multiplication factor for conversion to Angstroms
  unit_factor = 0.0
  
  if (user_unit == "a"):
    unit_name = "Angstrom"
    unit_factor = 1.0
  elif (user_unit == "m"):
    unit_name = "m"
    unit_factor = const.conv_mtoA
  elif (user_unit == "nm"):
    unit_name = "nm"
    unit_factor = const.conv_nmtoA
  elif (user_unit == "pm"):
    unit_name = "pm"
    unit_factor = const.conv_pmtoA
  else:
    # unknown unit
    print("unknown unit %s" % user_unit_orig)
    print("supported units A, m, nm, pm")
    sys.exit(2)
  # end handling distance units
  return((unit_name, unit_factor))

# **********************************************************************************************  
  
if __name__ == "__main__":
  print __doc__
  print("Usage as a standalone application/script is not supported at the moment")

