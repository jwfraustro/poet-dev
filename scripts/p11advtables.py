#! /usr/bin/env python

# $Author: patricio $
# $Revision: 725 $
# $Date: 2013-04-23 05:19:08 -0400 (Tue, 23 Apr 2013) $
# $HeadURL: file:///home/esp01/svn/code/python/pipeline/trunk/p10advtables.py $
# $Id: p10advtables.py 725 2013-04-23 09:19:08Z patricio $

"""
  Usage:
  ------
  Fill the fields commented with 'ENTER MANUALLY'.
  Then run from the promt:
  p10advtables.py filedir 


  Package Content:
  ----------------

To run this program, please fill in all the fields commented
'ENTER MANUALLY'.  Save and close the file then type 'p10advtables'
to run the program.  This version is specially designed for BLISS 
mapping and exponential & polynomial ramp models.  If you do not 
have a fit that uses these, please contact ccampo/kevin to get a 
version that works for your model fits.
"""

import os, sys, datetime
try:
  import cPickle as pickle
except:
  iport pickle
import numpy             as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
sys.path.append('../')
import run
plt.ion()

# Restore saved files:
event = run.p7Restore(filedir=sys.argv[1])
#event = run.p6Restore(filedir=sys.argv[1])  # OPTIONAL: Restore from p6.
planetname = event[0].planetname # Name of the planet
today = datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d")

# ENTER MANUALLY:  filename to save as
# set to None if you do not wish to save a file
fname = planetname + "_lightcurve-fit_" + today + "table"

# ENTER MANUALLY:  Ephemerides offset:
ephoffset = 2450000  # MJD - BJD in table

# ENTER MANUALLY:  Pretty-print the uncertainties
pretty = True 

# Name labels by event or by wavelength:
# ENTER MANUALLY:  choose: ['event' | 'wavelength']
to_label = "event"

# ENTER MANUALLY: Select how many tables and columns per table:
# nevents is a list of how many events per table do I want:
nevents = [len(event)]

# :::::::::::::::: End edditing ::::::::::::::::

# :::::::::::::::: Ancilliary routines ::::::::::::::::
def prettyprint(value, error, precision=2, exprep=False):
  """
  Pretty-print nominal value and uncertainty with 'precision' significant
  digits in the error.

  Parameters:
  -----------
  value: Scalar
         Nominal value.
  error: Scalar
         Uncertainty of value.
  precision: Integer or 'int'
         Number of significant digits in error. Set to 'int' for a precision
         of the integer part of error.
  exprep: Boolean
          Use exponential notation if True.

  Returns:
  --------
  String representation of 'value +/- error' either as:
      x.xx(ee)e+xx
  or as:
      xxx.xx(ee)

  Example:
  --------
prettyprint(123.454, 0.12)             
prettyprint(123.454, 1.12)             
prettyprint(123.454, 1.12, precision=1)
prettyprint(123.454, 1.12, precision=3)

prettyprint(123.45, 0.1234, exprep=1)
prettyprint(123.45, 0.1234, exprep=0)

prettyprint(123.45, 0.0234, exprep=1)
prettyprint(123.45, 0.0234, exprep=0)

prettyprint(123.454, 0.1234, exprep=0)
prettyprint(123.456, 0.1234, exprep=0)

# Fixme: put the period in the uncertainty.
prettyprint(123.454, 1.12) 
prettyprint(123.456, 1.12)

prettyprint(1234, 234, exprep=1)
prettyprint(1234, 234, exprep=0)

# Fixme: don't modify value.
prettyprint(1234, 234, exprep=0, precision=1)

prettyprint(1234,  34, exprep=0, precision=3)
prettyprint(1234,  34, exprep=0, precision="int")
prettyprint(1234,  34, exprep=0, precision='int')
prettyprint(1234, 134, exprep=0, precision='int')

  Modification History:
  ---------------------
  2013-04-17  patricio  Grabbed code from  http://stackoverflow.com/questions/
                                         6671053/python-pretty-print-errorbars
                                                 pcubillos@fulbrightmail.org
  """
  # base 10 exponents:
  x_exp  = int(np.floor(np.log10(value))) # position of first significant digit
  xe_exp = int(np.floor(np.log10(error))) # ditto
  # Integer part precision:
  if precision == "int":
    precision = xe_exp+1
  # Uncertainty:
  un_exp = xe_exp - precision + 1
  un_int = np.round(error*10**(-un_exp))
  # Nominal value:
  no_exp = un_exp
  no_int = np.round(value*10**(-no_exp))
  # format1:  nom(unc)exp
  fieldw = x_exp - no_exp
  fmt = '%%.%df' % fieldw
  result1 = (fmt + '(%.0f)e%d') % (no_int*10**(-fieldw), un_int, x_exp)
  # format2:  nom(unc)
  fieldw = max(0, -no_exp)
  fmt  = '%%.%df' % fieldw
  if xe_exp >= 0 and un_exp < 0: # If the period is inbetween the error:
    result2 = fmt%(no_int*10**no_exp) + "(" + str(un_int*10**un_exp) + ")"
  else:
    result2 = (fmt + '(%.0f)') % (no_int*10**no_exp, un_int*10**max(0, un_exp))
  #print(fmt)
  #print(precision, un_exp)
  #print(xe_exp, un_exp)
  #print(fieldw)
  #print(un_int*10**un_exp)
  # return representation:
  if exprep:
    ret = result1
  else:
    ret = result2
  return ret


def formatit(value, error, fmt, precision, pretty=False, param=None):
  """
  Makes a string a value with uncertainty in either:
    'value & error'
  or:
    'value(error)'
  format, and appends it to the list param.

  Parameters:
  -----------
  value: Scalar
  error: Scalar
  fmt: String
       Numeric format for the value & error case.
  precision: Integer
       Number of significant digits of the uncetainty for the pretty case.
  pretty: Boolean
       Pretty print the value and error.
  param: List
       List where to store the output string.

  Examples:
  ---------
  >>>formatit(364.41851893081, 328.696261, ".0f", 'int', True)
  '1364(329)'
  >>>formatit(364.41851893081, 328.696261, ".0f", 'int', False)
  '1364 & 329'

  >>>formatit(0.4994872096, 0.003791894, ".4f", 2, True)
  '0.4995(38)'
  >>>formatit(0.4994872096, 0.003791894, ".4f", 2, False)
  '0.4995 & 0.0038'

  Modification History:
  ---------------------
  2013-04-18  patricio  Initial implementation.  pcubillos@fulbrightmail.org
  """
  if not pretty:
    format = "%" + fmt +" & %" + fmt
    number = format%(value, error)
  else:
    number = prettyprint(value, error, precision=precision)
  if param is not None:
    param.append(number)
  else:
    return number


def find(object, list, none=-1):
  """
  Find if any element of list is a parameter of object. return its
  value from if exists.

  Parameters:
  -----------
  object: Object
          Object where to look in.
  list:   List
          List of elements to search for.
  none:   Any
          Return if no element in list is in object.
  Modification History:
  ---------------------
  2013-04-18  patricio  Initial implementation.   pcubillos@fulbrightmail.org
  """
  common_element = set(dir(object)) & set(list)
  try:
    val = eval("object.%s"%common_element.pop())
  except:
    val = none
  return val


def addpar(table, pname, val, start, end, paramlen=64, valuelen=15,
          multicolumn=True, format="%s"):
  """
  Adds a &-separated string element to the table list (LaTeX format).
  E.g.:   ["Param name    & val[start]  & ...  & val[end]  \\"]

  Parameters:
  -----------
  table: List
         The list where to append the new parameter.
  pname: String
         Parameter name.
  val:   1D ndarray
         Array of values.
  start: Integer
         Index of first element in val to include.
  end:   Integer
         Index of last  element in val to include.
  paramlen: Integer
            Length of the parameter
  valuelen: Integer
            Length of each element in val.
  multicolumn: Boolean
          Include multicolumn command for single values.
  format: String
          Printing format for numerical values.

  Modification History:
  ---------------------
  2013-04-17  patricio  Re-implementation from previous code.
                        Added documentation.        pcubillos@fulbrightmail.org
  """
  try:
    # Add multicolumn if necessary:
    if (isinstance(val[start], str) and "&" in val[start]) or not multicolumn:
      parcode = format
    else:
      parcode = "\mctc{" + format + "}"
    # Construct the line:
    line = " & ".join([(parcode%p).ljust(valuelen) for p in val[start:end]])
    table.append(pname.ljust(paramlen) + "& " + line + "\\\\")
  except:
    print("Parameter:  '%s'  could not be appended to Table."%pname)


def ephcalc(event, fit, midpt):
  """
  Calculate ecltimeutc and ecltimetdb from ecltime

  Parameters:
  -----------
  event:
  fit:
  midpt:
  
  Modification History:
  ---------------------
  2013-04-17  patricio  Documented.   pcubillos@fulbrightmail.org
  """
  if hasattr(event, 'timestd'):
      print('Verify that the ephemeris is reported in BJD_' +
            event.timestd + '!')
      offset = event.bjdtdb.flat[0]-event.bjdutc.flat[0]
      if   event.timestd == 'utc':
          ephtimeutc = event.ephtime
          ephtimetdb = event.ephtime + offset
      elif event.timestd == 'tdb':
          ephtimetdb = event.ephtime
          ephtimeutc = event.ephtime - offset
      else:
          print('Assuming that the ephemeris is reported in BJD_UTC!')
          ephtimeutc = event.ephtime
          ephtimetdb = event.ephtime + offset
  else:
    print('Assuming that the ephemeris is reported in BJD_UTC!')
    offset = event.bjdtdb.flat[0]-event.bjdutc.flat[0]
    ephtimeutc = event.ephtime
    ephtimetdb = event.ephtime + offset
  fit.ecltimeutc = (np.floor((event.bjdutc.flat[0] - ephtimeutc)/event.period) +
                    fit.bestp[midpt]) * event.period + ephtimeutc
  fit.ecltimetdb = (np.floor((event.bjdtdb.flat[0] - ephtimetdb)/event.period) +
                    fit.bestp[midpt]) * event.period + ephtimetdb
  return

def gentable(ncolumns, offset, multicolumn=True):
  """
  Generate the table.

  Docstring me!
  """
  # Initial and end fit:
  fi, fe = offset, ncolumns+offset  
  # Parameter and Value length:
  if multicolumn:
    pl, vl = 62, 20
  else:
    pl, vl = 62, 16
  mc = multicolumn 
 
  table = []
  addpar(table,"\\colhead{Parameter}", labels, fi, fe, pl, vl, mc)
  table.append("\\hline")
  addpar(table,"Array position ($\\bar{x}$, pix)",xpos, fi, fe, pl, vl, mc,"%.2f")
  addpar(table,"Array position ($\\bar{y}$, pix)",ypos, fi, fe, pl, vl, mc,"%.2f")
  addpar(table,"Position consistency\\tablenotemark{a} ($\delta\sb{x}$, pix)",
                                                pcosx, fi, fe, pl, vl, mc, "%.3f")
  addpar(table,"Position consistency\\tablenotemark{a} ($\delta\sb{y}$, pix)",
                                                 pcosy, fi, fe, pl, vl, mc,"%.3f")
  addpar(table,"Aperture size (pix)",           photap, fi, fe, pl, vl, mc,"%.2f")
  addpar(table,"Sky annulus inner radius (pix)",skyin,  fi, fe, pl, vl, mc,"%.1f")
  addpar(table,"Sky annulus outer radius (pix)",skyout, fi, fe, pl, vl, mc,"%.1f")
  addpar(table,"Eclipse depth (\\%)",           depth,  fi, fe, pl, vl, mc)
  addpar(table,"Brightness Temperature (K)",    temp,   fi, fe, pl, vl, mc)
  addpar(table,"Midpoint (orbits)",              midpt,      fi, fe, pl, vl, mc)
  addpar(table,"Transit midpoint (MJD\\sb{UTC})",midtimeutc, fi, fe, pl, vl, mc)
  addpar(table,"Transit midpoint (MJD\\sb{TDB})",midtimetdb, fi, fe, pl, vl, mc)
  addpar(table,"Eclipse duration (hr)",         width,  fi, fe, pl, vl, mc)
  addpar(table,"Ingress/Egress time (hr)",      t12,    fi, fe, pl, vl, mc)
  addpar(table,"System flux: $F\sb{s}$ (\\micro Jy)", flux,  fi, fe, pl, vl, mc)
  addpar(table,"$R\\sb{p}/R\\sb{\star}",  rprs, fi, fe, pl, vl, mc)
  addpar(table,"cos i",                   cosi, fi, fe, pl, vl, mc)
  addpar(table,"$a/R\\sb{\\star}$",       ars,  fi, fe, pl, vl, mc)
  addpar(table,"Ramp: $R(t)$",            ramp, fi, fe, pl, vl, mc)
  addpar(table,"Ramp, $r\sb{0}$",         r0,   fi, fe, pl, vl, mc)
  addpar(table,"Ramp, $r\sb{1}$",         r1,   fi, fe, pl, vl, mc)
  addpar(table,"Ramp, $r\sb{2}$",         r2,   fi, fe, pl, vl, mc)
  addpar(table,"Ramp, $r\sb{3}$",         r3,   fi, fe, pl, vl, mc)
  addpar(table,"Ramp, $r\sb{4}$",         r4,   fi, fe, pl, vl, mc)
  addpar(table,"Ramp, $r\sb{5}$",         r5,   fi, fe, pl, vl, mc)
  addpar(table,"Ramp, $r\sb{6}$",         r6,   fi, fe, pl, vl, mc)
  addpar(table,"Ramp, $r\sb{7}$",         r7,   fi, fe, pl, vl, mc)
  addpar(table,"Ramp, $t\sb{0}$",         t0,   fi, fe, pl, vl, mc)
  addpar(table,"BLISS map ($M(x,y)$)", isbliss, fi, fe, pl, vl, mc)
  addpar(table,"Minimum number of points per bin", minnumpts, fi, fe, pl, vl, mc)
  addpar(table,"Total frames",          totfrm,   fi, fe, pl, vl, mc)      
  addpar(table,"Frames used",           goodfrm,  fi, fe, pl, vl, mc)
  addpar(table,"Rejected frames (\\%)", rejfrm,   fi, fe, pl, vl, mc, "%.2f")
  addpar(table,"Free parameters",       freepar,  fi, fe, pl, vl, mc)
  addpar(table,"AIC value", aic,  fi, fe, pl, vl, mc, "%.1f")
  addpar(table,"BIC value", bic,  fi, fe, pl, vl, mc, "%.1f")
  addpar(table,"SDNR",      sdnr, fi, fe, pl, vl, mc, "%.7f")
  addpar(table,"Uncertainty scaling factor", chifact, fi, fe, pl, vl, mc, "%.3f")
  addpar(table,"Percentage of photon-limited S/N (\\%)",
                photonSNR, fi, fe, pl, vl, mc, "%.2f")
  return "\n".join(table)


# :::::::::::::::: Main ::::::::::::::::

numevents  = len(event)          # Number of events
fit = []                         # Collect the light curve fits
for i in np.arange(numevents):
  fit.append(event[i].fit[0])

nfit = len(fit)

# Calculate the cumulative number of parameters per fit:
cumulativenparams = [0] 
for i in np.arange(nfit):
  cumulativenparams.append(fit[i].nump + cumulativenparams[-1])

cumulativenparams = np.asarray(cumulativenparams)

# Check which fits are joint:
jointindex = np.arange(nfit)
for i in np.arange(nfit):
  # Shared parameters in fit[i]:
  shared = fit[i].stepsize[np.where(fit[i].stepsize<0)]
  for q in shared:
    shpar = np.abs(q) - 1 # Shared par
    ifit = np.where(np.asarray(cumulativenparams) > shpar)[0][0] - 1
    if ifit < jointindex[i]: 
      jointindex[i] = ifit  # Lowest fit index in the group

jointruns = []
for i in np.arange(nfit):
  group = np.where(jointindex == i)[0]
  if len(group) > 0:
    jointruns.append(list(group))

# So, is this a joint fit?
joint = len(jointruns) < nfit

# Event name labels:
elabels = []
for i in np.arange(numevents):
  elabels.append(event[i].eventname)

# Wavelengths labels:
wavelenghts = ["3.6 microns", "4.5 microns", "5.8 microns",
               "8.0 microns", "16 microns",  "24 microns" ]
wlabels = []
for i in np.arange(numevents):
  wlabels.append(wavelenghts[event[i].photchan-1])

if to_label == "event":
  labels = elabels
elif to_label == "wavelength":
  labels = wlabels

# Calculate the ephemeris for eclipses:
for j in range(nfit):
  if hasattr(fit[j],'ecltime'):
      ephcalc(event[j], fit[j], fit[j].i.midpt)
      del fit[j].ecltime
  elif hasattr(fit[j],'ecltime2'):
      ephcalc(event[j], fit[j], fit[j].i.midpt2)
      del fit[j].ecltime2
  elif hasattr(fit[j],'ecltime3'):
      ephcalc(event[j], fit[j], fit[j].i.midpt3)
      del fit[j].ecltime3

# Hack: recalculate point consistency:
pcosx, pcosy = [], []
for j in np.arange(nfit):
  event[j].xprecision = np.sqrt(np.mean(np.ediff1d(fit[j].xuc)**2.0))
  event[j].yprecision = np.sqrt(np.mean(np.ediff1d(fit[j].yuc)**2.0))

# Make lists for the parameters:
nfit    = len(fit)
xpos,   ypos,  pcosx, pcosy    = [], [], [], []      # Centering
photap, skyin, skyout          = [], [], []          # Photometry
midpt,  depth, width, t12, t34 = [], [], [], [], []  # Eclipse params
rprs, cosi, ars                = [], [], []          # Transit
midtimeutc, midtimetdb, flux, temp = [], [], [], []  # Mid-point timing
ramp, r0, r1, r2, r3           = [], [], [], [], []  # Ramp name and parameters
r4,   r5, r6, r7, t0           = [], [], [], [], []
isbliss, minnumpts             = [], []              # BLISS map
totfrm, goodfrm , rejfrm       = [], [], []          # Frames stats
freepar, aic, bic, sdnr, chifact, photonSNR = [], [], [], [], [], [] # Stats


# get eachparameter with the proper ammount of decimal places and error
for j in range(nfit):
  xpos    .append( fit[j].xuc.mean()   )
  ypos    .append( fit[j].yuc.mean()   )
  pcosx   .append( event[j].xprecision )
  pcosy   .append( event[j].yprecision )
  photap  .append( event[j].photap     )
  skyin   .append( event[j].skyin      )
  skyout  .append( event[j].skyout     )
  if "tbm" in dir(fit[j]):
    temp.append(formatit(fit[j].tbm, fit[j].tbsd, ".0f", 'int', pretty))
  # Eclipse parameters:
  idx = find(fit[j].i, ["midpt", "midpt2", "midpt3"]) # Get index
  if idx >= 0:
    formatit(fit[j].bestp[idx  ], fit[j].typerr[idx  ], ".4f", 2, pretty, midpt)
    formatit(fit[j].bestp [idx+1]*event[j].period*24.0,
             fit[j].typerr[idx+1]*event[j].period*24.0, ".2f", 2, pretty, width)
    formatit(fit[j].bestp [idx+2]*100.0,
             fit[j].typerr[idx+2]*100.0,                ".3f", 2, pretty, depth)
    formatit(fit[j]. bestp[idx+3]*event[j].period*24.0,
             fit[j].typerr[idx+3]*event[j].period*24.0, ".3f", 2, pretty, t12)
    formatit(fit[j]. bestp[idx+4]*event[j].period*24.0,
             fit[j].typerr[idx+4]*event[j].period*24.0, ".3f", 2, pretty, t34)
    formatit(fit[j].bestp[idx+5], fit[j].typerr[idx+5], ".1f", 2, pretty, flux)
  # Eclipse BJD times:
  utc     = find(fit[j], ['ecltimeutc', 'ecltimeutc2', 'ecltimeutc3'])
  tdb     = find(fit[j], ['ecltimetdb', 'ecltimetdb2', 'ecltimetdb3'])
  timeerr = find(fit[j], ['ecltimeerr', 'ecltimeerr2', 'ecltimeerr3'])
  if utc > 0:
    formatit(utc-ephoffset, timeerr, ".4f", 3, pretty, midtimeutc)
  if tdb > 0:
    formatit(tdb-ephoffset, timeerr, ".4f", 3, pretty, midtimetdb)
  # Transit parameters:
  idx = find(fit[j].i, ["trmidpt", "trmidpt2"])
  if idx >= 0:
    mjdoffset = event[j].params.tuoffset - ephoffset
    formatit(fit[j].bestp[idx]+mjdoffset,              fit[j].typerr[idx],
             ".4f", 3, pretty, midtimeutc)
    formatit(fit[j].bestp[idx]+mjdoffset+65.183/86400, fit[j].typerr[idx],
             ".4f", 3, pretty, midtimeutc)
    formatit(fit[j].bestp[idx+1], fit[j].typerr[idx+1], ".4f", 3, pretty, rprs)
    formatit(fit[j].bestp[idx+2], fit[j].typerr[idx+2], ".6f", 3, pretty, cosi)
    formatit(fit[j].bestp[idx+3], fit[j].typerr[idx+3], ".3f", 3, pretty, ars )
    formatit(fit[j].bestp[idx+4], fit[j].typerr[idx+4], ".1f", 3, pretty, flux)

for j in range(nfit):
  try:
    index = fit[j].functypes.index("ramp")
    ramp.append(fit[j].model[index])
  except:
    ramp.append("None")
  idx = find(fit[j].i, ["rem", "fem", "re2m1", "ser0", "selr0",     # Ramp r0
                          "seqr0", "se2r0"])
  if idx >= 0:
    formatit(fit[j].bestp[idx], fit[j].typerr[idx], ".2f", 2, pretty, r0)
  else:
    r0.append("$\cdots$")
  idx = find(fit[j].i, ["ser1", "selr1", "selr1", "seqr1", "se2r1", # Ramp r1
                          "ret0", "fet0", "re2t1"])
  if idx >= 0:
    formatit(fit[j].bestp[idx], fit[j].typerr[idx], ".2f", 2, pretty, r1)
  else:
    r1.append("$\cdots$")
  idx = find(fit[j].i, ['linr2', 'seqr2', 'selr2', 'qrr2'])         # Ramp r2
  if idx >= 0:
    formatit(fit[j].bestp[idx], fit[j].typerr[idx], ".6f", 2, pretty, r2)
  else:
    r2.append("$\cdots$")
  idx = find(fit[j].i, ['seqr3', 'qrr3'])  # Ramp r3
  if idx >= 0:
    formatit(fit[j].bestp[idx], fit[j].typerr[idx], ".6f", 2, pretty, r3)
  else:
    r3.append("$\cdots$")
  idx = find(fit[j].i, ['re2m2', 'se2r4']) # Ramp r4
  if idx >= 0:
    formatit(fit[j].bestp[idx], fit[j].typerr[idx], ".6f", 2, pretty, r4)
  else:
    r4.append("$\cdots$")
  idx = find(fit[j].i, ['re2t2', 'se2r5']) # Ramp r5
  if idx >= 0:
    formatit(fit[j].bestp[idx], fit[j].typerr[idx], ".6f", 2, pretty, r5)
  else:
    r5.append("$\cdots$")
  idx = find(fit[j].i, ['llr6', 'lqr6', 'logr6', 'l4qr6']) # Ramp r6
  if idx >= 0:
    formatit(fit[j].bestp[idx], fit[j].typerr[idx], ".6f", 2, pretty, r6)
  else:
    r6.append("$\cdots$")
  idx = find(fit[j].i, ['lqr7', 'logr7', 'l4qr7'])         # Ramp r7
  if idx >= 0:
    formatit(fit[j].bestp[idx], fit[j].typerr[idx], ".6f", 2, pretty, r7)
  else:
    r7.append("$\cdots$")
  idx = find(fit[j].i, ['llt0', 'lqt0', 'logt0', 'l4qt0']) # Ramp t0
  if idx >= 0:
    formatit(fit[j].bestp[idx], fit[j].typerr[idx], ".6f", 2, pretty, t0)
  else:
    t0.append("$\cdots$")

for j in range(nfit):
  # BLISS map:
  if "ipmap" in fit[j].functypes:
    isbliss  .append("Yes")
    minnumpts.append(str(event[j].params.minnumpts[0]))
  else:
    isbliss  .append("No")
    minnumpts.append("$\cdots$")
  # Stats:
  totfrm   .append( np.size(event[j].phase)      )
  goodfrm  .append( fit[j].nobj                  )
  rejfrm   .append( (totfrm[j] - goodfrm[j])*100.0/totfrm[j])
  freepar  .append( fit[j].numfreepars           )
  aic      .append( fit[j].aic                   )
  bic      .append( fit[j].bic                   )
  sdnr     .append( np.std( fit[j].normresiduals))
  chifact  .append( (fit[j].sigma/fit[j].rawsigma)[0])
  photonSNR.append(100*fit[j].sn/fit[j].photsn   )

# Check for Joint fits and Update BIC and Number of free Parameters:
if joint:
  for i in range(len(jointruns)):
    chisq, nobj, numfreepars = 0, 0, 0
    for j in jointruns[i]:
      chisq += np.sum((fit[j].residuals/event[j].fit[0].sigma)**2)
      nobj  += fit[j].nobj
      numfreepars += fit[j].numfreepars
    for j in jointruns[i]:
      aic[j] = chisq + 2*numfreepars
      bic[j] = chisq +   numfreepars*np.log(nobj)


ntables = len(nevents)
offset  = [0]
for i in np.arange(ntables-1):
  offset.append(nevents[i])

multicolumn = not pretty

for i in np.arange(ntables):
  table = gentable(nevents[i], offset[i], multicolumn)

  if fname:
    outfile = file(sys.argv[1] + "/" + fname + str(i+1) + ".txt", 'w')
    outfile.write(table)
    outfile.close()
