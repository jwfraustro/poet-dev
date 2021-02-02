# $Author: patricio $
# $Revision: 304 $
# $Date: 2010-07-13 11:36:20 -0400 (Tue, 13 Jul 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/centerdriver.py $
# $Id: centerdriver.py 304 2010-07-13 15:36:20Z patricio $

import numpy as np
import sys
import least_asymmetry as ctr
import imageedit   as ie
import psf_fit     as pf
import cgaussian    as g
import disk
import matplotlib.pyplot as plt
import cenlight

def centerdriver(method, data, guess, trim, radius, size,
                 mask=None, uncd=None, fitbg=1, maskstar=True,
                 expand=5.0, psf=None, psfctr=None, noisepix=False,
                 npskyrad=(0,0)):
  """
  Use the center method to find the center of a star in data, starting
  from position guess.
  
  Parameters:
  -----------
  method:   string
            Name of the centering method to use.
  data:     2D ndarray
            Array containing the star image.
  guess:    2 element 1D array
            y, x initial guess position of the target.
  trim:     integer
            Semi-length of the box around the target that will be trimmed.
  radius:   float
            Least-asymmetry parameter. See err_fasym_c.
  size:     float
            Least-asymmetry parameter. See err_fasym_c.  
  mask:     2D ndarray
            A mask array of bad pixels. Same shape as data.
  uncd:     2D ndarray
            An array containing the uncertainty values of data. Same
            shape as data.
  noisepix: bool
            Boolean flag to calculate and return noise pixels.
  npskyrad: 2 element iterable
            Radius of sky annuli used in noise pixel calculation
          
  Returns:
  --------
  A y,x tuple (scalars) with the coordinates of center of the target
  in data.
  
  Example:
  --------
  nica

  Modification History:
  ---------------------
  2010-11-23 patricio   Written by Patricio Cubillos
                        pcubillos@fulbrightmail.org
  2016-12-11 jh         Fixed None comparisons. Cleaned up docstring.
  2018-01-05 zacchaeus  updated for python3
                        zaccysc@gmail.com
  """

  # Default mask: all good
  if type(mask) == type(None):
    mask = np.ones(np.shape(data))

  # Default uncertainties: flat image
  if type(uncd) == type(None):
    uncd = np.ones(np.shape(data))

  # Trim the image if requested
  if trim != 0:
    # Integer part of center
    cen = np.rint(guess)
    # Center in the trimed image
    loc = (trim, trim)
    # Do the trim:
    img, msk, err = ie.trimimage(data, cen[0], cen[1], loc[0], loc[1], mask=mask, uncd=uncd)
  else:
    cen = np.array([0,0])
    loc = np.rint(guess)
    img, msk, err = data, mask, uncd


  # If all data is bad:
  if not np.any(msk):
    raise Exception('Bad Frame Exception!')

  weights = 1.0/np.abs(err)
  extra = []

  # Get the center with one of the methods:
  if   method == 'fgc':
    par, err = g.fitgaussian(img, yxguess=loc, mask=msk, weights=weights,
                         fitbg=fitbg, maskg=maskstar)
    y,    x    = par[2:4]
    #print('y: {:.3f}\tx: {:.3f}'.format(y, x))    
    # array([yerr, xerr, ywidth, xwidth])
    extra = np.concatenate((err[2:4], par[0:2]))
  elif method == 'rfgc':
    par, err = g.rotfitgaussian(img, yxguess=loc, mask=msk, weights=weights,
                                fitbg=fitbg, maskg=maskstar)
    y,    x    = par[2:4]
    #print('y: {:.3f}\tx: {:.3f}'.format(y, x))

    # array([yerr, xerr, ywidth, xwidth, rot])
    extra = np.concatenate((err[2:4], par[0:2], [par[5]]))
  elif method == 'col':
    y, x = cenlight.col(img, loc, npskyrad, mask=msk)
#    print(y, x)
  elif method == 'ccl':
    y, x = cenlight.ccl(img, trim, loc, npskyrad, mask=msk)
  elif method == 'lag':
    pos, asymarr = ctr.actr(img, loc, asym_rad=radius,
                    asym_size=size, method='gaus')
    y = pos[0]
    x = pos[1]
  elif method == 'lac':
    pos, asymarr = ctr.actr(img, loc, asym_rad=radius,
                    asym_size=size, method='col')
    y = pos[0]
    x = pos[1]
  elif method == 'bpf' or method == 'ipf':
    y, x, flux, sky = pf.spitzer_fit(img, msk, weights, psf, psfctr, expand,
                                    method)
    extra = flux, sky

  # Make trimming correction and return
  if noisepix == True:
    N = calcnoisepix(img, y, x, npskyrad, mask=msk)
    return ((y, x) + cen - trim), extra, N
  else:
    return ((y, x) + cen - trim), extra


def calcnoisepix(im, y, x, npskyrad, mask=None):
    '''
    Implementation of noise pixel calculation as described in the IRAC
    handbook. The box_centroider.pro routine was used as reference.

    Formula is N = sum(image)**2/sum(image**2).

    Inputs
    ------
    im: array-like
        Intended to be a 2D image, although any array should work.

    y: float
        Y position of the target

    x: float
        X position of the target

    npskyrad: 2-tuple of floats
        Radii for sky annulus (sky should be subtracted before
        calculating noise pixels in case it varies significantly,
        as is the case for spitzer subarrays)

    mask: boolean array
        Mask for the image. Same shape as im.

    Returns
    -------
    N: float
        Noise pixels

    Revisions
    ---------
    rchallen Initial Implementation   rchallen@knights.ucf.edu
    '''

    # Create a mask if not supplied. Otherwise, make sure mask is
    # boolean
    if type(mask) == type(None):
        mask = np.ones(np.shape(im), dtype=bool)
    else:
        mask = mask.astype(bool)

    skymsk = disk.disk(npskyrad[1], (x,y), im.shape) ^ \
             disk.disk(npskyrad[0], (x,y), im.shape)
    
    sky = np.average(im[mask*skymsk])
    
    flux = np.sum(im[mask] - sky)

    N = flux**2/np.sum((im[mask] - sky)**2)

    return N
