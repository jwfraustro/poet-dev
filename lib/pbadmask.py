# $Author: patricio $
# $Revision: 288 $
# $Date: 2010-06-24 03:33:44 -0400 (Thu, 24 Jun 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/poet_badmask.py $
# $Id: poet_badmask.py 288 2010-06-24 07:33:44Z patricio $


import numpy  as np
import sigrej as sr

def poet_badmask(data,  uncd,  pmask, pcrit, 
                 dmask, dcrit, fp, nimpos,  sigma=None):

  """
    This function generates a bad pixel mask from Spitzer time-series
    photometry data.

    Parameters:
    -----------
    data:    ndarray
             array of shape (maxnimpos,ny,nx,npos) , where nx and ny
             are the image dimensions, maxnimpos is the maximum number
             of images in the largest set, and npos is the number of
             sets (or 'positions').
    uncd:    ndarray 
             Uncertainties of data.
    pmask:   ndarray
             The permanent bad pixel mask for the instrument.
    pcrit:   scalar
             A bitmask indicating which bits in Pmask are critical
             problems for which we should flag a bad pixel.
    dmask:   The per-frame bad pixel masks for the dataset.  Same
		shape as Data, maybe different type.
    dcrit:
    fp:      2D ndarray
             Per-frame parameters, of shape (npos, maxnimpos)
    nimpos:  1D ndarray
             zero-based index of the last valid image in each position
             of data.
    sigma:  1D ndarray
            Successive sigma-rejection threshholds, passed to sigrej.
            If not defined, the data check is skipped, so that another
            routine can be used for that step.  Still allocates the
            array and initializes it with pre-flagged bad pixels (NaNs
            and prior masks).

    Return:
    -------
    ndarray
    This function returns a byte array, with the same shape as data,
    where 1 indicates the corresponding pixel in Data is good and 0
    indicates it is bad.

    Modification History:
    ---------------------
    2005-xx-xx  jh       Written by: Joseph Harrington, Cornell
                         jh@oobleck.astro.cornell.edu
                         and Statia Luszcz, Cornell
		         shl35@cornell.edu
    2005-10-13 jh        Restructured and largely rewritten.  Header added.
    2005-10-26 jh        Renamed rejct to nsigrej, updated comments.
    2005-11-21 jh        Added SIGMA keyword instead of hard-coding it.
    2005-11-24 jh        Changed badmask to mask, mask to smask, added nsstrej.
    2005-11-25 jh        Changed to use FP.
    2007-05-29 khorning  Handle data with only 3 dimensions
    2007-06-28 jh        Cleaned up indentation.
    2008-05-08 kevin     Added uncertainty array 'uncd', applied NaN test.
    2010-11-01 patricio  Converted to python. pcubillos@fulbrightmail.org
    2017-06-20 zacchaeus Fixed None comparisons.
  """

  # sizes
  maxnimpos, ny, nx, npos = np.shape(data)

  # allocate bad pixel mask
  mask = np.zeros((maxnimpos, ny, nx, npos), np.byte)

  # flag existing frames as good
  for pos in np.arange(npos): 
    mask[0:nimpos[pos], :, :, pos] = 1 

  # apply permanently-bad pixel mask
  pmaskcrit = pmask & pcrit
  if np.amax(pmask) > 0:
    permmask =  1 - ( pmaskcrit > 0 )
    mask *= permmask.reshape( (1,) + np.shape(permmask) + (1,) )

  # apply per-frame bad pixel masks
  dmaskcrit = dmask & dcrit
  mask[np.where(dmaskcrit > 0)] = 0

  # flag NaNs in the data and uncertainties
  bad = np.where( (np.isfinite(data) == 0) | (np.isfinite(uncd) == 0) )
  mask[bad] = 0

  # How many spacecraft-flagged bad pixels?
  # Spitzer rejects
  fp.nsstrej = np.sum(np.sum(1 - mask, axis=1), axis=1) # collapse along y and x

  # Reject outlying pixels, if sigma is provided
  if sigma is not None:
    for pos in np.arange(npos):
       mask[0:nimpos[pos], :, :, pos] = \
          sr.sigrej(data[0:nimpos[pos], :, :, pos], sigma)

    # How many more bad pixels did we reject?
    # our rejects
    fp.nsigrej  = np.sum(np.sum(1 - mask, axis=1), axis=1)
    fp.nsigrej -= fp.nsstrej
    fp.nsigrej  = np.transpose(fp.nsigrej)

  fp.nsstrej = np.transpose(fp.nsstrej)
  return mask
