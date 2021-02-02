# $Author: carthik $
# $Revision: 267 $
# $Date: 2010-06-08 22:33:22 -0400 (Tue, 08 Jun 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/sigrej.py $
# $Id: sigrej.py 267 2010-06-09 02:33:22Z carthik $


import numpy as np
import medstddev as msd

def sigrej(data, sigma, mask=None,     estsig=None,   ival=False, axis=0,
           fmean=False, fstddev=False, fmedian=False, fmedstddev=False):
  """This function flags outlying points in a data set using sigma
    rejection.

    Parameters:
    -----------
    data:   Array of points to apply sigma rejection to.
    sigma:  1D array of sigma multipliers for each iteration
            of sigma rejection.  Number of elements determines
            number of iterations.
    mask:   (optional) byte array, with the same shape as
            Data, where 1 indicates the corresponding element in
            Data is good and 0 indicates it is bad.  Only
            rejection of good-flagged data will be further
            considered.  This input mask is NOT modified in the
            caller.
    estsig: [nsig] array of estimated standard deviations to
            use instead of calculated ones in each iteration.
            This is useful in the case of small datasets with
            outliers, in which case the calculated standard
            deviation can be large if there is an outlier and
            small if there is not, leading to rejection of
            good elements in a clean dataset and acceptance of
            all elements in a dataset with one bad element.
            Set any element of estsig to a negative value to
            use the calculated standard deviation for that
            iteration.
    ival:    If True, return 2D array giving the median and
             standard deviation (with respect to the median) before
             each iteration.
    axis:    Axis number along which the data run.  Other axes contain
             different data sets.
    fmean:   If True, return the mean of the accepted data
    fstddev: If True, return the standard deviation of the accepted 
     	     data with respect to the mean
    fmedian: If True, return the median of the accepted data
    fmedstddev: If True, return the standard deviation of the accepted
       	    data with respect to the median

    Return:
    -------
    This function returns a mask of accepted values in the data.  The
    mask is a Bool array of the same shape as Data.  In the mask, True
    indicates good data, False indicates an outlier in the corresponding
    location of Data.

    If any of the return flags was set, the mask returns as the first
    element of a tuple, which also contains the requested diagnostic
    returns, in the order given above.

    Notes:
    ------
    sigrej flags as outliers points a distance of 'sigma' times the standard
    deviation from the median.  Unless given as a positive value in
    estsig, standard deviation is calculated with respect to the
    median, using medstddev.  For each successive iteration and value of
    sigma, sigrej recalculates the median and standard deviation from
    the set of 'good' (not masked) points, and uses these new values in
    calculating further outliers. The final mask contains a value of True
    for every 'inlier' and False for every outlying data point.

    Example:
    --------
    >>> x = np.array([65., 667, 84, 968, 62, 70, 66, 78, 47, 71, 56, 65, 60])
    >>> mask, ival, fmean, fstddev, fmedian, fmedstddev = \
                  sr.sigrej(x, [2,1], ival=True, fmean=True, 
                            fstddev=True, fmedian=True, fmedstddev=True)
    >>> print(maks)
    [ True False  True False  True  True  True  True  True  True  True  True
      True]
    >>> print(ival)
    [[  66.           65.5       ]
     [ 313.02675604  181.61572819]]
    >>> print(fmean)
    65.8181818182
    >>> print(fstddev)
    10.1174916043
    >>> print(fmedian)
    65.0
    >>> print(fmedstddev)
    10.1538170163

    Modification History:
    ---------------------
    2005-01-18 statia   Written by Statia Luszcz, Cornell. 
                        shl35@cornell.edu
    2005-01-19 statia   Changed function to return mask, rather than a
                        list of outlying and inlying points# added final
                        statistics keywords
    2005-01-20 jh       Joe Harrington, Cornell, jh@oobleck.astro.cornell.edu
	                Header update.  Added example.
    2005-05-26 jh       Fixed header typo.
    2006-01-10 jh       Moved definition, added test to see if all
                        elements rejected before last iteration (e.g.,
                        dataset is all NaN).  Added input mask, estsig.
    2010-11-01 patricio Converted to python. pcubillos@fulbrightmail.org
    2016-12-10 jh       Changed comparisons to None to comparisons to type(None).
                        Cleaned up example, order of returned params,
                        and parameter and return documentation.
  """

  # Get sizes
  dims = list(np.shape(data))
  nsig = np.size(sigma)
  if nsig == 0:
    nsig  = 1
    sigma = [sigma]

  if type(mask) == type(None):
    mask = np.ones(dims, bool)

  # defining estsig makes the logic below easier
  if type(estsig) == type(None):
    estsig = - np.ones(nsig)

  # Return parameters:
  retival       = ival
  retfmean      = fmean
  retfstddev    = fstddev
  retfmedian    = fmedian
  retfmedstddev = fmedstddev

  # Remove axis
  del(dims[axis])
  ival = np.empty( (2, nsig) + tuple(dims) )
  ival[:] = np.nan

  # Iterations
  for iter in np.arange(nsig):
    
    if estsig[iter] > 0:   # if we dont have an estimated std dev.
      # Calculations
      # FINDME: This hard-coded double loop breaks the code's
      # generality!  Make a view into the array with the data on one
      # axis and the datasets on another, or else use axis= properly.
      # or use Masked Arrays.  See AST 5765 code in transit project.
      for   j in np.arange(dims[0]):
        for i in np.arange(dims[1]):
          ival[0, iter, j, i] = np.median(data[:,j,i][np.where(mask[:,j,i])])
      # note: broadcasting to a slice of ival
      ival[1,iter] = estsig

      # Fixes
      count = np.sum(mask, axis=axis)
      # note: broadcasting to a slice of ival
      (ival[1,iter])[np.where(count == 0)] = np.nan

    else:
      # note: broadcasting to a slice of ival
      ival[1,iter], ival[0,iter] = msd.medstddev(data, mask, axis=axis,
                                                     medi=True)
    # Update mask
    # note: a slice of ival
    # FINDME: If something is nan, this breaks and throws invalid value warning.
    mask *= ( (data >= (ival[0,iter] - sigma[iter] * ival[1,iter])) &
              (data <= (ival[0,iter] + sigma[iter] * ival[1,iter])) )


  # the return arrays
  ret = (mask,)
  if retival:
    ret = ret + (ival,)

  # final calculations
  if retfmean or retfstddev:
    count   = np.sum(mask, axis=axis)
    fmean   = np.nansum(data*mask, axis=axis)

    # calculate only where there are good pixels
    goodvals = np.isfinite(fmean) * (count>0)
    if np.ndim(fmean) == 0 and goodvals:
      fmean /= count
    else:
      fmean[np.where(goodvals)] /= count[np.where(goodvals)]

    if retfstddev:
      resid   = (data-fmean)*mask
      fstddev = np.sqrt( np.sum( resid**2, axis=axis ) /(count - 1) )  
      if np.ndim(fstddev) == 0:
        if count == 1:
          fstddev = 0.0
      else:
        fstddev[np.where(count == 1)] = 0.0

  if retfmedian or retfmedstddev:
    fmedstddev, fmedian = msd.medstddev(data, mask, axis=axis, medi=True)

  # the returned final arrays
  if retfmean:
    ret= ret + (fmean,)
  if retfstddev:
    ret= ret + (fstddev,)
  if retfmedian: 
    ret= ret + (fmedian,)
  if retfmedstddev:
    ret= ret + (fmedstddev,)

  if len(ret) == 1:
    return ret[0]
  return ret
