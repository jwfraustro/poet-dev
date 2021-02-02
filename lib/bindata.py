# $Author: patricio $
# $Revision: 670 $
# $Date: 2012-07-24 12:51:52 -0400 (Tue, 24 Jul 2012) $
# $HeadURL: file:///home/esp01/svn/code/python/pipeline/trunk/lib/bindata.py $
# $Id: bindata.py 670 2012-07-24 16:51:52Z patricio $

import numpy as np

def bindata(nbins, median=[], mean=[], std=[], weighted=[], binsize=0):
  """
    Bin data, using median, mean, or weighted binning.

    Parameters:
    -----------
    nbins: Scalar
           Number of bins.
    median:   List of arrays to be median-binned.
    mean:     List of arrays to be mean-binned.
    std:      List of arrays to bin as standard deviation.
    weighted: List of arrays to be weighted binned. std[0] will be used as
              weights.

    Notes:
    ------
    All arrays must have the same lenght.

    Modification history:
    ---------------------
    2012-09-04  patricio  Written by Patricio Cubillos.
                          pcubillos@fulbrightmail.org, UCF
  """
  # Number of arrays in each list:
  nmedian,   nmean,   nstd,   nweighted   = len(median), len(mean), \
                                            len(std),    len(weighted)
  
  # Get lenght of arrays:
  if nmedian > 0:
    nobj = len(median[0])
  elif nmean > 0:
    nobj = len(mean[0])
  else:
    nobj = len(std[0])

  # If binsize is defined, override nbins value:
  if   binsize > 0:
    nbins = int(np.around(float(nobj)/binsize))
  #elif binsize == 0:
  binsize = float(nobj)/nbins

  # Initialize output:
  medianbin, meanbin, stdbin, weightedbin = [], [], [], []
  for b in np.arange(nmedian):
    medianbin.  append(np.zeros(nbins))
  for b in np.arange(nmean):
    meanbin.    append(np.zeros(nbins))
  for b in np.arange(nstd):
    stdbin.     append(np.zeros(nbins))
  for b in np.arange(nweighted):
    weightedbin.append(np.zeros(nbins))

  # Do the binning:    
  for i in np.arange(nbins):
    start = int( (i  ) * binsize )
    end   = int( (i+1) * binsize )

    for b in np.arange(nmedian):
      medianbin[b][i] = np.median(median[b][start:end])
    for b in np.arange(nmean):
      meanbin[b][i]   = np.mean(mean[b][start:end])  
    for b in np.arange(nstd):
      stdbin[b][i]    = np.sqrt( 1.0 / np.sum(1/std[b][start:end]**2) )
    for b in np.arange(nweighted):
      weightedbin[b][i] = (np.sum(weighted[b][start:end]/std[0][start:end]**2) /
                           np.sum(1/std[0][start:end]**2))

  # Concatenate all bins and return it:
  if nmean > 0:
    medianbin.extend(meanbin)
  if nstd > 0:
    medianbin.extend(stdbin)
  if nweighted > 0:
    medianbin.extend(weightedbin)
  return medianbin

def subarnbin(arr, event):
    '''
    Bins the given array according to subarray number. Uses a mean.

    Parameters
    ----------
    arr: ndarray
         Array to be binned

    event: POET event object
           The POET event object which contains at least event.subarn,
           an array which lists the frame number within each subarray
           frame set.

           e.g. [0, 1, 2, ..., 63, 64, 0, 1, ..., 63, 64]

    Returns
    -------
    newarr: ndarray
            The binned array.

    History
    -------
    2018-08-20 rchallen@knights.ucf.edu
               Initial implementation.
    '''
    newarr = []

    # Number of frames in the array
    nfrm = len(arr)

    # Array that will be populated with the items to bin based on
    # subarray number
    tobin = []
    
    for i in range(len(arr)):
        # If we are still adding to the binning array (i.e., we
        # are still increasing in subarray number), append the
        # next item
        if event.fp.subarn[0][i] - event.fp.subarn[0][i-1] > 0:
            tobin.append(arr[i-1])
        # If we have reset subarray number back to 0, append the last
        # item, take the average, add the average to the new binned
        # array, and empty the list of items to be binned
        else:
            tobin.append(arr[i-1])
            avg = np.average(tobin)
            newarr.append(avg)
            tobin = []

    newarr = np.array(newarr)

    return newarr

def subarnvar(arr, event):
    '''
    Calculates the variance of given array over each frame set. See
    subarnbin() for reference. Inputs and outputs are analogous.
    '''
    newarr = []

    # Number of frames in the array
    nfrm = len(arr)

    # Array that will be populated with the items to bin based on
    # subarray number
    tobin = []
    
    for i in range(len(arr)):
        # If we are still adding to the binning array (i.e., we
        # are still increasing in subarray number), append the
        # next item
        if event.fp.subarn[0][i] - event.fp.subarn[0][i-1] > 0:
            tobin.append(arr[i-1])
        # If we have reset subarray number back to 0, append the last
        # item, take the average, add the average to the new binned
        # array, and empty the list of items to be binned
        else:
            tobin.append(arr[i-1])
            var = np.var(tobin)
            newarr.append(var)
            tobin = []

    return newarr
