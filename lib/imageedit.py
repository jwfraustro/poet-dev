# $Author: patricio $
# $Revision: 301 $
# $Date: 2010-07-10 03:33:44 -0400 (Sat, 10 Jul 2010) $
# $HeadURL: file:///home/esp01/svn/code/python/branches/patricio/photpipe/lib/imageedit.py $
# $Id: imageedit.py 301 2010-07-10 07:33:44Z patricio $

import numpy as np
"""
    Name
    ----
    Image Edit

    File
    ----
    imageedit.py

    Description
    -----------
    Routines for editting 2D array images, allows to cut and paste one
    array into other.

    Package Contents
    ----------------
    trimimage(data, (yc,xc), (yr,xr), mask=None, uncd=None, oob=0):
        Extracts a rectangular area of an image masking out of bound pixels.

    pasteimage(data, subim, (dyc,dxc), (syc,sxc)=(None,None)):
        Inserts subim array into data, matching the specified coordinates.

    Examples:
    ---------

    Revisions
    ---------
    2010-07-11  patricio  Added documentation.      pcubillos@fulbrightmail.org
    1018-01-05  zacchaeus updated for python3       zaccysc@gmail.com
"""

def trimimage(data, yc,xc, yr,xr, mask=None, uncd=None, oob=0):

  """
    Extracts a rectangular area of an image masking out of bound pixels.

    Parameters:
    ----------
    data  : 2D ndarray
            Image from which to extract a subimage.
    yc, xc: 2 element integer tuple
            Position in data where the extracted image will be centered. 
    yr, xr: 2 element integer tuple
            Semi-length of the extracted image.  For example, yr=2
            gives a 5-pixel-high subimage.
    mask  : 2D ndarray
            If specified, this routine will extract the mask subimage 
            as well.  Out-of-bounds pixels will have value oob.
            This mask is float, to allow for arbitrary oob values.  Be
            careful, any nonzero number is True when cast to bool,
            even -1.
    uncd  : 2D ndarray
            If specified, this routine will extract the uncd subimage 
            as well.
    oob   : scalar
            Value for out-of-bounds pixels in the mask. Default is 0.

    Returns
    -------
    Tuple, up to three 2D ndarray images containing the extracted image, 
    its mask (if specified), and an uncd array (if specified). The shape 
    of each array is (2*yr+1, 2*xr+1). 

    Example:
    -------
    >>> import numpy as np
    >>> from imageedit import *

    >>> # Create a data image and its mask
    >>> data  = np.arange(25).reshape(5,5)
    >>> print(data)
    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]
     [20 21 22 23 24]]
    >>> msk  = np.ones(np.shape(data))
    >>> msk[1:4,2] = 0

    >>> # Extract a subimage centered on (3,1) of shape (3,5) 
    >>> dyc,dxc = 3,1
    >>> subim, mask = trimimage(data, (dyc,dxc), (1,2), mask=msk)
    >>> print(subim)
    [[  0.  10.  11.  12.  13.]
     [  0.  15.  16.  17.  18.]
     [  0.  20.  21.  22.  23.]]
    >>> print(mask)
    [[ 0.  1.  1.  0.  1.]
     [ 0.  1.  1.  0.  1.]
     [ 0.  1.  1.  1.  1.]]

    >>> # Set out of bound pixels in the mask to -1:
    >>> subim, mask = trimimage(data, (dyc,dxc), (1,2), mask=msk, oob=-1)
    >>> print(mask)
    [[-1.  1.  1.  0.  1.]
     [-1.  1.  1.  0.  1.]
     [-1.  1.  1.  1.  1.]]

    Revisions
    ---------
    2010-07-11  patricio  Added documentation.      pcubillos@fulbrightmail.org
    2016-12-11  jh        Fixed array comparisons with None. Updated docstring.
    2017-02-06  rchallen  Cast indices to integers  rchallen@knights.ucf.edu
  """

  # As of latest Python we must make sure slices are with integers
  if type(xc) != int:
    xc = int(xc)
  if type(yc) != int:
    yc = int(yc)
  if type(yr) != int:
    yr = int(yr)
  if type(xr) != int:
    xr = int(xr)
  
  # Shape of original data
  ny, nx = np.shape(data)

  # The extracted image and mask
  im = np.zeros((2*yr+1, 2*xr+1))

  # coordinates of the limits of the extracted image
  uplim = yc + yr + 1  # upper limit
  lolim = yc - yr      # lower limit
  rilim = xc + xr + 1  # right limit
  lelim = xc - xr      # left  limit

  # Ranges (in the original image):
  bot = np.amax((0,  lolim)) # bottom
  top = np.amin((ny, uplim)) # top
  lft = np.amax((0,  lelim)) # left
  rgt = np.amin((nx, rilim)) # right

  im[bot-lolim:top-lolim, lft-lelim:rgt-lelim] = data[bot:top, lft:rgt]
  ret = im

  if type(mask) != type(None):
    ma = np.zeros((2*yr+1, 2*xr+1)) + oob   # The mask is initialized to oob
    ma[bot-lolim:top-lolim, lft-lelim:rgt-lelim] = mask[bot:top, lft:rgt]
    ret = (ret, ma)

  if type(uncd) != type(None):
    un = np.zeros((2*yr+1, 2*xr+1)) + np.amax(uncd[bot:top, lft:rgt])
    un[bot-lolim:top-lolim, lft-lelim:rgt-lelim] = uncd[bot:top, lft:rgt]
    ret = (ret, un) if type(mask)==type(None) else ret + (un,)

  return ret



'''
# function never used?
def pasteimage(data, subim, (dyc,dxc), (syc,sxc)=(None,None)):

  """
    Inserts the subim array into data, the data coordinates (dyc,dxc)
    will match the subim coordinates (syc,sxc). The arrays can have not 
    overlapping pixels. 

    Parameters:
    ----------
    data    : 2D ndarray
              Image where subim will be inserted.
    subim   : 2D ndarray
              Image so be inserted.
    dyc, dxc: 2 elements scalar tuple
              Position in data that will match the (syc,sxc) position 
              of subim.
    syc, sxc: 2 elements scalar tuple
              Semi-length of the extracted image. if not specified, 
              (syc,sxc) will be the center of subim.

    Returns
    -------
    The data array with the subim array inserted, according to the
    given coordinates.

    Example:
    -------

    >>> from imageedit import *
    >>> import numpy as np
    >>> # Create an array and a subimage array to past in.
    >>> data  = np.zeros((5,5), int)
    >>> subim = np.ones( (3,3), int)
    >>> subim[1,1] = 2
    >>> print(data)
    [[0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]]
    >>> print(subim)
    [[1 1 1]
     [1 2 1]
     [1 1 1]]

    >>> # Define the matching coordinates
    >>> dyc,dxc = 3,1
    >>> syc,sxc = 1,1
    >>> # Paste subim into data
    >>> pasteimage(data, subim, (dyc,dxc), (syc,sxc))
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [1, 1, 1, 0, 0],
     [1, 2, 1, 0, 0],
     [1, 1, 1, 0, 0]]

    >>> # Paste subim into data without a complete overlap between images
    >>> data    = np.zeros((5,5), int)
    >>> dyc,dxc = 2,5
    >>> pasteimage(data, subim, (dyc,dxc), (syc,sxc))
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1],
     [0, 0, 0, 0, 1],
     [0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0]]

    Revisions
    ---------
    2010-07-11  patricio  Added documentation.      pcubillos@fulbrightmail.org
  """

  # Shape of the arrays
  dny, dnx = np.shape(data)
  sny, snx = np.shape(subim)

  if (syc,sxc)==(None,None):
    syc, sxc = sny/2, snx/2

  # left limits:
  led = dxc - sxc
  if led > dnx:  # the entire subimage is out of bounds
    return data

  les = np.amax([0,-led])  # left lim of subimage
  led = np.amax([0, led])  # left lim of data


  # right limits:
  rid = dxc + snx - sxc
  if rid < 0:    # the entire subimage is out of bounds
    return data

  ris = np.amin([snx, dnx - dxc + sxc])  # right lim of subimage
  rid = np.amin([dnx, rid])              # right lim of data


  # lower limits:
  lod = dyc - syc
  if lod > dny:  # the entire subimage is out of bounds
    return data

  los = np.amax([0,-lod])  # lower lim of subimage
  lod = np.amax([0, lod])  # lower lim of data


  # right limits:
  upd = dyc + sny - syc
  if upd < 0:    # the entire subimage is out of bounds
    return data

  ups = np.amin([sny, dny - dyc + syc])  # right lim of subimage
  upd = np.amin([dny, upd])              # right lim of data


  data[lod:upd, led:rid] = subim[los:ups, les:ris]

  return data

'''
