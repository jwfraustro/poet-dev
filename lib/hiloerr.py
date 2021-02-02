"""
 NAME:
	HILOERR

 PURPOSE:
	This function computes two-sided error bars for a set of
	values.  It treats Data as a representative set of
	measurements of a quantity, and returns the distance in
	parameter space from Value (default: MEAN(Data)) to the low
	and high bounds of the given CONFidence interval centered on
	Value.  CONF defaults to the Gaussian 1-sigma confidence.

 CATEGORY:
	Statistics.

 CALLING SEQUENCE:

	Result = HILOERR(Data, Value)

 INPUTS:
	Data:	An array of any non-complex numerical type and size
		containing representative measurements of a quantity.
		Bootstrap Monte-Carlo results are often used here.
	Value:	The nominal value of the dataset.  Optional.  If not
		defined, Value is set to MEAN(Data) IN THE CALLER.

 KEYWORD PARAMETERS:
	CONF:	Half the confidence interval desired.  I.e., the
		distance in probability space from Value to the lower
		and upper error bounds.  Optional.  If not set, CONF
		is set to erf(1d/sqrt(2d)) / 2d = 0.34134475 IN THE
		CALLER.
	IVAL:	(returned) The index of the nominal value.  This is an
		interpolated quantity, so it may not be an integer.
	ILO:	(returned) The index of the lower error bound.  This
		is an interpolated quantity, so it may not be an integer.
	IHI:	(returned) The index of the upper error bound.  This
		is an interpolated quantity, so it may not be an
		integer.

 OUTPUTS:
	This function returns a 2-element array giving the distances
	in parameter space from Value to the lower and upper bounds of
	the confidence (error) interval.

 PROCEDURE:
	The function sorts the data, finds (by spline interpolation)
	the index of the nominal value, counts up and down an
	appropriate number of points to find the indices of the lower
	and upper confidence interval bounds, and interpolates to find
	the corresponding parameter values.

 EXAMPLE:

	data = randomn(seed, 1000000, /double) * 2d + 5d
	value = mean(data)
	print, value, hiloerr(data, value)

 MODIFICATION HISTORY:
 	Written by:	Joseph Harrington, Cornell.  	2006-04-25
			jh@oobleck.astro.cornell.edu
	Removed nonunique values from data when finding ival
			Kevin Stevenson, UCF		2008-06-04
			kbstevenson@gmail.com
	Rewrote in python
			Kevin Stevenson, UCF		2008-07-08
			kbstevenson@gmail.com
	Fixed comparisons to None
			Zacchaeus, UCF			2017-06-20
			zaccysc@gmail.com
"""

def hiloerr(data, value = None, conf = 0.34134475):

   import numpy as np

   if value is None:   
      value = np.median(data)
   
   sdat  = np.unique(data)      # sorted, unique values
   ndat  = len(sdat)            # number of points
   idat  = np.arange(ndat)
   ival  = np.interp([value], sdat, idat)
   ilo   = ival - conf * ndat     # interpolated index of low value
   ihi   = ival + conf * ndat     # interpolated index of high value
   loval = np.interp(ilo, idat, sdat)
   hival = np.interp(ihi, idat, sdat)
   loerr = loval - value
   hierr = hival - value
   
   return loerr[0], hierr[0]
