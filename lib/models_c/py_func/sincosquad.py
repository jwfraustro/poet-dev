
import numpy as np
import numexpr as ne

def sincosquad(rampparams, t, etc = []):
   """
  This function creates a model that fits a sinusoid and quadratic.

  Parameters
  ----------
    a/b:       amplitude
    p1/p2:	   period
    t1/t2/t3:  phase/time offset
    c:         vertical offset
    r1/r2:     polynomial coefficients
    t:	    Array of time/phase points

  Returns
  -------
	This function returns an array of y values...

  Revisions
  ---------
  2016-06-29	Ryan Challener, UCF 
                rchallen@knights.ucf.edu
                Original version
   """

   a     = rampparams[0]
   p1    = rampparams[1]
   t1    = rampparams[2]
   b     = rampparams[3]
   p2    = rampparams[4]
   t2    = rampparams[5]
   c     = rampparams[6]
   r2    = rampparams[7]
   r1    = rampparams[8]
   t3    = rampparams[9]
   pi    = np.pi

   return ne.evaluate('a*sin(2*pi*(t-t1)/p1) + b*cos(2*pi*(t-t2)/p2) + r2*(t - t3)**2 + r1*(t - t3) + c')

