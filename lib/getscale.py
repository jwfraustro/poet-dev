import numpy    as np
import models_c as mc

def burnin_scale(datacube, thinning):
  """
  Estimate the typical scale of the distribution of a Markov Chain run.

  Parameters:
  -----------
  datacube: [Type] 3D ndarray
            A cube of data with axes: (nchains, nparameters, niterations)
  thinning: [Type] Scalar
            An integer 

  Return:
  -------
  scale: [Type] 1D ndarray
         The standard deviation of the combined chains for each
         parameter in the datacube.

  Modification History:
  ---------------------
  2013-02-19  patricio  Initial implementation.  pcubillos@fulbrightmail.org
  """
  # Gen number of free parameters:
  npars = np.shape(datacube)[1]
  scale = np.zeros(npars)
  # Calculate the scale as the standard deviation of the chain:
  for par in np.arange(npars):
    scale[par] = np.std(datacube[:, par, ::thinning].flatten())
  return scale


def chisq_scale(fit, params, stepsize, pmin, pmax, evalmodel, precision=0.01):
  """
  Determine the half difference of parameter values where chisq has
  increased by 1 with respect to the minimum chisq.

  Parameters:
  -----------
  fit: [Type] Object
       A fit object from a POET event.
  params: [Type] 1D ndarray
       The best fitting parameters.
  stepsize: [Type] 1D ndarray
       The MCMC jump stepsize. Same shape as params. 
  pmin: [Type] 1D ndarray
       The lower boundary of the fitting parameters.
  pmax: [Type] 1D ndarray
       The lower boundary of the fitting parameters.
  evalmodel: [Type] Function
       The modeling function.
  precision: [Type] Scalar
       Tolerance limit to accept the parameter: |Delta(chisq)| < precision.
       
  Return:
  -------
  scale: [Type] 1D ndarray
         Half-difference of greater and smaller parameters where chisq has
         increased by 1.

  Pseudo-code:
  ------------
  - Get minimum chi square evaluating model with best fitting parameters:
      chisq_min  
  - Vary each parameter (fixing the rest at ther best fitting values),
    to find the values greater than (param_hi) and smaller than
    (param_lo) the best fitting value, where:
      chisq(param_hi) = chisq(param_lo) = chisq_min + 1
  - Return the half-difference:
      scale = (param_hi-param_lo)/2

  Notes:
  ------
  This code uses the hunt algorithm to find the hi and lo params.
  See Figure 3.1.1.b of Numerical Recipes Third Edition.

  Modification History:
  ---------------------
  2013-02-21  patricio  Initial implementation.  pcubillos@fulbrightmail.org
  """
  npars = len(params)

  # Best chisq:
  bestchisq = 0.0
  
  # Evaluate model using best values and calculate best chisq:
  bestchisq = calc_chisq(fit, params, evalmodel)

  # Use stepsize to find the free and shared parameter indices:
  ifree  = np.where(stepsize  > 0)[0]
  ishare = np.where(stepsize  < 0)[0]

  # Allocate the lower and higher parameters:
  lohiparams = np.zeros(2*npars)

  # Set initial sampling jump at 1e-2 of the best-parameter value:
  jump = np.zeros(len(params))
  jump[ifree] = np.abs(np.asarray(params[ifree]) * 1e-2)
  # For smaller and larger values:
  jump = np.concatenate( (-jump,jump) )

  # Loop over each par:
  for p in np.arange(len(jump)):
    neval = 0 # Number of evaluations

    # The parameter index:
    parindex = p % npars

    samplepars = np.copy(params) # Copy of fitting params to evaluate
    pjump = jump[p]              # jump in parameter 'p'
    plow  = params[parindex]     # Lower constrain value of the parameter
    phigh = params[parindex] + pjump # Higher constrain value

    # Is the target value confined?: bestchisq - chisq[phigh] > 1
    confined = False  
    # Is the target value out of bounds?: phigh > pmax or phigh < pmin.
    # Also used to skip non-fitting parameters.
    outofbounds = not parindex in ifree

    # Confine the parameter:
    while (not confined) and (not outofbounds):
      # Constrain value within bounds:
      phigh = np.clip(phigh, pmin[parindex], pmax[parindex])

      # Eval chisq:
      samplepars[parindex] = phigh
      for i in ishare:
        samplepars[i] = samplepars[int(np.abs(stepsize[i])-1)]
      newchisq = calc_chisq(fit, samplepars, evalmodel)
      neval += 1

      # Evaluate confine condition:
      if newchisq - bestchisq > 1.0: # I confined it
        confined = True
      else:                          # I'm still looking
        if (phigh == pmin[parindex]) or (phigh == pmax[parindex]):
          outofbounds = True 
        else:
          pjump  *= 1.5
          plow = phigh
          phigh += pjump


    # Once confined, hunt down:
    if not outofbounds:
      phalf = (plow + phigh)/2
      # Eval chisq at half point:
      samplepars[parindex] = phalf
      for i in ishare:
        samplepars[i] = samplepars[int(np.abs(stepsize[i])-1)]
      newchisq = calc_chisq(fit, samplepars, evalmodel)
      neval += 1
      
      while np.abs(newchisq - bestchisq - 1) > precision:
        if newchisq - bestchisq > 1.0:
          phigh = phalf
        else:
          plow = phalf
        phalf = (plow + phigh)/2

        # Eval chisq:
        samplepars[parindex] = phalf
        for i in ishare:
          samplepars[i] = samplepars[int(np.abs(stepsize[i])-1)]
        newchisq = calc_chisq(fit, samplepars, evalmodel)
        neval += 1

      lohiparams[p] = phalf
    else:  # If out of bounds
      lohiparams[p] = phigh

  # return the half-difference:
  scale = np.zeros(npars)
  scale[ifree] = (lohiparams[npars:] - lohiparams[:npars])[ifree] / 2.0
  return scale


def calc_chisq(fit, params, evalmodel):
  """
  Calculate chi square.

  Parameters:
  -----------
  fit: [Type] Object
       An event fit's instance.
  params: [Type] 1D ndarray
       The current set of fitting parameter to evaluate the model.
  evalmodel: [Type] Function
       The model function.

  Return:
  -------
  chisq: Scalar
         Chi square of the model fit.

  Modification history:
  ---------------------
  2013-02-21  patricio   Initial implementation.   pcubillos@fulbrightmail.org
  """
  chisq = 0.0
  # Evaluate model and calculate best chisq:
  for j in np.arange(len(fit)):
    ymodel = evalmodel(params, fit[j])
    chisq += mc.chisquared(ymodel, fit[j].flux, fit[j].sigma)

  if len(fit[j].ipriors) > 0:
    pbar = fit[j].priorvals[:,0]
    psigma = np.zeros(len(pbar))

    for i in np.arange(len(fit[j].ipriors)):
      if params[fit[j].ipriors[i]] < pbar[i]:
        psigma[i] = fit[j].priorvals[i,1]
      else:
        psigma[i] = fit[j].priorvals[i,2]
      chisq += ((params[fit[j].ipriors[i]]-pbar[i])/psigma[i])**2.0

  return chisq
