import numpy as np
import time, timer
import gelman_rubin as gr
import models_c     as mc

def mcmc(params, pmin, pmax, stepsize, parscale, numit, iortholist, fit,
         nchains=10, walk='mrw', gamma=0.0, grtest=False, bound=True,
         factor=0.05):
    """
    Run Markov Chain Monte Carlo model fitting using the
    Metropolis-Hastings algorithm.

    Parameters:
    -----------
    params: 1D ndarray
          Array of initial guess for parameters
    pmin: 1D ndarray
          Array of parameter minimum values
    pmax: 1D ndarray
          Array of parameter maximum values
    stepsize: 1D ndarray
          Array of 1-sigma change in parameter per iteration
    parscale: 1D ndarray
          Array of the the parameter jump scale for use in DEMC.
    numit: Scalar
          Number of iterations to perform
    iortholist: 1D ndarray
          List of orthogonalizing parameters.
    fit: List of fit instances of an event.
    nchains: Scalar
          number of parallel chains to run
    walk: String
          Random walk for the Markov chain: 
          'demc': differential evolution MC.
          'mrw' : Metropolis random walk.
    gamma: Scalar
          Scaling factor for DEMC jumps.
    grtest: Boolean
          Do Gelman and Rubin convergence test. 
    bound: Boolean
          Use bounded-eclipse constrain for parameters (start after the 
          first frame, end before the last frame).
    factor: Scalar
          Shrinking factor of parscale in DEMC.

    Returns:
    --------
    This function returns an array of the best fitting parameters,
    an array of all parameters over all iterations, and numaccept.

    Modification history:
    ---------------------
    2008-05-02  Written by:  Kevin Stevenson, UCF
                kevin218@knights.ucf.edu
    2008-06-21  kevin     Finished updating
    2009-11-01  kevin     Updated for multi events:   
    2010-06-09  kevin     Updated for ipspline, nnint & bilinint
    2011-07-06  kevin     Updated for Gelman-Rubin statistic
    2011-07-22  kevin     Added principal component analysis
    2011-10-11  kevin     Added priors
    2012-09-03  patricio  Added Differential Evolution MC. Documented.
                pcubillos@fulbrightmail.org, UCF
    2013-02-19  patricio  Added parscale and factor parameter. Added 
                          support distribution (e) to DEMC.
    """
    numaccept  = np.zeros(nchains)
    nump       = len(stepsize)
    allparams  = np.zeros((nchains, nump, numit))
    ifree      = np.where(stepsize  > 0)[0] # Free params index
    iequal     = np.where(stepsize  < 0)[0] # Shared params index
    inotfixed  = np.where(stepsize != 0)[0] # Free and shared params index
    outside    = np.zeros(nump)             # Out-of-bounds counter
    numevents  = len(fit)
    intsteps   = np.min((numit/5, 1e5)) 

    if np.size(params) == np.size(stepsize):
      # Make it 2D shaped:
      params = np.repeat(np.atleast_2d(params), nchains, 0)
      if nchains > 1:
        for p in ifree:
          # A random number between pmin and pmax: 
          #params[1:, p] = np.random.uniform(pmin[p], pmax[p], (nchains-1))
          params[1:, p] = np.random.normal(params[0,p], stepsize[p], nchains-1)
          # Stay within boundaries:
          params[np.where(params[:,p] < pmin[p]),p] = pmin[p]
          params[np.where(params[:,p] > pmax[p]),p] = pmax[p]

    # Update shared parameters:
    for i in iequal:
      params[:,i] = params[:, int(np.abs(stepsize[i])-1)]

    # Get best-fitting parameters and chi-square:
    bestp     = np.copy(fit[0].fitparams)
    bestchisq = np.copy(fit[0].chisq)
    currchisq = np.ones(nchains)

    # Calculate chi-squared for model type using current params
    for   c in np.arange(nchains):
      nextp = np.copy(params[c]) # Proposed parameters
      pedit = np.copy(params[c]) # Editable parameters
      for j in np.arange(numevents):
        ymodel = np.ones(fit[j].nobj)
        for k in np.arange(fit[j].numm):
          if   fit[j].functypes[k] == 'ortho':
            pedit[iortholist[j]] = fit[j].funcs[k](pedit[iortholist[j]],
                                              fit[j].funcx[k], fit[j].etc[k])
          elif fit[j].functypes[k] == 'ipmap':
            fit[j].etc[k]   = ymodel
            ymodel *= fit[j].funcs[k](pedit[fit[j].iparams[k]], 
                                 fit[j].funcx[k], fit[j].etc[k])
          elif fit[j].functypes[k] == 'posoffset':
            # Record change in Position 0 => cannot orthogonalize
            # position parameters:
            ymodel *= fit[j].funcs[k](nextp[fit[j].iparams[k]],
                                      fit[j].funcx[k], fit[j].etc[k])
          else:
            ymodel *= fit[j].funcs[k](pedit[fit[j].iparams[k]],
                                      fit[j].funcx[k], fit[j].etc[k])

        # Calculate chi-square:
        currchisq[c] += mc.chisquared(ymodel, fit[j].flux, fit[j].sigma)

        # Apply prior, if exists:
        if len(fit[j].ipriors) > 0:
          pbar   = fit[j].priorvals[:,0]  # prior mean
          psigma = np.zeros(len(pbar))    # prior standard deviation
          # Determine psigma based on which side of asymmetric
          # Gaussian nextp is on:
          for i in np.arange(len(fit[j].ipriors)):
            if nextp[fit[j].ipriors[i]] < pbar[i]:
              psigma[i] = fit[j].priorvals[i,1]
            else:
              psigma[i] = fit[j].priorvals[i,2]
            currchisq[c] += ((nextp[fit[j].ipriors[i]]-pbar[i])/psigma[i])**2.0

      if currchisq[c] < bestchisq:
        bestchisq = currchisq[c]
        bestp     = np.copy(params[c])

    # Generate random numbers from mcmc:
    numfree     = len(ifree)
    numnotfixed = len(inotfixed)
    # Random steps for the Metropolis Random Walk:
    step  = np.random.normal(0, stepsize[ifree], (nchains, numit, numfree))

    # Uniform random distribution for the Metropolis acceptance rule:
    unif = np.random.uniform(0, 1, (nchains, numit))

    # DE-MC Set up:
    if gamma == 0:
      gamma = 2.4 / np.sqrt(2*numfree)
    step2 = np.random.normal(0, parscale[ifree]*factor,
                             (nchains, numit, numfree))
    # step3 = np.random.uniform(-parscale[ifree]*factor,parscale[ifree]*factor,
    #                           (nchains, numit, numfree))

    # Start timer:
    clock = timer.Timer(numit,progress = np.arange(0.05,1.01,0.05))

    # Run Metropolis-Hastings Monte Carlo algorithm 'numit' times
    for   n in np.arange(numit):
      for c in np.arange(nchains):
        nextp = np.copy(params[c]) # Proposed parameters

        # Metropolis Random Walk
        #Take step in random direction for adjustable parameters
        if walk == "mrw":
          nextp[ifree] = params[c,ifree] + step[c,n]

        # Differential Evolution MC:
        elif walk == "demc":
          r1 = int(np.random.uniform(0,nchains))
          r2 = int(np.random.uniform(0,nchains))
          while r1 == c:
            r1 = int(np.random.uniform(0,nchains))
          while r2 == c:
            r2 = int(np.random.uniform(0,nchains))
          jump = gamma * (params[r1,ifree] - params[r2, ifree]) + step2[c,n]
          nextp[ifree] = params[c,ifree] + jump

        # Check for new steps outside boundaries:
        ioutside = np.where(np.bitwise_or(nextp < pmin, nextp > pmax))[0]
        if (len(ioutside) > 0):
          nextp[ioutside] = np.copy(params[c,ioutside])
          outside[ioutside] += 1
        if bound:
          for j in np.arange(numevents):
            ioutside = bounds(nextp, fit[j])
            if (len(ioutside) > 0):
              nextp[ioutside] = np.copy(params[c,ioutside])
              outside[ioutside] += 1

        # Update parameters equal to other parameters:
        for i in iequal:
          nextp[i] = nextp[int(np.abs(stepsize[i])-1)]

        # Compute next chi squared and acceptance value:
        pedit     = np.copy(nextp)
        nextchisq = 0
        for j in np.arange(numevents):
          ymodel = np.ones(fit[j].nobj)
          for k in np.arange(fit[j].numm):
            if   fit[j].functypes[k] == 'ortho':
              # Modify copy of nextp only
              pedit[iortholist[j]] = fit[j].funcs[k](pedit[iortholist[j]],
                                                fit[j].funcx[k], fit[j].etc[k])
            elif fit[j].functypes[k] == 'ipmap':
              ipmodel, allknots = fit[j].funcs[k](pedit[fit[j].iparams[k]],
                                      fit[j].funcx[k], ymodel, retbinflux=True)
              ymodel *= ipmodel
              # Compress allknots by subtracting 1, multiplying by
              # 300000, then convert to int16:
#              print("np.int16((allknots[np.where(allknots > 0)]-1.)*300000).shape:", np.int16((allknots[np.where(allknots > 0)]-1.)*300000).shape) # zindme
              np.save(fit[j].allknotpid,
                      np.int16((allknots[np.where(allknots > 0)]-1.)*300000))
            elif fit[j].functypes[k] == 'posoffset':
              # Record change in Pos. 0 => can't orthogonalize
              # position parameters:
              ymodel *= fit[j].funcs[k](nextp[fit[j].iparams[k]],
                                   fit[j].funcx[k], fit[j].etc[k])
            else:
              ymodel *= fit[j].funcs[k](pedit[fit[j].iparams[k]],
                                   fit[j].funcx[k], fit[j].etc[k])

          # Calculate chi-square:
          nextchisq += mc.chisquared(ymodel, fit[j].flux, fit[j].sigma)

          # Apply prior, if exists:
          if len(fit[j].ipriors) > 0:
            pbar   = fit[j].priorvals[:,0] # prior mean
            psigma = np.zeros(len(pbar))   # prior standard deviation
            # Determine psigma:
            for i in range(len(fit[j].ipriors)):
              if nextp[fit[j].ipriors[i]] < pbar[i]:
                psigma[i] = fit[j].priorvals[i,1]
              else:
                psigma[i] = fit[j].priorvals[i,2]
              nextchisq += ((nextp[fit[j].ipriors[i]]-pbar[i])/psigma[i])**2
            del(pbar, psigma)
          del(ymodel, k)

        # Calculate acceptance probability:
        accept = np.exp(0.5 * (currchisq[c] - nextchisq))
        if accept >= unif[c,n]:
          numaccept[c] += 1
          params[c]  = np.copy(nextp)
          currchisq[c]  = nextchisq
          if currchisq[c] < bestchisq:
            bestp     = np.copy(params[c])
            bestchisq = currchisq[c]

        allparams[c,:,n] = params[c]

        # Print intermediate info:
        if ((n+1) % intsteps == 0) and (n > 0) and c==0: 
          print("\n" + time.ctime())
          print("Number of times parameter tries to step outside its prior:")
          print(outside)
          print("Current Best Parameters: ")
          print(bestp)
          # Flush allknots to file:
          for j in range(numevents):
              for k in np.arange(fit[j].numm):
                  if fit[j].functypes[k] == 'ipmap':
                      fit[j].allknotpid.flush()

          # Apply Gelman-Rubin statistic:
          if grtest:
            psrf = gr.convergetest(allparams[:, inotfixed, :n+1])
            print("Gelman-Rubin statistic for free parameters:")
            print(psrf)
            if np.all(psrf < 1.01):
              print("All parameters have converged to within 1% of unity.\n")
              #allparams = allparams[:,0:n+1]
              #break
        
      clock.check(n+1)

    return allparams, numaccept, bestp, bestchisq


def bounds(params, fit):
  """
    Determine if params return an eclipse model starting before/after the
    first/last frame of the lightcurve.

    parameters:
    -----------
    params: 1D ndarray
            list of lightcurve fitting parameters.
    fit: A fits instance.

    Modification history:
    ---------------------
    2012-09-10  patricio  Written by Patricio Cubillos. 
                          pcubillos@fulbrigmail.org
  """
  ioutside = []

  # parameter names:
  midpt = ['midpt', 'midpt2', 'midpt3']
  width = ['width', 'width2', 'width3']

  for i in np.arange(len(midpt)):
    # find indices for eclipse midpoint and width
    if midpt[i] in dir(fit.i):
      mindex = eval('fit.i.%s + fit.indparams[0]'%midpt[i])
      windex = eval('fit.i.%s + fit.indparams[0]'%width[i])

      # evaluate if out-of-bounds:
      if (params[mindex] - params[windex]/2 < fit.timeunit[ 0] or
          params[mindex] + params[windex]/2 > fit.timeunit[-1] ):
        ioutside.append(mindex)
        ioutside.append(windex)

  return ioutside
