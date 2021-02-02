from __future__ import print_function
import numpy   as np
import numexpr as ne
import matplotlib.pyplot as plt
import scipy.optimize    as op
import mpl_toolkits.mplot3d as m3d
import os, sys, time
import readeventhdf
import plots
import smoothing
import models_c  as mc
import mcmc      as mcmc
import bindata   as bd
import paramedit as pe
import correlated_noise as cn
import getscale as gs
sys.path.append('mccubed')
sys.path.append('lib/mccubed')
import MCcubed as mc3
"""
  this module runs Markov Chain Monte Carlo model fitting using the
  Metropolis-Hastings algorithm or Differential Evolution MC.

  Main Routines:
  --------------
  setup:    Set up the event for MCMC simulation. Initialize the models and
            performs a least-squares fit.
  runburn:  Run the burn-in MCMC, produce trace plots and normalized plots of 
            the models for the last iteration.
  finalrun: Run the final MCMC simulation (starting from last iteration in 
            burn-in), produce plots, and save results. 

  Auxiliary routines:
  -------------------
  get_minnumpts: Produce mask of minimum number of points per bin condition.
  evalmodel: Evaluate light-curve model with given set of parameters.
  noeclipse: Evaluate a no-eclipse light-curve model.
  residuals: Calculate the residual between the lightcurves and models.
  modelfit:  Least-squares wrapper for ligtcurve fitting.
  runmcmc:   MCMC simulation wrapper.

  Modification History:
  ---------------------
    Written by: Kevin Stevenson, UCF   
    2008-07-02  kevin218@knights.ucf.edu
    2008-09-08  kevin     Finished initial version
    2009-11-01  kevin     Updated for multi events
    2010-06-28  kevin     Added ip interpolation
    2010-08-03  kevin     Added minnumpts and fixipmap
    2010-08-19  kevin     Least-squares with multi events
    2010-11-17  kevin     Use phase or bjd as time unit
    2010-12-11  kevin     Minor improvements from pipeline meeting
    2010-12-31  kevin     Added Ballard IP method
    2011-01-25  kevin     Added sinusoidal-type models
    2011-03-19  kevin     Added bjdutc and bjdtdb
    2011-03-31  kevin     Write allparams to file
    2011-07-18  kevin     Added autocorr & 2D hist plots
    2011-07-26  kevin     Added param orthogonalization
    2011-10-11  kevin     Added priors
    2012-09-10  patricio  Added DEMC. Separated main modules into setup,
                          burnin, and final MCMC.  Added documentation.
                          Heavy code cleaning.      pcubillos@fulbrightmail.org
    2013-02-26  patricio  Added Support distribution (e) to DEMC. Freed prior
                          paramerters in minimizer, with prior-penalty.
    2017-02-07  rchallen  Updates to work with current python.
                          rchallen@knights.ucf.edu
    2018-01-06  zacchaeus updated to python3
"""

def setup(event, num=0, printout=sys.stdout,  mode='turtle', numit=None):
    """
    Set up the event for MCMC simulation. Initialize the models and
    performs a least-squares fit.
    
    Code content:
    ------------
    - Initialize event.fit instances.
    - Read-in, mask, sort, and flatten data.
    - Read initparams file.
    - Initialize models.
    - Set up priors.
    - Clip data.
    - set up Bliss map.
    - Set up time units.
    - Initialize ortho variables
    - Set up independent parameters for light-curve models.
    - Create binned time-data.
    - Do Least-square fitting with initial parameters.
    - Scale data uncertainties to get reduced chisq == 1.0
    - Re-do least-squares fitting with new uncertainties.
    - Save best-fitting parameters to initvals file.
    - Start burn-in MCMC.

    Parameters:
    -----------
    event: List of event instances
    num:   Scalar
           Identification number for the set of lightcurve models
    printout: File object for directing print statements
    mode:  String
           MCMC mode: ['burn' | 'continue' | 'final']
    numit: Scalar
           Number of MCMC iterations.

    Modification history:
    ---------------------
    2012-09-10  patricio  Added Documentation. Heavy code cleaning.
    """

    # If allplots=False: the correlation and histogram figures are not produced.
    # If normflux=Flase: flux is not normalized wrt each position's median value
    #                    This should only be used when using normflux model.

    nbins        = []
    normflux     = []
    numevents    = len(event)
    nummodels    = np.zeros(numevents, dtype=int)
    fit          = []
    params       = []
    pmin         = []
    pmax         = []
    stepsize     = []
    numparams    = [0]
    iparams      = []
    parlist      = []
    pcounter     = 0

    """
      Get light-curve data.
      Load models, and initial parameters.
      Fills: nummodels, fit, params, pmin, pmax, stepsize, numparams, iparams,
      parlist, functype, pcounter.
    """

    for j in range(numevents):
      # Initialize fits class:
      fit.append(readeventhdf.fits()) 
      # Attach it to the event:
      event[j].fit.append(fit[j])
      fit[j].model = event[j].params.model[num]

      fit[j].numfigs = 20

      # Update event.params
      event[j].params.numit = event[0].params.numit.astype(np.int)
      fit[j].nbins = event[j].params.nbins
      normflux.append(event[j].params.normflux)
      if hasattr(event[j].params, 'nchains') == False:
          event[j].params.nchains = 4
      if hasattr(event[j].params, 'newortho') == False:
          event[j].params.newortho = False
      event[j].isortho = False

      # Read in data
      fit[j].mflux = np.mean(event[j].aplev[np.where(event[j].good)])
      # Normalize flux at each position if using > 1 position
      if event[j].npos > 1 and normflux[j]:
        # Calculate mean flux per position:
        fit[j].posmflux = np.zeros((event[j].good.shape[0],1))
        for i in range(event[j].good.shape[0]):
          posgood = np.where(event[j].good[i])
          fit[j].posmflux[i] = np.mean(event[j].aplev[i, posgood])

        fit[j].fluxuc  = (event[j].aplev/fit[j].posmflux)[
                                           np.where(event[j].good)]*fit[j].mflux

        fit[j].sigmauc = (event[j].aperr/fit[j].posmflux)[
                                           np.where(event[j].good)]*fit[j].mflux
      else:
          fit[j].fluxuc   = event[j].aplev[np.where(event[j].good)]
          fit[j].sigmauc  = event[j].aperr[np.where(event[j].good)]

      fit[j].phaseuc  = event[j].phase [np.where(event[j].good)]
      fit[j].yuc      = event[j].y     [np.where(event[j].good)]
      fit[j].xuc      = event[j].x     [np.where(event[j].good)]
      fit[j].time     = event[j].time  [np.where(event[j].good)]
      fit[j].bjdcor   = event[j].bjdcor[np.where(event[j].good)]
      fit[j].juldat   = event[j].juldat[np.where(event[j].good)]
      if event[j].method == 'fgc' or event[j].method == 'rfgc':
          fit[j].ysiguc = event[j].fp.ysig[np.where(event[j].good)]
          fit[j].xsiguc = event[j].fp.xsig[np.where(event[j].good)]
      if hasattr(event[j], 'bjdutc') == False:
          event[j].bjdutc = event[j].juldat + event[j].bjdcor/86400.0
      if hasattr(event[j], 'bjdtdb') == False:
          event[j].bjdtdb = event[j].bjdutc
          print('****WARNING: BJD_TDB does not exist. Using BJD_UTC.')
      fit[j].bjdutcuc = event[j].bjdutc[np.where(event[j].good)]
      fit[j].bjdtdbuc = event[j].bjdtdb[np.where(event[j].good)]
      fit[j].posuc    = event[j].pos   [np.where(event[j].good)]
      fit[j].aoruc    = event[j].fp.aor[np.where(event[j].good)]
      fit[j].frmvisuc = event[j].frmvis[np.where(event[j].good)]
      if hasattr(event[j], 'apdata'):
          fit[j].apdatauc = event[j].apdata[np.where(event[j].good)]
       
      # Sort data by bjdutcuc if using > position
      if event[j].npos > 1:
          isort           = np.argsort(fit[j].bjdutcuc)
          fit[j].phaseuc  = fit[j].phaseuc [isort]
          fit[j].bjdutcuc = fit[j].bjdutcuc[isort]
          fit[j].bjdtdbuc = fit[j].bjdtdbuc[isort]
          fit[j].fluxuc   = fit[j].fluxuc  [isort]
          fit[j].sigmauc  = fit[j].sigmauc [isort]
          fit[j].yuc      = fit[j].yuc     [isort]
          fit[j].xuc      = fit[j].xuc     [isort]
          fit[j].time     = fit[j].time    [isort]
          fit[j].posuc    = fit[j].posuc   [isort]
          fit[j].aoruc    = fit[j].aoruc   [isort]
          fit[j].frmvisuc = fit[j].frmvisuc[isort]
          if hasattr(event[j], 'apdata'):
              fit[j].apdatauc = fit[j].apdatauc[isort]
          if event[j].method == 'fgc' or event[j].method == 'rfgc':
              fit[j].ysiguc = fit[j].ysiguc[isort]
              fit[j].xsiguc = fit[j].xsiguc[isort]

      # Obtain initial model parameters
      nummodels[j] = len(fit[j].model)
      if event[j].params.modelfile.__class__ == str:
        fit[j].modelfile = event[j].ancildir + '/' + event[j].params.modelfile
      if len(event[j].params.modelfile) == len(event[j].params.model):
        fit[j].modelfile = event[j].ancildir + '/' + event[j].params.modelfile[num] 
      else:
        fit[j].modelfile = event[j].ancildir + '/' + event[j].params.modelfile[0]
      parlist.append(pe.read(fit[j].modelfile, fit[j].model, event[j]))
      for i in np.arange(nummodels[j]):
        pars = parlist[j][i][2]
        if fit[j].model[i] == 'ipspline':
          # Read number of knots along x and y:
          numipyk, numipxk = pars[0]
          # Initial intra-pixel parameters:
          temppars = [np.ones(numipxk*numipyk)*1.0,
                      np.ones(numipxk*numipyk)*pars[1][0],
                      np.ones(numipxk*numipyk)*pars[2][0],
                      np.ones(numipxk*numipyk)*pars[3][0]]
          pars = temppars
          
        params    = np.concatenate((params,   pars[0]),0)
        pmin      = np.concatenate((pmin,     pars[1]),0)
        pmax      = np.concatenate((pmax,     pars[2]),0)
        stepsize  = np.concatenate((stepsize, pars[3]),0)
        numparams = np.concatenate((numparams,
                                    [numparams[-1] + len(pars[0])]),0)
        iparams.append(range(pcounter, pcounter+len(pars[0])))
          
        # Check for shared parameters, and make changes to
        # parameter values and limits:
        for k in range(len(pars[0])):
          if pars[3][k] < 0:

            params [pcounter+k] = params [int(-pars[3][k]-1)]
            pmin   [pcounter+k] = pmin   [int(-pars[3][k]-1)]
            pmax   [pcounter+k] = pmax   [int(-pars[3][k]-1)]
            #iparams[-1][k]      = int(-pars[3][k]-1)

        pcounter += len(pars[0])

      # Setup models by declaring functions, function types,
      # parameter names, indices, and save extentions:
      func, funct, fit[j].parname, fit[j].i, fit[j].saveext = \
            mc.setupmodel(fit[j].model, fit[j].i)
      fit[j].funcs     = func
      fit[j].functypes = funct
      if np.where(funct == 'ipmap') != (nummodels[j] - 1):
        print("ERROR: The interpolation model must be the last model listed.")
        return

    """
      Set priors, quadrant positions, and clip points.
      Set up intrapixel model.
    """
    cummodels    = [0] # Cumulative list of number of models
    ifreepars    = np.array([], dtype=int) 
    isprior      = np.array([], dtype=int) 
    numfreepars  = np.array(np.where(stepsize > 0)).flatten().size
    isipmapping  = False
    print("\nCurrent event & model:", file=printout)
    for j in range(numevents):
      print(event[j].eventname, file=printout)
      print(fit[j].model,       file=printout)
      cummodels.append(int(cummodels[j] + nummodels[j]))
      # Number of models in event[j]
      fit[j].numm = cummodels[j+1] - cummodels[j]
      # Index of the models in event[j]
      fit[j].indmodels = np.arange(cummodels[j], cummodels[j+1])

      # Number of parameters in event[j]
      fit[j].nump = numparams[cummodels[j+1]] - numparams[cummodels[j]]
      # Index of parameters in event[j]
      fit[j].indparams = np.arange(numparams[cummodels[j]],
                                   numparams[cummodels[j+1]])

      fit[j].pmin     = pmin    [fit[j].indparams]
      fit[j].pmax     = pmax    [fit[j].indparams]
      fit[j].stepsize = stepsize[fit[j].indparams]
      # fit[j].parscale = np.zeros(np.shape(stepsize)) + 1e-30
      fit[j].iparams = iparams[cummodels[j]:cummodels[j+1]]

      # Check for priors, assign indices and
      #   priorvals=[mean, low, high] to fit:
      fit[j].ipriors   = []
      fit[j].priorvals = []
      fit[j].isprior   = np.zeros(fit[j].nump)

      if (hasattr(event[j].params, 'priorvars') and
          len(event[j].params.priorvars) > 0):
          i = 0
          for pvar in event[j].params.priorvars:
              ipvar = getattr(fit[j].i, pvar) + numparams[cummodels[j]]
              fit[j].ipriors.append(ipvar)
              fit[j].isprior[ipvar-numparams[cummodels[j]]] = 1
              params[ipvar] = event[j].params.priorvals[i][0]
              i += 1
          fit[j].priorvals = event[j].params.priorvals
      isprior = np.concatenate((isprior, fit[j].isprior), 0)

      # Specify indices of free and nonprior parameters:
      # Index of first and last model for event j: 
      mini, mend = numparams[cummodels[j]], numparams[cummodels[j+1]]
      fit[j].ifreepars    = np.where(stepsize[fit[j].indparams] >  0)[0].flatten()
      fit[j].nonfixedpars = np.where(stepsize[fit[j].indparams] != 0)[0].flatten()

      ifreepars = np.concatenate((ifreepars, fit[j].ifreepars + mini), 0)
      fit[j].numfreepars = fit[j].ifreepars.size

      # Determine the centroid location relative to the center of pixel:
      # Record in which quadrant the center o flight falls:
      fit[j].nobjuc   = fit[j].fluxuc.size
      fit[j].quadrant = np.zeros(fit[j].nobjuc)
      fit[j].numq     = np.zeros(4)
      yround          = np.round(np.median(fit[j].yuc))
      xround          = np.round(np.median(fit[j].xuc))
      fit[j].y        = fit[j].yuc - yround
      fit[j].x        = fit[j].xuc - xround

      print("Positions are with respect to (y,x): " + str(int(yround)) +
            ", " + str(int(xround)), file=printout)
      # Determine pixel quadrant based on ydiv and xdiv boundary conditions:

      upq   = fit[j].y > event[j].params.ydiv  # in Upper quad 
      leftq = fit[j].x < event[j].params.xdiv  # in Left  quad

      fit[j].quadrant[np.where(  upq & 1-leftq)] = 1  # upper right
      fit[j].quadrant[np.where(1-upq & 1-leftq)] = 2  # lower right
      fit[j].quadrant[np.where(1-upq &   leftq)] = 3  # lower left

      fit[j].numq[0] = np.sum(  upq &   leftq) # number of points in quads
      fit[j].numq[1] = np.sum(  upq & 1-leftq)
      fit[j].numq[2] = np.sum(1-upq & 1-leftq)
      fit[j].numq[3] = np.sum(1-upq &   leftq)

      print("Number of frames per pixel quadrant:", file=printout)
      print(fit[j].numq, file=printout)

      # Clip first and last points if requested:
      if len(event[j].params.preclip) == len(event[j].params.model):
        preclip  = event[j].params.preclip[num]
      else:
        preclip  = event[j].params.preclip[0]
      if len(event[j].params.postclip) == len(event[j].params.model):
        postclip = fit[j].nobjuc - event[j].params.postclip[num]
      else:
        postclip = fit[j].nobjuc - event[j].params.postclip[0]

      fit[j].preclip  = preclip
      fit[j].postclip = postclip

      # Define clipmask: 1 is kept, 0 is clipped
      fit[j].clipmask = np.zeros(fit[j].nobjuc)
      fit[j].clipmask[preclip:postclip] = 1
      # Use interclip if it exists to clip points in the middle
      if hasattr(event[j].params, 'interclip'):
          interclip   = event[j].params.interclip
          for i in range(len(interclip)):
              fit[j].clipmask[interclip[i][0]:interclip[i][1]] = 0

      # Define ipmask: 1 is used in the intrapixel interpolation model, 0 not
      fit[j].ipmaskuc = np.ones(fit[j].nobjuc)
      if hasattr(event[j].params, 'ipclip'):
        ipclip = event[j].params.ipclip
        for i in range(len(ipclip)):
          fit[j].ipmaskuc[ipclip[i][0]:ipclip[i][1]] = 0

      # Define intrapixel mask, (y,x) position, and number of points
      # after clipping:
      fit[j].ipmask   = np.copy(fit[j].ipmaskuc[np.where(fit[j].clipmask)])
      fit[j].position = np.array([fit[j].y[np.where(fit[j].clipmask)], 
                                  fit[j].x[np.where(fit[j].clipmask)], 
                                  fit[j].quadrant[np.where(fit[j].clipmask)]])
      if event[j].method == 'fgc' or event[j].method == 'rfgc':
        fit[j].sigpos   = np.array([fit[j].ysiguc[np.where(fit[j].clipmask)],
                                    fit[j].xsiguc[np.where(fit[j].clipmask)]])
        fit[j].sigposuc = np.array([fit[j].ysiguc,
                                    fit[j].xsiguc])
      fit[j].nobj     = fit[j].position[0].size

      # check for default params.xstep and params.ystep values
      event[j].params.ystep = [step if step != 0 else event[j].yprecision
                               for step in event[j].params.ystep]
      event[j].params.xstep = [step if step != 0 else event[j].xprecision
                               for step in event[j].params.xstep]

      # Calculate minnumptsmask for at least minnumpts in each bin:
      fit[j].minnumptsmask = np.ones(fit[j].nobj, dtype=int)
      for k in np.arange(fit[j].numm):
        if fit[j].functypes[k] == 'ipmap':
          # Calculate fit[j].minnumptsmask (implicit in get_minnumpts):
          get_minnumpts(event[j].params, fit[j], num, mode=1)

      # Redefine clipped variables based on minnumpts for IP mapping 
      fit[j].clipmask[np.where(fit[j].clipmask)] *= fit[j].minnumptsmask
      cmask = np.where(fit[j].clipmask) # clip masked indices
      fit[j].phase    = fit[j].phaseuc [cmask]
      fit[j].bjdutc   = fit[j].bjdutcuc[cmask]
      fit[j].bjdtdb   = fit[j].bjdtdbuc[cmask]
      if hasattr(fit[j], 'sigpos'):
          fit[j].sigpos = np.array([fit[j].ysiguc[cmask],
                                    fit[j].xsiguc[cmask]])
      fit[j].flux     = np.copy(fit[j].fluxuc  [cmask])
      fit[j].sigma    = np.copy(fit[j].sigmauc [cmask])
      fit[j].pos      = np.copy(fit[j].posuc   [cmask])
      fit[j].frmvis   = np.copy(fit[j].frmvisuc[cmask])
      fit[j].ipmask   = np.copy(fit[j].ipmaskuc[cmask])
      fit[j].aor      = np.copy(fit[j].aoruc   [cmask])

      # FINDME: may not need with new pipeline.
      if hasattr(event[0], 'apdata'):
        fit[j].apdata   = np.copy(fit[j].apdatauc[cmask])
      fit[j].position = np.array([fit[j].y       [cmask], 
                                  fit[j].x       [cmask], 
                                  fit[j].quadrant[cmask]])
      fit[j].nobj     = fit[j].flux.size
      fit[j].positionuc = np.array([fit[j].y, fit[j].x, fit[j].quadrant])

      # Print number of data points:
      print("Total observed points:      %7i"
            %np.size(event[j].phase), file=printout)
      print("Raw light-curve points:     %7i"
            %fit[j].nobjuc, file=printout)
      print("Clipped light-curve points: %7i"
            %(fit[j].nobj + np.sum(1-fit[j].minnumptsmask)), file=printout)
      print("BLISS masked-out points:    %7i"
            %np.sum(1-fit[j].minnumptsmask), file=printout)
      print("Fitted light-curve points:  %7i"
            %fit[j].nobj, file=printout)

      # Determine bin location for each position
      fit[j].isipmapping = False
      fit[j].numknots    = 0
      fit[j].ballardip   = np.ones(fit[j].flux.size)
      fit[j].ballardipuc = np.ones(fit[j].fluxuc.size)
      for k in np.arange(fit[j].numm):
        if fit[j].functypes[k] == 'ipmap':
          isipmapping           = True
          fit[j].isipmapping    = True # Using mapping for IP sensitivity
          fit[j].wherebinflux   = []   # List of size = # of bins, def which points fall into each bin
          fit[j].wherebinfluxuc = []   # Un-clipped version of above
          fit[j].wbfipmask      = []   # Intrapixel masked version of wherebinflux
          fit[j].wbfipmaskuc    = []   # Un-clipped version of above
          fit[j].binloc   = np.zeros((2, fit[j].nobj),   dtype=int) - 1
          fit[j].binlocuc = np.zeros((2, fit[j].nobjuc), dtype=int) - 1
          # Read bin sizes
          # Calculate wherebinfluxes: 
          get_minnumpts(event[j].params, fit[j], num, mode=2,
                        printout=printout, verbose=True)

          fit[j].numknots = np.sum(fit[j].binfluxmask)
          # Read smoothing paramters:
          nindex, sindex = 0, 0
          if len(event[j].params.nx) == len(event[j].params.model):
            nindex = num # if one nx per model is provided
          ny = event[j].params.ny[nindex]
          nx = event[j].params.nx[nindex] 
          if len(event[j].params.sx) == len(event[j].params.model):
            sindex = num # if one sx per model is provided
          sy = event[j].params.sy[sindex]
          sx = event[j].params.sx[sindex]

          # Adjust weighting based on number of points in bin
          weightbinfluxmask   = fit[j].binfluxmask
          weightbinfluxmaskuc = fit[j].binfluxmaskuc

          # Calculate smoothing kernel
          fit[j].smoothingp = [ny, nx, sy, sx]
          fit[j].kernel   = smoothing.gauss_kernel_mask(ny, nx,
                                  sy, sx, weightbinfluxmask)
          fit[j].kerneluc = smoothing.gauss_kernel_mask(ny, nx,
                                  sy, sx, weightbinfluxmaskuc)
          fit[j].binfluxmask   = fit[j].binfluxmask  .flatten()
          fit[j].binfluxmaskuc = fit[j].binfluxmaskuc.flatten()

          # Determine distances to four nearest grid points
          # Used for bilinear interpolation
          # ORDER [grid #, (y-y1)/ystep, (y2-y)/ystep, (x-x1)/xstep, (x2-x)/xstep)
          print('Computing distances to four nearest bins.')
          fit[j].griddist   = np.ones((4, fit[j].nobj  ))
          fit[j].griddistuc = np.ones((4, fit[j].nobjuc))
          ysize, xsize = np.shape(fit[j].numpts)
          ygrid, xgrid = fit[j].ygrid, fit[j].xgrid
          num1 = num
          if len(event[j].params.ystep) != len(event[j].params.model):
            num1 = 0
          ystep = fit[j].ystep #event[j].params.ystep[num1]
          xstep = fit[j].xstep #event[j].params.xstep[num0]
          for m in range(ysize-1):
            wherey = np.where(np.bitwise_and(
                               fit[j].position[0] > ygrid[m  ], 
                               fit[j].position[0] < ygrid[m+1]))[0]
            for n in range(xsize-1):
              wherexy = wherey[np.where(np.bitwise_and(
                         fit[j].position[1,[wherey]] > xgrid[n  ],
                         fit[j].position[1,[wherey]] < xgrid[n+1])[0])[0]]
              if len(wherexy) > 0:
                fit[j].binloc[1, wherexy] = gridpt = m*xsize + n
                # If there are no points in one or more bins:
                if (len(fit[j].wbfipmask[gridpt        ]) == 0) or \
                   (len(fit[j].wbfipmask[gridpt      +1]) == 0) or \
                   (len(fit[j].wbfipmask[gridpt+xsize  ]) == 0) or \
                   (len(fit[j].wbfipmask[gridpt+xsize+1]) == 0):
                    # Set griddist = nearest bin (use nearest
                    # neighbor interpolation)
                    for loc in wherexy:
                      if   loc in fit[j].wherebinflux[gridpt        ]:
                        fit[j].griddist[0, loc] = 0
                        fit[j].griddist[2, loc] = 0
                      elif loc in fit[j].wherebinflux[gridpt      +1]:
                        fit[j].griddist[0, loc] = 0
                        fit[j].griddist[3, loc] = 0
                      elif loc in fit[j].wherebinflux[gridpt+xsize  ]:
                        fit[j].griddist[1, loc] = 0
                        fit[j].griddist[2, loc] = 0
                      elif loc in fit[j].wherebinflux[gridpt+xsize+1]:
                        fit[j].griddist[1, loc] = 0
                        fit[j].griddist[3, loc] = 0
                else:
                  # Calculate griddist normally for bilinear interpolation
                  fit[j].griddist[0, wherexy] = np.array((fit[j].position[0][wherexy]-ygrid[m])  /ystep)
                  fit[j].griddist[1, wherexy] = np.array((ygrid[m+1]-fit[j].position[0][wherexy])/ystep)
                  fit[j].griddist[2, wherexy] = np.array((fit[j].position[1][wherexy]-xgrid[n])  /xstep)
                  fit[j].griddist[3, wherexy] = np.array((xgrid[n+1]-fit[j].position[1][wherexy])/xstep)
          # Repeat for uc
          for m in range(ysize-1):
            wherey = np.where(np.bitwise_and(fit[j].y > ygrid[m  ],
                                             fit[j].y < ygrid[m+1]))[0]
            for n in range(xsize-1):
              wherexy = wherey[np.where(np.bitwise_and(
                                   fit[j].x[wherey] > xgrid[n  ],
                                   fit[j].x[wherey] < xgrid[n+1]))[0]]
              if len(wherexy) > 0:
                fit[j].binlocuc[1, wherexy] = gridpt = m*xsize + n
                # If there are no points in one or more bins:
                if (len(fit[j].wbfipmaskuc[gridpt        ]) == 0) or \
                   (len(fit[j].wbfipmaskuc[gridpt      +1]) == 0) or \
                   (len(fit[j].wbfipmaskuc[gridpt+xsize  ]) == 0) or \
                   (len(fit[j].wbfipmaskuc[gridpt+xsize+1]) == 0):
                  # Set griddist = nearest bin
                  for loc in wherexy:
                    if loc in fit[j].wherebinfluxuc[gridpt        ]:
                      fit[j].griddistuc[0, loc] = 0
                      fit[j].griddistuc[2, loc] = 0
                    if loc in fit[j].wherebinfluxuc[gridpt      +1]:
                      fit[j].griddistuc[0, loc] = 0
                      fit[j].griddistuc[3, loc] = 0
                    if loc in fit[j].wherebinfluxuc[gridpt+xsize  ]:
                      fit[j].griddistuc[1, loc] = 0
                      fit[j].griddistuc[2, loc] = 0
                    if loc in fit[j].wherebinfluxuc[gridpt+xsize+1]:
                      fit[j].griddistuc[1, loc] = 0
                      fit[j].griddistuc[3, loc] = 0
                else:
                  # Calculate griddist for bilinear interpolation:
                  fit[j].griddistuc[0, wherexy] = np.array((fit[j].y[wherexy]-ygrid[m])  /ystep)
                  fit[j].griddistuc[1, wherexy] = np.array((ygrid[m+1]-fit[j].y[wherexy])/ystep)
                  fit[j].griddistuc[2, wherexy] = np.array((fit[j].x[wherexy]-xgrid[n])  /xstep)
                  fit[j].griddistuc[3, wherexy] = np.array((xgrid[n+1]-fit[j].x[wherexy])/xstep)

          # Combine model parameters into one list:
          fit[j].posflux  = [fit[j].y[np.where(fit[j].clipmask)], 
                             fit[j].x[np.where(fit[j].clipmask)], 
                             fit[j].flux,
                             fit[j].wbfipmask,
                             fit[j].binfluxmask,
                             fit[j].kernel,
                             fit[j].smoothingp,
                             fit[j].binloc,
                             fit[j].griddist,
                             fit[j].gridshape,
                             event[j].params.issmoothing]
          fit[j].posfluxuc= [fit[j].y,
                             fit[j].x,
                             fit[j].fluxuc,
                             fit[j].wbfipmaskuc,
                             fit[j].binfluxmaskuc,
                             fit[j].kerneluc,
                             fit[j].smoothingp,
                             fit[j].binlocuc,
                             fit[j].griddistuc,
                             fit[j].gridshape,
                             event[j].params.issmoothing]

        if fit[j].functypes[k] == 'ballardip':
          print("Computing Ballard intrapixel effect:")
          ipparams = [params[fit[j].i.sigmay], params[fit[j].i.sigmax],
                      int(params[fit[j].i.nbins])]
          position = [fit[j].position, fit[j].ballardip, fit[j].flux]
          fit[j].ballardip = mc.ballardip(ipparams, position ,
                                          etc=[fit[j].ballardip])

    """
      Set up time units
      Set up models
    """
    # Indices of non-prior parameters:
    inonprior  = np.where(stepsize > 0)[0] 
    #inonprior  = np.where(np.bitwise_and(stepsize > 0, isprior == 0))[0] 
    # List of variable indices to orthogonalize:
    iortholist = []
    # Orthogonal parameter names:
    opname = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    # Assigns phase/bjdutc/bjdtdb or x,y-position as independent
    # variable for model fitting:
    for j in np.arange(numevents):
      # Use orbits or days as unit of time:
      if   (hasattr(event[j].params, 'timeunit') and
             (event[j].params.timeunit == 'days' or
              event[j].params.timeunit == 'days-utc') ):
          fit[j].timeunit   = fit[j].bjdutc   - event[j].params.tuoffset
          fit[j].timeunituc = fit[j].bjdutcuc - event[j].params.tuoffset
      elif (hasattr(event[j].params, 'timeunit') and
              event[j].params.timeunit == 'days-tdb'  ):
          fit[j].timeunit   = fit[j].bjdtdb   - event[j].params.tuoffset
          fit[j].timeunituc = fit[j].bjdtdbuc - event[j].params.tuoffset
      else:
          fit[j].timeunit   = fit[j].phase
          fit[j].timeunituc = fit[j].phaseuc
          event[j].params.timeunit = 'orbits'
          event[j].params.tuoffset = 0.0

      # Initialize ortho variables
      fit[j].iortholist2 = []
      if hasattr(event[j].params, 'ortholist'):
        if len(event[j].params.ortholist) == len(event[j].params.model):
          ortholist = event[j].params.ortholist[num] 
        else:
          ortholist = event[j].params.ortholist[0]
      else:
        ortholist = []
      diminvtrans       = len(ortholist)
      # Inverse transformation matrix, initialized to 1
      fit[j].invtrans   = np.matrix(np.identity(diminvtrans))
      # Transformation matrix, initialized to 1
      fit[j].trans      = np.matrix(np.identity(diminvtrans))
      # Origin of coordinate system, initialized to 0
      fit[j].origin     = np.zeros(diminvtrans)
      # Normalizing uncertainties, initialized to 1
      fit[j].orthosigma = np.ones(diminvtrans)                
      iortholist.append([])
      
      # Assign independent variable and extra parameter for each model type:
      fit[j].funcx   = []
      fit[j].funcxuc = []
      fit[j].etc     = []
      for k in np.arange(fit[j].numm):
        if fit[j].functypes[k] == 'ecl/tr':
          fit[j].funcx.  append(fit[j].timeunit  )
          fit[j].funcxuc.append(fit[j].timeunituc)
          if (hasattr(event[j].params, 'timeunit') and
              event[j].params.timeunit in ['days-tdb','days-utc','days']):
               fit[j].etc.append([event[j].period])
          else:
            fit[j].etc.append([1.0])
        elif fit[j].functypes[k] == 'ramp':
          fit[j].funcx.  append(fit[j].timeunit  )
          fit[j].funcxuc.append(fit[j].timeunituc)
        elif fit[j].functypes[k] == 'sinusoidal':
          fit[j].funcx.  append(fit[j].timeunit  )
          fit[j].funcxuc.append(fit[j].timeunituc)
        elif fit[j].functypes[k] == 'ippoly':
          if fit[j].model[k] == 'ipspline':
            fit[j].funcx.  append(fit[j].position  )
            fit[j].funcxuc.append(fit[j].positionuc)
            # Create knots for ipspline:
            fit[j].etc.append(np.meshgrid(
                 np.linspace(fit[j].position[1].min(), fit[j].position[1].max(), numipxk), 
                 np.linspace(fit[j].position[0].min(), fit[j].position[0].max(), numipyk)))
          elif fit[j].model[k] in ['posfluxlinip', 'linip']:
            fit[j].wherepos   = []
            fit[j].whereposuc = []
            fit[j].meanypos   = []
            fit[j].meanxpos   = []
            for i in range(event[j].npos):
              wherepos = np.where(fit[j].pos == i)[0]
              fit[j].wherepos.append(wherepos)
              fit[j].meanypos.append(np.mean(fit[j].position[0][wherepos]))
              fit[j].meanxpos.append(np.mean(fit[j].position[1][wherepos]))
              fit[j].whereposuc.append(np.where(fit[j].posuc == i)[0])
            fit[j].funcx.  append([fit[j].position,   fit[j].nobj,
                                   fit[j].wherepos                  ])
            fit[j].funcxuc.append([fit[j].positionuc, fit[j].nobjuc,
                                   fit[j].whereposuc                ])
            fit[j].etc.append([fit[j].meanypos, fit[j].meanxpos])
          elif fit[j].model[k] in ['quadsigip']:
            fit[j].funcx.  append(fit[j].sigpos  )
            fit[j].funcxuc.append(fit[j].sigposuc)
            fit[j].etc.append([])
          else:
            fit[j].funcx.  append(fit[j].position  )
            fit[j].funcxuc.append(fit[j].positionuc)
            fit[j].etc.append([])
        elif fit[j].functypes[k] == 'ballardip':
          fit[j].funcx.  append([fit[j].position, fit[j].ballardip,fit[j].flux])
          fit[j].funcxuc.append([fit[j].positionuc, fit[j].ballardipuc,
                          fit[j].fluxuc])
        elif fit[j].functypes[k] == 'posoffset':
          fit[j].wherepos   = []
          fit[j].whereposuc = []
          if fit[j].model[k] == 'aorflux':
            ngroup = event[j].naor
            group   = fit[j].aor
            groupuc = fit[j].aoruc                    
          else: # fit[j].model[k] == 'posflux':
            ngroup  = event[j].npos
            group   = fit[j].pos
            groupuc = fit[j].posuc
          for i in range(ngroup):
            fit[j].wherepos  .append(np.where(group   == i)[0])
            fit[j].whereposuc.append(np.where(groupuc == i)[0])
          fit[j].funcx.  append([fit[j].nobj,   fit[j].wherepos  ])
          fit[j].funcxuc.append([fit[j].nobjuc, fit[j].whereposuc])
        elif fit[j].functypes[k] == 'vissen':
          fit[j].funcx.  append([fit[j].frmvis,   event[j].params.vsknots])
          fit[j].funcxuc.append([fit[j].frmvisuc, event[j].params.vsknots])
        elif fit[j].functypes[k] == 'flatf':
          fit[j].funcx.  append(fit[j].apdata  )
          fit[j].funcxuc.append(fit[j].apdatauc)
        elif fit[j].functypes[k] == 'ipmap':
          fit[j].funcx.  append(fit[j].posflux  )
          fit[j].funcxuc.append(fit[j].posfluxuc)
        elif fit[j].functypes[k] == 'ortho':
          event[j].isortho = True
          # Create or load invserse transformation matrix and
          # new coordinate's origin:
          if event[j].params.newortho == False:
            for f in os.listdir('.'):
              if f.endswith(event[j].eventname + '-ortho-' +
                            fit[j].saveext + '.npz'        ):
                print('Loading orthogonalization save file:', f,
                      file=printout)
                fit[j].orthofile = f
                orthofile = np.load(f)
                if len(orthofile['origin']) == diminvtrans:
                  fit[j].invtrans   = np.matrix(orthofile['invtrans'])
                  fit[j].trans      = np.matrix(orthofile['trans'])
                  fit[j].origin     = orthofile['origin']
                  fit[j].orthosigma = orthofile['sigma' ]
                else:
                  print('WARNING: Shape of saved inverse transformation ' +
                        'matrix does not match number of elements in '    +
                        'ortholist. Using identity matrix instead.',
                        file=printout)
          # Load iortholist
          m = 0
          fit[j].iortholist  = []
          for orthovar in ortholist:
            ivar = getattr(fit[j].i, orthovar)
            fit[j].iortholist.append(ivar)
            fit[j].iortholist2.append(ivar + fit[j].indparams[0])
            #params[ivar + numparams[cummodels[j]]] -= fit[j].origin[m]
            m += 1
          fit[j].funcx.  append(fit[j].invtrans)
          fit[j].funcxuc.append(fit[j].invtrans)
          fit[j].etc.append([fit[j].origin,fit[j].orthosigma])
          # Set orthogonal parameter names
          fit[j].opname   = np.copy(fit[j].parname)
          fit[j].opname[fit[j].iortholist] = opname[:diminvtrans]

        # Set up fit.etc:
        if fit[j].functypes[k] not in ['ecl/tr', 'ippoly', 'ortho']:
          fit[j].etc.append([])

    # Bin time-data:
    for j in np.arange(numevents):
      # Clipped data:
      fit[j].binphase, fit[j].binbjdutc, fit[j].binbjdtbd =  \
            bd.bindata(fit[j].nbins, median=[fit[j].phase, fit[j].bjdutc, 
                                             fit[j].bjdtdb])
      # Un-clipped data:
      fit[j].binphaseuc, fit[j].binbjdutcuc, fit[j].binbjdtdbuc = \
           bd.bindata(fit[j].nbins, median=[fit[j].phaseuc, fit[j].bjdutcuc, 
                                            fit[j].bjdtdbuc])

    # Assign abscissa time unit (orbits or days)
    for j in range(numevents):
      if event[j].params.timeunit == 'orbits':
        fit[j].tuall      = event[j].phase.flatten() # Include all frames
        fit[j].timeunit   = fit[j].phase             # Use only clipped frames
        fit[j].timeunituc = fit[j].phaseuc           # Use only clipped frames
        fit[j].abscissa   = fit[j].binphase          # Binned   clipped frames
        fit[j].abscissauc = fit[j].binphaseuc        # Binned unclipped
        fit[j].xlabel     = 'Orbital Phase'
      elif event[j].params.timeunit == 'days-utc':
        fit[j].tuall      = event[j].bjdutc.flatten() - event[j].params.tuoffset
        fit[j].timeunit   = fit[j].bjdutc      - event[j].params.tuoffset
        fit[j].timeunituc = fit[j].bjdutcuc    - event[j].params.tuoffset
        fit[j].abscissa   = fit[j].binbjdutc   - event[j].params.tuoffset
        fit[j].abscissauc = fit[j].binbjdutcuc - event[j].params.tuoffset
        fit[j].xlabel     = 'BJD_UTC - ' + str(event[j].params.tuoffset)
      elif event[j].params.timeunit == 'days-tdb':
        fit[j].tuall      = event[j].bjdtdb.flatten() - event[j].params.tuoffset
        fit[j].timeunit   = fit[j].bjdtdb      - event[j].params.tuoffset
        fit[j].timeunituc = fit[j].bjdtdbuc    - event[j].params.tuoffset
        fit[j].abscissa   = fit[j].binbjdtdb   - event[j].params.tuoffset
        fit[j].abscissauc = fit[j].binbjdtdbuc - event[j].params.tuoffset
        fit[j].xlabel     = 'BJD_TDB - ' + str(event[j].params.tuoffset)
      else:
        fit[j].tuall      = event[j].bjdutc.flatten() - event[j].params.tuoffset
        fit[j].timeunit   = fit[j].bjdutc      - event[j].params.tuoffset
        fit[j].timeunituc = fit[j].bjdutcuc    - event[j].params.tuoffset
        fit[j].abscissa   = fit[j].binbjdutc   - event[j].params.tuoffset
        fit[j].abscissauc = fit[j].binbjdutcuc - event[j].params.tuoffset
        fit[j].xlabel     = 'BJD - ' + str(event[j].params.tuoffset)

    # Least-squares fit:
    if event[0].params.leastsq:
      print("Calculating least-squares fit.")
      # Initial minimization with prior parameters fixed:
      if np.any(isprior != 0):
        herenonprior = np.where(np.bitwise_and(stepsize > 0, isprior == 0))[0]
        for   j in np.arange(numevents):
          for i in np.arange(len(fit[j].ipriors)):
            params[fit[j].ipriors[i]] = fit[j].priorvals[i,0]
        output = modelfit(params, herenonprior, stepsize, fit, pmin, pmax,
                          full=True, nopriors=True)
      # Recalculate, now also fitting the prior parameters:
      output = modelfit(params, inonprior, stepsize, fit, pmin, pmax,
                        full=True)

    # Evaluate model using current values:
    for j in range(numevents):
      fit[j].fit0, fit[j].binipflux, fit[j].binipstd = \
                   evalmodel(params, fit[j], getbinflux=True, getbinstd=True)

      for k in np.arange(fit[j].numm):
        if fit[j].functypes[k] == 'ipmap':
          fit[j].binipflux = fit[j].binipflux.reshape(fit[j].gridshape)
          fit[j].binipstd  = fit[j].binipstd. reshape(fit[j].gridshape)

      # Reduced chi-square:
      fit[j].redchisq = np.sum(((fit[j].fit0 - fit[j].flux) / 
                        fit[j].sigma)**2.0) / (fit[j].nobj - fit[j].numfreepars)
      print("Reduced Chi-square: " + str(fit[j].redchisq), file=printout)

    # Since Spitzer over-estimates errors,
    # Modify sigma such that reduced chi-square = 1
    for j in np.arange(numevents):
      # Uncertainty of data from raw light curve:
      fit[j].rawsigma      = fit[j].sigma
      fit[j].rawsigmauc    = fit[j].sigmauc
      # Scaled data uncertainty such reduced chi-square = 1.0
      fit[j].scaledsigma   = fit[j].sigma   * np.sqrt(fit[j].redchisq)
      fit[j].scaledsigmauc = fit[j].sigmauc * np.sqrt(fit[j].redchisq)

      if event[0].params.chi2flag:
        fit[j].sigma   = fit[j].scaledsigma  
        fit[j].sigmauc = fit[j].scaledsigmauc

    # New Least-squares fit using modified sigma values:
    if event[0].params.leastsq:
      print("Re-calculating least-squares fit with new errors.")
      output = modelfit(params, inonprior, stepsize, fit, pmin, pmax,
                        verbose=True)

    # Calculate current chi-square and store it in fit[0]:
    fit[0].chisq = 0.0
    for j in np.arange(numevents):
      model = evalmodel(params, fit[j])
      fit[0].chisq += np.sum(((model - fit[j].flux)/fit[j].sigma)**2.0)
      
    # Store current fitting parameters in fit[0]:
    fit[0].fitparams = params

    # Calculate jump scale for DEMC:
    # parscale = gs.chisq_scale(fit, params, stepsize, pmin, pmax, evalmodel)

    # for i in inonprior:
    #   if parscale[i] > np.abs(params[i]) and parscale[i] < stepsize[i]:
    #     parscale[i] = stepsize[i]
    #   if parscale[i] == 0:
    #     parscale[i] = stepsize[i] * 0.1
    #   # FINDME: These ifs should never happen if chisq_scale were perfect.
    # print("Parameters scale:\n" +  str(parscale[inonprior]), file=printout)
    # for j in np.arange(numevents):
    #   fit[j].parscale = parscale[fit[j].indparams]

    # Update best-fit parameter list with best fitting parameters:
    for   j in np.arange(numevents):
      for k in np.arange(fit[j].numm):
        if fit[j].model[k] != 'ipspline': # FINDME: what?
          parlist[j][k][2][0] = params[fit[j].iparams[k]]

      # Update initial parameters:
      pe.write(fit[j].modelfile, parlist[j])
#      ier = os.system("cp %s %s/." % (fit[j].modelfile, event[j].modeldir))

      # Bin unclipped-data:
      fit[j].binstduc, fit[j].binfluxuc = bd.bindata(fit[j].nbins,
                                std=[fit[j].sigmauc], weighted=[fit[j].fluxuc])

      fit[j].binstd, fit[j].binflux = bd.bindata(fit[j].nbins,
                                std=[fit[j].sigma], weighted=[fit[j].flux])
      binsize = fit[j].nobj//fit[j].nbins
      if fit[j].preclip > 0:
        fit[j].preclipflux  = fit[j].fluxuc [:fit[j].preclip]
        fit[j].preclipsigma = fit[j].sigmauc[:fit[j].preclip]
        fit[j].binprecstd, fit[j].binprecflux = bd.bindata(fit[j].nbins,
                    std=[fit[j].preclipsigma], weighted=[fit[j].preclipflux], 
                    binsize=binsize)

      if fit[j].postclip < fit[j].nobjuc:
        fit[j].postclipflux  = fit[j].fluxuc [fit[j].postclip:]
        fit[j].postclipsigma = fit[j].sigmauc[fit[j].postclip:]
        fit[j].binpoststd, fit[j].binpostflux = bd.bindata(fit[j].nbins,
                     std=[fit[j].postclipsigma], weighted=[fit[j].postclipflux],
                      binsize=binsize)

      # Array of all parameter of models in fit[j]:
      fit[j].modelpars = params[fit[j].indparams]
      fit[j].chainend  = np.atleast_2d(params[fit[j].indparams])


def finalrun(event, num=0, printout=sys.stdout, numit=None):
  """
  Run the final MCMC simulation (starting from last iteration in burn-in),
  produce plots, and save results.
  
  Code content:
  -------------
  - Perform MCMC simulation.
  - Save and delete allparams.
  - Compute each best fit model on their own.
  - Calculate autocorrelation of free parameters.
  - Compute ortho transformation matrix.
  - Calculate mean, medianm and standard deviation of MCMC parameters.
  - Compute Bayesian Information Criterion.
  - Produce Normalized and normalized-binned models.
  - Calculate SDNR.
  - Produce binned, normalized-binned, trace, autocorrelation, 2D- and 
    1D-histogram, Bliss map, pointing histogram nad RMS vs bin-size plots.

  Parameters:
  -----------
    event: List of event instances
    num:   Scalar
           Identification number for the set of lightcurve models.
    printout: File object for directing print statements
    numit: Scalar
           Number of MCMC iterations.

  Modification history:
  ---------------------
  2012-09-10  patricio  Added DEMC. Added Documentation. 
                        Heavy code cleaning.  pcubillos@fulbrightmail.org
  2017-02-07  rchallen  Updated for current python indexing rules
                        rchallen@knights.ucf.edu
  """

  # Load up data:
  # fit is a list of fit objects, each of which applies to
  # the corresponding event object. Multiple fit objects can
  # go with a single event object if multiple model sets were
  # specified (num designates which one)
  numevents = len(event)
  fit = []
  for j in np.arange(numevents):
    fit.append(event[j].fit[num])

  # Run MCMC:
  if numit == None:
    numit  =  event[0].params.numit[1]

  allparams = runmcmc(event, fit, numit, event[0].params.rwalk, mode='final',
                 grtest=True, printout=printout, bound=event[0].params.boundecl)

  for j in np.arange(numevents):
    # Record best models for each model type:
    fit[j].allparams = allparams[fit[j].indparams]
    fit[j].bestp     = fit[0].fitparams[fit[j].indparams]
    fit[j].bestfit, fit[j].bestfituc = evalmodel(fit[0].fitparams, fit[j], 
                                                 getuc=True)

    # Save chain to file:
    fit[j].allparamsfile = (event[j].modeldir + "/d-" +
               event[j].eventname + '-allparams-'  + fit[j].saveext + '.npy')
    pid = open(fit[j].allparamsfile, 'wb')
    np.save(pid, allparams[:,fit[j].indparams])
    pid.close()

  del(allparams)

  for j in np.arange(numevents):
    # Get system flux:
    fit[j].systemflux = 1

    fluxes = ['flux',  'flux2', 'flux3', 'trflux', 'trflux2', 
              'trspf', 'trspf2']
    for i in np.arange(len(fluxes)):
      if fluxes[i] in dir(fit[j].i):
        index = eval('fit[j].i.%s'%fluxes[i])
        fit[j].systemflux *= fit[j].bestp[index]

    # Compute separate best eclispe, ramp, and intra-pixel models,
    # and best fit parameters:
    fit[j].bestecl      = np.ones(fit[j].nobj)
    fit[j].bestramp     = np.ones(fit[j].nobj)
    fit[j].bestsin      = np.ones(fit[j].nobj)
    fit[j].bestip       = np.ones(fit[j].nobj)
    fit[j].bestpos      = np.ones(fit[j].nobj)
    fit[j].bestvs       = np.ones(fit[j].nobj)
    fit[j].bestff       = np.ones(fit[j].nobj)
    fit[j].bestmip      = np.ones(fit[j].nobj)
    fit[j].bestpip      = []
    fit[j].bestfitmip   = []
    fit[j].bestecluc    = np.ones(fit[j].nobjuc)
    fit[j].bestrampuc   = np.ones(fit[j].nobjuc)
    fit[j].bestsinuc    = np.ones(fit[j].nobjuc)
    fit[j].bestipuc     = np.ones(fit[j].nobjuc)
    fit[j].bestposuc    = np.ones(fit[j].nobjuc)
    fit[j].bestvsuc     = np.ones(fit[j].nobjuc)
    fit[j].bestffuc     = np.ones(fit[j].nobjuc)
    fit[j].bestmipuc    = np.ones(fit[j].nobjuc)
    fit[j].bestfitmipuc = []

    etc   = np.ones(fit[j].nobj)
    etcuc = np.ones(fit[j].nobjuc)
    bestp = fit[0].fitparams
    for k in np.arange(fit[j].numm):
      if fit[j].functypes[k] not in ['ipmap']:
        model   = fit[j].funcs[k](bestp[fit[j].iparams[k]],
                                  fit[j].funcx[k],   fit[j].etc[k])
        modeluc = fit[j].funcs[k](bestp[fit[j].iparams[k]],
                                  fit[j].funcxuc[k], fit[j].etc[k])
        etc, etcuc = etc*model, etcuc*modeluc

      if   fit[j].functypes[k] == 'ecl/tr':
        fit[j].bestecl   *= model
        fit[j].bestecluc *= modeluc
      elif fit[j].functypes[k] == 'ramp':
        fit[j].bestramp   *= model 
        fit[j].bestrampuc *= modeluc
      elif fit[j].functypes[k] == 'sinusoidal':
        fit[j].bestsin   *= model
        fit[j].bestsinuc *= modeluc
      elif fit[j].functypes[k] == 'ippoly':
        fit[j].bestip    *= model
        fit[j].bestipuc  *= modeluc
        fit[j].bestpip    = bestp[fit[j].iparams[k]]
      elif fit[j].functypes[k] == 'posoffset':
        fit[j].bestpos   *= model
        fit[j].bestposuc *= modeluc
      elif fit[j].functypes[k] == 'vissen':
        fit[j].bestvs    *= model
        fit[j].bestvsuc  *= modeluc
        fit[j].bestpvs    = bestp[fit[j].iparams[k]]
      elif fit[j].functypes[k] == 'flatf':
        fit[j].bestff    *= model
        fit[j].bestffuc  *= modeluc
      elif fit[j].functypes[k] == 'ipmap':
        fit[j].bestfitmip   = etc
        fit[j].bestfitmipuc = etcuc

        fit[j].bestmip, fit[j].binipflux = \
                   fit[j].funcs[k](bestp[fit[j].iparams[k]], fit[j].funcx[k],
                                   fit[j].bestfitmip,   retbinflux=True)
        fit[j].bestmipuc, fit[j].binipfluxuc = \
                  fit[j].funcs[k](bestp[fit[j].iparams[k]], fit[j].funcxuc[k],
                                   fit[j].bestfitmipuc, retbinflux=True)
        fit[j].bestmipuc *= ( np.mean(fit[j].bestmip) /
                    np.mean(fit[j].bestmipuc[np.where(fit[j].clipmask)]) )
        fit[j].bestmipuc[np.where(fit[j].clipmask)] = fit[j].bestmip
        fit[j].binipflux   = fit[j].binipflux.reshape(  fit[j].gridshape)
        fit[j].binipfluxuc = fit[j].binipfluxuc.reshape(fit[j].gridshape)
        fit[j].bestpmip    = bestp[fit[j].iparams[k]]

  print("Calculating autocorrelation of free parameters.")
  maxstepsize = 1
  for j in range(numevents):
    acsize = np.int(np.amin(((event[0].params.numit[1]-
                              event[0].params.numit[0])*
                             event[0].params.nchains, 1e4)))
    fit[j].autocorr = np.zeros((len(fit[j].nonfixedpars), acsize))
    fit[j].ess      = np.zeros(len(fit[j].nonfixedpars))
    k = 0
    for i in fit[j].nonfixedpars:
      meanapi  = np.mean(fit[j].allparams[i])
      autocorr = np.correlate(fit[j].allparams[i,:acsize]-meanapi, 
                              fit[j].allparams[i,:acsize]-meanapi, mode='full')
      fit[j].autocorr[k] = (autocorr[np.size(autocorr)//2:] / np.max(autocorr))
      # Calculate effective sample size (ESS, see Kass et al.; 1998)
      # First instance where autocorr < 0.01:
      if np.min(fit[j].autocorr[k]) < 0.01:
        cutoff = np.where(fit[j].autocorr[k] < 0.01)[0][0]
      # Use the whole array if it never gets below the arbitrary
      # limit.
      else:
        cutoff = -1
        if event[0].params.mcmc: # Only meaningful if MCMC is run
          print("WARNING: autocorrelation never gets below 0.01. "  +
                 "Do not trust effective sample size for non-fixed " +
                 "parameter " + str(k))
      fit[j].ess[k] = numit / (1 + 2*np.sum(fit[j].autocorr[k,:cutoff]))
      k += 1
    miness = np.min(fit[j].ess)

    # If MCMC is not run, the fake MCMC data will break the ESS calcualation.
    # This is a workaround.
    if np.isnan(miness):
        print("Error with effective sample size in one or more " +
              "parameters. Ignore if MCMC was not run. Fixing " +
              "sample size to the entire sample.")
        miness = numit
        
    print(event[j].eventname, "effective sample size:", np.round(miness),
          file=printout)
    fit[j].thinning = np.round(numit/miness)
    # Set params.thinning if it was not defined before:
    if (not hasattr(event[j].params, 'thinning') or
        event[j].params.thinning == None):
      print(event[j].eventname, "New thinning:", fit[j].thinning, file=printout)
    maxstepsize = int(np.amax((maxstepsize, fit[j].thinning)))
  # Compute transformation matrix and plot orthogonal parameter correlations:
  for j in range(numevents):
    if event[j].params.newortho:
      print("Computing transformation matrix for %s." %event[j].eventname)
      steps = int(np.ceil(event[0].params.numit[0]/2000.))
      # Orthogonalization using Principal Component Analysis (PCA) class:
      orthodata = fit[j].allparams[fit[j].iortholist,::steps]
      fit[j].ortho = plt.mlab.PCA(orthodata.T)  # ortho object
      fit[j].origin = fit[j].ortho.mu           # means of orthodata
      # Compute inverse transformation matrix
      # Y = trans * (data-mu).T
      # (data-mu).T = invtrans * Y
      # Note: np.matrix type, not np.array
      fit[j].trans      = np.matrix(fit[j].ortho.Wt)
      fit[j].invtrans   = np.matrix(np.linalg.inv(fit[j].trans))
      fit[j].orthosigma = fit[j].ortho.sigma
      print(event[j].eventname + " Inverse transformation matrix:",
            file=printout)
      print(fit[j].invtrans, file=printout)

      # Update best-fit orthogonal parameters
      # fit[j].origin = np.copy(neworigin)
      fit[j].bestop[fit[j].iortholist] = \
          mc.orthoTrans(fit[j].bestp[fit[j].iortholist], fit[j].trans,
                        [fit[j].origin, fit[j].orthosigma])
      bestop[iortholist[j]] = mc.orthoTrans(bestp[iortholist[j]],
                        fit[j].trans, [fit[j].origin, fit[j].orthosigma])
      print(event[j].eventname + " best parameters after orthogonalization:",
            file=printout)
      print(fit[j].bestop, file=printout)

      # Correlation plot of original parameters:
      savefile = (event[j].modeldir + "/" + event[j].eventname + "-fig%d-" +
                  fit[j].saveext + ".png")

      fignum   = 6000 + num*fit[j].numfigs + j*100
      plots.hist2d(event[j], fit[j], fignum, savefile%fignum,
                   allparams=orthodata, iparams=fit[j].iortholist)

      # Correlation plot of transformed parameters:
      fignum   = 6010 + num*fit[j].numfigs + j*100
      plots.hist2d(event[j], fit[j], fignum, savefile%fignum,
                   allparams=fit[j].ortho.Y.T, parname=fit[j].opname,
                   iparams=fit[j].iortholist)

  # Calculate mean, median and standard deviations of output parameters
  medianp, meanp, std = [], [], []
  for j in np.arange(numevents):
    fit[j].medianp = np.median(fit[j].allparams[:, ::maxstepsize], axis=1)
    fit[j].meanp   = np.mean(  fit[j].allparams[:, ::maxstepsize], axis=1)
    fit[j].std     = np.std(   fit[j].allparams[:, ::maxstepsize], axis=1)
    medianp = np.concatenate((medianp, fit[j].medianp), 0)
    meanp   = np.concatenate((meanp,   fit[j].meanp),   0)
    std     = np.concatenate((std,     fit[j].std),     0)

  for j in np.arange(numevents):
    # Compute residuals and reduced chi-squared:
    fit[j].residuals   = fit[j].flux - fit[j].bestfit
    fit[j].redchisq    = (np.sum((fit[j].residuals / fit[j].sigma)**2) /
                             (fit[j].nobj - fit[j].numfreepars) )
    fit[j].oldredchisq = (np.sum((fit[j].residuals / fit[j].rawsigma)**2) /
                             (fit[j].nobj - fit[j].numfreepars) )
    print("Median parameters with standard deviations:", file=printout)
    print(np.vstack((fit[j].medianp,fit[j].std)).T,      file=printout)
    print("Best parameters:", file=printout)
    print(fit[j].bestp,       file=printout)
    print("Reduced chi-square: " + str(fit[j].redchisq), file=printout)

    # COMPUTE BAYESIAN INFORMATION CRITERION (BIC)
    # Residuals are nomalized by dividing by the variance of the
    # residuals from: the previous model fit from the same channel,
    # or from itself if this is the first model of this channel.
    # This results in the same normalizing factor for all models of
    # the same channel.
    try:
      fit[j].aic = (np.sum((fit[j].residuals/event[j].fit[0].sigma)**2) +
                    2*fit[j].numfreepars )
      fit[j].bic = (np.sum((fit[j].residuals/event[j].fit[0].sigma)**2) +
                    fit[j].numfreepars*np.log(fit[j].nobj) )
      print("AIC = " + str(fit[j].aic), file=printout)
      print("BIC = " + str(fit[j].bic), file=printout)

#      plotdata = open("plotpoint.txt", "a")
#      print(fit[j].bic, file=plotdata)
#      plotdata.close()
      
    except:
      fit[j].aic = 0
      fit[j].bic = 0
      print("Error computing AIC/BIC.")

    # Create model without the eclipse:
    fit[j].noecl = noeclipse(bestp, fit[j], fixip=fit[j].bestmip)
    fit[j].ramp, fit[j].rampuc = noeclipse(bestp, fit[j], fixip=fit[j].bestmip,
                                fixipuc=fit[j].bestmipuc, skip=['sinusoidal'])

    # Normalize data and best-fitting model:
    fit[j].normflux      = (fit[j].flux      / fit[j].ramp).flatten()
    fit[j].normsigma     = (fit[j].sigma     / fit[j].ramp).flatten()
    fit[j].normbestfit   = (fit[j].bestfit   / fit[j].ramp).flatten() 
    fit[j].normresiduals = (fit[j].residuals / fit[j].ramp).flatten()
    fit[j].normfluxuc    = (fit[j].fluxuc    / fit[j].rampuc).flatten()
    fit[j].normsigmauc   = (fit[j].sigmauc   / fit[j].rampuc).flatten()
    fit[j].normbestfituc = (fit[j].bestfituc / fit[j].rampuc).flatten() 
    
    binsize = fit[j].nobj//fit[j].nbins
    # Bin clipped data:
    fit[j].binbestfit, fit[j].binnoecl, fit[j].normbinbest, fit[j].normbinstd, \
       fit[j].normbinflux, fit[j].normbinresiduals = bd.bindata(fit[j].nbins,
       mean=[fit[j].bestfit, fit[j].noecl, fit[j].normbestfit], 
       std=[fit[j].normsigma], weighted=[fit[j].normflux, fit[j].normresiduals])

    # Binned preclip:
    if fit[j].preclip > 0:
      fit[j].preclipflux  = fit[j].normfluxuc   [:fit[j].preclip]
      fit[j].preclipsigma = fit[j].normsigmauc  [:fit[j].preclip]
      fit[j].preclipmodel = fit[j].normbestfituc[:fit[j].preclip]
      fit[j].preclipphase = fit[j].timeunituc   [:fit[j].preclip]
      fit[j].preclipresiduals = fit[j].preclipflux - fit[j].preclipmodel
      fit[j].precbinphase, fit[j].precbinstd, fit[j].precbinflux, \
        fit[j].precbinresiduals = bd.bindata(fit[j].nbins, 
        mean=[fit[j].preclipphase], std=[fit[j].preclipsigma], 
        weighted=[fit[j].preclipflux, fit[j].preclipresiduals], binsize=binsize)

    # Binned postclip:
    if fit[j].postclip < fit[j].nobjuc:
      fit[j].postclipflux  = fit[j].normfluxuc   [fit[j].postclip:]
      fit[j].postclipsigma = fit[j].normsigmauc  [fit[j].postclip:]
      fit[j].postclipmodel = fit[j].normbestfituc[fit[j].postclip:]
      fit[j].postclipphase = fit[j].timeunituc   [fit[j].postclip:]
      fit[j].postclipresiduals = fit[j].postclipflux - fit[j].postclipmodel
      fit[j].postbinphase, fit[j].postbinstd, fit[j].postbinflux, \
        fit[j].postbinresiduals = bd.bindata(fit[j].nbins, 
        mean=[fit[j].postclipphase], std=[fit[j].postclipsigma], 
        weighted=[fit[j].postclipflux, fit[j].postclipresiduals], 
        binsize=binsize)

    # Compute SDNR:
    fit[j].sdnr = np.std(fit[j].normresiduals)
    
    # Sort flux by x, y, and radial positions:
    yy  = np.sort(fit[j].position[0]) * 1.0
    xx  = np.sort(fit[j].position[1]) * 1.0
    yflux = (fit[j].flux / (fit[j].bestecl * fit[j].bestramp *
                            fit[j].bestsin * fit[j].bestpos  * fit[j].bestvs *
                            fit[j].bestff))[np.argsort(fit[j].position[0])]
    xflux = (fit[j].flux / (fit[j].bestecl * fit[j].bestramp *
                            fit[j].bestsin * fit[j].bestpos  * fit[j].bestvs *
                            fit[j].bestff))[np.argsort(fit[j].position[1])]
    ybestip = fit[j].bestip   [np.argsort(fit[j].position[0])] * \
              fit[j].bestmip  [np.argsort(fit[j].position[0])] * \
              fit[j].ballardip[np.argsort(fit[j].position[0])]
    xbestip = fit[j].bestip   [np.argsort(fit[j].position[1])] * \
              fit[j].bestmip  [np.argsort(fit[j].position[1])] * \
              fit[j].ballardip[np.argsort(fit[j].position[1])]

    # Sort flux by frmvis:
    fvsort = np.sort(fit[j].frmvis)
    vsflux = ( fit[j].flux / ( fit[j].bestecl * fit[j].bestramp *
                  fit[j].bestsin * fit[j].bestip * fit[j].bestpos *
                  fit[j].bestff  * fit[j].bestmip))[np.argsort(fit[j].frmvis)]

    # Bin data using weighted average:
    fit[j].binxx         = np.zeros(fit[j].nbins)
    fit[j].binyy         = np.zeros(fit[j].nbins)
    fit[j].binxflux      = np.zeros(fit[j].nbins)
    fit[j].binyflux      = np.zeros(fit[j].nbins)
    fit[j].binxflstd     = np.zeros(fit[j].nbins)
    fit[j].binyflstd     = np.zeros(fit[j].nbins)
    fit[j].binxbestip    = np.zeros(fit[j].nbins)
    fit[j].binybestip    = np.zeros(fit[j].nbins)
    fit[j].binxbipstd    = np.zeros(fit[j].nbins)
    fit[j].binybipstd    = np.zeros(fit[j].nbins)
    fit[j].binresstd     = np.zeros(fit[j].nbins)

    for i in range(fit[j].nbins):
      start = int(1.0* i   *fit[j].nobj/fit[j].nbins)
      end   = int(1.0*(i+1)*fit[j].nobj/fit[j].nbins)
      wherex = np.bitwise_and(xx >= xx[0] +  i   *(xx[-1]-xx[0])/fit[j].nbins,
                              xx <= xx[0] + (i+1)*(xx[-1]-xx[0])/fit[j].nbins)
      wherey = np.bitwise_and(yy >= yy[0] +  i   *(yy[-1]-yy[0])/fit[j].nbins,
                              yy <= yy[0] + (i+1)*(yy[-1]-yy[0])/fit[j].nbins)

      xxrange    =      xx[np.where(wherex)]
      xfluxrange =   xflux[np.where(wherex)]
      xbestiprng = xbestip[np.where(wherex)]
      yyrange    =      yy[np.where(wherey)]
      yfluxrange =   yflux[np.where(wherey)]
      ybestiprng = ybestip[np.where(wherey)]
      
      fit[j].binxx[i]      = np.mean(xxrange)
      fit[j].binyy[i]      = np.mean(yyrange)
      fit[j].binxflux[i]   = np.mean(xfluxrange)
      fit[j].binyflux[i]   = np.mean(yfluxrange)
      fit[j].binxflstd[i]  = np.std (xfluxrange) / np.sqrt(xfluxrange.size)
      fit[j].binyflstd[i]  = np.std (yfluxrange) / np.sqrt(yfluxrange.size)
      fit[j].binxbestip[i] = np.mean(xbestiprng)
      fit[j].binybestip[i] = np.mean(ybestiprng)
      fit[j].binxbipstd[i] = np.std (xbestiprng) / np.sqrt(xbestiprng.size)
      fit[j].binybipstd[i] = np.std (ybestiprng) / np.sqrt(ybestiprng.size)
      fit[j].binresstd[i]  = np.std (fit[j].residuals[start:end]) / \
                             np.sqrt(np.size(fit[j].residuals))

  # Calculate AIC/BIC value for joint model fits:
  if numevents > 1:
    aic  = 0
    bic  = 0
    nobj = 0
    for j in range(numevents):
      aic  += sum((fit[j].residuals/fit[j].sigma)**2)
      bic  += sum((fit[j].residuals/fit[j].sigma)**2)
      nobj += fit[j].nobj
    aic += fit[j].numfreepars * 2
    bic += fit[j].numfreepars*np.log(fit[j].nobj)
    print("AIC for joint model fit = " + str(aic), file=printout)
    print("BIC for joint model fit = " + str(bic), file=printout)

  print('Produce figures.')
  for j in range(numevents):
    # Plot binned data with best fit:
    savefile = event[j].modeldir + "/" + event[j].eventname + \
               "-fig%d-" + fit[j].saveext+".png"
    fignum   = 6001 + num*fit[j].numfigs + j*100
    plots.binlc(event[j], fit[j], fignum, savefile%fignum, j=j)

    # Plot normalized binned data and best fit:
    fignum   = 6002 + num*fit[j].numfigs + j*100
    if hasattr(event[j].params, 'interclip'):
      interclip = event[j].params.interclip
    else:
      interclip = None
    plots.normlc(event[j], fit[j], fignum, savefile%fignum, j=j,
                 interclip=interclip)

    if event[0].params.allplots:
      # allparams trace parameters values for all steps:
      fignum   = 6003 + num*fit[j].numfigs + j*100
      plots.trace(event[j], fit[j], fignum, savefile%fignum)

      # allparams autocorrelation plot:
      fignum   = 6004 + num*fit[j].numfigs + j*100
      plots.autocorr(event[j], fit[j], fignum, savefile%fignum)

      # allparams correlation plots with 2D histograms:
      fignum   = 6005 + num*fit[j].numfigs + j*100
      plots.hist2d(event[j], fit[j], fignum, savefile%fignum)

      # allparams 1D histograms:
      fignum   = 6006 + num*fit[j].numfigs + j*100
      plots.histograms(event[j], fit[j], fignum, savefile%fignum)

      # Plot projections of position sensitivity along x and y
      fignum   = 6007 + num*fit[j].numfigs + j*100
      plots.ipprojections(event[j], fit[j], fignum, savefile%fignum)

      # Plot Fourier transform of residuals
      fignum   = 6010 + num*fit[j].numfigs + j*100
      plots.fourier(event[j], fit[j], fignum, savefile%fignum)

      if event[j].isortho  and not event[j].params.newortho:
        # allorthop trace parameters values for all steps:
        fignum   = 6013 + num*fit[j].numfigs + j*100
        plots.trace(event[j], fit[j], fignum, savefile=savefile%fignum,
                    allparams=fit[j].allorthop, parname=fit[j].opname)

        # allorthop autocorrelation plot:
        fignum   = 6014 + num*fit[j].numfigs + j*100
        plots.autocorr(event[j], fit[j], fignum, savefile=savefile%fignum,
                       allparams=fit[j].allorthop, parname=fit[j].opname)
            
        # allorthop correlation plots with 2D histograms:
        fignum   = 6015 + num*fit[j].numfigs + j*100
        plots.hist2d(event[j], fit[j], fignum, savefile=savefile%fignum,
                     allparams=fit[j].allorthop, parname=fit[j].opname)
            
        #allorthop 1D HISTOGRAMS
        fignum   = 6016 + num*fit[j].numfigs + j*100
        plots.histograms(event[j], fit[j], fignum, savefile=savefile%fignum,
                         allparams=fit[j].allorthop, parname=fit[j].opname)
  
    if fit[j].isipmapping:
      # Minimum number of acceptable points in a bin
      # Plot Bliss Map
      fignum   = 6008 + num*fit[j].numfigs + j*100
      plots.blissmap(event[j], fit[j], fignum, savefile=savefile%fignum,
                     minnumpts=fit[j].minnumpts)
      
      # Plot pointing histogram
      fignum   = 6009 + num*fit[j].numfigs + j*100
      plots.pointingHist(event[j], fit[j], fignum, savefile=savefile%fignum,
                         minnumpts=fit[j].minnumpts)

    # Plot RMS vs. bin size:
    if hasattr(event[j].params, 'rmsbins'):
      maxbins = event[j].params.rmsbins
    else:
      maxbins = int(fit[j].normresiduals.size//2)
    fit[j].rms, fit[j].stderr, fit[j].binsz, fit[j].rmserr = \
              cn.computeRMS(fit[j].normresiduals, binstep=1, maxnbins=maxbins, isrmserr=1)

    # Compute standard error for noisy data, instead of denoised data
    if (hasattr(event[j].params, 'noisysdnr') and
        event[j].params.noisysdnr != None):
      fit[j].stderr = cn.computeStdErr(event[j].params.noisysdnr,
                                      fit[j].normresiduals.size, fit[j].binsz)
    fignum   = 6011 + num*fit[j].numfigs + j*100
    try:
      plots.rmsplot(event[j], fit[j], fignum, savefile=savefile%fignum)
    except:
      pass

  # # Plot visit sensitivity and model:
  # if 'vissen' in functype:  # FINDME
  #   numvsmodels = 0
  #   for i in functype:
  #     if i == 'vissen':
  #       numvsmodels += 1
    
  #   plt.figure(612 + num*fit[j].numfigs) # ??
  #   plt.clf()
  #   a = plt.suptitle(event[0].filename + ' Visit # vs. Sensitivity', size=16)
  #   k = 1
  #   for j in range(numevents):
  #     for i in range(cummodels[j],cummodels[j+1]):
  #       if functype[i] == 'vissen':
  #         a = plt.subplot(numvsmodels,1,k)
  #         a = plt.errorbar(fit[j].binfrmvis, fit[j].binvsflux,
  #                          fit[j].binvsflstd, fmt='go', label='Data')
  #         a = plt.plot(fit[j].frmvis, fit[j].bestvs, 'k.', label='Model')
  #         for model in fit[j].model:
  #           if (model == 'vsspline'):
  #             a = plt.plot(event[j].params.vsknots, fit[j].bestpvs, 'bs')
  #         a = plt.ylabel('Flux Sensitivity')
  #         a = plt.xlabel('Visit Number')
  #         a = plt.legend(loc='best')
  #         a.fontsize=8
  #   plt.savefig(event[j].modeldir + "/" + event[0].filename +"-fig"+str(num*fit[j].numfigs+1608)
  #               + "-" + saveext + ".png")

  print('Writing save files.')
  for j in range(numevents):
    # write fit[j].allparams to file, then delete:
    fit[j].allparamsfile = event[j].modeldir + "/d-" + event[j].eventname + \
                           '-allparams-' + fit[j].saveext + '.npy'
    pid = open(fit[j].allparamsfile, 'wb')
    np.save(pid, fit[j].allparams)
    pid.close()
    del fit[j].allparams, fit[j].allorthop

    if event[j].params.newortho:
      # Write ortho save file:
      fit[j].orthofile = "d-" + event[j].eventname + '-ortho-' + \
                         fit[j].saveext + '.npz'
      pid  = open(fit[j].orthofile, 'wb')
      np.savez(pid, invtrans=fit[j].invtrans, trans=fit[j].trans,
               origin=fit[j].origin, sigma=fit[j].orthosigma)
      pid.close()

  # Reassign modified fit objects to the event
  #for j in np.arange(numevents):
  #    event[j].fit[num] = fit[j]

  return


def get_minnumpts(params, fit, num, mode=1, printout=None, verbose=False):
  """
  Produce mask of minimum number of points per bin condition.

  Parameters:
  -----------
  params: events.params instance
  fit: fit instance
  num: Scalar
       Identification number for the set of lightcurve models
  mode: Scalar
        1 = Calculate minnumptsmask
        2 = Calculate: binfluxmask, numpts, binloc, wherebinflux, wbfipmask
  verbose: Boolean
           If True print results to printout

  Modification History:
  ---------------------
  2012-09-10  patricio  Written by Patricio Cubillos fom old p6model code.
  """

  num1 = num
  if len(params.ystep) != len(params.model):
    num1 = 0
  ystep, xstep = params.ystep[num1], params.xstep[num1] 
  yfactor  = 10.0
  xfactor  = 10.0
  ymin     = np.floor(yfactor*fit.y.min())/yfactor - ystep
  ymax     = np.ceil (yfactor*fit.y.max())/yfactor + ystep
  xmin     = np.floor(xfactor*fit.x.min())/xfactor - xstep
  xmax     = np.ceil (xfactor*fit.x.max())/xfactor + xstep
  # Number of bins in y,x dimensions
  ysize    = int((ymax-ymin)/ystep + 1)
  xsize    = int((xmax-xmin)/xstep + 1)
  ygrid, ystep  = np.linspace(ymin, ymax, ysize, retstep=True)
  xgrid, xstep  = np.linspace(xmin, xmax, xsize, retstep=True)
  if mode == 2:
    fit.ygrid, fit.xgrid = ygrid, xgrid
    fit.xygrid = np.meshgrid(xgrid, ygrid)
    fit.ystep, fit.xstep = ystep, xstep
    fit.gridshape = np.shape(fit.xygrid[0])
    fit.binfluxmask   = np.zeros((ysize, xsize), dtype=int)
    fit.binfluxmaskuc = np.zeros((ysize, xsize), dtype=int)
    fit.numpts        = np.zeros((ysize, xsize)) # Number of points per bin
    fit.numptsuc      = np.zeros((ysize, xsize))

  # Minimum number of acceptable points in a bin
  if params.minnumpts.__class__ == int:
      fit.minnumpts = params.minnumpts
  elif np.size(params.minnumpts) == len(params.model): # One for each fit
      fit.minnumpts = params.minnumpts[num]
  else:
      fit.minnumpts = params.minnumpts[0]

  if verbose:
    print('Step size in y = ' + str(ystep), file=printout)
    print('Step size in x = ' + str(xstep), file=printout)
    print('Ignoring bins with < ' + str(fit.minnumpts) + ' points.',
          file=printout)
    print('Computing bin for each position.')

  for m in np.arange(ysize):
    wbftemp   = np.where(np.abs(fit.position[0]-ygrid[m]) < (ystep/2.0))[0]
    wbftempuc = np.where(np.abs(fit.y          -ygrid[m]) < (ystep/2.0))[0]
    pos1   = fit.position[1, [wbftemp]]
    pos1uc = fit.x[wbftempuc] 
    for n in np.arange(xsize):
        wbf   = wbftemp  [np.where((np.abs(pos1 -xgrid[n]) < (xstep/2.))[0])]
        wbfuc = wbftempuc[np.where(np.abs(pos1uc-xgrid[n]) < (xstep/2.))    ]
        wbfipmask   = wbf  [np.where(fit.ipmask  [wbf  ] == 1)]
        wbfipmaskuc = wbfuc[np.where(fit.ipmaskuc[wbfuc] == 1)]
        if mode == 1 and len(wbfipmask) < fit.minnumpts:
          fit.minnumptsmask[wbf] = 0
        if mode == 2:
          if len(wbfipmask) >= fit.minnumpts:
            fit.binfluxmask  [m,n] = 1 # FINDME: move this out of for
            fit.numpts       [m,n] = len(wbfipmask)
            fit.binloc[0, wbf] = m*xsize + n
            fit.wherebinflux.append(wbf)
            fit.wbfipmask.  append(wbfipmask)
          else:
            fit.wherebinflux.append([])
            fit.wbfipmask.   append([])
          # Do the same for unclipped data:
          if len(wbfipmaskuc) > 0:
            fit.binfluxmaskuc[m,n] = 1
            fit.numptsuc     [m,n] = len(wbfipmaskuc)
            fit.binlocuc[0, wbfuc] = m*xsize + n
            fit.wherebinfluxuc.append(wbfuc)
            fit.wbfipmaskuc.   append(wbfipmaskuc)
          else:
            fit.wherebinfluxuc.append([])
            fit.wbfipmaskuc.   append([])

  return

def mc3evalmodel(params, fits=None,  getuc=False,     getipflux=False, 
                 getbinflux=False,   getbinstd=False, getipfluxuc=False, 
                 getbinfluxuc=False, skip=[], fixip=None, fixipuc=None):
    """
    Function to evaluate the light curve model, for use in MC3.
    Evaluates the model for all fits given, rather than a single
    fit object. 
    """

    models = []

    numevents = len(fits)

    # Parse the params between the fits
    counter = 0
    for j in np.arange(numevents):
        fits[j].jointiparams = []
        for k in np.arange(fits[j].numm):
            nmodelpars = len(fits[j].iparams[k])
            fits[j].jointiparams.append(range(counter, counter + nmodelpars))
            counter += nmodelpars
    
    for j in np.arange(numevents):
        ymodel = np.ones(fits[j].nobj)
        for k in np.arange(fits[j].numm):
            if fits[j].functypes[k] == 'ortho':
                print('Ortho funcs no longer supported.')
            elif fits[j].functypes[k] == 'ipmap':
                fits[j].etc[k] = ymodel
                ymodel *= fits[j].funcs[k](params[fits[j].jointiparams[k]],
                                           fits[j].funcx[k], fits[j].etc[k])
            elif fits[j].functypes[k] == 'posoffset':
                ymodel *= fits[j].funcs[k](params[fits[j].jointiparams[k]],
                                           fits[j].funcx[k], fits[j].etc[k])
            else:
                ymodel *= fits[j].funcs[k](params[fits[j].jointiparams[k]],
                                           fits[j].funcx[k], fits[j].etc[k])
        models.append(ymodel)
    
    return np.concatenate(models)

def evalmodel(params, fit,        getuc=False,     getipflux=False, 
              getbinflux=False,   getbinstd=False, getipfluxuc=False, 
              getbinfluxuc=False, skip=[], fixip=None, fixipuc=None):
  """
  Evaluate the light-curve model for a single event.

  Parameters:
  -----------
  params: 1D ndarray
          List of lightcurve parameters
  fit: A fit instance
  getuc: Boolean
         Evaluate and return lightcurve for unclipped data.
  getipflux:     Boolean
                 Return intrapixel model
  getbinflux:    Boolean
                 Return binned ipflux map.
  getbinstd:     Boolean
                 Return binned ipflux standard deviation.
  getipfluxuc:   Boolean
                 Return intrapixel model for unclipped data.
  getbinfluxcuc: Boolean
                 Return binned ipflux map for unclipped data.
  skip: List of strings
        List of names of models not to evaluate.
  fixip: Intrapixel model
         If provided use this model for intrapixel.
  fixipuc: Intrapixel model for unclipped data
           If provided use this model for unclipped intrapixel.

  Modification history:
  ---------------------
  2012-09-10  patricio  Written from old p6model module. Documented.
  """
  ipflux, binipflux, binstd = 1, 1, 1  # Dummy variables values

  # Evaluate model:
  fit0 = np.ones(fit.nobj) # fit[j].bestfit   
  for k in np.arange(fit.numm):
    if   fit.functypes[k] == 'ortho':
      pass
    elif fit.functypes[k] == 'ipmap':
      if fixip is not None:
        fit0 *= fixip
      else:
        ipflux, binipflux, binstd = fit.funcs[k](params[fit.iparams[k]],
                          fit.funcx[k], fit0, retbinflux=True, retbinstd=True)
        fit0 *= ipflux
    elif fit.functypes[k] == 'ballardip':
      fit.ballardip  = fit.funcs[k](params[fit.iparams[k]],
                                         fit.funcx[k], etc=[fit0])
      fit0 *= fit.ballardip
    else:
      if fit.functypes[k] not in skip:
        fit0 *= fit.funcs[k](params[fit.iparams[k]],
                             fit.funcx[k], fit.etc[k])

  # Repeat for unclipped data:
  if getuc or getipfluxuc or getbinfluxuc:
    ipfluxuc, binipfluxuc = 1, 1  # Dummy variables values

    fituc0 = np.ones(fit.nobjuc)
    for k in np.arange(fit.numm):
      if   fit.functypes[k] == 'ortho':
        pass
      elif fit.functypes[k] == 'ipmap':
        if fixipuc is not None:
          fituc0 *= fixipuc
        else:
          ipfluxuc, binipfluxuc = fit.funcs[k](params[fit.iparams[k]],
                                     fit.funcxuc[k], fituc0, retbinflux=True)
          ipfluxuc[np.where(fit.clipmask)] = ipflux
          fituc0 *= ipfluxuc
      elif fit.functypes[k] == 'ballardip':
        fit.ballardipuc    = fit.funcs[k](params[fit.iparams[k]],
                                                fit.funcxuc[k], etc=[fituc0])
        fit.ballardipuc[np.where(fit.clipmask)] = fit.ballardip
        fituc0 *= fit.ballardipuc
      else:
        if fit.functypes[k] not in skip:
          fituc0 *= fit.funcs[k](params[fit.iparams[k]],
                                 fit.funcxuc[k], fit.etc[k])

  # Return statement:
  if ( not getipflux  and not getuc        and not getbinflux  and
       not getbinstd  and not getipfluxuc  and not getbinfluxuc  ):
    # Return the fit alone:
    return fit0

  # Else, return a list with the requested values:
  ret = [fit0]
  if getipflux:
    ret.append(ipflux)  
  if getuc:
    ret.append(fituc0)
  if getbinflux:
    ret.append(binipflux)
  if getbinstd:
    ret.append(binstd)
  if getipfluxuc:
    ret.append(ipfluxuc)  
  if getbinfluxuc:
    ret.append(binipfluxuc)
  return ret


def noeclipse(params, fit, fixip=None, fixipuc=None, skip=[]):
  noepars = np.copy(params) # no-eclipse parameters 

  # Create model without the eclipse:
  depths = ['depth', 'depth2', 'depth3', 'gdepth', 'trrprs', 'trrprs2', 'rprs', 'rprs2', 'rp_rs', 'trdepth']
  for i in np.arange(len(depths)):
    if depths[i] in dir(fit.i):
      noepars[eval('fit.i.%s + fit.indparams[0]'%depths[i])] = 0.0

  if fixipuc is not None:
    getuc = True
  else:
    getuc = False
  noecl = evalmodel(noepars, fit, getuc=getuc, 
                    skip=skip, fixip=fixip, fixipuc=fixipuc)
  return noecl


def residuals(freepars, params, inonprior, fit, nopriors=False):
  """
  Calculate the residual between the lightcurves and models.

  Parameters:
  -----------
  freepars: 1D ndarray
            Array of fitting light-curve model parameters.
  params: 1D ndarray 
            Array of light-curve model parameters.
  inonprior: 1D ndarray
            Array with the indices of freepars.
  fit: List of fits instances of an event.
  nopriors: Boolean
            Do not add priors penalty to chi square in minimization.

  Modification history:
  ---------------------
  2012-09-10  patricio  Written from old p6model module. Documented.
  2013-02-26  patricio  Added priors penalization and nopriors option.
  2017-02-07  rchallen  Updated for current python indexing rules.
                        rchallen@knights.ucf.edu
  """

  # The output:
  residuals = []        # Residuals from the fit
  prior_residuals = []  # Residuals from the prior penalization

  # Number of events:
  numevents = len(fit)

  # Update the fitting parameters:
  pmin, pmax, stepsize  = [], [], []
  for j in np.arange(numevents): 
    pmin     = np.concatenate((pmin,     fit[j].pmin),     0)
    pmax     = np.concatenate((pmax,     fit[j].pmax),     0)
    stepsize = np.concatenate((stepsize, fit[j].stepsize), 0)

  # Fitting parameters:
  params[inonprior] = freepars

  # Check min and max boundaries:
  params[np.where(params < pmin)] = pmin[np.where(params < pmin)]
  params[np.where(params > pmax)] = pmax[np.where(params > pmax)]

  # Update shared parameters:
  for i in np.arange(len(stepsize)):
    if stepsize[i] < 0:
      params[i] = params[int(-stepsize[i]-1)]

  # Evaluate model for each event:
  for j in np.arange(numevents):
    model = evalmodel(params, fit[j])

    # Calculate event's residuals and concatenate:
    ev_res = (model - fit[j].flux)/fit[j].sigma
    residuals = np.concatenate((residuals, ev_res))

    # Apply priors penalty if exists:
    if len(fit[j].ipriors) > 0:
      pbar = fit[j].priorvals[:,0]
      psigma = np.zeros(len(pbar))
      for i in np.arange(len(fit[j].ipriors)):
        if params[fit[j].ipriors[i]] < pbar[i]:
          psigma[i] = fit[j].priorvals[i,1]
        else:
          psigma[i] = fit[j].priorvals[i,2]
        #priorchisq += ((params[fit[j].ipriors[i]]-pbar[i])/psigma[i])**2.0
        prior_residuals.append((params[fit[j].ipriors[i]]-pbar[i])/psigma[i])

  # chisq = np.sum(residuals**2) + priorchisq
  # pseudoresiduals = np.sqrt(chisq/len(residuals)*np.ones(len(residuals)) )
  if nopriors:
    return residuals
  return np.concatenate((residuals, prior_residuals))


def modelfit(params, inonprior, stepsize, fit, pmin, pmax,
             verbose=False, full=False,
             retchisq=False, nopriors=False):
  """
  Least-squares fitting wrapper.

  Parameters:
  -----------
  params: 1D ndarray 
          Array of light-curve model parameters.
  inonprior: 1D ndarray
             Array with the indices of freepars.
  stepsize: 1D ndarray
            Array of fitting light-curve model parameters.
  fit: List of fits instances of an event.
  verbose:  Boolean
            If True print least-square fitting message.
  full:     Boolean
            If True return full output in scipy's leastsq and print message.
  retchisq: Boolean
            Return the best-fitting chi-square value.
  nopriors: Boolean
            Do not add priors penalty to chi square in minimization.

  Modification history:
  ---------------------
  2012-09-10  patricio  Written from old p6model module. Documented.
  2013-02-26  patricio  Added priors penalization and nopriors option.
  """

  fitting = op.least_squares(residuals, params[inonprior],
                             args=(params, inonprior, fit, nopriors),
                             bounds=(pmin[inonprior], pmax[inonprior]),
                             ftol=3e-16, xtol=3e-16, gtol=3e-16)
                             
  #fitting = op.leastsq(residuals, params[inonprior],
  #      args=(params, inonprior, fit, nopriors), factor=100, ftol=1e-16,
  #      xtol=1e-16, gtol=1e-16, diag=1./stepsize[inonprior], full_output=full)

  # Unpack least-squares fitting results:
  if full:
    output, mesg, err = fitting.x, fitting.message, fitting.status 
    print(mesg)
  else:
    output, mesg, err = fitting.x, fitting.message, fitting.status

  # Print least-squares flag message:
  if verbose:
    if (err >= 1) and (err <= 4):
      print("Fit converged without error.")
    else:
      print("WARNING: Error with least squares fit!")

  # Print out results:
  if full or verbose:
    print("Least squares fit best parameters:")
    print(output)

  # Count how many priors:
  npriors = 0
  for j in np.arange(len(fit)):
    npriors += len(fit[j].ipriors)

  # Return chi-square of the fit:
  if retchisq:
    fit_residuals = residuals(params[inonprior], params, inonprior, fit)
    chisq = np.sum(fit_residuals)**2.0
    return output, chisq

  return output


def runmcmc(event, fit, numit, walk, mode, grtest, printout, bound):
  """
  MCMC simulation wrapper.

  Code content:
  -------------
  - Set up parameters, limits, etc.
  - Orthogonalize if requested
  - Run MCMC
  - De-orthogonalize
  - Check for better fitting parameters.
  - Re-run least-squares if necessary.
  - Calculate and print acceptance rate.

  Parameters:
  -----------
  event: List of event instances
  fit: List of fit instances of an event.
    numit: Scalar
           Number of MCMC iterations.
  walk: String
        Random walk for the Markov chain: 
        'demc': differential evolution MC.
        'mrw': Metropolis random walk.
  mode: String
           MCMC mode: ['burn' | 'continue' | 'final']
  grtest: Boolean
          Do Gelman and Rubin convergence test. 
  printout: File object for directing print statements
  bound: Boolean
         Use bounded-eclipse constrain for parameters (start after the 
         first frame, end before the last frame).

  Modification history:
  ---------------------
  2012-09-10  patricio  Written from old p6model module. Documented.
  """

  numevents = len(fit)

  ninitial = 1   # Number of initial params. sets to start MCMC

  fitpars = np.zeros((ninitial, 0))
  pmin, pmax, stepsize, parscale, isprior, iortho = [], [], [], [], [], []
  ntotal = 0 # total number of parameters:
  for j in np.arange(numevents):
    fitpars   = np.hstack((fitpars, fit[j].chainend))
    pmin      = np.concatenate((pmin,     fit[j].pmin),      0)
    pmax      = np.concatenate((pmax,     fit[j].pmax),      0)
    stepsize  = np.concatenate((stepsize, fit[j].stepsize),  0)
#    parscale  = np.concatenate((parscale, fit[j].parscale),  0)
    isprior   = np.concatenate((isprior,  fit[j].isprior),   0)
    iortho.append(fit[j].iortholist2)
    ntotal += fit[j].nump

    # Open process ID to save allknots:
    fit[j].allknotfile = (event[j].modeldir + "/d-" + event[j].eventname
                          + '-allknots-' + fit[j].saveext + '.npy')
    fit[j].allknotpid = open(fit[j].allknotfile, 'wb')

  inonprior = np.where(stepsize > 0)[0]

  # Fix IP mapping to bestmip if requested (for final run):
  if mode == 'final':
    for j in np.arange(numevents):
      if hasattr(event[j].params, 'isfixipmap') and event[j].params.isfixipmap:
        for k in np.arange(fit[j].numm):
          if fit[j].functypes[k] == 'ipmap':
            foo = fit[j].funcx.pop(k), fit[j].funcxuc.pop(k)
            fit[j].funcx.append(  [fit[j].bestmip,   fit[j].binipflux,
                                   np.zeros(len(fit[j].wbfipmask))])
            fit[j].funcxuc.append([fit[j].bestmipuc, fit[j].binipfluxuc,
                                   np.zeros(len(fit[j].wbfipmaskuc))])
            fit[j].funcs[k] = mc.fixipmapping

  # Transform parameters in ortholist to orthogonal parameters
  fitop = np.copy(fitpars)
  for j in np.arange(numevents):
    if event[j].isortho and not event[j].params.newortho:
      for c in np.arange(ninitial):
        fitop[c,iortho[j]] = mc.orthoTrans(fitpars[c,iortho[j]],
                            fit[j].trans, [fit[j].origin, fit[j].orthosigma])

  # Run MCMC:
  #print(time.ctime() + ' : ' + mode + ' MCMC simulation', file=printout)
  #print('Number of iterations: %d'%numit, file=printout)
  #allorthop, numaccept, bestop, bestchisq = mcmc.mcmc(fitop, pmin, pmax,
  #                            stepsize, parscale, numit, iortho, fit,
  #                            event[0].params.nchains, walk=walk, 
  #                            grtest=grtest, bound=bound)

  # Independent params for mc3 function 
  indparams = [fit]

  # Set up a 1D data and uncert arrays for mc3
  data   = []
  uncert = []
  for i in range(numevents):
    data.append(  fit[i].flux)
    uncert.append(fit[i].sigma)
  data   = np.concatenate(data)
  uncert = np.concatenate(uncert)

  # Set up mc3 priors
  prior, priorlow, priorup = [], [], []

  for j in range(numevents):
    nump = 0
    fitprior    = np.zeros(len(fit[j].pmin))
    fitpriorlow = np.zeros(len(fit[j].pmin))
    fitpriorup  = np.zeros(len(fit[j].pmin))
    if hasattr(event[j].params, "priorvars"):
        for pvar in event[j].params.priorvars:
            if hasattr(fit[j].i, pvar):
                fitprior   [getattr(fit[j].i, pvar)] = fit[j].priorvals[nump][0]
                fitpriorlow[getattr(fit[j].i, pvar)] = fit[j].priorvals[nump][1]
                fitpriorup [getattr(fit[j].i, pvar)] = fit[j].priorvals[nump][2]
            else:
                print("Prior variable " + pvar + " not recognized.")
            nump += 1
    prior    = np.concatenate((prior,    fitprior),    0)
    priorlow = np.concatenate((priorlow, fitpriorlow), 0)
    priorup  = np.concatenate((priorup,  fitpriorup),  0)

  # Make this a user setting at some point
  grbreak = 0

  nsamples = event[0].params.nchains * numit
  nchains  = event[0].params.nchains
  nproc    = event[0].params.nchains
  burnin   = event[0].params.numit[0]
      
  # Run MCMC:
  if event[0].params.mcmc:
      mc3out = mc3.mcmc(data=data, uncert=uncert, func=mc3evalmodel,
                        indparams=indparams, params=fitop[0],
                        pmin=pmin, pmax=pmax, stepsize=stepsize,
                        prior=prior, priorlow=priorlow,
                        priorup=priorup, walk=walk, nsamples=nsamples,
                        nchains=nchains, nproc=nchains, burnin=burnin,
                        leastsq=event[0].params.leastsq,
                        chisqscale=event[0].params.chi2flag,
                        grtest=grtest, grbreak=grbreak,
                        full_output=False, chireturn=True)

      bestop    = mc3out[0]
      CRlo      = mc3out[1]
      CRhi      = mc3out[2]
      stdp      = mc3out[3]
      posterior = mc3out[4]
      Zchain    = mc3out[5]
      bestchisq, redchisq, chifactor, bic = mc3out[6]

      #numaccept = np.sum(numaccept)
      print(time.ctime() + ' : End mcmc simulation', file=printout)
  else: # Make up some values to prevent crashes
      print("WARNING: MCMC will not be run. MCMC best fit set to " +
            "least-squares fit. MCMC array filled with zeros.")
      nfree = stepsize[np.where(stepsize>0.0)].size
      bestop    = fitop[0]
      CRlo      = fitop[0] - 0.1 * fitop[0]
      CRhi      = fitop[0] + 0.1 * fitop[0]
      stdp      = fitop[0] * 0.1
      posterior = np.zeros((nsamples - burnin * nchains, nfree))
      
      Zchain    = np.ones(nsamples)
      # Set chisq high so code doesn't think the MCMC found a good fite
      bestchisq, redchisq, chifactor, bic = 1e300, 1e300, 1e300, 1e300

  # Convert mc3 output to format of old MCMC output so
  # that it will function with the rest of the code
  # This requires we add rows for each of the fixed parameters
  # and reshape into a 3D array of (nchains, nparams, niter)
  # where niter is the number of iterations per chain
  totaliter = posterior.shape[0]
  allorthop = np.zeros((ntotal, totaliter))
  niter = totaliter//nchains
  counter = 0
  for i in range(ntotal):
    if stepsize[i] == 0.0:
      allorthop[i] = fitop[0][i]
    elif stepsize[i] < 0.0:
      allorthop[i] = allorthop[int(-1*stepsize[i]-1)]
    else:
      allorthop[i] = posterior[:,counter]
      counter += 1

  allp = np.zeros((nchains, ntotal, niter))

  for i in range(nchains):
    for j in range(ntotal):
      allp[i,j] = allorthop[j,niter*i:niter*(i+1)]

  allorthop = np.copy(allp)
      
  # Transform parameters in ortholist to original parameters:
  bestp     = np.copy(bestop)
  allparams = np.copy(allorthop)
  for j in range(numevents):
    # Record trace of decorrelated parameters, if any, otherwise
    # same as fit[j].allparams
    if mode == 'final': # store values
      fit[j].allorthop  = np.copy(allorthop[:,fit[j].indparams])
      fit[j].bestop     = np.copy(bestp    [  fit[j].indparams])

    if event[j].isortho and not event[j].params.newortho:
      for c in np.arange(event[0].params.nchains):
        bestp    [  iortho[j]] = mc.orthoInvTrans(bestop   [iortho[j]],
                          fit[j].invtrans, [fit[j].origin, fit[j].orthosigma])
        allparams[c,iortho[j]] = mc.orthoInvTrans(allorthop[c,iortho[j]],
                          fit[j].invtrans, [fit[j].origin, fit[j].orthosigma])
      del(allorthop)

  # Reshape allparams joining chains for final results:
  if mode == 'final':
    allp = np.zeros((ntotal, totaliter))
    for par in np.arange(ntotal):
      allp[par] = allparams[:,par,:].flatten()
    del(allparams)
    allparams = allp
    
  # Checks best parameters against minimizer output:
  if bestchisq < fit[0].chisq:
    print("MCMC found a better fit.",   file=printout)
    print("Current best parameters:",   file=printout)
    print(fit[0].fitparams[inonprior],  file=printout)
    # Re-run minimizer:
    if event[0].params.leastsq:
      output = modelfit(bestp, inonprior, stepsize, fit, pmin, pmax,
                        verbose=True)
      print("New minimizer values:",     file=printout)
      print(bestp, file=printout)
      #print(fit[0].fitparams[inonprior], file=printout)

    # Re-calculate chi-square:
    fit[0].chisq = 0.0
    for j in np.arange(numevents):
      model = evalmodel(bestp, fit[j])
      fit[0].chisq += np.sum(((model - fit[j].flux)/fit[j].sigma)**2.0)
      fit[j].bic = (np.sum(((model - fit[j].flux)/fit[j].sigma)**2.0) +
                    fit[j].numfreepars*np.log(fit[j].nobj))
    # Update best-fitting parameters in fit[0]:
    fit[0].fitparams = bestp

    # Update best-fit parameter list with best fitting parameters:
    parlist = []
    for   j in np.arange(numevents):
      parlist.append(pe.read(fit[j].modelfile, fit[j].model, event[j]))
      for k in np.arange(fit[j].numm):
        if fit[j].model[k] != 'ipspline': # what?
          parlist[j][k][2][0] = bestp[fit[j].iparams[k]]
      # Update parameters:
      pe.write(fit[j].modelfile, parlist[j])

#    ier = os.system("cp %s %s/." % (fit[j].modelfile, event[j].modeldir))

  # Calculate best fit, SDNR, BIC, bsigchi, etc.
  for j in np.arange(numevents):
    # Evaulate model with best parameters
    fit[j].bestfit,     fit[j].bestmip,   fit[j].bestfituc,    \
      fit[j].binipflux, fit[j].bestmipuc, fit[j].binipfluxuc = \
            evalmodel(fit[0].fitparams, fit[j], getuc=True, getipflux=True,
                        getbinflux=True, getipfluxuc=True, getbinfluxuc=True)

    # Calculate residuals
    fit[j].residuals = fit[j].flux - fit[j].bestfit
    ramp = noeclipse(fit[0].fitparams, fit[j], fixip=fit[j].bestmip, 
                     skip=['sinusoidal'])
    fit[j].normresiduals = (fit[j].residuals / ramp).flatten()
    # SDNR for best fit
    fit[j].sdnr    = np.std(fit[j].normresiduals)
    # Compute BIC  for best-fitting model:
    fit[j].bic = (np.sum((fit[j].residuals/event[j].fit[0].sigma)**2) +
                  fit[j].numfreepars*np.log(fit[j].nobj) )
    # Calculate binned-sigma chi-squared (Deming et al., 2015)
    fit[j].bsigchi = cn.bsigchi(fit[j])

    # Bin results:
    fit[j].binfit0, fit[j].binbestfit = bd.bindata(fit[j].nbins, 
                                        mean=[fit[j].fit0, fit[j].bestfit])   

    fit[j].allknotpid.close()
    del(fit[j].allknotpid)

    # Save chains end point:
    if(mode != "final"):
        fit[j].chainend = allparams[:, fit[j].indparams, -1]
    else:
        fit[j].chainend = allparams[fit[j].indparams, -1]

  return allparams
