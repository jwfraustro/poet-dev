#! /usr/bin/env python2

# Implementation of PLD algorithm
# Team members:
#   Ryan Challener (rchallen@knights.ucf.edu)
#   Em DeLarme
#   Andrew Foster
#
# History:
#   2015-11-02 em       Initial implementation
#   2015-11-20 rchallen Updates for use with POET
#   2016-10-07 rchallen Plotting functions
#   2017-02-20 rchallen Lots of bug fixes. Made git repo.
# See git repo for full history.

# from __future__ imports must occur at the beginning of the file
from __future__ import print_function

# standard libraries
import configparser
import matplotlib.pyplot as plt
import multiprocessing   as mp
import numpy             as np
import os
import scipy.optimize    as sco
import shutil
import sys
import time
import gc
from sklearn    import linear_model

# custom libraries
sys.path.insert(0, "./lib/mccubed/")
sys.path.insert(1, "./lib")
import run
import copy
import irsa
import kurucz_inten
import logger
import manageevent  as me
import MCcubed      as mc3
import reader3      as rd
import readeventhdf
import zen_funcs    as zf
import zenplots     as zp
import models_c     as mc
import paramedit    as pe

def main(rundir, cfile=None, cfilename=None):
    '''
    One function to rule them all.
    '''
    # Set up logging of all print statements in this main file
    logfile = 'zen.log'
    templogfile = rundir + '/' + logfile + '.tmp'
    log = logger.Logger(templogfile)
    print("Start: %s" % time.ctime(), file=log)
    
    configobjs = []

    # eventlist is a list of events for each model set
    # eventlistlist is a list of eventlists. For example,
    # if you have 3 model sets and 2 data sets, eventlist
    # will be the events with optimized photometry for
    # each data set and model (so length 3) and eventlistlist contains
    # all the necessary events for joint fitting this scenario
    # (currently unsupported, but future update may change that)
    eventlistlist = []
    # Read the config file into a dictionary
    print("Reading the config file(s).")

    # If no obj is given, read them all
    if type(cfile) == type(None):
        confignames = []
        for fname in os.listdir(rundir):
            if (fname.endswith("-zen.cfg")):
                confignames.append(fname)
        confignames.sort()
        for fname in confignames:
            config = configparser.ConfigParser()
            config.read(rundir + '/' + fname)
            configobjs.append(config)
    # If a filename is provided
    elif isinstance(cfile, str):
        confignames = [cfile]
        config = configparser.ConfigParser()
        config.read(rundir + '/' + cfile)
        configobjs.append(config)
    # Otherwise, use just the object received    
    else:
        configobjs.append(cfile)
        confignames = [cfilename]

    nevents = len(configobjs)

    for m in range(nevents):
        eventlist = []
        fit = []
        nmodelsets = len(configobjs[m]['EVENT']['models'].split('\n'))
        if nmodelsets > 1 and nevents > 1:
            print("WARNING: multiple model sets not supported with" +
                  "joint fits. Please choose a single model for each" +
                  "event.")
            sys.exit()
            
        for n in range(nmodelsets):            
            # Initialize fit object (we don't yet know which event object
            # to attach it to)
            fit.append(readeventhdf.fits())

            # Fill in fit options
            zf.fitopt(fit[n], configobjs[m], rundir, n)

        if nevents > 1 and len(fit[n].bintry) > 1:
            print("WARNING: bin size optimization not supported with" +
                  " joint fits due to issues with shared parameters between" +
                  " data sets. Please set bintry to a single value for" +
                  " each data set (can be different for each).")
            sys.exit()

        for n in range(nmodelsets):
            # Get initial parameters and stepsize arrays from the config
            fit[n].modelfile = rundir + '/' + fit[n].modelfile

            nmodels = len(fit[n].modelstrs)

            parlist = pe.read(fit[n].modelfile,
                              fit[n].modelstrs,
                              None,
                              npldpars=fit[n].npix)
            
            fit[n].params   = []
            fit[n].pmin     = []
            fit[n].pmax     = []
            fit[n].npars    = []
            fit[n].stepsize = []
            for i in np.arange(nmodels):
                pars = parlist[i][2]
                fit[n].params    = np.concatenate((fit[n].params,     pars[0]),  0)
                fit[n].pmin      = np.concatenate((fit[n].pmin,       pars[1]),  0)
                fit[n].pmax      = np.concatenate((fit[n].pmax,       pars[2]),  0)
                fit[n].stepsize  = np.concatenate((fit[n].stepsize,   pars[3]),  0)
                fit[n].npars     = np.concatenate((fit[n].npars, [len(pars[0])]),0)

            # Currently there's a bug in numpy that converts concatenated
            # lists of ints to floats if one list is empty. This is a
            # workaround
            fit[n].npars = [int(p) for p in fit[n].npars]

            fit[n].modelfuncs, fit[n].modeltypes, fit[n].parnames, fit[n].i, fit[n].saveext = \
                        mc.setupmodel(fit[n].modelstrs, fit[n].i, fit[n].npix)

            # Parse priors
            nump = 0
            fit[n].prior    = np.zeros(len(fit[n].parnames))
            fit[n].priorlow = np.zeros(len(fit[n].parnames))
            fit[n].priorup  = np.zeros(len(fit[n].parnames))
            if hasattr(fit[n], "priorvars"):
                if len(fit[n].priorvals) % 3 != 0:
                    print("WARNING: priorvals not specified correctly.")
                for pvar in fit[n].priorvars:
                    if hasattr(fit[n].i, pvar):
                        fit[n].prior   [getattr(fit[n].i, pvar)] = fit[n].priorvals[3*nump]
                        fit[n].priorlow[getattr(fit[n].i, pvar)] = fit[n].priorvals[3*nump+1]
                        fit[n].priorup [getattr(fit[n].i, pvar)] = fit[n].priorvals[3*nump+2]
                    else:
                        print("Prior variable " + pvar + " not recognized.")
                    nump += 1

            fit[n].numm = len(fit[n].modelfuncs)

            nbin  = len(fit[n].bintry)
            ncent = len(fit[n].centdir)
            nphot = len(fit[n].photdir)

            # Set up multiprocessing
            jobs = []
            # Multiprocessing requires 1D arrays (if we use shared memory)
            chisqarray = mp.Array('d', np.zeros(nbin  *
                                                nphot *
                                                ncent))
            chislope   = mp.Array('d', np.zeros(nbin  *
                                                nphot *
                                                ncent))

            # Load the data (images are the same regardless of cent and
            # phot, so we can load prior to the loop)
            event_data = me.loadevent(rundir + '/' + fit[n].eventname + "_ini",
                                      load=['data'])
            data = event_data.data

            # Giant loop over all specified apertures and centering methods
            for l in range(nphot):
                for k in range(ncent):            
                    # Load the POET event object (up through p5)
                    print("Loading the POET event object.", file=log)
                    print("Ap:   " + fit[n].photdir[l], file=log)
                    print("Cent: " + fit[n].centdir[k], file=log)
                    centloc = '/'.join([rundir, fit[n].centdir[k], '']) 
                    photloc = '/'.join([rundir, fit[n].centdir[k],
                                        fit[n].photdir[l], ''])
                    if os.path.isdir(photloc):
                        event = me.loadevent(photloc + fit[n].eventname + "_p5c")
                    else:
                        print("Unable to find "
                              + fit[n].centdir[k] + '/'
                              + fit[n].photdir[l] +
                              ". Skipping.", file=log)
                        fill = np.ones(nbin) * np.inf
                        chisqarray[     nbin*l+nbin*nphot*k:
                                   nbin+nbin*l+nbin*nphot*k] = fill
                        chislope  [     nbin*l+nbin*nphot*k:
                                   nbin+nbin*l+nbin*nphot*k] = fill
                        continue

                    phase = event.phase

                    # Create masks
                    preclipmask  = phase > fit[n].preclip
                    postclipmask = phase < fit[n].postclip
                    fit[n].clipmask = np.logical_and(preclipmask, postclipmask)

                    for i in range(fit[n].ninterclip):
                        interclipmask = np.logical_or(phase < fit[n].interclip[2*i  ],
                                                      phase > fit[n].interclip[2*i+1])
                        fit[n].clipmask = np.logical_and(fit[n].clipmask, interclipmask)

                    fit[n].mask = np.logical_and(   fit[n].clipmask, event.good)


                    npos = data.shape[0]
                    if npos > 1:
                        mflux = np.mean(event.fp.aplev[np.where(event.good)])
                        posmflux = np.zeros((event.good.shape[0],1))
                        for i in range(event.good.shape[0]):
                            posgood = np.where(event.good[i])
                            posmflux[i] = np.mean(event.fp.aplev[i, posgood])
                        event.fp.aplev = event.fp.aplev / posmflux * mflux
                    
                    phasegood = event.phase[fit[n].mask]

                    phot    = event.fp.aplev[fit[n].mask]
                    photerr = event.fp.aperr[fit[n].mask]

                    normfactor = np.average(phot)

                    phot    /= normfactor
                    photerr /= normfactor

                    # Make sure phase is ascending
                    ind = np.argsort(phasegood)
                    phasegood = phasegood[ind]
                    phot      = phot[ind]
                    photerr   = photerr[ind]

                    # Identify the bright pixels to use
                    print("Identifying brightest pixels.", file=log)
                    boxsize = 10

                    xavg, yavg, rows, cols, pixels = zf.pldpixcoords(event,
                                                                     data,
                                                                     fit[n].npix,
                                                                     boxsize,
                                                                     fit[n].mask)

                    print("Doing preparatory calculations.", file=log)
                    phat, dP = zf.zen_init(data, pixels)

                    phatgood = np.zeros(len(fit[n].mask))

                    # Mask out the bad images in phat
                    for i in range(fit[n].npix):
                        tempphat = phat[:,i].copy()
                        tempphatgood = tempphat[fit[n].mask[0]]
                        if i == 0:
                            phatgood = tempphatgood.copy()
                        else:
                            phatgood = np.vstack((phatgood, tempphatgood))

                    # Invert the new array because I lack foresight
                    phatgood  = phatgood.T

                    # Check if maximum binning will work
                    nfreep = np.sum(np.array(fit[n].stepsize) > 0)
                    if len(phot) // np.max(fit[n].bintry) <= nfreep:
                        warnstr = ("Warning! Maximum bin size too large! " +
                                   "Reduce below {} and rerun.")
                        print(warnstr.format(len(phot)//(nfreep+1)),
                              file=log)
                        return

                    # If doing a joint fit, we need to avoid bin size
                    # optimization, because shared parameters across
                    # data sets will cause unintended behavior
                    # For a joint fit to get to this point, it should have
                    # nbin=1, nphot=1, ncent=1. In which case, we can
                    # just set each 1-element array to a single
                    # value and the code will behave correctly
                    if nevents > 1:
                        chisqarray[     nbin*l+nbin*nphot*k:
                                   nbin+nbin*l+nbin*nphot*k] = np.ones(nbin)
                        chislope  [     nbin*l+nbin*nphot*k:
                                   nbin+nbin*l+nbin*nphot*k] = np.ones(nbin) * fit[n].slopethresh
                        continue

                    # Optimize bin size                
                    # Initialize processes
                    p = mp.Process(target=zf.do_bin,
                                   args=(fit[n].bintry, phasegood, phatgood,
                                         phot, photerr, fit[n].modelfuncs,
                                         fit[n].modeltypes, fit[n].params,
                                         fit[n].npars, fit[n].npix,
                                         fit[n].stepsize, fit[n].pmin,
                                         fit[n].pmax,
                                         fit[n].parnames,
                                         chisqarray, chislope, l, k, nphot))

                    # Start process
                    jobs.append(p)
                    p.start()

                    # This intentionally-infinite loop continuously
                    # calculates the number of running processes, then
                    # exits if the number of processes is less than
                    # the number requested. This allows additional
                    # processes to spawn as other finish, which is
                    # more efficient than waiting for them all to
                    # finish since some processes can take much longer
                    # than others
                    while True:
                        procs = 0
                        for proc in jobs:
                            if proc.is_alive():
                                procs += 1

                        if procs < fit[n].nprocbin:
                            break

                        # Save the CPU some work.
                        time.sleep(0.1)

                    # Reduce memory usage (otherwise, extra memory
                    # is used while the next object is being loaded)
                    del(phase)
                    del(event)
                    gc.collect()                

            # Make sure all processes finish
            for proc in jobs:
                proc.join()

            fit[n].chisqarray = np.asarray(chisqarray).reshape((ncent,
                                                                nphot,
                                                                nbin))
            fit[n].chislope   = np.asarray(chislope  ).reshape((ncent,
                                                                nphot,
                                                                nbin))

            # Initialize bsig to something ridiculous
            fit[n].bsig = np.inf
            # Determine best binning
            # We also demand that the slope be less than a
            # value, because Deming does and if the slope is
            # too far off from -1/2, binning is not improving the
            # fit in a sensible way
            if all(i >= fit[n].slopethresh for i in fit[n].chislope.flatten()):
                print("Slope threshold too low. Increase and rerun.", file=log)
                print("Setting threshold to 0 so run can complete.", file=log)
                fit[n].slopethresh = 0

            for i in range(ncent):
                for j in range(nphot):
                    for k in range(nbin):
                        if (fit[n].chisqarray[i,j,k] <  fit[n].bsig and
                            fit[n].chislope[i,j,k]   <= fit[n].slopethresh):
                            fit[n].bsig   = fit[n].chisqarray[i,j,k]
                            fit[n].bsigsl = fit[n].chislope  [i,j,k]
                            fit[n].icent = i
                            fit[n].iphot = j
                            fit[n].ibin  = k

            if nevents == 1: # Output is nonsense for joint fits
                print("Best aperture:  " +     fit[n].photdir[fit[n].iphot],
                      file=log)
                print("Best centering: " +     fit[n].centdir[fit[n].icent],
                      file=log)
                print("Best binning:   " + str(fit[n].bintry[ fit[n].ibin]),
                      file=log)
                print("Slope of SDNR vs Bin Size: " + str(fit[n].bsigsl),
                      file=log)

            # Create an output directory if not done yet
            fit[n].outdir = '/'.join([rundir,
                                      fit[n].centdir[fit[n].icent],
                                      fit[n].photdir[fit[n].iphot],
                                      fit[n].outdir, ''])
            if not os.path.isdir(fit[n].outdir):           
                os.makedirs(fit[n].outdir)

            # Write configs to output
            for fname in confignames:
                with open(fit[n].outdir + fname, 'w') as newfile:
                    configobjs[0].write(newfile)


            # Make plot of log(bsig) and slope vs phot, cent, bin
            zp.bsigvis( fit[n], savedir=fit[n].outdir)
            zp.chislope(fit[n], savedir=fit[n].outdir)

            # Reload the event object
            centloc = '/'.join([rundir, fit[n].centdir[fit[n].icent], ''])
            photloc = '/'.join([rundir, fit[n].centdir[fit[n].icent],
                                        fit[n].photdir[fit[n].iphot], ''])

            print("Reloading best POET object.", file=log)
            event = me.loadevent(photloc + fit[n].eventname + "_p5c")
            # Adding the fit object to its event
            event.fit = []
            event.fit.append(fit[n])

            phase = event.phase

            preclipmask  = phase > fit[n].preclip
            postclipmask = phase < fit[n].postclip
            fit[n].clipmask = np.logical_and(preclipmask, postclipmask)
            for i in range(fit[n].ninterclip):
                interclipmask = np.logical_or(phase < fit[n].interclip[2*i  ],
                                              phase > fit[n].interclip[2*i+1])
                fit[n].clipmask = np.logical_and(fit[n].clipmask, interclipmask)        
            fit[n].mask = np.logical_and(   fit[n].clipmask, event.good)

            npos = data.shape[0]

            if npos > 1:
                mflux = np.mean(event.fp.aplev[np.where(event.good)])
                posmflux = np.zeros((event.good.shape[0],1))
                for i in range(event.good.shape[0]):
                    posgood = np.where(event.good[i])
                    posmflux[i] = np.mean(event.fp.aplev[i, posgood])
                event.fp.aplev = event.fp.aplev / posmflux * mflux

            phot    = event.fp.aplev[fit[n].mask]
            photerr = event.fp.aperr[fit[n].mask]

            # Make sure phase is ascending
            ind = np.argsort(phasegood)
            phasegood = phasegood[ind]
            phot      = phot[ind]
            photerr   = photerr[ind]

            # Identify the bright pixels to use
            print("Identifying brightest pixels.", file=log)
            xavg, yavg, rows, cols, pixels = zf.pldpixcoords(event, data,
                                                             fit[n].npix,
                                                             boxsize,
                                                             fit[n].mask)

            zp.pixels(event.meanim[:,:,0], pixels,
                      np.ceil(np.sqrt(fit[n].npix)),
                      xavg, yavg, fit[n].eventname, savedir=fit[n].outdir)

            print("Redoing preparatory calculations.", file=log)
            phat, dP = zf.zen_init(data, pixels)

            phatgood = np.zeros(len(fit[n].mask))

            # Mask out the bad images in phat
            for i in range(fit[n].npix):
                tempphat = phat[:,i].copy()
                tempphatgood = tempphat[fit[n].mask[0]]
                if i == 0:
                    phatgood = tempphatgood.copy()
                else:
                    phatgood = np.vstack((phatgood, tempphatgood))

            # Invert the new array because I lack foresight
            phatgood  = phatgood.T
            phasegood = event.phase[fit[n].mask]

            print("Rebinning to the best binning.", file=log)
            fit[n].binbest = fit[n].bintry[fit[n].ibin]

            binphase, binphot, binphoterr = zf.bindata(phasegood, phot,
                                                       fit[n].binbest,
                                                       yerr=photerr)

            binphotnorm    = binphot    / phot.mean()
            binphoterrnorm = binphoterr / phot.mean()

            for j in range(fit[n].npix):
                if j == 0:
                    _,     binphat = zf.bindata(phasegood,
                                                phatgood[:,j],
                                                fit[n].binbest)
                else:
                    _, tempbinphat = zf.bindata(phasegood,
                                                phatgood[:,j],
                                                fit[n].binbest)
                    binphat = np.column_stack((binphat, tempbinphat))

            fit[n].binphase   = binphase
            fit[n].binphot    = binphot
            fit[n].binphoterr = binphoterr
            fit[n].binphat    = binphat

            fit[n].binphoterrnorm = binphoterrnorm
            fit[n].binphotnorm    = binphotnorm

            fit[n].phase   = phasegood
            fit[n].phot    = phot
            fit[n].photerr = photerr
            fit[n].phat    = phatgood

            eventlist.append(event)
        eventlistlist.append(eventlist)

    # Set up for joint fits and run MCMC
    for n in range(nmodelsets):
        fits = [eventlistlist[i][n].fit[0] for i in range(nevents)]
        mc3y, mc3yerr = [], []
        params, pmin, pmax, stepsize, parnames = [], [], [], [], []
        prior, priorlow, priorup = [], [], []
        for i in range(nevents):
            escale = 1.
            if fits[0].chisqscale:
                print("Rescaling uncertainties for " + fits[i].eventname,
                      file=log)
                ss = fits[i].stepsize.copy()
                # Hacky fix for joint fit issues
                # Ideally we would interpret negative step sizes as
                # whether they set params equal within an event (and
                # use those setting) or whether they set params equal
                # between events and do the following. 
                ss[np.where(ss < 0)] = 1e-5
                indparams = [fits[i].binphase, fits[i].binphat,
                             fits[i].modelfuncs, fits[i].modeltypes,
                             fits[i].npars]
                chisq, _, _, _ = mc3.fit.modelfit(fits[i].params,
                                                  zf.zen, fits[i].binphotnorm,
                                                  fits[i].binphoterrnorm,
                                                  indparams=indparams,
                                                  stepsize=ss,
                                                  pmin=fits[i].pmin,
                                                  pmax=fits[i].pmax,
                                                  prior=fits[i].prior,
                                                  priorlow=fits[i].priorlow,
                                                  priorup=fits[i].priorup)
                nfreep = np.sum(fits[i].stepsize > 0)
                escale = np.sqrt(chisq / (fits[i].binphotnorm.size - nfreep))
                fits[i].binphoterrnorm *= escale
                fits[i].binphoterr     *= escale
            mc3y     = np.concatenate((mc3y,     fits[i].binphotnorm))
            mc3yerr  = np.concatenate((mc3yerr,  fits[i].binphoterrnorm))
            params   = np.concatenate((params,   fits[i].params))
            pmin     = np.concatenate((pmin,     fits[i].pmin))
            pmax     = np.concatenate((pmax,     fits[i].pmax))
            stepsize = np.concatenate((stepsize, fits[i].stepsize))
            parnames = np.concatenate((parnames, fits[i].parnames))
            prior    = np.concatenate((prior,    fits[i].prior))
            priorlow = np.concatenate((priorlow, fits[i].priorlow))
            priorup  = np.concatenate((priorup,  fits[i].priorup))

        # And we're off!    
        print("Beginning MCMC.", file=log)


        mcout = mc3.mcmc(data=mc3y, uncert=mc3yerr, func=zf.mc3zen,
                         indparams=[fits], parname=parnames,
                         params=params, pmin=pmin, pmax=pmax,
                         stepsize=stepsize, prior=prior,
                         priorlow=priorlow, priorup=priorup,
                         walk=fits[0].walk, nsamples=fits[0].nsamples,
                         nchains=fits[0].nchains, nproc=fits[0].nchains,
                         burnin=fits[0].burnin, leastsq=fits[0].leastsq,
                         chisqscale=False,
                         grtest=fits[0].grtest, grbreak=fits[0].grbreak,
                         plots=fits[0].plots,
                         savefile=fits[0].outdir+fits[0].savefile,
                         log=fits[0].outdir+fits[0].mcmclog,
                         chireturn=True)

        bp, CRlo, CRhi, stdp, posterior, Zchain, chiout = mcout

        bpchisq, redchisq, chifactor, bic = chiout

        for fit in fits:
            fit.bic       = bic
            fit.chifactor = chifactor
            fit.bpchisq   = bpchisq
            fit.redchisq  = bpchisq
            fit.bp        = bp
            fit.crlo      = CRlo
            fit.crhi      = CRhi
            fit.stdp      = stdp

        # Parse results between fit objects
        counter = 0
        for m in range(nevents):
            event = eventlistlist[m][n]
            fit   = eventlistlist[m][n].fit[0]
            fit.bp   = bp  [counter:counter+np.sum(fit.npars)]
            fit.stdp = stdp[counter:counter+np.sum(fit.npars)]
            counter += np.sum(fit.npars)
        
        # Post-fit analysis
        for m in range(nevents):
            event = eventlistlist[m][n]
            fit   = eventlistlist[m][n].fit[0]

            fit.binbestfit = zf.zen(fit.bp, fit.binphase, fit.binphat,
                                    fit.modelfuncs, fit.modeltypes, fit.npars)

            # Update errors
            fit.binphoterr     *= chifactor
            fit.binphoterrnorm *= chifactor

            # Make a list of best parameters for each model
            bplist = []
            parind = 0

            for i in range(len(fit.modelstrs)):
                bplist.append(fit.bp[parind:parind+fit.npars[i]])
                parind += fit.npars[i]

            # Calculate model fit without the eclispe
            noeclfit = zf.noeclipse(fit.bp, fit.binphase, fit.binphat,
                                    fit.modelfuncs, fit.modeltypes, fit.npars,
                                    fit.parnames)

            # In case of multiple ecl/tr models, we subtract 1 from each
            # and then add it back in at the end
            fit.bestecl = np.zeros(len(fit.binphase))
            for i in range(len(fit.modelfuncs)):
                if fit.modeltypes[i] == 'ecl/tr':
                    fit.bestecl += (fit.modelfuncs[i](bplist[i], fit.binphase) - 1)

            fit.bestecl += 1

            # Make plots
            print("Making plots.", file=log)
            fit.binnumplot = int(len(fit.binphot)/fit.nbinplot)

            if fit.binnumplot == 0:
                fit.binnumplot = 1

            pbinphase, pbinphot, pbinphoterr = zf.bindata(fit.binphase,
                                                          fit.binphot,
                                                          fit.binnumplot,
                                                          yerr=fit.binphoterr)
            pbinphase, pbinnoeclfit          = zf.bindata(fit.binphase,
                                                          noeclfit,
                                                          fit.binnumplot)
            pbinphase, pbinbestecl           = zf.bindata(fit.binphase,
                                                          fit.bestecl,
                                                          fit.binnumplot)
            pbinphase, pbinbestfit           = zf.bindata(fit.binphase,
                                                          fit.binbestfit,
                                                          fit.binnumplot)


            pbinphotnorm    = pbinphot    / pbinphot.mean()
            pbinphoterrnorm = pbinphoterr / pbinphot.mean()

            zp.normlc(pbinphase, pbinphotnorm, pbinphoterrnorm,
                      pbinnoeclfit, pbinbestecl, fit.binphase,
                      fit.bestecl, 1, title=fit.titles,
                      eventname=fit.eventname, savedir=fit.outdir)

            zp.models(fit, savedir=fit.outdir)

        # Skip post-fit analysis if not desired (saves considerable time)
        if not fit.postanal:
            continue
        
        for m in range(nevents):
            event = eventlistlist[m][n]
            fit   = eventlistlist[m][n].fit[0]
            # Calculate eclipse times in BJD_UTC and BJD_TDB
            # Code adapted from POET p7
            print('Calculating eclipse times in Julian days', file=log)
            offset = event.bjdtdb.flat[0] - event.bjdutc.flat[0]
            if   event.timestd == 'utc':
                fit.ephtimeutc = event.ephtime
                fit.ephtimetdb = event.ephtime + offset
            elif event.timestd == 'tdb':
                fit.ephtimetdb = event.ephtime
                fit.ephtimeutc = event.ephtime - offset
            else:
                print('Assuming that ephemeris is reported in BJD_UTC. Verify!',
                      file=log)
                fit.ephtimeutc = event.ephtime
                fit.ephtimetdb = event.ephtime + offset

            print('BJD_TDB - BJD_UTC = ' + str(offset * 86400.) + ' seconds.',
                  file=log)

            fit.bestmidpt  = fit.bp[  fit.parnames.index('Eclipse Phase')]
            fit.ecltimeerr = fit.stdp[fit.parnames.index('Eclipse Phase')]*event.period

            startutc = event.bjdutc.flat[0]
            starttdb = event.bjdtdb.flat[0]

            fit.ecltimeutc = (np.floor((startutc-fit.ephtimeutc)/event.period) +
                              fit.bestmidpt) * event.period + fit.ephtimeutc
            fit.ecltimetdb = (np.floor((starttdb-fit.ephtimetdb)/event.period) +
                              fit.bestmidpt) * event.period + fit.ephtimetdb

            print('Eclipse time = ' + str(fit.ecltimeutc)
                  + '+/-' + str(fit.ecltimeerr) + ' BJD_UTC', file=log)
            print('Eclipse time = ' + str(fit.ecltimetdb)
                  + '+/-' + str(fit.ecltimeerr) + ' BJD_TDB', file=log)

            # Brightness temperature calculation
            print('Starting Monte-Carlo Temperature Calculation', file=log)    
            kout = kurucz_inten.read(event.kuruczfile, freq=True)

            filterf = np.loadtxt(event.filtfile, unpack=True)
            filterf = np.concatenate((filterf[0:2,::-1].T,[filterf[0:2,0]]))

            logg     = np.log10(event.tep.g.val*100.)
            loggerr  = np.log10(event.tep.g.uncert*100.)
            tstar    = event.tstar
            tstarerr = event.tstarerr

            # Find index of depth
            countfix = 0
            for i in range(len(fit.parnames)):
                if fit.parnames[i] in ['Depth', 'depth', 'Maximum Eclipse Depth', 'Eclipse Depth']:
                    idepth = i

            # Count number of fixed parameters prior to the depth
            # parameter, to adjust the idepth
            for i in range(idepth):
                if fit.stepsize[i] <= 0:
                    countfix += 1

            idepthpost = idepth - countfix

            depthpost = posterior[:,idepthpost]

            if posterior.shape[0] < fit.numcalc:
                print("WARNING: not enough samples for Temperature Monte-Carlo!",
                      file=log)
                print("Reducing numcalc to match size of MCMC posterior.",
                      file=log)
                fit.numcalc  = posterior.shape[0]
                slicenum = posterior.shape[0] // fit.numcalc # always 1, but for clarity
                slicelim = slicenum * fit.numcalc
            else:
                # Since slice step must be an integer, we need to calculate
                # the limit of the posterior to slice such that we get
                # an array of the correct length
                slicenum = posterior.shape[0] // fit.numcalc
                slicelim = slicenum * fit.numcalc

            bsdata    = np.zeros((3,fit.numcalc))

            # Use every nth eclipse depth except the 0th
            bsdata[0] = depthpost[:slicelim:slicenum]
            bsdata[1] = np.random.normal(logg,  loggerr,  fit.numcalc)
            bsdata[2] = np.random.normal(tstar, tstarerr, fit.numcalc)

            tb, tbg, numnegf, fmfreq = zf.calcTb(bsdata, kout, filterf, event)

            tbm   = np.median(tb [np.where(tb  > 0)])
            tbsd  = np.std(   tb [np.where(tb  > 0)])
            tbgm  = np.median(tbg[np.where(tbg > 0)])
            tbgsd = np.std(   tbg[np.where(tbg > 0)])

            print('Band-center brightness temp = '
                  + str(round(tbgm,  2)) + ' +/- '
                  + str(round(tbgsd, 2)) + ' K', file=log)
            print('Integral    brightness temp = '
                  + str(round(tbm,  2)) + ' +/- '
                  + str(round(tbsd, 2)) + ' K', file=log)

            event.fit[0].fluxuc   = event.fp.aplev[np.where(event.good)] 
            event.fit[0].clipmask = fit.clipmask[np.where(event.good)]
            event.fit[0].flux     = event.fp.aplev[fit.mask] # Clipped flux
            event.fit[0].bestfit  = zf.zen(bp, fit.phase, fit.phat, fit.modelfuncs,
                                        fit.modeltypes, fit.npars) # Best fit (norm)

            # Data from plot
            event.fit[0].pbinphase       = pbinphase
            event.fit[0].pbinphot        = pbinphot
            event.fit[0].pbinphoterr     = pbinphoterr
            event.fit[0].pbinnoeclfit    = pbinnoeclfit
            event.fit[0].pbinbestfit     = pbinbestfit
            event.fit[0].pbinphotnorm    = pbinphotnorm
            event.fit[0].pbinphoterrnorm = pbinphoterrnorm

            # Temperatures
            event.fit[0].tbm   = tbm
            event.fit[0].tbsd  = tbsd
            event.fit[0].tbgm  = tbgm
            event.fit[0].tbgsd = tbgsd

            # Optimal phot description
            event.fit[0].bestphotdir   = fit.photdir[fit.iphot]
            event.fit[0].bestcentdir   = fit.centdir[fit.icent]
            event.fit[0].bestbinsize   = fit.bintry [fit.ibin]



            # Write IRSA table and FITS file
            if not os.path.exists(fit.outdir + 'irsa'):
                os.mkdir(fit.outdir + 'irsa')

            # Set the topstring
            topstring = zf.topstring(fit.papername, fit.month, fit.year,
                                     fit.journal, fit.instruments,
                                     fit.programs, fit.authors)

            irsa.do_irsa(event, event.fit[0], directory=fit.outdir,
                         topstring=topstring)

            
    print("Saving.")
    for n in range(nmodelsets):
        for i in range(nevents):
            event = eventlistlist[i][n]
            fit   = eventlistlist[i][n].fit[0]    
            run.p6Save(event, fit.outdir)
            
    minbic = np.inf
    for n in range(nmodelsets):
        for i in range(nevents):
            event = eventlistlist[i][n]
            fit   = eventlistlist[i][n].fit[0]
            print("For models " + ' '.join(fit.modelstrs) + ":", file=log)
            print("Best aperture:  " +     fit.photdir[fit.iphot], file=log)
            print("Best centering: " +     fit.centdir[fit.icent], file=log)
            print("Best binning:   " + str(fit.bintry[ fit.ibin]), file=log)
            minbic = np.min((minbic, fit.bic))

    print("Models\tBIC\tdelBIC", file=log)            
    for n in range(nmodelsets):
        for i in range(nevents):
            event = eventlistlist[i][n]
            fit   = eventlistlist[i][n].fit[0]            
            print(' '.join(fit.modelstrs) + '\t' + str(fit.bic) +
                  '\t' + str(fit.bic - minbic), file=log)
    
    print("End:  %s" % time.ctime(), file=log)

    log.close()
    for n in range(nmodelsets):
        for i in range(nevents):
            fit   = eventlistlist[i][n].fit[0]
            shutil.copy(templogfile, fit.outdir + logfile)

    # Delete temporary log
    os.unlink(templogfile)

    # Return directory of output (not used for joint fits)
    return fit.outdir, fit.centdir[fit.icent], fit.photdir[fit.iphot], chiout

if __name__ == "__main__":
    main(*sys.argv[1:])
