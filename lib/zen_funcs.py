import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import linear_model
import sys
import time

sys.path.insert(1, "./mccubed/")
sys.path.append('lib')
import kurucz_inten
import MCcubed as mc3
import models_c as mc
import transplan_tb

def zen_init(data, pixels):
  """
  This function does the initial calculations for pixel-level decorrelation.

  Parameters:
  -----------
  data: ndarray
    3D float array of images

  pixels: ndarray
    2D array coordinates of pixels to consider
    EX: array([[ 0,  1],
               [ 2,  3],
               [ 4,  5],
               [ 6,  7],
               [ 8,  9],
               [10, 11],
               [12, 13],
               [14, 15],
               [16, 17],
               [18, 19]])

  Returns:
  --------

  Example:
  --------
  >>> import numpy as np
  >>> data = np.arange(200).reshape(8,5,5)
  >>> pixels = [[1,2],[2,1],[2,2],[2,3],[3,2]]
  >>> res = zen_init(data,pixels)

  Modification History:
  ---------------------
  2015-11-02 em      Initial implementation
  2015-11-20 rchallen Generalized for any given pixels

  """
  # Set number of frames and image dimensions
  nframes, ny, nx, nsets = np.shape(data)

  # Set number of pixels
  npix = len(pixels)
  
  # Initialize array for flux values of pixels
  p = np.zeros((nframes, npix))

  # Extract flux values from images
  for i in range(npix):
    for j in range(nframes):
      p[j,i] = data[j, pixels[i][0], pixels[i][1],0]

  # Initialize array for normalized flux of pixels
  phat = np.zeros(p.shape)
  
  # Remove astrophysics by normalizing
  for t in range(nframes):
    phat[t] = p[t]/np.sum(p[t])

  # Calculate mean flux through all frames at each pixel
  pbar    = np.mean(phat, axis=0)

  # Difference from the mean flux
  dP      = phat - pbar

  return(phat, dP)


def pldpixcoords(event, data, npix, boxsize, mask):
  nx = data.shape[1]
  ny = data.shape[2]

  xavg = np.int(np.round(np.average(event.fp.x)))
  yavg = np.int(np.round(np.average(event.fp.y)))

  photavg = np.average(data[mask[0], yavg-boxsize:yavg+boxsize,
                                  xavg-boxsize:xavg+boxsize],
                       axis=0)[:,:,0]

  photavgflat = photavg.flatten()

  flatind = photavgflat.argsort()[-npix:]

  rows = flatind // photavg.shape[1]
  cols = flatind %  photavg.shape[0]

  pixels = []

  for i in range(npix):
    pixels.append([rows[i]+yavg-boxsize,cols[i]+xavg-boxsize])

  return xavg, yavg, rows, cols, pixels

def fitopt(fit, configobj, rundir, n):
  '''
  Places all configuration options in the fit object, and sets
  up defaults where appropriate.
  '''
  #### EVENT OPTIONS ####
  configdict = configobj['EVENT']
  
  fit.eventname = configdict['eventname']

  fit.modelfile = configdict['modelfile']

  fit.outdir    = configdict['outdir']
  
  fit.centdir   = configdict['cent'].split()
  fit.photdir   = configdict['phot'].split()

  if fit.centdir == ['all']:
    centdir = []
    for cent in os.listdir(rundir):
      if os.path.isdir(rundir + '/' + cent):
        centdir.append(cent)
    # The 'plots' directory is made by p6, and does not
    # contain photometry directories
    if 'plots' in centdir:
      centdir.remove('plots')
    fit.centdir = centdir

  if fit.photdir == ['all']:
    photdir = []
    for cent in fit.centdir:
      for phot in os.listdir(rundir + '/' + cent):
        if os.path.isdir(rundir + '/'
                         + cent + '/'
                         + phot):
          photdir.append(phot)
    fit.photdir = np.unique(photdir)

  fit.npix = int(configdict['npix'])
  # Gets only the nth sublist of models
  fit.modelstrs = [s for s in configdict['models'].split('\n')[n].split()]

  fit.slopethresh = configdict.getfloat('slopethresh')

  fit.nbinplot = configdict.getint('nbinplot')
  fit.nprocbin = configdict.getint('nprocbin')

  fit.preclip  = configdict.getfloat('preclip')
  fit.postclip = configdict.getfloat('postclip')

  if 'interclip' in configdict.keys():
    fit.interclip = [float(s) for s in configdict['interclip'].split()]
    if len(fit.interclip) % 2.0 != 0:
      print('Warning! Interclip ranges specifed incorrectly!')
    fit.ninterclip = int(len(fit.interclip) / 2)
  else:
    fit.interclip = [0, 0]
    fit.ninterclip = 1

  fit.bintry = [int(s) for s in configdict['bintry'].split()]

  fit.maxbinsize = int(configdict['maxbinsize'])

  fit.numcalc = configdict.getint('numcalc')

  if fit.bintry == [0]:
    fit.bintry = 2 ** np.arange(0, np.int(np.log2(fit.maxbinsize)) + 1)

  if 'priorvars' in configdict.keys():
    fit.priorvars = configdict['priorvars'].split()
    fit.priorvals = [float(s) for s in configdict['priorvals'].split()]

  fit.papername   = configdict['papername']
  fit.month       = configdict['month']
  fit.year        = configdict['year']
  fit.journal     = configdict['journal']
  fit.instruments = configdict['instruments']
  fit.programs    = configdict['programs']
  fit.authors     = configdict['authors']

  fit.postanal    = configdict.getboolean('postanal')

  #### MCMC OPTIONS ####
  configdict = configobj['MCMC']

  fit.plots      = configdict.getboolean('plots')
  fit.bins       = configdict.getboolean('bins')
  fit.titles     = configdict.getboolean('titles')
  fit.chisqscale = configdict.getboolean('chisqscale')
  fit.grtest     = configdict.getboolean('grtest')
  fit.grbreak    = configdict.getboolean('grbreak')
  fit.leastsq    = configdict.getboolean('leastsq')

  fit.walk = configdict['walk']

  fit.nsamples = int(float(configdict['nsamples']))
  fit.nchains  = int(float(configdict['nchains']))
  fit.burnin   = int(float(configdict['burnin']))

  fit.savefile = configdict['savefile']
  fit.mcmclog  = configdict['logfile']
  
def zen(par, x, phat, models, modeltypes, npars):
  """
  Zen function. Combines the specified models additively.

  Parameters:
  -----------
  par: ndarray
    All parameters for all models
  x: ndarray
    Locations to evaluate eclipse model
  phat: ndarray
    Array of p_hat for the PLD model
  models: list
    List of functions to combine
  npars: list
    List of ints to separate par array into subarrays
    for each model in models
    

  Returns:
  --------
  y: ndarray
    Model evaluated at x

  Modification History:
  ---------------------
  2015-11-02 em      Initial implementation.
  2015-11-20 rchallen Adaptation for use with MCcubed

  Notes:
  ------
  Only allows for quadratic ramp functions
  """

  # Initialize output array
  y = np.zeros(len(x))

  # Set up parlist (list of numpy arrays of model parameters
  # for each model in models)
  parlist = []
  parind  = 0

  for i in range(len(models)):
    parlist.append(par[parind:parind+npars[i]])
    parind += npars[i]
  
  for i in range(len(models)):
    currmodel = models[i](parlist[i], x, phat)
    y += currmodel
    # All eclipse models are with respect to 1,
    # but we want them to be with respect to 0
    # for an additive model like PLD
    if modeltypes[i] == 'ecl/tr':
      y -= 1

  return y

def mc3zen(pars, fits=None):
  """
  Function to evaluate the light curve model in MC3. Evaluates the 
  model for all fits given, to facilitate joint fits.
  """
  nfits = len(fits)

  models = []

  counter = 0
  for j in range(nfits):
    fits[j].jointiparams = range(counter, counter + np.sum(fits[j].npars))
    counter += np.sum(fits[j].npars)

  for j in range(nfits):
    models.append(zen(pars[fits[j].jointiparams], fits[j].binphase,
                      fits[j].binphat, fits[j].modelfuncs,
                      fits[j].modeltypes, fits[j].npars))

  return np.concatenate(models)

def pld(params, x, phat):
  '''
  Computes the actual PLD model.
  '''
  y = 0
  
  for i in range(len(params)):
       y += params[i] * phat[:,i]

  return y

def pldcross2(params, x, phat):
  '''
  Computes the 2nd-order cross terms of the PLD model.
  '''
  y = 0

  npix = phat.shape[1]

  counter = 0
  for i in range(npix):
      for j in range(i, npix):
          y += params[counter] * phat[:,i] * phat[:,j]
          counter += 1

  return y

def bindata(x, y, ppb, yerr=None):
    nbin = int(len(x)/ppb)

    binx = np.zeros(nbin)
    biny = np.zeros(nbin)

    # make sure arrays are sorted by x
    if x.size == y.size:
      sind = np.argsort(x)
      x = x[sind]
      y = y[sind]

    if type(yerr) != type(None):
        binyerr = np.zeros(nbin)

    for i in range(nbin):
        binx[i] = np.mean(x[i*ppb:(i+1)*ppb])
        biny[i] = np.mean(y[i*ppb:(i+1)*ppb])
        if type(yerr) != type(None):
            binyerr[i] = np.mean(yerr[i*ppb:(i+1)*ppb])/ppb**.5

    if type(yerr) != type(None):
        return binx, biny, binyerr
    else:
        return binx, biny


def reschisq(y, x, yerr, zeropoint):
    '''
    Little function to calculate chisq of a log data set against
    a line with slope -0.5. Used to check residual binning.
    '''
    m = -0.5
    line = 10.0**(np.log10(zeropoint) + m * np.log10(x))
    diff = y - line
    # We use median of the error here just because the first implementation
    # of PLD does. It is probably better to weight each point
    # individually (and in fact, weighting them equally should not
    # have any effect).

    # Determine median in the manner IDL does. Since err is
    # always increasing, we can just index at the middle
    if len(yerr) % 2 == 0:
      errmed = yerr[len(yerr)//2]
    else:
      errmed = np.median(yerr)
      
    chisq = np.sum(diff**2/errmed**2)
    # Actually fit a line and find the slope. This will be used later
    # to discard some fits
    fit = np.polyfit(np.log10(x), np.log10(y), 1)
    slope = fit[0]
    return chisq, slope

def do_bin(bintry, phasegood, phatgood, phot, photerr, modelfuncs,
           modeltypes, params, npars, npix, stepsize, pmin, pmax,
           parnames, chisqarray, chislope, photind, centind, nphot,
           regress=False, plot=False):
    '''
    Function to be launched with multiprocessing.

    Notes
    -----
    This function modifies (fills in) the passed chisqarray variable.
    It returns nothing because this function is meant to be used
    with multiprocessing.
    '''
    
    # Optimize bin size
    print("Calculating unbinned SDNR")
    indparams = [phasegood, phatgood, modelfuncs, modeltypes, npars]
    dummy, fitbestp, model, dummy = mc3.fit.modelfit(params, zen,
                                                     phot, photerr,
                                                     indparams,
                                                     stepsize,
                                                     pmin, pmax)

    zeropoint = np.std((phot - model)/noeclipse(fitbestp, phasegood, phatgood,
                                                modelfuncs, modeltypes, npars,
                                                parnames))
    
    print("SDNR of unbinned model: " + str(zeropoint))
    
    print("Optimizing bin size.")
    for i in range(len(bintry)):
        # Bin the phase and phat
        for j in range(npix):
            if j == 0:
                binphase,     binphat = bindata(phasegood,
                                                phatgood[:,j],
                                                bintry[i])
            else:
                binphase, tempbinphat = bindata(phasegood,
                                                phatgood[:,j],
                                                bintry[i])
                binphat = np.column_stack((binphat, tempbinphat))
                # Bin the photometry and error
                # Phase is binned again but is identical to
                # the previously binned phase.
        binphase, binphot, binphoterr = bindata(phasegood, phot,
                                                bintry[i], yerr=photerr)

        # Normalize
        photnorm    = phot    #/ phot.mean()
        photerrnorm = photerr #/ phot.mean()

        binphotnorm    = binphot    #/ phot.mean()
        binphoterrnorm = binphoterr #/ phot.mean()

        # Number of parameters we are fitting
        # This loop removes fixed parameters from the
        # count
        nparam = len(stepsize)
        for ss in stepsize:
            if ss <= 0:
                nparam -= 1

        #Minimize chi-squared for this bin size
        indparams = [binphase, binphat, modelfuncs, modeltypes, npars]
        chisq, fitbestp, model, dummy = mc3.fit.modelfit(params, zen,
                                                         binphotnorm,
                                                         binphoterrnorm,
                                                         indparams,
                                                         stepsize,
                                                         pmin, pmax,
                                                         lm=True)
        unbinnedres = photnorm - zen(fitbestp, phasegood, phatgood, modelfuncs,
                                     modeltypes, npars)
        
        # Normalize residuals
        noeclfit = noeclipse(fitbestp, phasegood, phatgood, modelfuncs,
                             modeltypes, npars, parnames)
        normubinres = unbinnedres / noeclfit

        # Calculate model on unbinned data from parameters of the
        # chi-squared minimization with this bin size. Calculate
        # residuals for binning
        sdnr        = []
        binlevel    = []
        err         = []
        resppb      = 1.
        resbin = float(len(phasegood))
        num    = float(len(normubinres))
        sigma = np.std(normubinres, ddof=1) * np.sqrt(num   /(num   -nparam))
        
        # Bin the residuals, calculate SDNR of binned residuals. Do this
        # until you are binning to <= 16 points remaining.
        while resbin > 16:
            xrem = int(num - resppb * resbin)
            # This part is gross. Need to clean up
            dummy, binnedres = bindata(phasegood[xrem:],
                                       normubinres[xrem:],
                                       int(resppb))
            sdnr.append(np.std(binnedres, ddof=1) *
                        np.sqrt(resbin/(resbin-nparam)))
            binlevel.append(resppb)
            ebar  = sigma/np.sqrt(2 * resbin)
            err.append(ebar)
            resppb *= 2
            resbin = np.floor(num/resppb)
            
        sdnr[0] = sigma
        err[0]  = sigma/np.sqrt(2*num)

        # Calculate chisquared of the various SDNR wrt line of slope -0.5
        # passing through the SDNR of the unbinned residuals
        # Record chisquared
        sdnr     = np.asarray(sdnr)
        binlevel = np.asarray(binlevel)
        sdnrchisq, slope = reschisq(sdnr, binlevel, err, zeropoint)

        # Some diagnostic plotting
        if plot == True:
            ax = plt.subplot(111)
            ax.set_xscale('log')
            ax.set_yscale('log')
            plt.errorbar(binlevel, sdnr, yerr=err, fmt='o')
            plt.plot(binlevel,
                     10**(np.log10(sdnr[0]) + (-.5) * np.log10(binlevel)))
            #plt.xlim((10**-.1,10**2.8))
            #plt.ylim((10**-3.8,10**-2.4))
            plt.savefig('bsigplot.png')
            plt.clf()
            
        print(" Ap: "    + str(photind) +
              " Cent: "  + str(centind) +
              " Bin: "   + str(i)       +
              " Chisq: " + str(sdnrchisq))
        chisqarray[i+len(bintry)*photind+len(bintry)*nphot*centind] = sdnrchisq
        chislope[  i+len(bintry)*photind+len(bintry)*nphot*centind] = slope

def calcTb(bsdata, kout, filterf, event):
    '''
    Monte-Carlo temperature calculation. Taken from POET, P7.

    Params
    ------
    bsdata: array
        3 x numcalc array of eclipse depths from the MCMC, a normal
        sampling of distribution of the log(g) of the star, and a 
        normal sampling of the temperature of the star

    kout: array
        output of the kurucz_inten.read() function in frequency space

    filterf: array
        read-in filter file. Format is weird

    event: POET event object

    Returns
    -------
    tb: array
        array of integral brightness temperatures from the Monte-Carlo

    tbg: array
        array of band-center brightness temperatures from the
        Monte-Carlo

    numnegf: integer
        number of -ve flux values in allparams

    fmfreq: None
        Option for transplan_tb function. Set to none in the function
        so it has no effect...
    '''
    #reload(transplan_tb)
    
    kinten, kfreq, kgrav, ktemp, knainten, khead = kout
    ffreq     = event.c / (filterf[:,0] * 1e-6)
    ftrans    = filterf[:,1]
    sz        = bsdata[1].size
    tb        = np.zeros(sz)
    tbg       = np.zeros(sz)
    numnegf   = 0               #Number of -ve flux values in allparams
    #guess    = 1        #1: Do not compute integral
    complete  = 0
    fmfreq    = None
    kstar     = kurucz_inten.interp2d(kinten, kgrav, ktemp,
                                      bsdata[1], bsdata[2])
    fmstar    = None
    if (event.tep.rprssq.val > 0):
      arat      = np.random.normal(event.tep.rprssq.val,
                                   event.tep.rprssq.uncert, sz)
    else:
      if (event.tep.rprs.val < 0):
        print("WARNING: radius ratio undefined in TEP file.")
      arat      = np.random.normal(event.tep.rprs.val**2,
                                   event.tep.rprs.uncert*np.sqrt(2*event.tep.rprs.val), sz)
    for i in range(sz):
        if bsdata[0,i] > 0:
            fstar   = np.interp(ffreq, kfreq, kstar[i])
            tb[i], tbg[i] = transplan_tb.transplan_tb(arat[i], bsdata[0,i],
                                                      bsdata[1,i], bsdata[2,i],
                                                      kfreq=kfreq, kgrav=kgrav,
                                                      ktemp=ktemp,
                                                      kinten=kinten,
                                                      ffreq=ffreq,
                                                      ftrans=ftrans,
                                                      fmfreq=fmfreq,
                                                      fstar=fstar,
                                                      fmstar=fmstar)
        else:
            numnegf += 1
        if (i % (sz / 5) == 0): 
            print(str(complete * 20) + "% complete at " + time.ctime())
            complete += 1
    return tb, tbg, numnegf, fmfreq

def buildlist(stringlist):
    '''
    Builds a string list into a list for use in real text.
    '''
    string_text = ''
    
    if len(stringlist) == 1:
        string_text += stringlist[0]
    elif len(stringlist) == 2:
        string_text += (stringlist[0] + ' and ' + stringlist[1])
    else:
        for i in range(len(stringlist)):
            if i == 0:
                string_text += (stringlist[0])
            elif i == len(stringlist) - 1:
                string_text += (', and ' + stringlist[i])
            else:
                string_text += (', '     + stringlist[i])

    return string_text
            

def topstring(papername, month, year, journal, instruments,
              programs, authors):
        '''
        Makes a topstring for FITS and IRSA files.
        '''

        authors     = list(authors.split('\n'))
        instruments = list(instruments.split(' '))
        programs    = list(programs.split(' '))

        authors_text     = buildlist(authors)
        instruments_text = buildlist(instruments)
        programs_text    = buildlist(programs)

        # Set the topstring
        topstring = "This file contains the light curves from the paper:  " + papername + " by " + authors_text + ", which was submitted in " + month + " " + year + " to " + journal + ". The data are from the " + instruments_text + " on the NASA Spitzer Space Telescope, programs " + programs_text + ", which are availabe from the public Spitzer archive (http://sha.ipac.caltech.edu). The paper cited above and its electronic supplement, of which this file is a part, describe the observations and data analysis. The data are in an ASCII table that follows this header. The TTYPE* keywords describe the columns in the ASCII table. Units are microJanskys for flux, pixels for distance, and seconds for time. All series (pixel positions, frame number, etc.) are zero-based. Data in a table row are valid if GOOD equals 1, invalid otherwise."

        # Split up topstring so that it fits in a FITS header
        linelen = 65

        for i in range(2*len(topstring)):
            # Initialize reformatted string
            if i % linelen == 0 and i == linelen:
                topstring_reformat = topstring[:i]
                # Set a mark so we know where to begin the next chunk
                mark = i
            elif i % linelen == 0 and i != linelen and i != 0:
                # Add another line of linelen characters to the topstring
                # starting from the mark
                topstring_reformat += ('\n' + topstring[mark:i])
                # Update the mark
                mark = i

        # Add on any remaining trailing characters
        topstring_reformat += topstring[mark:]

        # Strip off extra newlines at the end
        topstring_reformat = topstring_reformat.rstrip('\n') 

        return topstring_reformat

def noeclipse(pars, phase, phat, modelfuncs, modeltypes, npars, parnames):
  '''
  Calculates the ZEN model with eclipse depth set to 0.
  '''
  noeclpars = np.zeros(len(pars))

  depthstrs = ['Eclipse Depth',
               'Radius Ratio',
               'Maximum Eclipse Depth']

  for i in range(len(noeclpars)):
    if parnames[i] in depthstrs:
      noeclpars[i] = 0
    else:
      noeclpars[i] = pars[i]

  # Calculate model fit without the eclipse
  noeclmodel = zen(noeclpars, phase, phat, modelfuncs, modeltypes, npars)

  return noeclmodel
  
  

