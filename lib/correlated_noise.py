

import numpy as np
import bindata as bd


# COMPUTE ROOT-MEAN-SQUARE AND STANDARD ERROR OF DATA FOR VARIOUS BIN SIZES
def computeRMS(data, maxnbins=None, binstep=1, isrmserr=False):
    ''''
    Compute root-mean-square and standard error of data over various
    bin sizes.

    Revisions
    ---------
    ????-??-?? unknown  Initial Implementation. 
    2017-02-07 rchallen Updated for current python indexing rules.
                        rchallen@knights.ucf.edu
    '''
               
    #data    = fit.normresiduals
    #maxnbin = maximum # of bins
    #binstep = Bin step size
    
    # bin data into multiple bin sizes
    npts    = data.size
    if maxnbins is None:
        maxnbins = npts/10.
    binsz   = np.arange(1, maxnbins+binstep, step=binstep)
    nbins   = np.zeros(binsz.size, dtype=int)
    rms     = np.zeros(binsz.size)
    rmserr  = np.zeros(binsz.size)
    for i in range(binsz.size):
        nbins[i] = int(np.floor(data.size/binsz[i]))
        bindata   = np.zeros(nbins[i], dtype=float)
        # bin data
        # ADDED INTEGER CONVERSION, mh 01/21/12
        for j in range(int(nbins[i])):
            bindata[j] = data[j*binsz[i]:(j+1)*binsz[i]].mean()
        # get rms
        rms[i]    = np.sqrt(np.mean(bindata**2))
        rmserr[i] = rms[i]/np.sqrt(2.*int(nbins[i]))
    # expected for white noise (WINN 2008, PONT 2006)
    stderr = (data.std()/np.sqrt(binsz))*np.sqrt(nbins/(nbins - 1.))
    if isrmserr == True:
        return rms, stderr, binsz, rmserr
    else:
        return rms, stderr, binsz

# Compute standard error
def computeStdErr(datastd, datasize, binsz):
    #datastd  = fit.normresiduals.std()
    #datasize = fit.normresiduals.size
    #binsz    = array of bins
    
    nbins   = np.zeros(binsz.size)
    for i in range(binsz.size):
        nbins[i] = int(np.floor(datasize/binsz[i]))
    stderr = (datastd/np.sqrt(binsz))*np.sqrt(nbins/(nbins - 1.))
    return stderr

'''
    # plot results
    plt.figure(1)
    plt.clf()
    plt.axes([0.12, 0.12, 0.82, 0.82])
    plt.loglog(binsz, rms, color='black', lw=1.5, label='RMS')    # our noise
    plt.loglog(binsz, stderr, color='red', ls='-', lw=2, label='Std. Err.') # expected noise
    plt.xlim(0, binsz[-1]*2)
    plt.ylim(rms[-1]/2., rms[0]*2.)
    plt.xlabel("Bin Size", fontsize=14)
    plt.ylabel("RMS", fontsize=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.legend()
    plt.text(binsz[-1], rms[0], "Channel {0}".format(chan+1), fontsize=20,
             ha='right')
    plt.savefig("wa012bs{0}1-noisecorr.ps".format(chan+1))
    plt.title("Channel {0} Noise Correlation".format(chan+1), fontsize=25)
    plt.savefig("wa012bs{0}1-noisecorr.png".format(chan+1))
'''

def bsigchi(fit):
    '''
    Some code from ZEN which calculates the binned-sigma
    relation chi-squared used in the PLD method.
    '''
    zeropoint = fit.sdnr
    sdnr     = []
    binlevel = []
    err      = []
    resppb = 1.
    resbin = float(len(fit.abscissa))
    num    = float(len(fit.normresiduals))
    sigma = np.std(fit.normresiduals)

    while resbin > 16:
        binnedres = bd.bindata(int(resbin), mean = [fit.normresiduals])
        sdnr.append(np.std(binnedres))
        binlevel.append(resppb)
        ebar = sigma/np.sqrt(2. * resbin)
        err.append(ebar)
        resppb *= 2
        resbin = np.floor(num/resppb)

    sdnr[0] = sigma
    err[0]  = sigma/np.sqrt(2*num)

    sdnr = np.asarray(sdnr)
    binlevel = np.asarray(binlevel)
    sdnrchisq, slope = reschisq(sdnr, binlevel, err, zeropoint)

    return sdnrchisq

def reschisq(y, x, yerr, zeropoint):
    '''
    Little function to calculate chisq of a log data set against
    a line with slope -0.5. Used to check residual binning.

    Taken from ZEN.
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
      errmed = yerr[int(len(yerr)/2)]
    else:
      errmed = np.median(yerr)
      
    chisq = np.sum(diff**2/errmed**2)
    # Actually fit a line and find the slope. This will be used later
    # to discard some fits
    fit = np.polyfit(np.log10(x), np.log10(y), 1)
    slope = fit[0]
    return chisq, slope
  
