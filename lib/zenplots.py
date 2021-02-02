import matplotlib.pyplot as plt
import numpy as np
pltfmt  = ['b-',  'g-',  'r-',  'c-',  'm-',  'y-',  'k-',  'w-' ]

def normlc(phase, phot, photerr, noecl, bestecl, ubphase, ubbestecl,
           fignum, j=0, title=False, eventname='', savedir = './'):
    '''
    Plots a normalized light curve for a PLD model. Note that phase, phot,
    photerr, noecl should be matching in size and ubphase, bestecl
    should also match in size, but they can be binned differently.
    '''
    plt.rcParams.update({'legend.fontsize':13})
    plt.figure(fignum, figsize=(8,6))
    plt.clf()
    
    a = plt.axes([0.15,0.35,0.8,0.55])
    a.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.4f'))
    if type(title) != type(None):
        plt.title("Normalized Light Curve")
        
    plt.errorbar(phase, phot-noecl+1, photerr, fmt='ko',
                 ms=4, lw=1, label='Binned Data', zorder=0)

    plt.plot(ubphase, ubbestecl, pltfmt[j], label='Best Fit', lw=2,
             zorder=1)

    plt.setp(a.get_xticklabels(), visible = False)

    plt.yticks(size=13)
    plt.ylabel('Normalized Flux',size=14)
    
    xmin, xmax = plt.xlim()

    plt.axes([0.15,0.1,0.8,0.2])
    flatline = np.zeros(len(phase))
    plt.plot(phase, phot-noecl-bestecl+1, 'ko',ms=4)
    plt.plot(phase, flatline,'k:',lw=1.5)
    plt.xlim(xmin,xmax)
    plt.xticks(size=13)
    plt.yticks(size=13)
    plt.xlabel('Phase',size=14)
    plt.ylabel('Residuals',size=14)
    plt.legend(loc='best')


    plt.savefig(savedir + eventname + '-normlc.png')

def bsigvis(fit, savedir='./'):
    '''
    Plots the log10 of bsig in separate panels for each bin size.
    '''
    ncent, nphot, nbin = fit.chisqarray.shape

    fig = plt.figure(figsize=(16,8))

    # Make sure all panels scale equally
    vmin = np.min(np.log10(fit.chisqarray[np.isfinite(fit.chisqarray)]))
    vmax = np.max(np.log10(fit.chisqarray[np.isfinite(fit.chisqarray)]))

    # Define a list of tuples with custom data types for sorting (purely
    # alphabetical is not quite right)
    photsplit = []
    photdtype = [('prefix', 'S10'), ('scale', float), ('offset', float)]
    for i in range(nphot):
      photsplit.append((fit.photdir[i][ :2],
                        np.float(fit.photdir[i][2:5])/100,
                        np.float(fit.photdir[i][5:9])/100))

    photsplit = np.array(photsplit, dtype=photdtype)

    photsortind = np.argsort(photsplit, order=['prefix', 'scale', 'offset'])

    chisqsort = fit.chisqarray[:,photsortind,:]
    photsort  = np.array(fit.photdir)[photsortind]

    # Make a panel for each bin size
    for i in range(nbin):
      fig.add_subplot(nbin, 1, i+1)
      plt.imshow(np.log10(chisqsort[:,:,i]),
                 vmin=vmin, vmax=vmax)
      plt.autoscale(False)
      plt.yticks(np.arange(ncent), labels=fit.centdir)
      if i == nbin-1:
        plt.xticks(np.arange(nphot), labels=photsort, rotation=90)
      else:
        plt.xticks([])
      plt.title("Bin size: {}".format(2**i), fontsize=6)

    fig.tight_layout()

    plt.savefig(savedir + fit.eventname + '-logbsig.png')
    plt.clf()

def chislope(fit, savedir='/'):
    '''
    Plots the bsig fit slope in separate panels for each bin size.
    Very similar to bsigvis() function.
    '''
    ncent, nphot, nbin = fit.chislope.shape

    fig = plt.figure(figsize=(16,8))

    vmin = np.min(fit.chislope[np.isfinite(fit.chislope)])
    vmax = np.max(fit.chislope[np.isfinite(fit.chislope)])

    photsplit = []
    photdtype = [('prefix', 'S10'), ('scale', float), ('offset', float)]
    for i in range(nphot):
      photsplit.append((fit.photdir[i][ :2],
                        np.float(fit.photdir[i][2:5])/100,
                        np.float(fit.photdir[i][5:9])/100))

    photsplit = np.array(photsplit, dtype=photdtype)

    photsortind = np.argsort(photsplit, order=['prefix', 'scale', 'offset'])

    chislsort = fit.chislope[:,photsortind,:]
    photsort  = np.array(fit.photdir)[photsortind]

    for i in range(nbin):
      fig.add_subplot(nbin, 1, i+1)
      plt.imshow(chislsort[:,:,i],
                 vmin=vmin, vmax=vmax)
      plt.autoscale(False)
      plt.yticks(np.arange(ncent), labels=fit.centdir)
      if i == nbin-1:
        plt.xticks(np.arange(nphot), labels=photsort, rotation=90)
      else:
        plt.xticks([])
      plt.title("Bin size: {}".format(2**i), fontsize=6)

    fig.tight_layout()

    plt.savefig(savedir + fit.eventname + '-chislope.png')    
    plt.clf()

def pixels(im, pixels, trim, x, y, eventname, savedir='./'):
    '''
    Plots the pixel parameter numbers over a zoomed image of the
    star.

    Parameters
    ----------
    im: 2d array
        Image to be plotted.

    pixels: 2d list
        List of lists, where the first dimension is pixel number
        and the second dimension is y and x position.

    trim: float
        Size of window to plot.

    x, y: floats
        Position to center the window.

    savedir: string
        Location to save the plot.
    '''
    fig, ax = plt.subplots()
    ax.imshow(im, cmap='gray')
    for i in range(len(pixels)):
        ax.text(pixels[i][1], pixels[i][0], i+1, color='red',
                ha='center', va='center')

    ax.set_ylim((y-trim,y+trim))
    ax.set_xlim((x-trim,x+trim))
    fig.tight_layout()
    plt.savefig(savedir + eventname + '-pixels.png')
    plt.clf()
        
def models(fit, savedir='./'):
    '''
    '''
    phase      = fit.phase
    phat       = fit.phat
    modelfuncs = fit.modelfuncs
    modeltypes = fit.modeltypes
    npars      = fit.npars
    bp         = fit.bp
    
    nplots = 0
    for i in range(len(modelfuncs)):
        if modeltypes[i] == 'pld':
            nplots += npars[i]
            nplots += 1
        else:
            nplots += 1
            
    fig, axlist = plt.subplots(nrows=nplots, figsize=(16,8))

    parlist = []
    parind  = 0
    axind   = 0
    
    for i in range(len(modelfuncs)):
        parlist.append(bp[parind:parind+npars[i]])
        parind += npars[i]
        
    for i in range(len(modelfuncs)):
        y = modelfuncs[i](parlist[i], phase, phat)
        if modeltypes[i] == 'ecl/tr':
            y -= 1
        axlist[axind].plot(phase, y, label=modeltypes[i])
        axind += 1
        if modeltypes[i] == 'pld':
            for j in range(len(parlist[i])):
                axlist[axind].plot(phase, phat[:,j]*parlist[i][j],
                                   label='Pixel '+str(j+1))
                axind += 1

    for i in range(nplots):
        axlist[i].legend(loc='upper right', fontsize=6)
        if i == nplots-1:
            axlist[i].set_xlabel('Phase')
        else:
            axlist[i].set_xticks([])

    plt.subplots_adjust(hspace=0.1, left=0.05, right=0.95,
                        top=0.95, bottom=0.05)
    plt.savefig(savedir + fit.eventname + '-models.png')

    plt.clf()
                        


