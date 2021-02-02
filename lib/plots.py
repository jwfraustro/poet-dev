"""
Revisions:
----------
    2017-06-20  Zacchaeus  Fixed comparison to None.
"""
import numpy as np
import matplotlib.pyplot as plt
from pynfft import NFFT
import sigrej as sr
import os


#SET PLOTTING FORMAT
# Matplotlib only supports 8 different colors in format strings :(
# (we don't include white)
ebfmt   = ['bo',  'go',  'ro',  'co',  'mo',  'yo',  'ko' ]
pltfmt  = ['b-',  'g-',  'r-',  'c-',  'm-',  'y-',  'k-' ]
pltfmt2 = ['b--', 'g--', 'r--', 'c--', 'm--', 'y--', 'k--']


# Plot binned data and best fit
def binlc(event, fit, fignum, savefile=None, istitle=True, j=0):
    plt.rcParams.update({'legend.fontsize':13})
    plt.figure(fignum)
    plt.clf()
    if istitle:
        plt.suptitle(event.eventname + ' Binned Data With Best Fit',size=16)
        plt.title(fit.model, size=10)
    if fit.preclip > 0:
      plt.errorbar(fit.precbinphase, fit.binprecflux, fit.binprecstd, fmt='o',
                   color="0.7", ms=4, linewidth=1)
    if fit.postclip < fit.nobjuc:
      plt.errorbar(fit.postbinphase, fit.binpostflux, fit.binpoststd, fmt='o',
                   color="0.7", ms=4, linewidth=1)
    plt.errorbar(fit.abscissa, fit.binflux, fit.binstd, fmt='ko',
                 ms=4, linewidth=1, label='Binned Data')
    plt.plot(fit.abscissa, fit.binnoecl,   'k-',      lw=1, label='No Eclipse')
    plt.plot(fit.abscissa, fit.binbestfit, pltfmt[j%len(pltfmt)], lw=1, label='Best Fit')

    #plt.errorbar(fit.abscissauc, fit.binfluxuc, fit.binstduc, fmt='ko', 
    #                 ms=4, linewidth=1, label='Binned Data')
    #plt.plot(fit.abscissa, fit.binnoecl,   'k-',      label='No Eclipse')
    #plt.plot(fit.abscissa, fit.binbestfit, pltfmt[j%len(pltfmt)], label='Best Fit')
    plt.xticks(size=13)
    plt.yticks(size=13)
    plt.xlabel(fit.xlabel, size=14)
    plt.ylabel('Flux',     size=14)
    plt.legend(loc='best')
    if savefile is not None:
        plt.savefig(savefile)
    return

# Plot normalized data, best fit, and residuals
def normlc(event, fit, fignum, savefile=None, istitle=True, j=0, interclip=None):
    plt.rcParams.update({'legend.fontsize':13})
    plt.figure(fignum, figsize=(8,6))
    plt.clf()
    # Normalized subplot
    a = plt.axes([0.15,0.35,0.8,0.55])
    a.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.4f'))
    if istitle:
        plt.suptitle(event.eventname + ' Normalized Binned Data With Best Fit',
                     size=16)
        plt.title(fit.model, size=10)
    if fit.preclip > 0:
      plt.errorbar(fit.precbinphase, fit.precbinflux, fit.precbinstd, fmt='o', 
                   color='0.7', ms=4, lw=1)
    if fit.postclip < fit.nobjuc:
      plt.errorbar(fit.postbinphase, fit.postbinflux, fit.postbinstd, fmt='o',
                   color='0.7', ms=4, lw=1)

    plt.errorbar(fit.abscissa, fit.normbinflux, fit.normbinstd, fmt='ko',
                 ms=4, lw=1, label='Binned Data')
    plt.plot(fit.timeunit, fit.normbestfit, pltfmt[j%len(pltfmt)], label='Best Fit', lw=2)

    ymin, ymax = plt.ylim()
    if interclip is not None:
      for i in range(len(interclip)):
        plt.plot(fit.timeunituc   [interclip[i][0]:interclip[i][1]], 
                 fit.normbestfituc[interclip[i][0]:interclip[i][1]], 'w-', lw=2)
    plt.setp(a.get_xticklabels(), visible = False)
    plt.yticks(size=13)
    plt.ylabel('Normalized Flux',size=14)
    plt.ylim(ymin,ymax)
    plt.legend(loc='best')
    xmin, xmax = plt.xlim()

    # Residuals subplot
    plt.axes([0.15,0.1,0.8,0.2])
    flatline = np.zeros(len(fit.abscissa))
    plt.plot(fit.abscissa, fit.normbinresiduals, 'ko',ms=4)
    plt.plot(fit.abscissa, flatline,'k:',lw=1.5)
    plt.xlim(xmin,xmax)
    plt.xticks(size=13)
    plt.yticks(size=13)
    plt.xlabel(fit.xlabel,size=14)
    plt.ylabel('Residuals',size=14)
    if savefile is not None:
        plt.savefig(savefile)
    return
    
# Trace plots
def trace(event, fit, fignum, savefile=None, allparams=None, parname=None, 
          iparams=None, thinning=None, istitle=True):
    if thinning is None:
        thinning = event.params.thinning
    if allparams is None:
        allparams = fit.allparams
    if parname is None:
        parname   = fit.parname
    if iparams is None:
        nonfixedpars = fit.nonfixedpars
    else:
        nonfixedpars = range(allparams.shape[0])
        if parname is None:
            parname   = np.array(fit.parname)[iparams]
    plt.figure(fignum, figsize=(8,8))
    plt.clf()
    if istitle:
        plt.suptitle(event.eventname + ' Trace Plots',size=16)
    numfp     = len(nonfixedpars)
    plt.subplots_adjust(left=0.15,right=0.95,bottom=0.10,top=0.90,hspace=0.20,wspace=0.20)
    k = 1
    for i in nonfixedpars:
        a = plt.subplot(numfp,1,k)
        if parname[i].startswith('System Flux'):
            a.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.0f'))
        plt.plot(allparams[i],'.')
        s = parname[i].replace(',','\n')
        plt.ylabel(s,size=10, multialignment='center')
        plt.yticks(size=10)
        if i == nonfixedpars[-1]:
            plt.xticks(size=10)
            plt.xlabel('MCMC step number',size=10)
        else:
            plt.xticks(visible=False)
        k += 1
    
    if savefile is not None:
        plt.savefig(savefile)
    return

# Autocorrelation plots of trace values
def autocorr(event, fit, fignum, savefile=None, allparams=None, parname=None, iparams=None, thinning=None, istitle=True):
    if thinning is None:
        thinning = event.params.thinning
    if allparams is None:
        allparams = fit.allparams
    if parname is None:
        parname   = fit.parname
    if iparams is None:
        nonfixedpars = fit.nonfixedpars
    else:
        nonfixedpars = range(allparams.shape[0])
        if parname is None:
            parname   = np.array(fit.parname)[iparams]
    plt.figure(fignum, figsize=(8,8))
    plt.clf()
    if istitle:
        plt.suptitle(event.eventname + ' Autocorrelation Plots',size=16)
    numfp     = len(nonfixedpars)
    plt.subplots_adjust(left=0.15,right=0.95,bottom=0.10,top=0.90,hspace=0.20,wspace=0.20)
    k = 0
    for i in nonfixedpars:
        a = plt.subplot(numfp,1,k+1)
        plt.plot(fit.autocorr[k],'-', lw=2)
        line = np.zeros(len(fit.autocorr[k]))
        plt.plot(line, 'k-')
        s = parname[i].replace(',','\n')
        plt.ylabel(s,size=10, multialignment='center')
        plt.yticks(size=10)
        if i == nonfixedpars[-1]:
            plt.xticks(size=10)
            plt.xlabel('MCMC step number (up to first 10,000)',size=10)
        else:
            plt.xticks(visible=False)
        k += 1
    
    if savefile is not None:
        plt.savefig(savefile)
    return

# Correlation plots with 2D histograms
def hist2d(event, fit, fignum, savefile=None, allparams=None, parname=None, iparams=None, thinning=None, istitle=True):
    if thinning is None:
        thinning = event.params.thinning
    if allparams is None:
        allparams = fit.allparams
    if parname is None:
        parname   = fit.parname
    if iparams is None:
        nonfixedpars = fit.nonfixedpars
    else:
        nonfixedpars = range(allparams.shape[0])
        parname   = np.array(parname)[iparams]
    #palette = plt.matplotlib.colors.LinearSegmentedColormap('jet2',plt.cm.datad['jet'],65536)
    palette = plt.matplotlib.colors.LinearSegmentedColormap.from_list('YlOrRd2',plt.cm.datad['YlOrRd'],65536)
    #palette = plt.cm.YlOrRd
    #palette = plt.cm.RdBu_r
    palette.set_under(color='w')
    plt.figure(fignum, figsize=(8,8))
    plt.clf()
    if istitle:
        plt.suptitle(event.eventname + ' Correlation Plots with 2D Histograms',size=16)
    numfp     = len(nonfixedpars)
    paramcorr = np.corrcoef(allparams)
    h     = 1
    m     = 1
    plt.subplots_adjust(left=0.15,right=0.95,bottom=0.15,top=0.95,hspace=0.15,wspace=0.15)
    for i in nonfixedpars[1:numfp]:
        n     = 0
        for k in nonfixedpars[0:numfp-1]:
            if i > k:
                a = plt.subplot(numfp-1,numfp-1,h)
                #a.set_axis_bgcolor(plt.cm.YlOrRd(np.abs(paramcorr[m,n])))
                if parname[i].startswith('System Flux'):
                    a.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.0f'))
                if parname[k].startswith('System Flux'):
                    a.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.0f'))
                if k == nonfixedpars[0]:
                    plt.yticks(size=11)
                    s = parname[i].replace(',','\n')
                    plt.ylabel(s, size = 12, multialignment='center')
                else:
                    a = plt.yticks(visible=False)
                if i == nonfixedpars[numfp-1]:
                    plt.xticks(size=11,rotation=90)
                    s = parname[k].replace(',','\n')
                    plt.xlabel(s, size = 12)
                else:
                    a = plt.xticks(visible=False)
                hist2d, xedges, yedges = np.histogram2d(allparams[k,0::thinning],
                                                        allparams[i,0::thinning],20,density=True)
                vmin = np.min(hist2d[np.where(hist2d > 0)])
                #largerhist = np.zeros((22,22))
                #largerhist[1:-1,1:-1] = hist2d
                a = plt.imshow(hist2d.T,extent=(xedges[0],xedges[-1],yedges[0],yedges[-1]), cmap=palette, 
                               vmin=vmin, aspect='auto', origin='lower') #,interpolation='bicubic')
            h += 1
            n +=1
        m +=1
    
    if numfp > 2:
        a = plt.subplot(numfp-1, numfp-1, numfp-1, frameon=False)
        a.yaxis.set_visible(False)
        a.xaxis.set_visible(False)
        a = plt.imshow([[0,1],[0,0]], cmap=plt.cm.YlOrRd, visible=False)
        a = plt.text(1.4, 0.5, 'Normalized Point Density', rotation='vertical', ha='center', va='center')
        a = plt.colorbar()
    else:
        a = plt.colorbar()
    if savefile is not None:
        plt.savefig(savefile)
    return

# 1D histogram plots
def histograms(event, fit, fignum, savefile=None, allparams=None, 
               parname=None, iparams=None, thinning=None, istitle=True):
  if thinning is None:
    thinning = event.params.thinning
  if allparams is None:
    allparams = fit.allparams
  if parname is None:
    parname   = fit.parname
  if iparams is None:
    nonfixedpars = fit.nonfixedpars
  else:
    nonfixedpars = range(allparams.shape[0])
    if parname is None:
      parname   = np.array(fit.parname)[iparams]
  j          = 1
  numfp      = len(nonfixedpars)
  histheight = np.min((int(4*np.ceil(numfp/3.)),8))
  if histheight == 4:
    bottom = 0.23
    hspace = 0.40
  elif histheight == 8:
    bottom = 0.13
    hspace = 0.40
  else:
    bottom = 0.12
    hspace = 0.65
  plt.figure(fignum, figsize=(8,histheight))
  plt.clf()
  if istitle:
    a = plt.suptitle(event.eventname + ' Histograms', size=16)
  for i in nonfixedpars:
    a = plt.subplot(np.ceil(numfp/3.),3,j)
    if parname[i].startswith('System Flux'):
      a.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.0f'))
    plt.xticks(size=12,rotation=90)
    plt.yticks(size=12)
    #plt.axvline(x=fit.meanp[i,0])
    plt.xlabel(parname[i], size=14)
    a  = plt.hist(allparams[i,0::thinning], 20, density=False, label=str(fit.meanp[i]))
    j += 1
  plt.subplots_adjust(left=0.07,right=0.95,bottom=bottom,top=0.95,hspace=hspace,wspace=0.25)
  if savefile is not None:
    plt.savefig(savefile)
  return

# Projections of position sensitivity along x and y
def ipprojections(event, fit, fignum, savefile=None, istitle=True):
    plt.rcParams.update({'legend.fontsize':11})
    plt.figure(fignum, figsize=(8,4))
    plt.clf()
    if istitle:
        a = plt.suptitle(event.eventname + ' Projections of Position Sensitivity', size=16)
    yround = fit.yuc[0] - fit.y[0]
    xround = fit.xuc[0] - fit.x[0]
    plt.subplot(1,2,1)
    plt.errorbar(yround+fit.binyy, fit.binyflux, fit.binyflstd, fmt='ro', label='Binned Flux',
                 zorder=1)
    plt.plot(yround+fit.binyy, fit.binybestip, 'k-', lw=2, label='BLISS Map', zorder=2)
    plt.xlabel('Pixel Postion in y', size=14)
    plt.ylabel('Normalized Flux', size=14)
    plt.xticks(rotation=90)
    plt.legend(loc='best')
    plt.subplot(1,2,2)
    plt.errorbar(xround+fit.binxx, fit.binxflux, fit.binxflstd, fmt='bo', label='Binned Flux',
                 zorder=1)
    plt.plot(xround+fit.binxx, fit.binxbestip, 'k-', lw=2, label='BLISS Map', zorder=2)
    plt.xlabel('Pixel Postion in x', size=14)
    plt.xticks(rotation=90)
    #a = plt.ylabel('Normalized Flux', size=14)
    a = plt.legend(loc='best')
    plt.subplots_adjust(left=0.11,right=0.97,bottom=0.20,top=0.90,wspace=0.20)
    if savefile is not None:
        plt.savefig(savefile)
    return

# BLISS map
def blissmap(event, fit, fignum, savefile=None, istitle=True, minnumpts=1):
    #palette   = plt.matplotlib.colors.LinearSegmentedColormap('jet3',plt.cm.datad['jet'],16384)
    palette = plt.cm.RdBu_r
    palette.set_under(alpha=0.0, color='w')
    # Determine size of non-zero region
    vmin = fit.binipflux[np.where(fit.binipflux > 0)].min()
    vmax = fit.binipflux.max()
    yround = fit.yuc[0] - fit.y[0]
    xround = fit.xuc[0] - fit.x[0]
    xmin = fit.xygrid[0][np.where(fit.numpts>=minnumpts)].min() + xround
    xmax = fit.xygrid[0][np.where(fit.numpts>=minnumpts)].max() + xround
    ymin = fit.xygrid[1][np.where(fit.numpts>=minnumpts)].min() + yround
    ymax = fit.xygrid[1][np.where(fit.numpts>=minnumpts)].max() + yround
    ixmin = np.where(fit.xygrid[0] + xround == xmin)[1][0]
    ixmax = np.where(fit.xygrid[0] + xround == xmax)[1][0]
    iymin = np.where(fit.xygrid[1] + yround == ymin)[0][0]
    iymax = np.where(fit.xygrid[1] + yround == ymax)[0][0]
    # Plot
    plt.figure(fignum, figsize=(8,6))
    plt.clf()
    if istitle:
        a = plt.suptitle(event.eventname + ' BLISS Map', size=16)
    if fit.model.__contains__('nnint'):
        interp = 'nearest'
    else:
        interp = 'bilinear'
    #MAP
    a = plt.axes([0.11,0.10,0.75,0.80])
    plt.imshow(fit.binipflux[iymin:iymax+1,ixmin:ixmax+1], cmap=palette, vmin=vmin, vmax=vmax, origin='lower', 
               extent=(xmin,xmax,ymin,ymax), aspect='auto', interpolation=interp)
    plt.ylabel('Pixel Position in y', size=14)
    plt.xlabel('Pixel Position in x', size=14)
    if ymin < -0.5+yround:
        plt.hlines(-0.5+yround, xmin, xmax, 'k')
    if ymax >  0.5+yround:
        plt.hlines( 0.5+yround, xmin, xmax, 'k')
    if xmin < -0.5+xround:
        plt.vlines(-0.5+xround, ymin, ymax, 'k')
    if xmax >  0.5+xround:
        plt.vlines( 0.5+xround, ymin, ymax, 'k')
    #COLORBAR
    a = plt.axes([0.90,0.10,0.01,0.8], frameon=False)
    a.yaxis.set_visible(False)
    a.xaxis.set_visible(False)
    a = plt.imshow([[vmin,vmax],[vmin,vmax]], cmap=palette, aspect='auto', visible=False)
    plt.colorbar(a, fraction=3.0)
    if savefile is not None:
        plt.savefig(savefile)
    return


# Pointing Histogram
def pointingHist(event, fit, fignum, savefile=None, istitle=True, minnumpts=1):
    #palette   = plt.matplotlib.colors.LinearSegmentedColormap('jet3',plt.cm.datad['jet'],16384)
    palette = plt.matplotlib.colors.LinearSegmentedColormap.from_list('YlOrRd2',plt.cm.datad['YlOrRd'],65536)
    palette.set_under(alpha=0.0, color='w')
    # Determine size of non-zero region
    vmin = 1
    vmax = fit.numpts.max()
    yround = fit.yuc[0] - fit.y[0]
    xround = fit.xuc[0] - fit.x[0]
    xmin = fit.xygrid[0][np.where(fit.numpts>=minnumpts)].min() + xround
    xmax = fit.xygrid[0][np.where(fit.numpts>=minnumpts)].max() + xround
    ymin = fit.xygrid[1][np.where(fit.numpts>=minnumpts)].min() + yround
    ymax = fit.xygrid[1][np.where(fit.numpts>=minnumpts)].max() + yround
    ixmin = np.where(fit.xygrid[0] + xround == xmin)[1][0]
    ixmax = np.where(fit.xygrid[0] + xround == xmax)[1][0]
    iymin = np.where(fit.xygrid[1] + yround == ymin)[0][0]
    iymax = np.where(fit.xygrid[1] + yround == ymax)[0][0]
    # Plot
    plt.figure(fignum, figsize=(8,6))
    plt.clf()
    if istitle:
        a = plt.suptitle(event.eventname + ' Pointing Histogram', size=16)
    if fit.model.__contains__('nnint'):
        interp = 'nearest'
    else:
        interp = 'bilinear'
    #MAP
    a = plt.axes([0.11,0.10,0.75,0.80])
    plt.imshow(fit.numpts[iymin:iymax+1,ixmin:ixmax+1], cmap=palette, vmin=vmin, vmax=vmax,
               origin='lower', extent=(xmin,xmax,ymin,ymax), aspect='auto', interpolation=interp)
    plt.ylabel('Pixel Position in y', size=14)
    plt.xlabel('Pixel Position in x', size=14)
    if ymin < -0.5+yround:
        plt.hlines(-0.5+yround, xmin, xmax, 'k')
    if ymax >  0.5+yround:
        plt.hlines( 0.5+yround, xmin, xmax, 'k')
    if xmin < -0.5+xround:
        plt.vlines(-0.5+xround, ymin, ymax, 'k')
    if xmax >  0.5+xround:
        plt.vlines( 0.5+xround, ymin, ymax, 'k')
    #COLORBAR
    a = plt.axes([0.90,0.10,0.01,0.8], frameon=False)
    a.yaxis.set_visible(False)
    a.xaxis.set_visible(False)
    a = plt.imshow([[vmin,vmax],[vmin,vmax]], cmap=palette, aspect='auto', visible=False)
    plt.colorbar(a, fraction=3.0)
    if savefile is not None:
        plt.savefig(savefile)
    return

# Plot RMS vs. bin size looking for time-correlated noise
def rmsplot(event, fit, fignum, savefile=None, istitle=True, binstep = 5, stderr=None, normfactor=None):
    idur = fit.bestp[fit.i.t12]   * event.period * 86400  # ingress duration in seconds
    edur = fit.bestp[fit.i.width] * event.period * 86400  # eclipse duration in seconds
    if stderr is None:
        stderr = fit.stderr
    if normfactor is None:
        normfactor = stderr[0]
    plt.rcParams.update({'legend.fontsize':11})
    plt.figure(fignum, figsize=(8,6))
    plt.clf()
    if istitle:
        a = plt.suptitle(event.eventname + ' Correlated Noise', size=16)
    plt.errorbar(fit.binsz[::binstep]*event.framtime, fit.rms[::binstep], 
                 fit.rmserr[::binstep], fmt='k-', ecolor='black', lw=1, 
                 capsize=0, label='Fit RMS')
    plt.loglog(fit.binsz[::1]*event.framtime, fit.stderr[::1], color='red', 
               ls='-', lw=2, label='Std. Err.')
    plt.ylim(fit.rms[-1]/2., fit.rms[0]*2.0)
    ymin, ymax = plt.ylim()
    plt.vlines(idur, ymin, ymax, 'b', 'dotted', lw=2)  #Ingress duration
    plt.vlines(edur, ymin, ymax, 'g', 'dashed', lw=2)  #eclipse duration
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.legend(loc='upper right')
    plt.xlabel("Bin Size (sec)", fontsize=14)
    plt.ylabel("Normalized RMS", fontsize=14)
    if savefile is not None:
        plt.savefig(savefile)
    return

# Plot RMS vs. bin size (with RMS uncertainties) looking for time-correlated noise
def rmsploterr(event, fit, fignum, savefile=None, istitle=True, stderr=None, normfactor=None):
    if stderr is None:
        stderr = fit.stderr
    if normfactor is None:
        normfactor = stderr[0]
    plt.rcParams.update({'legend.fontsize':11})
    plt.figure(fignum, figsize=(8,6))
    plt.clf()
    if istitle:
        a = plt.suptitle(event.eventname + ' Correlated Noise', size=16)
    plt.loglog(fit.binsz, fit.rms/normfactor, color='black', lw=1.5, label='Fit RMS')    # our noise
    plt.loglog(fit.binsz, stderr/normfactor, color='red', ls='-', lw=2, label='Std. Err.') # expected noise
    plt.xlim(0, fit.binsz[-1]*2)
    plt.ylim(stderr[-1]/normfactor/2., stderr[0]/normfactor*2.)
    plt.xlabel("Bin Size", fontsize=14)
    plt.ylabel("Normalized RMS", fontsize=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.legend()
    if savefile is not None:
        plt.savefig(savefile)
    return

def chainsnlc(normfit, normbinflux, normbinsd, fit, fignum, 
              savefile=None, istitle=True, j=0, interclip=None):
  """
  Plot chains end normalized lightcurves:
  """
  plt.rcParams.update({'legend.fontsize':13})
  plt.figure(fignum, figsize=(8,10))
  plt.clf()

  # Normalized subplot
  numplots = len(normfit)
  ncolumns = 3
  nrows = int(np.ceil(1.0 * numplots / ncolumns))
  for c in np.arange(numplots):
    a = plt.subplot(nrows, ncolumns, c+1)
    a.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.4f'))
    if istitle:
      plt.title('Chain %2d'%(c+1), size=10)

    plt.errorbar(fit.abscissauc, normbinflux[c], normbinsd[c], fmt='ko',
                 ms=4, lw=1)
    plt.plot(fit.timeunit, normfit[c], pltfmt[j%len(pltfmt)], lw=2)
    if interclip is not None:
      for i in np.arange(len(interclip)):
        plt.plot(fit.tuall[interclip[i][0]:interclip[i][1]], 
                   np.ones(interclip[i][1]-interclip[i][0]), '-w', lw=3)

        plt.plot(fit.timeunituc   [interclip[i][0]:interclip[i][1]],
                 fit.normbestfituc[interclip[i][0]:interclip[i][1]], 'w-', lw=2)

    # plt.setp(a.get_xticklabels(), visible = False)
    plt.yticks(size=13)
    if (c+1) % ncolumns == 1:
      plt.ylabel('Normalized Flux', size=10)
    else:
      a.set_yticklabels([''])
    if numplots - (c+1) < ncolumns:
      plt.xlabel(fit.xlabel, size=10)
    else:
      a.set_xticklabels([''])

  plt.subplots_adjust(hspace=0.2, wspace=0.1,right=0.99)
  plt.suptitle('Last Burn-in Iteration Models',size=16)
  if savefile is not None:
    plt.savefig(savefile)

  return

def burntraces(event, fit, fignum, savefile=None, allparams=None, parname=None,
               iparams=None, thinning=1, istitle=True):
  if allparams is None:
    allparams = fit.allparams
  if parname is None:
    parname   = fit.parname
  if iparams is None:
    nonfixedpars = fit.nonfixedpars
  else:
    nonfixedpars = range(allparams.shape[1])
    if parname is None:
      parname   = np.array(fit.parname)[iparams]

  nchains = event.params.nchains # number of chains 
  chainlen = allparams.shape[2]  # length of chains

  plt.figure(fignum, figsize=(8,8))
  plt.clf()
  if istitle:
    plt.suptitle(event.eventname + ' Trace Plots',size=16)
  numfp = len(nonfixedpars)
  k = 1
  for i in nonfixedpars:
    a = plt.subplot(numfp, 1, k)
    if parname[i].startswith('System Flux'):
      a.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%0.0f'))
    s = parname[i].replace(',','\n')
    plt.ylabel(s, size=10, multialignment='center')
    for c in np.arange(nchains):
      plt.plot(np.arange(0, chainlen, thinning), allparams[c,i,::thinning])
    plt.yticks(size=10)
    if i == nonfixedpars[-1]:
      plt.xticks(size=10)
      plt.xlabel('MCMC burn-in iteration', size=10)
    else:
      plt.xticks(visible=False)
    k += 1

  plt.subplots_adjust(left=0.15,   right=0.95, bottom=0.10, top=0.90,
                      hspace=0.20, wspace=0.20)
  if savefile is not None:
    plt.savefig(savefile)
  return

def fourier(event, fit, fignum, savefile):
    x = fit.timeunit
    y = fit.residuals

    shift1 = x[0]
    newx = x - shift1

    stretch1 = 1/x[-1]
    newx *= stretch1

    shift2 = 0.5
    newx -= shift2

    M = len(y)
    N = M//2

    nfft_obj = NFFT(N, M=M)
    nfft_obj.x = newx
    nfft_obj.precompute()

    nfft_obj.f = y

    nfft_obj.adjoint()

    power = np.zeros(len(nfft_obj.f_hat))
    for i in range(len(nfft_obj.f_hat)):
        power[i] = np.abs(nfft_obj.f_hat[i])**2

    k = np.arange(-N//2, N//2, 1)
    
    plt.clf()
    plt.plot(k, power)
    plt.xlabel('Frequency (periods/dataset)')
    plt.ylabel('Power')
    plt.title('Power spectrum of Fourier transform of residuals')
    plt.xlim((0,M//500))
    #plt.ylim((0,max(power[N/2:N/2+M/500])))
    if savefile != None:
        plt.savefig(savefile)
    return

def makep6plots(lines, plotdir):
    names = lines.keys()

    sigiter = (4, 4)

    sdnr  = np.zeros(len(names))
    bic   = np.zeros(len(names))
    depth = np.zeros(len(names))
    derr  = np.zeros(len(names))
    bsig  = np.zeros(len(names))
    method = []

    methoddtype = [('cent', 'S10'),
                   ('prefix', 'S10'),
                   ('scale', float),
                   ('offset', float),
                   ('model', 'S10')]
    
    for i, name in enumerate(lines):
        sdnr[i]  = lines[name][0][4]
        bic[i]   = lines[name][0][5]
        depth[i] = lines[name][0][6]
        derr[i]  = lines[name][0][7]
        bsig[i]  = lines[name][0][8]
        method.append((lines[name][0][0],
                       lines[name][0][1],
                       lines[name][0][2],
                       lines[name][0][3],
                       lines[name][0][9]))

    method = np.array(method, dtype=methoddtype)

    methodsortind = np.argsort(method, order=['cent', 'prefix', 'scale',
                                              'offset', 'model'])

    sdnrsort  = sdnr [methodsortind]
    bicsort   = bic  [methodsortind]
    depthsort = depth[methodsortind]
    derrsort  = derr [methodsortind]
    bsigsort  = bsig [methodsortind]

    bestind = np.where(bsigsort == np.min(bsigsort))
    sdnrind = np.where(sdnrsort == np.min(sdnrsort))

    namesort  = np.array(list(names))[methodsortind]

    print("Lowest binned-sigma chisq: {}".format(namesort[bestind]))
    print("Lowest SDNR:               {}".format(namesort[sdnrind]))

    textx = 0.85
    texty = 0.95

    fig = plt.figure(figsize=(20,8))
    msk = sr.sigrej(depthsort, sigiter)
    nrej = depthsort.size - np.sum(msk)
    plt.errorbar(np.arange(len(names))[msk], depthsort[msk],
                 derrsort[msk], fmt='ko')
    plt.xticks(np.arange(len(names)), namesort, rotation='vertical', fontsize=6)
    plt.errorbar(np.arange(len(names))[bestind],
                 depthsort[bestind],
                 derrsort[bestind],
                 fmt='ro')                       
    plt.ylabel("Depth (%)")
    plt.text(textx, texty, "Note: {} outliers hidden".format(nrej),
             transform=plt.gca().transAxes)
    plt.savefig(plotdir + "/depth.pdf", bbox_inches='tight')
    plt.savefig(plotdir + "/depth.png", bbox_inches="tight")
    plt.savefig(plotdir + "/depth.ps",  transparent=True)
    plt.close()

    fig = plt.figure(figsize=(20,8))
    msk = sr.sigrej(sdnrsort, sigiter)
    nrej = depthsort.size - np.sum(msk)
    plt.scatter(np.arange(len(names))[msk], sdnrsort[msk], c='black')
    plt.xticks(np.arange(len(names)), namesort, rotation='vertical', fontsize=6)
    plt.scatter(np.arange(len(names))[bestind],
                sdnrsort[bestind],
                c='red')
    plt.ylabel("SDNR")
    plt.text(textx, texty, "Note: {} outliers hidden".format(nrej),
             transform=plt.gca().transAxes)
    plt.savefig(plotdir + "/sdnr.pdf", bbox_inches='tight')
    plt.savefig(plotdir + "/sdnr.png", bbox_inches='tight')
    plt.savefig(plotdir + "/sdnr.ps",  transparent=True)
    plt.close()

    fig = plt.figure(figsize=(20,8))
    msk = sr.sigrej(bicsort, sigiter)
    nrej = depthsort.size - np.sum(msk)
    plt.scatter(np.arange(len(names))[msk], bicsort[msk], c='black')
    plt.xticks(np.arange(len(names)), namesort, rotation='vertical', fontsize=6)
    plt.scatter(np.arange(len(names))[bestind],
                bicsort[bestind],
                c='red')
    plt.ylabel("BIC")
    plt.text(textx, texty, "Note: {} outliers hidden".format(nrej),
             transform=plt.gca().transAxes)
    plt.savefig(plotdir + "/BIC.pdf", bbox_inches='tight')
    plt.savefig(plotdir + "/BIC.png", bbox_inches='tight')
    plt.savefig(plotdir + "/BIC.ps",  transparent=True)
    plt.close()

    fig = plt.figure(figsize=(20,8))
    msk = sr.sigrej(bsigsort, sigiter)
    nrej = depthsort.size - np.sum(msk)    
    plt.scatter(np.arange(len(names))[msk], bsigsort[msk], c='black')
    plt.xticks(np.arange(len(names)), namesort, rotation='vertical', fontsize=6)
    plt.scatter(np.arange(len(names))[bestind],
                bsigsort[bestind],
                c='red')
    plt.ylabel(r"$\chi^2_{bin}$")
    plt.text(textx, texty, "Note: {} outliers hidden".format(nrej),
             transform=plt.gca().transAxes)   
    plt.savefig(plotdir + "/bsig.pdf", bbox_inches='tight')
    plt.savefig(plotdir + "/bsig.png", bbox_inches='tight')
    plt.savefig(plotdir + "/bsig.ps",  transparent=True)
    plt.close()

def getinfo(file):
  """
  Extract info from results.txt file.

  Parameters:
  -----------
  file: String
        Path to file. This must be a file output from p6model.

  Results:
  --------
  model: List
        The name of the models in file.
  sdnr: List
       SDNR values for each model.
  bic: List
       BIC values for each model.
  depth: List
       Eclipse depth for each model. 
  deptherr: List
       Uncertainty in the eclipse depth for each model.
  nmodels: Integer
       Number of models in the file.

  Modification History:
  ---------------------
  2013-02-08  patricio  Initial implementation.   pcubillos@fulbrightmail.org
  """
  # Read file:
  data = open(file, "r")
  lines = data.readlines()
  data.close()
  # Count number of models fitted:
  nmodels = lines.count("Current event & model:\n")
  # Output info:
  bic, sdnr, model, depth, deptherr, bsigchi = [], [], [], [], [], []
  # Get BIC values:
  for line in lines:
    if line.startswith("BIC ="):
      bic.append( float(line.strip().split()[-1]))
  # Get depths, errors, and model:
  lines.reverse()
  index = len(lines) - 1 - lines.index(
          "Best-fit eclipse depths or transit radius ratios with errors:\n")
  lines.reverse()
  depthline = 2 + index
  for m in np.arange(nmodels):
    line = lines[depthline + m].strip().split()
    if len(line) == 0:
        break
    depth.   append(float(line[0]))
    deptherr.append(float(line[1]))
    model.   append(      line[2] )
    bsigchi. append(float(line[3]))
  nmodels = len(model)
  # Get SDNR, read the last m lines in the file:
  for m in np.arange(nmodels):
    try:
      sdnr.append(float(lines[m - nmodels].split()[1]) )
    except:
      print("Error reading from results while making p6 plots.")
  return model, sdnr, bic, depth, deptherr, bsigchi #, nmodels

def p6plots(modeldir, paths, directory, fontsize=14):

    lines = {}
    info  = {}

    # Model names dictionary:
    model_name = {
        "m1bli":"No ramp",         "m1":"No ramp",
        "m1afbli":"AOR",           "m1af":"AOR",
        "m1lnafbli":"AOR-Linear",  "m1lnaf":"AOR-Linear",
        "m1lgbli":"Logramp",       "m1lg":"Logramp",
        "m1sebli":"Rising exp",    "m1se":"Rising exp",
        "m1qdbli":"Quadratic",     "m1qd":"Quadratic",
        "m1llbli":"LogLinear",     "m1ll":"LogLinear",
        "m1lnbli":"Linear"}

    print(paths)

    for path in paths:

        path += '/' + modeldir
        path = os.path.abspath(path)

        if not os.path.isdir(path):
            print('Unable to retrieve results from', path)
            continue

        pathdir = os.listdir(path)

        pointfile = None
        for file in pathdir:
            if file == "plotpoint.txt":
                pointfile = file
            if file.startswith("results") and file.endswith('.txt'):
                resfile = file

        try:
            model, sdnr, bic, depth, derr, bsigchi = getinfo(path + '/' + resfile)
        except Exception as E:
            print("Unable to read output from: " + path + '/' + resfile + ". p6"
                  " probably crashed here.", flush=True)
            raise RuntimeError("Did p6 fully run? (look for EARLIEST tracebac"
                    "k.) Note that p6 won't run in an existing dir"
                    "ectory in order to not accidentally clobber dat"
                    "a.") from E

        centdir, photdir = path.split('/')[-3:-1]

        if centdir.find('_') != -1:
            cent = centdir[:centdir.index('_')]
        else:
            cent = centdir

        if photdir.startswith('ap'):
            aptype = 'ap'
            aper = float(photdir[2:5]) / 100
            off  = float(photdir[5:9]) / 100

        elif photdir.startswith('va'):
            aptype = 'va'
            aper = float(photdir[2:5]) / 100
            off  = float(photdir[5:9]) / 100
            
        elif photdir.startswith('el'):
            aptype = 'el'
            aper = float(photdir[2:5]) / 100
            off  = float(photdir[5:9]) / 100            

        elif photdir == 'optimal':
            aptype = 'optimal'
            aper = 0.0
            off =  0.0

        elif photdir == 'psffit':
            aptype = 'psffit'
            aper =  0.0
            off =   0.0

        else:
            print("unable to identify photometry from folder name"
                  "for path:", path)

        for i in range(len(model)):
            try:
                line = '_'.join([cent,
                                 aptype,
                                 str(aper)+["","+"][off>=0]+str(off),
                                 model_name[model[i]]])
            except:
                line = '_'.join([cent,
                                 aptype,
                                 str(aper)+["","+"][off>=0]+str(off),
                                 model[i]])
                print("Model:", model[i], "has no map in dict model_name"
                      " in func p6plots in file plots.py")
            if not line in lines.keys():
                lines[line] = []
            lines[line].append([cent, aptype, aper, off,
                                sdnr[i], bic[i],
                                depth[i], derr[i],
                                bsigchi[i], model[i]])

            if not line in info.keys():
                info[line] = {}
            info[line][off] = {'model': model[i], 'centdir': centdir,
                          "photdir": photdir}

    #convert dictionary entries from lists to numpy arrays
    for line in lines.keys():
        data = np.array(lines[line])
        lines[line] = data[data.argsort(0)[:, 1]]

    # create plot directory.
    plotdir = directory + '/plots'
    if not os.path.isdir(plotdir):
        os.mkdir(plotdir)

    # create plot subdirectory
    plotdir += "/" + modeldir
    if os.path.isdir(plotdir):
        print("plot directory already exists. Writing over plots...")
    else:
        os.mkdir(plotdir)
        
    makep6plots(lines, plotdir)

    return lines, info
