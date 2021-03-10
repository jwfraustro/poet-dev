#! /usr/bin/env python
"""
 MODIFICATION HISTORY:
    Written by:	Kevin Stevenson, UCF  	2008-07-02
                kevin218@knights.ucf.edu
    Finished initial version:            kevin   2008-09-08
    Updated for multi events:            kevin   2009-11-01
    Added ip interpolation:              kevin   2010-06-28
    Added x,y precision calc:            kevin   2010-07-02
    Switched pyfits to astropy.io.fits:  garland 2014-08-13
                                         zabblleon@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import time, os
import pickle
import readeventhdf
import manageevent as me
import suntimecorr as stc
import time2phase2 as tp
import logedit     as le
import bindata as bd
import utc_tt


def checks2(filename, num):
    numfigs = 10

    print('MARK: ' + time.ctime() + ' : Starting Checks')

    #RESTORE HDF5 SAVE FILES FROM IDL
    event = readeventhdf.ReadEventHDF(filename)
    obj   = event.eventname

    print("Current event     = " + obj)
    print("Kurucz file       = " + event.kuruczfile)
    print("Filter file       = " + event.filtfile)

    #IF USING OPTIMAL PHOTOMETRY, MOVE OPHOTLEV TO APLEV
    try:
        event.good  = event.ogood
        event.aplev = event.ophotlev
        event.aperr = event.ophoterr
        print('Photometry method = OPTIMAL')
    except:
        print('Photometry method = APERTURE')

    #Number of good frames should be > 95%
    print("Good Frames [%]   = " + str(np.mean(event.good)*100))
	
    #VERIFY LIGHT-TIME CORRECTION
    #File could be located in /home/jh/ast/esp01/data/ or /home/esp01/data/
    hfile = event.dpref + event.aorname[0] + event.bcddir + event.fpref +           \
            event.aorname[0] + '_' + str(int(event.expid.flat[0])).zfill(4) + '_' + \
            str(int(event.dce.flat[0])).zfill(4) + event.pipev + event.bcdsuf
    try:
        image, event.header = fits.getdata(hfile, header=True)
        print('Light-time correction: ' + str(event.bjdcor.flat[0]) + ' = ' + \
              str((event.header['HMJD_OBS'] - event.header['MJD_OBS'])*86400))
    except:
        print('Could not verify light-time correction.')

    print('Mean, std dev of x center = ' + str(np.mean(event.x)) + ' +/- ' + str(np.std(event.x)))
    print('Mean, std dev of y center = ' + str(np.mean(event.y)) + ' +/- ' + str(np.std(event.y)))
    event.xprecision = [np.mean(np.abs(event.x[:-1] - event.x[1:])), np.std (np.abs(event.x[:-1] - event.x[1:]))]
    event.yprecision = [np.mean(np.abs(event.y[:-1] - event.y[1:])), np.std (np.abs(event.y[:-1] - event.y[1:]))]
    print('Mean, std dev of x precision = ' + 
                    str(np.round(event.xprecision[0],4)) +
          ' +/- ' + str(np.round(event.xprecision[1],4)) + ' pixels.')
    print('Mean, std dev of y precision = ' + 
                    str(np.round(event.yprecision[0],4)) +
          ' +/- ' + str(np.round(event.yprecision[1],4)) + ' pixels.')
    print('Center & photometry aperture sizes = ' + str(event.centap) + ', ' + str(event.photap) + ' pixels.')
    print('Period = ' + str(event.period) + ' +/- ' + str(event.perioderr) + ' days')
    print('Ephemeris = ' + str(event.ephtime) + ' +/- ' + str(event.ephtimeerr) + ' JD')
	
    #CHOOSE ONLY GOOD FRAMES FOR PLOTTING
    phase = event.phase[np.where(event.good == 1)]
    aplev = event.aplev[np.where(event.good == 1)]

    #COMPUTE X AND Y PIXEL LOCATION RELATIVE TO...
    if event.npos > 1:
        #CENTER OF EACH PIXEL
        y = (event.y - np.round(event.y))[np.where(event.good == 1)]
        x = (event.x - np.round(event.x))[np.where(event.good == 1)]
    else:
        #CENTER OF MEDIAN PIXEL
        y = (event.y - np.round(np.median(event.y)))[np.where(event.good == 1)]
        x = (event.x - np.round(np.median(event.x)))[np.where(event.good == 1)]

    #SORT aplev BY x, y AND radial POSITIONS
    rad    = np.sqrt(x**2 + y**2)
    xx     = np.sort(x)
    yy     = np.sort(y)
    rr     = np.sort(rad)
    xaplev = aplev[np.argsort(x)]
    yaplev = aplev[np.argsort(y)]
    raplev = aplev[np.argsort(rad)]

    #BIN RESULTS FOR PLOTTING POSITION SENSITIVITY EFFECT
    nobj      = aplev.size
    nbins     = 120
    binxx     = np.zeros(nbins)
    binyy     = np.zeros(nbins)
    binrr     = np.zeros(nbins)
    binxaplev = np.zeros(nbins)
    binyaplev = np.zeros(nbins)
    binraplev = np.zeros(nbins)
    binxapstd = np.zeros(nbins)
    binyapstd = np.zeros(nbins)
    binrapstd = np.zeros(nbins)
    binphase  = np.zeros(nbins)
    binaplev  = np.zeros(nbins)
    binapstd  = np.zeros(nbins)
    for i in range(nbins):
        start        = int(1.*i*nobj/nbins)
        end          = int(1.*(i+1)*nobj/nbins)
        binxx[i]     = np.mean(xx[start:end])
        binyy[i]     = np.mean(yy[start:end])
        binrr[i]     = np.mean(rr[start:end])
        binxaplev[i] = np.median(xaplev[start:end])
        binyaplev[i] = np.median(yaplev[start:end])
        binraplev[i] = np.median(raplev[start:end])
        binxapstd[i] = np.std(xaplev[start:end]) / np.sqrt(end-start)
        binyapstd[i] = np.std(yaplev[start:end]) / np.sqrt(end-start)
        binrapstd[i] = np.std(raplev[start:end]) / np.sqrt(end-start)
        binphase[i]  = np.mean(phase[start:end])
        binaplev[i]  = np.median(aplev[start:end])
        binapstd[i]  = np.std(aplev[start:end]) / np.sqrt(end-start)

    #PLOT 
    plt.figure(501+numfigs*num)
    plt.clf()
    #plt.plot(phase, aplev, '.', ms=1)
    plt.errorbar(binphase,binaplev,binapstd,fmt='bo',linewidth=1)
    plt.title(obj + ' Phase vs. Binned Flux')
    plt.xlabel('Orbital Phase')
    plt.ylabel('Flux')
    plt.savefig(str(obj) + "-fig" + str(501+numfigs*num) + ".png")

    #PLOT
    plt.figure(502+numfigs*num, figsize=(8,12))
    plt.clf()
    plt.subplot(2,1,1)
    plt.title(obj + ' Position vs. Binned Flux')
    plt.errorbar(binyy, binyaplev, binyapstd, fmt='ro', label='y')
    #plt.plot(y, aplev, 'r.', ms=1)
    plt.ylabel('Flux')
    plt.legend()
    plt.subplot(2,1,2)
    plt.errorbar(binxx, binxaplev, binxapstd, fmt='bo', label='x')
    #plt.plot(x, aplev, 'b.', ms=1)
    plt.xlabel('Pixel Postion')
    plt.ylabel('Flux')
    plt.legend()
    plt.savefig(str(obj) + "-fig" + str(502+numfigs*num) + ".png")

    #PLOT 
    plt.figure(503+numfigs*num)
    plt.clf()
    plt.plot(phase, x, 'b.', ms=1)
    plt.plot(phase, y, 'r.', ms=1)
    plt.title(obj + ' Phase vs. Position')
    plt.xlabel('Orbital Phase')
    plt.ylabel('Pixel Position')
    plt.legend('xy')
    plt.savefig(str(obj) + "-fig" + str(503+numfigs*num) + ".png")

    #PLOT 
    plt.figure(504+numfigs*num)
    plt.clf()
    #plt.plot(np.sqrt(x**2+y**2), aplev, 'b.', ms=1)
    plt.errorbar(binrr, binraplev, binrapstd, fmt='bo', label='r')
    plt.title(obj + ' Radial Distance vs. Flux')
    plt.xlabel('Distance From Center of Pixel')
    plt.ylabel('Flux')
    plt.legend()
    plt.savefig(str(obj) + "-fig" + str(504+numfigs*num) + ".png")
    
    #PLOT MEANIM
    #plt.figure(104)
    #plt.imshow(event.meanim)
    #plt.title(obj + 'Mean Image')
    #Plot 'x' at mean star center
    #plt.savefig(obj + "-fig104.png")

    plt.show()

    print('MARK: ' + time.ctime() + ' : End p5checks\n')

    return event

def checks1(eventname, cwd, period=None, ephtime=None):

  owd = os.getcwd()
  #os.chdir(cwd)

  # Load the Event
  event = me.loadevent(eventname)
  
  # Create a log
  oldlogname = event.logname
  logname = event.eventname + "_p5.log"
  log = le.Logedit(logname, oldlogname)
  log.writelog('\nStart Checks: ' + time.ctime())

  # If p5 run after p3: we are using results from PSFfit: 
  if not hasattr(event, "phottype"):
    event.phottype = "psffit"
    try:
      os.mkdir("psffit/")
    except:
      pass
    #os.chdir("psffit/")

  # Move frame parameters to fit Kevin's syntax:
  # event.fp.param --> event.param
  event.filenames = event.fp.filename
  event.x         = event.fp.x
  event.y         = event.fp.y
  event.time      = event.fp.time
  event.pos       = event.fp.pos
  event.frmvis    = event.fp.frmvis
  event.filename  = event.eventname

  event.aplev      = event.fp.aplev
  event.background = event.fp.skylev
  event.good       = event.fp.good    

  if   event.phottype == "aper":
    event.aperr     = event.fp.aperr
    log.writelog('Photometry method is APERTURE')
  elif event.phottype == "var":
    event.aperr     = event.fp.aperr
    log.writelog('Photometry method is VARIABLE APERTURE')
  elif event.phottype == "ell":
    event.aperr     = event.fp.aperr
    log.writelog('Photometry method is ELLIPTICAL APERTURE')
  elif event.phottype == "psffit":
    # FINDME: do something with aperr
    event.aperr = .0025*np.mean(event.aplev)*np.ones(np.shape(event.aplev))
    log.writelog('Photometry method is PSF FITTING')
  elif event.phottype == "optimal":
    event.aperr = event.fp.aperr
    log.writelog('Photometry method is OPTIMAL')

    
  # UPDATE period AND ephtime
  if period is not None:
    event.period     = period[0]
    event.perioderr  = period[1]
  if ephtime is not None:
    event.ephtime    = ephtime[0]
    event.ephtimeerr = ephtime[1]

  log.writelog("\nCurrent event = " + event.eventname)
  log.writelog("Kurucz file     = " + event.kuruczfile)
  log.writelog("Filter file     = " + event.filtfile)


  # Light-time correction to BJD:

  # Julian observation date
  #event.juldat = event.jdjf80 + event.fp.time / 86400.0
  event.juldat = event.fp.juldat = event.j2kjd + event.fp.time / 86400.0

  if not event.ishorvec:
    log.writeclose('\nHorizon file not found!')
    return
  print("Calculating BJD correction...")

  event.fp.bjdcor = np.zeros(event.fp.juldat.shape)
  # Sometimes bad files are just missing files, in which case they have
  # times of 0, which causes problem in the following interpolation. So
  # we must mask out these files. We don't use the event.fp.good mask
  # because we may want to know the bjd of bad images

  nonzero = np.where(event.fp.time != 0.0)
  event.fp.bjdcor[nonzero] = stc.suntimecorr(event.ra, event.dec,
                                             event.fp.juldat[nonzero],
                                             event.horvecfile)

  # Get bjd times:
  event.bjdcor = event.fp.bjdcor
  #event.bjddat = event.fp.juldat + event.fp.bjdcor / 86400.0
  event.bjdutc = event.fp.juldat + event.fp.bjdcor / 86400.0   # utc bjd date
  event.bjdtdb = np.empty(event.bjdutc.shape)
  for i in range(event.bjdtdb.shape[0]):
    event.bjdtdb[i] = utc_tt.utc_tdb(event.bjdutc[i],
                                     event.topdir+'/'+event.leapdir)   # terrestial bjd date

  # ccampo 3/18/2011: check which units phase should be in
  try:
    if event.tep.ttrans.unit == "BJDTDB":
        event.timestd  = "tdb"
        event.fp.phase = tp.time2phase(event.bjdtdb, event.ephtime,
                                       event.period, event.ecltype)
    else:
        event.timestd  = "utc"
        event.fp.phase = tp.time2phase(event.bjdutc, event.ephtime,
                                       event.period, event.ecltype)
  except:
    event.timestd  = "utc"
    event.fp.phase = tp.time2phase(event.bjdutc, event.ephtime,
                                   event.period, event.ecltype)

  # assign phase variable
  event.phase = event.fp.phase

  # ccampo 3/18/2011: moved this above
  # Eclipse phase, BJD
  #event.fp.phase = tp.time2phase(event.fp.juldat + event.fp.bjdcor / 86400.0,
  #                               event.ephtime, event.period, event.ecltype)

  # verify leapsecond correction
  hfile = event.filenames[0,0]
  try:
    image, event.header = fits.getdata(hfile, header=True)
    dt  = ((event.bjdtdb - event.bjdutc)*86400.0)[0, 0]
    dt2 = event.header['ET_OBS'] - event.header['UTCS_OBS']
    log.writelog('Leap second correction : ' + str(dt) + ' = ' + str(dt2))
  except:
    log.writelog('Could not verify leap-second correction.')

  log.writelog('Min and Max light-time correction: ' +
               np.str(np.amin(event.fp.bjdcor)) + ', ' +
               np.str(np.amax(event.fp.bjdcor)) + ' seconds')

  # Verify light-time correction
  try:
    image, event.header = fits.getdata(hfile, header=True)
    try:
      log.writelog('BJD Light-time correction: ' + str(event.bjdcor[0,0]) +
          ' = ' + str((event.header['BMJD_OBS']-event.header['MJD_OBS'])*86400))
    except:
      log.writelog('HJD Light-time correction: ' + str(event.bjdcor[0,0]) +
          ' = ' + str((event.header['HMJD_OBS']-event.header['MJD_OBS'])*86400))
  except:
    log.writelog('Could not verify light-time correction.')

  # Number of good frames should be > 95%
  log.writelog("Good Frames = %7.3f"%(np.mean(event.good)*100)+ " %")

  log.writelog('\nCentering:     X mean     X stddev  Y mean     Y stddev')
  for pos in range(event.npos):
    log.writelog('position %2d:'%pos+
                 ' %10.5f'%np.mean(event.x[pos, np.where(event.good[pos])]) + 
                 ' %9.5f'%np.std(  event.x[pos, np.where(event.good[pos])]) +
                 ' %10.5f'%np.mean(event.y[pos, np.where(event.good[pos])]) + 
                 ' %9.5f'%np.std(  event.y[pos, np.where(event.good[pos])]) )

  # COMPUTE RMS POSITION CONSISTENCY
  event.xprecision = np.sqrt(np.mean(np.ediff1d(event.x)**2))
  event.yprecision = np.sqrt(np.mean(np.ediff1d(event.y)**2))

  log.writelog('RMS of x precision = '    + 
               str(np.round(event.xprecision,4)) + ' pixels.')
  log.writelog('RMS of y precision = '    + 
               str(np.round(event.yprecision,4)) + ' pixels.')
  if event.phottype == "aper":
    log.writelog('\nCenter & photometry half-width/aperture sizes = ' + 
                 str(event.ctrim) + ', ' + str(event.photap) + ' pixels.')
  log.writelog('Period = ' + str(event.period)    + ' +/- ' + 
               str(event.perioderr) + ' days')
  log.writelog('Ephemeris = ' + str(event.ephtime)    + ' +/- ' + 
               str(event.ephtimeerr) + ' JD')

  # Compute elliptical area if gaussian centering
  if event.method == 'fgc' or event.method == 'rfgc':
      event.fp.ellarea = np.pi * (3 * event.fp.xsig) * (3 * event.fp.ysig)

  fmt1 = ['bo','go','yo','ro','ko','co','mo','bs','gs','ys','rs','ks','cs','ms']
  fmt2 = ['b,','g,','y,','r,']
  fmt3 = ['b.','g.','y.','r.']

  plt.figure(501)
  plt.clf()
  plt.figure(502, figsize=(8,12))
  plt.clf()
  plt.figure(503)
  plt.clf()
  plt.figure(504)
  plt.clf()
  plt.figure(505)
  plt.clf()

  for pos in range(event.npos):
    wheregood = np.where(event.good[pos, :])
    # CHOOSE ONLY GOOD FRAMES FOR PLOTTING
    phase      = event.phase      [pos, :][wheregood]
    aplev      = event.aplev      [pos, :][wheregood]
    jdtime     = event.bjdutc     [pos, :][wheregood]
    background = event.background [pos, :][wheregood]
    noisepix   = event.fp.noisepix[pos, :][wheregood]
    if event.method == "fgc" or event.method == "rfgc":
        ellarea = event.fp.ellarea[pos, :][wheregood]
        rot     = event.fp.rot    [pos, :][wheregood]
    # COMPUTE X AND Y PIXEL LOCATION RELATIVE TO ...
    if event.npos > 1:
      # CENTER OF EACH PIXEL
      y = (event.y[pos, :] - np.round(event.y[pos, :]))[wheregood]
      x = (event.x[pos, :] - np.round(event.x[pos, :]))[wheregood]
    else:
      # CENTER OF MEDIAN PIXEL
      y = (event.y[pos, :] - np.round(np.median(event.y)))[wheregood]
      x = (event.x[pos, :] - np.round(np.median(event.x)))[wheregood]

    # SORT aplev BY x, y AND radial POSITIONS
    rad    = np.sqrt(x**2 + y**2)
    xx     = np.sort(x)
    yy     = np.sort(y)
    rr     = np.sort(rad)
    xaplev = aplev[np.argsort(x)]
    yaplev = aplev[np.argsort(y)]
    raplev = aplev[np.argsort(rad)]
  
    # BIN RESULTS FOR PLOTTING POSITION SENSITIVITY EFFECT
    nobj      = aplev.size
    nbins     = 120//event.npos
    binxx     = np.zeros(nbins)
    binyy     = np.zeros(nbins)
    binrr     = np.zeros(nbins)
    binxaplev = np.zeros(nbins)
    binyaplev = np.zeros(nbins)
    binraplev = np.zeros(nbins)
    binxapstd = np.zeros(nbins)
    binyapstd = np.zeros(nbins)
    binrapstd = np.zeros(nbins)
    binphase  = np.zeros(nbins)
    binaplev  = np.zeros(nbins)
    binapstd  = np.zeros(nbins)
    binnpix   = np.zeros(nbins)
    for i in range(nbins):
        start        = int(1.* i   *nobj/nbins)
        end          = int(1.*(i+1)*nobj/nbins)
        binxx[i]     = np.mean(xx[start:end])
        binyy[i]     = np.mean(yy[start:end])
        binrr[i]     = np.mean(rr[start:end])
        binxaplev[i] = np.median(xaplev[start:end])
        binyaplev[i] = np.median(yaplev[start:end])
        binraplev[i] = np.median(raplev[start:end])
        binxapstd[i] = np.std(xaplev[start:end]) / np.sqrt(end-start)
        binyapstd[i] = np.std(yaplev[start:end]) / np.sqrt(end-start)
        binrapstd[i] = np.std(raplev[start:end]) / np.sqrt(end-start)
        binphase[i]  = np.mean(phase[start:end])
        binaplev[i]  = np.median(aplev[start:end])
        binapstd[i]  = np.std(aplev[start:end]) / np.sqrt(end-start)
        binnpix[i]   = np.mean(noisepix[start:end])

    # PLOT 1: flux
    plt.figure(501)
    plt.errorbar(binphase, binaplev, binapstd, fmt=fmt1[pos], 
                 linewidth=1, label=('pos %i'%(pos)))
    plt.title(event.planetname + ' Phase vs. Binned Flux')
    plt.xlabel('Orbital Phase')
    plt.ylabel('Flux')
    plt.legend(loc='best')

    # PLOT 2: position-flux
    plt.figure(502)
    plt.subplot(2,1,1)
    plt.title(event.planetname + ' Position vs. Binned Flux')
    plt.errorbar(binyy, binyaplev, binyapstd, fmt=fmt1[pos], 
                 label=('pos %i y'%(pos)))
    plt.ylabel('Flux')
    plt.legend(loc='best')
    plt.subplot(2,1,2)
    plt.errorbar(binxx, binxaplev, binxapstd, fmt=fmt1[pos], 
                 label=('pos %i x'%(pos)))
    plt.xlabel('Pixel Postion')
    plt.ylabel('Flux')
    plt.legend(loc='best')
  
    #PLOT 3: position-phase
    plt.figure(503)

    plt.plot(phase, x, 'b,')
    plt.plot(phase, y, 'r,')
    plt.title(event.planetname + ' Phase vs. Position')
    plt.xlabel('Orbital Phase')
    plt.ylabel('Pixel Position')
    plt.legend('xy')

    #PLOT 4: flux-radial distance
    plt.figure(504)
    plt.errorbar(binrr, binraplev, binrapstd, fmt=fmt1[pos], 
                 label=('pos %i'%(pos)))
    plt.title(event.planetname + ' Radial Distance vs. Flux')
    plt.xlabel('Distance From Center of Pixel')
    plt.ylabel('Flux')
    plt.legend(loc='best')

    # ::::::::::: Background setting :::::::::::::::::
    if np.size(background) !=0:
      # number of points per bin:
      npoints = 42
      nbins = int(np.size(background)//npoints)
      medianbg = np.zeros(nbins)
      bphase   = np.zeros(nbins)  # background bin phase
      bintime  = np.zeros(nbins)  # background bin JD time
      for i in range(nbins):
        start        = int(1.0* i   *npoints)
        end          = int(1.0*(i+1)*npoints)
        medianbg[i]  = np.median(background[start:end])
        bphase[i]    = np.mean(  phase     [start:end])
        bintime[i]   = np.mean(  jdtime    [start:end])

      # PLOT 5: background-phase
      day = int(np.floor(np.amin(jdtime)))
      timeunits1 = jdtime  - day
      timeunits2 = bintime - day
      xlabel = 'JD - ' + str(day)
      if event.ecltype == 's':
        timeunits1 = phase
        timeunits2 = bphase
        xlabel = 'Phase'

      plt.figure(505)
      plt.plot(timeunits1, background, color='0.45', linestyle='None',
               marker=',')
      if np.size(background) > 10000:
        plt.plot(timeunits2, medianbg, fmt2[pos], label='median bins')
      plt.title(event.planetname + ' Background level')
      plt.xlabel(xlabel)
      plt.ylabel('Flux')
      plt.plot(timeunits1[0], background[0], color='0.45', linestyle='None',
               marker=',', label='all points')
      plt.legend(loc='best')

    else:
      print("WARNING: background has zero size.")

    #PLOT 7: Noise Pixels Binned
    plt.figure(507)
    plt.scatter(binphase, binnpix)
    plt.xlabel("Orbital Phase")
    plt.ylabel("Noise Pixels")
    plt.title(event.planetname + " Binned Noise Pixels")   

    #PLOT 8: Noise Pixel Variance
    plt.figure(508)
    npixvar        = bd.subarnvar(noisepix, event)
    subarnbinphase = bd.subarnbin(phase,    event)
    plt.scatter(subarnbinphase, npixvar, s=1)
    plt.xlabel("Orbital Phase")
    plt.ylabel("Noise Pixel Variance")
    plt.title(event.planetname + " Noise Pixels Variance")

    #PLOT 9 and 10: Elliptical Area and Variance
    if event.method == 'fgc' or event.method == 'rfgc':
      plt.figure(509)
      plt.scatter(phase, ellarea, s=0.1)
      plt.xlabel("Orbital Phase")
      plt.ylabel("Elliptical Area")
      plt.title(event.planetname + " Gaussian Centering Elliptical Area")

      plt.figure(510)
      ellareavar = bd.subarnvar(ellarea, event)
      plt.scatter(subarnbinphase, ellareavar, s=1)
      plt.xlabel("Orbital Phase")
      plt.ylabel("Elliptical Area Variance")
      plt.title(event.planetname + " Elliptical Area Variance")

    if event.method == 'rfgc':
      plt.figure(511)
      plt.scatter(phase, rot%(np.pi/2) * 180/np.pi, s=1)
      plt.xlabel("Orbital Phase")
      plt.ylabel("Rotation (deg)")
      plt.title(event.planetname + " Gaussian Centering Rotation")
  
  #PLOT 6: Preflash
  if event.havepreflash:
    plt.figure(506)
    plt.errorbar((event.prefp.time[0]-event.prefp.time[0,0])/60.,
                 event.prefp.aplev[0], yerr=event.prefp.aperr[0], fmt="o")
    plt.xlabel("Time since start of preflash  (minutes)")
    plt.ylabel("Flux")
    plt.title(event.planetname + " Preflash")
        
  figname1  = str(event.eventname) + "-fig501.png"
  figname2  = str(event.eventname) + "-fig502.png"
  figname3  = str(event.eventname) + "-fig503.png"
  figname4  = str(event.eventname) + "-fig504.png"
  figname5  = str(event.eventname) + "-fig505.png"
  figname6  = str(event.eventname) + "-fig506.png"
  figname7  = str(event.eventname) + "-fig507.png"
  figname8  = str(event.eventname) + "-fig508.png"
  figname9  = str(event.eventname) + "-fig509.png"
  figname10 = str(event.eventname) + "-fig510.png"
  figname11 = str(event.eventname) + "-fig511.png"

  plt.figure(501)
  plt.savefig(figname1)
  plt.figure(502)
  plt.savefig(figname2)
  plt.figure(503)
  plt.savefig(figname3)
  plt.figure(504)
  plt.savefig(figname4)
  plt.figure(505)
  plt.savefig(figname5)
  
  plt.figure(506)
  if event.havepreflash:
    plt.savefig(figname6)

  plt.figure(507)
  plt.savefig(figname7)
  plt.figure(508)
  plt.savefig(figname8)
  
  if event.method == 'fgc' or event.method == 'rfgc':
      plt.figure(509)
      plt.savefig(figname9)
      plt.figure(510)
      plt.savefig(figname10)

  if event.method == 'rfgc':
      plt.figure(511)
      plt.savefig(figname11)

  # Saving
  me.saveevent(event, event.eventname + "_p5c")

  cwd += "/"
  # Print outputs, end-time, and close log.
  log.writelog("Output files:")
  log.writelog("Data:")
  log.writelog(" " + cwd + event.eventname + "_p5c.dat")
  log.writelog("Log:")
  log.writelog(" " + cwd + logname)
  log.writelog("Figures:")
  log.writelog(" " + cwd + figname1)
  log.writelog(" " + cwd + figname2)
  log.writelog(" " + cwd + figname3)
  log.writelog(" " + cwd + figname4)
  log.writelog(" " + cwd + figname5)
  if event.havepreflash:
    log.writelog(" " + cwd + figname6)
  log.writelog(" " + cwd + figname7)
  log.writelog(" " + cwd + figname8)
  if event.method == 'fgc' or event.method == 'rfgc':
    log.writelog(" " + cwd + figname9)
    log.writelog(" " + cwd + figname10)
  if event.method == 'rfgc':
    log.writelog(" " + cwd + figname11)
  log.writeclose('\nEnd Checks: ' + time.ctime())

  #os.chdir(owd)

  return event


