import sys, time, os
import numpy  as np
import astropy.io.fits as fits

import pbright2flux as btf
import pbadmask     as pbm
import pchunkbad    as pcb
import logedit      as le
import manageevent  as me
import timer        as t
import poet

def badpix(eventname, cwd):
  """
  Modification History:
  ---------------------
  2010-??-??  patricio  Initial Python implementation
  2014-08-13  garland   switched the pyfits package to astropy.io.fits
  	              zabblleon@gmail.com 
  2017-06-20  zacchaeus Fixed None comparisons
                        zaccysc@gmail.com
  """

  owd = os.getcwd()

  tini = time.time()

  # Load the event
  event = me.loadevent(os.path.join(cwd, eventname))
  # Load the data
  me.updateevent(event, os.path.join(cwd, eventname), event.loadnext)

  # Create a new log starting from the old one.
  oldlogname = event.logname
  logname = event.eventname + ".log"
  log = le.Logedit(logname, oldlogname)
  event.logname = logname
  log.writelog('\nMARK: ' + time.ctime() + ': Starting p2badpix.')

  # ccampo 3/18/2011: do this in p5
  # Julian observation date
  #event.fp.juldat = event.jdjf80 + event.fp.time / 86400.0
  

  # ::::::::::::::::::::::: UNCERTAINTIES ::::::::::::::::::::::::::::::::
  # IRAC subarray data come with bogus uncertainties that are not linearly
  # related to photon noise.  We scale them later, using the reduced chi
  # squared from the model fit.

  # ::::::::::::::::::::::: FLUX CONVERSION :::::::::::::::::::::::::::::
  # Do we want flux (uJy/pix) or surface brightness (MJy/sr) units?  If
  # doing photometry, convert to flux.  Since we care about relative
  # numbers, it doesn't really matter.

  # Convert from surface brightness (MJy/sr) to flux units (uJy/pix)
  if event.fluxunits:
    log.writelog('Converting surface brightness to flux')
    event.data, event.uncd = btf.poet_bright2flux(event.data, event.uncd,
                                                  event.posscl)
    if event.havepreflash:
      event.predata,  event.preuncd  = btf.poet_bright2flux(event.predata,
                                                event.preuncd, event.posscl )
    if event.havepostcal:
      event.postdata, event.postuncd = btf.poet_bright2flux(event.postdata, 
                                                event.postuncd, event.posscl)

  else:
    log.writelog('Did not convert bright to flux.')


  # Mean Background Estimate, from zodi model
  event.estbg = ( np.mean(event.fp.zodi[np.where(event.fp.exist)]) +
                  np.mean(event.fp.ism [np.where(event.fp.exist)]) +
                  np.mean(event.fp.cib [np.where(event.fp.exist)]) )

  if event.fluxunits:
    event.estbg *= ( event.srperas * 1e12 * np.mean(event.posscl[0,:]) * 
                                            np.mean(event.posscl[1,:]) )

  # Bad Pixel Masking
  log.writelog('Find and fix bad pixels')

  # Get permanent bad pixel mask.
  if not event.ispmask[0]:
    log.writelog('\nPermanent Bad pixel mask not found!')
  else:
    hdu = fits.open(event.pmaskfile[0])
    if hdu[0].header['bitpix'] == -32:  # if data type is float
      hdu[0].scale(type='int16')        # cast it down to int16
    event.pmask = hdu[0].data


  # IRS FIX:
  # IRS data contains the blue peak subarray while its pmask contains
  # the whole array (Hard coding)
  if event.photchan == 5:
    event.pmask = event.pmask[3:59,86:127]


  # Do NOT define sigma, we have a different scheme for finding baddies
  # adds Spitzer rejects: fp.nsstrej  &  our rejects: fp.nsigrej
  event.mask = pbm.poet_badmask(event.data,   event.uncd,
                                event.pmask,  event.inst.pcrit,
                                event.bdmskd, event.inst.dcrit,
                                event.fp,     nimpos=event.nimpos)

  # User rejected pixels:
  if event.userrej is not None:
    for i in range(np.shape(event.userrej)[0]):
      event.mask[:, event.userrej[i,0], event.userrej[i,1], :] = 0
    event.fp.userrej = np.sum(np.sum(1-event.mask, axis=1), axis=1)
    event.fp.userrej = np.transpose(event.fp.userrej) - event.fp.nsstrej
  else:
    event.fp.userrej = np.zeros((event.npos, event.maxnimpos))


  # define sigma here.
  # adds median sky: fp.medsky
  event.meanim = pcb.poet_chunkbad(event.data,  event.uncd, 
                                   event.mask,  event.nimpos, 
                                   event.sigma, event.szchunk,
                                   event.fp,    event.nscyc   )

  log.writelog('Masks combined')

  # Repeat procedure for preflash and postcal data:
  if event.havepreflash:
    event.premask = pbm.poet_badmask(event.predata,   event.preuncd, 
                                     event.pmask,     event.inst.pcrit,
                                     event.prebdmskd, event.inst.dcrit,
                                     event.prefp,     nimpos=event.prenimpos)
    if event.userrej is not None:
      for i in range(np.shape(event.userrej)[0]):
        event.premask[:, event.userrej[i,0], event.userrej[i,1], :] = 0
      event.prefp.userrej = np.sum(np.sum(1-event.premask, axis=1), axis=1)
      event.prefp.userrej =np.transpose(event.prefp.userrej)-event.prefp.nsstrej
    else:
      event.prefp.userrej = np.zeros((event.npos, event.premaxnimpos))

    event.premeanim = pcb.poet_chunkbad(event.predata, event.preuncd, 
                                        event.premask, event.prenimpos,
                                        event.sigma,   event.szchunk,
                                        event.prefp,   event.nscyc     )
  if event.havepostcal:
    event.postmask = pbm.poet_badmask(event.postdata,   event.postuncd, 
                                      event.pmask,      event.inst.pcrit,
                                      event.postbdmskd, event.inst.dcrit,
                                      event.postfp,   nimpos=event.postnimpos)

    if event.userrej is not None:
      for i in range(np.shape(event.userrej)[0]):
        event.postmask[:, event.userrej[i,0], event.userrej[i,1], :] = 0
      event.postfp.userrej = np.sum(np.sum(1-event.postmask, axis=1), axis=1)
      event.postfp.userrej = np.transpose(event.postfp.userrej) - \
                             event.postfp.nsstrej
    else:
      event.postfp.userrej = np.zeros((event.npos, event.postmaxnimpos))

    event.postmeanim = pcb.poet_chunkbad(event.postdata,  event.postuncd,
                                         event.postmask, event.postnimpos,
                                         event.sigma,    event.szchunk,
                                         event.postfp,   event.nscyc     )
    for pos in range(event.npos):
      fits.writeto(event.eventname + "_medpostcal.fits",
                 event.postmeanim[:,:,pos], clobber=True)

    # Delete post calibration data:
    event.havepostcal = False
    del(event.postdata)
    del(event.postmask)
    del(event.postuncd)
    del(event.postbdmskd)


  # Save the data
  if event.instrument == 'mips':   
    todel = ['bdmskd', 'brmskd']  # what to delete
  else:
    todel = ['bdmskd']

  me.saveevent(event, os.path.join(event.topdir, event.eventname + "_bpm"),
            save=['data', 'uncd', 'mask'], delete=todel)

  # Print time elapsed and close log:
  log.writelog("Output files:")
  log.writelog("Data:")
  log.writelog(" " + cwd + '/' + event.eventname + "_bpm.dat")
  log.writelog(" " + cwd + '/' + event.eventname + "_bpm.h5")
  log.writelog("Log:")
  log.writelog(" " + cwd + '/' + logname)

  dt = t.hms_time(time.time()-tini)
  log.writeclose('\nBad pixel masking time (h:m:s):  %s '%dt)

  #os.chdir(owd)

  if event.runp3:
      #poet.p(3)
      os.system("python3 poet.py p3")
