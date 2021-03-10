"""
Revisions
---------
2018-01-05 zacchaeus  updated to python3
"""


import numpy as np
import time, os, copy, sys
import apphot_c       as ap
import reader3        as rd
import logedit        as le
import timer          as t
import manageevent    as me
import poet
from multiprocessing import Process, Array

import imageedit   as ie
import psf_fit     as pf
import optphot     as op
import matplotlib.pyplot as plt


def photometry(event, pcf, photdir, mute, owd):

  tini = time.time()

  # Create photometry log
  logname = event.logname
  log = le.Logedit(photdir + "/" + logname, logname)
  log.writelog("\nStart " + photdir + " photometry: " + time.ctime())

  parentdir = os.getcwd() + "/"
  #os.chdir(photdir)

  # Parse the attributes from the control file to the event:
  attrib = vars(pcf)
  keys = attrib.keys()
  for key in keys:
    setattr(event, key, attrib.get(key))

  maxnimpos, npos = event.maxnimpos, event.npos
  # allocating frame parameters: 
  event.fp.aplev     = np.zeros((npos, maxnimpos)) 
  event.fp.aperr     = np.zeros((npos, maxnimpos)) 
  event.fp.nappix    = np.zeros((npos, maxnimpos)) 
  event.fp.skylev    = np.zeros((npos, maxnimpos)) 
  event.fp.skyerr    = np.zeros((npos, maxnimpos)) 
  event.fp.nskypix   = np.zeros((npos, maxnimpos)) 
  event.fp.nskyideal = np.zeros((npos, maxnimpos)) 
  event.fp.status    = np.zeros((npos, maxnimpos)) 
  event.fp.good      = np.zeros((npos, maxnimpos))

  # For interpolated aperture photometry, we need to "interpolate" the
  # mask, which requires float values. Thus, we convert the mask to
  # floats (this needs to be done before processes are spawned or memory
  # usage balloons).
  if event.mask.dtype != float:
    event.mask = event.mask.astype(float)

  # Aperture photometry:
  if event.phottype == "aper": # not event.dooptimal or event.from_aper is None:

    # Multy Process set up:
    # Shared memory arrays allow only 1D Arrays :(
    aplev     = Array("d", np.zeros(npos*maxnimpos))# aperture flux
    aperr     = Array("d", np.zeros(npos*maxnimpos))# aperture error
    nappix    = Array("d", np.zeros(npos*maxnimpos))# number of aperture pixels
    skylev    = Array("d", np.zeros(npos*maxnimpos))# sky level
    skyerr    = Array("d", np.zeros(npos*maxnimpos))# sky error
    nskypix   = Array("d", np.zeros(npos*maxnimpos))# number of sky pixels
    nskyideal = Array("d", np.zeros(npos*maxnimpos))# ideal number of sky pixels
    status    = Array("d", np.zeros(npos*maxnimpos))# apphot return status
    good      = Array("d", np.zeros(npos*maxnimpos))# good flag
    # Size of chunk of data each core will process:
    chunksize = maxnimpos//event.ncores + 1

    event.aparr = np.ones(npos*maxnimpos) * event.photap + event.offset

    print("Number of cores: " + str(event.ncores))
    # Start Muti Procecess:
    processes = []
    for nc in range(event.ncores):
      start =  nc    * chunksize # Starting index to process
      end   = (nc+1) * chunksize # Ending   index to process
      proc = Process(target=do_aphot, args=(start, end, event, log, mute,
                                            aplev, aperr,
                                            nappix, skylev, skyerr, nskypix,
                                            nskyideal, status, good, 0))
      processes.append(proc)
      proc.start()

    # Make sure all processes finish their work:
    for nc in range(event.ncores):
      processes[nc].join()


    # Put the results in the event. I need to reshape them:
    event.fp.aplev     = np.asarray(aplev    ).reshape(npos,maxnimpos)
    event.fp.aperr     = np.asarray(aperr    ).reshape(npos,maxnimpos)
    event.fp.nappix    = np.asarray(nappix   ).reshape(npos,maxnimpos)
    event.fp.skylev    = np.asarray(skylev   ).reshape(npos,maxnimpos)
    event.fp.skyerr    = np.asarray(skyerr   ).reshape(npos,maxnimpos)
    event.fp.nskypix   = np.asarray(nskypix  ).reshape(npos,maxnimpos)
    event.fp.nskyideal = np.asarray(nskyideal).reshape(npos,maxnimpos)
    event.fp.status    = np.asarray(status   ).reshape(npos,maxnimpos)
    event.fp.good      = np.asarray(good     ).reshape(npos,maxnimpos)

    # raw photometry (no sky subtraction):
    event.fp.apraw = ( event.fp.aplev + ( event.fp.skylev * event.fp.nappix ) )

    # Print results into the log if it wasn't done before:
    for pos in range(npos):
      for i in range(event.nimpos[pos]):
        log.writelog('\nframe =%7d       '%i                 + 
                       'pos   =%5d       '%pos               +
                       'y =%7.3f       '  %event.fp.y[pos,i] + 
                       'x =%7.3f'         %event.fp.x[pos,i] + '\n' +
                       'aplev =%11.3f   ' %event.fp.aplev    [pos,i] + 
                       'aperr =%9.3f   '  %event.fp.aperr    [pos,i] +
                       'nappix =%6.2f'    %event.fp.nappix   [pos,i] + '\n' +
                       'skylev=%11.3f   ' %event.fp.skylev   [pos,i] + 
                       'skyerr=%9.3f   '  %event.fp.skyerr   [pos,i] +
                       'nskypix=%6.2f   ' %event.fp.nskypix  [pos,i] + 
                       'nskyideal=%6.2f'  %event.fp.nskyideal[pos,i] + '\n' +
                       'status=%7d       '%event.fp.status   [pos,i] + 
                       'good  =%5d'       %event.fp.good   [pos,i], mute=True)

  elif event.phottype == "var": # variable aperture radius

    # Multy Process set up:
    # Shared memory arrays allow only 1D Arrays :(
    aplev     = Array("d", np.zeros(npos*maxnimpos))# aperture flux
    aperr     = Array("d", np.zeros(npos*maxnimpos))# aperture error
    nappix    = Array("d", np.zeros(npos*maxnimpos))# number of aperture pixels
    skylev    = Array("d", np.zeros(npos*maxnimpos))# sky level
    skyerr    = Array("d", np.zeros(npos*maxnimpos))# sky error
    nskypix   = Array("d", np.zeros(npos*maxnimpos))# number of sky pixels
    nskyideal = Array("d", np.zeros(npos*maxnimpos))# ideal number of sky pixels
    status    = Array("d", np.zeros(npos*maxnimpos))# apphot return status
    good      = Array("d", np.zeros(npos*maxnimpos))# good flag
    # Size of chunk of data each core will process:
    chunksize = maxnimpos//event.ncores + 1

    event.aparr = event.fp.noisepix[0]**.5 * event.photap + event.offset

    print("Number of cores: " + str(event.ncores))
    # Start Muti Procecess:
    processes = []
    for nc in range(event.ncores):
      start =  nc    * chunksize # Starting index to process
      end   = (nc+1) * chunksize # Ending   index to process
      proc = Process(target=do_aphot, args=(start, end, event, log, mute,
                                            aplev, aperr,
                                            nappix, skylev, skyerr, nskypix,
                                            nskyideal, status, good, 0))
      processes.append(proc)
      proc.start()

    # Make sure all processes finish their work:
    for nc in range(event.ncores):
      processes[nc].join()


    # Put the results in the event. I need to reshape them:
    event.fp.aplev     = np.asarray(aplev    ).reshape(npos,maxnimpos)
    event.fp.aperr     = np.asarray(aperr    ).reshape(npos,maxnimpos)
    event.fp.nappix    = np.asarray(nappix   ).reshape(npos,maxnimpos)
    event.fp.skylev    = np.asarray(skylev   ).reshape(npos,maxnimpos)
    event.fp.skyerr    = np.asarray(skyerr   ).reshape(npos,maxnimpos)
    event.fp.nskypix   = np.asarray(nskypix  ).reshape(npos,maxnimpos)
    event.fp.nskyideal = np.asarray(nskyideal).reshape(npos,maxnimpos)
    event.fp.status    = np.asarray(status   ).reshape(npos,maxnimpos)
    event.fp.good      = np.asarray(good     ).reshape(npos,maxnimpos)

    # raw photometry (no sky subtraction):
    event.fp.apraw = ( event.fp.aplev + ( event.fp.skylev * event.fp.nappix ) )

    # Print results into the log if it wasn't done before:
    for pos in range(npos):
      for i in range(event.nimpos[pos]):
        log.writelog('\nframe =%7d       '%i                 + 
                       'pos   =%5d       '%pos               +
                       'y =%7.3f       '  %event.fp.y[pos,i] + 
                       'x =%7.3f'         %event.fp.x[pos,i] + '\n' +
                       'aplev =%11.3f   ' %event.fp.aplev    [pos,i] + 
                       'aperr =%9.3f   '  %event.fp.aperr    [pos,i] +
                       'nappix =%6.2f'    %event.fp.nappix   [pos,i] + '\n' +
                       'skylev=%11.3f   ' %event.fp.skylev   [pos,i] + 
                       'skyerr=%9.3f   '  %event.fp.skyerr   [pos,i] +
                       'nskypix=%6.2f   ' %event.fp.nskypix  [pos,i] + 
                       'nskyideal=%6.2f'  %event.fp.nskyideal[pos,i] + '\n' +
                       'status=%7d       '%event.fp.status   [pos,i] + 
                       'good  =%5d'       %event.fp.good   [pos,i], mute=True)

  elif event.phottype == "ell": # elliptical
    # Multy Process set up:
    # Shared memory arrays allow only 1D Arrays :(
    aplev     = Array("d", np.zeros(npos*maxnimpos))# aperture flux
    aperr     = Array("d", np.zeros(npos*maxnimpos))# aperture error
    nappix    = Array("d", np.zeros(npos*maxnimpos))# number of aperture pixels
    skylev    = Array("d", np.zeros(npos*maxnimpos))# sky level
    skyerr    = Array("d", np.zeros(npos*maxnimpos))# sky error
    nskypix   = Array("d", np.zeros(npos*maxnimpos))# number of sky pixels
    nskyideal = Array("d", np.zeros(npos*maxnimpos))# ideal number of sky pixels
    status    = Array("d", np.zeros(npos*maxnimpos))# apphot return status
    good      = Array("d", np.zeros(npos*maxnimpos))# good flag
    # Size of chunk of data each core will process:
    chunksize = maxnimpos//event.ncores + 1

    print("Number of cores: " + str(event.ncores))
    # Start Muti Procecess:
    processes = []
    for nc in range(event.ncores):
      start =  nc    * chunksize # Starting index to process
      end   = (nc+1) * chunksize # Ending   index to process
      proc = Process(target=do_aphot, args=(start, end, event, log, mute,
                                            aplev, aperr,
                                            nappix, skylev, skyerr, nskypix,
                                            nskyideal, status, good, 0))
      processes.append(proc)
      proc.start()

    # Make sure all processes finish their work:
    for nc in range(event.ncores):
      processes[nc].join()


    # Put the results in the event. I need to reshape them:
    event.fp.aplev     = np.asarray(aplev    ).reshape(npos,maxnimpos)
    event.fp.aperr     = np.asarray(aperr    ).reshape(npos,maxnimpos)
    event.fp.nappix    = np.asarray(nappix   ).reshape(npos,maxnimpos)
    event.fp.skylev    = np.asarray(skylev   ).reshape(npos,maxnimpos)
    event.fp.skyerr    = np.asarray(skyerr   ).reshape(npos,maxnimpos)
    event.fp.nskypix   = np.asarray(nskypix  ).reshape(npos,maxnimpos)
    event.fp.nskyideal = np.asarray(nskyideal).reshape(npos,maxnimpos)
    event.fp.status    = np.asarray(status   ).reshape(npos,maxnimpos)
    event.fp.good      = np.asarray(good     ).reshape(npos,maxnimpos)

    # raw photometry (no sky subtraction):
    event.fp.apraw = ( event.fp.aplev + ( event.fp.skylev * event.fp.nappix ) )

    # Print results into the log if it wasn't done before:
    for pos in range(npos):
      for i in range(event.nimpos[pos]):
        log.writelog('\nframe =%7d       '%i                 + 
                       'pos   =%5d       '%pos               +
                       'y =%7.3f       '  %event.fp.y[pos,i] + 
                       'x =%7.3f'         %event.fp.x[pos,i] + '\n' +
                       'aplev =%11.3f   ' %event.fp.aplev    [pos,i] + 
                       'aperr =%9.3f   '  %event.fp.aperr    [pos,i] +
                       'nappix =%6.2f'    %event.fp.nappix   [pos,i] + '\n' +
                       'skylev=%11.3f   ' %event.fp.skylev   [pos,i] + 
                       'skyerr=%9.3f   '  %event.fp.skyerr   [pos,i] +
                       'nskypix=%6.2f   ' %event.fp.nskypix  [pos,i] + 
                       'nskyideal=%6.2f'  %event.fp.nskyideal[pos,i] + '\n' +
                       'status=%7d       '%event.fp.status   [pos,i] + 
                       'good  =%5d'       %event.fp.good   [pos,i], mute=True)
        
  elif event.phottype == "psffit":
    event.fp.aplev  = event.fp.flux
    event.fp.skylev = event.fp.psfsky
    event.fp.good   = np.zeros((event.npos, event.maxnimpos))
    for pos in range(event.npos):
      event.fp.good[pos,0:event.nimpos[pos]] = 1

  elif event.phottype == "optimal":
    # utils for profile construction:
    pshape = np.array([2*event.otrim+1, 2*event.otrim+1])
    subpsf = np.zeros(np.asarray(pshape, int)*event.expand)
    x = np.indices(pshape)

    clock = t.Timer(np.sum(event.nimpos),
                    progress=np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1.1]))

    for  pos in range(npos):
      for i in range(event.nimpos[pos]):

        # Integer part of center of subimage:
        cen = np.rint([event.fp.y[pos,i], event.fp.x[pos,i]])
        # Center in the trimed image:
        loc = (event.otrim, event.otrim)
        # Do the trim:
        img, msk, err = ie.trimimage(event.data[i,:,:,pos], *cen, *loc,
                         mask=event.mask[i,:,:,pos], uncd=event.uncd[i,:,:,pos])

        # Center of star in the subimage:
        ctr = (event.fp.y[pos,i]-cen[0]+event.otrim,
               event.fp.x[pos,i]-cen[1]+event.otrim)

        # Make profile:
        # Index of the position in the supersampled PSF:
        pix = pf.pos2index(ctr, event.expand)
        profile, pctr = pf.make_psf_binning(event.psfim, pshape, event.expand,
                                            [pix[0], pix[1], 1.0, 0.0],
                                            event.psfctr, subpsf)

        #subtract the sky level:
        img -= event.fp.psfsky[pos,i]
        # optimal photometry calculation:
        immean, uncert, good = op.optphot(img, profile, var=err**2.0, mask=msk)

        event.fp.aplev [pos, i] = immean
        event.fp.aperr [pos, i] = uncert
        event.fp.skylev[pos, i] = event.fp.psfsky[pos,i]
        event.fp.good  [pos, i] = good

        # Report progress:
        clock.check(np.sum(event.nimpos[0:pos]) + i, name=event.centerdir)

  # START PREFLASH EDIT :::::::::::::::::::::::::::::::::::::

  # Do aperture on preflash data:
  if event.havepreflash:
    print("\nStart preflash photometry:")
    premaxnimpos = event.premaxnimpos
    preaplev     = Array("d", np.zeros(npos * premaxnimpos))
    preaperr     = Array("d", np.zeros(npos * premaxnimpos))
    prenappix    = Array("d", np.zeros(npos * premaxnimpos))
    preskylev    = Array("d", np.zeros(npos * premaxnimpos))
    preskyerr    = Array("d", np.zeros(npos * premaxnimpos))
    preskynpix   = Array("d", np.zeros(npos * premaxnimpos))
    preskyideal  = Array("d", np.zeros(npos * premaxnimpos))
    prestatus    = Array("d", np.zeros(npos * premaxnimpos))
    pregood      = Array("d", np.zeros(npos * premaxnimpos))

    # Start Procecess:
    mute = False
    proc = Process(target=do_aphot, args=(0, event.prenimpos[0], event, log,
                                          mute, preaplev, preaperr,
                                          prenappix, preskylev, preskyerr,
                                          preskynpix, preskyideal, prestatus,
                                          pregood, 1))
    proc.start()
    proc.join()

    # Put the results in the event. I need to reshape them:
    event.prefp.aplev  = np.asarray(preaplev ).reshape(npos,premaxnimpos)
    event.prefp.aperr  = np.asarray(preaperr ).reshape(npos,premaxnimpos)
    event.prefp.nappix = np.asarray(prenappix).reshape(npos,premaxnimpos)
    event.prefp.status = np.asarray(prestatus).reshape(npos,premaxnimpos)
    event.prefp.skylev = np.asarray(preskylev).reshape(npos,premaxnimpos)
    event.prefp.good   = np.asarray(pregood  ).reshape(npos,premaxnimpos)

    # raw photometry (no sky subtraction):
    event.prefp.aplev = ( event.prefp.aplev +
                          (event.prefp.skylev * event.prefp.nappix) )
    # END PREFLASH EDIT :::::::::::::::::::::::::::::::::::::::

  if event.method in ["bpf"]: 
    event.ispsf = False

  # PSF aperture correction:
  if event.ispsf and event.phottype == "aper":
    log.writelog('Calculating PSF aperture:')
    event.psfim = event.psfim.astype(np.float64)

    imerr = np.ones(np.shape(event.psfim))
    imask = np.ones(np.shape(event.psfim))
    skyfrac = 0.1
    
    event.aperfrac, ape, event.psfnappix, event.psfskylev, sle, \
         event.psfnskypix, event.psfnskyideal, event.psfstatus  \
                   = ap.apphot_c(event.psfim, imerr, imask,
                                 event.psfctr[0], event.psfctr[1],
                                 event.photap * event.psfexpand,
                                 event.skyin  * event.psfexpand,
                                 event.skyout * event.psfexpand,
                                 skyfrac, event.apscale, event.skymed)

    event.aperfrac += event.psfskylev * event.psfnappix 

    event.fp.aplev /= event.aperfrac
    event.fp.aperr /= event.aperfrac

    log.writelog('Aperture contains %f of PSF.'%event.aperfrac)

  if event.ispsf and event.phottype == "var":
    log.writelog('Calculating PSF aperture:')
    event.psfim = event.psfim.astype(np.float64)

    imerr = np.ones(np.shape(event.psfim))
    imask = np.ones(np.shape(event.psfim))
    skyfrac = 0.1

    avgap = np.mean(event.aparr)
    
    event.aperfrac, ape, event.psfnappix, event.psfskylev, sle, \
         event.psfnskypix, event.psfnskyideal, event.psfstatus  \
                   = ap.apphot_c(event.psfim, imerr, imask,
                                 event.psfctr[0], event.psfctr[1],
                                 avgap        * event.psfexpand,
                                 event.skyin  * event.psfexpand,
                                 event.skyout * event.psfexpand,
                                 skyfrac, event.apscale, event.skymed)

    event.aperfrac += event.psfskylev * event.psfnappix 

    event.fp.aplev /= event.aperfrac
    event.fp.aperr /= event.aperfrac

    log.writelog('Aperture contains %f of PSF.'%event.aperfrac)

  if event.ispsf and event.phottype == "ell":
    log.writelog('Calculating PSF aperture:')
    event.psfim = event.psfim.astype(np.float64)

    imerr = np.ones(np.shape(event.psfim))
    imask = np.ones(np.shape(event.psfim))
    skyfrac = 0.1

    avgxwid = np.mean(event.fp.xsig * event.photap)
    avgywid = np.mean(event.fp.ysig * event.photap)
    avgrot  = np.mean(event.fp.rot)
    
    event.aperfrac, ape, event.psfnappix, event.psfskylev, sle, \
         event.psfnskypix, event.psfnskyideal, event.psfstatus  \
                   = ap.elphot_c(event.psfim, imerr, imask,
                                 event.psfctr[0], event.psfctr[1],
                                 avgxwid * event.psfexpand,
                                 avgywid * event.psfexpand,
                                 avgrot,
                                 event.skyin  * event.psfexpand,
                                 event.skyout * event.psfexpand,
                                 skyfrac, event.apscale, event.skymed)

    event.aperfrac += event.psfskylev * event.psfnappix 

    event.fp.aplev /= event.aperfrac
    event.fp.aperr /= event.aperfrac

    log.writelog('Aperture contains %f of PSF.'%event.aperfrac)  

  # Sadly we must do photometry for every aperture used
  # Possibly use a range and interpolate? Might be an option
  # for the future to speed this up.
  # This is commented out, as it seems to just remove the corrections
  # made by variable or elliptical photometry
  # if event.ispsf and (event.phottype == "var" or event.phottype == "ell"):
  #   log.writelog('Calculating PSF aperture. This may take some time.')
  #   event.psfim = event.psfim.astype(np.float64)

  #   imerr = np.ones(np.shape(event.psfim))
  #   imask = np.ones(np.shape(event.psfim))
  #   skyfrac = 0.1

  #   aperfrac     = Array("d", np.zeros(npos*maxnimpos))# psf flux
  #   aperfracerr  = Array("d", np.zeros(npos*maxnimpos))# psf flux error
  #   psfnappix    = Array("d", np.zeros(npos*maxnimpos))# psf aperture pix num
  #   psfsky       = Array("d", np.zeros(npos*maxnimpos))# psf sky level
  #   psfskyerr    = Array("d", np.zeros(npos*maxnimpos))# psf sky error
  #   psfnskypix   = Array("d", np.zeros(npos*maxnimpos))# psf sky pix num
  #   psfnskyideal = Array("d", np.zeros(npos*maxnimpos))# psf ideal sky pix num
  #   psfstatus    = Array("d", np.zeros(npos*maxnimpos))# psf return status
  #   psfgood      = Array("d", np.zeros(npos*maxnimpos))# psf good flag

  #   processes=[]
  #   for nc in range(event.ncores):
  #     start =  nc    * chunksize
  #     end   = (nc+1) * chunksize
  #     proc = Process(target=do_aphot_psf, args=(start, end, event, log, mute,
  #                                               aperfrac, aperfracerr,
  #                                               psfnappix,
  #                                               psfsky, psfskyerr,
  #                                               psfnskypix, psfnskyideal,
  #                                               psfstatus, psfgood))

  #     processes.append(proc)
  #     proc.start()

  #   for nc in range(event.ncores):
  #     processes[nc].join()

  #   # Reshape
  #   event.aperfrac     = np.asarray(aperfrac    ).reshape(npos,maxnimpos)
  #   event.aperfracerr  = np.asarray(aperfracerr ).reshape(npos,maxnimpos)
  #   event.psfnappix    = np.asarray(psfnappix   ).reshape(npos,maxnimpos)
  #   event.psfsky       = np.asarray(psfsky      ).reshape(npos,maxnimpos)
  #   event.psfskyerr    = np.asarray(psfskyerr   ).reshape(npos,maxnimpos)
  #   event.psfnskypix   = np.asarray(psfnskypix  ).reshape(npos,maxnimpos)
  #   event.psfnskyideal = np.asarray(psfnskyideal).reshape(npos,maxnimpos)
  #   event.psfstatus    = np.asarray(psfstatus   ).reshape(npos,maxnimpos)
  #   event.psfgood      = np.asarray(psfgood     ).reshape(npos,maxnimpos)

  #   event.aperfrac += event.psfsky * event.psfnappix 

  #   event.fp.aplev /= event.aperfrac
  #   event.fp.aperr /= event.aperfrac

  #   log.writelog('Aperture contains average %f of PSF.'%np.mean(event.aperfrac))

  # save
  print("\nSaving ...")
  # denoised data:
  if event.denphot:
    killdata = 'dendata'
  else:
    killdata = 'data'
  me.saveevent(event, event.eventname + "_pht", delete=[killdata, 'uncd',
                                                        'mask'])

  # Print time elapsed and close log:
  cwd = os.getcwd() + "/"
  log.writelog("Output files (" + event.photdir + "):")
  log.writelog("Data:")
  log.writelog(" " + cwd + event.eventname + "_pht.dat")
  log.writelog("Log:")
  log.writelog(" " + cwd + logname)

  dt = t.hms_time(time.time()-tini)
  log.writeclose("\nEnd Photometry. Time (h:m:s):  %s "%dt  +
                 "  (" + photdir + ")")
  print("--------------  ------------\n")

  #os.chdir(owd)

  if event.runp5:
      os.system("python3 poet.py p5 %s/%s"%(event.centerdir, event.photdir))
      #poet.poet("p5", event.centerdir + '/' + event.photdir)

def run_photometry(eventname, cwd):
  """
  Load the event.
  Read the control file.
  Launch a thread for each centering run.
  """

  owd = os.getcwd()
  #os.chdir(cwd)
  config = os.path.basename(eventname)[:-4] + '.pcf'
  pcfs = rd.read_pcf(config, 'photometry')

  if len(pcfs) == 1: #, I may be in photdir to re-run:
    pcf = pcfs[0]
    if pcf.offset < 0:
      sign = '-'
    else:
      sign = '+'
    # Get name of photometry dir:
    if   pcf.phottype == "psffit":
      photdir = 'psffit'
    elif pcf.phottype == "optimal":
      photdir = 'optimal'
    elif pcf.phottype == "var":        
      photdir = ('va%03d'%(pcf.photap*100) + sign +
                 '%03d'%(np.abs(pcf.offset*100)))
    elif pcf.phottype == "ell":
      photdir = ('el%03d'%(pcf.photap*100) + sign +
                 '%03d'%(np.abs(pcf.offset*100)))
    else: # pcf[0].phottype == "aper": 
      photdir = ('ap%03d'%(pcf.photap*100) + sign +
                 '%03d'%(np.abs(pcf.offset*100)))

    # Append suffix to folder if suplied:
    if pcf.pcfname is not None:
      photdir += "_" + str(pcf.pcfname)

    # If I am in the photometry dir already:
    if cwd[-len(photdir):] == photdir:
      # Go to dir where poet3 files were saved.
      cwd = cwd[:-len(photdir)]
      #os.chdir(cwd)

    mute = False  # print to screen

  else:
    mute = True   # do not print to screen


  # Load the event data:
  if pcfs[0].denphot:  # Use denoised data if requested:
    readdata = 'dendata'
  else:
    readdata = 'data'
  event = me.loadevent(eventname, load=[readdata,'uncd','mask'])


  # Loop over each run:
  for pcf in pcfs:

    # Make a copy of the event:
    this_event = copy.copy(event)
    
    if pcf.offset < 0:
      sign = '-'
    else:
      sign = '+'

    # Get name of photometry dir:
    if   pcf.phottype == "psffit":
      photdir = 'psffit'
    elif pcf.phottype == "optimal":
      photdir = 'optimal'
    elif pcf.phottype == "var":
      photdir = ('va%03d'%(pcf.photap*100) + sign +
                 '%03d'%(np.abs(pcf.offset*100)))
    elif pcf.phottype == "ell":
      photdir = ('el%03d'%(pcf.photap*100) + sign +
                 '%03d'%(np.abs(pcf.offset*100)))
    else:
      photdir = ('ap%03d'%(pcf.photap*100) + sign +
                 '%03d'%(np.abs(pcf.offset*100)))

    # Append suffix to folder if suplied:
    if pcf.pcfname is not None:
      photdir += "_" + str(pcf.pcfname)
    this_event.photdir = photdir

    # Create the photometry directory if it doesn't exist:
    if not os.path.exists(photdir): 
      os.mkdir(photdir)

    # copy photom.pcf in photdir
    pcf.make_file(photdir + '/' + config, 'photometry')

    # Launch the thread:
    p = Process(target=photometry, args=(this_event, pcf, photdir, mute, owd))
    p.start()

  #os.chdir(owd)

def do_aphot(start, end, event, log, mute, aplev, aperr, nappix, skylev, skyerr,
             nskypix, nskyideal, status, good, datatype):
  """
    Notes:
    ------
    Medium level routine that performs aperture photometry.
    Each thread from the main routine (photometry) will run do_aphot once.
    do_aphot stores the values in the shared memory arrays.
  """
  # Initialize a Timer to report progress (use first Process):
  if start == 0: 
    clock = t.Timer(event.npos*end,
                    progress=np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1.1]))

  y, x = event.fp.y, event.fp.x
  if datatype == 0:
    data = event.data
    mask = event.mask
    imer = event.uncd
    nimpos = event.nimpos
  elif datatype == 1:  # Preflash
    data = event.predata
    mask = event.premask
    imer = event.preuncd
    nimpos = event.prenimpos
    y[:] = np.mean(y)
    x[:] = np.mean(x)

  if event.denphot:
    data = event.dendata

  if event.phottype == 'ell':
    ywid = event.fp.ysig * event.photap + event.offset
    xwid = event.fp.xsig * event.photap + event.offset   
    rot  = event.fp.rot 

  for pos in range(event.npos):
    # Recalculate star and end indexes. Care not to go out of bounds:
    end   = np.amin([end,   nimpos[pos]])
    start = np.amin([start, nimpos[pos]])

    for i in range(start, end):
      # Index in the share memory arrays:
      # FINDME: bug fix loc
      loc = pos * event.maxnimpos + i
      # Calculate aperture photometry:
      if event.phottype == 'ell':
        aphot = ap.elphot_c(data[i,:,:,pos], imer[i,:,:,pos],
                            mask[i,:,:,pos],  y[pos,i], x[pos,i],
                            xwid[pos,i], ywid[pos,i], rot[pos,i],
                            event.skyin, event.skyout,
                            event.skyfrac, event.apscale, event.skymed)
      else:
        aphot = ap.apphot_c(data[i,:,:,pos], imer[i,:,:,pos],
                            mask[i,:,:,pos],  y[pos,i], x[pos,i],
                            event.aparr[i], event.skyin, event.skyout,
                            event.skyfrac, event.apscale, event.skymed)

      # Store values:
      aplev  [loc], aperr  [loc], nappix   [loc], skylev[loc], \
       skyerr[loc], nskypix[loc], nskyideal[loc], status[loc] = aphot

      if status[loc] == 0:
        good[loc] = 1 # good flag

      # Print to screen only if one core:
      if event.ncores == 1 and (not mute): # (end - start) == event.maxnimpos:
        if event.verbose:
          print('\nframe =%7d       '       %i               + 
                         'pos   =%5d       '%pos             +
                         'y =%7.3f       '  %y[pos,i]        + 
                         'x =%7.3f'         %x[pos,i]        + '\n' +
                         'aplev =%11.3f   ' %aplev     [loc] + 
                         'aperr =%9.3f   '  %aperr     [loc] +
                         'nappix =%6.2f'    %nappix    [loc] + '\n' +
                         'skylev=%11.3f   ' %skylev    [loc] + 
                         'skyerr=%9.3f   '  %skyerr    [loc] +
                         'nskypix=%6.2f   ' %nskypix   [loc] + 
                         'nskyideal=%6.2f'  %nskyideal [loc] + '\n' +
                         'status=%7d       '%status    [loc] + 
                         'good  =%5d'       %good      [loc])

          perc = 100.0*(np.sum(event.nimpos[:pos])+i+1)/np.sum(event.nimpos)
          hms = clock.hms_left(np.sum(event.nimpos[0:pos]) + i)
          print("progress: %6.2f"%perc + "%  -  Remaining time (h:m:s):" + hms)

      # Fisrt Process when ncores > 1:
      if start == 0:
        if mute or event.ncores > 1: 
          clock.check(pos*end + i, name=event.photdir)

def do_aphot_psf(start, end, event, log, mute, aplev, aperr, nappix,
                 skylev, skyerr, nskypix, nskyideal, status, good):
  '''
  Same as do_aphot, except acts on a single image (the PSF). Used for
  multiprocessing. Omits some log printing.
  '''
  
  # Initialize a Timer to report progress (use first Process):
  if start == 0: 
    clock = t.Timer(event.npos*end,
                    progress=np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1.1]))

  x, y = event.psfctr

  data  = event.psfim
  mask  = np.ones(np.shape(event.psfim))
  imerr = np.ones(np.shape(event.psfim))

  nimpos = event.nimpos

  if event.phottype == 'ell':
    ywid = event.fp.ysig * event.photap
    xwid = event.fp.xsig * event.photap
    rot  = event.fp.rot 

  for pos in range(event.npos):
    # Recalculate star and end indexes. Care not to go out of bounds:
    end   = np.amin([end,   nimpos[pos]])
    start = np.amin([start, nimpos[pos]])

    for i in range(start, end):
      # Index in the share memory arrays:
      loc = pos * event.maxnimpos + i
      # Calculate aperture photometry:
      if event.phottype == 'ell':
        aphot = ap.elphot_c(data, imerr, mask, x, y,
                            xwid[pos,i]  * event.psfexpand,
                            ywid[pos,i]  * event.psfexpand,
                            rot[pos,i],
                            event.skyin  * event.psfexpand,
                            event.skyout * event.psfexpand,
                            event.skyfrac, 1, event.skymed)
      else:
        aphot = ap.apphot_c(data, imerr, mask, x, y,
                            event.photap * event.psfexpand,
                            event.skyin  * event.psfexpand,
                            event.skyout * event.psfexpand,
                            event.skyfrac, 1, event.skymed)

      # Store values:
      aplev  [loc], aperr  [loc], nappix   [loc], skylev[loc], \
       skyerr[loc], nskypix[loc], nskyideal[loc], status[loc] = aphot

      if status[loc] == 0:
        good[loc] = 1 # good flag

      #perc = 100.0*(np.sum(event.nimpos[:pos])+i+1)/np.sum(event.nimpos)
      #hms = clock.hms_left(np.sum(event.nimpos[0:pos]) + i)
      #print("progress: %6.2f"%perc + "%  -  Remaining time (h:m:s):" + hms)

      # Fisrt Process when ncores > 1:
      if start == 0:
        if mute or event.ncores > 1: 
          clock.check(pos*end + i, name=event.photdir)
