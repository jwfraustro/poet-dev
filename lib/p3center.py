import numpy  as np
import astropy.io.fits as fits
import sys, time, os, shutil, copy
import reader3      as rd
import logedit      as le
import manageevent  as me
import centerdriver as cd
import timer        as t
import poet
from multiprocessing import Process, Array

"""
POET_P3CENTERING WORKFLOW:
--------------------------
This beautiful piece of code consists of three sections:
- run_centering
- centering
- do_center

Modification History:
---------------------
2010-??-??  patricio  Initial Pyhton implementation
2014-08-13  garland   Switched the pyfits package to astropy.io.fits
	              zabblleon@gmail.com 
2017-06-20 zacchaeus  Fixed None comparisons
                      zaccysc@gmail.com
"""


def centering(event, pcf, centerdir, owd):
  
  os.chdir(centerdir)

  tini = time.time()

  # Create centering log
  log = le.Logedit(event.logname, event.logname)
  log.writelog("\nStart " + centerdir + " centering: " + time.ctime())

  # Parse the attributes from the control file to the event:
  attrib = vars(pcf)
  keys = attrib.keys()
  for key in keys:
    setattr(event, key, attrib.get(key))

  # Check least asym parameters work:
  if event.method in ['lac', 'lag']:
    if event.ctrim < (event.cradius + event.csize) and event.ctrim !=0:
      event.ctrim = event.cradius + event.csize + 1
      log.writelog('Trim radius is too small, changed to: %i'%event.ctrim)
    if event.psfctrim < (event.psfcrad + event.psfcsize) and event.psfctrim !=0:
      event.psfctrim = event.psfcrad + event.psfcsize + 1
      log.writelog('PSF Trim radius is too small, changed to: %i'
                   %event.psfctrim)

  # Centering bad pixel mask:
  centermask = np.ones((event.ny, event.nx))
  if event.ymask is not None:
    ymask = np.asarray(event.ymask, int)
    xmask = np.asarray(event.xmask, int)
    for i in range(len(ymask)):
      centermask[ymask[i], xmask[i]] = 0

  # PSF:
  # Re-evaluate if a PSF has been redefined:
  if event.newpsf is not None:
    event.ispsf = os.path.isfile(event.newpsf)
    if event.ispsf:
      event.psffile = event.newpsf
      log.writelog('The PSF file has been redefined!')
      log.writelog("PSF:     " + event.psffile)

  # PSF Centering:
  if event.ispsf:
    event.psfim = fits.getdata(event.psffile)
    # Guess of the center of the PSF (center of psfim)
    psfctrguess = np.asarray(np.shape(event.psfim))//2
    # Do not find center of PSF:
    if event.nopsfctr:
      event.psfctr = psfctrguess
    # Find center of PSF:
    else:
      if event.method == "bpf" or event.method == "ipf":
        method = "fgc"
      else:
        method = event.method
      event.psfctr, extra = cd.centerdriver(method, event.psfim, psfctrguess,
                                            event.psfctrim, event.psfcrad,
                                            event.psfcsize,
                                            npskyrad=(event.npskyin,
                                                      event.npskyout))
    log.writelog('PSF center found.')
  else:
    event.psfim  = None
    event.psfctr = None
    log.writelog('No PSF supplied.')
  # Find center of the mean Image:
  event.targpos = np.zeros((2, event.npos))

  # Override target position estimate if specified
  if type(pcf.srcesty) != type(None) and type(pcf.srcestx) != type(None):
    srcesty = str(pcf.srcesty).split(',')
    srcestx = str(pcf.srcestx).split(',')
    
    if len(srcestx) != len(srcesty):
      print("WARNING: Length of srcest inputs do not match!")
    if len(srcestx) != event.npos or len(srcesty) != event.npos:
      print("WARNING: Length of srcest inputs do not match npos!")
    if len(srcestx) > 1 or len(srcesty) > 1:
      print("Verify that srcest override order matches telescope pos order.")

    for pos in range(event.npos):
      event.srcest[0, pos] = srcesty[pos]
      event.srcest[1, pos] = srcestx[pos]
        
  for pos in range(event.npos):
    print("Fitting mean image at pos: " + str(pos))
    meanim = event.meanim[:,:,pos]      
    guess  = event.srcest[:, pos]
    targpos, extra = cd.centerdriver(event.method, meanim,
                                     guess, event.ctrim,
                                     event.cradius, event.csize,
                                     fitbg=event.fitbg, psf=event.psfim,
                                     psfctr=event.psfctr, expand=event.expand, npskyrad=(event.npskyin, event.npskyout))

    event.targpos[:,pos] = targpos
  log.writelog("Center position(s) of the mean Image(s):\n" +
               str(np.transpose(event.targpos)))

  # Inclusion ::::::::
  # Multy Process set up:
  # Shared memory arrays allow only 1D Arrays :(
  x        = Array("d", np.zeros(event.npos * event.maxnimpos))
  y        = Array("d", np.zeros(event.npos * event.maxnimpos))
  xerr     = Array("d", np.zeros(event.npos * event.maxnimpos))
  yerr     = Array("d", np.zeros(event.npos * event.maxnimpos))
  xsig     = Array("d", np.zeros(event.npos * event.maxnimpos))
  ysig     = Array("d", np.zeros(event.npos * event.maxnimpos))
  rot      = Array("d", np.zeros(event.npos * event.maxnimpos))
  noisepix = Array("d", np.zeros(event.npos * event.maxnimpos))
  flux     = Array("d", np.zeros(event.npos * event.maxnimpos))
  sky      = Array("d", np.zeros(event.npos * event.maxnimpos))
  goodfit  = Array("d", np.zeros(event.npos * event.maxnimpos))

  # Size of chunk of data each core will process:
  chunksize = event.maxnimpos//event.ccores + 1
  print("Number of cores: " + str(event.ccores))

  # Start Muti Procecess: ::::::::::::::::::::::::::::::::::::::
  processes = []
  for nc in range(event.ccores):
    start =  nc    * chunksize # Starting index to process
    end   = (nc+1) * chunksize # Ending   index to process
    proc = Process(target=do_center, args=(start, end, event, centermask, log,
                                           x,    y, flux, sky, goodfit,
                                           xerr, yerr, xsig, ysig, noisepix,
                                           rot))
    processes.append(proc)
    proc.start()
  # Make sure all processes finish their work:
  for nc in range(event.ccores):
    processes[nc].join()

  # Put the results in the event. I need to reshape them:
  event.fp.x         = np.asarray(x       ).reshape(event.npos,event.maxnimpos)
  event.fp.y         = np.asarray(y       ).reshape(event.npos,event.maxnimpos)
  event.fp.xerr      = np.asarray(xerr    ).reshape(event.npos,event.maxnimpos)
  event.fp.yerr      = np.asarray(yerr    ).reshape(event.npos,event.maxnimpos)
  event.fp.noisepix  = np.asarray(noisepix).reshape(event.npos,event.maxnimpos)
  # If Gaussian fit:
  if event.method == 'fgc' or event.method == 'rfgc':
    event.fp.xsig    = np.asarray(xsig    ).reshape(event.npos,event.maxnimpos)
    event.fp.ysig    = np.asarray(ysig    ).reshape(event.npos,event.maxnimpos)
    event.fp.rot     = np.asarray(rot     ).reshape(event.npos,event.maxnimpos)
  # If PSF fit:
  if event.method in ["ipf", "bpf"]: 
    event.fp.flux    = np.asarray(flux    ).reshape(event.npos,event.maxnimpos)
    event.fp.psfsky  = np.asarray(sky     ).reshape(event.npos,event.maxnimpos)
    event.fp.goodfit = np.asarray(goodfit ).reshape(event.npos,event.maxnimpos)
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # Pixel R position:
  event.fp.r = np.sqrt((event.fp.x % 1.0 - 0.5)**2.0 + 
                       (event.fp.y % 1.0 - 0.5)**2.0 )

  log.writelog("End frames centering.")

  # Save
  print("\nSaving")
  if event.denoised:
    me.saveevent(event, event.eventname + "_ctr", save=['dendata', 'data',
                                                        'uncd', 'mask'])
  else:
    me.saveevent(event, event.eventname + "_ctr", save=['data', 'uncd', 'mask'])

  # Print time elapsed and close log:
  cwd = os.getcwd()
  log.writelog("Output files (" + event.centerdir + "):")
  log.writelog("Data:")
  log.writelog(" " + cwd + '/' + event.eventname + "_ctr.dat")
  log.writelog(" " + cwd + '/' + event.eventname + "_ctr.h5")
  log.writelog("Log:")
  log.writelog(" " + cwd + '/' + event.logname)

  dt = t.hms_time(time.time()-tini)
  log.writeclose("\nEnd Centering. Time (h:m:s):  %s"%dt  +
                 "  (" + event.centerdir + ")")
  print("-------------  ------------\n")

  os.chdir(owd)

  if event.runp4:
      os.system("python3 poet.py p4 %s"%event.centerdir)
      #poet.poet("p4", event.centerdir)

def run_centering(eventname, cwd):
  """
  Read the control file.
  Load the event.
  Launch a thread for each centering run.
  """

  owd = os.getcwd()
  os.chdir(cwd)
  config = os.path.basename(eventname)[:-4] + '.pcf'
  pcfs = rd.read_pcf(config, 'centering')

  if len(pcfs) == 1: #, I may be in the center dir, to re-run: 
    # Get name of centering dir:
    pcf = pcfs[0]
    centerdir = pcf.method
    if pcf.pcfname is not None:
      centerdir += "_" + str(pcf.pcfname)

    if cwd[-len(centerdir):] == centerdir:
      # Go to dir where poet2 files were saved.
      cwd = cwd[:-len(centerdir)]
      os.chdir(cwd)

  # Load the event:
  try:
    event = me.loadevent(eventname, load=['dendata', 'data','uncd','mask'])
    print("Performing centering on denoised data")
  except:
    event = me.loadevent(eventname, load=['data','uncd','mask'])

  # Loop over each run:
  for pcf in pcfs:

    # Make a copy of the event:
    this_event = copy.copy(event)

    # Name of the directory to put the results:
    centerdir = pcf.method
    if pcf.pcfname is not None:
      centerdir += "_" + str(pcf.pcfname)
    this_event.centerdir = centerdir
    
    # Create the centering directory if it doesn't exist:
    if not os.path.exists(centerdir): 
      os.mkdir(centerdir)

    # copy the photometry and centering configuration into the
    # centering directory
    filename = centerdir + '/' + event.eventname + '.pcf'
    pcf.make_file(filename, 'centering')
    rd.copy_config(config, ['photometry'], filename)

    # Launch the thread:
    p = Process(target=centering, args=(this_event, pcf, centerdir, owd))
    p.start()

  os.chdir(owd)


def do_center(start, end, event, centermask, log, x, y, flux, sky, goodfit, xerr, yerr, xsig, ysig, noisepix, rot):

  # Initialize a Timer to report progress:
  if start == 0:  # Only for the fisrt chunk
    clock = t.Timer(event.npos*end,
                    progress=np.array([0.05, 0.1, 0.2, 0.3, 0.4,  0.5,
                                       0.6,  0.7, 0.8, 0.9, 0.99, 1.1]))

  # Use denoised data if exists:
  if event.denoised:
    data = event.dendata
  else:
    data = event.data

  # Finally, do the centering:
  for  pos in range(event.npos):
    # Recalculate star/end, Care not to go out of bounds:
    end   = np.amin([end,   event.nimpos[pos]])
    start = np.amin([start, event.nimpos[pos]]) # is this necessary?
    if event.noctr:   # Just use the mean x,y in this case
      y[pos*event.maxnimpos+start:pos*event.maxnimpos+end] = event.targpos[0, pos]
      x[pos*event.maxnimpos+start:pos*event.maxnimpos+end] = event.targpos[1, pos]
    else:
      for im in range(start, end):
        # Index in the share memory arrays:
        ind = pos * event.maxnimpos + im
        try:
          if event.weights:   # weight by uncertainties in fitting?
            uncd = event.uncd[im,:,:,pos]
          else:
            uncd = None
          # Do the centering:
          position, extra, noisepixels = cd.centerdriver(event.method,
                                  data[im,:,:,pos],
                                  event.targpos[:,pos], event.ctrim,
                                  event.cradius, event.csize,
                                  (event.mask[im,:,:,pos]*centermask).astype(int),
                                  uncd, fitbg=event.fitbg,
                                  expand=event.expand,
                                  psf=event.psfim,
                                  psfctr=event.psfctr,
                                  noisepix=True,
                                  npskyrad=(event.npskyin, event.npskyout))
          #print("im: %3i  pos: %2i  y: %6.3f  x: %6.3f ind: %3i"%(im, pos, position[0], position[1], ind))
          y[ind], x[ind] = position
          if event.method == "ipf" or event.method == "bpf":
            flux[ind] = extra[0]
            sky [ind] = extra[1]
            # FINDME: define some criterion for good/bad fit.
            goodfit[ind] = 1
          if event.method == "fgc" or event.method == "rfgc":
            yerr[ind] = extra[0]
            xerr[ind] = extra[1]
            ysig[ind] = extra[2]
            xsig[ind] = extra[3]
          if event.method == "rfgc":
            rot [ind] = extra[4]
          noisepix[ind] = noisepixels   
        except Exception as e:
          y[ind], x[ind] = event.targpos[:, pos]
          flux[ind], sky[ind] = 0.0, 0.0
          goodfit[ind] = 0
          log.writelog("Centering failed in im, pos: %5i"%im + ", %2i"%pos)

        if start == 0: 
          # Report progress:
          clock.check(pos*end + im, name=event.centerdir)
