import os, sys, re, time
import numpy  as np
import astropy.io.fits as fits

import matplotlib.pyplot as plt
import sexa2dec      as s2d
import pdataread     as pdr
import reader3       as rd
import tepclass      as tc
import instrument    as inst
import logedit       as le
import timer         as t
import poet
from manageevent import *
from univ import Univ

class Event(Univ):
  """
  Modification History:
  ---------------------
  2010-??-??  patricio     
  2014-08-13  garland   switched the pyfits package to astropy.io.fits
  	              zabblleon@gmail.com 
  2017-06-20  Zacchaeus Fixed comparisons to None
                      zaccysc@gmail.com
  """

  def __init__(self, eventpcf, cwd):

    owd = os.getcwd()
    # os.chdir(cwd)
    tini = time.time()

    # Open new log based on the file name
    logname = os.path.join(owd, cwd, eventpcf[:-4] + "_ini.log")
    log = le.Logedit(logname)
    self.logname = logname

    # initialize Univ
    Univ.__init__(self)
    pcf, = rd.read_pcf(os.path.join(cwd, eventpcf), 'event', expand=False)
    self.initpars(pcf, log)
    self.calc(pcf, log)
    self.read(log)
    self.check(log)
    self.save()

    # Print time elapsed and close log:
    log.writelog("\nOutput files:")
    log.writelog("Data:")
    log.writelog(" " + cwd + '/' + self.eventname + "_ini.dat")
    log.writelog(" " + cwd + '/' + self.eventname + "_ini.h5")
    log.writelog("Log:")
    log.writelog(" " + logname)
    log.writelog("Figures:")
    log.writelog(" " + cwd + '/' + self.eventname + "-fig101.png")

    dt = t.hms_time(time.time()-tini)
    log.writeclose('\nEnd init and read. Time (h:m:s):  %s'%dt)

    if self.runp2:
      os.system("python3 poet.py p2")
      #poet.poet("p2")

  def initpars(self, pcf, log):
    """
      add docstring.
    """

    log.writelog( 'MARK: ' + time.ctime() + ' : New Event Started ')

    # Read Planet Parameters From TepFile
    # ccampo 5/27/2011:
    # origtep is a tepfile with its original units (not converted)
    self.origtep      = tc.tepfile(os.path.join(pcf.topdir[0], pcf.tepfile[0]), conv=False)
    tep               = tc.tepfile(os.path.join(pcf.topdir[0], pcf.tepfile[0]))
    self.tep          = tep
    self.ra           = tep.ra.val
    self.dec          = tep.dec.val
    self.rstar        = tep.rs.val
    self.rstarerr     = tep.rs.uncert
    self.metalstar    = tep.feh.val
    self.metalstarerr = tep.feh.uncert
    self.tstar        = tep.ts.val
    self.tstarerr     = tep.ts.uncert
    self.logg         = tep.loggstar.val
    self.loggerr      = tep.loggstar.uncert
    self.rplan        = tep.rp.val
    self.rplanerr     = tep.rp.uncert
    self.semimaj      = tep.a.val
    self.semimajerr   = tep.a.uncert
    self.incl         = tep.i.val
    self.inclerr      = tep.i.uncert
    self.ephtime      = tep.ttrans.val
    self.ephtimeerr   = tep.ttrans.uncert
    self.period       = tep.period.val      / 86400.0  # conv to days
    self.perioderr    = tep.period.uncert   / 86400.0  # ditto
    self.transdur     = tep.transdur.val
    self.transdurerr  = tep.transdur.uncert
    self.arat    = (self.rplan / self.rstar)**2
    self.araterr = 2*(self.arat)*np.sqrt( (self.rplanerr/self.rplan)**2 + 
                                          (self.rstarerr/self.rstar)**2   )

    # position corrections:
    if pcf.ra[0] is not None:
      self.ra  = s2d.sexa2dec(pcf.ra[0] ) /  12.0 * np.pi 

    if pcf.dec[0] is not None:
      self.dec = s2d.sexa2dec(pcf.dec[0]) / 180.0 * np.pi

    # Convert units to uJy/pix
    self.fluxunits    = pcf.fluxunits[0]

    # Initialize control file parameters
    self.planetname = pcf.planetname
    if np.size(self.planetname) > 1:
      self.planetname = ' '.join(self.planetname)
    else:
      self.planetname = self.planetname[0]
    self.planet       = pcf.planet[0]
    self.ecltype      = pcf.ecltype[0]
    self.photchan     = pcf.photchan[0]
    if self.photchan < 5:
      self.instrument = 'irac'
    elif self.photchan == 5:
      self.instrument = 'irs'
    else:
      self.instrument = 'mips'

    # Instrument contains the instrument parameters
    # FINDME: inherit Instrument instead of adding as a parameter ?
    self.inst = inst.Instrument(self.photchan)

    self.visit     = pcf.visit[0]
    self.sscver    = pcf.sscver[0]

    # Directories
    self.topdir    = pcf.topdir[0]
    self.datadir   = pcf.datadir[0].split('/')
    self.dpref     = (os.path.join(self.topdir, *self.datadir, self.sscver, 'r' ))

    # aors
    self.aorname   = np.array(pcf.aorname, dtype=np.str_)# get aorname as string
    self.aortype   = np.array(pcf.aortype)

    # Number of aors per event
    self.naor      = np.size(self.aorname[np.where(self.aortype == 0)]) 

    # Number of position and noddings 
    self.npos      = pcf.npos[0]
    self.nnod      = pcf.nnod[0]

    # Run next steps:
    self.runp2 = pcf.runp2[0]
    self.runp3 = pcf.runp3[0]

    # Ancil files
    self.hordir    = pcf.hordir[0].split('/')
    self.leapdir   = pcf.leapdir[0].split('/')
    self.kuruczdir = pcf.kuruczdir[0].split('/')
    self.filtdir   = pcf.filtdir[0].split('/')
    self.psfdir    = pcf.psfdir[0].split('/')

    pmaskfile = pcf.pmaskfile[0]
    self.pmaskfile  = [os.path.join(self.dpref+str(aorname), *self.inst.caldir.split('/'), pmaskfile)
                       for aorname in self.aorname[np.where(self.aortype == 0)]]

    self.horvecfile = os.path.join(self.topdir, *self.hordir, pcf.horfile[0])
    self.kuruczfile = os.path.join(self.topdir, *self.kuruczdir, pcf.kuruczfile[0])

    filt = pcf.filtfile
    if self.photchan < 5:
      filt = re.sub('CHAN', str(self.photchan), filt[0])
    elif self.photchan == 5:
      filt = filt[1]
    else: # self.photchan == 5:
      filt = filt[2]
    self.filtfile   = os.path.join(self.topdir, *self.filtdir, filt)

    # Default PSF file:
    if self.photchan < 5  and pcf.psffile[0] == "default":
      self.psffile = os.path.join(self.topdir, *self.psfdir, 'IRAC PSF', 'IRAC.%i.PRF.5X.070312.fits'%self.photchan )
    # User specified PSF file: 
    else:
      self.psffile =  os.path.join(self.topdir, *self.psfdir, pcf.psffile[0])

    # Bad pixels:
    # Chunk size
    self.szchunk      = pcf.szchunk[0]
    # Sigma rejection threshold
    self.sigma        = pcf.sigma
    # User rejected pixels
    self.userrej      = pcf.userrej
    if self.userrej[0] is not None:
      self.userrej = self.userrej.reshape(np.size(self.userrej)/2, 2)
    else:
      self.userrej = None
      
    # set event directory
    self.eventdir = os.getcwd()

    # Denoise:
    self.denoised = False  # Has the data been denoised? 
                           # (modified in pdenoise.py)

  def calc(self, pcf, log):
    """
      Add docstring.
    """
    # Instrument Channel
    self.spitzchan = ( self.photchan if self.photchan <= 4 else  
                                  (0 if self.photchan == 5 else 1) )

    # Name of the event
    comment = str(pcf.comment[0]) if pcf.comment[0] is not None else ''
    self.eventname = ( self.planet           + self.ecltype +
                       str(self.photchan) + str(self.visit) +
                       comment)

    # Added to check whether ancillary data files exist
    self.ispmask  = np.zeros(self.naor, bool)
    for i in range(self.naor):
      self.ispmask[i] = os.path.isfile(self.pmaskfile[i])
    self.ishorvec     = os.path.isfile(self.horvecfile)
    self.iskurucz     = os.path.isfile(self.kuruczfile)
    self.isfilt       = os.path.isfile(self.filtfile  )
    self.ispsf        = os.path.isfile(self.psffile   )

    # Calibration aors:
    self.havepreflash = np.any(self.aortype==1)
    self.havepostcal  = np.any(self.aortype==2)
    self.prenaor      = np.size(self.aorname[np.where(self.aortype==1)])
    self.postnaor     = np.size(self.aorname[np.where(self.aortype==2)])
    self.preaorname   = self.aorname[np.where(self.aortype==1)]
    self.postaorname  = self.aorname[np.where(self.aortype==2)]

    # Array containing the number of expositions per AOR:
    self.nexpid  = np.zeros(self.naor, np.long)
    self.prenexpid  = np.zeros(self.prenaor,  np.long)
    self.postnexpid = np.zeros(self.postnaor, np.long)

    # compile patterns: lines ending with each suffix 
    bcdpattern    = re.compile("(^[^._]" + ".+" + self.inst.bcdsuf    + ")\n",
                               flags=re.M)
    bdmskpattern  = re.compile("(^[^._]" + ".+" + self.inst.bdmsksuf  + ")\n",
                               flags=re.M)
    bdmsk2pattern = re.compile("(^[^._]" + ".+" + self.inst.bdmsksuf2 + ")\n",
                               flags=re.M)

    # Flag in case we don't find any mask files
    self.nomask = False
 
    # Make list of files in each AOR:
    self.bcdfiles = []
    #for aornum in range(self.naor):
    for aornum in (i for i, atype in enumerate(self.aortype) if atype == 0):
      dir = os.path.relpath(os.path.join(self.dpref + self.aorname[aornum] + self.inst.bcddir))
      frameslist = os.listdir(dir)
      framesstring = '\n'.join(frameslist) + '\n'

      # find the data files
      bcdfiles = bcdpattern.findall(framesstring)
      # and sort them
      self.bcdfiles.append(sorted(bcdfiles))

      # find bdmask suffix:
      if bdmskpattern.findall(framesstring) != []:
        self.masksuf = self.inst.bdmsksuf
      elif bdmsk2pattern.findall(framesstring) != []:
        self.masksuf = self.inst.bdmsksuf2
      else:
        log.writelog("WARNING: No mask files found.")
        self.nomask = True

      # get first index of exposition ID, number of expID, and ndcenum
      #                    expid      dcenum     pipev
      first = re.search("_([0-9]{4})_([0-9]{4})_([0-9])",self.bcdfiles[-1][0])
      last  = re.search("_([0-9]{4})_([0-9]{4})_([0-9])",self.bcdfiles[-1][-1])

      self.expadj         = int(first.group(1))
      self.nexpid[aornum] = int(last.group(1)) + 1 - self.expadj
      #self.nexpid[aornum] = len(self.bcdfiles[0])
      self.ndcenum        = int(last.group(2)) + 1
      self.pipev          = int(last.group(3))

      # A catch for missing files (this has happened once in hundreds
      # of hours of observation....). Will not work if nnod > 1
      if self.ndcenum == 0:
        if self.nexpid[aornum] != len(self.bcdfiles[0]):
          self.nexpid[aornum] = len(self.bcdfiles[0])
          print("WARNING: Possible missing files detected. Adjusting nexpid.")

    # List of preflash calibration AORs:
    self.prebcdfiles = []
    for aornum in range(self.prenaor):
      folder = self.dpref + self.preaorname[aornum] + self.inst.bcddir 
      preframesstring = '\n'.join(os.listdir(folder)) + '\n'
      self.prebcdfiles.append(sorted(bcdpattern.findall(preframesstring)))

      first = re.search("_([0-9]{4})_([0-9]{4})_([0-9])",
                        self.prebcdfiles[-1][0])
      last  = re.search("_([0-9]{4})_([0-9]{4})_([0-9])",
                        self.prebcdfiles[-1][-1])
      self.prenexpid[aornum] = int(last.group(1)) + 1 - int(first.group(1))
      self.prendcenum = int(last.group(2)) + 1

    # List of post-calibration AORs:
    self.postbcdfiles = []
    for aornum in range(self.postnaor):
      folder = self.dpref + self.postaorname[aornum] + self.inst.bcddir 
      postframesstring = '\n'.join(os.listdir(folder)) + '\n'
      self.postbcdfiles.append(sorted(bcdpattern.findall(postframesstring)))

      first = re.search("_([0-9]{4})_([0-9]{4})_([0-9])",
                        self.postbcdfiles[-1][0])
      last  = re.search("_([0-9]{4})_([0-9]{4})_([0-9])",
                        self.postbcdfiles[-1][-1])
      self.postnexpid[aornum] = int(last.group(1)) + 1 - int(first.group(1))
      self.postndcenum = int(last.group(2)) + 1

    # pick a random image, not the first
    file = bcdfiles[-2]
    data, head = fits.getdata(os.path.join(dir, file), header=True,
                              ignore_missing_end=True)

    # data size
    shape = data.shape
    if data.ndim >= 3:
      self.nz = shape[0]
      self.ny = shape[1]
      self.nx = shape[2]
    else:
      self.nz = 1
      self.ny = shape[0]
      self.nx = shape[1]

    # Number of lines in the header:
    self.nh = len(head)#.items())

    # Number of small, medium, and big cycles:
    if self.instrument == 'irac':
      self.nbcyc = 1
      self.nmcyc = np.sum(self.nexpid)
      self.nscyc = self.ndcenum if self.nz == 1 else self.nz
      self.prenmcyc  = np.sum(self.prenexpid)
      self.postnmcyc = np.sum(self.postnexpid)
    elif self.instrument == 'irs':
      self.nbcyc = np.sum(self.nexpid)/self.nnod
      self.nmcyc = self.ndcenum
      self.nscyc = 1
    else: # self.instrument == 'mips'
      self.nbcyc = np.sum(self.nexpid)/self.nnod
      self.nmcyc = (self.ndcenum - 1)/7
      self.nscyc = 7

    # Max. number of images per position:
    if self.instrument == 'mips':
      self.maxnimpos  = int(self.nbcyc * (self.nmcyc + 1))
    else:
      self.maxnimpos  = int(np.sum(self.nexpid) * self.ndcenum *
                            self.nz / self.nnod)

    # Calibration maxnimpos:
    if self.havepreflash:
      self.premaxnimpos  = np.sum(self.prenexpid ) * self.prendcenum
    if self.havepostcal:
      self.postmaxnimpos = np.sum(self.postnexpid) * self.postndcenum


    try:
      self.framtime = head['FRAMTIME']   # interval between exposure starts
    except:
      self.framtime = 0.0
    try:
      self.exptime  = head['EXPTIME']    # effective exposure time
    except:
      self.exptime  = None
    try:
      self.gain     = head['GAIN']       # e/DN conversion
    except:
      try:
        self.gain   = head['GAIN1']      # IRS peak-up camera
      except:
        self.gain   = None

    self.bunit      = head['BUNIT']      # Units of image data
    self.fluxconv   = head['FLUXCONV']   # Flux Conv factor (MJy/Str per DN/sec)
    self.posscl       = np.zeros((2,self.npos))
    self.posscl[0, :] = np.abs(head['PXSCAL2']) # ["/pix] axis 2 @ CRPIX1,CRPIX2
    self.posscl[1, :] = np.abs(head['PXSCAL1']) # ["/pix] axis 1 @ CRPIX1,CRPIX2

    if self.spitzchan != head['CHNLNUM']:  # Spitzer photometry channel
      log.writelog( 'poet_calc: photometry channel unexpected')

    # Frequency calculated from wavelength 
    self.freq = self.c / self.inst.spitzwavl


  def read(self, log):
    """
      add docstring.
      Read Data
    """
    pdr.poet_dataread(self, log=log)
    if self.havepreflash:
      pdr.poet_dataread(self, type=1, log=log)
    if self.havepostcal:
      pdr.poet_dataread(self, type=2, log=log)

  def check(self, log):
    """
      add docstring.
    """

    # Source estimated position
    self.srcest = np.zeros((2,self.npos))
    # Changed to median in case of crazy outliers
    for p in range(self.npos):
      self.srcest[0,p] = np.round(np.median(self.fp.heady[p,0:self.nimpos[p]]))
      self.srcest[1,p] = np.round(np.median(self.fp.headx[p,0:self.nimpos[p]]))

    # Plot a reference image
    image = np.zeros((self.ny, self.nx))
    for pos in range(self.npos):
      image += self.data[0,:,:,pos]

    image[np.where(np.isfinite(image) != True)] = 0
    plt.figure(101, (10,9))
    plt.clf()
    plt.imshow(image, interpolation='nearest', origin='lower', cmap=plt.cm.gray)
    plt.plot(self.srcest[1,:], self.srcest[0,:],'r+')
    plt.xlim(0,self.nx-0.5)
    plt.ylim(0,self.ny-0.5)
    plt.title(self.eventname + ' reference image')
    plt.savefig(os.path.join(self.topdir, self.eventname + "-fig101.png"))

    # Throw a warning if the source estimate position lies outside of
    # the image.
    wrning = False
    if (np.any(self.srcest[1,:] < 0) or np.any(self.srcest[1,:] > self.nx - 1)
     or np.any(self.srcest[0,:] < 0) or np.any(self.srcest[0,:] > self.ny - 1)):
      wrning = True

    # Write to log
    log.writelog('\n%s: event %s'%(self.planetname, self.eventname))
    log.writelog('nexpid  = ' + np.str(self.nexpid))
    log.writelog('ndcenum = %d'%self.ndcenum)
    #log.writelog('you should see %d positions in the output image '%self.npos +
    #            '(red crosses)')
    log.writelog("Target guess position:\n" + str(self.srcest[0:2,:]))
    if wrning:
       log.writelog('Source position estimate out of bounds!')
    log.writelog('nimpos  = ' + np.str(self.nimpos))
    log.writelog('Read %d frames\n'%np.sum(self.nimpos))

    # Report files not found:
    print("Ancil Files:")
    if not self.ispmask[0]:
      log.writelog('Pmask:   File not found!')
    else:
      log.writelog("Pmask:   " + self.pmaskfile[0])
    if not self.ishorvec:
      log.writelog('Horizon: File not found!')
    else:
      log.writelog("Horizon: " + self.horvecfile)
    if not self.iskurucz:
      log.writelog('Kurucz:  File not found!')
    else:
      log.writelog("Kurucz:  " + self.kuruczfile)
    if not self.isfilt:
      log.writelog('Filter:  File not found!')
    else:
      log.writelog("Filter:  " + self.filtfile)
    if not self.ispsf:
      log.writelog('PSF:     Not supplied.')
    else:
      log.writelog("PSF:     " + self.psffile)

    if self.exptime  is None:
      log.writelog("Exposure time undefined.")
    if self.gain     is None:
      log.writelog("Gain undefined.")

  def save(self):
    # what to load in p2_badpix
    self.loadnext = ['data', 'uncd', 'bdmskd']

    print("Saving event.")
    
    if self.instrument == 'mips':
      self.loadnext.append('brmskd')
      saveevent(self, os.path.join(self.topdir, self.eventname + "_ini"),
                save=['data', 'uncd', 'head', 'bdmskd', 'brmskd'])
    else:
      saveevent(self, os.path.join(self.topdir, self.eventname + "_ini"), delete=['brmskd'],
                save=['data', 'uncd', 'head', 'bdmskd'])

