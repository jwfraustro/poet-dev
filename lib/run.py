#! /usr/bin/env python

# $Author$
# $Revision$
# $Date$
# $HeadURL$
# $Id$

"""
TO EXECUTE DIRECTLY FROM BASH, USE THE FOLLOWING SYNTAX:
./run.py p5idl directory
./run.py p5 directory clip
./run.py p6 modeldir directory clip idl
./run.py p7 directory idl islak
./run.py p8 directory idl eclphase
./run.py p9 directory idl
Arguments after directory are optional but must remain in order. 
modeldir:    String to append to the default directory name.
            A new directory is ALWAYS created in BASH mode.
directory:   Location of the savefile to restore, default is './'.
clip:      
           data set to model and plot only a portion of the entire light curve.
           Good for modeling a transit or eclipse in an around-the-orbit data
           set.
           Syntax is start:end (eg. 0:70000 or -60500:None), defult is None
           for full data set.
idl:       False (default), set True when using IDL photometry.
islak:     True (default), set False to not load 'allknots' into memory and
           skip Figs 701 & 702.
Figures will NOT display in BASH mode, but they will still save.
"""

from __future__ import print_function
import os, sys
os.environ['OMP_NUM_THREADS']='12'
try:
    sys.path.remove('/home/esp01/code/lib/python')
except:
    pass

def in_ipython():
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        return True

import numpy       as np
try:
    import cPickle   as pickle
except:
    import pickle
import np_unpickle as unpic
import matplotlib  as mpl
import multiprocessing as mp

if not in_ipython():
    mpl.use("Agg")

import matplotlib.pyplot as plt
import datetime, shutil, time
import reader3  as rd
import p5checks as p5
import p6model  as p6
import p7anal   as p7
import p8tables as p8
import p9figs   as p9
import manageevent as me
import printoutput as po
import plots
import mccubed as mc3

isinteractive = False   # Set True and use '-pylab' to turn on
                        # displaying plots in interactive mode.

# Use the commands below when in interactive mode.  Only copy and
# paste the lines that you need.
def interactive():
    '''IDL'''
    # Run p5checks or restore after p5checks
    # directory:   Location of the savefile to restore.
    events = p5idl(directory='./')
    events = p5idlRestore(directory='./')

    '''POET'''
    # Restore after p5checks
    # directory:   Location of the savefile to restore.

    # clip:      Clip data set to model and plot only a portion of the entire
    #            light curve.  Good for modeling a transit or eclipse in an 
    #            around-the-orbit data set.  Syntax is "<start>:<end>" or the
    #            tuple: (start, end)
    #            (eg. "0:70000" or -60500,None to use last 60500 points).
    events = poetRestore(directory='../')

    # Run p6model
    # event:     If not specified, event will be restored using poetRestore 
    #            or p5idlRestore.
    # modeldir:  Creates a new model directory when with the date if None
    #            (default). If modeldir is a string, make that the modeldir's
    #            name.
    # directory:   If event is not specified, location of the savefile to
    #            restore.
    # idl:       False (default), set True when using IDL photometry.
    p6model(events)
    p6model(events, modeldir=False)
    p6model(events, modeldir='test')

    # Run p7anal, p8tables, and p9figs
    # event:     If not specified, event will be restored using p6Restore.
    # directory:   If event is not specified, location of the savefile to restore.
    # idl:       False (default), set True when using IDL photometry.
    # islak:     True (default), set False to not load 'allknots' into memory 
    #            and skip Figs 701 & 702 (p7anal only).
    # eclphase:  Nominal eclipse phase, default is 0.5 (p8tables only).
    p7anal(events)
    p8tables(events)
    p9figs(events)
    
    # Restore after p6model or p7anal, if necessary
    # You can use os.listdir('./') to find the desired model directory.
    # directory:   Location of the savefile to restore.
    # idl:       False (default), set True when using IDL photometry.
    events = p6Restore(directory = '2011-02-25_14:53-testdir')
    events = p6Restore(directory='./')
    events = p7Restore(directory='./')
    
    # Overwrite the savefile for a single event after running poetRestore, 
    # p6model or p7anal.
    # Do not use these commands unless you are sure you want to overwrite an
    # existing savefile.
    # directory:   Location of the savefile to restore (poetSave only).
    #  (poetSave only).
    poetSave(event[0], directory='../')
    p6Save(event[0])
    p7Save(event[0])

    return

########################################################
#                                                      #
#   DO NOT RUN CODE INTERACTIVELY BEYOND THIS POINT!   #
#                                                      #
########################################################

# RUN p5checks AFTER IDL PHOTOMETRY
def p5idl(directory):

    # Read in HDF5 file
    hdf5files  = []
    for fname in os.listdir(directory):
        if (fname.endswith(".h5")):
            hdf5files.append(fname)
    hdf5files.sort()

    print('Using the following HDF5 files:', hdf5files)

    #Run p5checks
    events     = []
    filename  = ''
    for i, hdf5file in enumerate(hdf5files):
        event = p5.checks2(hdf5file, i)
        filename += event.eventname
        events.append(event)

    #p5checks - save
    for event in events:
        event.filename = filename
        savefile  = "d-" + event.eventname + "-5checks.dat"
        handle    = open(savefile, 'wb')
        pickle.dump(event, handle)
        handle.close()

    return events

# RESTORE SAVEFILE AFTER p5checks USING IDL PHOTOMETRY
def p5idlRestore(directory='./'):

    loadfile = []
    for fname in os.listdir(directory):
        if (fname.endswith("5checks.dat")):
            loadfile.append(fname)
    loadfile.sort()

    events = []
    for lfile in loadfile:
        print("Loading " + lfile)
        handle = open(lfile, 'rb')
        try:
            events.append(pickle.load(handle))
        except:
            events.append(unpic.unpickle_old_pyfits(lfile))
        handle.close()

    return events

# RESTORE SAVEFILE FROM POET
def poetRestore(directory='../', clip=None, rundir=None, modeldir=None, params_override=None):
    
    files    = []
    events   = []
    filename = ''
    for fname in os.listdir(directory):
        if (fname.endswith("_p5c.dat")):
            files.append(fname[:-4])
    files.sort()

    if len(files) == 0:
        print('Cannot find any files to restore.')
        return []

    for f in files:

        # Load event
        event = me.loadevent(directory + '/' + f)
        events.append(event)
        print('Finished loading: ' + event.eventname)
        filename = filename + event.eventname
        event.ancildir   = directory + '/' + modeldir + '/'


        # Clip data set to model and plot only a portion of the entire
        # light curve.  Good for modeling a transit or eclipse in an
        # around-the-orbit data set
        if clip is not None and clip != 'None':

            if type(clip) == str:

                #Convert from string to 2 ints
                start, end = clip.split(':',1)

                try:
                    start = int(start)
                except:
                    print("Error with format of optional variable clip.")
                    return []

                try:
                    end   = int(end)
                except:
                    end   = None

            else:
                if len(clip) == 2:
                    start, end = clip
                else:
                    start = clip[0]
                    end   = None

            # Use only data points from 'start' to 'end'
            event.phase      = event.phase [:,start:end]
            event.aplev      = event.aplev [:,start:end]
            event.aperr      = event.aperr [:,start:end]
            event.good       = event.good  [:,start:end]
            event.time       = event.time  [:,start:end]
            event.y          = event.y     [:,start:end]
            event.x          = event.x     [:,start:end]
            event.juldat     = event.juldat[:,start:end]
            event.bjdutc     = event.bjdutc[:,start:end]
            event.bjdtdb     = event.bjdtdb[:,start:end]

    for event in events:

        # Copy params file into output dir.
        paramsfile = event.eventname + '.pcf'
        event.paramsfile = directory + '/' + modeldir + '/' + paramsfile
        if not os.path.isfile(event.paramsfile):

            if params_override:
                mod = {"params" : params_override}
            else:
                mod = {}
            rd.copy_config(rundir + '/' + paramsfile, ['params'],
                           event.paramsfile, 'w', mod)

        # Copy initial values file into output dir
        initvfile = rd.read_pcf(event.paramsfile, "params",
                                simple=True).modelfile
        event.initvalsfile = directory + '/' + modeldir + '/' + initvfile
        if not os.path.isfile(event.initvalsfile):

            if os.path.isfile(rundir + '/' + initvfile):
                shutil.copy(rundir + '/' + initvfile,
                            event.initvalsfile)

            else:
                shutil.copy(rundir + '/initvals.txt',
                            event.initvalsfile)

    return events

# OVERWRITE event SAVEFILE AFTER RUNNING p5checks in POET
def poetSave(event, directory='../'):
    me.saveevent(event, directory + "/" + event.eventname + "_p5c")

# RUN p6model
def p6model(events=None, directory='./', modeldir=None, clip=None,
            idl=False, mode='full', numit=None, nodate=True, rundir=None,
            justplot=False, isinteractive=isinteractive, control=None,
            retmoddir=False, params_override=None):

    # adjust modeldir
    if modeldir is None or not nodate:
        date = datetime.datetime.today().strftime("%Y-%m-%d_%H_%M")

        # if modeldir is not None:
        #     modeldir = date + modeldir + '/'

    # If a control is given, recursively
    # call on all paths specified in control.
    if control is not None:

        if isinstance(control, dict):
            p6pcf = rd.Pcf(control)
        else:
            p6pcf = rd.read_pcf(control, 'p6', simple=True,
                                d1lists=["centering", "photometry"],
                                d2lists=["path"])

        paths = getp6filedirs(p6pcf, directory)

        if modeldir is None:
            modeldir = p6pcf.modeldir

        if modeldir is None:
            modeldir = date
        elif not nodate:
            modeldir = date + modeldir + '/'

        if p6pcf.threads < 1:
            print("ERROR: less than 1 threads specified in p6.pcf."
                  "\nExiting...")
            return

        procs = []
        for path in paths:
            while len(mp.active_children()) >= p6pcf.threads:
                time.sleep(10)
            p = mp.Process(target=p6model,
                           kwargs={"events":events, "directory":path,
                                   "modeldir":modeldir, "clip":clip,
                                   "idl":idl, "mode":mode, "rundir":rundir,
                                   "numit":numit, "nodate":True,
                                   "isinteractive":isinteractive,
                                   "control":None,
                                   "params_override":params_override})
            p.start()
            procs.append(p)

        for proc in procs:
            proc.join()

        print("Making p6 plots.")
        if retmoddir:
            return modeldir, plots.p6plots(modeldir, paths, directory)
        return plots.p6plots(modeldir, paths, directory)

    if modeldir is None:
        modeldir = date

    # Create new directory for model run with custom name:
    if os.path.isdir(directory + '/' + modeldir):
        print("path :", directory + '/' + modeldir,
              "already exists.\nSkipping this p6 run...")
        return "skip"

    try:
        os.mkdir(directory + '/' + modeldir)
    except:
        print("WARNING: Specified model directory already exists.")

    if events == None:
        if idl:
            events = p5idlRestore(directory)
        else:
            events = poetRestore(directory, clip, rundir, modeldir,
                                 params_override=params_override)

    numevents = len(events)

    if numevents == 0:
        print("Event object is empty.")
        return

    # Reload model parameters:
    nummodels = np.zeros(numevents, dtype=int)
    # attributes to be converted to 1d-lists
    d1lists = ['modelfile', 'numit', 'preclip', 'postclip',
               'priorvars', 'xstep', 'ystep', 'minnumpts',
               'sssteps', 'nx', 'ny', 'sx', 'sy']
    # attributes to be converted to 2d-lists
    d2lists = ['model', 'ortholist', 'interclip', 'priorvals',
               'ipclip']
    # attributes to be converted to 2d-lists
    nparrs = ['numit', 'priorvals']
    for i, event in enumerate(events):

        event.params = rd.read_pcf(event.paramsfile, 'params',
                                       simple=True, d1lists=d1lists,
                                       d2lists=d2lists, nparrs=nparrs)
        if params_override:
            for key in params_override:
                setattr(event.params, key, params_override[key])

        event.fit = []
        event.modeldir = directory + '/' + modeldir
        nummodels[i] = len(event.params.model)
        if i > 0 and nummodels[i] != nummodels[i-1]:
            print("WARNING: Number of models in each event does not"
                  + "match.")

    # Initialize output type: stdout, a file object or a file
    printout = po.init(events[0].params.printout, events)

    # Execute p6 setup
    for i in range(nummodels.min()):
        p6.setup(events, i, printout, mode=mode, numit=numit)

    printout = po.init(events[0].params.printout, events)
    dataout = open(directory + '/plotpoint.txt', 'a')
    for i in range(nummodels.min()):
        p6.finalrun(events, i, printout, numit=numit)
        for event in events:
            p6Save(event, directory + '/' + modeldir)

    # Print parameters used for comparison:
    print("\nBest-fit eclipse depths or transit radius ratios with"
          + " errors:", file=printout)
    for event in events:
        event.minbic = np.inf  # Min BIC value of all fits for one
                               # event
        print(event.eventname, file=printout)
        print(event.eventname, file=dataout)
        for fit in event.fit:

            if hasattr(fit.i,'depth'):
                print(fit.bestp[fit.i.depth],
                      fit.std  [fit.i.depth],
                      fit.saveext, fit.bsigchi, file=printout)
                print(fit.bestp[fit.i.depth],
                      fit.std  [fit.i.depth],
                      fit.saveext, fit.bsigchi, file=dataout)

            if hasattr(fit.i,'depth2'):
                print(fit.bestp[fit.i.depth2],
                      fit.std  [fit.i.depth2],
                      fit.saveext, fit.bsigchi, file=printout)
                print(fit.bestp[fit.i.depth2],
                      fit.std  [fit.i.depth2],
                      fit.saveext, fit.bsigchi, file=dataout)

            if hasattr(fit.i,'depth3'):
                print(fit.bestp[fit.i.depth3], 
                      fit.std  [fit.i.depth3],
                      fit.saveext, fit.bsigchi, file=printout)
                print(fit.bestp[fit.i.depth3], 
                      fit.std  [fit.i.depth3],
                      fit.saveext, fit.bsigchi, file=dataout)

            if hasattr(fit.i,'trrprs'):
                print(fit.bestp[fit.i.trrprs], 
                      fit.std  [fit.i.trrprs],
                      fit.saveext, fit.bsigchi, file=printout)
                print(fit.bestp[fit.i.trrprs], 
                      fit.std  [fit.i.trrprs],
                      fit.saveext, fit.bsigchi, file=dataout)

            if hasattr(fit.i,'trrprs2'):
                print(fit.bestp[fit.i.trrprs2], 
                      fit.std  [fit.i.trrprs2],
                      fit.saveext, fit.bsigchi, file=printout)
                print(fit.bestp[fit.i.trrprs2], 
                      fit.std  [fit.i.trrprs2],
                      fit.saveext, fit.bsigchi, file=dataout)

            if hasattr(fit.i,'rprs'):
                print(fit.bestp[fit.i.rprs], 
                      fit.std  [fit.i.rprs],
                      fit.saveext, fit.bsigchi, file=printout)
                print(fit.bestp[fit.i.rprs], 
                      fit.std  [fit.i.rprs],
                      fit.saveext, fit.bsigchi, file=dataout)

            if hasattr(fit.i,'rprs2'):
                print(fit.bestp[fit.i.rprs2], 
                      fit.std  [fit.i.rprs2],
                      fit.saveext, fit.bsigchi, file=printout)
                print(fit.bestp[fit.i.rprs2], 
                      fit.std  [fit.i.rprs2],
                      fit.saveext, fit.bsigchi, file=dataout)

            if hasattr(fit.i,'trdepth'):
                print(fit.bestp[fit.i.trdepth], 
                      fit.std  [fit.i.trdepth],
                      fit.saveext, fit.bsigchi, file=printout)
                print(fit.bestp[fit.i.trdepth], 
                      fit.std  [fit.i.trdepth],
                      fit.saveext, fit.bsigchi, file=dataout)

            if hasattr(fit.i,'gdepth'):
                print(fit.bestp[fit.i.gdepth], 
                      fit.std  [fit.i.gdepth],
                      fit.saveext, fit.bsigchi, file=printout)
                print(fit.bestp[fit.i.gdepth], 
                      fit.std  [fit.i.gdepth],
                      fit.saveext, fit.bsigchi, file=dataout)                

            event.minbic = np.min((event.minbic,fit.bic))

    # Print SDNR for joint fit if more than one event:
    if len(events) > 1:

        # Joint-fit normalized residuals:
        jnres = np.array([])
        for event in events:

            # Force the normalized residuals to 1D shape:
            nres = event.fit[0].normresiduals.flatten()
            # Concatenate values:
            jnres = np.concatenate((jnres,nres))

        # Joint SDNR:
        jsdnr = np.std(jnres)
        print("\nJoint-fit SDNR: %9.7f"%jsdnr, file=printout)

    print("\n S/N SDNR \xce\x94BIC MODEL NUMIT "
          + "BIN_SZ(y,x) MinNumPts", file=printout)
    # Delta = '\xce\x94' in utf-8
    for event in events:

        print(event.eventname, file=printout)
        print(event.eventname, file=dataout)
        minbic = event.minbic
        for i, fit in enumerate(event.fit):
            try:

                sdnr  = fit.sdnr
                bic   = fit.bic - minbic
                model = fit.saveext
                numit = event.params.numit[1]

                if len(event.params.ystep) == \
                   len(event.params.model):
                    ystep, xstep = event.params.ystep[i],\
                                   event.params.xstep[i]

                else:
                    ystep, xstep = event.params.ystep[0],\
                                   event.params.xstep[0] 

                if len(event.params.minnumpts) == \
                   len(event.params.model):
                    minnumpts = event.params.minnumpts[i] 

                else:
                    minnumpts = event.params.minnumpts[0]

                # extract string code
                ec ='%8.4f %9.7f %8.1f %11s %7.1e %6.3f,%5.3f %4.0f'

                print(sdnr, file=dataout)
                if hasattr(fit.i,'depth'):
                    snr = fit.bestp[fit.i.depth] /\
                          fit.std  [fit.i.depth]
                    print(ec%(snr, sdnr, bic, model, numit, ystep,
                              xstep, minnumpts), file=printout)

                if hasattr(fit.i,'depth2'):
                    snr = fit.bestp[fit.i.depth2] /\
                          fit.std  [fit.i.depth2]
                    print(ec%(snr, sdnr, bic, model, numit, ystep,
                              xstep, minnumpts), file=printout)

                if hasattr(fit.i,'depth3'):
                    snr = fit.bestp[fit.i.depth3] /\
                          fit.std  [fit.i.depth3]
                    print(ec%(snr, sdnr, bic, model, numit, ystep,
                              xstep, minnumpts), file=printout)

                if hasattr(fit.i,'trrprs'):
                    snr = fit.bestp[fit.i.trrprs] /\
                          fit.std  [fit.i.trrprs]
                    print(ec%(snr, sdnr, bic, model, numit, ystep,
                              xstep, minnumpts), file=printout)

                if hasattr(fit.i,'trrprs2'):
                    snr = fit.bestp[fit.i.trrprs2] /\
                          fit.std  [fit.i.trrprs2]
                    print(ec%(snr, sdnr, bic, model, numit, ystep,
                              xstep, minnumpts), file=printout)

                if hasattr(fit.i,'rprs'):
                    snr = fit.bestp[fit.i.rprs] /\
                          fit.std  [fit.i.rprs]
                    print(ec%(snr, sdnr, bic, model, numit, ystep,
                              xstep, minnumpts), file=printout)

                if hasattr(fit.i,'rprs2'):
                    snr = fit.bestp[fit.i.rprs2] /\
                          fit.std  [fit.i.rprs2]
                    print(ec%(snr, sdnr, bic, model, numit, ystep,
                              xstep, minnumpts), file=printout)

                if hasattr(fit.i,'trdepth'):
                    snr = fit.bestp[fit.i.trdepth] /\
                          fit.std  [fit.i.trdepth]
                    print(ec%(snr, sdnr, bic, model, numit, ystep,
                              xstep, minnumpts), file=printout)

                if hasattr(fit.i,'gdepth'):
                    snr = fit.bestp[fit.i.gdepth] /\
                          fit.std  [fit.i.gdepth]
                    print(ec%(snr, sdnr, bic, model, numit, ystep,
                              xstep, minnumpts), file=printout)          

            except:
                print("Error calculating values. %13s" %
                        event.fit[i].saveext, file=printout)

    po.close(printout)

    if isinteractive == False:
       plt.close('all')

    else:
       plt.show()

    return event.modeldir

# Save event after p6model
def p6Save(event, savedir, mode='burn'):
    savefile  = "d-" + event.eventname + "-6model.dat"
    handle    = open(savedir + '/' + savefile, 'wb')
    pickle.dump(event, handle)
    handle.close()
    return

# Restore savefile after p6model
def p6Restore(directory='./', idl=False):

    loadfile = []
    for fname in os.listdir(directory):
        if (fname.endswith("6model.dat")):
            loadfile.append(directory + "/" + fname)
    loadfile.sort()

    #print("Loading files:", loadfile)
    #loadfile = [loadfile[-1]]   #***EDIT THIS LINE MANUALLY***
    if len(loadfile) == 0:
        print('Cannot find any files to restore.')
        return []

    events = []
    for lfile in loadfile:
        print("Loading " + lfile)
        handle = open(lfile, 'rb')
        #try:
        events.append(pickle.load(handle))
        #except:
            #events.append(unpic.unpickle_old_pyfits(lfile))
        handle.close()

    return events

#RUN 7anal
def p7anal(events=None, directory='./', idl=False, islak=False,
            isinteractive=isinteractive):

    if events == None:
        events = p6Restore(directory, idl)

    nummodels = np.array([len(event.params.model) for event in events],
                             dtype=int)

    printout = po.init(events[0].params.printout, events)
    for j, event in enumerate(events):
        print("\n" + event.eventname, file=printout)
        minnum = nummodels.min()
        for i in range(minnum):
            print("\nCurrent model = " + str(event.params.model[i]),
                    file=printout)
            p7.stdanal(event, event.fit[i], j*minnum+i,
                    printout, islak)
            p7Save(event, directory)
    po.close(printout)

    if isinteractive == False:
        plt.close('all')
    else:
        plt.show()
    return

#SAVE event AFTER p7anal
def p7Save(event, savedir):
    savefile  = "d-" + event.eventname + "-7anal.dat"
    handle    = open(savedir + '/' + savefile, 'wb')
    pickle.dump(event, handle)
    handle.close()
    return

#RESTORE SAVEFILE AFTER p7anal
def p7Restore(directory='./', idl=False):

    loadfile = []
    for fname in os.listdir(directory):
        if (fname.endswith("7anal.dat")):
            loadfile.append(directory + "/" + fname)
    loadfile.sort()

    if len(loadfile) == 0:
        print('Cannot find any files to restore.')
        return []

    events = []
    for lfile in loadfile:
        print("Loading " + lfile)
        handle = open(lfile, 'rb')
        try:
            events.append(pickle.load(handle))
        except:
            events.append(unpic.unpickle_old_pyfits(lfile))
        handle.close()

    return events

# Run p8tables
def p8tables(events=None, directory='./', idl=False, eclphase=0.5):

    if events == None:
        events = p7Restore(directory, idl)

    nummodels = np.array([len(event.params.model) for event in events],
                             dtype=int)

    printout = po.init(events[0].params.printout, events)
    for event in events:
        print("\n" + event.eventname, file=printout)
        event.meanphase = eclphase
        for i in range(nummodels.min()):
            print("\nCurrent model = " + str(event.params.model[i]),
                  file=printout)
            p8.tables(event, i, printout)
    po.close(printout)

    return

# Run p9figs
def p9figs(events=None, directory='./', idl=False, titles=True,
            isinteractive=isinteractive):

    if events == None:
        events = p7Restore(directory, idl)

    nummodels = np.array([len(event.params.model) for event in events],
                         dtype=int)


    printout = po.init(events[0].params.printout, events)
    for j, event in enumerate(events):
        print("\n" + event.eventname, file=printout)
        nummin = nummodels.min()
        for i in range(nummin):
            print("\nCurrent model = " + str(event.params.model[i]),
                    file=printout)
            p9.figs (event, i, j*nummin+i, titles=titles)
    po.close(printout)

    if isinteractive == False:
        plt.close('all')
    else:
        plt.show()

    return

def getp6filedirs(data, directory):

    # retrieve centers if not explicitly given
    if len(data.centering) != 0 and data.centering[0] == 'all':

        centerings = ['fgc', 'col', 'ccl', 'lag', 'lac', 'ipf', 'bpf']
        centers = []

        for name in os.listdir(directory):

            if name.count("_"):
                code = name[:name.index("_")]
            else:
                code = name

            if code in centerings and os.path.isdir(directory + '/' + name):
                centers.append(directory + '/' + name)

    # if centering is specified, use those
    else:
        centers = [directory + '/' + name for name in data.centering]

    paths = []

    # search for phot dirs if not explicitly given
    if len(data.photometry) != 0 and data.photometry[0] == 'all':
        for center in centers:
            for name in os.listdir(center):
                if os.path.isdir(center + '/' + name):
                    paths.append(center + '/' + name + '/')

    # otherwise, use given photometry
    else:
        for center in centers:
            for phot in data.photometry:

                path = center + '/' + phot

                if os.path.isdir(path):
                    paths.append(path)

                else:
                    print("unable to find cent/phot directory:", path)

#    for path in data.path:
        

    return paths

