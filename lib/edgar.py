#! /usr/bin/env python3

# Easy
# Distributed
# Global
# Aggregator of
# Runs


import astropy.io.fits as fits
import configparser    as cfg
import datetime
import multiprocessing
import numpy           as np
import os
import shutil
import sys
import time
sys.path.append('lib')

import instrument as inst
import models_c   as mc
import paramedit
import poet
import reader3    as rd


cfgdir     = 'lib/cfgs/'
logf       = 'edgar.log'


def edgar(*args, **kwargs):
    """This function reads the "poet.pcf" file in the current working
directory, and runs the various parts of POET on the specified
data. EDGAR keeps track of memory and CPU usage to partially
serialize the highly parallelized code so large amount of computation
can be performed with minimal interaction (It doesn't run everything
at once and crash your computer).

Parameters
----------
task : string, optional
    Specifies which task to begin with. Can take on values in {"read",
    "mask", "p6_init", "p6_full", "p6_check", "p6_final", "finish",
    "zen"}. If not specified, EDGAR will start in "read". After
    finishing one task, EDGAR will proceed to the next task in the
    list unless otherwise specified in the "poet.pcf" control file.

*args : strings, optional
    optional sequence of arguments passed to the starting task. In
    general, you will only use this after running EDGAR, during which
    EDGAR will produce print statements giving what arguments to use
    here to resume from a given spot. See individual task functions if
    you would like to abuse this functionality (not recommended).

Returns
-------
string
    string returned by the last task attempted. Will be "quit" if the
task did not fully run. Will be "Done" if everything completed
successfully.

Notes
-----
Running and part of EDGAR will set all "runp#" parameters in your
event pcf to false.

The functionality of each task is documented in that task's function's
docstring. The function name is the same as the string corresponding
to it {read, mask, p6_init, p6_full, p6_check, p6_final, finish, zen}.

Each function receives the edpcf and poetpcf objects instead of
rereading it so EDGAR can be run multiple times concurrently from the
same working directory (start EDGAR, edit configs, start another EDGAR)

    """
    edpcf     = rd.read_pcf("poet.pcf", "EDGAR", simple=True,
                            d2lists=['p6model', 'zenmodels'])
    poetpcf    = rd.read_pcf("poet.pcf", "POET", simple=True)
    args = [poetpcf, edpcf] + list(args)

    if "task" in kwargs.keys():
        task = kwargs["task"]
    elif len(args) > 2:
        task = args.pop(2)
    else:
        task = "read"

    edPrint("\n"+(79*'=')+'\nNew EDGAR run starting with task:', task,
            "\nRundir is:", poetpcf.rundir,
            "\nTime is:", datetime.datetime.now(),
            "\n"+(79*'='))

    if edpcf.cores < 1:
        edPrint("Cannot use less than 1 core. Exiting...")
        return ()

    # change config to not run next portions on their own
    pcfname = os.path.relpath(os.path.join(poetpcf.rundir, poetpcf.eventname+'.pcf'))
    eventpcf = open(pcfname, 'r').readlines()
    for i, line in enumerate(eventpcf):
        for num in ('2', '3', '4', '5'):
            if line.startswith("runp"+num):
                eventpcf[i] = "runp" + num + " False\n"
    open(pcfname, 'w').writelines(eventpcf)

    # Run parts of EDGAR starting from where is specified
    resume = [read, mask, centphot, p6_init, p6_full, p6_check, p6_final,
              finish, zen]
    start  = {
        "read" : 0,
        "mask" : 1,
        "centphot" : 2,
        "p6_init" : 3,
        "p6_full" : 4,
        "p6_check" : 5,
        "p6_final" : 6,
        "finish" : 7,
        "zen" : 8
        }
    for func in resume[start[task]:]:
        args = func(*args)
        if isinstance(args, str):
            return args

def read(poetpcf, edpcf):
    """Reads in data and generates an event object (runs p1).

Parameters
----------
poetpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    POET section of poet.pcf. This indirectly specifies the run
    directory and the event pcf to use.

edpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    EDGAR section of poet.pcf.

Returns
-------
string or tuple
    If read completes successfully, returns a tuple of (poetpcf,
    edpcf) reader3.pcf objects. If read is unsuccessful, returns the
    string "quit".

Notes
-----
If you want to pass in your own pcf object, you should generate them like so:
poetpcf = rd.read_pcf("poet.pcf", "POET", simple=True)
edpcf   = rd.read_pcf("poet.pcf", "EDGAR", simple=True,
                                d1lists=['zenmodels'])

    """

    if isinstance(poetpcf, str):
        poetpcf = rd.read_pcf("poet.pcf", "POET", simple=True)
    if isinstance(edpcf, str):
        edpcf     = rd.read_pcf("poet.pcf", "EDGAR", simple=True,
                                d1lists=['zenmodels'])

    edPrint("starting initial read in.")

    returnPrint("read")

    p1size = guessp1size(poetpcf)
    if p1size > edpcf.memory:
        edPrint("You asked me to start a POET run. However, creating an Event"
                " object requires approximately", p1size, "MB of RAM and you "
                "only allotted me", edpcf.memory, "MB of RAM. Either increase 'mem"
                "ory' in poet.pcf or choose a smaller set of data.")
        return "quit"

    #run p1
    poet.p(1, poetpcf=poetpcf)

    return (poetpcf, edpcf)

def mask(poetpcf, edpcf):
    """Generates a bad pixel mask for the data in your event (runs p2).

Parameters
----------
poetpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    POET section of poet.pcf. This indirectly specifies the run
    directory and the event pcf to use.

edpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    EDGAR section of poet.pcf.

Returns
-------
string or tuple
    If mask completes successfully, returns a tuple of (poetpcf,
    edpcf) reader3.pcf objects. If read is unsuccessful, returns the
    string "quit".

Notes
-----
If you want to pass in your own pcf object, you should generate them like so:
poetpcf = rd.read_pcf("poet.pcf", "POET", simple=True)
edpcf   = rd.read_pcf("poet.pcf", "EDGAR", simple=True,
                                d1lists=['zenmodels'])

    """

    edPrint("starting to mask bad pixels.")

    returnPrint("mask")

    # file size in bytes
    datsize = os.path.getsize(os.path.join(poetpcf.rundir, poetpcf.eventname+"_ini.dat"))
    h5size  = os.path.getsize(os.path.join(poetpcf.rundir, poetpcf.eventname+"_ini.h5" ))
    # total size in MB
    p2size = 200 + (datsize + h5size)/1024**2
    
    if p2size > edpcf.memory:
        edPrint("Generating bad pixel masks will require", p2size, "MB of RAM"
                " but you only allotted me", edpcf.memory, "MB of RAM. Either incr"
                "ease 'memory' in poet.pcf or reduce the memory required by p"
                "2.")
        return "quit"

    # run p2
    poet.p(2, poetpcf=poetpcf)

    return (poetpcf, edpcf)

def centphot(poetpcf, edpcf):
    """Runs centering, photometry, and checks (p3, p4, & p5). 

Parameters
----------
poetpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    POET section of poet.pcf. This indirectly specifies the run
    directory and the event pcf to use.

edpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    EDGAR section of poet.pcf.

Returns
-------
string or tuple
    If centphot completes successfully, returns a tuple of (poetpcf,
    edpcf) reader3.pcf objects. If read is unsuccessful, returns the
    string "quit".

Notes
-----
If you want to pass in your own pcf object, you should generate them like so:
poetpcf = rd.read_pcf("poet.pcf", "POET", simple=True)
edpcf   = rd.read_pcf("poet.pcf", "EDGAR", simple=True,
                                d1lists=['zenmodels'])
"""
    edPrint("I am queuing up centering and photometry tasks...")
    returnPrint("centphot")

    config = poetpcf.eventname+'.pcf'

    # get pcfs and queue up tasks
    cpcfs = rd.read_pcf(os.path.join(poetpcf.rundir, config), 'centering')
    ppcfs = rd.read_pcf(os.path.join(poetpcf.rundir, config), 'photometry')
    p3tasks = []
    p4tasks = []
    p5tasks = []
    for cpcf in cpcfs:

        # make directory and place pcfs
        center = cpcf.method if cpcf.pcfname is None else cpcf.method + '_' + cpcf.pcfname
        cdir = os.path.join(poetpcf.rundir, center)
        os.makedirs(cdir, exist_ok=True)
        cpcf.make_file(os.path.join(cdir, config), "centering")
        rd.copy_config(os.path.join(poetpcf.rundir, config), 'photometry', os.path.join(cdir, config))

        # add task
        p3task = Task(poetpcf.rundir, poetpcf.eventname, cpcf.ccores, poet.p, args=(3, center), kwargs={"poetpcf":poetpcf})
        p3tasks.append(p3task)

        for ppcf in ppcfs:

            # check that elliptical photometry is only used on fgc or rfgc
            if ppcf.phottype == 'ell':
                if cpcf.method not in ['fgc', 'rfgc']:
                    continue
            # make photometry directory and place pcfs
            photom = photname(ppcf)
            pdir = os.path.join(cdir, photom)
            os.makedirs(pdir, exist_ok=True)
            ppcf.make_file(os.path.join(pdir, config), "photometry")

            # add photometry tasks to queue
            p4task = Task(poetpcf.rundir, poetpcf.eventname, ppcf.ncores, poet.p,
                         args=(4, center+'/'+photom),
                         prereqs=[p3task], kwargs={"poetpcf":poetpcf})
            p4tasks.append(p4task)
            p5task = Task(poetpcf.rundir, poetpcf.eventname, 1, poet.p, args=(5, center+'/'+photom),
                          prereqs=[p4task], kwargs={"poetpcf":poetpcf})
            p5tasks.append(p5task)

    edPrint("I am completing p3-5 in a convenient order")
    curr_core  = 0
    curr_mem   = 200
    curr_tasks = 'p3'
    tasks      = p3tasks + p4tasks + p5tasks
    while True:

        for task in tasks:
            # if task is finished, remove its mem and core
            # contributions to counters
            if task.started and not task.thread.is_alive():
                curr_core -= task.cores
                curr_mem  -= task.mem
                tasks.remove(task)
            elif task.started:
                continue
            # otherwise try to start the task
            else:
                cores, mem = task.check(curr_core, edpcf.cores, curr_mem,
                                        edpcf.memory)
                curr_core += cores
                curr_mem  += mem
                if hasattr(task, "mem"):
                    if task.mem > 200 + edpcf.memory:
                        edPrint("Woops, it seems I don't have enough memory a"
                                "located to finish these centering and photo"
                                "metry runs. Increase 'memory' in poet.pcf?")
                        return 'quit'
                if task.cores > edpcf.cores:
                    edPrint("Woops, it seems I don't have enough cores a"
                            "located to finish these centering and photo"
                            "metry runs. Increase 'cores' in poet.pcf?")
                    return 'quit'
                        

        # change task set once you are ready
        if tasks == []:
            return (poetpcf, edpcf)

        # don't use up an entire cpu
        time.sleep(1)

def p6_init(poetpcf, edpcf):
    """Runs p6 on the test centering and photometry using every model
combination to determine the best model combination.

Parameters
----------
poetpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    POET section of poet.pcf. This specifies the run directory and the
    event pcf to use.

edpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    EDGAR section of poet.pcf. This specifies the test centering and
    photometry as well as the models to run

Returns
-------
string
    If read is unsuccessful, returns the string "quit".

tuple
    If p6_init completes successfully, returns a tuple of (poetpcf,
    edpcf, model) where the first two are the pcf objects passed in, and
    model is a space separated list of p6 models.

Notes
-----
This function will edit hd209bs24.pcf and replace the [p6] and
[params] sections. After p6 is run, it will return hd209bs24.pcf to
its original state, but will put the sections that were used in the
files params_init.pcf and p6_init.pcf as a record of what was done

If you want to pass in your own pcf object, you should generate them like so:
poetpcf = rd.read_pcf("poet.pcf", "POET", simple=True)
edpcf   = rd.read_pcf("poet.pcf", "EDGAR", simple=True,
                                d1lists=['zenmodels'])

The following setting in the params section of poet.pcf will be overwritten:
model, mcmc, chi2flag (mcmc set to False, chi2flag set to 0)

    """
    returnPrint("p6_init")
    edPrint("I am running p6 for the first time to Identify the optimal set o"
            "f models.")

    if not edpcf.runp6:
        edPrint("`runp6` is false in poet.pcf, skipping p6...\nIf you would like to run ZEN though EDGAR, run \"poet.py ed zen\" from the command line or \"poet.p('ed', 'zen')\" from a python session.")
        return 'quit'

    # find the number of threads for p6
    threads = edpcf.cores
    if threads < 1:
        edPrint("I am unable to run when you"
                " only allotted me", edpcf.cores, "cores for this run.\n")
        return "quit"

    numit = 0
    nchains = 0
    p6mem = guessp6size(poetpcf.rundir, nchains,
                        poetpcf.eventname, edpcf.p6model,
                        edpcf.test_centering, edpcf.test_photometry,
                        numit)

    # adjust cpu count based on memory
    if 200 + p6mem*threads > edpcf.memory:
        if 200 + p6mem > edpcf.memory:
            edPrint("Cannot run p6 with only", edpcf.memory, "MB of RAM allocated "
                    "when running p6 requires", 200 + p6mem, "MB of RAM.")
            return "quit"

        threads = int((edpcf.memory - 200)/p6mem)
        edPrint("I am reducing number of cores used to compensate for a lack "
                "of allotted RAM.")

    config = poetpcf.eventname+'.pcf'

    # use custom config overrides
    params_override = {
        "mcmc"     : False,
        "model"    : edpcf.p6model,
        "chi2flag" : 0
    }
    p6_override = {
        "centering"  : [edpcf.test_centering],
        "photometry" : [edpcf.test_photometry],
        "modeldir"   : 'dry_run',
        "threads"    : threads
    }

    # run p6
    lines, info = poet.p(6, nodate=False, poetpcf=poetpcf,
                         control=p6_override,
                         params_override=params_override)

    edPrint("I am finding the best model based on BIC values.")

    # Find the best model
    bestline = None
    bestoff  = None
    bestval  = None
    for line in lines.keys():
        for aper in lines[line]:
            if bestval is None or aper[5] < bestval:
                bestline = line
                bestoff  = float(aper[3])
                bestval  = aper[5]

    bestmodel = info[bestline][bestoff]['model']

    # find model that was best
    for mod in edpcf.p6model:
        modeljazz = mc.setupmodel(mod)
        if modeljazz[-1] == bestmodel:
            model = ' '.join(mod)
            break

    edPrint("I have found '"+model+"' to be the best model.")

    return poetpcf, edpcf, model,

def p6_full(poetpcf, edpcf, model):
    """Runs p6 on every centering and photometry using the model
combination passed in to determine the best centering and photometry
combination.

Parameters
----------
poetpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    POET section of poet.pcf. This specifies the run directory and the
    event pcf to use.

edpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    EDGAR section of poet.pcf. This specifies the amount of computer
    resources to use

model : string
    Space separated list of models to combine. This will be inserted
    into the p6 section of eventname. See Notes below

Returns
-------
string
    If read is unsuccessful, returns the string "quit".

tuple
    If p6_full completes successfully, returns a tuple of (poetpcf,
    edpcf, centering, photometry, model) where the first two are the
    pcf objects passed in, centering is the string giving the
    centering directory, photometry a string giving the photometry
    directory, and model a space separated list of p6 models. This is
    EDGAR's best guess of the best centering, photometry, and model

Notes
-----
This function will edit hd209bs24.pcf and replace the [p6] and
[params] sections. After p6 is run, it will return hd209bs24.pcf to
its original state, but will put the sections that were used in the
files params_full.pcf and p6_full.pcf as a record of what was done

If you want to pass in your own pcf object, you should generate them like so:
poetpcf = rd.read_pcf("poet.pcf", "POET", simple=True)
edpcf   = rd.read_pcf("poet.pcf", "EDGAR", simple=True,
                                d1lists=['zenmodels'])

The following setting in the params section of poet.pcf will be overwritten:
model, mcmc, chi2flag (mcmc set to False, chi2flag set to 0)

    """

    returnPrint("p6_full", model)

    edPrint("I am running p6 on all centering/photometry with the decided"
            " model:", model, ". This will Identify the optimal centering"
            " and photometry methods.")

    # find the number of threads for p6
    threads = edpcf.cores
    if threads < 1:
        edPrint("I am unable to run when you"
                "only allotted me", edpcf.cores, "cores for this run.\n")
        return 'quit'

    # check memory usage
    numit = 0
    nchains = 0
    p6mem = guessp6size(poetpcf.rundir, nchains,
                        poetpcf.eventname, [model], edpcf.test_centering,
                        edpcf.test_photometry,
                        numit)

    # adjust cpu count based on memory
    if 200 + p6mem*threads > edpcf.memory:
        if 200 + p6mem > edpcf.memory:
            edPrint("Cannot run p6 with only", edpcf.memory, "MB of RAM allocated "
                    "when running p6 requires", 200 + p6mem, "MB of RAM.")
            return "quit"

        threads = int((edpcf.memory - 200)/p6mem)
        edPrint("I am reducing number of cores used to compensate for a lack "
                "of allotted RAM.")

    # use custom config overrides
    p6_override = {
        'centering': ['all'],
        'photometry': ['all'],
        'modeldir': None,
        'threads': threads}

    params_override = {
        'mcmc': False,
        'model': [model.split()],
        'chi2flag': 1}

    # run p6
    lines, info = poet.p(6, nodate=False, poetpcf=poetpcf,
                         control=p6_override,
                         params_override=params_override)

    edPrint("I am finding the best centering/photometry based on binned-sigma"
            " chi-squared")

    # find best centering/photometry
    bestline = None
    bestoff  = None
    bestval  = None
    for line in lines.keys():
        for aper in lines[line]:
            if bestline is None or aper[8] < bestval:
                bestline = line
                bestoff  = float(aper[3])
                bestval  = aper[8]

    centering = info[bestline][bestoff]['centdir']
    photometry = info[bestline][bestoff]['photdir']

    edPrint("I have found a centering method of", centering, "with a"
            "photometry of", photometry, "using '", model, "' as the model to"
            "yield the most accurate results.")

    return poetpcf, edpcf, centering, photometry, model

def p6_check(poetpcf, edpcf, centering, photometry, model):
    """Runs p6 on the centering and photometry passed in but on every
model in edpcf and compares to ensure that a different model is not
chosen. This is done in the rare situation that a different model is
chosen, in which case a human should continue the analysis as this
should not happen in theory

Parameters
----------
poetpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    POET section of poet.pcf. This specifies the run directory and the
    event pcf to use.

edpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    EDGAR section of poet.pcf. This specifies the the models to run

centering : string
    The name of the centering directory.

photometry : string
    The name of the photometry directory.

model : string
    Space separated model. This model is not the only one that is run,
    but an error will occurred if this model does not yield the lowest
    BIC.

Returns
-------
string
    If read is unsuccessful, returns the string "quit".

tuple
    If p6_check completes successfully, returns a tuple of (poetpcf,
    edpcf, centering, photometry, model) where the first two are the
    pcf objects passed in, centering is the string giving the
    centering directory, photometry a string giving the photometry
    directory, and model a space separated list of p6 models. This is
    EDGAR's best guess of the best centering, photometry, and model.

Notes
-----
This function will edit hd209bs24.pcf and replace the [p6] and
[params] sections. After p6 is run, it will return hd209bs24.pcf to
its original state, but will put the sections that were used in the
files params_check.pcf and p6_check.pcf as a record of what was done

If you want to pass in your own pcf object, you should generate them like so:
poetpcf = rd.read_pcf("poet.pcf", "POET", simple=True)
edpcf   = rd.read_pcf("poet.pcf", "EDGAR", simple=True,
                                d1lists=['zenmodels'])

The following setting in the params section of poet.pcf will be overwritten:
model, mcmc, chi2flag (mcmc set to False, chi2flag set to 0)

    """

    edPrint("I am rerunning all models on the centering and photometry select"
            "ed in the full phase to make sure that the optimal model did not"
            " change from using a different centering and photometry. (You wi"
            "ll get a lengthy error message if this happens.)")

    returnPrint("p6_check", centering, photometry, model)
    
    # find the number of threads for p6
    threads = edpcf.cores
    if threads < 1:
        edPrint("I am unable to run when you"
                "only allotted me", edpcf.cores, "cores for this run.\n")
        return 'quit'

    # check memory usage
    numit = 0
    nchains = 0
    p6mem = guessp6size(poetpcf.rundir, nchains,
                        poetpcf.eventname, edpcf.p6model,
                        centering, photometry,
                        numit)

    # adjust cpu count based on memory
    if 200 + p6mem*threads > edpcf.memory:
        if 200 + p6mem > edpcf.memory:
            edPrint("Cannot run p6 with only", edpcf.memory, "MB of RAM allocated "
                    "when running p6 requires", 200 + p6mem, "MB of RAM.")
            return "quit"

        threads = int((edpcf.memory - 200)/p6mem)
        edPrint("I am reducing number of cores used to compensate for a lack "
                "of allotted RAM.")

    config = poetpcf.eventname + '.pcf'

    # use custom config overrides
    p6_override = {
        'header': 'p6',
        'centering': [centering],
        'photometry': [photometry],
        'modeldir': 'check',
        'threads': threads}

    params_override = {
        'mcmc': False,
        'model': edpcf.p6model,
        'chi2flag': 0}

    # run p6
    lines, info = poet.p(6, nodate=False, poetpcf=poetpcf,
                         control=p6_override,
                         params_override=params_override)

    edPrint("I am finding the best model based on BIC values."
            "This should be the same as calculated before.")

    # Find the best model
    bestline = None
    bestoff  = None
    bestval  = None
    for line in lines.keys():
        for aper in lines[line]:
            if bestval is None or aper[5] < bestval:
                bestline = line
                bestoff  = float(aper[3])
                bestval  = aper[5]

    bestmodel = info[bestline][bestoff]['model']

    # find model that was best
    for mod in edpcf.p6model:
        modeljazz = mc.setupmodel(mod)
        if modeljazz[-1] == bestmodel:
            newmodel = ' '.join(mod)
            break

    if newmodel != model:
        edPrint("After running p6 on the selected centering and photometry, "
                "a different best model was obtained. This is irregular and "
                "beyond the scope of EDGAR's functionality. You will have to r"
                "erun and finish the analysis by hand. The model obtained in"
                " the init phase was:", model, ". This was done with centeri"
                "ng:", edpcf.test_centering, "; and photometry:",
                edpcf.test_photometry, ". The model obtained in the check p"
                "hase (this phase) is:", newmodel, ". This was run on center"
                "ing:", centering, "; and photometry:", photometry, ". Good "
                "Luck!\n\nWhen you get a good result, you can run p7-10 with"
                " by passing the arguments 'ed', 'finish', '<p6ouputdir>' to "
                "poet in that order where 'p6outputdir' a full path. Also, Z"
                "EN can be run through EDGAR with arguments 'ed', 'zen'")
        return 'quit'

    edPrint("models agree")

    return poetpcf, edpcf, centering, photometry, model

def p6_final(poetpcf, edpcf, centering, photometry, model):
    """Runs p6 one last time on the chosen centering, photometry, and
model with a higher mcmc iteration.

Parameters
----------
poetpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    POET section of poet.pcf. This specifies the run directory and the
    event pcf to use.

edpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    EDGAR section of poet.pcf. This specifies the the models to run

centering : string
    The name of the centering directory.

photometry : string
    The name of the photometry directory.

model : string
    Space separated model to be run.

Returns
-------
string
    If read is unsuccessful, returns the string "quit".

tuple
    If p6_final completes successfully, returns a tuple of (poetpcf,
    edpcf) which are the pcf objects passed in.

Notes
-----
This function will edit hd209bs24.pcf and replace the [p6] and
[params] sections. After p6 is run, it will return hd209bs24.pcf to
its original state, but will put the sections that were used in the
files params_final.pcf and p6_final.pcf as a record of what was done

If you want to pass in your own pcf object, you should generate them like so:
poetpcf = rd.read_pcf("poet.pcf", "POET", simple=True)
edpcf   = rd.read_pcf("poet.pcf", "EDGAR", simple=True,
                                d1lists=['zenmodels'])

The following setting in the params section of poet.pcf will be overwritten:
model, mcmc, chi2flag (mcmc set to True, chi2flag set to 1)

    """

    edPrint("I am rerunning with more iterations on the centering/photometry"
            "with the lowest bsigchi using the model selected from the init p"
            "hase (and confirmed in the check phase).")

    returnPrint("p6_final", centering, photometry, model)

    # grab params section
    config = poetpcf.eventname + '.pcf'
    params = rd.read_pcf(poetpcf.rundir+'/'+config, 'params',
                         simple=True, d1lists=('numit',))

    # find the number of threads for p6
    threads = edpcf.cores // params.nchains
    if threads < 1:
        edPrint("I am unable to run", params.nchains, "chains when you"
                "only allotted me", edpcf.cores, "cores for this run.\n")
        return 'quit'

    # check memory usage
    p6mem = guessp6size(poetpcf.rundir, params.nchains,
                        poetpcf.eventname, [model],
                        centering, photometry,
                        params.numit[1])

    # adjust cpu count based on memory
    if 200 + p6mem*threads > edpcf.memory:
        if 200 + p6mem > edpcf.memory:
            edPrint("Cannot run p6 with only", edpcf.memory, "MB of RAM allocated "
                    "when running p6 requires", 200 + p6mem, "MB of RAM.")
            return "quit"

        threads = int((edpcf.memory - 200)/p6mem)
        edPrint("I am reducing number of cores used to compensate for a lack "
                "of allotted RAM.")


    # moving params file to rundir with chosen model
    p6_override = {
        'header': 'p6',
        'centering': [centering],
        'photometry': [photometry],
        'modeldir': 'final',
        'threads': threads}

    params_override = {
        'model': [model.split()],
        'mcmc': True,
        'chi2flag': 1}

    moddir = poet.p(6, nodate=False, retmoddir=True, poetpcf=poetpcf,
                    control=p6_override, params_override=params_override)[0]

    edPrint("Everything seems to have worked out fine. Check out",
            '/'.join([poetpcf.rundir, centering, photometry, moddir]),
            "to see the results.")

    p6outdir = '/'.join([centering, photometry, moddir])

    return poetpcf, edpcf, p6outdir,

def finish(poetpcf, edpcf, p6outdir):
    """Runs p7 through 10 on the specified output directory

Parameters
----------
poetpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    POET section of poet.pcf. This specifies the run directory and the
    event pcf to use.

edpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    EDGAR section of poet.pcf. This specifies the the models to run

p6outdir : string
    relative path to p6 output directory to be run on.

Returns
-------
string
    If read is unsuccessful, returns the string "quit".

tuple
    If finish completes successfully, returns a tuple of (poetpcf,
    edpcf) which are the pcf objects passed in.

Notes
----
If you want to pass in your own pcf object, you should generate them like so:
poetpcf = rd.read_pcf("poet.pcf", "POET", simple=True)
edpcf   = rd.read_pcf("poet.pcf", "EDGAR", simple=True,
                                d1lists=['zenmodels'])

    """

    returnPrint("finish", p6outdir)

    if not edpcf.runp7_10:
        edPrint("`runp7_10` is false in poet.pcf, skipping p7 through p10...")
        return (poetpcf, edpcf)

    p710size = guessp710size(p6outdir, poetpcf)

    if p710size > edpcf.memory:
        edPrint("You did not give me enough memory in poet.pcf to finish runn"
                "ing p7 through p10.")
        return 'quit'

    # run the rest of POET
    edPrint("p6 ran to completion, running analysis (p7)...")
    poet.p(7, p6outdir, poetpcf=poetpcf)
    edPrint("p7 ran to completion, generating tables (p8)...")
    poet.p(8, p6outdir, poetpcf=poetpcf)
    edPrint("p8 ran to completion, generating figures (p9)...")
    poet.p(9, p6outdir, poetpcf=poetpcf)
    edPrint("p9 ran to completion, generating IRSA tables (p10)...")
    poet.p(10, p6outdir, poetpcf=poetpcf)
    edPrint("p10 ran to completion, check", p6outdir, "in your specified run"
            " directory for the results.")


    return (poetpcf, edpcf)

def zen(poetpcf, edpcf):
    """Runs ZEN.

For each zenmodel specified in edpcf, this function runs ZEN on every
centering and photometry (call these full runs). Then, for each model,
ZEN is run again, this time only on the best centering and photometry
as determined by the corresponding full run with binning set to 1 (call
these BIC runs). The BIC run with the lowest BIC is selected and the
full run with the same model is the "best". A report string indicating
the "best" result directory, as well as all directories is printed.

Parameters
----------
poetpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    POET section of poet.pcf. This specifies the run directory and the
    event pcf to use.

edpcf : reader3.pcf, string
    String giving location of poet.pcf, or pcf object read from the
    EDGAR section of poet.pcf. This specifies the the models to run

Returns
-------
string
    If read is unsuccessful, returns the string "quit". Otherwise, returns
    "Done".

Notes
-----
The configuration for each model, [model], is found by searching the appropriate run directory for a file named zen_[model].cfg. If that file is not found, a new config is generated from a template.

The following settings in zen_*.cfg will be overwritten by this function: eventname, nchains, cent, phot, maxbinsize

If you want to pass in your own pcf object, you should generate them like so:
poetpcf = rd.read_pcf("poet.pcf", "POET", simple=True)
edpcf   = rd.read_pcf("poet.pcf", "EDGAR", simple=True,
                                d1lists=['zenmodels'])

    """

    edPrint("Running ZEN on specified models")
    returnPrint("zen")

    if not edpcf.runzen:
        edPrint("`runzen` is false in poet.pcf, returning...")
        return 'quit'

    eventpcf = poetpcf.rundir + '/' + poetpcf.eventname + '.pcf'


    # find example centering and photometry dir
    cpcf = rd.read_pcf(eventpcf, "centering")[0]
    ppcf = rd.read_pcf(eventpcf, "photometry")[0]
    centdir = cpcf.method if cpcf.pcfname is None else cpcf.method + '_' + cpcf.pcfname
    photdir = photname(ppcf)

    # check memory usage
    filename = poetpcf.rundir+'/'+poetpcf.eventname+'-zen.cfg'
    zenmem = guesszensize(poetpcf, edpcf, centdir, photdir, filename)
    if zenmem > edpcf.memory:
        edPrint("EDGAR does not have enough memory allotted to run the with t"
                "he specified number of cores. Either increase \"memory\" or "
                "decrease \"zenchains\" in \"EDGAR\" portion of poet.pcf and rer"
                "un.")
        returnPrint("zen")
        return 'quit'

    # Run zen with each model to determine the best cent/ap combo for each model
    bestdir    = {}
    compareBIC = {}
    bicdir     = {}
    for i, model_set in enumerate(edpcf.zenmodels):

        model_set = ' '.join(model_set)
        # move config files around
        config = cfg.ConfigParser()
        config.read([filename])
        config['EVENT']['models'] = model_set
        date = datetime.datetime.today().strftime("%Y-%m-%d_%H_%M")
        config['EVENT']['outdir'] = date+'_zen_model%d'%i+"_full"
        config['EVENT']['cent']   = 'all'
        config['EVENT']['phot']   = 'all'

        # run ZEN
        outdir, centdir, photdir, chiout = poet.p("zen", config, poetpcf=poetpcf, cfilename='zen.cfg')
        bestdir[model_set] = outdir

        # Rerun each model on their respective best combo with bintry=1 to
        # compare across models.
        config = cfg.ConfigParser()
        config.read([filename])
        config['EVENT']['models']     = model_set
        config['EVENT']['cent']       = centdir
        config['EVENT']['phot']       = photdir
        config['EVENT']['bintry']     = '1'
        date = datetime.datetime.today().strftime("%Y-%m-%d_%H_%M")
        config['EVENT']['outdir'] = date+'_zen_model%d'%i+"_BIC"
        config['MCMC']['chisqscale'] = "False"
        

        # run zen bintry=1
        outdir, centdir, photdir, chiout = poet.p("zen", config, poetpcf=poetpcf, cfilename='zen.cfg')
        compareBIC[model_set] = chiout[3]
        bicdir[model_set]     = outdir

    edPrint("Ran the following ZEN models:", edpcf.zenmodels)

    # find best BIC
    bestmodel = None
    bestBIC   = None
    for model_set in compareBIC:
        if bestmodel is None or compareBIC[model_set] < bestBIC:
            bestmodel = model_set
            bestBIC   = compareBIC[model_set]

    edPrint("After rerunning models on their best cent/ap combinations with bi"
            "ntry=1, BICs were compared indicating that", bestmodel, "was the "
            "best. The output can be found at:\n", bestdir[bestmodel], "\nHowev"
            "er, you should still check to make sure that the ZEN runs all con"
            "verged. Here are the directories of the full runs' can be found a"
            "t:\n" + "\n".join(model_set+"\n"+bestdir[model_set] for model_set in bestdir),
            "\nand here are the directories of the mini runs that were conducted"
            " to compare BIC values:\n" + "\n".join(model_set+"\n"+bicdir[model_set]
                                                    for model_set in bicdir))
    edPrint("All Done!")

    return 'Done'


class Task:
    """Object used to queue up p3, p4, and p5 runs (tasks).

Attributes
----------
eventname : str
    eventname
rundir : str
    relative path to run directory
started : bool
    True if task has begun
prereqs : itterable
    itterable of tasks that must be completed prior to starting this task
cores : int
    number of cores that this task will take
args : itterable
    arguments passed to the run
thread : multiprocessing.Process
    thread to be run.

    """

    def __init__(self, rundir, eventname, cores, func, args=(), kwargs={}, prereqs=[]):
        """Parameters
----------
rundir : str
    relative path to run directory
eventname : str
    eventname
cores : int
    number of cores that this tasks will use
func : function
    function to be run. This will be passed to a multiprocessing.Process constructor
args : itterable
    args to be passed to multiprocessing.Process constructor
kwargs : map
    kwargs passed to multiprocessing.Process constructor
prereqs : itterable
    itterable of tasks which must be completed prior to running this task.
        """

        assert cores > 0, "Edgar: Task cannot have 0 or fewer cores."

        self.eventname = eventname
        self.rundir    = rundir
        self.started   = False
        self.prereqs   = prereqs
        self.cores     = cores
        self.args      = args
        self.thread    = multiprocessing.Process(target=func, args=args,
                                                 kwargs=kwargs)

    def check(self, curr_cores, max_cores, curr_mem, max_mem):
        """Checks to see if self.thread can be started (prereqs finished,
enough cores/memory available). If self.thread can be started, then it
is started.

Parameters
----------
curr_cores : int
    current number of cores in use
max_cores : int
    maximum number of cores to be used
curr_mem : float
    current amount of memory in use
max_mem : float
    maximum amount of memory to be used

Returns
-------
int
    number of additional cores used after this function call.
float
    additional memory (in MB) used after this function call.

Notes
-----
The task's mem attribute is not calculated until this function is
called after all prereqs have completed as the amount of memory used
by this task once started is dependent on the size of files output by
its prereqs.

        """

        # do nothing if this would surpass max core count
        if self.cores + curr_cores > max_cores:
            return 0, 0
        # do nothing if a prereq has not started or finished
        for prereq in self.prereqs:
            if prereq.thread.is_alive() or not prereq.started:
                return 0, 0
        # check how much memory will be used
        self.mem = self.get_mem()
        if self.mem + curr_mem > max_mem:
            return 0, 0
        # do nothing if this thread has already started
        if self.started == True:
            edPrint("Tried to start process twice. Not allowing...")
            return 0, 0

        self.started = True
        self.thread.start()
        return self.cores, self.mem

    def get_mem(self):
        """Returns the amount of memory that will be used by this task. Will
raise an error if the files necessary to determine this task's memory
usage have not been populated. (i.e. if this is a p4 task and the
corresponding p3 run has not finished, this function will stat p3's
output, which doesn't exist, and crash.

        """
        if self.args[0] == 3:
            datsize = os.path.getsize('/'.join([self.rundir,
                                                self.eventname+"_bpm.dat"]))
            h5size  = os.path.getsize('/'.join([self.rundir,
                                                self.eventname+"_bpm.h5"]))
        elif self.args[0] == 4:
            centerdir = self.args[1].split('/')[0]
            datsize = os.stat('/'.join([self.rundir, centerdir,
                                        self.eventname+"_ctr.dat"])).st_size
            h5size  = os.stat('/'.join([self.rundir, centerdir,
                                        self.eventname+"_ctr.h5"])).st_size \
                            * 1.5
        elif self.args[0] == 5:
            datsize = os.stat('/'.join([self.rundir, self.args[1],
                                        self.eventname+"_pht.dat"])).st_size
            h5size  = 0
        else:
            edPrint("If this statement printed, burn everything.")
        return 200 + (datsize + h5size)/1024**2


def edPrint(*args, **kwargs):
    """Prints output to standard out, prepended by "\nEdgar:", as well as
to the log file (without "\nEdgar:"). Can be used identically to the
standard print, except you can't specify the file keyword argument.

    """
    print("\nEdgar:", *args, **kwargs)
    print(*args, file=open(logf, 'a'), **kwargs)

def returnPrint(*args):
    """Prints a return message. arguments should be those that will be
passed to edgar() in order to resume from a given point.

    """
    edPrint("To resume from this point, run \"poet.py ed '"+"' '".join(args)+
            "'\" from a terminal or \"poet.p('ed', '"+"', '".join(args)+"')"
            "\" from a python session.")

def photname(pcf):
    """From an input photometry pcf, generates the corresponding photometry
directory name.

    """

    phottype = pcf.phottype
    if phottype == 'optimal':
        return 'optimal' if pcf.pcfname is None else 'optimal_' + pcf.pcfname
    elif phottype == 'psffit':
        return 'psffit' if pcf.pcfname is None else 'psffit_' + pcf.pcfname
    elif phottype == 'aper' or phottype == 'var' or phottype == 'ell':
        ap   =  '%03i' % (pcf.photap * 1e2)
        off  = '%+04i' % (pcf.offset *100)
        # sin  = '%02i' % pcf.skyin
        # sout = '%02i' % pcf.skyout
        if phottype == 'ell': phottype = 'el'
        name = phottype[0:2] + ap + off
        return name if pcf.pcfname is None else name + '_' + pcf.pcfname


def guessp1size(poetpcf):
    """From and event pcf, guesses how much memory running p1 will take."""

    # the size of a p1 object is roughly 4.047 (rounding to 4.1) times
    # the size of one of its four arrays (data, uncertainty, bdmask,
    # and brmask). To guess the size of an event generated in p1, we
    # need to find the size of one of these arrays which at the time
    # of this writing, follows the formula:
    #
    #     arrsize = 80 + 16*dim + 8*#ofvals [bytes] (assuming 64 bit values)
    #
    # Changes to this formula in future versions should be
    # insignificant (unless 128 or higher bit values are
    # used). Additionally, running python and importing everything
    # uses about 116 MB (res) of RAM. I round this up to 200 MB for
    # safesies and because more memory will be used to make other
    # things. Hence the final equation I will use is:
    #
    #     size = 4.1 * (arrsize / 1024**2) + 200


    # get values from pcf
    pcf,     = rd.read_pcf(poetpcf.rundir+'/'+poetpcf.eventname+'.pcf', "event", expand=False)
    npos     = pcf.npos[0]
    nnod     = pcf.nnod[0]
    aornames = np.array(pcf.aorname)
    aortypes = np.array(pcf.aortype)

    # find dimensions of array which are ny, nx, npos, maxnimpos
    instru     = inst.Instrument(pcf.photchan[0])
    naor       = np.size(aornames[np.where(aortypes == 0)])
    nexpid     = np.empty(naor, np.long)
    for i, aorname in enumerate(aornames[:naor]):
        bcddir = os.path.join(*[*pcf.topdir[0].split("/"), *pcf.datadir[0].split("/"),
                           pcf.sscver[0], 'r'+str(aorname),
                           *instru.bcddir.split("/")])
        files = sorted([f for f in os.listdir(bcddir)
                        if f.endswith(instru.bcdsuf)])
        first, last = files[0], files[-1]
        fexpid = first.split('_')[-4]
        expid, dcenum = last.split('_')[-4:-2]
        nexpid[i] = int(expid) - int(fexpid) + 1
    ndcenum = int(dcenum) + 1

    header = fits.getheader(bcddir+'/'+first)

    # number of dimension is hard coded to be 4 in
    # pdataread.poet_dataread() and throughout the code
    dim = 4

    if header['NAXIS'] >=3:
        nz = header['NAXIS1']
        ny = header['NAXIS2']
        nx = header['NAXIS3']
    else:
        nz = 1
        ny = header['NAXIS1']
        nx = header['NAXIS2']

    # equations taken from p1event
    if instru.name == 'mips':
        maxnimpos = int((np.sum(nexpid)/nnod) * (((ndcenum - 1)/7) + 1))
    else:
        maxnimpos = int(np.sum(nexpid) * ndcenum * nz / nnod)

    # in bytes
    arrsize = 80 + 16 * dim + 8 * (nx * ny * npos * maxnimpos)

    # in MB
    return 4.1 * (arrsize / 1024**2) + 200

def guessp6size(rundir, nchains, eventname, models, center, phot, numit):
    """Guesses the amount of memory running p6 will take on a single
centering and photometry.

Parameters
----------
rundir : str
    relative path to run directory.
nchains : int
    number of parallel chains to be used in p6.
eventname : str
    eventname
models : itterable
    itterable of model sets.
center
    centering directory name.
phot
    photometry directory name.
numit
    the number of MCMC iterations that will be performed.

Returns
-------
float
    amount of memory in MB that will (in theory) be used.

    """

    # the size of a p6 object is roughly the size of p5's output. 
    # The size of the 'allparams' array follows the followings formula:
    #
    #     arrsize = 80 + 16*dim + 8*#ofvals [bytes] (assuming 64 bit values)
    #
    # which may exist thrice (it gets copied and deleted in a couple
    # places). Estimating 200 MB for imports and various variables,
    # the total memory used should be:
    #
    #     mem = 200 + 3*(arrsize/1024**2) + p5size [MB]

    # assume the various p5 event objects are of similar size
    p5ename = '/'.join([rundir, center, phot, eventname+"_p5c.dat"])
    # in MB
    p5size = os.stat(p5ename).st_size/1024**2

    models_array = paramedit.init_models()
    maxparams = 0
    for model_set in models:
        nparams = 0
        for model_part in model_set:
            for modelname, modelheader, array in models_array:
                if modelname == model_part:
                    nparams += array.shape[1]
        maxparams = max(maxparams, nparams)

    # in Bytes
    arraysize = 80 + 16*3 + 8*nchains*numit*maxparams

    # in MB
    return 200 + 3*(arraysize/1024**2) + p5size

def guessp710size(p6dir, poetpcf):
    """guesses the amount of memory that will be used to run p7 through p10

Parameters
----------
p6dir : str
    relative path including rundir to the p6 output directory
poetpcf : reader3.Pcf
    pcf object containing the POET section of poet.pcf

Returns
-------
float
    amount of memory that will (in theory) be used to run p7-p10

    """

    p6dir = poetpcf.rundir+'/'+p6dir
    params = rd.read_pcf('/'.join([p6dir, poetpcf.eventname+'.pcf']), "params", simple=True, d1lists=('numit',), d2lists=('model',), nparrs=('numit',))
    modeljazz = mc.setupmodel(params.model[0])
    allparams = os.stat(p6dir+'/d-'+poetpcf.eventname+'-allparams-'+modeljazz[-1]+'.npy').st_size
    event     = os.stat(p6dir+'/d-'+poetpcf.eventname+'-6model.dat').st_size
    return 200 + (allparams + event)/1024**2

def guesszensize(poetpcf, edpcf, centdir, photdir, fname):
    """Guesses the amount of memory that will be used to run ZEN.

Parameters
----------
poetpcf : reader3.Pcf
    pcf object containing the POET section of poet.pcf
edpcf : reader3.Pcf
    pcf object containing the EDGAR section of poet.pcf
centdir : str
    name of centering directory
photdir : str
    name of photometry directory

Returns
-------
float
    approximate amount of memory that will be used running ZEN

    """

    p1dat = os.stat(
        '/'.join((poetpcf.rundir, poetpcf.eventname+"_ini.dat"))
    ).st_size
    p1h5  = os.stat(
        '/'.join((poetpcf.rundir, poetpcf.eventname+"_ini.h5"))
    ).st_size
    p5dat = os.stat(
        '/'.join((poetpcf.rundir, centdir, photdir, poetpcf.eventname+"_p5c.dat"))
    ).st_size

    config = cfg.ConfigParser()
    config.read([fname])
    zenchains = int(config["MCMC"]["nchains"])

    return 200 + (zenchains * p5dat + p1dat + p1h5/2)  / 1024**2
if __name__ == "__main__":
    args = sys.argv[1:]
    kwargs = poet.getkwargs(args)
    edgar(*args, **kwargs)
