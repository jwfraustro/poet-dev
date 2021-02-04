#! /usr/bin/env python3
"""
TO EXECUTE, USE THE FOLLOWING SYNTAX:

[] = optional ordered arguments
{} = optional key-word arguments
[{}] = can be specified in order or as key-word arguments

./poet.py p1
    Description:
            Reads in initial data and saves to a file

./poet.py p2
    Description:
            Performs a bad pixel mask to                                     #

./poet.py p3 [directory]
    Description:
            Preforms centering                                               #
    directory:
            Default: current working directory
            If specified, run p3 on given directory using the 'center.pcf'
              in that directory.

./poet.py p4 directory
    directory:
            This can be given as <center> or <center>/<photom>.
            If <center> is given, p4 is run on all photometry specified in
              <center>/photom.pcf.
            If <center>/<photom> is given, p4 is run only on that photometry
              using <center>/<photom>/photom.pcf

./poet.py p5 directory [{ephtime period}]
    Description:
    directory:
            Given as <center>/<photom>.
            p5 is run on this directory
    ephtime:
            Default: None
                                                                             #
    period:
            Default: None
                                                                             #
    
./poet.py zen [cfile {outdirext}]
    Description:
    cfile:
    outdirext:

./poet.py p6 [directory {modeldir topdir clip idl mode numit nodate justplot}]
    Description:
    directory:
            Default: current working directory
            Given as <center>/<photom> or .
            If omitted (or .), runs p6 on all <center>/<photom> specified in
              p6.pcf in the run directory. Basic results are written to a
              cache in the run directory and are plotted in plots (parallel
              to run.)
            If given, runs p6 on the given directory. Plots will NOT be made
              until you run './poet.py p6 justplot=True' (which will only
              plot output specified in p6.pcf.)
    modeldir:
            Default: None
            This specifies the name of the output directory for p6.
            If not specified, the output directory will be named by the date
              regardless of the value of nodate.
            This value overrides the modeldir value given in p6.pcf
    clip:
            Default: None
            Clip data set to model and plot only a portion of the entire
              light curve.  Good for modeling a transit or eclipse in an 
              around-the-orbit data set.  Syntax is <start>:<end>
    idl:
                                                                             #
    mode:
            Default: full
            If full, runs in burn followed by a run in final
            If burn,
            If final,
            If continue,
    numit:
            The number of MCMC iterations. If not specified, the number of
              iterations specified in the params pcf is used.
                                                                             #
    nodate:
            Default: True
            If False, The date is put at the beginning of the output
    justplot:
            Default: False
            Specifying modeldir will override modeldir specified in p6.pcf.
            
./poet.py p7 directory [{idl islak}]
    Description:
    directory:
    idl:
    islak:

./poet.py p8 directory [{idl eclphase}]
    Description:
    directory:
    idl:
    eclphase:

./poet.py p9 directory [{idl titles}]
    Description:
    directory:
            
    idl:
            Default: False
                                                                             #
    titles:
            Default: True
            If False, titles are not included in plots

"""


import sys, os, re

# Some SciPy functions use the OpenBLAS library to run subprocesses
# on linear algebra solvers and the like. Normally this is a good thing,
# but we only want POET to use the specified number of threads, so we
# only allow one thread per POET instance. Paradoxically, this speeds
# things up considerably, as the subprocesses eat up processors that could
# be doing other things more efficiently. Some users may not find this
# to be the case depending on their setup, and may wish to comment out
# the following line.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib as mpl

def in_ipython():
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        return True

if not in_ipython():
    mpl.use("Agg")

rootdir = os.getcwd()

libdir = os.path.dirname(os.path.abspath(__file__)) + "/lib"

# add lib directory to path
sys.path.insert(0, libdir)
# add least asymmetry dir to path
sys.path.insert(1, libdir + '/la')

import p1event  as p1
import p2badpix as p2
import p3center as p3
import p4photom as p4
import p5checks as p5
import p10irsa  as p10
import pdenoise as pd
import edgar
import reader3  as rd
import run
import zen


def poet(func, *args, **kwargs):

    # read config
    if "poetpcf" in kwargs.keys():
        poetpcf = kwargs.pop("poetpcf")
    else:
        poetpcf = rd.read_pcf("poet.pcf", 'POET', simple=True)
    rundir    = os.path.abspath(poetpcf.rundir)
    eventname = poetpcf.eventname

    # extract key-word arguments from args
    args = list(args)
    kwargs.update(getkwargs(args))
#    args = [formt(arg) for arg in args]

    # If the user is using EDGAR, switch to that immediately.
    if func == "ed" or func == "edgar" or func == "poe":
        return edgar.edgar(*args, **kwargs)

    # reformat func if given as int or with p omitted
    if func in range(1, 12):
        func = str(func)
    if func in ['1', '2', 'd', '3', '4', '5', '6', '7', '8', '9',
                '10', '11', '12']:
        func = 'p' + func

    # find proper directory to run in
    if len(args) > 0 and func != "zen":
        directory = '/'.join([rundir, args.pop(0)])
    else:
        directory = rundir
    directory = os.path.abspath(directory)

    # run POET
    if func == "p1":
        return p1.Event(eventname+'.pcf', directory, **kwargs)

    elif func == "p2":
        return p2.badpix(eventname+'_ini', directory, **kwargs)

    elif func == "pd":
        pd.run_denoising(eventname+'_bpm', directory, **kwargs)

    elif func == "p3":
        if os.path.isfile(directory+'/'+eventname+'_den.dat'):
            p3.run_centering(directory+'/'+eventname+'_den', directory,
                             **kwargs)
        elif os.path.isfile(directory+'/'+eventname+'_bpm.dat'):
            p3.run_centering(directory+'/'+eventname+'_bpm', directory,
                             **kwargs)
        else:
            rootdir = os.path.dirname(directory)
            if os.path.isfile(rootdir+'/'+eventname+'_den.dat'):
                p3.run_centering(rootdir+'/'+eventname+'_den', directory,
                                 **kwargs)
            else:
                p3.run_centering(rootdir+'/'+eventname+'_bpm', directory,
                                 **kwargs)

    elif func == "p4":
        if os.path.isfile(directory+'/'+eventname+'_ctr.dat'):
            p4.run_photometry(directory+'/'+eventname+'_ctr', directory,
                              **kwargs)
        else:
            rootdir = os.path.dirname(directory)
            p4.run_photometry(rootdir+'/'+eventname+'_ctr', directory,
                              **kwargs)

    elif func == "p5":
        p5.checks1(directory+'/'+eventname+'_pht', directory, *args, **kwargs)

    elif func == "zen":
        if len(args) > 0:
            return zen.main(rundir, *args, **kwargs)
        else:
            return zen.main(rundir, None, **kwargs)

    elif func == "p6":
        if directory != rundir:
            return run.p6model(None, directory, *args,
                           rundir=rundir, **kwargs)

        else:
            if "control" not in kwargs.keys():
                kwargs["control"] = directory+'/'+eventname+'.pcf'

            return run.p6model(None, directory, rundir=rundir, **kwargs)

    elif func == "p7":
        run.p7anal(None, directory, *args, **kwargs)

    elif func == "p8":
        run.p8tables(None, directory, *args, **kwargs)

    elif func == "p9":
        run.p9figs(None, directory, *args, **kwargs)

    elif func == "p10":
        # The config is not in the p6output directory
        cfgdir = directory + '/../../../'
        p10.make_irsa(None, directory, config=cfgdir+'/p10.pcf',
                      *args, **kwargs)

    else:
        print("Unrecognized function:", func)

    return


def getkwargs(args):

    kwargs = {}
    remove = []

    for arg in args:
        if '=' in arg:
            remove.append(arg)
            info = arg.split('=')
            kwargs[info[0]] = formt(info[1])

    for arg in remove:
        args.remove(arg)

    return kwargs


def formt(val):
    """
    Turns string into float/int/True/False/None if need be.
    """
    # if val is a list, recursively format the list
    if isinstance(val, (list, tuple)):
        return [formt(elt) for elt in val]

    # convert val to bool or None.
    if val == 'True' or val == 'true':
        return True
    elif val == 'False' or val == 'true':
        return False
    elif val == 'None' or val == 'none':
        return None

    # convert to float or int.
    try:
        val = float(val)
    except ValueError:
        return val
    if val.is_integer():
        return int(val)
    return val

p = poet

if __name__ == '__main__':
    poet(sys.argv[1], *sys.argv[2:])

