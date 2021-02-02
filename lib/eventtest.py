#! /usr/bin/env python3
"""This is a module for testing POET. This runs in Python 3 and
should be able to load output from python 2 sessions and compare with
output from Python 3 sessions."""


import instrument   as inst
import numpy        as np
import p1event      as p1
import pdataread    as pd
import tepclass     as tep
import manageevent  as me
import readeventhdf as rdhdf
import reader3      as rd3


# items to not compare
_exclude = {
    p1.Event : {"initpars", 'save', 'calc', 'read', 'check',
                'eventdir', 'tep', 'inst', 'logname',
                'ancildir', 'paramsfile', 'modeldir',
                'initvalsfile'},
    inst.Instrument : set(),
    pd.FrameParameters : set(),
    rdhdf.fits : {'modelfile', 'burnparamsfile',
                  'allparamsfile', 'allknotfile'},
    rd3.Data : set(),
    rdhdf.indices : set(),
    tep.tepfile : {"convunits", "fname", "version"}
}
# tepfields to check as a generic as opposed to a tepclass.param
_tepSpecial   = {"fname", "version"}


#def easyCompare(loc1, loc2, 


#def compareEvents(loc1, loc2, **kwargs):
#ecode='wa012bs12', centdirs=[], photdirs=[], flerr=.0001, excludeEvent=[], excludeTep=[], excludeFp=[]):
"""
Compares two events or two itterables of events. Lists of events
must be of same length and same order within list. Prints differences
to screen.

Parameters
----------
loc1, loc2: string, list, tuple, p1.Event
    These fields are a bit flexible and their type decides how they are
     interpreted.
    If string, these are interpreted as the 'run' directories to load events
     from.
    If list or tuple, these are expected to be iterables of p1.Event objects.
     The other fields will be ignored in this case (except flerr)
    If p1.Event object, the function acts similarly to if they were iterables.
    I
Returns
-------
    

Example
-------
>>> # this assumes you ran with fgc centering aper photometry size 2.0
>>> #  (inner and outer annul as default)
>>> 
>>> import manageevent as me
>>> import eventtest as test
>>> 
>>> here  = 'run/'          # path to first run directory
>>> there = '../test/run/'  # path to second run directory
>>> files = 'wa012bs12_ini', 'wa012bs12_bpm', 'fgc/wa012bs12_ctr', 'fgc/ap2000715/wa012bs12_pht', 'fgc/ap2000715/wa012bs12_p5c'
>>> events1 = [me.loadevent( here + file) for file in files]
>>> events2 = [me.loadevent(there + file) for file in files]
>>> 
>>> test.compareEvents(events1, events2)
>>> 
>>> # to check p6 and p6 events (these should have many differences):
>>> import run
>>> here  = here  + "fgc/ap2000715/p6output/"
>>> there = there + "fgc/ap2000715/p6output/"
>>> events1 = [run.p6Restore( here + 'd-wa012bs12-6model'), 
...            run.p7Restore( here + d-wa012bs12-7anal)]
>>> events2 = [run.p6Restore(there + 'd-wa012bs12-6model'), 
...            run.p7Restore(there + d-wa012bs12-7anal)]
>>> 
>>> test.compareEvents(events1, events2)
"""
'''    kwargs['excludeEvent'] = kwargs.get('excludeEvent',
                                        set()).union(_excludeEvent)
    kwargs['excludeTep']   = kwargs.get('excludeTep',
                                        set()).union(_excludeTep)
    kwargs['excludeFp']    = kwargs.get('excludeFp', set())

    if isinstance(loc1, p1.Event):
        events1, events2, names = [loc1], [loc2], ['Event']

    elif isinstance(loc1, (list, tuple)):
        events1, events2, names = loc1, loc2, np.arange(len(loc2))

    else:

        events1, events2, names = [], [], []

        ecode = kwargs['ecode']

        file = '/' + ecode + '_ini'
        _append(events1, events2, names, loc1, loc2, file, 'p1', ['data', 'uncd'])

        file = '/' + ecode + '_bpm'
        _append(events1, events2, names, loc1, loc2, file, 'p2',
                ['data', 'uncd', 'mask'])
            
        for cdir in kwargs['centdirs']:

            file = '/' + cdir + '/' + ecode + '_ctr'
            name = 'p3 ' + cdir
            _append(events1, events2, names, loc1, loc2, file, name,
                    ['data', 'uncd', 'mask'])

            for pdir in kwargs['photdirs']:

                file = '/' + cdir + '/' + pdir + '/' + ecode + '_pht'
                name = 'p4 ' + cdir + ' ' + pdir
                _append(events1, events2, names, loc1, loc2, file, name)

                file = '/' + cdir + '/' + pdir + '/' + ecode + '_p5c'
                name = 'p5 ' + cdir + ' ' + pdir
                _append(events1, events2, names, loc1, loc2, file, name)

    for ind, (event1, event2, name) in enumerate(zip(events1, events2, names)):

        print("\n\nComparing:", name)
        print(79*'-'+'\n')

        edir1 = [attrib for attrib in dir(event1)
                 if attrib[0] != '_' and
                 attrib not in kwargs['excludeEvent']]
        edir2 = [attrib for attrib in dir(event2)
                 if attrib[0] != '_' and
                 attrib not in kwargs['excludeEvent']]

        compareDirs(edir1, edir2, '\b', ind)
                
        for attrib in edir1:

            elt1 = getattr(event1, attrib)
            elt2 = getattr(event2, attrib)

            if isinstance(elt1, np.ndarray):
                compareNdarr(elt1, elt2, attrib, ind, **kwargs)

            elif isinstance(elt1, tep.tepfile):
                compareTep(elt1, elt2, attrib, ind, **kwargs)

            elif isinstance(elt1, inst.Instrument):
                compareInst(elt1, elt2, attrib, ind, **kwargs)

            elif isinstance(elt1, pd.FrameParameters):
                compareFp(elt1, elt2, attrib, ind, **kwargs)

            else:
                compareGen(elt1, elt2, attrib, ind, **kwargs)

    return events1, events2, names

def _append(events1, events2, names, loc1, loc2, file, name, include=[]):
    events1.append(me.loadevent(loc1 + file, load=include))
    events2.append(me.loadevent(loc2 + file, load=include))
    names.append(name)
'''

def compare(elt1, elt2, attrib='event'):

    if isinstance(elt1, (p1.Event, inst.Instrument, pd.FrameParameters,
                         rdhdf.fits, rd3.Data, rdhdf.indices)):
        compareCustom(elt1, elt2, attrib, _exclude[elt1.__class__])

    elif isinstance(elt1, np.ndarray):
        compareNdarr(elt1, elt2, attrib)

    elif isinstance(elt1, tep.tepfile):
        compareTep(elt1, elt2, attrib, _exclude[tep.tepfile])

    else:
        compareGen(elt1, elt2, attrib)

def compareCustom(obj1, obj2, attrib, exclude=set()):

    odir = compareDirs(obj1, obj2, attrib, exclude)

    for param in odir:
        compare(getattr(obj1, param), getattr(obj2, param), attrib+'.'+param)

def compareTep(tep1, tep2, attrib, exclude=set()):

    tdir = compareDirs(tep1, tep2, attrib, exclude)

    for param in tdir:
        param1 = getattr(tep1, param)
        param2 = getattr(tep2, param)
        for field in "val", "uncert", "ref", "unit":
            field1 = getattr(param1, field)
            field2 = getattr(param2, field)
            compareGen(field1, field2, attrib+'.'+param+'.'+field)

    for param in _tepSpecial:
        param1 = getattr(tep1, param)
        param2 = getattr(tep2, param)
        compareGen(param1, param2, attrib+"."+param)

def compareGen(gen1, gen2, attrib):

    if isinstance(gen1, (list, tuple)):
        for i , (elt1, elt2) in enumerate(zip(gen1, gen2)):
            compare(elt1, elt2, attrib+"["+str(i)+"]")
        return

    if isinstance(gen1, np.bytes_):
        gen1 = gen1.astype(str)

    if isinstance(gen2, np.bytes_):
        gen2 = gen2.astype(str)

    if not gen1 == gen2:
        print('\nattribute', attrib, "differs.")
        print("first", attrib, ":\n", gen1)
        print("second", attrib, ":\n", gen2)

def compareNdarr(arr1, arr2, attrib):
    if not isinstance(arr2, np.ndarray):
        print("attribute", attrib, "is ndarray in first object but not array",
              "in second object. casting to ndarray...")
        arr2 = np.array(arr2)

    if arr1.dtype.type == np.bytes_:
        arr1 = arr1.astype(str)
    if arr2.dtype.type == np.bytes_:
        arr2 = arr2.astype(str)

    diffarr = arr1 != arr2

    if issubclass(arr1.dtype.type, (np.number, float, int)):
        nans1 = np.isnan(arr1)
        nans2 = np.isnan(arr2)
        if nans1.sum() != 0:
            arr1 = arr1[~nans1]
            arr2 = arr2[~nans1]

        if (nans1 != nans2).sum() != 0:
            print(attrib, 'has different nans.')

        nonzeros = np.where(arr1!=0)
        nzeros = (arr1==0).sum()
        arr1 = arr1[nonzeros]
        arr2 = arr2[nonzeros]

        diff = diffarr[~nans1].sum()

    else:
        diff = diffarr.sum()

    if diff != 0:
        print("\nattibute", attrib, "differs in", diff, "places.")
        print("same in", (~diffarr).sum(), 'places.')
        arrdiff = arr2 - arr1
        reletivediff = arrdiff / arr1
        print("excluding", nzeros, "zero values.")
        if reletivediff.size != 0:
            print("average absolute relative error:", np.abs(reletivediff).mean())
            print("maximum absolute relative error:", np.abs(reletivediff).max())
        else:
            print("no values remaining due to excluded zeros")
            
def compareDirs(elt1, elt2, attrib, exclude=set()):

    dir1 = [param for param in dir(elt1)
             if param[0] != '_' and param not in exclude]
    dir2 = [param for param in dir(elt2)
             if param[0] != '_' and param not in exclude]

    for param in dir1:
        if param not in dir2:
            print(param, "in first", attrib, "but not in second", attrib)
            print("ignoring", attrib + '.' + param + "...")
            dir1.remove(param)

    for param in dir2:
        if param not in dir1:
            print(param, "in second", attrib, "but not in first", attrib)
            print("ignoring ", attrib + '.' + param + "...")
            dir2.remove(param)

    return dir1
