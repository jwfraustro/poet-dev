"""
    This class loads a control file (pcf) and lets you
    querry the parameters and values.

    Constructor Parameters:
    -----------------------
    file : A control file containing the parameters and values.

    Notes:
    ------
    A parameter can have one or more values, differet parameters can
    have different number of values.

    The function Param.get(index) automatically interprets the type of the
    values. If they can be cast into a numeric value retuns a numeric
    value, otherwise returns a string.

    Examples:
    --------
    >>> # Load a pcf file:
    >>> import reader3 as rd
    >>> reload(rd)
    >>> pcf = rd.Pcffile('/home/patricio/ast/esp01/anal/wa011bs11/run/wa011bs11.pcf')

    >>> Each parameter has the attribute value, wich is a ndarray:
    >>> pcf.planet.value
    array(['wa011b'], 
          dtype='|S6')

    >>> # To get the n-th value of a parameter use pcffile.param.get(n):
    >>> # if it can't be converted to a number/bool/etc, it returns a string.
    >>> pcf.planet.get(0)
    'wa011b'
    >>> pcf.photchan.get(0)
    1
    >>> pcf.fluxunits.get(0)
    True

    >>> # Use pcffile.param.value[n] to get the n-th value as string:
    >>> pcf.aorname.get(0)
    38807808
    >>> pcf.aorname.value[0]
    '38807808'

    >>> # The function pcffile.param.getarr() returns the numeric/bool/etc
    >>> # values of a parameter as a nparray:
    >>> pcf.sigma.value
    array(['4.0', '4.0'], 
          dtype='|S5')
    >>> pcf.sigma.getarr()
    array([4.0, 4.0], dtype=object)


    Modification History:
    --------------------
    2009-01-02 chris      Initial Version.  
                          by Christopher Campo      ccampo@gmail.com 
    2010-03-08 patricio   Modified from ccampo version.
                          by Patricio Cubillos      pcubillos@fulbrightmail.org
    2010-10-27 patricio   Docstring updated
    2011-02-12 patricio   Merged with ccampo's tepclass.py
    2017-06-23 zacchaeus  added a simple read function
    2018-06-24 zacchaeus  rewrote everything to use configparser and dicts
"""


import configparser
import numpy as np


class Pcf:

    def __init__(self, pcfdict):

        for key, val in pcfdict.items():
            setattr(self, key, val)

    def make_file(self, name, header, mode='w'):

        file = open(name, mode)

        file.write("[" + header + "]\n")

        for key, val in vars(self).items():
            if isinstance(val, list):
                val = [str(elt) for elt in val]
                file.write('   '.join([key]+val) + '\n')
            else:
                file.write(key + '   ' + str(val)+ '\n')

        file.write("\n")
        file.close()


def read_pcf(filename, header, expand=True, simple=False, d1lists=[], d2lists=[], nparrs=[]):
    """
    This function generates Pcf objects from config files.

    Parameters
    ----------
    filename: string, file pointer
        location to read the configuration from.
    header: str
        The name of the section of the config to read. This will also read all
        sections that start with `header` unless you specify a simple read
        (so specifying "photometry" would include "photometry1" and
        "photometry2").
    expand: bool, optional
        If True, will generate a list of Pcfs with every combination of the
        values specified. Specifying `simple` overrides this functionality.
        If False, the values will all be lists
    simple: bool, optional
        If True, returns a Data objects with each parameter specified in the
        config as a parameter of the returned object.
    d1lists: iterable, optional
        Used for simple reads, ignored otherwise. specifies which parameters
        should be interpreted as a one-dimensional list.
    d2lists: iterable, optional
        Used for simple reads, ignored otherwise. specifies which parameters 
        should be interpreted as a two-dimensional list.
    nparrs: iterable, optinal
        Used for simple reads, ignored otherwise. specifies which parameters
        should be converted to a numpy array at read in.
    Returns
    -------
    Pcfs: list
        A list of Pcf objects (or Data objects for simple reads)
    Examples
    --------
    >>> # setting expand to False keeps inputs as lists
    >>> import reader3 as rd
    >>> pcfs = rd.read_pcf("wa012bs12.pcf", "event", False)
    >>> print(type(pcfs))
    list
    >>> pcf = pcfs[0]
    >>> print(pcf.runp2)
    [False]

    >>> # By default, `expand` is True so each entry is a single value
    >>> pcfs = rd.read_pcf("wa012bs12.pcf", "centering")
    >>> print(pcfs[0].method)
    "fgc"

    >>> # This must be the exact name of the section header because it is a
    >>> # simple read.
    >>> pcf = rd.read_pcf("wa012bs12.pcf", "params", simple=True,
                          d2list=['model', 'numit'], nparrs=['numit])
    >>> print(pcf.model)
    [['linramp', 'madelecl'],
     ['madelecl']]
    >>> print(pcf.numit)
    [[1e3, 1e5]]
    >>> print(pcf.nchains)
    10
    """

    # 
    config = configparser.ConfigParser(delimiters=' ', comment_prefixes='#', inline_comment_prefixes='#', empty_lines_in_values=False)
    config.read(filename)

    if simple:
        return simple_read(config[header], d1lists, d2lists, nparrs)

    section_dicts = []
    sections = (config[head] for head in config.sections()
                if head.startswith(header))
    for section in sections:

        section_dict = {key : formt(section[key].split()) for key in section}
        section_dicts.append(section_dict)

    if not expand:
        return [Pcf(pcfdict) for pcfdict in section_dicts]

    all_dicts = []
    for section_dict in section_dicts:

        expanded_dicts = [{}]
        for key, values in section_dict.items():

            copies = [ [pcfdict.copy() for pcfdict in expanded_dicts]
                       for i in range(len(values)-1)]

            for pcfdicts, value in zip(copies + [expanded_dicts], values):
                for pcfdict in pcfdicts:
                    pcfdict[key] = value

            for copy in copies:
                expanded_dicts.extend(copy)

        all_dicts.extend(expanded_dicts)

    return [Pcf(pcfdict) for pcfdict in all_dicts]

def copy_config(src, keep_sections, dest, mode='a', mod={}):

    config = configparser.ConfigParser(delimiters=' ', comment_prefixes='#', inline_comment_prefixes='#', empty_lines_in_values=False)
    config.read(src)

    for conf_section in config.sections():
        keep = False
        for keep_section in keep_sections:
            if conf_section.startswith(keep_section):
                keep = True
        if not keep:
            config.remove_section(conf_section)

    for sect in mod:
        for conf_section in config.sections():
            if conf_section.startswith(sect):
                for key in mod[sect]:
                    if isinstance(mod[sect][key], str):
                        config[conf_section][key] = mod[sect][key]
                    elif isinstance(mod[sect][key], int):
                        config[conf_section][key] = str(mod[sect][key])
                    elif isinstance(mod[sect][key], list):
                        try:
                            config[conf_section][key] = '\n'.join([' ' + ' '.join(items) for items in mod[sect][key]])
                        except:
                            config[conf_section][key] = ' '.join(mod[sect][key])

    config.write(open(dest, mode))

def formt(val):
    """
    Turns string into float/int/True/False/None if need be.
    """
    # if val is a list, reccursively format the list
    if isinstance(val, list):
        return [formt(elt) for elt in val]

    # convert val to bool, none, float, or int as needed
    if val == 'True' or val == 'true':
        return True
    elif val == 'False' or val == 'true':
        return False
    elif val == 'None' or val == 'none':
        return None

    try:
        val = float(val)
    except ValueError:
        return val
    
    if val.is_integer():
        return int(val)
    return val

def simple_read(config, d1lists, d2lists, nparrs):

    pcf = Pcf({})
    for param in config:
        # add to object in appropriate manner
        if param in d1lists:
            setattr(pcf, param, formt(config[param].split()))
        elif param in d2lists:
            setattr(pcf, param, [formt(line.split())
                                 for line in config[param].split('\n')])
        else:
            setattr(pcf, param, formt(config[param]))
        if param in nparrs:
            setattr(pcf, param, np.array(getattr(pcf, param)))

    return pcf

def readnpy(file, rows):
    """
Read a npy file using 1.0 syntax. (2.0 is only ever used if the header exceeds
65535 bytes.)

POET writes arrays with np.save one line of a 2d array at a time to avoid
having the whole array in memmory at any point in time. This is non-standard
so this function reconstructs the data from our writes.

Parameters
----------
filename: string or file object
    file containging the array.

Returns
-------
array: numpy.memmap
    array without headers strewn thoughout array.
"""
    if isinstance(file, str):
        file = open(file, 'rb')

    header    = file.readline()
    headerlen = len(header)
    # npy headers start with 10 bytes of we dont care. What remains is
    # a python valid dictionary.
    header    = eval(header[10:].decode())
    cols,     = header['shape']
    dtype     = np.dtype(header['descr'])
    size      = headerlen // dtype.itemsize

    assert size == headerlen / dtype.itemsize, """things are bad. The only thing to do now is refactor p6 and mcmc to use numpy.memmaps to write to files instead of using arrays then writing those arrays with np.save to the same file."""

    return np.memmap(file, dtype=dtype, mode='r').reshape(rows, -1)[:, size:]
