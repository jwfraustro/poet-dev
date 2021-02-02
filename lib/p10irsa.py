'''
This module replaces the script which was previously used
to generate IRSA tables and FITS files.
'''

from __future__ import print_function
import numpy as np
import irsa
import os
import run

def make_irsa(events=None, directory='./', config='p10.pcf'):

    cfg = open(config, 'r')

    lines = cfg.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n')
        
    authors     = []
    instruments = []
    programs    = []

    # Read the file.
    # Yes, this is rudimentary.
    i = 0
    while lines[i] != 'end':
        if lines[i] != '':
            if lines[i][0] == '#':
                i += 1
        if lines[i] == '':
            i += 1
        if lines[i] == 'papername':
            i += 1
            papername = lines[i]
            i += 1
        if lines[i] == 'authors':
            i += 1
            while lines[i] != '':
                authors.append(lines[i])
                i += 1
        if lines[i] == 'month':
            i += 1
            month = lines[i]
            i += 1
        if lines[i] == 'year':
            i += 1
            year = lines[i]
            i += 1
        if lines[i] == 'journal':
            i += 1
            journal = lines[i]
            i += 1
        if lines[i] == 'instruments':
            i += 1
            while lines[i] != '':
                instruments.append(lines[i])
                i += 1
        if lines[i] == 'programs':
            i += 1
            while lines[i] != '':
                programs.append(lines[i])
                i += 1

    # Reformat lists into text 3 cases:
    # 1. Item is the last in a list with more than 1 entry
    # 2. Item is the only item in the list
    # 3. Any other situation
    authors_text = ''
    for i in range(len(authors)):
        if i == len(authors) - 1 and len(authors) > 1:
            authors_text += ('and ' + authors[i])
        elif len(authors) == 1:
            authors_text += (authors[i])        
        else:
            authors_text += (authors[i] + ', ')

    programs_text = ''
    for i in range(len(programs)):
        if i == len(programs) - 1 and len(programs) > 1:
            programs_text += ('and ' + programs[i])
        elif len(programs) == 1:
            programs_text += (programs[i])    
        else:
            programs_text += (programs[i] + ', ')

    instruments_text = ''
    for i in range(len(instruments)):
        if i == len(instruments) - 1 and len(instruments) > 1:
            instruments_text += ('and ' + instruments[i])
        elif len(instruments) == 1:
            instruments_text += (instruments[i])
        else:
            instruments_text += (instruments[i] + ', ')

    
    # Set the topstring. 
    topstring = "This file contains the light curves from the paper:  " + papername + " by " + authors_text + ", which was submitted in " + month + " " + year + " to " + journal + ". The data are from the " + instruments_text + " on the NASA Spitzer Space Telescope, programs " + programs_text + ", which are availabe from the public Spitzer archive (http://sha.ipac.caltech.edu). The paper cited above and its electronic supplement, of which this file is a part, describe the observations and data analysis. The data are in an ASCII table that follows this header. The TTYPE* keywords describe the columns in the ASCII table. Units are microJanskys for flux, pixels for distance, and seconds for time. All series (pixel positions, frame number, etc.) are zero-based. Data in a table row are valid if GOOD equals 1, invalid otherwise."

    # FITS files are silly things, and so we need our topstring
    # to only have, at most, 70 characters per line. But we want
    # it to be readable, so we need to split lines intelligently.
    # Loop goes farther than len(topstring) because we are 'inserting'
    # characters
    linelen = 65
    
    for i in range(2*len(topstring)):
        # Initialize reformatted string
        if i % linelen == 0 and i == linelen:
            topstring_reformat = topstring[:i]
            # Set a mark so we know where to begin the next chunk
            mark = i
        elif i % linelen == 0 and i != linelen and i != 0:
            # Add another line of linelen characters to the topstring
            # starting from the mark
            topstring_reformat += ('\n' + topstring[mark:i])
            # Update the mark
            mark = i

    # Add on any remaining trailing characters
    topstring_reformat += topstring[mark:]

    # Strip off extra newlines at the end
    topstring_reformat = topstring_reformat.rstrip('\n')
            
    if events == None:
        events = run.p7Restore(directory)

    if not os.path.isdir(directory + '/irsa'):
        os.makedirs(directory + '/irsa')

    for i in np.arange(len(events)):
        irsa.do_irsa(events[i], events[i].fit[0],
                     topstring=topstring_reformat,
                     directory=directory)

    return
