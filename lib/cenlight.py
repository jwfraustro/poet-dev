# $Author: kmcin $
# $Revision: initial $
# $Date: 2019-09-05 19:54:41 $

import numpy as np
import least_asymmetry as ctr
import disk


def col(data, loc, npskyrad, mask=None):
    data_copy = data.copy()

    if type(mask) == type(None):
        mask = np.ones(np.shape(data_copy), dtype=bool)
    else:
        mask = mask.astype(bool)

    sky_ann = disk.disk(npskyrad[1], (loc[1],loc[0]), data_copy.shape) ^ \
              disk.disk(npskyrad[0], (loc[1],loc[0]), data_copy.shape)

    sky_mask = sky_ann * mask * np.isfinite(data_copy)
    
    sky = np.median(data_copy[np.where(sky_mask)])

    data_copy -= sky
    
    return ctr.col(data_copy)



def ccl(data, rad, loc, npskyrad, mask=None):
    data_copy = data.copy()

    if type(mask) == type(None):
        mask = np.ones(np.shape(data_copy), dtype=bool)
    else:
        mask = mask.astype(bool)

    sky_ann = disk.disk(npskyrad[1], (loc[1],loc[0]), data_copy.shape) ^ \
              disk.disk(npskyrad[0], (loc[1],loc[0]), data_copy.shape)

    sky_mask = sky_ann * mask * np.isfinite(data_copy)
    
    sky = np.median(data_copy[np.where(sky_mask)])

    data_copy -= sky
    
    x_data = data_copy.shape[1]
    y_data = data_copy.shape[0]
    
    x_rad = (x_data - 1)/2
    y_rad = (y_data - 1)/2
    
    weights = np.ones(data_copy.shape, dtype=float)
    ind = np.indices(weights.shape)
    
    x_ind = ind[1] - x_rad
    y_ind = ind[0] - y_rad
    weights[np.where(np.sqrt(x_ind**2 + y_ind**2)>rad)] = 0
    
    return ctr.col(data_copy, weights=weights)
