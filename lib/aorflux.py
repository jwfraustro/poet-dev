import numpy as np

def aorflux(posparams, x, etc = []):
  """
    Create a model that fits the median flux for each aor.

    Parameters
    ----------
    posparams: Position parameters
    x:         
    etc:       Not used.

    Returns
    -------
    This function returns an array of y values ...

    Revisions
    ---------
    2012-07-03    Patricio   pcubillos@fulbrightmail.org
                  Adapted from posflux.
  """
  nobj, wherepos = x
  normfactors = np.ones(nobj)
  # SET posparams[0] = 1/ (PRODUCT OF posparams[1:])
  # posparams[0] = 1/np.product(posparams[1:len(wherepos)])
  for i in np.arange(len(wherepos)):
    normfactors[wherepos[i]] = posparams[i]

  return normfactors
