import numpy as np
import matplotlib.pyplot as plt

def asech(eclparams, t, etc = []):
  """
  Calculates an asymmetric hyperbolic secant transit model.
  See Rappaport et al. (2014) and Croll et al. (2015).
  """
  tau0 = eclparams[0] # Appr. transit midpoint
  c    = eclparams[1] # Appr. the transit depth
  tau1 = eclparams[2] # Appr. ingress time
  tau2 = eclparams[3] # Appr. egress time
  flux = eclparams[4] # System flux

  y = 1 - 2 * c * 1. / (np.exp(-(t-tau0)/tau1) +
                        np.exp( (t-tau0)/tau2))

  return y*flux
