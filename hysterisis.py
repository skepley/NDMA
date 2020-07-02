import numpy as np
import scipy
from hill_model import *
from itertools import product
from saddle_node import *
from models import ToggleSwitch


def find_max_hysteresis(SNode, parameters, gamma0=1, x0=[1, 2]):
    # core element
    def distance_function(param):
        return hysteresis(SNode, param, gamma0, x0)

    optimum_parameters = scipy.optimize.minimize(distance_function, parameters, method='nelder-mead')
    return optimum_parameters


def hysteresis(SN, par0, gamma0, x0):
    # find two different saddles and return their distance in "gamma"
    jSearchNodes = np.linspace(gamma0/ 10, 10 * gamma0, 25)
    jSols = SN.find_saddle_node(1, gamma0, par0,  freeParameterValues=jSearchNodes)
    gamma1 = jSols[0]
    gamma2 = jSols[-1]
    return np.abs(gamma1-gamma2)


# testing
decay = np.array([1, 1], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
n0 = 4.1
SN_main = SaddleNode(f)

# ==== find saddle node minimizer for some initial parameter choice
p0 = np.array([1, 5, 3, 1, 6, 3],
              dtype=float)  # (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)
p1 = np.array([1, 4, 3, 1, 1, 5, 3], dtype=float)

localMinimum = find_max_hysteresis(SN_main, p1)
print(localMinimum)