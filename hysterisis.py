"""
Setting up the hysteresis problem

    Output: best parameters for hysteresis
    Other files required: Hill*, saddle_node

    Author: Elena Queirolo

"""


import numpy as np
import scipy
from hill_model import *
from itertools import product
from saddle_node import *


def find_max_hysteresis(SNode, parameters, gamma0=3, x0=[1, 2]):
    # core element
    def distance_function(param):
        return hysteresis(SNode, param, gamma0, x0)

    optimum_parameters = scipy.optimize.minimize(distance_function, parameters, method='nelder-mead')
    return optimum_parameters


def hysteresis(SN, par0, gamma0, x0):
    # find two different saddles and return their distance in "gamma"

    [gamma1, gamma2] = find_two_saddles(SN, par0, gamma0, x0)
    return np.abs(gamma1-gamma2)


def find_two_saddles(SN, par0, gamma0, x0):
    gamma1 = gamma0
    gamma2 = gamma0
    delta = 0.1
    iter = 0
    while gamma1 == gamma2 and iter < 100:
        [gamma1, x1] = SN(par0, gamma0 - iter * delta, x0)
        [gamma2, x2] = SN(par0, gamma0 + iter * delta, x0)
    if gamma1 == gamma2:
        return np.inf
    else:
        return gamma1,  gamma2


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
p1 = np.array([1, 4, 3, 1, 5, 3], dtype=float)

localMinimum = find_max_hysteresis(SN_main,p1)
print(localMinimum)