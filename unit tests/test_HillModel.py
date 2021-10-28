"""
Function and design testing for the HillModel class


    Output: output
    Other files required: none

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 4/6/20; Last revision: 6/23/20
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from hill_model import *


def nan_array(m, n):
    """Return an m-by-n numpy array or vector of np.nan values"""

    # if m ==1:  # return a vector
    #     return np.array([np.nan for idx in range(n)])
    # else:  # return a matrix
    nanArray = np.empty((m, n))
    nanArray[:] = np.nan
    return nanArray


# ============= set up the Network 12 example to test on =============
nCoordinate = 3

# set all parameters as variable
gamma = np.array([np.nan for idx in range(nCoordinate)])  # set both decay rates as variables
interactionSigns = [[1, 1, 1], [1, 1], [1]]  # all interactions are activation
interactionTypes = [[3], [2], [1]]  # all interactions are single summand
interactionIndex = [[0, 1, 2], [0, 2], [0]]
parameter = [nan_array(len(interactionIndex[idx]), 4) for idx in range(nCoordinate)]

f = HillModel(gamma, parameter, interactionSigns, interactionTypes, interactionIndex)  # define HillModel
# get easy access to Hill productionComponents
f0 = f.coordinates[0]
f1 = f.coordinates[1]
f2 = f.coordinates[2]



# set evaluation parameters and state variables
gammaVals = np.array([10 * (j + 1) for j in range(nCoordinate)])
pHill = np.array([1, 2, 4, 3])  # choose some Hill function parameters to use for all Hill functions.
pVals = [ezcat(*len(interactionIndex[idx]) * [pHill]) for idx in range(nCoordinate)]
x = np.array([4, 4, 0], dtype=float)
p = ezcat(*[ezcat(*tup) for tup in zip(gammaVals, pVals)])  # this only works when all parameters are variable

print(f(x, p))
print(f.dx(x,p))

