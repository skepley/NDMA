"""
Function and design testing for the HillComponent class

    Output: output
    Other files required: none

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 4/3/20; Last revision: 4/3/20
"""

import numpy as np
from hill_model import *
from itertools import product

gamma = 1.2
interactionSign = [1, -1, 1, -1]
interactionType = [2, 1, 1]
fullParm = np.array([[1.1, 2.4, 1, 2],
                     [1.2, 2.3, 2, 3],
                     [1.3, 2.2, 3, 2],
                     [1.4, 2.1, 4, 3]], dtype=float)
x = np.array([3., 2, 4, 1], dtype=float)

parameter1 = np.copy(fullParm)
pVars = tuple(zip(*product(range(4), range(4))))  # set all parameters as variable
parameter1[pVars] = np.nan
p = fullParm[pVars]  # get floating points for all variable parameters
f = HillCoordinate(parameter1, interactionSign, interactionType, [1, 0, 1, 2, 3])
print(f(x, p))
print(f.dx(x, p))  # derivative is embedded back as a vector in R^6

stopHere
parameter2 = np.copy(fullParm)
p2Vars = [[0, -1], [1, 0]]  # set n_1, and ell_2 as variable parameters
parameter2[0, -1] = parameter1[1, 0] = np.nan
p2 = np.array([gamma, fullParm[0, -1], fullParm[1, 0]], dtype=float)
f2 = HillCoordinate(parameter2, interactionSign, interactionType, [0, 1, 2])  # gamma is a variable parameter too
print(f2(x, p2))
print(f2.dx(x, p2))
print(f2.dn(x, p2))
print(f2.diff(x, p2, 1))
print(f2.dx2(x, p2))

# check that diff and dn produce equivalent derivatives
parameter3 = np.copy(fullParm)
p3Vars = [[0, -1], [1, -1]]  # set n_1, and n_2 as variable parameters
parameter3[0, -1] = parameter3[1, -1] = np.nan
p3 = np.array([fullParm[0, -1], fullParm[1, -1]], dtype=float)
f3 = HillCoordinate(parameter3, interactionSign, interactionType, [0, 1, 2],
                    gamma=gamma)  # gamma is a variable parameter too
print([f3.diff(x, p3, j) for j in range(f3.nVariableParameter)])

# check summand evaluations
parameter4 = np.repeat(np.nan, 12).reshape(3, 4)
interactionType = [2, 1]
interactionSign = [1, 1, -1]
p4 = np.arange(12)
f4 = HillCoordinate(parameter4, interactionSign, interactionType, [0, 1, 2, 3], gamma=gamma)
print(f4.diff_interaction(x, p4, 1))
print(f4.diff(x, p4))
print(f4.dx2(x, p4))
