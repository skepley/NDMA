"""
Function and design testing for the HillComponent class

    Output: output
    Other files required: none

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 4/3/20; Last revision: 4/3/20
"""

import numpy as np
from hill_model import HillCoordinate, is_vector

gamma = 1.2
interactionSign = [1, -1]
interactionType = [2]
fullParm = np.array([[1.1, 3, 5, 4.1],
                     [1, 2.2, 6, 3.3]], dtype=float)
x = np.array([3., 2, 2, 1, 2, 3])  # assume these HillCoordinates are coordinates in a vector field on R^6

parameter1 = np.copy(fullParm)
p1Vars = [[0, 0], [1, 1], [1, -1]]  # set ell_1, delta_2, and n_2 as variable parameters
parameter1[0, 0] = parameter1[1, 1] = parameter1[1, -1] = np.nan
p1 = np.array([fullParm[0, 0], fullParm[1, 1], fullParm[1, -1]], dtype=float)
f1 = HillCoordinate(parameter1, interactionSign, interactionType, [0, 1, 2], gamma=gamma)
print(f1(x, p1))
print(f1.dx(x, p1))  # derivative is embedded back as a vector in R^6
# print([f1.diff(j, x, p1) for j in range(3)])


parameter2 = np.copy(fullParm)
p2Vars = [[0, -1], [1, 0]]  # set n_1, and ell_2 as variable parameters
parameter2[0, -1] = parameter1[1, 0] = np.nan
p2 = np.array([gamma, fullParm[0, -1], fullParm[1, 0]], dtype=float)
f2 = HillCoordinate(parameter2, interactionSign, interactionType, [0, 1, 2])  # gamma is a variable parameter too
print(f2(x, p2))
print(f2.dx(x, p2))
# print([f2.diff(j, x, p2) for j in range(3)])
print(f2.dn(x, p2))
print(f2.diff(1, x, p2))

# check that diff and dn produce equivalent derivatives
parameter3 = np.copy(fullParm)
p3Vars = [[0, -1], [1, -1]]  # set n_1, and n_2 as variable parameters
parameter3[0, -1] = parameter3[1, -1] = np.nan
p3 = np.array([fullParm[0, -1], fullParm[1, -1]], dtype=float)
f3 = HillCoordinate(parameter3, interactionSign, interactionType, [0, 1, 2],
                    gamma=gamma)  # gamma is a variable parameter too
print([f3.diff(j, x, p3) for j in range(f3.nVariableParameter)])

# check summand evaluations
parameter4 = np.repeat(np.nan, 12).reshape(3, 4)
interactionType = [2, 1]
interactionSign = [1, 1, -1]
f4 = HillCoordinate(parameter4, interactionSign, interactionType, [0, 1, 2, 3], gamma=gamma)
summand = [[0, 1], [2], [3, 4]]
