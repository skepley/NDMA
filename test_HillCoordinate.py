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
componentParm = np.array([[1.1, 2.4, 1, 2],
                          [1.2, 2.3, 2, 3],
                          [1.3, 2.2, 3, 2],
                          [1.4, 2.1, 4, 3]], dtype=float)
x = np.array([3., 2, 4, 1], dtype=float)

parameter1 = np.copy(componentParm)
pVars = tuple(zip(*product(range(4), range(4))))  # set all parameters as variable
parameter1[pVars] = np.nan
p = ezcat(gamma, componentParm[pVars])  # get floating points for all variable parameters
f = HillCoordinate(parameter1, interactionSign, interactionType, [1, 0, 1, 2, 3])
y = (x, p)
yx = f.dx(x, p)
yxx = f.dx2(x, p)
yxxx = f.dx3(x, p)
yp = f.diff(x, p)
ypx = f.dxdiff(x, p)
ypxx = f.dx2diff(x, p)
ypp = f.diff2(x, p)
yppx = f.dxdiff2(x, p)





# # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
# DxH = f.diff_component(x, p, [1, 0], fullTensor=True)
# DxxH = f.diff_component(x, p, [2, 0], fullTensor=True)
# DxxxH = f.diff_component(x, p, [3, 0], fullTensor=True)
#
# # get tensors for derivatives of p o H(x) (outer terms of chain rule)
# Dp = f.diff_interaction(x, p, 1)  # 1-tensor
# D2p = f.diff_interaction(x, p, 2)  # 2-tensor
# D3p = f.diff_interaction(x, p, 3)  # 3-tensor




stopHere
parameter2 = np.copy(componentParm)
p2Vars = [[0, -1], [1, 0]]  # set n_1, and ell_2 as variable parameters
parameter2[0, -1] = parameter1[1, 0] = np.nan
p2 = np.array([gamma, componentParm[0, -1], componentParm[1, 0]], dtype=float)
f2 = HillCoordinate(parameter2, interactionSign, interactionType, [0, 1, 2])  # gamma is a variable parameter too
print(f2(x, p2))
print(f2.dx(x, p2))
print(f2.dn(x, p2))
print(f2.diff(x, p2, 1))
print(f2.dx2(x, p2))

# check that diff and dn produce equivalent derivatives
parameter3 = np.copy(componentParm)
p3Vars = [[0, -1], [1, -1]]  # set n_1, and n_2 as variable parameters
parameter3[0, -1] = parameter3[1, -1] = np.nan
p3 = np.array([componentParm[0, -1], componentParm[1, -1]], dtype=float)
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
