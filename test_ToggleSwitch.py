"""
Testing and analysis of the ToggleSwitch model

    Other files required: models, hill_model

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 6/9/20; Last revision: 6/24/20
"""

import numpy as np
from hill_model import *
from models import ToggleSwitch

# TESTING FOR TOGGLE SWITCH
# ============= set up the toggle switch example to test on =============
nCoordinate = 2
gamma = np.array([1, 1], dtype=float)
hill = 4.1


componentParmValues = [np.array([1, 5, 3], dtype=float), np.array([1, 6, 3], dtype=float)]
parameter1 = [np.copy(cPValue) for cPValue in componentParmValues]
compVars = [[j for j in range(3)] for i in range(nCoordinate)]  # set all parameters as variable
for i in range(nCoordinate):
    parameter1[i][compVars[i]] = np.nan

gammaVar = np.array([np.nan, np.nan])  # set both decay rates as variables
f = ToggleSwitch(gammaVar, parameter1)
f0 = f.coordinates[0]
f1 = f.coordinates[1]
H1 = f0.components[0]
H2 = f1.components[0]

# set some data to check evaluations with
x = np.array([4, 3], dtype=float)
x1 = x[0]
x2 = x[1]
p = ezcat(*[ezcat(*tup) for tup in zip(gamma, componentParmValues)])  # this only works when all parameters are variable
p0 = ezcat(p[:4], hill)  # parameters for f1 and f2 function calls
p1 = ezcat(p[4:], hill)
H1p = p0[1:]  # parameters for H1 and H2 function calls
H2p = p1[1:]

print('f = \n', f(x, hill, p))
print('dx = \n', f.dx(x, hill, p))
print('dx2 = \n', f.dx2(x, hill, p))
print('dx3 = \n', f.dx3(x, hill, p))
print('diff = \n', f.diff(x, hill, p))
# print('dxdiff = \n', f.dxdiff(x, hill, p))
# print('diff2 = \n', f.diff2(x, hill, p))
# print('dx2diff = \n', f.dx2diff(x, hill, p))
# print('dxdiff2 = \n', f.dxdiff2(x, hill, p))

#
# print(f.diff(x, hill, p, diffIndex=0))
# eq = f.find_equilibria(10, hill, p)
# f.plot_nullcline(hill, p)
