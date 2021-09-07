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

# ============= set up the toggle switch example to test on =============
nCoordinate = 2
gamma = np.array([1, 1], dtype=float)
componentParmValues = [np.array([1, 5, 3, 4.1], dtype=float), np.array([1, 6, 3, 4.1], dtype=float)]
parameter1 = [np.copy(cPValue) for cPValue in componentParmValues]
compVars = [[j for j in range(4)] for i in range(nCoordinate)]  # set all parameters as variable
for i in range(nCoordinate):
    parameter1[i][compVars[i]] = np.nan
gammaVar = np.array([np.nan, np.nan])  # set both decay rates as variables
interactionSigns = [[-1], [-1]]
interactionTypes = [[1], [1]]
interactionIndex = [[1], [0]]
f = HillModel(gammaVar, parameter1, interactionSigns, interactionTypes,
              interactionIndex)  # define HillModel for toggle switch by inheritance
f1 = f.coordinates[0]
f2 = f.coordinates[1]

x = np.array([4, 3], dtype=float)
p = ezcat(*[ezcat(*tup) for tup in zip(gamma, componentParmValues)])  # this only works when all parameters are variable
p1 = p[:5]
p2 = p[5:]

# H1 = f.components[0]
# H2 = f.components[1]
# H3 = f.components[2]
# H4 = f.components[3]
# pComp = [p[1:5], p[5:9], p[9:13], p[13:]]



y = f(x, p)
yx = f.dx(x,p)
yp = f.diff(x,p)
yxx = f.dx2(x,p)




# test Hill model equilibrium finding
eq = f.find_equilibria(10, p)
print(eq)
# added vectorized evaluation of Hill Models - DONE


# plot nullclines and equilibria
plt.close('all')
Xp = np.linspace(0, 10, 100)
Yp = np.linspace(0, 10, 100)
Z = np.zeros_like(Xp)

N1 = f1(np.row_stack([Z, Yp]), p1) / gamma[0]  # f1 = 0 nullcline
N2 = f2(np.row_stack([Xp, Z]), p2) / gamma[1]  # f2 = 0 nullcline

plt.figure()
plt.scatter(eq[0, :], eq[1, :])
plt.plot(Xp, N2)
plt.plot(N1, Yp)
