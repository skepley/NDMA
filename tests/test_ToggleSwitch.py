"""
Testing and analysis of the ToggleSwitch model

    Other files required: models, hill_model

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 6/9/2020 
"""

from ndma.hill_model import *
from ndma.examples.TS_model import ToggleSwitch
import matplotlib.pyplot as plt

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
f1 = f.coordinates[0]
f2 = f.coordinates[1]
H1 = f1.productionComponents[0]
H2 = f2.productionComponents[0]

# set some data to check evaluations with
x = np.array([4, 3], dtype=float)
x1 = x[0]
x2 = x[1]
p = ezcat(*[ezcat(*tup) for tup in zip(gamma, componentParmValues)])  # this only works when all parameters are variable
f1p = ezcat(p[:4], hill)  # parameters for f1 and f2 function calls
f2p = ezcat(p[4:], hill)
H1p = f1p[1:]  # parameters for H1 and H2 function calls
H2p = f2p[1:]
print(p)
print('f=\n', f(x, hill, p))
print('diff =\n', f.diff(x, hill, p, diffIndex=0))
print('diff =\n', f.diff(x, hill, p))
print('dx =\n', f.dx(x, hill, p))
print('dx2 =\n', f.dx2(x, hill, p))
eq = f.find_equilibria(10, hill, p)
print('eq=\n', eq)

'''
should be

f=
 [-0.5        -0.58914356]
diff =
 [ 0.         -0.31043881]
diff =
 [[ 0.         -4.          1.          0.5         1.70833333  0.
   0.          0.          0.        ]
 [-0.31043881  0.          0.          0.          0.         -3.
   1.          0.23514274  1.47477518]]
dx =
 [[-1.         -1.70833333]
 [-1.10608138 -1.        ]]
dx2 =
 [[[0.         0.        ]
  [0.         0.56944444]]
 [[0.8770754  0.        ]
  [0.         0.        ]]]
eq=
 [[1.16100039 6.88005099]
 [3.58500776 2.9506309 ]
 [5.80337075 1.37597185]]
'''

f.plot_nullcline(hill, p)
plt.show()
