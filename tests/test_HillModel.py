"""
Function and design testing for the HillModel class


    Output: output
    Other files required: none

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 4/6/2020 
"""
from ndma.hill_model import *
from ndma.model import Model
from ndma.activation import HillActivation

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
productionSigns = [[1, 1, 1], [1, 1], [1]]  # all production terms are activating
productionTypes = [[3], [2], [1]]  # all production terms in a single summand
productionIndex = [[0, 1, 2], [0, 2], [0]]
parameter = [nan_array(len(productionIndex[idx]), 4) for idx in range(nCoordinate)]

f = Model(gamma, parameter, productionSigns, productionTypes, productionIndex)  # define HillModel
# get easy access to Hill productionComponents
f0 = f.coordinates[0]
f1 = f.coordinates[1]
f2 = f.coordinates[2]

# set evaluation parameters and state variables
gammaVals = np.array([10 * (j + 1) for j in range(nCoordinate)])
pHill = np.array([1, 2, 4, 3])  # choose some Hill function parameters to use for all Hill functions.
pVals = [ezcat(*len(productionIndex[idx]) * [pHill]) for idx in range(nCoordinate)]
x = np.array([4, 3, 2], dtype=float)
p = ezcat(*[ezcat(*tup) for tup in zip(gammaVals, pVals)])  # this only works when all parameters are variable
# set up callable copy of Hill function which makes up all production terms
H = HillActivation(1, ell=pHill[0], delta=pHill[1], theta=pHill[2], hillCoefficient=pHill[3])

# check f evaluation
print('f eval')
y = f(x, p)
print(y)
print(-gammaVals[0] * x[0] + H(x[0]) + H(x[1]) + H(x[2]))
print(-gammaVals[1] * x[1] + H(x[0]) + H(x[2]))
print(-gammaVals[2] * x[2] + H(x[0]))
print('\n')

# check dx evaluation
print('dx eval')
yx = f.dx(x, p)
print(yx)
print(np.array([
    [-gammaVals[0] + H.dx(x[0], []), H.dx(x[1], []), H.dx(x[2], [])],
    [H.dx(x[0], []), -gammaVals[1], H.dx(x[2], [])],
    [H.dx(x[0], []), 0, -gammaVals[2]]
]))
print('\n')

# check diff evaluation
print('diff eval')
yp = f.diff(x, p)
print(yp)
print('\n')


# check dx2 evaluation
print('dx2 eval')
yxx = f.dx2(x, p)
print(yxx)

# check dxdiff evaluation
print('dxdiff eval')
ypx = f.dxdiff(x, p)
print(ypx)

# check diff2 evaluation
ypp = f.diff2(x, p)
print(ypp)