"""
Sample saddle node bifurcations over a fiber of the dsgrn coordinate projection map.

    Output: output
    Other files required: none
    See also: OTHER_SCRIPT_NAME,  OTHER_FUNCTION_NAME
   
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 8/19/20; Last revision: 8/19/20
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from hill_model import *
from saddle_node import SaddleNode
from models import ToggleSwitch

plt.close('all')

# ============ initialize toggle switch Hill model and SaddleNode instance  ============
# This uses the non-dimensionalized parameters i.e. gamma_1 = 1theta_1 = theta_2 = 1.
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)


def sampler():
    """Sample non-hill parameters for the non-dimensionalized toggle switch over a fiber. This assumes that
    theta_1 = theta_2 = gamma_1 = 1 and other constraints which define the fiber. Returns a vector in R^5 of the form:
     (ell_1, delta_1, gamma_1, ell_2, delta_2). For this example we pick the upper right corner of the central DSGRN
     parameter region which are the parameters satisfying: alpha_i = theta_i = 1. The coordinates for this fiber in R^3
     are (delta_1, gamma_2, delta_2)."""

    # pick ell_2, delta_2
    ell_1 = 1.0  # fix alpha_1 = 1 = theta_1 which puts us on the line y = 2 in DSGRN coordinates
    delta_1 = 1.0 * np.random.random_sample()  # sample in (0, 1)

    # pick gamma_2
    gammaScale = 1 + 9 * np.random.random_sample()  # gamma scale in [1, 10]
    g = lambda x: x if np.random.randint(2) else 1 / x
    gamma_2 = g(gammaScale)  # sample in (.1, 10) to preserve symmetry between genes

    # pick ell_2, delta_2
    ellByGamma_2 = 1.0  # fix alpha_2 = 1 = theta_2 which puts us on the line x = 2 in DSGRN coordinates
    deltaByGamma_2 = np.random.random_sample()  # sample in (0, 1)
    ell_2 = ellByGamma_2 * gamma_2
    delta_2 = deltaByGamma_2 * gamma_2
    return ezcat(ell_1, delta_1, gamma_2, ell_2, delta_2)


def count_eq(hill, p, gridDensity=10):
    """Count the number of equilibria found for a given parameter"""
    if is_vector(hill):
        countVector = np.zeros_like(hill)
        for j in range(len(countVector)):
            countVector[j] = count_eq(hill[j], p)
        return countVector
    else:
        eq = f.find_equilibria(gridDensity, hill, p)
        if eq is not None:
            return np.shape(eq)[1]  # number of columns is the number of equilibria found
        else:
            return 0


def estimate_saddle_node(hill, p, gridDensity=10):
    """Attempt to predict whether p admits any saddle-node points by counting equilibria at each value in the hill vector.
    If any values return multiple equilibria, attempt to bound the hill parameters for which these occur. Otherwise,
    return an empty interval."""

    hillIdx = 0
    numEquilibria = 1
    hill = ezcat(1, hill)  # append 1 to the front of the hill vector

    while numEquilibria < 2 and hillIdx < len(hill) - 1:
        hillMin = hill[hillIdx]  # update lower hill coefficient bound
        hillMax = hill[hillIdx + 1]  # update upper hill coefficient bound
        numEquilibria = count_eq(hillMax, p, gridDensity)
        hillIdx += 1  # increment hill index counter

    if numEquilibria < 2:
        return None  # p is monostable

    else:  # p is a candidate for a saddle node parameter
        return np.array([hillMin, hillMax])


# ============ Sample the fiber and find saddle node points ============
# generate some sample parameters
nSample = 10 ** 3
hillRange = [2, 1000]
hillDensity = [25, 5, 5]  # coarse, fine, ultrafine node density
parameterData = np.array([sampler() for j in range(nSample)])
nCourseHill = hillDensity[0]  # number of nodes used for for selection based on equilibrium counting
nFineHill = hillDensity[1]  # number of nodes for refined Hill candidate interval subdivision (equilibria are recomputed)
nUltraFineHill = hillDensity[2]  # number of initial Hill candidates for each refined candidate (equilibria are NOT recomputed)
coarseInitialHillData = np.linspace(hillRange[0], hillRange[1],
                                    nCourseHill)  # hill coefficient vector to use for candidate selection by counting equilibria

# compute saddle nodes on samples
t0 = time.time()
monostableParameters = []  # list for indices of monostable parameters
badCandidates = []  # list for parameters which pass the candidate check but fail to find a saddle node
SNParameters = []  # list for parameters where a saddle node is found
for j in range(nSample):
    p = parameterData[j]
    candidateInterval = estimate_saddle_node(coarseInitialHillData, p)
    # print('Coarse grid: {0}'.format(candidateInterval))
    if candidateInterval is not None:  # p should have a saddle node point
        jSols = np.array([])  # initialize list to hold solutions
        fineInitialHillData = np.linspace(*candidateInterval, nFineHill)
        # print('Medium grid: {0}'.format(fineInitialHillData))
        fineHillStep = fineInitialHillData[1] - fineInitialHillData[0]  # step size for ultra-fine Hill grid
        fattenGrid = ezcat(fineInitialHillData[0], fineInitialHillData,
                           fineInitialHillData[-1])  # append double copies of the first and last initial condition
        for k in range(1, len(fattenGrid) - 1):
            hill = fattenGrid[k]
            ultraFineHill = np.arange(fattenGrid[k - 1], fattenGrid[k + 1],
                                      fineHillStep / (
                                              nUltraFineHill - 1 - 1e-13))  # get additional grid of hill coefficients
            # print('Fine grid: {0}'.format(ultraFineHill))
            jkSols = SN.find_saddle_node(0, hill, p, freeParameterValues=ultraFineHill)
            jSols = ezcat(jkSols)

        if len(jSols) > 0:
            jSols = np.unique(np.round(jSols, 10))  # uniquify solutions
            SNParameters.append((j, jSols))

        else:
            badCandidates.append(j)  # this parameter passed the selection but failed to return any saddle not points

    else:  # this is a monostable parameter
        monostableParameters.append(j)

    print(j)

tf = time.time() - t0
print('Computation time: {0} hours'.format(tf/(24*60)))
np.savez('upperRightFiberData', parameterData, hillRange, hillDensity, monostableParameters, badCandidates, SNParameters)

npData = np.load('upperRightFiberData.npz', allow_pickle=True)
parameterData = npData['arr_0.npy']
hillRange = npData['arr_1.npy']
hillDensity = npData['arr_2.npy']
monostableParameters = npData['arr_3.npy']
badCandidates = npData['arr_4.npy']
SNParameters = npData['arr_5.npy']
# print(badCandidates)
# print(SNParameters)
