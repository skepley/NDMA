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
from saddle_node import *
from models import ToggleSwitch
from scipy.interpolate import griddata


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
        equilibria = []
        for j in range(len(countVector)):
            countVector[j], equilibria[j] = count_eq(hill[j], p)
        return countVector
    else:
        eq = f.find_equilibria(gridDensity, hill, p)
        if eq is not None:
            return np.shape(eq)[1], eq  # number of columns is the number of equilibria found
        else:
            eq = f.find_equilibria(gridDensity * 2, hill, p)
            return np.shape(eq)[1], eq


def estimate_saddle_node(hill, p, gridDensity=10):
    """Attempt to predict whether p admits any saddle-node points by counting equilibria at each value in the hill vector.
    If any values return multiple equilibria, attempt to bound the hill parameters for which these occur. Otherwise,
    return an empty interval."""

    hillIdx = 0
    hill = ezcat(1, hill)  # append 1 to the front of the hill vector

    numEquilibria, Eq = count_eq(hill[0], p, gridDensity)
    numEquilibriaInf, Eqs = count_eq(hill[-1], p, gridDensity)

    hill_for_saddle = []
    equilibria_for_saddle = []

    if numEquilibriaInf > 1:
        n_steps = int(np.ceil((hill[-1] - hill[0])/5))
        #try:
        hill_SN, eqs = bisection(hill[0], hill[-1], p, n_steps)

        #except TypeError:
        #    print(hill[0], hill[-1], p, n_steps)

        hill_for_saddle.append(hill_SN)
        equilibria_for_saddle.append(eqs)
        return hill_for_saddle, equilibria_for_saddle

    while hillIdx < len(hill) - 1:
        hillMin = hill[hillIdx]  # update lower hill coefficient bound
        hillMax = hill[hillIdx + 1]  # update upper hill coefficient bound
        numEquilibriaOld = numEquilibria
        numEquilibria, eq = count_eq(hillMax, p, gridDensity)
        hillIdx += 1  # increment hill index counter
        if numEquilibria - numEquilibriaOld != 0:
            n_steps = int(np.ceil(log((hillMax - hillMin) / 5)))
            hill_SN, equilibria = bisection(hillMin, hillMax, p, n_steps)
            hill_for_saddle.append(hill_SN)
            equilibria_for_saddle.append(equilibria)

    return hill_for_saddle, equilibria_for_saddle


def bisection(hill0, hill1, p, n_steps):

    if n_steps is 0:
        return np.array([hill0, hill1])

    nEq0, Eq0 = count_eq(hill0, p)
    nEq1, Eq1 = count_eq(hill1, p)
    for i in range(n_steps):
        if hill1 - hill0 > 1:
            print('bisection ', i)
            hill_middle = (hill0 + hill1)/2
            nEqmiddle, EqMiddle = count_eq(hill_middle, p)

            if nEqmiddle == nEq0:
                hill0 = hill_middle
                nEq0 = nEqmiddle
                Eq0 = EqMiddle
            elif nEqmiddle == nEq1:
                hill1 = hill_middle
                nEq1 = nEqmiddle
                Eq1 = EqMiddle
            else:
                return hill_middle, EqMiddle
        else:
            break
    if nEq0 > nEq1:
        return hill0, Eq0
    else:
        return hill1, Eq1


def heat_coordinates(alpha, beta, alphaMax):
    """Returns the DSGRN heat map coordinates For a parameter of the form (alpha, beta) where
    alpha = ell / gamma and beta = (ell + delta) / gamma"""

    if beta < 1:  # (ell + delta)/gamma < theta
        x = beta

    elif alpha > 1:  # theta < ell/gamma
        x = 2 + (alpha - 1) / (alphaMax - 1)

    else:  # ell/gamma < theta < (ell + delta)/gamma
        x = 1 + (1 - alpha) / (beta - alpha)
    return x


def dsgrn_heat_plot(parameterData, colorData, alphaMax, heatMin=1000):
    """Produce a heat map plot of a given choice of toggle switch parameters using the specified color map data.
    ParameterData is a N-by-5 matrix where each row is a parameter of the form (ell_1, delta_1, gamma_2, ell_2, delta_2)"""

    nParameter = len(parameterData)
    alpha_1 = parameterData[:,0]
    beta_1 = parameterData[:,1]
    alpha_2 = parameterData[:, 3] / parameterData[:, 2]
    beta_2 = (parameterData[:, 3] + parameterData[:, 4]) / parameterData[:, 2]
    x = np.array([heat_coordinates(alpha_1[j], beta_1[j], alphaMax) for j in range(nParameter)])
    y = np.array([heat_coordinates(alpha_2[j], beta_2[j], alphaMax) for j in range(nParameter)])
    z = np.array([np.min([val, heatMin]) for val in colorData])

    # define grid.
    plt.figure()
    xGrid = np.linspace(0, 3, 100)
    yGrid = np.linspace(0, 3, 100)
    # grid the data.
    zGrid = griddata((x, y), z, (xGrid[None, :], yGrid[:, None]), method='linear')
    # contour the gridded data, plotting dots at the randomly spaced data points.
    zmin = 0
    zmax = np.nanmax(zGrid)
    palette = plt.matplotlib.colors.LinearSegmentedColormap('jet3', plt.cm.datad['jet'], 2048)
    palette.set_under(alpha=0.0)
    plt.imshow(zGrid, extent=(0, 3, 0, 3), cmap=palette, origin='lower', vmin=zmin, vmax=zmax, aspect='auto',
               interpolation='bilinear')
    # CS = plt.contour(xGrid, yGrid, zGrid, 15, linewidths=0.5, colors='k')
    # CS = plt.contourf(xGrid, yGrid, zGrid, 15, cmap=plt.cm.jet)
    plt.colorbar()  # draw colorbar
    # plot data points.
    plt.scatter(x, y, marker='o', c='k', s=2)
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    # plt.title('griddata test (%d points)' % npts)
    plt.show()
    for i in range(1, 3):
        plt.plot([i, i], [0, 3], 'k')
        plt.plot([0, 3], [i, i], 'k')



# ============ Sample the fiber and find saddle node points ============
# generate some sample parameters

nSample = 30
hillRange = [2, 1000]
hillDensity = [25]  # coarse, fine, ultrafine node density
parameterData = np.array([sampler() for j in range(nSample)])
nCourseHill = hillDensity[0]  # number of nodes used for for selection based on equilibrium counting
coarseInitialHillData = [hillRange[0], hillRange[1]]  # hill coefficient vector to use for candidate selection by counting equilibria

# compute saddle nodes on samples
t0 = time.time()
monostableParameters = []  # list for indices of monostable parameters
badCandidates = []  # list for parameters which pass the candidate check but fail to find a saddle node
SNParameters = []  # list for parameters where a saddle node is found
for j in range(nSample):
    p = parameterData[j]
    hill_for_saddle, equilibria_for_saddle = estimate_saddle_node(coarseInitialHillData, p)
    # print('Coarse grid: {0}'.format(candidateInterval))
    if hill_for_saddle is []:
        monostableParameters.append(j) # this is a monostable parameter
    else:
        while hill_for_saddle:  # p should have at least one saddle node point
            candidateHill = np.array(hill_for_saddle.pop())
            equilibria = np.array(equilibria_for_saddle.pop())
            print(equilibria.shape)
            SN_candidate_eq = SN_candidates_from_bisection(equilibria)
            jkSols = SN.find_saddle_node(0, candidateHill, p, equilibria=SN_candidate_eq)
            jSols = ezcat(jkSols)
            if len(jSols) > 0:
                jSols = np.unique(np.round(jSols, 10))  # uniquify solutions
                SNParameters.append((j, jSols))
            else:
                badCandidates.append((j, candidateHill, equilibria))  # this parameter passed the selection but failed to return any saddle not points

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

point_1 = badCandidates[1]
par = parameterData[point_1[0]]

f.plot_nullcline(point_1[1][0], par, domainBounds=((0.8, 1.3), (0.8, 1.3)))