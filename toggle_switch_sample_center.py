"""
Analysis of dynamics for the toggle switch restricted to the central DSGRN parameter region

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 6/30/20; Last revision: 6/30/20
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from hill_model import *
from saddle_node import SaddleNode
from models import ToggleSwitch

# set some parameters to test using MATLAB toggle switch for ground truth
decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)

f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)

# # =========== NON-DIMENSIONALIZED UNIFORM:  23.36 percent, 5,545 max Hill, 3.393 mean Hill
# saveFile = 'tsDataCenter'
# hillInitialData = np.linspace(2, 20, 10)
# hill = 3
#
#
# def sample_center():
#     """Return a sample point in the DSGRN central region"""
#     theta_1 = theta_2 = gamma_1 = 1.  # set fixed parameters for from non-dimensionalization
#     ell_1 = np.random.random_sample()  # sample in (0, 1)
#     u_1 = 1. + np.random.random_sample()  # sample ell_1 + delta_1 in (1,2)
#     delta_1 = u_1 - ell_1
#
#     gamma_2 = 0.1 + 9.9 * np.random.random_sample()  # set gamma_2 in (0.1, 10)
#     ell_2 = gamma_2 * np.random.random_sample()  # sample ell_1 in (0, gamma_2)
#     u_2 = gamma_2 + gamma_2 * np.random.random_sample()  # sample ell_2 + delta_2 in (gamma_2, 2*gamma_2)
#     delta_2 = u_2 - ell_2
#     return ezcat(gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)


# # =========== NON-DIMENSIONALIZED AWAY FROM BOUNDARY:  31.21 percent, 5.719 max Hill, 0.917 mean Hill
# saveFile = 'tsDataCenterMid'
# hillInitialData = np.linspace(2, 20, 10)
# hill = 3
# def sample_center():
#     """Return a sample point in the DSGRN central region"""
#     theta_1 = theta_2 = gamma_1 = 1.  # set fixed parameters for from non-dimensionalization
#     ell_1 = 0.5 * np.random.random_sample()  # sample in (0, 0.5)
#     u_1 = 1.5 + 0.5 * np.random.random_sample()  # sample ell_1 + delta_1 in (1.5, 2)
#     delta_1 = u_1 - ell_1
#
#     gamma_2 = 0.1 + 9.9 * np.random.random_sample()  # set gamma_2 in (0.1, 10)
#     ell_2 = 0.5 * gamma_2 * np.random.random_sample()  # sample ell_1 in (0, 0.5*gamma_2)
#     u_2 = 1.5 * gamma_2 + 0.5 * gamma_2 * np.random.random_sample()  # sample ell_2 + delta_2 in (1.5*gamma_2, 2*gamma_2)
#     delta_2 = u_2 - ell_2
#     return ezcat(gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)

# =========== NON-DIMENSIONALIZED AWAY FROM BOUNDARY, ORDERED LINEAR DECAY (g2 < g1): 29.82 percent, 5.346 max Hill,  0.875 mean Hill
saveFile = 'tsDataCenterMidGammaOrdered'
hillInitialData = np.linspace(2, 20, 10)
hill = 3


def sample_center():
    """Return a sample point in the DSGRN central region"""
    theta_1 = theta_2 = gamma_1 = 1.  # set fixed parameters for from non-dimensionalization
    ell_1 = 0.5 * np.random.random_sample()  # sample in (0, 0.5)
    u_1 = 1.5 + 0.5 * np.random.random_sample()  # sample ell_1 + delta_1 in (1.5, 2)
    delta_1 = u_1 - ell_1
    # node 2 parameters
    gamma_2 = 0.1 + 0.9 * np.random.random_sample()  # set gamma_2 in (0.1, 1)
    ell_2 = 0.5 * gamma_2 * np.random.random_sample()  # sample ell_1 in (0, 0.5*gamma_2)
    u_2 = 1.5 * gamma_2 + 0.5 * gamma_2 * np.random.random_sample()  # sample ell_2 + delta_2 in (1.5*gamma_2, 2*gamma_2)
    delta_2 = u_2 - ell_2
    return ezcat(gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)


# compute saddle nodes on samples
nSample = 10 ** 4
parameterData = np.array([sample_center() for j in range(nSample)])
t0 = time.time()
allSols = np.zeros(nSample)
for j in range(nSample):
    p = parameterData[j]
    jSols = SN.find_saddle_node(0, hill, p, freeParameterValues=hillInitialData)
    if len(jSols) > 0:
        allSols[j] = np.min(jSols)
    if j % 1000 == 0:
        print(j)

tf = time.time() - t0
print('Finished in {0} minutes \n'.format(round(tf / 60)))
np.savez(saveFile, hillInitialData, np.array([hill]), allSols, parameterData)

npData = np.load(saveFile + '.npz')
hillInitialData = npData['arr_0.npy']
hill = npData['arr_1.npy']
allSols = npData['arr_2.npy']
parameterData = npData['arr_3.npy']
# dsgrnData = np.array([dsgrn_region(parameterData[j]) for j in range(np.shape(parameterData)[0])])
nnzSols = np.array([parameterData[j] for j in range(np.shape(parameterData)[0]) if allSols[j] > 0])
print('{0} percent \n'.format(len(nnzSols) / 10 ** 2))
print('Max Hill: {0} \n'.format(np.max(allSols)))
print('Mean Hill: {0} \n'.format(np.mean(allSols)))
