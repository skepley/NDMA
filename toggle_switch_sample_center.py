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

# ==== find saddle node for a parameter choice
rhoInitData = np.linspace(2, 10, 5)
rho = 5
t0 = time.time()
nSample = 10 ** 4
parameterData = np.zeros([nSample, 8])  # initialize parameter samples

# =========== set theta_i in (ell_i, ell_i + delta_i) /gamma_i  --- 1.41 percent
# parameterData[:, [0, 1, 2, 4, 5, 6]] = np.random.random_sample([nSample, 6])  # set random values for each gamma, ell, delta
# parameterData[:, 3] = parameterData[:, 1] / parameterData[:, 0] + np.random.random_sample(nSample
#                                                                                           ) * parameterData[:, 2] / parameterData[:, 0]
# parameterData[:, 7] = parameterData[:, 5] / parameterData[:, 4] + np.random.random_sample(nSample
#                                                                                           ) * parameterData[:, 6] / parameterData[:, 4]


# # =========== set theta_i as the midpoint of (ell_i, ell_i + delta_i) /gamma_i  --- 1.61 percent
# parameterData[:, [0, 1, 2, 4, 5, 6]] = np.random.random_sample([nSample, 6])  # set random values for each gamma, ell, delta in [0,1]
# parameterData[:, 3] = (parameterData[:, 1] + parameterData[:, 2]) / (2 * parameterData[:, 0])
# parameterData[:, 7] = (parameterData[:, 5] + parameterData[:, 6]) / (2 * parameterData[:, 4])


# # =========== set theta_i as the midpoint of (ell_i, ell_i + delta_i) /gamma_i  --- 2.65 percent
# parameterData[:, [1, 2, 5, 6]] = 10 * np.random.random_sample([nSample, 4])  # set random values for each ell, delta in [0,10], gamma = 1
# parameterData[:, 0] = 1  # set gamma_1
# parameterData[:, 4] = 1  # set gamma_2
# parameterData[:, 3] = (parameterData[:, 1] + parameterData[:, 2]) / 2  # set theta_1
# parameterData[:, 7] = (parameterData[:, 5] + parameterData[:, 6]) / 2  # set theta_2


# # =========== set theta_i as the midpoint of (ell_i, ell_i + delta_i) /gamma_i  --- 2.88 percent
# parameterData[:, [1, 2, 5, 6]] = 25 * np.random.random_sample(
#     [nSample, 4])  # set random values for each ell, delta in [0,10], gamma = 1
# parameterData[:, 0] = 1  # set gamma_1
# parameterData[:, 4] = 1  # set gamma_2
# parameterData[:, 3] = (parameterData[:, 1] + parameterData[:, 2]) / 2  # set theta_1
# parameterData[:, 7] = (parameterData[:, 5] + parameterData[:, 6]) / 2  # set theta_2


# # =========== NON-DIMENSIONALIZED COMPUTATION: set theta_i = gamma_1 = 1, \ell_i < 1 < \ell_i + \delta_i, gamma_2 in (0,2]  --- 21.26 percent
# parameterData[:, [1, 5]] = np.random.random_sample([nSample, 2])  # set random values for ell_i in [0,1]
# parameterData[:, [2, 6]] = 1 + np.random.random_sample([nSample, 2])  # set random values for delta_i in [1,2]
# parameterData[:, 4] = 2 * np.random.random_sample(nSample)  # set random values for gamma_2 in [0,2]
# parameterData[:, [0, 3, 7]] = 1  # set gamma_1, theta_1, theta_2 = 1


# =========== NON-DIMENSIONALIZED COMPUTATION: set theta_i = gamma_1 = 1, \ell_i < 0.5, 1.5 < \ell_i + \delta_i, gamma_2 in (0,2]  --- 54.44  percent
parameterData[:, [1, 5]] = 0.5 * np.random.random_sample([nSample, 2])  # set random values for ell_i in [0, 0.5]
parameterData[:, [2, 6]] = 1.5 + np.random.random_sample([nSample, 2])  # set random values for delta_i in [1.5, 2.5]
parameterData[:, 4] = 2 * np.random.random_sample(nSample)  # set random values for gamma_2 in [0,2]
parameterData[:, [0, 3, 7]] = 1  # set gamma_1, theta_1, theta_2 = 1

allSols = np.zeros(nSample)
for j in range(nSample):
    p = parameterData[j, :]
    jSols = SN.find_saddle_node(0, rho, p, freeParameterValues=rhoInitData)
    if len(jSols) > 0:
        allSols[j] = np.min(jSols)
    if j % 1000 == 0:
        print(j)

tf = time.time() - t0
print(tf)

# np.savez('tsDataCenter', rhoInitData, np.array([rho]), allSols, parameterData)
# np.savez('tsDataCenterMidpoint', rhoInitData, np.array([rho]), allSols, parameterData)
# np.savez('tsDataCenterMidpoint10', rhoInitData, np.array([rho]), allSols, parameterData)
# np.savez('tsDataCenterMidpoint25', rhoInitData, np.array([rho]), allSols, parameterData)
np.savez('tsDataNonDimMidpoint', rhoInitData, np.array([rho]), allSols, parameterData)




# npData = np.load('tsData.npz')
# allSols = npData['arr_2.npy']
# parameterData = npData['arr_3.npy']
# dsgrnData = np.array([dsgrn_region(parameterData[j]) for j in range(np.shape(parameterData)[0])])
nnzSols = np.array([parameterData[j] for j in range(np.shape(parameterData)[0]) if allSols[j] > 0])
print(len(nnzSols))
