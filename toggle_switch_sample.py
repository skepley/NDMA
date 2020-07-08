"""
Analysis of dynamics for the toggle switch
   
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 6/29/20; Last revision: 6/29/20
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from hill_model import *
from saddle_node import SaddleNode
from models import ToggleSwitch

np.set_printoptions(precision=10, floatmode='maxprec_equal')

# set some parameters to test using MATLAB toggle switch for ground truth
decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)

f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)


# gamma_1 = 1.
# gamma_2 = 1.
# ell_1 = np.linspace(1, 5, 10)
# delta_1 = np.linspace(1, 5, 10)
# theta_1 = np.linspace(1, 15, 10)
# ell_2 = np.linspace(1, 5, 10)
# delta_2 = np.linspace(1, 5, 10)
# theta_2 = np.linspace(1, 15, 10)
# parameterGrid = np.reshape(np.meshgrid(gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2), [8, 10 ** 6])
# allSols = np.zeros(np.shape(parameterGrid)[1])


# ==== find saddle node for a parameter choice
# rhoInitData = np.linspace(2, 10, 5)
# rho = 5
# t0 = time.time()
# nSample = 10 ** 4
# parameterData = np.random.random_sample([nSample, 8])
# allSols = np.zeros(nSample)
# for j in range(nSample):
#     p = parameterData[j, :]
#     jSols = SN.find_saddle_node(0, rho, p, freeParameterValues=rhoInitData)
#     if len(jSols) > 0:
#         allSols[j] = np.min(jSols)
#     if j % 1000 == 0:
#         print(j)
#
# tf = time.time() - t0
# print(tf)
# np.savez('tsData', rhoInitData, np.array([rho]), allSols, parameterData)


def dsgrn_region(parameter):
    """Return a dsgrn classification of a parameter as an integer in {1,...,9}"""

    def factor_slice(gamma, ell, delta, theta):
        T = gamma * theta
        if T <= ell:
            return 0
        elif ell < T <= ell + delta:
            return 1
        else:
            return 2

    return 1 + 3 * factor_slice(*parameter[[0, 1, 2, 7]]) + factor_slice(*parameter[[4, 5, 6, 3]])

npData = np.load('tsData.npz')
rhoInitData = npData['arr_0.npy']
rho = 5
allSols = npData['arr_2.npy']
parameterData = npData['arr_3.npy']
dsgrnData = np.array([dsgrn_region(parameterData[j]) for j in range(np.shape(parameterData)[0])])
nnzSols = np.array([parameterData[j] for j in range(np.shape(parameterData)[0]) if allSols[j] > 0])
solByRegion = np.array([dsgrn_region(p) for p in nnzSols])
countByRegion = np.array([sum(solByRegion == j) for j in range(1, 10)])
percentByRegion = np.reshape(100 / len(nnzSols) * countByRegion, [3, 3])
print(percentByRegion)

oddBalls = np.array([p for p in nnzSols if dsgrn_region(p) != 5])
nOddBalls = len(oddBalls)

# oddBallSN = []
# for j in range(nOddBalls):
#     p = nnzSols[j]
#     jSols = SN.find_saddle_node(0, rho, p, freeParameterValues=rhoInitData)
#     print(j, jSols)
#     oddBallSN.append(jSols)

j = 45
p = nnzSols[j]
jSols = SN.find_saddle_node(0, rho, p, freeParameterValues=rhoInitData)
for hill in jSols:
    plt.figure()
    f.plot_nullcline(ezcat(hill, nnzSols[j]), nNodes=200)

for hill in np.linspace(jSols[0] - 1, jSols[1] + 10, 10):
    plt.figure()
    f.plot_nullcline(ezcat(hill, nnzSols[j]), nNodes=1000)
