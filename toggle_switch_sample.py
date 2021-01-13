"""
Sample and analyze saddle node bifurcations for the toggle switch
   
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
from scipy.interpolate import griddata
from toggle_switch_heat_functionalities import sampler
# from binData import bin_data
import numpy.ma as ma

plt.close('all')
# ============ all parameters are variables ============
# decay = np.array([np.nan, np.nan], dtype=float)
# p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
# p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)


# ============ non-dimensionalized version ============
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)

f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)





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


def grid_lines():
    """Add grid lines to a dsgrn coordinate plot"""
    for i in range(1, 3):
        plt.plot([i, i], [0, 3], 'k')
        plt.plot([0, 3], [i, i], 'k')
    return

def dsgrn_plot(parameterArray, alphaMax):
    """A scatter plot in DSGRN coordinates of a M-by-5 dimensional array. These are nondimensional parameters with rows
    of the form: (ell_1, delta_1, gamma_2, ell_2, delta_2)."""

    alpha1 = parameterArray[:, 0]
    beta1 = parameterArray[:, 0] + parameterArray[:, 1]
    alpha2 = parameterArray[:, 3] / parameterArray[:, 2]
    beta2 = (parameterArray[:, 3] + parameterArray[:, 4]) / parameterArray[:, 2]
    x = np.array(
        [heat_coordinates(alpha1[j], beta1[j], alphaMax) for j in range(len(alpha1))])
    y = np.array(
        [heat_coordinates(alpha2[j], beta2[j], alphaMax) for j in range(len(alpha2))])
    plt.scatter(x, y, marker='o', c='k', s=2)
    grid_lines()


# ============ DENSE HILL COEFFICIENTS ============

# # compute saddle nodes on samples
# nSample = 10 ** 4
# parameterData = np.array([sampler() for j in range(nSample)])
# t0 = time.time()
# hillInitialData = np.linspace(2, 100, 25)
# hill = 1
# allSols = np.zeros([nSample, 2])
# for j in range(nSample):
#     p = parameterData[j]
#     jSols = SN.find_saddle_node(0, hill, p, freeParameterValues=hillInitialData)
#     if len(jSols) == 1:
#         allSols[j] = ezcat(jSols, 0)
#     elif len(jSols) == 2:
#         allSols[j] = jSols
#     if j % 1000 == 0:
#         print(j)
#
# tf = time.time() - t0
# print(tf)
# np.savez('UniformTSData', hillInitialData, np.array([hill]), allSols, parameterData)
# npData = np.load('UniformTSData.npz')


# ============ SPARSE HILL COEFFICIENTS ============

# # compute saddle nodes on samples
nSample = 10 ** 3
parameterData = np.array([sampler() for j in range(nSample)])
t0 = time.time()
hillInitialData = np.linspace(2, 1000, 250)
hill = 1
allSols = np.zeros([nSample, 2])
for j in range(nSample):
    p = parameterData[j]
    jSols = SN.find_saddle_node(0, hill, p, freeParameterValues=hillInitialData)
    if len(jSols) == 1:
        allSols[j] = ezcat(jSols, 0)
    elif len(jSols) == 2:
        allSols[j] = jSols
    if j % 1000 == 0:
        print(j)

tf = time.time() - t0
print(tf)
np.savez('UniformTSDataLong', hillInitialData, np.array([hill]), allSols, parameterData)
npData = np.load('UniformTSDataLong.npz')

hillInitialData = npData['arr_0.npy']
hill = npData['arr_1.npy'][0]
allSols = npData['arr_2.npy']
parameterData = npData['arr_3.npy']
nSample = np.shape(parameterData)[0]

hillInitialData = hillInitialData[::10]  # downsample by an order of magnitude
eqCounts = np.zeros([nSample, len(hillInitialData)])
for j in range(nSample):
    print(j)
    eqCounts[j] = count_eq(hillInitialData, parameterData[j])

np.savez('eqCounts', eqCounts)
eqData = np.load('eqCounts.npz')
eqCounts = eqData['arr_0.npy']

# plot samples with more than 1 equilibrium at any parameter value
plt.figure()
SNCandidateIdx = [j for j in range(nSample) if np.any(eqCounts[j] > 1)]
SNCandidate = parameterData[SNCandidateIdx]
dsgrn_plot(SNCandidate, 1.5)

# plot samples with 1 equilibrium at every parameter value
plt.figure()
monostableIdx = [j for j in range(nSample) if np.all(eqCounts[j] <= 1)]
monostableData = parameterData[monostableIdx]
dsgrn_plot(monostableData, 1.5)


# plot all samples
plt.figure()
dsgrn_plot(parameterData, 1.5)


stopHere

nnzIdx = allSols[:, 0] > 0
hillThreshold = np.percentile(allSols[nnzIdx, 0], 95)
goodIdx = nnzIdx * (allSols[:, 0] < hillThreshold)  # throw out possible bad data points
minSols = allSols[goodIdx, 0]
alpha_1 = parameterData[goodIdx, 0]
beta_1 = parameterData[goodIdx, 1]
alpha_2 = parameterData[goodIdx, 3] / parameterData[goodIdx, 2]
beta_2 = (parameterData[goodIdx, 3] + parameterData[goodIdx, 4]) / parameterData[goodIdx, 2]
x = np.array([heat_coordinates(alpha_1[j], beta_1[j]) for j in range(len(alpha_1))])
y = np.array([heat_coordinates(alpha_2[j], beta_2[j]) for j in range(len(alpha_2))])
z = np.array([np.min([minSols[j], 1000]) for j in range(len(minSols))])

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

# # PLOT COLOR MAP BY BINNING
# plt.figure()
# # enter the gridding.  imagine drawing a symmetrical grid over the
# # plot above.  the binsize is the width and height of one of the grid
# # cells, or bins in units of x and y.
# binsize = 0.3
# grid, bins, binloc = bin_data(x, y, z, binsize=binsize)  # see this routine's docstring
#
# # minimum values for colorbar. filter our nans which are in the grid
# zmin = grid[np.where(np.isnan(grid) == False)].min()
# zmax = grid[np.where(np.isnan(grid) == False)].max()
#
# # colorbar stuff
# palette = plt.matplotlib.colors.LinearSegmentedColormap('jet3', plt.cm.datad['jet'], 2048)
# palette.set_under(alpha=0.0)
#
# # plot the results.  first plot is x, y vs z, where z is a filled level plot.
# extent = (x.min(), x.max(), y.min(), y.max())  # extent of the plot
# plt.subplot(1, 2, 1)
# plt.imshow(grid, extent=extent, cmap=palette, origin='lower', vmin=zmin, vmax=zmax, aspect='auto',
#            interpolation='bilinear')
# plt.colorbar()
#
# plt.show()
# for i in range(1, 3):
#     plt.plot([i, i], [0, 3], 'k')
#     plt.plot([0, 3], [i, i], 'k')


# dsgrnData = np.array([dsgrn_region(parameterData[j]) for j in range(np.shape(parameterData)[0])])
# nnzSols = np.array([parameterData[j] for j in range(np.shape(parameterData)[0]) if allSols[j] > 0])
# solByRegion = np.array([dsgrn_region(p) for p in nnzSols])
# countByRegion = np.array([sum(solByRegion == j) for j in range(1, 10)])
# percentByRegion = np.reshape(100 / len(nnzSols) * countByRegion, [3, 3])
# print(percentByRegion)
#
# oddBalls = np.array([p for p in nnzSols if dsgrn_region(p) != 5])
# nOddBalls = len(oddBalls)

# oddBallSN = []
# for j in range(nOddBalls):
#     p = nnzSols[j]
#     jSols = SN.find_saddle_node(0, hill, p, freeParameterValues=hillInitialData)
#     print(j, jSols)
#     oddBallSN.append(jSols)

# j = 45
# p = nnzSols[j]
# jSols = SN.find_saddle_node(0, hill, p, freeParameterValues=hillInitialData)
# for hill in jSols:
#     plt.figure()
#     f.plot_nullcline(ezcat(hill, nnzSols[j]), nNodes=200)

#
# for hill in np.linspace(jSols[0] - 1, jSols[1] + 10, 10):
#     plt.figure()
#     f.plot_nullcline(ezcat(hill, nnzSols[j]), nNodes=1000)
