"""
Troubleshoot problems with the bootstrap algorithm for finding equilibria in the Toggle Switch

    Output: output
    Other files required: none
    See also: OTHER_SCRIPT_NAME,  OTHER_FUNCTION_NAME
   
    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 9/30/2021
"""
"""
This code creates the heat map, as presented in the article.
The heat map indicates the value of the Hill coefficient in which a saddle node is taking place depending on the parameter.
It also consider the parameter projection into [0,3]x[0,3] thanks to the DSGRN region definition
"""

from hill_model import *
from saddle_finding_functionalities import *
from toggle_switch_heat_functionalities import *
import numpy as np
import matplotlib.pyplot as plt
from models.TS_model import ToggleSwitch
import sys
from create_dataset import *

# define the saddle node problem for the toggle switch
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)

# use dataset creation
# size of the sample
n_sample = 100
file_name = 'TS_data_100000.npz'
try:
    np.load(file_name)
except FileNotFoundError:
    n = 100
    create_dataset_TS(n, file_name)
    print('this happened')

file_storing = 'heat_map.npz'
data, regions, coefs = load_dataset(file_name)

parameterArray = np.transpose(data)
eqNewton = []
eqBootstrap = []
hillCoef = 4
badParms = []
for j, parm in enumerate(parameterArray):
    eqNewton.append(f.find_equilibria(10, hillCoef, parm, bootstrap=False))
    # eqBootstrap.append(f.find_equilibria(10, hillCoef, parm))
    eqBootstrap.append(f.bootstrap_enclosure(hillCoef, parm)[1])
    if eqBootstrap[-1] is None:
        badParms.append((j, parameterArray[j]))

p = badParms[0][1]

# Now blow up the bootstrap enclosure method
fullParm = f.parse_parameter(
    hillCoef, p)  # concatenate all parameters into a vector with hill coefficients sliced in
P0, P1 = parameterByCoordinate = f.unpack_parameter(fullParm)  # unpack variable parameters by component
g0, p0 = f.coordinates[0].parse_parameters(P0)
g1, p1 = f.coordinates[1].parse_parameters(P1)
H0 = f.coordinates[0].productionComponents[0]
H1 = f.coordinates[1].productionComponents[0]
x0Bounds = (1 / g0) * H0.image(p0[0])
x1Bounds = (1 / g1) * H1.image(p1[0])
u0 = np.array(list(zip(x0Bounds, x1Bounds))).flatten()  # zip initial bounds

# G0 = lambda x : (1 / g0) * H0(x[1], p0[0])
# G1 = lambda x : (1 / g1) * H1(x[0], p1[0])

# iterate the bootstrap map to obtain an enclosure
tol = 1e-13
Phi = f.bootstrap_map(hillCoef, p)
maxIter = 10
u = u0
notConverged = True
nIter = 0
while nIter < maxIter and notConverged:
    uNew = Phi(u)
    tol_loc = np.linalg.norm(uNew - u)
    if nIter>3:
        uveryOld = uOld
    uOld = u
    notConverged = np.linalg.norm(uNew - u) > tol
    u = uNew
    print(u)
    nIter += 1