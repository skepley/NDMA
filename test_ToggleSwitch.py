"""
One line description of what the script performs (H1 line)
  
Optional file header info (to give more details about the function than in the H1 line)
Optional file header info (to give more details about the function than in the H1 line)
Optional file header info (to give more details about the function than in the H1 line)

    Output: output
    Other files required: none
    See also: OTHER_SCRIPT_NAME,  OTHER_FUNCTION_NAME
   
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 6/9/20; Last revision: 6/9/20
"""

import numpy as np
from hill_model import *

# set some parameters to test using MATLAB toggle switch for ground truth
decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)

f = ToggleSwitch(decay, [p1, p2])
f0 = f.coordinates[0]
f1 = f.coordinates[1]
F0 = HillCoordinate(np.array(4 * [np.nan]), [-1], [1], [0, 1])
F1 = HillCoordinate(np.array(4 * [np.nan]), [-1], [1], [1, 0])

H0 = f0.components[0]
H1 = f1.components[0]
n0 = 4.1

x0 = np.array([4, 3])
p0 = np.array([1, 1, 5, 3, 1, 1, 6, 3],
              dtype=float)  # (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)
P0 = ezcat(p0[0:4], n0)
P1 = ezcat(p0[4:], n0)
# print(f0.dx(x0, P0))
# print(F0.dx(x0, P0))
# print(f0.dx2(x0, P0))
# print(F0.dx2(x0, P0))
print(f0.dn(x0, P0))
print(f0.diff(x0, P0, 4))
print(f1.dn(x0, P1))
print(F1.diff(x0, P1, 4))
print(f0.dndx(x0, P0))
print(F0.dxdiff(x0, P0, 4))
