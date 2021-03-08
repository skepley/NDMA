"""
Function and design testing for the SaddleNode class

    Output: output
    Other files required: none

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 4/13/20; Last revision: 4/13/20
"""
import numpy as np
from hill_model import HillComponent, HillCoordinate,  HillModel, find_root
from saddle_node import SaddleNode
from models import ToggleSwitch


# set some parameters to test using MATLAB toggle switch for ground truth
decay = np.array([1, 1], dtype=float)
p1 = np.array([1, 5, 3], dtype=float)
p2 = np.array([1, 6, 3], dtype=float)
x = np.array([4, 3])

f = ToggleSwitch(decay, [p1, p2])
f1 = f.coordinates[0]
f2 = f.coordinates[1]
H1 = f1.components[0]
H2 = f2.components[0]
hill = 4.1

SN = SaddleNode(f)
eq = f.find_equilibria(10, hill)
# v0 = np.array([1, -.7])
v0 = np.array([1, 1])
x0 = eq[:, 1]
u0 = np.concatenate((x0, v0, np.array(hill)), axis=None)

uSol = find_root(SN.zero_map, SN.diff_zero_map, u0, diagnose=True)
print(uSol)
xSol, vSol, nSol = [uSol.x[idx] for idx in [[0, 1], [2, 3], [4]]]

# plot nullclines
plt.close('all')
plt.figure()
f.plot_nullcline(4.1)
plt.figure()
f.plot_nullcline(nSol)