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
    Date: 5/15/20; Last revision: 5/15/20
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from hill_model import ToggleSwitch, SaddleNode, find_root


def SN_root(SNinstance, parameter, u0):
    """Return a single root of the SaddleNode rootfinding problem"""

    root = find_root(lambda u: SNinstance.zero_map(u, parameter), lambda u: SNinstance.diff_zero_map(u, parameter), u0,
                     diagnose=True)
    return root


def SN_call(SNinstance, parameter, n0):
    """Temporary call method for SaddleNode class"""

    equilibria = SNinstance.model.find_equilibria(10, n0, parameter)
    initial_eigenvector = np.array([1, -.7])
    saddleNodePoints = list(filter(lambda root: root.success,
                                   [SN_root(SNinstance, parameter,
                                            np.concatenate((equilibria[:, j], initial_eigenvector, n0)))
                                    for j in
                                    range(equilibria.shape[1])]))  # return equilibria which converged
    # saddleNodePoints = list([SN_root(SNinstance, parameter,
    #                                         np.concatenate((equilibria[:, j], initial_eigenvector, n0)))
    #                                 for j in
    #                                 range(equilibria.shape[1])])  # return equilibria which converged
    hillCoefficients = np.concatenate(([sol.x[-1] for sol in saddleNodePoints], np.array([np.inf])))
    return np.min(hillCoefficients)


# set some parameters to test using MATLAB toggle switch for ground truth
decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)

f = ToggleSwitch(decay, [p1, p2])
f1 = f.coordinates[0]
f2 = f.coordinates[1]
H1 = f1.components[0]
H2 = f2.components[0]
n0 = 4.1

x0 = np.array([4, 3])
p0 = np.array([1, 1, 5, 3, 1, 1, 6, 3],
              dtype=float)  # (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)
SN = SaddleNode(f)

# ==== find saddle node for a parameter choice
p0 = np.array([1, 1, 5, 3, 1, 1, 6, 3], dtype=float)
SNpoints = SN_call(SN, p0, np.array([4.1]))
print(SNpoints)
# v0 = np.array([1, -.7])
# eq0 = f.find_equilibria(10, n0, p0)
# x0 = eq0[:, -1]
# u0 = np.concatenate((x0, v0, np.array(n0)), axis=None)
# u0Sol = SN_call_temp(SN, p0, u0)
# # print(u0Sol)
# x0Sol, v0Sol, n0Sol = [u0Sol.x[idx] for idx in [[0, 1], [2, 3], [4]]]
# compare to u0Sol = [ 4.55637172,  2.25827744,  0.82199933, -0.56948846,  3.17447061]
