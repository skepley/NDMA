"""
This is a simple use of the build-in ToggleSwitch class and a small show of the functionalities of the library
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path

from saddle_finding_functionalities import count_eq, saddle_node_search
from saddle_node import SaddleNode
from models.TS_model import ToggleSwitch
from create_dataset import create_dataset_ToggleSwitch, subsample

# defining the parameters to be np.nan keeps them free to be determined at computation
decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)

# using the special class to inherit the sturcture AND the equal Hill parameters
f = ToggleSwitch(decay, [p1, p2])
print("The Hill model is")
print(f)
print("where all Hill coefficients are set to be equal")

hill_vec = [1, 5, 10, 30]
gridDensity = 5

parameter = np.array([3,4,5,3,2.4])
n_eqs = np.empty([1,1])
for hill in hill_vec:
    n_eqs = n_eqs.append(count_eq(f, hill, parameter))
for hill, n_eq in zip(hill_vec, n_eqs):
    print("At Hill coefficient ", hill," we find ", n_eq," equilibria")

# if the number of equilibria changes, let's look for saddle nodes!
if n_eqs[0] != n_eqs[-1]:
    SN = SaddleNode(f)
    hillRange = [1, 5, 10, 15, 20, 25, 30] # a finer comb o look for the saddle node interval
    SNParameters, badCandidates = saddle_node_search(f, hillRange, parameter, bisectionBool=True)
    print(SNParameters)


