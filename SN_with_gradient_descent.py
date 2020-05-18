"""
Minimization example with respect to HIll Coefficient for the toggle switch example
   
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 5/15/20; Last revision: 5/15/20
"""

import numpy as np
import matplotlib.pyplot as plt
from hill_model import *
from saddle_node import *

# set some parameters
decay = np.array([1, 1], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
n0 = 4.1
SN = SaddleNode(f)

# ==== find saddle node minimizer for some initial parameter choice
p0 = np.array([1, 5, 3, 1, 6, 3],
              dtype=float)  # (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)
p1 = np.array([1, 4, 3, 1, 5, 3], dtype=float)

localMinimum = SN.find_minimizer(p1)
print(localMinimum)
