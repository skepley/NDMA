"""
Function and design testing for the HillComponent class

    Output: output
    Other files required: none

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 4/3/20; Last revision: 4/3/20
"""

import numpy as np
from hill_model import HillCoordinate, isvector

gamma = 1.2
interactionSign = [1, -1]
parameter1 = np.array([[np.nan, 3, 5, 4.1],
                       [1, np.nan, 6, np.nan]], dtype=float)

x = np.array([3., 2, 2, 1, 2, 3])  # assume these HillCoordinates are coordinates in a vector field on R^6
p1 = np.array([1.1, 2.2, 3.3])

f1 = HillCoordinate(parameter1, interactionSign, [2], [0, 1, 2], gamma=gamma)
print(f1(x, p1))
print(f1.dx(x, p1))   # derivative is embedded back as a vector in R^6
print([f1.diff(j, x, p1) for j in range(3)])


parameter2 = np.array([[1.1, 3, 5, np.nan],
                       [np.nan, 2.2, 6, 3.3]], dtype=float)
f2 = HillCoordinate(parameter2, interactionSign, [2], [0, 1, 2])
p2 = np.array([gamma, 4.1, 1])
print(f2(x, p2))
print(f2.dx(x, p2))
print([f2.diff(j, x, p2) for j in range(3)])
