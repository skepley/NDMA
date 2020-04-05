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
    Date: 4/3/20; Last revision: 4/3/20
"""

import numpy as np
from hill_model import HillComponent, HillCoordinate

gamma = 1.2
p1 = np.array([np.nan, 3, 5, 4.1], dtype=float)
p2 = np.array([1, np.nan, 6, np.nan], dtype=float)
parameter = np.row_stack([p1, p2])
interactionSign = [1, -1]
x = np.array([3, 2, 2, 1, 2, 3])
p = np.array([1, 2, 3])

f1 = HillCoordinate(parameter, interactionSign, [2], [0, 1, 2], gamma=gamma)
print(f1(x, p))
