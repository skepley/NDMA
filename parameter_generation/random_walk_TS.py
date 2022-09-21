import DSGRN
import numpy as np
import sys
import os
import inspect
from DSGRN_tools import *
from models.TS_model import ToggleSwitch
from tools_random_walk import *
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from create_dataset import *
from toggle_switch_heat_functionalities import *


decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
TS = ToggleSwitch(decay, [p1, p2])

index = 0
point_test = np.array([2, 1.25, 1.75, .25, .5])

points_test = np.array([[2, 1.25, 1.75, .25, .5], [2, 1.25, 1.75, .25, .5]])

alphaMax = [2, 0.14]

print(parameter_to_region(points_test, alphaMax=alphaMax))
print('This point should be in region 0!')
parameter_to_region(point_test, alphaMax=alphaMax)

pointbad = np.array([2, 1.25, 1.75, -.25, .5])
print(parameter_to_region(pointbad, alphaMax=alphaMax))

# following a random walk approach

bool_region = lambda x:  parameter_to_region(x, alphaMax=alphaMax) == index

point0 = np.array([2, 1.25, 1.75, .25, .5])
point1 = restricted_random_step(point0, bool_region)

many_points = brownian_motion_in_region(point0, bool_region, n_steps=10**4)
# if alphaMax is fixed as previously, all points show up
dsgrn_plot(many_points.T, alphaMax=alphaMax)
plt.savefig('fixed_alphaMax.png')
plt.show()

# otherwise, some of them get NEGATIVE!
dsgrn_plot(many_points.T, alphaMax=None)
plt.savefig('None_alphaMax.png')
plt.show()

print(99)
