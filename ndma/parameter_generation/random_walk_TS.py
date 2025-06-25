import inspect

from ndma.basic_models.TS_model import ToggleSwitch
from ndma.parameter_generation.DSGRN_tools import *
from tools_random_walk import *
from assess_distribution import check_convergence, convergence_rate
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from toggle_switch_heat_functionalities import *



decay = np.array([1, np.nan], dtype=float)
p1 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, 1], dtype=float)  # (ell_2, delta_2, theta_2)
TS = ToggleSwitch(decay, [p1, p2])

index = 0
point_test = np.array([.25, .5, 2, 1.25, .5]) #  ell1, delta1, gamma2, ell2, delta2

points_test = np.array([[.25, .5, 2, 1.25, .5],[.25, .5, 2, 1.25, .5]])

alphaMax = np.array([2, 0.14])

try:
    parameter_to_region(points_test, alphaMax=alphaMax)
except ValueError:
    pass

alphaMax = np.array([2, 1.4])

print(parameter_to_region(points_test, alphaMax=alphaMax))
print('This point should be in region 0!')
parameter_to_region(point_test, alphaMax=alphaMax)

pointbad = np.array([2, 1.25, 1.75, -.25, .5])
print(parameter_to_region(pointbad, alphaMax=alphaMax))

# following a random walk approach

bool_region = lambda x:  parameter_to_region(x, alphaMax=alphaMax) == index

point0 = np.array([.25, .5, 2, 1.25, .5],  ndmin=2)
point1 = restricted_random_step(point0, bool_region)

many_points = brownian_motion_in_region(point0, bool_region, n_steps=10**4)
# if alphaMax is fixed as previously, all points show up
dsgrn_plot(many_points.T, alphaMax=alphaMax)
plt.savefig('fixed_alphaMax.png')
plt.show()

# check_convergence(points_0, points_1, selected_points=None, threshold=10**-3):

points_0 = end_multiple_brownian_in_region(point0, bool_region, n_steps=10**2, n_points=500)
points_1 = end_multiple_brownian_in_region(points_0, bool_region, n_steps=10)
dsgrn_plot(points_1.T, alphaMax=alphaMax)
plt.show()
print('Computing convergence', time.asctime(time.localtime()))
conv_10_2 = convergence_rate(points_0, points_1)
print('Convergence after 10**2 = ', conv_10_2)

conv_10_2 = check_convergence(points_0, points_1)

print('Convergence after 10**2 = ', conv_10_2)

points_0 = end_multiple_brownian_in_region(point0, bool_region, n_steps=10**3, n_points=500)
points_1 = end_multiple_brownian_in_region(points_0, bool_region, n_steps=10)
dsgrn_plot(points_1.T, alphaMax=alphaMax)
plt.show()
print('Computing convergence', time.asctime(time.localtime()))
conv_10_2 = convergence_rate(points_0, points_1)
print('Convergence after 10**2 = ', conv_10_2)


points_0 = end_multiple_brownian_in_region(point0, bool_region, n_steps=10**4, n_points=500)
points_1 = end_multiple_brownian_in_region(points_0, bool_region, n_steps=10**2)
dsgrn_plot(points_1.T, alphaMax=alphaMax)
plt.show()
print('Computing convergence', time.asctime(time.localtime()))
conv_10_4 = convergence_rate(points_0, points_1)
print('Convergence after 10**4 = ', conv_10_4)

conv_10_4 = check_convergence(points_0, points_1)

print('Convergence after 10**4 = ', conv_10_4)

