import numpy as np
import matplotlib.pyplot as plot
import scipy.optimize as opt
from tools_random_walk import *
from assess_distribution import check_convergence, convergence_rate
import time


def narrow_escape_region(x, f):
    x1 = x[0]
    if np.abs(x1) > 4.5:
        return 0
    x2 = x[1]
    if f(x1) > x2 and x2 > - f(x1):
        return 1
    return 0


def box(x):
    return np.all(np.abs(x) <= 1)


def scatter_plot(x, n_region):
    regions = np.array([n_region(x_i) for x_i in x.T])
    for i in range(2):
        plot.plot(x[0, regions == i], x[1, regions == i], '*')
    plot.show()


n_points = 1000
'''
bool_region = lambda x: box(x)

x0 = np.array([0, 0], ndmin=2)
points_0 = end_multiple_brownian_in_region(x0, bool_region, n_steps=10 ** 3, n_points=n_points)
points_1 = end_multiple_brownian_in_region(points_0, bool_region, n_steps=100)
plot.plot(points_0[:, 0], points_0[:, 1], 'g*')
plot.plot(points_1[:, 0], points_1[:, 1], 'r*')
plot.show()

print('Computing convergence', time.asctime(time.localtime()))
convergence = convergence_rate(points_0, points_1, points_0[:300, :])
print('Convergence = ', convergence, 'computed at ', time.asctime(time.localtime()))
'''

# X = (np.random.random(size=[2, 5000])-0.5) * 10
# scatter_plot(X, lambda x: narrow_escape_region(x, f_sym))
n_points = 1000
narrow = 0.03
f_sym = lambda x: (-x ** 4 + 20 * x ** 2) / 30 + narrow
f_asym = lambda x: (-x ** 4 + 20 * x ** 2 - 30 * x) / 30 + narrow
x0 = np.array([2, 0], ndmin=2)
bool_region = lambda x: narrow_escape_region(x, f_sym) == 1

print('Computing brownian motion', time.asctime(time.localtime()))

points_0_asym = end_multiple_brownian_in_region(x0, bool_region, step_size=0.1, n_steps=10 ** 3, n_points=n_points)
points_1_asym = end_multiple_brownian_in_region(points_0_asym, bool_region, n_steps=100)
x_plot = np.linspace(-4.5, 4.5, 200)
plot.plot(points_1_asym[:, 0], points_1_asym[:, 1], '*')
plot.plot(points_0_asym[:, 0], points_0_asym[:, 1], 'r*')
plot.plot(x_plot, f_sym(x_plot))
plot.plot(x_plot, -f_sym(x_plot))
plot.show()
print('Computing convergence', time.asctime(time.localtime()))
convergence_asym = convergence_rate(points_0_asym, points_1_asym)
print('Convergence_asym = ', convergence_asym, 'computed at ', time.asctime(time.localtime()))

x0 = np.array([[2, 0], [-2, 0]], ndmin=2)
points_0_sym = end_multiple_brownian_in_region(x0, bool_region, step_size=0.1, n_steps=10 ** 3, n_points=n_points / 2)
points_1_sym = end_multiple_brownian_in_region(points_0_sym, bool_region, n_steps=100)
x_plot = np.linspace(-4.5, 4.5, 200)
plot.plot(points_1_sym[:, 0], points_1_sym[:, 1], '*')
plot.plot(x_plot, f_sym(x_plot))
plot.plot(x_plot, -f_sym(x_plot))
plot.show()
print('Computing convergence', time.asctime(time.localtime()))
convergence_sym = convergence_rate(points_0_sym, points_1_sym)
print('Convergence_asym = ', convergence_sym, 'computed at ', time.asctime(time.localtime()))


print(99)
