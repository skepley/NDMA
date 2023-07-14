import numpy as np
import matplotlib.pyplot as plot
from tools_random_walk import *


def regions3D(x, f_1, f_2, f_3):
    if any(x < 0):
        return np.nan
    if f_2(*x) < f_3(*x):
        return 1
    elif f_1(*x) < f_3(*x):
        return 0
    elif f_3(*x) < f_1(*x):
        return 2
    return np.nan


"""
def random_change(step_size, x):
    h = np.random.normal(0, step_size, len(x))
    # adding a stronger likelihood of "coming back"
    if np.linalg.norm(x) > 1 and np.log2(np.linalg.norm(x)) > abs(np.random.normal(0, 6)):
        h = - np.abs(h)
    return h


def restricted_random_step(x, bool_region, step_size=0.1):
    if bool_region(x) is False:
        ValueError()
    h = random_change(step_size, x)
    iter = 0
    while iter < 10 and bool_region(x + h) is False:
        h = random_change(step_size, x)
        iter = iter + 1
        if iter == 10:
            if step_size > 10 ** -6:
                iter = 0
                step_size = 0.1 * step_size
            else:
                AttributeError()
    return x + h


def one_figure(point0, bool_region, ax, color, niter=10, step_size=0.2):
    for j in range(niter):
        point0 = restricted_random_step(point0, bool_region, step_size=step_size)
        ax.plot(point0[0],
                  point0[1], point0[2], '.', color=color)
    return
"""

f1 = lambda x, y, z: x
f2 = lambda x, y, z: x + y
f3 = lambda x, y, z: z
# this generate 3 regions (always f1<f2)
# f1<f3<f2, f2<f3, f3<f1

point0 = np.ones(3)
y_point0 = [1.5, 4, 0.5]
color_options = ['b', 'r', 'g']

regions3D(point0, f1, f2, f3)

# create x,y
xx, yy = np.meshgrid(range(4), range(4))

ax = plot.axes(projection='3d')
# the two planes dividing the space
ax.plot_surface(xx, yy, xx, alpha=0.4)
ax.plot_surface(xx, yy, xx + yy, alpha=0.4)

for region_index in range(3):
    bool_region = lambda x: regions3D(x, f1, f2, f3) == region_index
    point0[2] = y_point0[region_index]
    for i in range(10):
        one_figure(point0, bool_region, ax, color_options[region_index], niter=500)

plot.show()

