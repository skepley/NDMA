import numpy as np
import matplotlib.pyplot as plot
import scipy.optimize as opt
from tools_random_walk import *

def simple_region(x, *f):
    x1 = x[0]
    x2 = x[1]
    if x1 < 0 or x2 < 0:
        return np.nan
    for i in range(len(f)):
        if x2 < f[i](x1):
            return i
    return len(f)


f1 = lambda x: (x < 1) * (0.5 * x) + (x >= 1) * 0.5 * np.sqrt(np.abs(x))
f2 = lambda x: (x < 1) * (1.5 * x) + (x >= 1) * 1.5 * x ** 2

point0 = np.ones(2)
y_point0 = [0.25, 1, 2]
color_options = ['b', 'r', 'g']

fig, ax = plot.subplots()
x_plot = 0.01 * np.arange(0, 300)
ax.plot(x_plot, f1(x_plot))
ax.plot(x_plot, f2(x_plot))
ax.axis('equal')
ax.set(xlim=(0, 5), ylim=(0, 5))

for region_index in range(3):

    bool_region = lambda x: simple_region(x, f1, f2) == region_index

    point0[1] = y_point0[region_index]

    for i in range(10):
        one_figure(point0, bool_region, ax, color_options[region_index])

plot.show()

