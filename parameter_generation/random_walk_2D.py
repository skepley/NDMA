import numpy as np
import matplotlib.pyplot as plot
import scipy.optimize as opt


def simple_region(x, *f):
    x1 = x[0]
    x2 = x[1]
    if x1 < 0 or x2 < 0:
        return np.nan
    for i in range(len(f)):
        if x2 < f[i](x1):
            return i

    return len(f)


def random_step(x, step_size=0.1):
    h = np.random.normal(0, step_size, len(x))
    return x + h


def random_change(step_size, x):
    h = np.random.normal(0, step_size, len(x))

    # adding a stronger likelyhood of "coming back"
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


def one_figure(point0, ax, color, niter=1000):
    for j in range(niter):
        point1 = restricted_random_step(point0, bool_region)
        ax.plot(np.array([point0[0], point1[0]]),
                  np.array([point0[1], point1[1]]), '.', color=color)
        point0 = point1

    return


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
        one_figure(point0, ax, color_options[region_index])

plot.show()

