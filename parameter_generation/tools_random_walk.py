import numpy as np


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


def one_figure(point0, bool_region, ax, color, niter=1000):
    for j in range(niter):
        point0 = restricted_random_step(point0, bool_region)
        ax.plot(*point0, '.', color=color)
    return
