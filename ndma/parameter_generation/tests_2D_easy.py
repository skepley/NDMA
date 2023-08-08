import numpy as np
import matplotlib.pyplot as plot
import scipy.optimize as opt


def simple_region(x, f):
    x1 = x[:, 0]
    x2 = x[:, 1]
    assigned_region = np.zeros_like(x1)
    assigned_region[f(x1, x2)] = 1
    return assigned_region.astype(int)


def many_regions(x, *f):
    x1 = x[:, 0]
    x2 = x[:, 1]
    assigned_region = np.zeros_like(x1)
    for i in range(len(f)):
        assigned_region[f[i](x1, x2)] = i + 1
    return assigned_region.astype(int)


def scatter_plot(x, regions):
    for i in range(np.max(regions) + 1):
        plot.plot(x[regions == i, 0], x[regions == i, 1], '*')
    plot.show()


def coverage_ratio(regions):
    max_n_points = 0
    min_n_points = len(regions)
    for i in range(np.max(regions) + 1):
        points_in_region_i = np.sum(regions == i)
        max_n_points = np.maximum(max_n_points, points_in_region_i)
        min_n_points = np.minimum(min_n_points, points_in_region_i)
    return min_n_points / max_n_points


def distribution(par, size, dim=2):
    mean = par[:dim]
    cov = np.reshape(par[dim:], [dim, dim])
    return np.abs(np.random.multivariate_normal(mean, np.matmul(cov.transpose(), cov), size=size))


def return_loss_function(dim=2, *f):
    def loss_function(par):
        points2D = distribution(par, 100000, dim)
        return coverage_ratio(many_regions(points2D, *f))

    return loss_function


def numerical_diff(f, x):
    # f is 1 dimensional, while x is an n-array, the jacobian is then an n-array
    diff = np.zeros_like(x)
    h = diff
    epsilon = 10**-4
    for i in range(len(x)):
        h[i] = epsilon
        diff[i] = (f(x+h) - f(x))/epsilon
    return diff


# definitions
f_diag = lambda x, y: y < x
n = 2000
f_square = lambda x, y: y ** 2 < x
mean = np.array([3,2])
cov = 3 * np.identity(2)
par = np.append(mean, np.reshape(cov, -1))

# tests
points2D = np.abs(np.random.normal(scale=3, size=[2, n]))
simple_region(points2D, f_diag)

scatter_plot(points2D, simple_region(points2D, f_diag))

points2D = np.abs(np.random.multivariate_normal(mean, cov, size=n))
scatter_plot(points2D, simple_region(points2D, f_square))

points2D = np.abs(np.random.multivariate_normal(mean, cov, size=n))
scatter_plot(points2D, many_regions(points2D, f_diag, f_square))

coverage_ratio(many_regions(points2D, f_diag, f_square))

loss = return_loss_function(2, f_diag, f_square)
loss(par)

res = opt.minimize(loss, par, method='BFGS', jac=lambda x: numerical_diff(loss, x), options={'disp': True})
# covariance gets not positive-definite
points2D = distribution(res.x, 5000, 2)
scatter_plot(points2D, many_regions(points2D, f_diag, f_square))
loss(res.x)

print(99)