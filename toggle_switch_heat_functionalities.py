from hill_model import *

def sampler():
    """Sample parameters for the toggle switch other than the hill coefficient. This is a nondimensionalized sampler
     so it assumes that theta_1 = theta_2 = gamma_1 = 1 and returns a vector in R^5 of the form:
     (ell_1, delta_1, gamma_1, ell_2, delta_2)."""

    # pick ell_2, delta_2
    ell_1 = 1.5 * np.random.random_sample()  # sample in (0, 1.5)
    delta_1 = 1.5 * np.random.random_sample()  # sample in (0, 1.5)

    # pick gamma_2
    gammaScale = 1 + 9 * np.random.random_sample()  # gamma scale in [1, 10]
    g = lambda x: x if np.random.randint(2) else 1 / x
    gamma_2 = g(gammaScale)  # sample in (.1, 10) to preserve symmetry between genes

    # pick ell_2, delta_2
    ellByGamma_2 = 1.5 * np.random.random_sample()  # sample in (0, 1.5)
    deltaByGamma_2 = 1.5 * np.random.random_sample()  # sample in (0, 1.5)
    ell_2 = ellByGamma_2 * gamma_2
    delta_2 = deltaByGamma_2 * gamma_2
    return ezcat(ell_1, delta_1, gamma_2, ell_2, delta_2)

def heat_coordinate(alpha, beta, alphaMax):
    """Returns the DSGRN heat map coordinates For a parameter of the form (alpha, beta) where
    alpha = ell / gamma and beta = (ell + delta) / gamma"""

    if beta < 1:  # (ell + delta)/gamma < theta
        x = beta

    elif alpha > 1:  # theta < ell/gamma
        x = 2 + (alpha - 1) / (alphaMax - 1)

    else:  # ell/gamma < theta < (ell + delta)/gamma
        x = 1 + (1 - alpha) / (beta - alpha)
    return x

def heat_coordinates(alpha1, beta1, alpha2, beta2, alphaMax):
    """ take vector of coordinates and return vectors of x-coordinates and y-ccordinates"""
    x = np.array(
        [heat_coordinates(alpha1[j], beta1[j], alphaMax) for j in range(len(alpha1))])
    y = np.array(
        [heat_coordinates(alpha2[j], beta2[j], alphaMax) for j in range(len(alpha2))])
    return x, y


def grid_lines():
    """Add grid lines to a dsgrn coordinate plot"""
    for i in range(1, 3):
        plt.plot([i, i], [0, 3], 'k')
        plt.plot([0, 3], [i, i], 'k')
    return


def dsgrn_plot(parameterArray, alphaMax=None):
    """A scatter plot in DSGRN coordinates of a M-by-5 dimensional array. These are nondimensional parameters with rows
    of the form: (ell_1, delta_1, gamma_2, ell_2, delta_2)."""

    alpha1 = parameterArray[:, 0]
    beta1 = parameterArray[:, 0] + parameterArray[:, 1]
    alpha2 = parameterArray[:, 3] / parameterArray[:, 2]
    beta2 = (parameterArray[:, 3] + parameterArray[:, 4]) / parameterArray[:, 2]

    if alphaMax is None:
        alphaMax = np.max(np.max(alpha1), np.max(alpha2))

    x, y =  heat_coordinates(alpha1, beta1, alpha2, beta2, alphaMax)

    plt.scatter(x, y, marker='o', c='k', s=2)
    grid_lines()



