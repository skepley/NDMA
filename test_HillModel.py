"""
Function and design testing for the HillModel class


    Output: output
    Other files required: none

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 4/6/20; Last revision: 4/6/20
"""

import numpy as np
import matplotlib.pyplot as plt
from hill_model import HillComponent, HillCoordinate, HillModel


class ToggleSwitch(HillModel):
    """An implementation of a saddle node problem for the toggle switch as a HillModel subclass"""

    def __init__(self, gamma, fixedParameters):
        """Toggle switch construction where each node has fixed ell, delta, theta, gamma and both Hill coefficients
        are free (but identical) variables."""

        parameter = [np.insert(p, 3, np.nan) for p in fixedParameters]
        interactionSigns = [[-1], [-1]]
        interactionTypes = [[1], [1]]
        interactionIndex = [[1], [0]]
        super().__init__(gamma, parameter, interactionSigns, interactionTypes,
                         interactionIndex)  # define HillModel for toggle switch

        # Define Hessian functions for HillCoordinates. This is temporary until the general formulas for the HillCoordinate
        # class is implemented.
        setattr(self.coordinates[0], 'dx2',
                lambda x, hillCoefficient: np.array(
                    [[0, 0], [0, self.coordinates[0].components[0].dx2(x[1], hillCoefficient)]]))
        setattr(self.coordinates[1], 'dx2',
                lambda x, hillCoefficient: np.array(
                    [[self.coordinates[1].components[0].dx2(x[0], hillCoefficient), 0], [0, 0]]))

        setattr(self.coordinates[0], 'dndx',
                lambda x, hillCoefficient: np.array([0, self.coordinates[0].components[0].dndx(x[1], hillCoefficient)]))
        setattr(self.coordinates[1], 'dndx',
                lambda x, hillCoefficient: np.array([self.coordinates[1].components[0].dndx(x[0], hillCoefficient), 0]))

    def __call__(self, x, n):
        """Overload the toggle switch to identify the Hill coefficients"""

        return super().__call__(x, np.array([n, n]))

    def dx(self, x, n):
        """Overload the toggle switch derivative to identify the Hill coefficients"""

        return super().dx(x, np.array([n, n]))

    def dn(self, x, n):
        """Overload the toggle switch derivative to identify the Hill coefficients"""

        Df_dn = super().dn(x, np.array([n, n]))  # Return Jacobian with respect to n = (n1, n2)
        return np.sum(Df_dn, 1)  # n1 = n2 = n so the derivative is tbe gradient vector of f with respect to n

    def plot_nullcline(self, n, nNodes=100, domainBounds=(10, 10)):
        """Plot the nullclines for the toggle switch at a given parameter"""

        equilibria = self.find_equilibria(n, 10)
        Xp = np.linspace(0, domainBounds[0], nNodes)
        Yp = np.linspace(0, domainBounds[1], nNodes)
        Z = np.zeros_like(Xp)

        N1 = self.coordinates[0](np.row_stack([Z, Yp]), np.array([n])) / self.coordinates[0].gamma  # f1 = 0 nullcline
        N2 = self.coordinates[1](np.row_stack([Xp, Z]), np.array([n])) / self.coordinates[1].gamma  # f2 = 0 nullcline

        if equilibria.ndim == 0:
            pass
        elif equilibria.ndim == 1:
            plt.scatter(equilibria[0], equilibria[1])
        else:
            plt.scatter(equilibria[0, :], equilibria[1, :])

        plt.plot(Xp, N2)
        plt.plot(N1, Yp)


# set some parameters to test using MATLAB toggle switch for ground truth
gamma = np.array([1, 1], dtype=float)
# p1 = np.array([1, 5, 3, 4.1], dtype=float)
# p2 = np.array([1, 6, 3, 4.1], dtype=float)
p1 = np.array([1, 5, 3], dtype=float)
p2 = np.array([1, 6, 3], dtype=float)
parm = np.array([4.1, 4.1])
x0 = np.array([4, 3])
n0 = 4.1
# test Hill model code
ts = ToggleSwitch(gamma, [p1, p2])
print(ts(x0, n0))

# test Hill model equilibrium finding
# eq = ts.find_equilibria(parm, 10)
# print(eq)
# added vectorized evaluation of Hill Models - DONE


# plot nullclines and equilibria
# plt.close('all')
# Xp = np.linspace(0, 10, 100)
# Yp = np.linspace(0, 10, 100)
# Z = np.zeros_like(Xp)
#
# N1 = ts.coordinates[0](np.row_stack([Z, Yp])) / gamma[0]  # f1 = 0 nullcline
# N2 = ts.coordinates[1](np.row_stack([Xp, Z])) / gamma[1]  # f2 = 0 nullcline
#
# plt.figure()
# plt.scatter(eq[0, :], eq[1, :])
# plt.plot(Xp, N2)
# plt.plot(N1, Yp)
