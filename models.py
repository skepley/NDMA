"""
A separate file to store important HillModel subclasses for analysis or testing

    Other files required: hill_model

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 6/24/20; Last revision: 6/24/20
"""
from hill_model import *

class ToggleSwitch(HillModel):
    """Two-dimensional toggle switch construction inherited as a HillModel where each node has free (but identical)
    Hill coefficients and possibly some other parameters free."""

    def __init__(self, gamma, parameter):
        """Class constructor which has the following syntax:
        INPUTS:
            gamma - A vector in R^n of linear decay rates
            parameter - A length n list of K_i-by-3 parameter arrays with rows of the form (ell, delta, theta)"""

        parameter = [np.insert(parmList, 3, np.nan) for parmList in
                     parameter]  # append hillCoefficient as free parameter
        interactionSigns = [[-1], [-1]]
        interactionTypes = [[1], [1]]
        interactionIndex = [[1], [0]]
        super().__init__(gamma, parameter, interactionSigns, interactionTypes,
                         interactionIndex)  # define HillModel for toggle switch by inheritance
        self.hillIndexByCoordinate = self.variableIndexByCoordinate[1:] - np.array(range(1, 1 + self.dimension))

        # Define Hessian functions for HillCoordinates. This is temporary until the general formulas for the HillCoordinate
        # class is implemented.
        setattr(self.coordinates[0], 'dx2',
                lambda x, parm: np.array(
                    [[0, 0],
                     [0, self.coordinates[0].components[0].dx2(x[1], self.coordinates[0].parse_parameters(parm)[1])]]))
        setattr(self.coordinates[1], 'dx2',
                lambda x, parm: np.array(
                    [[self.coordinates[1].components[0].dx2(x[0], self.coordinates[1].parse_parameters(parm)[1]), 0],
                     [0, 0]]))

        setattr(self.coordinates[0], 'dndx',
                lambda x, parm: np.array(
                    [0, self.coordinates[0].components[0].dndx(x[1], self.coordinates[0].parse_parameters(parm)[1])]))
        setattr(self.coordinates[1], 'dndx',
                lambda x, parm: np.array(
                    [self.coordinates[1].components[0].dndx(x[0], self.coordinates[1].parse_parameters(parm)[1]), 0]))

    def parse_parameter(self, N, parameter):
        """Overload the parameter parsing for HillModels to identify all HillCoefficients as a single parameter, N. The
        parser Inserts copies of N into the appropriate Hill coefficient indices in the parameter vector."""

        return np.insert(parameter, self.hillIndexByCoordinate, N)

    def dn(self, x, N, parameter):
        """Overload the toggle switch derivative to identify the Hill coefficients which means summing over each
        gradient. This is a hacky fix and hopefully temporary. A correct implementation would just include a means to
        including the chain rule derivative of the hillCoefficient vector as a function of the form:
        Nvector = (N, N,...,N) in R^M."""

        Df_dHill = super().dn(x, N, parameter)  # Return Jacobian with respect to N = (N1, N2)  # OLD VERSION
        return np.sum(Df_dHill, 1)  # N1 = N2 = N so the derivative is tbe gradient vector of f with respect to N

    def plot_nullcline(self, n, parameter=np.array([]), nNodes=100, domainBounds=(10, 10)):
        """Plot the nullclines for the toggle switch at a given parameter"""

        equilibria = self.find_equilibria(25, n, parameter)
        Xp = np.linspace(0, domainBounds[0], nNodes)
        Yp = np.linspace(0, domainBounds[1], nNodes)
        Z = np.zeros_like(Xp)

        # unpack decay parameters separately
        gamma = np.array(list(map(lambda f_i, parm: f_i.parse_parameters(parm)[0], self.coordinates,
                                  self.unpack_variable_parameters(self.parse_parameter(n, parameter)))))
        N1 = (self(np.row_stack([Z, Yp]), n, parameter) / gamma[0])[0, :]  # f1 = 0 nullcline
        N2 = (self(np.row_stack([Xp, Z]), n, parameter) / gamma[1])[1, :]  # f2 = 0 nullcline

        if equilibria.ndim == 0:
            pass
        elif equilibria.ndim == 1:
            plt.scatter(equilibria[0], equilibria[1])
        else:
            plt.scatter(equilibria[0, :], equilibria[1, :])

        plt.plot(Xp, N2)
        plt.plot(N1, Yp)