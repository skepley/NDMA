"""
A separate file to store important HillModel subclasses for analysis or testing

    Other files required: hill_model

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 6/24/20; Last revision: 1/22/21
"""
from hill_model import *


class ToggleSwitch(HillModel):
    """Two-dimensional toggle switch construction inherited as a HillModel where each node has free (but identical)
    Hill coefficients, hill_1 = hill_2 = hill, and possibly some other parameters free. This is the simplest test case
    for analysis and also a canonical example of how to implement a HillModel in which some parameters are constrained
    by others."""

    def __init__(self, gamma, parameter, hill=np.nan):
        """Class constructor which has the following syntax:
        INPUTS:
            gamma - A vector in R^2 of linear decay rates or NaN if decays are variable parameters.
            parameter - A length-2 list of length-3 parameter vectors of the form (ell, delta, theta)
            hill - float for the (identified) Hill coefficient"""

        nComponent = 2  # number of total Hill components (edges in GRN)
        parameter = [np.insert(parmList, 3, hill) for parmList in
                     parameter]  # append hillCoefficient as free parameter
        interactionSigns = [[-1], [-1]]
        interactionTypes = [[1], [1]]
        interactionIndex = [[1], [0]]
        super().__init__(gamma, parameter, interactionSigns, interactionTypes,
                         interactionIndex)  # define HillModel for toggle switch by inheritance
        # insertion indices for HillCoefficients to expand the truncated parameter vector to a full parameter vector
        self.hillIndex = np.array(self.variableIndexByCoordinate[
                                  1:]) - 1  # indices of Hill coefficient parameters in the full parameter vector
        self.nonhillIndex = np.array([idx for idx in range(self.nVariableParameter) if
                                      idx not in self.hillIndex])  # indices of non Hill coefficient variable parameters in the full vector
        self.hillInsertionIndex = self.hillIndex - np.array(range(nComponent))
        self.nVariableParameter -= 1  # adjust variable parameter count by 1 to account for the identified Hill coefficients.

    def parse_parameter(self, *parameter):
        """Overload the generic parameter parsing for HillModels to identify all HillCoefficients as a single parameter, hill. The
        parser Inserts copies of hill into the appropriate Hill coefficient indices in the parameter vector.

        INPUT: parameter is an arbitrary number of inputs which must concatenate to the ordered parameter vector with hill as first component.
            Example: parameter = (hill, p) with p in R^{m-2}
        OUTPUT: A vector of the form:
            lambda = (gamma_1, ell_1, delta_1, theta_1, hill, gamma_2, ell_2, delta_2, theta_2, hill),
        where any fixed parameters are omitted."""
        parameterVector = ezcat(
            *parameter)  # concatenate input into a single vector. Its first component must be the common hill parameter for both coordinates
        hill, p = parameterVector[0], parameterVector[1:]
        return np.insert(p, self.hillInsertionIndex, hill)

    def diff(self, x, *parameter, diffIndex=None):
        """Overload the diff function to identify the Hill parameters"""

        fullDf = super().diff(x, *parameter)
        Dpf = np.zeros([self.dimension, self.nVariableParameter])  # initialize full derivative w.r.t. all parameters
        Dpf[:, 1:] = fullDf[:, self.nonhillIndex]  # insert derivatives of non-hill parameters
        Dpf[:, 0] = np.einsum('ij->i',
                              fullDf[:, self.hillIndex])  # insert sum of derivatives for identified hill parameters

        if diffIndex is None:
            return Dpf  # return the full vector of partials
        else:
            return np.squeeze(Dpf[:, np.array([diffIndex])])  # return only columns for the specified subset of partials

    def dxdiff(self, x, *parameter, diffIndex=None):
        """Overload the dxdiff function to identify the Hill parameters"""

        fullDf = super().dxdiff(x, *parameter)
        Dpf = np.zeros(
            2 * [self.dimension] + [self.nVariableParameter])  # initialize full derivative w.r.t. all parameters
        Dpf[:, :, 1:] = fullDf[:, :, self.nonhillIndex]  # insert derivatives of non-hill parameters
        Dpf[:, :, 0] = np.einsum('ijk->ij', fullDf[:, :,
                                            self.hillIndex])  # insert sum of derivatives for identified hill parameters

        if diffIndex is None:
            return Dpf  # return the full vector of partials
        else:
            return np.squeeze(
                Dpf[:, :, np.array([diffIndex])])  # return only columns for the specified subset of partials

    def diff2(self, x, *parameter, diffIndex=[None, None]):
        """Overload the diff2 function to identify the Hill parameters"""

        fullDf = super().diff2(x, *parameter)
        Dpf = np.zeros(
            [self.dimension] + 2 * [self.nVariableParameter])  # initialize full derivative w.r.t. all parameters
        Dpf[:, 1:, 1:] = fullDf[np.ix_(np.arange(self.dimension), self.nonhillIndex,
                                       self.nonhillIndex)]  # insert derivatives of non-hill parameters
        Dpf[:, 0, 0] = np.einsum('ijk->i', fullDf[np.ix_(np.arange(self.dimension), self.hillIndex,
                                                         self.hillIndex)])  # insert sum of derivatives for identified hill parameters

        if diffIndex[0] is None and diffIndex[1] is None:
            return Dpf  # return the full vector of partials
        else:
            return np.squeeze(
                Dpf[np.ix_(np.arange(self.dimension), diffIndex,
                           diffIndex)])  # return only slices for the specified subset of partials

    def dx2diff(self, x, *parameter, diffIndex=None):
        """Overload the dx2diff function to identify the Hill parameters"""

        fullDf = super().dx2diff(x, *parameter)
        Dpf = np.zeros(
            3 * [self.dimension] + [self.nVariableParameter])  # initialize full derivative w.r.t. all parameters
        Dpf[:, :, :, 1:] = fullDf[
            np.ix_(np.arange(self.dimension), np.arange(self.dimension), np.arange(self.dimension),
                   self.nonhillIndex)]  # insert derivatives of non-hill parameters
        Dpf[:, :, :, 0] = np.einsum('ijkl->ijk', fullDf[
            np.ix_(np.arange(self.dimension), np.arange(self.dimension), np.arange(self.dimension),
                   self.hillIndex)])  # insert sum of derivatives for identified hill parameters

        if diffIndex is None:
            return Dpf  # return the full vector of partials
        else:
            return np.squeeze(
                Dpf[np.ix_(np.arange(self.dimension), np.arange(self.dimension), np.arange(self.dimension),
                           diffIndex)])  # return only slices for the specified subset of partials

    def dxdiff2(self, x, *parameter, diffIndex=[None, None]):
        """Overload the dxdiff2 function to identify the Hill parameters"""

        fullDf = super().dxdiff2(x, *parameter)
        Dpf = np.zeros(
            2 * [self.dimension] + 2 * [self.nVariableParameter])  # initialize full derivative w.r.t. all parameters
        Dpf[:, :, 1:, 1:] = fullDf[
            np.ix_(np.arange(self.dimension), np.arange(self.dimension), self.nonhillIndex,
                   self.nonhillIndex)]  # insert derivatives of non-hill parameters
        Dpf[:, :, 0, 0] = np.einsum('ijkl->ij', fullDf[
            np.ix_(np.arange(self.dimension), np.arange(self.dimension), self.hillIndex,
                   self.hillIndex)])  # insert sum of derivatives for identified hill parameters

        if diffIndex[0] is None and diffIndex[1] is None:
            return Dpf  # return the full vector of partials
        else:
            return np.squeeze(
                Dpf[np.ix_(np.arange(self.dimension), np.arange(self.dimension), diffIndex,
                           diffIndex)])  # return only slices for the specified subset of partials

    def bootstrap_map(self, *parameter):
        """Return the bootstrap map for the toggle switch, Phi: R^4 ---> R^4 which is iterated to bound equilibrium enclosures"""

        fullParm = self.parse_parameter(
            *parameter)  # concatenate all parameters into a vector with hill coefficients sliced in
        P0, P1 = parameterByCoordinate = self.unpack_variable_parameters(
            fullParm)  # unpack variable parameters by component
        g0, p0 = self.coordinates[0].parse_parameters(P0)
        g1, p1 = self.coordinates[1].parse_parameters(P1)

        def H0(x):
            """Evaluate the function from R^2 to R defined by the first and 3rd components of Phi"""
            return (1 / g0) * self.coordinates[0].components[0](x[1], p0[0])

        def H1(x):
            """Evaluate the function from R^2 to R defined by the second and fourth components of Phi"""
            return (1 / g1) * self.coordinates[1].components[0](x[0], p1[0])

        def bootstrap(u):
            alpha, beta = np.split(u, 2)
            alphaNew = np.array([H0(beta), H1(beta)])
            betaNew = np.array([H0(alpha), H1(alpha)])
            return ezcat(alphaNew, betaNew)

        return bootstrap

    def bootstrap_enclosure(self, *parameter, tol=1e-13):
        """Return an enclosure of all equilibria for the toggle switch as a rectangle of one of the following forms:
    
        1. R = [a1, b1]x[a2,b2] and the returned array has rows of the form [ai, bi]. In this case the vector field has at least
        two equilibria at the coordinates, x1 = (min(ai), max(bi)) and x2 = (max(ai), min(bi)).
    
        2. R = [x1, x1]x[x2,x2] which is a degenerate rectangle returned as an array [x1,x2]. In this case the vector field has
        a unique equilibrium at the coordinates x = (x1, x2) which is always stable."""

        # get initial condition for Phi
        fullParm = self.parse_parameter(
            *parameter)  # concatenate all parameters into a vector with hill coefficients sliced in
        P0, P1 = parameterByCoordinate = self.unpack_variable_parameters(
            fullParm)  # unpack variable parameters by component
        g0, p0 = self.coordinates[0].parse_parameters(P0)
        g1, p1 = self.coordinates[1].parse_parameters(P1)
        H0 = self.coordinates[0].components[0]
        H1 = self.coordinates[1].components[0]
        x0Bounds = (1 / g0) * H0.image(p0[0])
        x1Bounds = (1 / g1) * H1.image(p1[0])
        u0 = np.array(list(zip(x0Bounds, x1Bounds))).flatten()  # zip initial bounds

        # iterate the bootstrap map to obtain an enclosure
        Phi = self.bootstrap_map(*parameter)
        maxIter = 100
        u = u0
        notConverged = True
        nIter = 0
        while nIter < maxIter and notConverged:
            uNew = Phi(u)
            tol_loc = np.linalg.norm(uNew - u)
            if nIter>3:
                uveryOld = uOld
            uOld = u
            notConverged = np.linalg.norm(uNew - u) > tol
            u = uNew
            nIter += 1

        if nIter == maxIter:
            print('Uh oh. The bootstrap map failed to converge')
            return u0, None

        # unzip i.e. (alpha, beta) ---> (a1, b1)x(a2, b2)
        alpha, beta = np.split(u, 2)
        intervalFactors = np.array(list(zip(alpha, beta)))
        return u0, np.unique(np.round(intervalFactors, 13), axis=1).squeeze()  # remove degenerate intervals and return

    def plot_nullcline(self, *parameter, nNodes=100, domainBounds=((0, 10), (0, 10))):
        """Plot the nullclines for the toggle switch at a given parameter"""

        X1, X2 = np.meshgrid(np.linspace(*domainBounds[0], nNodes), np.linspace(*domainBounds[1], nNodes))
        flattenNodes = np.array([np.ravel(X1), np.ravel(X2)])
        p1, p2 = self.unpack_variable_parameters(self.parse_parameter(*parameter))
        Z1 = np.reshape(self.coordinates[0](flattenNodes, p1), 2 * [nNodes])
        Z2 = np.reshape(self.coordinates[1](flattenNodes, p2), 2 * [nNodes])
        cs1 = plt.contour(X1, X2, Z1, [0], colors='g', alpha=0)
        cs2 = plt.contour(X1, X2, Z2, [0], colors='r', alpha=0)
        x1 = cs1.collections[0].get_paths()[0].vertices[:,0]
        y1 = cs1.collections[0].get_paths()[0].vertices[:,1]
        x2 = cs2.collections[0].get_paths()[0].vertices[:,0]
        y2 = cs2.collections[0].get_paths()[0].vertices[:,1]
        return x1, y1, x2, y2


    def find_equilibria(self, gridDensity, *parameter, uniqueRootDigits=5, bootstrap=True):
        """Overloading the HillModel find equilibrium method to use the bootstrap approach for the ToggleSwitch. The output
         is an array whose rows are coordinates of found equilibria for the ToggleSwitch."""

        if bootstrap:
            eqBound = self.bootstrap_enclosure(*parameter)[1]
            if is_vector(eqBound):  # only a single equilibrium given by the degenerate rectangle
                return eqBound
            else:
                return super().find_equilibria(gridDensity, *parameter, uniqueRootDigits=uniqueRootDigits, eqBound=eqBound)

        else:  # Use the old version inherited from the HillModel class. This should only be used to troubleshoot
            return super().find_equilibria(gridDensity, *parameter, uniqueRootDigits=uniqueRootDigits)

    def plot_equilibria(self, *parameter, nInitData=10):
        """Find equilibria at given parameter and add to current plot"""

        equilibria = self.find_equilibria(nInitData, *parameter)
        if equilibria.ndim == 0:
            warnings.warn('No equilibria found')
            return
        elif equilibria.ndim == 1:
            plt.scatter(equilibria[0], equilibria[1], color='b')
        else:
            plt.scatter(equilibria[0, :], equilibria[1, :], color='b')

    def dsgrn_region(self, *parameter):
        """Return a dsgrn parameter region for the toggle switch as an integer in {1,...,9}

        INPUT: parameter = (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_1 theta_2)
        with fixed parameters omitted."""

        print('deprecated. Use the dsgrn_coordinates functionality in the toggle_switch_heat_functionalities')

        def factor_slice(gamma, ell, delta, theta):
            T = gamma * theta
            if T <= ell:
                return 0
            elif ell < T <= ell + delta:
                return 1
            else:
                return 2

        fullParameterVector = self.parse_parameter(*parameter)
        DSGRNParameter = fullParameterVector[[0, 1, 2, 3, 5, 6, 7, 8]]  # remove Hill coefficients from parameter vector
        return 3 * (factor_slice(*DSGRNParameter[[0, 1, 2, 7]]) + 1) - factor_slice(*DSGRNParameter[[4, 5, 6, 3]])

