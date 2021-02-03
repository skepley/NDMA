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
            np.ix_(np.arange(self.dimension), np.arange(self.dimension), np.arange(self.dimension), self.nonhillIndex)]  # insert derivatives of non-hill parameters
        Dpf[:, :, :, 0] = np.einsum('ijkl->ijk', fullDf[np.ix_(np.arange(self.dimension), np.arange(self.dimension), np.arange(self.dimension), self.hillIndex)])  # insert sum of derivatives for identified hill parameters

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
            np.ix_(np.arange(self.dimension), np.arange(self.dimension), self.nonhillIndex, self.nonhillIndex)]  # insert derivatives of non-hill parameters
        Dpf[:, :, 0, 0] = np.einsum('ijkl->ij', fullDf[np.ix_(np.arange(self.dimension), np.arange(self.dimension), self.hillIndex, self.hillIndex)])  # insert sum of derivatives for identified hill parameters

        if diffIndex[0] is None and diffIndex[1] is None:
            return Dpf  # return the full vector of partials
        else:
            return np.squeeze(
                Dpf[np.ix_(np.arange(self.dimension), np.arange(self.dimension), diffIndex,
                           diffIndex)])  # return only slices for the specified subset of partials

    def plot_nullcline(self, *parameter, nNodes=100, domainBounds=((0, 10), (0, 10))):
        """Plot the nullclines for the toggle switch at a given parameter"""

        X1, X2 = np.meshgrid(np.linspace(*domainBounds[0], nNodes), np.linspace(*domainBounds[1], nNodes))
        flattenNodes = np.array([np.ravel(X1), np.ravel(X2)])
        p1, p2 = self.unpack_variable_parameters(self.parse_parameter(*parameter))
        Z1 = np.reshape(self.coordinates[0](flattenNodes, p1), 2 * [nNodes])
        Z2 = np.reshape(self.coordinates[1](flattenNodes, p2), 2 * [nNodes])
        plt.contour(X1, X2, Z1, [0], colors='g')
        plt.contour(X1, X2, Z2, [0], colors='r')

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


class ToggleSwitchPlus(HillModel):
    """Two-dimensional toggle switch with one gene self activating/repressing inherited as a HillModel. Each edge has
    free (but identical) Hill coefficients, hill_1 = hill_2 = hill_3 = hill, and possibly some other parameters free. This is the simplest test case
    for analysis and also a canonical example of how to implement a HillModel in which some parameters are constrained
    by others."""

    def __init__(self, gamma, parameter, selfEdgeSigns):
        """Class constructor which has the following syntax:
        INPUTS:
            gamma - A vector in R^2 of linear decay rates or NaN if decays are variable parameters.
            parameter - A length-2 list of parameter arrays of size k-by-3 for k in {1,2}. Each row of a parameter array
                has the form (ell, delta, theta).
            selfEdgeSigns - A length-2 list of interaction signs for the self edges which can be in {0, 1, -1}."""

        # append hillCoefficient as free parameter to each Hill Component
        def insert_nan(parameterArray):
            """Insert NaN values for Hill coefficients"""

            if is_vector(parameterArray):
                return np.insert(parameterArray, 3, np.nan)  # Add Hill coefficient placeholder to a single edge.
            else:
                return np.array(
                    [np.insert(row, 3, np.nan) for row in parameterArray])  # Add Hill coefficient to each row of array

        parameter = [insert_nan(parmArray) for parmArray in parameter]

        # Add interactions for the basic toggle switch
        interactionSigns = [[-1], [-1]]
        interactionTypes = [[1], [1]]
        interactionIndex = [[1], [0]]

        for j in range(2):  # add self interactions
            if selfEdgeSigns[j] != 0:
                interactionSigns[j].insert(j, selfEdgeSigns[j])
                interactionTypes[j].insert(j,
                                           1)  # This assumes only a product interaction function for both coordinates
                interactionIndex[j].insert(j, j)

        super().__init__(gamma, parameter, interactionSigns, interactionTypes,
                         interactionIndex)  # define HillModel for toggle switch by inheritance
        self.nComponent = np.sum(
            [self.coordinates[j].nComponent for j in range(2)])  # count total number of Hill components
        self.hillIndex = ezcat(
            *[self.variableIndexByCoordinate[j] + self.coordinates[j].variableIndexByComponent[1:] - 1 for j in
              range(2)])
        # insertion indices for HillCoefficients to expand the truncated parameter vector to a full parameter vector
        self.nonhillIndex = np.array([idx for idx in range(self.nVariableParameter) if
                                      idx not in self.hillIndex])  # indices of non Hill coefficient variable parameters in the full vector
        self.hillInsertionIndex = self.hillIndex - np.arange(self.nComponent)
        self.nVariableParameter -= (
                self.nComponent - 1)  # adjust variable parameter count to account for the identified Hill coefficients.

    def parse_parameter(self, *parameter):
        """Overload the generic parameter parsing for HillModels to identify all HillCoefficients as a single parameter, hill. The
        parser Inserts copies of hill into the appropriate Hill coefficient indices in the parameter vector.

        INPUT: parameter is an arbitrary number of inputs which must concatenate to the ordered parameter vector with hill as first component.
            Example: parameter = (hill, p) with p in R^{m-2}
        OUTPUT: An ordered parameter vector of the form:
            lambda = (gamma_1, ell_11, ... delta_1, theta_11, hill, gamma_12,...,hill, gamma_2, ell_21,...,hill),
        where any fixed parameters or parameters for missing self edges are omitted."""
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

    def plot_nullcline(self, *parameter, nNodes=100, domainBounds=((0, 10), (0, 10))):
        """Plot the nullclines for the toggle switch at a given parameter"""

        X1, X2 = np.meshgrid(np.linspace(*domainBounds[0], nNodes), np.linspace(*domainBounds[1], nNodes))
        flattenNodes = np.array([np.ravel(X1), np.ravel(X2)])
        p1, p2 = self.unpack_variable_parameters(self.parse_parameter(*parameter))
        Z1 = np.reshape(self.coordinates[0](flattenNodes, p1), 2 * [nNodes])
        Z2 = np.reshape(self.coordinates[1](flattenNodes, p2), 2 * [nNodes])
        plt.contour(X1, X2, Z1, [0], colors='g')
        plt.contour(X1, X2, Z2, [0], colors='r')

    def plot_equilibria(self, *parameter, nInitData=10):
        """Find equilibria at given parameter and add to current plot"""

        equilibria = self.find_equilibria(nInitData, *parameter)
        if equilibria.ndim == 0:
            warnings.warn('No equilibria found')
            return
        elif equilibria.ndim == 1:
            plt.scatter(equilibria[0], equilibria[1])
        else:
            plt.scatter(equilibria[0, :], equilibria[1, :])


class Network12(HillModel):
    """The best performing consistent 3-node network for producing robust hysteresis. Each edge has
    free (but identical) Hill coefficients, hill_1 = hill_2 = hill_3 = hill, and possibly some other parameters free."""

    def __init__(self, gamma, parameter):
        """Class constructor which has the following syntax:
        INPUTS:
            gamma - A vector in R^3 of linear decay rates or NaN if decays are variable parameters.
            parameter - A length-3 list of parameter arrays of size k_i-by-3 for k_i in {1,2,3}. Each row of a parameter array
                has the form (ell, delta, theta) and corresponds to an in-edge for node X_i."""

        # append hillCoefficient as free parameter to each Hill Component
        def insert_nan(parameterArray):
            """Insert NaN values for Hill coefficients"""

            if is_vector(parameterArray):
                return np.insert(parameterArray, 3, np.nan)  # Add Hill coefficient placeholder to a single edge.
            else:
                return np.array(
                    [np.insert(row, 3, np.nan) for row in parameterArray])  # Add Hill coefficient to each row of array

        parameter = [insert_nan(parmArray) for parmArray in parameter]

        # Add interactions for the basic toggle switch
        interactionSigns = [[1, 1, 1], [1, 1], [1]]  # all interactions are activation
        interactionTypes = [[3], [2], [1]]  # all interactions are single summand
        interactionIndex = [[0, 1, 2], [0, 2], [0]]

        super().__init__(gamma, parameter, interactionSigns, interactionTypes,
                         interactionIndex)  # define HillModel for toggle switch by inheritance

        self.nComponent = np.sum(
            [self.coordinates[j].nComponent for j in range(self.dimension)])  # count total number of Hill components
        self.hillIndex = ezcat(
            *[self.variableIndexByCoordinate[j] + self.coordinates[j].variableIndexByComponent[1:] - 1 for j in
              range(self.dimension)])
        # insertion indices for HillCoefficients to expand the truncated parameter vector to a full parameter vector
        self.nonhillIndex = np.array([idx for idx in range(self.nVariableParameter) if
                                      idx not in self.hillIndex])  # indices of non Hill coefficient variable parameters in the full vector
        self.hillInsertionIndex = self.hillIndex - np.arange(self.nComponent)
        self.nVariableParameter -= (
                self.nComponent - 1)  # adjust variable parameter count to account for the identified Hill coefficients.

    def parse_parameter(self, *parameter):
        """Overload the generic parameter parsing for HillModels to identify all HillCoefficients as a single parameter, hill. The
        parser Inserts copies of hill into the appropriate Hill coefficient indices in the parameter vector.

        INPUT: parameter is an arbitrary number of inputs which must concatenate to the ordered parameter vector with hill as first component.
            Example: parameter = (hill, p) with p in R^{m-2}
        OUTPUT: An ordered parameter vector of the form:
            lambda = (gamma_1, ell_11, ... delta_1, theta_11, hill, gamma_12,...,hill, gamma_2, ell_21,...,hill),
        where any fixed parameters or parameters for missing self edges are omitted."""
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
