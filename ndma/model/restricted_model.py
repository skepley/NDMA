"""
This class restricts the Hill model class to the case in which all Hill coefficients are equal.

This child class overwrites all functionalities of the Hill class
"""
import numpy as np

from ndma.activation import HillActivation
from ndma.model.model import Model, ezcat


class RestrictedHillModel(Model):
    """
    This subclass of the Hill model class automatically sets all Hill Coefficients to be equal, thus decreasing the  parameter space
    It can also be used as template for future applications were other parameters are set to be equal
    """

    def __init__(self, gamma, parameter, productionSign, productionType, productionIndex):
        """Class constructor which has the following syntax:
        INPUTS:
            gamma - A vector in R^n of linear decay rates
            parameter - A length n list of K_i-by-3 parameter arrays
                    Note: If K_i = 1 then productionSign[i] should be a vector, not a matrix i.e. it should have shape
                    (3,) as opposed to (1,3). If the latter case then the result will be squeezed since otherwise HillCoordinate
                    will throw an exception during construction of that coordinate.
            productionSign - A length n list of lists in F_2^{K_i}
            productionType - A length n list of length q_i lists describing an integer partitions of K_i
            productionIndex - A length n list whose i^th element is a length K_i list of global indices for the nonlinear
                interactions for node i. These are specified in any order as long as it is the same order used for productionSign
                and the rows of parameter. IMPORTANT: The exception to this occurs if node i has a self edge. In this case i must appear as the first
                index.
"""
        parameter_with_Hill = list(map(lambda parmArray: np.concatenate([parmArray, np.array([np.shape(parmArray)[0] * [
            np.nan]]).transpose()], axis=1), parameter))
        super().__init__(gamma, parameter_with_Hill, productionSign, productionType,
                         productionIndex, activationFunction=HillActivation)

        # # Alterations for identifying all Hill coefficients.
        self.hillIndex = self.hill_coefficient_idx()  # indices of Hill coefficient parameters in the full parameter
        # vector
        self.nonHillIndex = np.array([idx for idx in range(self.nParameter) if
                                      idx not in self.hillIndex])  # indices of non Hill coefficient variable parameters in the full vector
        n_terms = sum([len(productionSign[i]) for i in range(len(gamma))])
        self.hillInsertionIndex = self.hillIndex - np.array(range(n_terms))
        self.nReducedParameter = self.nParameter - (n_terms - 1)
        # adjust variable parameter count by the number of Hill coefficients detected to account for the
        # identified Hill coefficients.

    def hill_coefficient_idx(self):
        """Compute and return the variable parameter indices which correspond to Hill coefficient parameters"""

        def hill_coefficient_by_coordinate(idx):
            nParameterByComponent = self.coordinates[idx].nParameterByProductionIndex
            lastParameterIdx = self.parameterIndexByCoordinate[idx][-1]  # identify the last Hill coefficient and
            # subtract
            # off parameter indices by component. This way we do not need to consider whether gamma is fixed or
            # variable.
            return list(np.cumsum([lastParameterIdx] + [-pCount for pCount in nParameterByComponent[:0:-1]]))

        hillCoefficientIdx = []
        for i in range(self.dimension):
            hillCoefficientIdx += hill_coefficient_by_coordinate(i)

        return np.array(sorted(hillCoefficientIdx))

    def parse_parameter(self, *parameter):
        """Overload the generic parameter parsing for HillModels to identify all HillCoefficients as a single parameter, hill. The
        parser Inserts copies of hill into the appropriate Hill coefficient indices in the parameter vector.

        INPUT: parameter is an arbitrary number of inputs which must concatenate to the ordered parameter vector with hill as first component.
            Example: parameter = (hill, p) with p in R^{M-1}
        OUTPUT: A vector of size M+n_hill where the value of hill has been inserted into all HillCoefficient parameter locations."""
        parameterVector = ezcat(
            *parameter)  # concatenate input into a single vector. Its first component must be the common hill parameter for both coordinates
        hill, p = parameterVector[0], parameterVector[1:]
        return np.insert(p, self.hillInsertionIndex, hill)

    def unpack_parameter(self, parameter):
        """Unpack a parameter vector for the HillModel into disjoint parameter slices for each distinct coordinate"""
        parameter_all = self.parse_parameter(*parameter)
        return [parameter_all[idx] for idx in self.parameterIndexByCoordinate]

    def diff(self, x, *parameter, diffIndex=None):
        """Overload the diff function to identify the Hill parameters"""

        fullDf = super().diff(x, *parameter)
        Dpf = np.zeros(
            [self.dimension, self.nReducedParameter])  # initialize full derivative with respect to all parameters
        Dpf[:, 1:] = fullDf[:, self.nonHillIndex]  # insert derivatives of non-hill parameters
        Dpf[:, 0] = np.einsum('ij->i',
                              fullDf[:, self.hillIndex])  # insert sum of all derivatives for Hill coefficient
        # parameters
        if diffIndex is None:
            return Dpf  # return the full vector of partials
        else:
            return np.squeeze(Dpf[:, np.array([diffIndex])])  # return only columns for the specified subset of partials

    def dxdiff(self, x, *parameter, diffIndex=None):
        """Overload the dxdiff function to identify the Hill parameters"""

        fullDf = super().dxdiff(x, *parameter)
        Dpf = np.zeros(
            2 * [self.dimension] + [
                self.nReducedParameter])  # initialize full derivative with respect to all parameters
        Dpf[:, :, 1:] = fullDf[:, :, self.nonHillIndex]  # insert derivatives of non-hill parameters
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
            [self.dimension] + 2 * [
                self.nReducedParameter])  # initialize full derivative with respect to all parameters
        Dpf[:, 1:, 1:] = fullDf[np.ix_(np.arange(self.dimension), self.nonHillIndex,
                                       self.nonHillIndex)]  # insert derivatives of non-hill parameters
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
            3 * [self.dimension] + [
                self.nReducedParameter])  # initialize full derivative with respect to all parameters
        Dpf[:, :, :, 1:] = fullDf[
            np.ix_(np.arange(self.dimension), np.arange(self.dimension), np.arange(self.dimension),
                   self.nonHillIndex)]  # insert derivatives of non-hill parameters
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
            2 * [self.dimension] + 2 * [
                self.nReducedParameter])  # initialize full derivative with respect to all parameters
        Dpf[:, :, 1:, 1:] = fullDf[
            np.ix_(np.arange(self.dimension), np.arange(self.dimension), self.nonHillIndex,
                   self.nonHillIndex)]  # insert derivatives of non-hill parameters
        Dpf[:, :, 0, 0] = np.einsum('ijkl->ij', fullDf[
            np.ix_(np.arange(self.dimension), np.arange(self.dimension), self.hillIndex,
                   self.hillIndex)])  # insert sum of derivatives for identified hill parameters
        if diffIndex[0] is None and diffIndex[1] is None:
            return Dpf  # return the full vector of partials
        else:
            return np.squeeze(
                Dpf[np.ix_(np.arange(self.dimension), np.arange(self.dimension), diffIndex,
                           diffIndex)])  # return only slices for the specified subset of partials

    @classmethod
    def Model_from_Model(cls, A : Model):
        # Assumed no fixed parameters!
        gamma = np.nan + np.zeros(A.dimension)
        productionSign = [[j.sign for j in A.coordinates[i].productionComponents] for i in range(A.dimension)]
        productionType = [A.coordinates[i].productionType for i in range(A.dimension)]
        productionIndex = A.productionIndex
        parameter = [[np.nan +  np.zeros(3) for j in A.productionIndex[i]] for i in range(A.dimension)]
        return RestrictedHillModel(gamma, parameter, productionSign, productionType, productionIndex)


if __name__ == "__main__":
    hill = 4
    gamma = [np.nan, np.nan, np.nan, np.nan]
    p1 = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
    p2 = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
    p4 = np.array(
        [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
        dtype=float)
    parameter = [p1, p1, p1, p4]

    productionSign = [[1], [-1], [1], [1, -1, -1]]
    productionType = [[1], [1], [1], [1, 2]]
    productionIndex = [[1], [2], [3], [2, 1, 0]]
    g = Model(gamma, parameter, productionSign, productionType, productionIndex)
    print('Example model:\n', g)

    def fill_with_randoms(p):
        y = p
        for i in range(len(p)):
            for j in range(len(p[i])):
                y[i][j] = np.random.rand()
        return y

    x = np.random.random(4)
    pars = np.random.random(28)
    index_list = [4, 9, 14, 19, 23, 27]  # 4 is the number of gamma parameters
    for i in index_list:
        pars[i] = hill
    print('full model: ', g(x, pars))

    p1 = np.array([[np.nan, np.nan, np.nan]], dtype=float)
    p4 = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=float)
    parameter_small = [p1, p1, p1, p4]
    g_tilde = RestrictedHillModel(gamma, parameter_small, productionSign, productionType, productionIndex)
    pars_small = np.delete(pars, index_list, axis=0)
    g_tilde(x, hill, pars_small)

    print('difference between full model and identified Hill coefs = ',
          np.linalg.norm(g(x, pars) - g_tilde(x, hill, pars_small)))