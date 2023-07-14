"""
An implementation of the 6 node EMT network as a Hill model
    Other files required: hill_model.py

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 1/15/2021
"""

from bifurcation.saddlenode import *
from model.model import Model


class EMT(Model):
    """Six-dimensional EMT model construction inherited as a HillModel where each node has free Hill coefficients. This
     has a total of 12 edges and 54 parameters. The nodes are ordered as follows:
    0. TGF_beta
    1. miR200
    2. Snail1
    3. Ovol2
    4. Zeb1
    5. miR34a"""

    def __init__(self, gamma, parameter):
        """Class constructor which has the following syntax:
        INPUTS:
            gamma - A vector in R^6 of linear decay rates or NaN if decays are variable parameters.
            parameter - A length-6 list of parameter arrays of size K_i-by-4 where K_i is the number of incoming edges to
             node i. Each row of a parameter array has the form (ell, delta, theta, hill)."""

        # TODO: The productionIndex specified here is wrong. It should only include nonlinear global indices.

        parameter = list(map(lambda parmArray: np.concatenate([parmArray, np.array([np.shape(parmArray)[0] * [
            np.nan]]).transpose()], axis=1), parameter))  # Insert a nan value into the Hill coefficient spot of the
        # HillComponent parameter list associated to every edge
        productionSign = [[-1, -1], [-1, -1], [1, -1], [-1], [-1, 1, -1],
                          [-1, -1]]  # length 6 list of production signs for each node
        productionType = [len(sign) * [1] for sign in productionSign]  # all productions are products
        productionIndex = [[0, 1, 3], [1, 2, 4], [2, 0, 5], [3, 4], [4, 1, 2, 3], [5, 2, 4]]
        super().__init__(gamma, parameter, productionSign, productionType,
                         productionIndex)  # define HillModel for toggle switch by inheritance

        # # Alterations for identifying all Hill coefficients.
        self.hillIndex = self.hill_coefficient_idx()  # indices of Hill coefficient parameters in the full parameter
        # vector
        self.nonHillIndex = np.array([idx for idx in range(self.nParameter) if
                                      idx not in self.hillIndex])  # indices of non Hill coefficient variable parameters in the full vector
        self.hillInsertionIndex = self.hillIndex - np.array(range(12))
        self.nReducedParameter = self.nParameter - 11  # adjust variable parameter count by 11 to account for the 12
        # identified Hill
        # coefficients.

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
        OUTPUT: A vector of size M+12 where the value of hill has been inserted into all 12 HillCoefficient parameter locations."""
        parameterVector = ezcat(
            *parameter)  # concatenate input into a single vector. Its first component must be the common hill parameter for both coordinates
        hill, p = parameterVector[0], parameterVector[1:]
        return np.insert(p, self.hillInsertionIndex, hill)

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


if __name__ == "__main__":
    # set some parameters to test with
    gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
    edgeCounts = [2, 2, 2, 1, 3, 2]
    parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]

    f = EMT(gammaVar, parameterVar)
    # SNB = SaddleNode(f)
    #
    gammaValues = np.array([j for j in range(1, 7)])
    parmValues = [np.random.rand(*np.shape(parameterVar[node])) for node in range(6)]
    x = np.random.rand(6)
    p = ezcat(*[ezcat(ezcat(tup[0], tup[1].flatten())) for tup in
                zip(gammaValues, parmValues)])  # this only works when all parameters are variable
    hill = 4
    #
    print(np.shape(f(x, hill, p)))
    print(np.shape(f.dx(x, hill, p)))
    print(np.shape(f.diff(x, hill, p)))
    print(np.shape(f.dx2(x, hill, p)))
    print(np.shape(f.dxdiff(x, hill, p)))
    print(np.shape(f.diff2(x, hill, p)))
