"""
An implementation of a three node negative cyclic feedback oscillator. Each edge has free (but identical)
Hill coefficients, hill_1 = hill_2 = hill_3 = hill, and possibly some other parameters free. This has a total of 3 edges
and up to 13 variable parameters.

    Other files required: HillModel.py
    SEE ALSO: ./models/ToggleSwitch.py
   
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 3/8/21; Last revision: 3/8/21
"""

from hill_model import *


class Repressilator(HillModel):
    """Class definition inherited from HillModel with methods overloaded to identity Hill coefficients"""

    def __init__(self, gamma, parameter, variant=0):
        """Class constructor which has the following syntax:
        INPUTS:
            gamma - A vector in R^3 of linear decay rates or NaN if decays are variable parameters.
            parameter - A length-3 list of parameter arrays each of size 1-by-3. Each parameter array
                has the form (ell, delta, theta) and corresponds to the unique in-edge for node X_i.
            variant - Either 0 or 1 for all repressors or 1 repressor/2 activators in the cycle respectively."""

        # append hillCoefficient as free parameter to each Hill Component
        def insert_nan(parameterArray):
            """Insert NaN values for Hill coefficients"""

            if is_vector(parameterArray):
                return np.insert(parameterArray, 3, np.nan)  # Add Hill coefficient placeholder to a single edge.
            else:
                return np.array(
                    [np.insert(row, 3, np.nan) for row in parameterArray])  # Add Hill coefficient to each row of array

        parameter = [insert_nan(parmArray) for parmArray in parameter]

        # Add interactions defined by the topology and a choice of algebra.
        if variant == 0:
            interactionSigns = [[-1], [-1],
                                [-1]]  # all nonlinear interactions are repression (strong negative feedback)
        elif variant == 1:
            interactionSigns = [[-1], [1], [1]]  # 1 repression and 2 activation (weak negative feedback)

        interactionTypes = [[1], [1], [1]]  # all nonlinear interactions depend only on a single gene
        interactionIndex = [[1], [2], [0]]  # interactions defined by a single cycle

        super().__init__(gamma, parameter, interactionSigns, interactionTypes,
                         interactionIndex)  # define HillModel by inheritance

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

    def diff2(self, x, *parameter, diffIndex=(None, None)):
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

    def dxdiff2(self, x, *parameter, diffIndex=(None, None)):
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


# function testing
if __name__ == '__main__':
    # TEST VECTOR FIELD EVALUATION
    gammaVar = np.array(3 * [np.nan])  # set all decay rates as variables
    edgeCounts = [1, 1, 1]  # count incoming edges to each node to structure the parameter array
    parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]).squeeze() for nEdge in
                    edgeCounts]  # pass all parameters as variable
    f = Repressilator(gammaVar, parameterVar)

    gammaValues = np.array([10, 20, 30])  # choose some decays
    x = np.arange(1, 4)  # choose the point x = (1,2,3) in phase space to test evaluations
    hill = 2  # choose a Hill coefficient

    # choose parameters which make the Hill components integer valued. All ell = 3, delta = 100, hill = 2, and theta = kx for k in {1,2,3}
    # All Hill functions are positive so the values are:
    # H(x, (3, 100, x, 2)) = 53,    H(x, (3, 100, 2*x, 2)) = 23,   H(x, (3, 100, 3*x, 2)) = 13
    parmValues = [np.array([3, 100, x[0]]),
                  np.array([3, 100, 2 * x[1]]),
                  np.array([3, 100, 3 * x[2]])]
    p = ezcat(*[ezcat(ezcat(tup[0], tup[1].flatten())) for tup in
                zip(gammaValues, parmValues)])  # this only works when all parameters are variable
    p = np.array([1, 1, 4, 4, 1, 1, 4, 4, 1, 1, 4, 4])

    eqs = f.find_equilibria(10, hill, p)

    stopHere
    print('f = \n', f(x, hill, p))
    print('dx = \n', f.dx(x, hill, p))
    print('dx2 = \n', f.dx2(x, hill, p))
    print('dx3 = \n', f.dx3(x, hill, p))
    print('diff = \n', f.diff(x, hill, p))
    # print('dxdiff = \n', f.dxdiff(x, hill, p))
    # print('diff2 = \n', f.diff2(x, hill, p))
    # print('dx2diff = \n', f.dx2diff(x, hill, p))
    # print('dxdiff2 = \n', f.dxdiff2(x, hill, p))

    # TEST DERIVATIVE EVALUATION
    [f0, f1, f2] = f.coordinates
    pT = np.array([3, 100, x[0], 2])
    p0 = ezcat(gammaValues[0], parmValues[0], hill)
    p1 = ezcat(gammaValues[1], parmValues[1], hill)
    p2 = ezcat(gammaValues[2], parmValues[2], hill)

    # fix dx3 for HillCoordinate
    fullTensor = True
    DxH = f0.diff_component(x, p0, [1, 0], fullTensor=fullTensor)
    DxxH = f0.diff_component(x, p0, [2, 0], fullTensor=fullTensor)
    DxxxH = f0.diff_component(x, p0, [3, 0], fullTensor=fullTensor)

    # get tensors for derivatives of p o H(x) (outer terms of chain rule)
    Dp = f0.diff_interaction(x, p0, 1)  # 1-tensor
    D2p = f0.diff_interaction(x, p0, 2)  # 2-tensor
    D3p = f0.diff_interaction(x, p0, 3)  # 3-tensor

    term1 = np.einsum('ikq,qr,kl,ij', D3p, DxH, DxH, DxH)
    term2 = np.einsum('ik,kl,ijq', D2p, DxH, DxxH)
    term3 = np.einsum('ik,ij,klq', D2p, DxH, DxxH)
    term4 = np.einsum('il,lq,ijk', D2p, DxH, DxxH)
    term5 = np.einsum('i, ijkl', Dp, DxxxH)
#
#     # print(f0.dx(x, p0))
#     # print(f1.dx(x, p1))
#     # print(f2.dx(x, p2))
#
#     y0 = f0.dx2(x, p0)
#     y = f.dx2(x, hill, p)

#
# gamma, p2ByComponent = f2.parse_parameters(p2)
# Df = np.zeros(f2.dim, dtype=float)
# xLocal = x[
#     f2.interactionIndex]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{n_i}
# diffInteraction = f2.diff_interaction(x,
#                                       p2,
#                                       1)  # evaluate derivative of interaction function (outer term in chain rule)
# DHillComponent = np.array(
#     list(map(lambda H, x_k, parm: H.dx(x_k, parm), f2.components, xLocal,
#              p2ByComponent)))  # evaluate vector of partial derivatives for Hill components (inner term in chain rule)
# Df[
#     f2.interactionIndex] = diffInteraction * DHillComponent  # evaluate gradient of nonlinear part via chain rule
# Df[f2.index] -= gamma  # Add derivative of linear part to the gradient at this HillCoordinate
