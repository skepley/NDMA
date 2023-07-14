"""
A separate file to store important HillModel subclasses for analysis or testing

    Other files required: hill_model

    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 6/24/2020 
"""
from hill_model import *
from model.model import Model


class Network12(Model):
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
            [self.coordinates[j].nProduction for j in range(self.dimension)])  # count total number of Hill productionComponents
        self.hillIndex = ezcat(
            *[self.parameterIndexByCoordinate[j] + self.coordinates[j].productionParameterIndexRange[1:] - 1 for j in
              range(self.dimension)])
        # insertion indices for HillCoefficients to expand the truncated parameter vector to a full parameter vector
        self.nonhillIndex = np.array([idx for idx in range(self.nParameter) if
                                      idx not in self.hillIndex])  # indices of non Hill coefficient variable parameters in the full vector
        self.hillInsertionIndex = self.hillIndex - np.arange(self.nComponent)
        self.nParameter -= (
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
        Dpf = np.zeros([self.dimension, self.nParameter])  # initialize full derivative w.r.t. all parameters
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
            2 * [self.dimension] + [self.nParameter])  # initialize full derivative w.r.t. all parameters
        Dpf[:, :, 1:] = fullDf[:, :, self.nonhillIndex]  # insert derivatives of non-hill parameters
        Dpf[:, :, 0] = np.einsum('ijk->ij', fullDf[:, :,
                                            self.hillIndex])  # insert sum of derivatives for identified hill parameters

        if diffIndex is None:
            return Dpf  # return the full vector of partials
        else:
            return np.squeeze(
                Dpf[:, :, np.array([diffIndex])])  # return only columns for the specified subset of partials

