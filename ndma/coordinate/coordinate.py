"""
A description of what the script performs

    Output: output
    Other files required: none
   
    Author: Shane Kepley
    email: s.kepley@vu.nl
    Date: 7/14/23; Last revision: 7/14/23
"""
import textwrap
from itertools import permutations, product

import numpy as np
# from ndma.activation import HillActivation
from ndma.hill_model import ezcat, is_vector


def verify_call(func):
    """Evaluation method decorator for validating evaluation calls. This can be used when troubleshooting and
    testing and then omitted in production runs to improve efficiency. This decorates any Coordinate method which has
    inputs of the form (self, x, *parameter, **kwargs)."""

    def func_wrapper(*args, **kwargs):
        coordinateObj = args[
            0]  # a HillModel or HillCoordinate instance is passed as 1st position argument to evaluation
        # method
        x = args[1]  # state vector passed as 2nd positional argument to evaluation method
        if not is_vector(x):  # x must be a vector
            raise TypeError('Second argument must be a state vector.')

        N = coordinateObj.nState
        parameter = args[2]  # parameter vector passed as 3rd positional argument to evaluation method

        if len(x) != N:  # make sure state input is the correct size
            raise IndexError(
                'State vector for this evaluation should be size {0} but received a vector of size {1}'.format(N,
                                                                                                               len(x)))
        elif len(parameter) != coordinateObj.nParameter:
            raise IndexError(
                'Parsed parameter vector for this evaluation should be size {0} but received a vector of '
                'size {1}'.format(
                    coordinateObj.nParameter, len(parameter)))

        else:  # parameter and state vectors are correct size. Pass through to evaluation method
            return func(*args, **kwargs)

    return func_wrapper


class Coordinate:
    """Define a single scalar coordinate of a Hill system. This function always takes the form of a linear decay term
    and a nonlinear production term defined by composing a polynomial interaction function with HillCoordinates
    which each depend only on a single variable. Specifically, a HillCoordinate represents a function, f : R^K ---> R.
    which describes rate of production and decay of a scalar quantity x. If x does not have a nonlinear self production,
    (i.e. no self loop in the network topology) then this is a scalar differential equation taking the form
    x' = -gamma*x + p(H_1, H_2, ...,H_{K-1})
    where each H_i is a Hill function depending on a single state variable y_i != x. Otherwise, if x does contribute to
    its own nonlinear production, then f takes the form,
    x' = -gamma*x + p(H_1, H_2,...,H_K)
    where H_1 depends on x and H_2,...,H_K depend on a state variable y_i != x."""

    def __init__(self, gamma, parameter, productionSign, productionType, nStateVariables, activationFunction):
        """Hill Coordinate instantiation with the following syntax:
        INPUTS:
            gamma - (float) decay rate for this coordinate or NaN if gamma is a variable parameter which is callable as
                the first component of the parameter variable vector.
            parameter - (numpy array) A K-by-4 array of Hill component parameters with rows of the form [ell, delta, theta, hillCoefficient]
                Entries which are NaN are variable parameters which are callable in the function and all derivatives.
            productionSign - (list) A vector in F_2^K carrying the sign type for each Hill component
            productionType - (list) A vector describing the interaction type of the interaction function specified as an
                ordered integer partition of K.
            nStateVariables - (integer) Report how many state variables this HillCoordinate depends on. All evaluation methods will expect a state vector of
                this size.
            activationFunction - The name of the activation function for all nonlinear production terms.
            """

        # TODO: 1. Class constructor should not do work!
        self.gammaIsVariable = np.isnan(gamma)
        if ~np.isnan(gamma):
            self.gamma = gamma  # set fixed linear decay
        self.nState = nStateVariables  # dimension of state vector input to HillCoordinate
        self.parameterValues = parameter  # initialize array for the fixed (non-variable) parameter values
        self.nProduction = len(productionSign)  # number of incoming edges contributing to nonlinear production. In the
        # current version this is always equal to self.nState - 1 (no self edge) or self.nState (self edge)
        self.productionIndex = list(range(self.nState)[slice(-self.nProduction, None,
                                                             1)])  # state variable selection for the production term are the trailing
        # K variables. If this coordinate has a self edge this is the entire vector, otherwise, it selects all state
        # variables except the first state variable.
        self.activation = activationFunction
        self.productionComponents, self.nParameterByProductionIndex, self.productionParameterIndexRange = self.set_production(
            parameter, productionSign)
        self.productionType = productionType  # specified as an integer partition of K
        self.summand = self.set_summand()

        self.nParameter = sum(
            self.nParameterByProductionIndex) + int(
            self.gammaIsVariable)  # number of variable parameters for this coordinate.

    def parse_parameters(self, parameter):
        """Returns the value of gamma and slices of the parameter vector divided by component"""

        # If gamma is not fixed, then it must be the first coordinate of the parameter vector
        if self.gammaIsVariable:
            gamma = parameter[0]
        else:
            gamma = self.gamma
        return gamma, [parameter[self.productionParameterIndexRange[j]:self.productionParameterIndexRange[j + 1]] for
                       j in range(self.nProduction)]

    def parameter_to_production_index(self, linearIndex):
        """Convert a linear parameter index to an ordered pair, (i, j) where the specified parameter is the j^th variable
         parameter of the i^th Hill production function."""

        if self.gammaIsVariable and linearIndex == 0:
            print('production index for a decay parameter is undefined')
            raise KeyboardInterrupt
        componentIndex = np.searchsorted(self.productionParameterIndexRange,
                                         linearIndex + 0.5) - 1  # get the production index which contains the variable parameter. Adding 0.5
        # makes the returned value consistent in the case that the diffIndex is an endpoint of the variable index list
        parameterIndex = linearIndex - self.productionParameterIndexRange[
            componentIndex]  # get the local parameter index in the HillComponent for the variable parameter
        return componentIndex, parameterIndex

    def component_to_parameter_index(self, componentIdx, localIdx):
        """Given an input (i,j), return a linear index for the j^th local parameter of the i^th Hill component"""

        return self.productionParameterIndexRange[componentIdx] + localIdx

    @verify_call
    def __call__(self, x, parameter):
        """Evaluate the Hill coordinate on a vector of state and parameter variables. This is a
        map of the form  g: R^n x R^m ---> R where n is the number of state variables for this Hill coordinate (in the current version
        with Hill functions in the decay term this is either n = K or n = K + 1). m is the number of variable parameters for this
        Hill coordinate (at most m = 1 + 4K). When calling this function for (x_1,...,x_n) is is REQUIRED that the global index of
        x_1 is the state variable associated with this HillCoordinate."""

        # TODO: Currently the input parameter must be a numpy array even if there is only a single parameter.
        # Evaluate coordinate for a single x in R^n. Slice callable parameters into a list of length K. The j^th list contains the variable parameters belonging to
        # the j^th Hill function in the production term.

        gamma, parameterByComponent = self.parse_parameters(parameter)
        productionComponentValues = self.evaluate_production_components(x, parameter)
        summandValues = self.evaluate_summand(productionComponentValues)
        nonlinearProduction = np.prod(summandValues)
        # TODO: Below is the old version. This should be removed once the refactored classes are fully vetted.
        # nonlinearProduction = self.evaluate_production_interaction(
        #     productionHillValues)  # compose with production interaction function
        return -gamma * x[0] + nonlinearProduction

    def __repr__(self):
        """Return a canonical string representation of a Hill coordinate"""

        reprString = 'Hill Coordinate: \n' + 'Production Type: p = ' + (
                '(' + ')('.join(
            [' + '.join(['z_{0}'.format(idx + 1) for idx in summand]) for summand in self.summand]) + ')\n') + (
                             'Components: H = (' + ', '.join(
                         map(lambda i: 'H+' if i == 1 else 'H-', [H.sign for H in self.productionComponents])) + ') \n')

        # initialize index strings
        stateIndexString = 'State Variables: x = (x_i; '
        variableIndexString = 'Variable Parameters: lambda = ('
        if self.gammaIsVariable:
            variableIndexString += 'gamma, '

        for k in range(self.nProduction):
            idx = self.productionIndex[k]
            stateIndexString += 'x_{0}, '.format(idx)
            if self.productionComponents[k].variableParameters:
                variableIndexString += ', '.join(
                    [var + '_{0}'.format(idx) for var in self.productionComponents[k].variableParameters])
                variableIndexString += ', '

        # remove trailing commas and close brackets
        variableIndexString = variableIndexString[:-2]
        stateIndexString = stateIndexString[:-2]
        variableIndexString += ')\n'
        stateIndexString += ')\n'
        reprString += stateIndexString + '\n          '.join(textwrap.wrap(variableIndexString, 80))
        return reprString

    def evaluate_production_components(self, x, parameter):
        """Evaluate each HillComponent for the production term. Returns an ordered vector in R^K."""

        gamma, parameterByProductionComponent = self.parse_parameters(parameter)
        return np.array(
            list(map(lambda H, x_i, parm: H(x_i, parm), self.productionComponents, x[self.productionIndex],
                     parameterByProductionComponent)))  # evaluate Hill productionComponents

    def summand_index(self, componentIdx):
        """Returns the summand index of a component index. This is a map of the form, I : {1,...,K} --> {1,...,q} which
        identifies to which summand of the production interaction the k^th production component contributes."""

        return self.summand.index(filter(lambda L: componentIdx in L, self.summand).__next__())

    def evaluate_summand(self, componentValues):
        """Evaluate the summands of the production interaction function. This is a map taking values in R^q where the input is
        a vector in R^K obtained by evaluating the Hill production components. The component values which contribute to the same summand are then
        summed according to the productionType."""

        return np.array([np.sum(componentValues[self.summand[j]]) for j in range(len(self.summand))])

    def evaluate_production_interaction(self, componentValues):
        """Evaluate the production interaction function at vector of HillComponent values: (H1,...,HK). This is the second evaluation
        in the composition which defines the production term."""

        # TODO: This function is deprecated but it is still used in the HillModel.eq_interval method. This usage
        #  should be replaced with calls to evaluate_summand and np.prod instead.
        # print('Deprecation Warning: This function should no longer be called. Use the evaluate_summand method and '
        #       'np.prod() instead.')
        if len(self.summand) == 1:  # this is the all sum interaction type
            return np.sum(componentValues)
        else:
            return np.prod([sum([componentValues[idx] for idx in summand]) for summand in self.summand])

    def diff_production(self, x, parameter, diffOrder, diffIndex=None):
        """Return the differential of the specified order for the production interaction function in the coordinate specified by
        diffIndex. The differential is evaluated at the vector of HillComponent evaluations i.e. this function serves as the
         outer function call when evaluating chain rule derivatives for HillCoordinates with respect to state or parameter vectors.
         If diffIndex is not specified, it returns the full derivative as a vector with all K partials of
        order diffOrder."""

        def nonzero_index(order):
            """Return the indices for which the given order derivative of an interaction function is nonzero. This happens
            precisely for every multi-index in the tensor for which each component is drawn from a different summand."""

            summandTuples = permutations(self.summand, order)
            summandProducts = []  # initialize cartesian product of all summand tuples
            for tup in summandTuples:
                summandProducts += list(product(*tup))

            return np.array(summandProducts)

        nSummand = len(self.productionType)  # number of summands
        if diffIndex is None:  # compute the full differential of p as a vector in R^K with each component evaluated at
            # H_k(x_k, p_k).

            if diffOrder == 1:  # compute first derivative of interaction function composed with Hill Components
                if nSummand == 1:  # the all sum special case
                    return np.ones(self.nProduction)
                else:
                    productionComponentValues = self.evaluate_production_components(x, parameter)
                    summandValues = self.evaluate_summand(productionComponentValues)
                    fullProduct = np.prod(summandValues)
                    DxProducts = fullProduct / summandValues  # evaluate all partials only once using q multiplies. The m^th term looks like P/p_m.
                    return np.array([DxProducts[self.summand_index(k)] for k in
                                     range(
                                         self.nProduction)])  # broadcast values to all members sharing the same summand

            elif diffOrder == 2:  # compute second derivative of interaction function composed with Hill Components as a 2-tensor
                if nSummand == 1:  # the all sum special case
                    return np.zeros(diffOrder * [self.nProduction])  # initialize Hessian of interaction function

                elif nSummand == 2:  # the 2 summands special case
                    DpH = np.zeros(diffOrder * [self.nProduction])  # initialize derivative tensor
                    idxArray = nonzero_index(diffOrder)  # array of nonzero indices for derivative tensor
                    DpH[idxArray[:, 0], idxArray[:, 1]] = 1  # set nonzero terms to 1
                    return DpH

                else:
                    DpH = np.zeros(2 * [self.nProduction])  # initialize Hessian of interaction function
                    # compute Hessian matrix of interaction function by summand membership
                    productionComponentValues = self.evaluate_production_components(x, parameter)
                    summandValues = self.evaluate_summand(productionComponentValues)
                    fullProduct = np.prod(summandValues)
                    DxProducts = fullProduct / summandValues  # evaluate all partials using only nSummand-many multiplies
                    DxxProducts = np.outer(DxProducts,
                                           1.0 / summandValues)  # evaluate all second partials using only nSummand-many additional multiplies.
                    # Only the cross-diagonal terms of this matrix are meaningful.
                    for row in range(nSummand):  # compute Hessian of interaction function (outside term of chain rule)
                        for col in range(row + 1, nSummand):
                            Irow = self.summand[row]
                            Icolumn = self.summand[col]
                            DpH[np.ix_(Irow, Icolumn)] = DpH[np.ix_(Icolumn, Irow)] = DxxProducts[row, col]
                    return DpH

            elif diffOrder == 3:  # compute third derivative of interaction function composed with Hill Components as a 3-tensor
                if nSummand <= 2:  # the all sum or 2-summand special cases
                    return np.zeros(diffOrder * [self.nProduction])  # initialize Hessian of interaction function

                elif nSummand == 3:  # the 2 summands special case
                    DpH = np.zeros(diffOrder * [self.nProduction])  # initialize derivative tensor
                    idxArray = nonzero_index(diffOrder)  # array of nonzero indices for derivative tensor
                    DpH[idxArray[:, 0], idxArray[:, 1], idxArray[:, 2]] = 1  # set nonzero terms to 1
                    return DpH
                else:
                    raise KeyboardInterrupt

        else:  # compute a single partial derivative of p
            if diffOrder == 1:  # compute first partial derivatives
                if len(self.productionType) == 1:
                    return 1.0
                else:
                    productionComponentValues = self.evaluate_production_components(x, parameter)
                    summandValues = self.evaluate_summand(productionComponentValues)
                    I_k = self.summand_index(diffIndex)  # get the summand index containing the k^th Hill component
                    return np.prod(
                        [summandValues[m] for m in range(len(self.productionType)) if
                         m != I_k])  # multiply over
                # all summands which do not contain the k^th component
            else:
                raise KeyboardInterrupt

    def diff_production_component(self, x, parameter, diffOrder, *diffIndex, fullTensor=True):
        """Compute derivative of component vector, H = (H_1,...,H_K) with respect to state variables or parameters. This is
        the inner term in the chain rule derivative for the higher order derivatives of a HillCoordinate. diffOrder has the form
         [xOrder, parameterOrder] which specifies the number of derivatives with respect to state variables and parameter
         variables respectively. Allowable choices are: {[1,0], [0,1], [2,0], [1,1], [0,2], [3,0], [2,1], [1,2]}"""

        xOrder = diffOrder[0]
        parameterOrder = diffOrder[1]
        gamma, parameterByComponent = self.parse_parameters(parameter)
        xProduction = x[
            self.productionIndex]  # extract only the coordinates of x that contribute to the production term.

        if parameterOrder == 0:  # return partials of H with respect to x as a length K vector of nonzero values. dH is
            # obtained by taking the diag operator to broadcast this vector to a tensor of correct rank.

            if xOrder == 1:
                DH_nonzero = np.array(
                    list(map(lambda H_k, x_k, p_k: H_k.dx(x_k, p_k), self.productionComponents, xProduction,
                             parameterByComponent)))  # evaluate vector of first order state variable partial derivatives for Hill productionComponents
            elif xOrder == 2:
                DH_nonzero = np.array(
                    list(map(lambda H_k, x_k, p_k: H_k.dx2(x_k, p_k), self.productionComponents, xProduction,
                             parameterByComponent)))  # evaluate vector of second order state variable partial derivatives for Hill productionComponents
            elif xOrder == 3:
                DH_nonzero = np.array(
                    list(map(lambda H_k, x_k, p_k: H_k.dx3(x_k, p_k), self.productionComponents, xProduction,
                             parameterByComponent)))  # evaluate vector of third order state variable partial derivatives for Hill productionComponents

            if fullTensor:
                DH = np.zeros((1 + xOrder) * [self.nProduction])
                np.einsum(''.join((1 + xOrder) * 'i') + '->i', DH)[:] = DH_nonzero
                return DH
            else:
                return DH_nonzero

        elif parameterOrder == 1:  # return partials w.r.t parameters specified by diffIndex as a vector of nonzero productionComponents.

            if not diffIndex:  # no optional argument means return all component parameter derivatives (i.e. all parameters except gamma)
                diffIndex = list(range(int(self.gammaIsVariable), self.nParameter))
            parameterComponentIndex = [self.parameter_to_production_index(linearIdx) for linearIdx in
                                       diffIndex]  # a list of ordered pairs for differentiation parameter indices

            if xOrder == 0:  # Compute D_lambda(H)
                DH_nonzero = np.array(
                    list(map(lambda idx: self.productionComponents[idx[0]].diff(xProduction[idx[0]],
                                                                                parameterByComponent[idx[0]],
                                                                                idx[1]),
                             parameterComponentIndex)))  # evaluate vector of first order partial derivatives for Hill productionComponents

            elif xOrder == 1:
                DH_nonzero = np.array(
                    list(map(lambda idx: self.productionComponents[idx[0]].dxdiff(xProduction[idx[0]],
                                                                                  parameterByComponent[idx[0]],
                                                                                  idx[1]),
                             parameterComponentIndex)))  # evaluate vector of second order mixed partial derivatives for Hill productionComponents
            elif xOrder == 2:
                DH_nonzero = np.array(
                    list(map(lambda idx: self.productionComponents[idx[0]].dx2diff(xProduction[idx[0]],
                                                                                   parameterByComponent[idx[0]],
                                                                                   idx[1]),
                             parameterComponentIndex)))  # evaluate vector of third order mixed partial derivatives for Hill productionComponents

            if fullTensor:
                tensorDims = (1 + xOrder) * [self.nProduction] + [self.nParameter - self.gammaIsVariable]
                DH = np.zeros(tensorDims)
                nonzeroComponentIdx = list(zip(*parameterComponentIndex))[
                    0]  # zip into a pair of tuples for last two einsum indices
                nonzeroIdx = tuple((1 + xOrder) * [nonzeroComponentIdx] + [
                    tuple(range(tensorDims[-1]))])  # prepend copies of the Hill component index for xOrder derivatives
                DH[nonzeroIdx] = DH_nonzero
                return DH
            else:
                return DH_nonzero

        elif parameterOrder == 2:  # 2 partial derivatives w.r.t. parameters.

            if not diffIndex:  # no optional argument means return all component parameter derivatives twice (i.e. all parameters except gamma)
                from itertools import product
                diffIndex = []  # initialize a list of parameter pairs
                for idx in range(self.nProduction):
                    parameterSlice = range(self.productionParameterIndexRange[idx],
                                           self.productionParameterIndexRange[idx + 1])
                    diffIndex += list(product(parameterSlice, parameterSlice))

            parameterComponentIndex = [
                ezcat(self.parameter_to_production_index(idx[0]), self.parameter_to_production_index(idx[1])[1]) for idx
                in diffIndex]
            # a list of triples stored as numpy arrays of the form (i,j,k) where lambda_j, lambda_k are both parameters for H_i

            if xOrder == 0:
                DH_nonzero = np.array(
                    list(map(lambda idx: self.productionComponents[idx[0]].diff2(xProduction[idx[0]],
                                                                                 parameterByComponent[idx[0]],
                                                                                 idx[1:]),
                             parameterComponentIndex)))  # evaluate vector of second order pure partial derivatives for Hill productionComponents

            elif xOrder == 1:
                DH_nonzero = np.array(
                    list(map(lambda idx: self.productionComponents[idx[0]].dxdiff2(xProduction[idx[0]],
                                                                                   parameterByComponent[idx[0]],
                                                                                   idx[1:]),
                             parameterComponentIndex)))  # evaluate vector of third order mixed partial derivatives for Hill productionComponents

            if fullTensor:
                tensorDims = (1 + xOrder) * [self.nProduction] + 2 * [self.nParameter - self.gammaIsVariable]
                DH = np.zeros(tensorDims)
                nonzeroTripleIdx = list(zip(*parameterComponentIndex))
                nonzeroComponentIdx = nonzeroTripleIdx[0]
                nonzeroLambdaIdx = [tuple(
                    self.component_to_parameter_index(nonzeroComponentIdx[j], nonzeroTripleIdx[1][j]) - int(
                        self.gammaIsVariable) for j in
                    range(len(nonzeroComponentIdx))),
                    tuple(self.component_to_parameter_index(nonzeroComponentIdx[j],
                                                            nonzeroTripleIdx[2][j]) - int(self.gammaIsVariable) for j in
                          range(len(nonzeroTripleIdx[0])))
                ]

                nonzeroIdx = tuple((1 + xOrder) * [
                    nonzeroComponentIdx] + nonzeroLambdaIdx)  # prepend copies of the Hill component index for xOrder derivatives
                DH[nonzeroIdx] = DH_nonzero
                return DH
                # return DH, DH_nonzero, nonzeroIdx
            else:
                return DH_nonzero

    @verify_call
    def dx(self, x, parameter):
        """Return the derivative as a gradient vector evaluated at x in R^n and p in R^m"""

        gamma, parameterByComponent = self.parse_parameters(parameter)
        Df = np.zeros(self.nState, dtype=float)
        diffInteraction = self.diff_production(x, parameter,
                                               1)  # evaluate derivative of production interaction function (outer term in chain rule)
        DHillComponent = np.array(
            list(map(lambda H_k, x_k, p_k: H_k.dx(x_k, p_k), self.productionComponents, x[self.productionIndex],
                     parameterByComponent)))  # evaluate vector of partial derivatives for production Hill Components (inner term in chain rule)
        Df[
            self.productionIndex] = diffInteraction * DHillComponent  # evaluate gradient of nonlinear part via chain rule
        Df[0] -= gamma  # Add derivative of linear part to the gradient at this HillCoordinate
        return Df

    @verify_call
    def diff(self, x, parameter, diffIndex=None):
        """Evaluate the derivative of a Hill coordinate with respect to a parameter at the specified local index.
           The parameter must be a variable parameter for one or more HillComponents."""

        if diffIndex is None:  # return the full gradient with respect to parameters as a vector in R^m
            return np.array([self.diff(x, parameter, diffIndex=k) for k in range(self.nParameter)])

        else:  # return a single partial derivative as a scalar
            if self.gammaIsVariable and diffIndex == 0:  # derivative with respect to decay parameter
                return -x[0]
            else:  # First obtain a local index in the HillComponent for the differentiation variable
                diffComponent = np.searchsorted(self.productionParameterIndexRange,
                                                diffIndex + 0.5) - 1  # get the component which contains the differentiation variable. Adding 0.5
                # makes the returned value consistent in the case that the diffIndex is an endpoint of the variable index list
                diffParameterIndex = diffIndex - self.productionParameterIndexRange[
                    diffComponent]  # get the local parameter index in the HillComponent for the differentiation variable

                # Now evaluate the derivative through the HillComponent and embed into tangent space of R^n
                gamma, parameterByComponent = self.parse_parameters(parameter)
                xProduction = x[
                    self.productionIndex]  # extract only the coordinates of x that contribute to the production of this HillCoordinate
                diffInteraction = self.diff_production(x, parameter, 1,
                                                       diffIndex=diffComponent)  # evaluate outer term in chain rule
                dpH = self.productionComponents[diffComponent].diff(xProduction[diffComponent],
                                                                    parameterByComponent[
                                                                        diffComponent],
                                                                    diffParameterIndex)  # evaluate inner term in chain rule
                return diffInteraction * dpH

    @verify_call
    def dx2(self, x, parameter):
        """Return the second derivative (Hessian matrix) with respect to the state variable vector evaluated at x in
        R^n and p in R^m as a K-by-K matrix"""

        gamma, parameterByComponent = self.parse_parameters(parameter)
        xProduction = x[
            self.productionIndex]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{K}
        D2f = np.zeros(2 * [self.nState], dtype=float)

        D2HillComponent = np.array(
            list(map(lambda H_k, x_k, p_k: H_k.dx2(x_k, p_k), self.productionComponents, x[self.productionIndex],
                     parameterByComponent)))
        # evaluate vector of second partial derivatives for production Hill Components
        nSummand = len(self.productionType)  # number of summands

        if nSummand == 1:  # production is all sum
            D2Nonlinear = np.diag(D2HillComponent)
        # TODO: Adding more special cases for 2 and even 3 summand production types will speed up the computation quite a bit.
        #       This should be done if this method ever becomes a bottleneck.

        else:  # production interaction function contributes derivative terms via chain rule

            # compute off diagonal terms in Hessian matrix by summand membership
            productionComponentValues = self.evaluate_production_components(x, parameter)
            summandValues = self.evaluate_summand(productionComponentValues)
            fullProduct = np.prod(summandValues)
            DxProducts = fullProduct / summandValues  # evaluate all partials using only nSummand-many multiplies

            # initialize Hessian matrix and set diagonal terms
            DxProductsByComponent = np.array([DxProducts[self.summand_index(k)] for k in range(self.nProduction)])
            D2Nonlinear = np.diag(D2HillComponent * DxProductsByComponent)

            # set off diagonal terms of Hessian by summand membership and exploiting symmetry
            DxxProducts = np.outer(DxProducts,
                                   1.0 / summandValues)  # evaluate all second partials using only nSummand-many additional multiplies.
            # Only the cross-diagonal terms of this matrix are meaningful.

            offDiagonal = np.zeros_like(D2Nonlinear)  # initialize matrix of mixed partials (off diagonal terms)
            for row in range(
                    nSummand):  # compute Hessian of production interaction function (outside term of chain rule)
                for col in range(row + 1, nSummand):
                    offDiagonal[np.ix_(self.summand[row], self.summand[col])] = offDiagonal[
                        np.ix_(self.summand[col], self.summand[row])] = DxxProducts[row, col]

            DHillComponent = np.array(
                list(map(lambda H_k, x_k, p_k: H_k.dx(x_k, p_k), self.productionComponents, x[self.productionIndex],
                         parameterByComponent)))  # evaluate vector of partial derivatives for Hill productionComponents
            mixedPartials = np.outer(DHillComponent,
                                     DHillComponent)  # mixed partial matrix is outer product of gradients!
            D2Nonlinear += offDiagonal * mixedPartials
            # NOTE: The diagonal terms of offDiagonal are identically zero for any interaction type which makes the
            # diagonal terms of mixedPartials irrelevant
        D2f[np.ix_(self.productionIndex, self.productionIndex)] = D2Nonlinear
        return D2f

    @verify_call
    def dxdiff(self, x, parameter, diffIndex=None):
        """Return the mixed second derivative with respect to x and a scalar parameter evaluated at x in
        R^n and p in R^m as a gradient vector in R^K. If no parameter index is specified this returns the
        full second derivative as the m-by-K Hessian matrix of mixed partials"""

        if diffIndex is None:
            return np.column_stack(
                list(map(lambda idx: self.dxdiff(x, parameter, idx), range(self.nParameter))))

        else:
            D2f = np.zeros(self.nState, dtype=float)  # initialize derivative as a vector

            if self.gammaIsVariable and diffIndex == 0:  # derivative with respect to decay parameter
                D2f[0] = -1
                return D2f

            gamma, parameterByComponent = self.parse_parameters(parameter)
            xProduction = x[
                self.productionIndex]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{K}
            diffComponent = np.searchsorted(self.productionParameterIndexRange,
                                            diffIndex + 0.5) - 1  # get the component which contains the differentiation variable. Adding 0.5
            # makes the returned value consistent in the case that the diffIndex is an endpoint of the variable index list
            diffParameterIndex = diffIndex - self.productionParameterIndexRange[
                diffComponent]  # get the local parameter index in the HillComponent for the differentiation variable

            # initialize inner terms of chain rule derivatives of f
            # DH = np.zeros(2 * [self.nProduction])  # initialize diagonal tensor for DxH as a 2-tensor
            DHillComponent = np.array(
                list(map(lambda H_k, x_k, p_k: H_k.dx(x_k, p_k), self.productionComponents, xProduction,
                         parameterByComponent)))  # 1-tensor of partials for DxH
            # np.einsum('ii->i', DH)[:] = DHillComponent  # build the diagonal tensor for DxH
            DpH = self.productionComponents[diffComponent].diff(xProduction[diffComponent],
                                                                parameterByComponent[diffComponent],
                                                                diffParameterIndex)

            D2H = self.productionComponents[diffComponent].dxdiff(xProduction[diffComponent],
                                                                  parameterByComponent[diffComponent],
                                                                  diffParameterIndex)  # get the correct mixed partial derivative of H_k

            # initialize outer terms of chain rule derivatives of f
            Dp = self.diff_production(x, parameter, 1)[diffComponent]  # k^th index of Dp(H) is a 0-tensor (scalar)
            D2p = self.diff_production(x, parameter, 2)[diffComponent]  # k^th index of D^2p(H) is a 1-tensor (vector)

            D2f[self.productionIndex] += DpH * DHillComponent * D2p  # contribution from D2(p(H))*D_parm(H)*DxH
            D2f[self.productionIndex[diffComponent]] += D2H * Dp  # contribution from Dp(H)*D_parm(DxH)
            return D2f

    @verify_call
    def diff2(self, x, parameter, *diffIndex, fullTensor=True):
        """Return the second derivative with respect to parameters specified evaluated at x in
        R^n and p in R^m as a Hessian matrix. If no parameter index is specified this returns the
        full second derivative as the m-by-m Hessian matrix"""

        # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
        DlambdaH = self.diff_production_component(x, parameter, [0, 1], fullTensor=fullTensor)
        D2lambdaH = self.diff_production_component(x, parameter, [0, 2], fullTensor=fullTensor)

        # get tensors for derivatives of p o H(x) (outer terms of chain rule)
        Dp = self.diff_production(x, parameter, 1)  # 1-tensor
        D2p = self.diff_production(x, parameter, 2)  # 2-tensor

        if fullTensor:  # slow version to be used as a ground truth for testing
            term1 = np.einsum('ik,kl,ij', D2p, DlambdaH, DlambdaH)
            term2 = np.einsum('i,ijk', Dp, D2lambdaH)
            DpoH = term1 + term2
        else:
            raise ValueError

        if self.gammaIsVariable:
            D2lambda = np.zeros(2 * [self.nParameter])
            D2lambda[1:, 1:] = DpoH
            return D2lambda
        else:
            return DpoH

    @verify_call
    def dx3(self, x, parameter, fullTensor=True):
        """Return the third derivative (3-tensor) with respect to the state variable vector evaluated at x in
        R^n and p in R^m as a K-by-K matrix"""

        # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
        DxH = self.diff_production_component(x, parameter, [1, 0], fullTensor=fullTensor)
        DxxH = self.diff_production_component(x, parameter, [2, 0], fullTensor=fullTensor)
        DxxxH = self.diff_production_component(x, parameter, [3, 0], fullTensor=fullTensor)

        # get tensors for derivatives of p o H(x) (outer terms of chain rule)
        Dp = self.diff_production(x, parameter, 1)  # 1-tensor
        D2p = self.diff_production(x, parameter, 2)  # 2-tensor
        D3p = self.diff_production(x, parameter, 3)  # 3-tensor

        if fullTensor:  # slow version to be used as a ground truth for testing
            term1 = np.einsum('ikq,qr,kl,ij', D3p, DxH, DxH, DxH)
            term2 = np.einsum('ik,kl,ijq', D2p, DxH, DxxH)
            term3 = np.einsum('ik,ij,klq', D2p, DxH, DxxH)
            term4 = np.einsum('il,lq,ijk', D2p, DxH, DxxH)
            term5 = np.einsum('i, ijkl', Dp, DxxxH)
            return term1 + term2 + term3 + term4 + term5
        else:  # this code is the faster version but it is not quite correct. The .multiply method needs to be combined appropriately with
            # tensor reshaping.

            return D3p * DxH * DxH * DxH + 3 * D2p * DxH * DxxH + Dp * DxxxH

    @verify_call
    def dx2diff(self, x, parameter, fullTensor=True):
        """Return the third derivative (3-tensor) with respect to the state variable vector (twice) and then the parameter
        (once) evaluated at x in R^n and p in R^m as a K-by-K matrix"""

        # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
        DxH = self.diff_production_component(x, parameter, [1, 0], fullTensor=fullTensor)
        DxxH = self.diff_production_component(x, parameter, [2, 0], fullTensor=fullTensor)
        DlambdaH = self.diff_production_component(x, parameter, [0, 1],
                                                  fullTensor=fullTensor)  # m-vector representative of a pseudo-diagonal Km 2-tensor
        Dlambda_xH = self.diff_production_component(x, parameter, [1, 1],
                                                    fullTensor=fullTensor)  # m-vector representative of a pseudo-diagonal KKm 3-tensor
        Dlambda_xxH = self.diff_production_component(x, parameter, [2, 1],
                                                     fullTensor=fullTensor)  # m-vector representative of a pseudo-diagonal KKKm 4-tensor

        # get tensors for derivatives of p o H(x) (outer terms of chain rule)
        Dp = self.diff_production(x, parameter, 1)  # 1-tensor
        D2p = self.diff_production(x, parameter, 2)  # 2-tensor
        D3p = self.diff_production(x, parameter, 3)  # 3-tensor

        if fullTensor:  # slow version to be used as a ground truth for testing
            term1 = np.einsum('ikq,qr,kl,ij', D3p, DlambdaH, DxH, DxH)
            term2 = np.einsum('ik,kl,ijq', D2p, DxH, Dlambda_xH)
            term3 = np.einsum('ik,ij,klq', D2p, DxH, Dlambda_xH)
            term4 = np.einsum('il,lq,ijk', D2p, DlambdaH, DxxH)
            term5 = np.einsum('i, ijkl', Dp, Dlambda_xxH)
            DpoH = term1 + term2 + term3 + term4 + term5
        else:
            raise ValueError

        if self.gammaIsVariable:
            Dlambda_xx = np.zeros(2 * [self.nState] + [self.nParameter])
            Dlambda_xx[:, :, 1:] = DpoH
            return Dlambda_xx
        else:
            return DpoH

    @verify_call
    def dxdiff2(self, x, parameter, fullTensor=True):
        """Return the third derivative (3-tensor) with respect to the state variable vector (once) and the parameters (twice)
        evaluated at x in R^n and p in R^m as a K-by-K matrix"""

        # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
        DxH = self.diff_production_component(x, parameter, [1, 0], fullTensor=fullTensor)
        DlambdaH = self.diff_production_component(x, parameter, [0, 1], fullTensor=fullTensor)  # Km 2-tensor
        Dlambda_xH = self.diff_production_component(x, parameter, [1, 1], fullTensor=fullTensor)  # KKm 3-tensor
        D2lambdaH = self.diff_production_component(x, parameter, [0, 2], fullTensor=fullTensor)
        D2lambda_xH = self.diff_production_component(x, parameter, [1, 2], fullTensor=fullTensor)  # KKKm 4-tensor

        # get tensors for derivatives of p o H(x) (outer terms of chain rule)
        Dp = self.diff_production(x, parameter, 1)  # 1-tensor
        D2p = self.diff_production(x, parameter, 2)  # 2-tensor
        D3p = self.diff_production(x, parameter, 3)  # 3-tensor

        if fullTensor:  # slow version to be used as a ground truth for testing
            term1 = np.einsum('ikq,qr,kl,ij', D3p, DlambdaH, DlambdaH, DxH)
            term2 = np.einsum('ik,kl,ijq', D2p, DlambdaH, Dlambda_xH)
            term3 = np.einsum('ik,ij,klq', D2p, DxH, D2lambdaH)
            term4 = np.einsum('il,lq,ijk', D2p, DlambdaH, Dlambda_xH)
            term5 = np.einsum('i, ijkl', Dp, D2lambda_xH)
            DpoH = term1 + term2 + term3 + term4 + term5
        else:
            raise ValueError

        if self.gammaIsVariable:
            D2lambda_x = np.zeros([self.nState] + 2 * [self.nParameter])
            D2lambda_x[:, 1:, 1:] = DpoH
            return D2lambda_x
        else:
            return DpoH

    def set_production(self, parameter, productionSign):
        """Return a list of Hill functions contributing to the production term of this HillCoordinate"""

        def row2dict(row):
            """convert ordered row of parameter matrix to kwarg"""
            return {self.activation.PARAMETER_NAMES[j]: row[j] for j in range(len(self.activation.PARAMETER_NAMES)) if
                    not np.isnan(row[j])}

        # set up production Hill component functions
        if self.nProduction == 1:
            productionComponents = [self.activation(productionSign[0], **row2dict(parameter))]
        else:
            productionComponents = [self.activation(productionSign[k], **row2dict(parameter[k, :])) for k in
                                    range(self.nProduction)]  # list of ordered HillComponents for the production term

        # get a list of the number of variable parameters for each component in the production term.
        if self.nProduction == 1:  # production function consists of a single Hill function
            nParameterByProductionIndex = list(
                map(lambda j: np.count_nonzero(np.isnan(self.parameterValues)), range(self.nProduction)))
        else:  # production consists of multiple Hill functions
            nParameterByProductionIndex = list(
                map(lambda j: np.count_nonzero(np.isnan(self.parameterValues[j, :])), range(self.nProduction)))

        # get a list of endpoints for the concatenated parameter vector (linearly indexed) for each production component
        productionParameterIndexRange = np.cumsum([self.gammaIsVariable] + nParameterByProductionIndex)
        # endpoints for concatenated parameter vector split by production component. This is a
        # vector of length K+1. The kth component parameters are the slice productionParameterIndexRange[k:k+1] for k = 0...K-1

        return productionComponents, nParameterByProductionIndex, productionParameterIndexRange

    def set_summand(self):
        """Return the list of lists containing the summand indices defined by the production type.
        EXAMPLE:
            productionType = [2,1,3,1] returns the index partition [[0,1], [2], [3,4,5], [6]]"""

        sumEndpoints = np.insert(np.cumsum(self.productionType), 0,
                                 0)  # summand endpoint indices including initial zero
        localIndex = list(range(self.nProduction))
        return [localIndex[sumEndpoints[i]:sumEndpoints[i + 1]] for i in range(len(self.productionType))]

    def eq_interval(self, parameter=None):
        """Return a closed interval which must contain the projection of any equilibrium onto this coordinate"""

        if parameter is None:
            # all parameters are fixed
            # TODO: This should only require all ell, delta, and gamma variables to be fixed.
            minProduction = self.evaluate_production_interaction(
                [H.ell for H in self.productionComponents]) / self.gamma
            maxProduction = self.evaluate_production_interaction(
                [H.ell + H.delta for H in self.productionComponents]) / self.gamma

        else:
            # some variable parameters are passed in a vector containing all parameters for this Hill Coordinate
            gamma, parameterByComponent = self.parse_parameters(parameter)
            rectangle = np.row_stack(
                list(map(lambda H, parm: H.image(parm), self.productionComponents, parameterByComponent)))
            minProduction = self.evaluate_production_interaction(
                rectangle[:, 0]) / gamma  # min(f) = p(ell_1, ell_2,...,ell_K)
            maxProduction = self.evaluate_production_interaction(
                rectangle[:, 1]) / gamma  # max(f) = p(ell_1 + delta_1,...,ell_K + delta_K)

        return [minProduction, maxProduction]


