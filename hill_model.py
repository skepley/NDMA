"""
Classes and methods for constructing, evaluating, and doing parameter continuation of Hill Models

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 2/29/20; Last revision: 3/4/20
"""
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy import optimize
from numpy import log


def is_vector(array):
    """Returns true if input is a numpy vector"""

    return len(np.shape(array)) == 1


def ezcat(*coordinates):
    """A multiple dispatch concatenation function for numpy arrays. Accepts arbitrary inputs as int, float, tuple,
    list, or numpy array and concatenates into a vector returned as a numpy array. This is recursive so probably not
    very efficient for large scale use."""

    if len(coordinates) == 1:
        if isinstance(coordinates[0], list):
            return np.array(coordinates[0])
        elif isinstance(coordinates[0], np.ndarray):
            return coordinates[0]
        else:
            return np.array([coordinates[0]])

    try:
        return np.concatenate([coordinates[0], ezcat(*coordinates[1:])])
    except ValueError:
        return np.concatenate([np.array([coordinates[0]]), ezcat(*coordinates[1:])])


def find_root(f, Df, initialGuess, diagnose=False):
    """Default root finding method to use if one is not specified"""

    solution = optimize.root(f, initialGuess, jac=Df, method='hybr')  # set root finding algorithm
    if diagnose:
        return solution  # return the entire solution object including iterations and diagnostics
    else:
        return solution.x  # return only the solution vector


def full_newton(f, Df, x0, maxDefect=1e-13):
    """A full Newton based root finding algorithm"""

    def is_singular(matrix, rank):
        """Returns true if the derivative becomes singular for any reason"""
        return np.isnan(matrix).any() or np.isinf(matrix).any() or np.linalg.matrix_rank(matrix) < rank

    fDim = len(x0)  # dimension of the domain/image of f
    maxIterate = 100

    if not is_vector(x0):  # an array whose columns are initial guesses
        print('not implemented yet')

    else:  # x0 is a single initial guess
        # initialize iteration
        x = x0.copy()
        y = f(x)
        Dy = Df(x)
        iDefect = np.linalg.norm(y)  # initialize defect
        iIterate = 1
        while iDefect > maxDefect and iIterate < maxIterate and not is_singular(Dy, fDim):
            if fDim == 1:
                x -= y / Dy
            else:
                x -= np.linalg.solve(Dy, y)  # update x

            y = f(x)  # update f(x)
            print(y)
            Dy = Df(x)  # update Df(x)
            iDefect = np.linalg.norm(y)  # initialize defect
            print(iDefect)
            iIterate += 1

        if iDefect < maxDefect:
            return x
        else:
            print('Newton failed to converge')
            return np.nan


PARAMETER_NAMES = ['ell', 'delta', 'theta', 'hillCoefficient']  # ordered list of HillComponent parameter names


class HillComponent:
    """A component of a Hill system of the form ell + delta*H(x; ell, delta, theta, n) where H is an increasing or decreasing Hill function.
    Any of these parameters can be considered as a fixed value for a Component or included in the callable variables. The
    indices of the edges associated to ell, and delta are different than those associated to theta."""

    def __init__(self, interactionSign, **kwargs):
        """A Hill function with parameters [ell, delta, theta, n] of InteractionType in {-1, 1} to denote H^-, H^+ """
        # TODO: Class constructor should not do work!

        self.sign = interactionSign
        self.parameterValues = np.zeros(4)  # initialize vector of parameter values
        parameterNames = PARAMETER_NAMES.copy()  # ordered list of possible parameter names
        parameterCallIndex = {parameterNames[j]: j for j in range(4)}  # calling index for parameter by name
        for parameterName, parameterValue in kwargs.items():
            setattr(self, parameterName, parameterValue)  # fix input parameter
            self.parameterValues[
                parameterCallIndex[parameterName]] = parameterValue  # update fixed parameter value in evaluation vector
            del parameterCallIndex[parameterName]  # remove fixed parameter from callable list

        self.variableParameters = list(parameterCallIndex.keys())  # set callable parameters
        self.parameterCallIndex = list(parameterCallIndex.values())  # get indices for callable parameters
        self.fixedParameter = [parameterName for parameterName in parameterNames if
                               parameterName not in self.variableParameters]
        #  set callable parameter name functions
        for idx in range(len(self.variableParameters)):
            self.add_parameter_call(self.variableParameters[idx], idx)

    def __iter__(self):
        """Make iterable"""
        yield self

    def add_parameter_call(self, parameterName, parameterIndex):
        """Adds a call by name function for variable parameters to a HillComponent instance"""

        def call_function(self, parameter):
            """returns a class method which has the given parameter name. This method slices the given index out of a
            variable parameter vector"""
            return parameter[parameterIndex]

        setattr(HillComponent, parameterName, call_function)  # set dynamic method name

    def curry_parameters(self, parameter):
        """Returns a parameter evaluation vector in R^4 with fixed and variable parameters indexed properly"""
        parameterEvaluation = self.parameterValues.copy()  # get a mutable copy of the fixed parameter values
        parameterEvaluation[self.parameterCallIndex] = parameter  # slice passed parameter vector into callable slots
        return parameterEvaluation

    def __call__(self, x, parameter=np.array([])):
        """Evaluation method for a Hill component function instance"""

        # TODO: Handle the case that negative x values are passed into this function.

        ell, delta, theta, hillCoefficient = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        # compute powers of x and theta only once.
        xPower = x ** hillCoefficient
        thetaPower = theta ** hillCoefficient  # compute theta^hillCoefficient only once

        # evaluation rational part of the Hill function
        if self.sign == 1:
            evalRational = xPower / (xPower + thetaPower)
        elif self.sign == -1:
            evalRational = thetaPower / (xPower + thetaPower)
        return ell + delta * evalRational

    def __repr__(self):
        """Return a canonical string representation of a Hill component"""

        reprString = 'Hill Component: \n' + 'sign = {0} \n'.format(self.sign)
        for parameterName in ['ell', 'delta', 'theta', 'hillCoefficient']:
            if parameterName not in self.variableParameters:
                reprString += parameterName + ' = {0} \n'.format(getattr(self, parameterName))
        reprString += 'Variable Parameters: {' + ', '.join(self.variableParameters) + '}\n'
        return reprString

    def dx(self, x, parameter=np.array([])):
        """Evaluate the derivative of a Hill component with respect to x"""

        ell, delta, theta, hillCoefficient = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        # compute powers of x and theta only once.
        thetaPower = theta ** hillCoefficient
        xPowerSmall = x ** (hillCoefficient - 1)  # compute x^{hillCoefficient-1}
        xPower = xPowerSmall * x
        return self.sign * hillCoefficient * delta * thetaPower * xPowerSmall / ((thetaPower + xPower) ** 2)

    def dx2(self, x, parameter=np.array([])):
        """Evaluate the second derivative of a Hill component with respect to x"""

        ell, delta, theta, hillCoefficient = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        # compute powers of x and theta only once.
        thetaPower = theta ** hillCoefficient
        xPowerSmall = x ** (hillCoefficient - 2)  # compute x^{hillCoefficient-1}
        xPower = xPowerSmall * x ** 2
        return self.sign * hillCoefficient * delta * thetaPower * xPowerSmall * (
                (hillCoefficient - 1) * thetaPower - (hillCoefficient + 1) * xPower) / ((thetaPower + xPower) ** 3)

    def diff(self, diffIndex, x, parameter=np.array([])):
        """Evaluate the derivative of a Hill component with respect to a parameter at the specified local index.
        The parameter must be a variable parameter for the HillComponent."""

        diffParameter = self.variableParameters[diffIndex]  # get the name of the differentiation variable

        if diffParameter == 'ell':
            return 1.
        else:
            ell, delta, theta, hillCoefficient = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters
            xPower = x ** hillCoefficient

        if diffParameter == 'delta':
            thetaPower = theta ** hillCoefficient  # compute theta^hillCoefficient only once
            if self.sign == 1:
                dH = xPower / (thetaPower + xPower) ** 2
            else:
                dH = thetaPower / (thetaPower + xPower) ** 2

        elif diffParameter == 'theta':
            thetaPowerSmall = theta ** (hillCoefficient - 1)  # compute power of theta only once
            thetaPower = theta * thetaPowerSmall
            dH = self.sign(-delta * hillCoefficient * xPower * thetaPowerSmall) / ((thetaPower + xPower) ** 2)

        elif diffParameter == 'hillCoefficient':
            thetaPower = theta ** hillCoefficient
            dH = self.sign * delta * xPower * thetaPower * log(x / theta) / ((thetaPower + xPower) ** 2)

        return dH


    def diff2(self, diffIndex, x, parameter=np.array([])):
        """Evaluate the derivative of a Hill component with respect to a parameter at the specified local index.
        The parameter must be a variable parameter for the HillComponent."""

        # ordering of the variables decrease options
        if diffIndex[0]>diffIndex[1]:
            diffIndex = diffIndex[[1, 0]]

        diffParameter0 = self.variableParameters[diffIndex[0]]  # get the name of the differentiation variable
        diffParameter1 = self.variableParameters[diffIndex[1]]  # get the name of the differentiation variable

        if diffParameter0 == 'ell' or diffParameter1 == 'ell':
            return 0.
        else:
            ell, delta, theta, hillCoefficient = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters
            xPower = x ** hillCoefficient

        if diffParameter0 == 'theta' and diffParameter1 == 'theta':
            thetaPower_der_der = theta ** (hillCoefficient - 2)
            thetaPower_der = theta * thetaPower_der_der  # compute power of theta only once
            thetaPower = theta * thetaPower_der

        else:
            if diffParameter0 == 'theta' or diffParameter1 == 'theta':
                thetaPower_der = theta ** (hillCoefficient - 1)  # compute power of theta only once
                thetaPower = theta * thetaPower_der
            else:
                thetaPower = theta ** hillCoefficient

        if diffParameter0 == 'delta':
            if diffParameter1 == 'delta':
                return 0.
            if diffParameter1 == 'theta':
                dH = self.sign(-1 * hillCoefficient * xPower * thetaPower_der) / ((thetaPower + xPower) ** 2)
            if diffParameter1 == 'hillCoefficient':
                dH = self.sign * xPower * thetaPower * log(x / theta) / ((thetaPower + xPower) ** 2)

        elif diffParameter0 == 'theta':
            if diffParameter1 == 'theta':
                dH = self.sign * -delta * hillCoefficient * xPower * (thetaPower_der_der * ((thetaPower + xPower) ** 2) - thetaPower_der * 2 * (thetaPower + xPower) * thetaPower_der )/ ((thetaPower + xPower) ** 4)
            if diffParameter1 == 'hillCoefficient':
                dH = self.sign * delta * xPower * ( (thetaPower_der * log(x / theta) + thetaPower * (log(x) - 1/theta))
                                                    -  thetaPower_der * 2 * (thetaPower + xPower) * thetaPower_der )\
                     / ((thetaPower + xPower) ** 4)
                #dH = self.sign * -delta * hillCoefficient * xPower * thetaPowerSmall / ((thetaPower + xPower) ** 2)

        elif diffParameter0 == 'hillCoefficient':
            # then diffParameter1 = 'hillCoefficient'
            dH = self.sign * delta / ((thetaPower + xPower) ** 4) * (
                    log(x * theta) * log (x/theta) *( thetaPower + xPower) ** 2 -
                    log(x / theta) * 2 * (thetaPower + xPower) * (thetaPower*log(theta) + xPower * log(x))
                    )

        return dH

    def dxdiff(self, diffIndex, x, parameter=np.array([])):
        """Evaluate the derivative of a Hill component with respect to the state variable and a parameter at the specified
        local index.
        The parameter must be a variable parameter for the HillComponent."""

        diffParameter = self.variableParameters[diffIndex]  # get the name of the differentiation variable

        if diffParameter == 'ell':
            return 0.
        else:
            ell, delta, theta, hillCoefficient = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters
            xPower = x ** hillCoefficient
            xPower_der = hillCoefficient * x ** (hillCoefficient - 1)

        if diffParameter == 'delta':
            thetaPower = theta ** hillCoefficient  # compute theta^hillCoefficient only once
            if self.sign == 1:
                dH = xPower / (thetaPower + xPower) ** 2
                ddH = (xPower_der * ((thetaPower + xPower) ** 2) - xPower * 2 * (thetaPower + xPower) * xPower_der) / (thetaPower + xPower) ** 4
            else:
                dH = thetaPower / (thetaPower + xPower) ** 2
                ddH = thetaPower * 2 * (thetaPower + xPower) * xPower_der / (thetaPower + xPower) ** 4

        elif diffParameter == 'theta':
            thetaPowerSmall = theta ** (hillCoefficient - 1)  # compute power of theta only once
            thetaPower = theta * thetaPowerSmall
            dH = self.sign * (-delta * hillCoefficient * xPower * thetaPowerSmall) / ((thetaPower + xPower) ** 2)
            ddH = self.sign * (-delta * hillCoefficient * thetaPowerSmall) * \
                (xPower_der * ((thetaPower + xPower) ** 2) - xPower * 2 *
                 (thetaPower + xPower) * xPower_der) / (thetaPower + xPower) ** 4

        elif diffParameter == 'hillCoefficient':
            thetaPower = theta ** hillCoefficient
            dH = self.sign * delta * xPower * thetaPower * log(x / theta) / ((thetaPower + xPower) ** 2)
            ddH = self.sign * delta * thetaPower * ((1/x * xPower * log(1 / theta) +
                                                      xPower_der * log(x / theta)) * ((thetaPower + xPower) ** 2) -
                                                    xPower * 2 * (thetaPower + xPower) * xPower_der) / (thetaPower + xPower) ** 4

        return ddH


    def dn(self, x, parameter=np.array([])):
        """Returns the derivative of a Hill component with respect to n. """

        ell, delta, theta, hillCoefficient = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        # compute powers of x and theta only once.
        xPower = x ** hillCoefficient
        thetaPower = theta ** hillCoefficient
        return self.sign * delta * xPower * thetaPower * log(x / theta) / ((thetaPower + xPower) ** 2)

    def dndx(self, x, parameter=np.array([])):
        """Returns the mixed partials of a Hill component with respect to n and x"""

        ell, delta, theta, hillCoefficient = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        # compute powers of x and theta only once.
        thetaPower = theta ** hillCoefficient
        xPowerSmall = x ** (hillCoefficient - 1)  # compute x^{hillCoefficient-1}
        xPower = xPowerSmall * x
        return self.sign * delta * thetaPower * xPowerSmall * (
                hillCoefficient * (thetaPower - xPower) * log(x / theta) + thetaPower + xPower) / (
                       (thetaPower + xPower) ** 3)

    def image(self, parameter=None):
        """Return the range of this HillComponent given by (ell, ell+delta)"""

        if 'ell' in self.variableParameters:
            ell = self.ell(parameter)
        else:
            ell = self.ell

        if 'delta' in self.variableParameters:
            delta = self.delta(parameter)
        else:
            delta = self.delta

        return np.array([ell, ell + delta])


class HillCoordinate:
    """Define a coordinate of the vector field for a Hill system. This is a scalar equation taking the form
    x' = -gamma*x + p(H_1, H_2,...,H_k) where each H_i is a Hill function depending on x_i which is a state variable
    which regulates x"""

    def __init__(self, parameter, interactionSign, interactionType, interactionIndex, gamma=np.nan):
        """Hill Coordinate instantiation with the following syntax:
        INPUTS:
            gamma - (float) decay rate for this coordinate or NaN if gamma is a variable parameter which is callable as
                the first component of the parameter variable vector.
            parameter - (numpy array) A K-by-4 array of Hill component parameters with rows of the form [ell, delta, theta, hillCoefficient]
                Entries which are NaN are variable parameters which are callable in the function and all derivatives.
            interactionSign - (list) A vector in F_2^K carrying the sign type for each Hill component
            interactionType - (list) A vector describing the interaction type of the interaction function specified as an integer partition of K
            interactionIndex - (list) A length K+1 vector of global state variable indices. interactionIndex[0] is the global index
                for this coordinate and interactionIndex[1:] the indices of the K incoming interacting nodes"""

        # TODO: 1. Class constructor should not do work!
        #       2. Handing local vs global indexing of state variable vectors should be moved to the HillModel class instead of this class.
        self.gammaIsVariable = np.isnan(gamma)
        if ~np.isnan(gamma):
            self.gamma = gamma  # set fixed linear decay
        self.parameterValues = parameter  # initialize array of fixed parameter values
        self.nComponent = len(interactionSign)  # number of interaction nodes
        self.components = self.set_components(parameter, interactionSign)
        self.index = interactionIndex[0]  # Define this coordinate's global index
        self.interactionIndex = interactionIndex[1:]  # Vector of global interaction variable indices
        self.interactionType = interactionType  # specified as an integer partition of K
        self.summand = self.set_summand()
        if self.nComponent == 1:  # Coordinate has a single HillComponent
            self.nVarByComponent = list(
                map(lambda j: np.count_nonzero(np.isnan(self.parameterValues)), range(self.nComponent)))
        else:  # Coordinate has multiple HillComponents
            self.nVarByComponent = list(
                map(lambda j: np.count_nonzero(np.isnan(self.parameterValues[j, :])), range(self.nComponent)))
        # endpoints for concatenated parameter vector by coordinate
        self.variableIndexByComponent = np.cumsum([self.gammaIsVariable] + self.nVarByComponent)
        # endpoints for concatenated parameter vector by coordinate. This is a
        # vector of length K+1. The kth component parameters are the slice variableIndexByComponent[k:k+1] for k = 0...K-1
        self.nVariableParameter = sum(
            self.nVarByComponent) + self.gammaIsVariable  # number of variable parameters for this coordinate.

    def parse_parameters(self, parameter):
        """Returns the value of gamma and slices of the parameter vector divided by component"""

        # If gamma is not fixed, then it must be the first coordinate of the parameter vector
        if self.gammaIsVariable:
            gamma = parameter[0]
        else:
            gamma = self.gamma
        return gamma, [parameter[self.variableIndexByComponent[j]:self.variableIndexByComponent[j + 1]] for
                       j in range(self.nComponent)]

    def __call__(self, x, parameter=np.array([])):
        """Evaluate the Hill coordinate on a vector of (global) state variables and (local) parameter variables. This is a
        map of the form  g: R^n x R^m ---> R where n is the number of state variables of the Hill model and m is the number
        of variable parameters for this Hill coordinate"""

        # TODO: Currently the input parameter must be a numpy array even if there is only a single parameter.
        if is_vector(x):  # Evaluate coordinate for a single x in R^n
            # slice callable parameters into a list of length K. The j^th list contains the variable parameters belonging to
            # the j^th Hill component.
            gamma, parameterByComponent = self.parse_parameters(parameter)
            hillComponentValues = self.evaluate_components(x, parameter)
            nonlinearTerm = self.interaction_function(hillComponentValues)  # compose with interaction function
            return -gamma * x[self.index] + nonlinearTerm

        # TODO: vectorized evaluation is a little bit hacky and should be rewritten to be more efficient
        else:  # vectorized evaluation where x is a matrix of column vectors to evaluate
            return np.array([self(x[:, j], parameter) for j in range(np.shape(x)[1])])

    def evaluate_components(self, x, parameter):
        """Evaluate each HillComponent and return as a vector in R^K"""

        gamma, parameterByComponent = self.parse_parameters(parameter)
        return np.array(
            list(map(lambda H, idx, parm: H(x[idx], parm), self.components, self.interactionIndex,
                     parameterByComponent)))  # evaluate Hill components

    def evaluate_summand(self, x, parameter, m=None):
        """Evaluate the Hill summands"""

        gamma, parameterByComponent = self.parse_parameters(parameter)

        if m is None:  # Return all summand evaluations as a vector in R^q
            return np.array([self.evaluate_summand(x, parameter, m=summandIdx) for summandIdx in range(len(self.summand))])
        else:
            parm = parameterByComponent[m]
            interactionIndex = [self.interactionIndex[k] for k in self.summand[m-1]]
            componentValues = np.array(
                list(map(lambda k: self.components[k](x[interactionIndex[k]], parm), self.summand[m-1])))  # evaluate Hill components
            return np.sum(componentValues)


    def interaction_function(self, componentValues):
        """Evaluate the polynomial interaction function at a parameter in (0,inf)^{K}"""

        # return np.prod([sum([parm[idx] for idx in sumList]) for sumList in self.summand])

        if len(self.summand) == 1:  # this is the all sum interaction type
            return np.sum(componentValues)
        else:


            return




    def summand_index(self, componentIdx):
        """Returns the summand index of a component index. This is a map of the form, I : {1,...,K} --> {1,...,q}"""

        return self.summand.index(filter(lambda L: componentIdx in L, self.summand).__next__())

    def dx(self, x, parameter=np.array([])):
        """Return the derivative (gradient vector) evaluated at x in R^n and p in R^m as a row vector"""

        gamma, parameterByComponent = self.parse_parameters(parameter)
        dim = len(x)  # dimension of vector field
        Df = np.zeros(dim, dtype=float)
        xLocal = x[
            self.interactionIndex]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{K}
        diffInteraction = self.diff_interaction(xLocal)  # evaluate outer term in chain rule
        DHillComponent = np.array(
            list(map(lambda H, x, parm: H.dx(x, parm), self.components, xLocal,
                     parameterByComponent)))  # evaluate inner term in chain rule
        Df[
            self.interactionIndex] = diffInteraction * DHillComponent  # evaluate gradient of nonlinear part via chain rule
        Df[self.index] -= gamma  # Add derivative of linear part to the gradient at this HillCoordinate
        return Df

    def diff(self, diffIndex, x, parameter=np.array([])):
        """Evaluate the derivative of a Hill coordinate with respect to a parameter at the specified local index.
           The parameter must be a variable parameter for one or more HillComponents."""

        if self.gammaIsVariable and diffIndex == 0:  # derivative with respect to decay parameter
            return -1
        else:  # First obtain a local index in the HillComponent for the differentiation variable
            diffComponent = np.searchsorted(self.variableIndexByComponent,
                                            diffIndex + 0.5) - 1  # get the component which contains the differentiation variable. Adding 0.5
            # makes the returned value consistent in the case that the diffIndex is an endpoint of the variable index list
            diffParameterIndex = diffIndex - self.variableIndexByComponent[
                diffComponent]  # get the local parameter index in the HillComponent for the differentiation variable

            # Now evaluate the derivative through the HillComponent and embed into tangent space of R^n
            gamma, parameterByComponent = self.parse_parameters(parameter)
            xLocal = x[
                self.interactionIndex]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{K}
            diffInteraction = self.diff_interaction(xLocal)[diffComponent]  # evaluate outer term in chain rule
            # TODO: diffInteraction should allow calls to individual components. Currently it always returns the entire vector of
            #       derivatives for every component.
            dH = self.components[diffComponent].diff(diffParameterIndex, xLocal[diffComponent], parameterByComponent[
                diffComponent])  # evaluate inner term in chain rule
            return diffInteraction * dH

    def dn(self, x, parameter=np.array([])):
        """Evaluate the derivative of a HillCoordinate with respect to the vector of Hill coefficients as a row vector.
        Evaluation requires specifying x in R^n and p in R^m. This method does not assume that all HillCoordinates have
        a uniform Hill coefficient. If this is the case then the scalar derivative with respect to the Hill coefficient
        should be the sum of the gradient vector returned"""

        warnings.warn("The .dn method for HillComponents and HillCoordinates is deprecated. Use the .diff method instead.")
        gamma, parameterByComponent = self.parse_parameters(parameter)
        dim = len(x)  # dimension of vector field
        df_dn = np.zeros(dim, dtype=float)
        xLocal = x[
            self.interactionIndex]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{K}
        diffInteraction = self.diff_interaction(xLocal)  # evaluate outer term in chain rule
        dHillComponent_dn = np.array(
            list(map(lambda H, x, parm: H.dn(x, parm), self.components, xLocal,
                     parameterByComponent)))  # evaluate inner term in chain rule
        df_dn[
            self.interactionIndex] = diffInteraction * dHillComponent_dn  # evaluate gradient of nonlinear part via chain rule
        return df_dn

    def set_components(self, parameter, interactionSign):
        """Return a list of Hill components for this Hill coordinate"""

        def row2dict(row):
            """convert ordered row of parameter matrix to kwarg"""
            return {PARAMETER_NAMES[j]: row[j] for j in range(4) if
                    not np.isnan(row[j])}

        if self.nComponent == 1:
            return [HillComponent(interactionSign[0], **row2dict(parameter))]
        else:
            return [HillComponent(interactionSign[k], **row2dict(parameter[k, :])) for k in
                    range(self.nComponent)]  # list of Hill components

    def set_summand(self):
        """Return the list of lists containing the summand indices defined by the interaction type.
        EXAMPLE:
            interactionType = [2,1,3,1] returns the index partition [[0,1], [2], [3,4,5], [6]]"""

        sumEndpoints = np.insert(np.cumsum(self.interactionType), 0,
                                 0)  # summand endpoint indices including initial zero
        localIndex = list(range(self.nComponent))
        return [localIndex[sumEndpoints[i]:sumEndpoints[i + 1]] for i in range(len(self.interactionType))]

    def diff_interaction(self, xLocal):
        """Dummy functionality for evaluating the derivative of the interaction function"""

        if len(self.interactionType) == 1:
            return np.ones(len(xLocal))
        else:
            raise KeyboardInterrupt

    def summand_map(self, x, parameter=np.array([])):
        """Apply the summand map of the form: f_i : R^n ---> p = (p1,...,p_q) which is a partial composition
        with the interaction function."""

        H = self.__call__(x, parameter)
        return [sum(H[indexSet]) for indexSet in self.summand]

    def eq_interval(self, parameter=None):
        """Return a closed interval which must contain the projection of any equilibrium onto this coordinate"""

        if parameter is None:  # all parameters are fixed
            # TODO: This should only require all ell, delta, and gamma variables to be fixed.
            minInteraction = self.interaction_function([H.ell for H in self.components]) / self.gamma
            maxInteraction = self.interaction_function([H.ell + H.delta for H in self.components]) / self.gamma

        else:  # some variable parameters are passed in a vector containing all parameters for this Hill Coordinate
            gamma, parameterByComponent = self.parse_parameters(parameter)
            rectangle = np.row_stack(list(map(lambda H, parm: H.image(parm), self.components, parameterByComponent)))
            minInteraction = self.interaction_function(rectangle[:, 0]) / gamma  # min(f) = p(ell_1, ell_2,...,ell_K)
            maxInteraction = self.interaction_function(
                rectangle[:, 1]) / gamma  # max(f) = p(ell_1 + delta_1,...,ell_K + delta_K)
        return [minInteraction, maxInteraction]


class HillModel:
    """Define a Hill model as a vector field describing the derivatives of all state variables. The i^th coordinate
    describes the derivative of the state variable, x_i, as a function of x_i and its incoming interactions, {X_1,...,X_{K_i}}.
    This function is always a linear decay and a nonlinear interaction defined by a polynomial composition of Hill
    functions evaluated at the interactions. The vector field is defined coordinate-wise as a vector of HillCoordinate instances"""

    def __init__(self, gamma, parameter, interactionSign, interactionType, interactionIndex):
        """Class constructor which has the following syntax:
        INPUTS:
            gamma - A vector in R^n of linear decay rates
            parameter - A length n list of K_i-by-4 parameter arrays
            interactionSign - A length n list of vectors in F_2^{K_i}
            interactionType - A length n list of length q_i lists describing an integer partitions of K_i
            interactionIndex - A length n list whose i^th element is a length K_i list of global indices for the i^th incoming interactions"""

        # TODO: Class constructor should not do work!

        self.dimension = len(gamma)  # Dimension of vector field i.e. n
        self.coordinates = [HillCoordinate(parameter[j], interactionSign[j],
                                           interactionType[j], [j] + interactionIndex[j], gamma=gamma[j]) for j in
                            range(self.dimension)]
        # A list of HillCoordinates specifying each coordinate of the vector field
        self.nVarByCoordinate = [fi.nVariableParameter for fi in
                                 self.coordinates]  # number of variable parameters by coordinate
        self.variableIndexByCoordinate = np.insert(np.cumsum(self.nVarByCoordinate), 0,
                                                   0)  # endpoints for concatenated parameter vector by coordinate
        self.nVariableParameter = sum(self.nVarByCoordinate)  # number of variable parameters for this HillModel

    def parse_parameter(self, *parameter):
        """Default parameter parsing if input is a single vector simply returns the same vector. Otherwise, it assumes
        input parameters are provided in order and concatenates into a single vector. This function is included in
        function calls so that subclasses can redefine function calls with customized parameters and overload this
        function as needed. Overloaded versions should take a variable number of numpy arrays as input and must always
        return a single numpy vector as output."""

        if parameter:
            return np.concatenate(parameter)
        else:
            return np.array([])

    def unpack_variable_parameters(self, parameter):
        """Unpack a parameter vector for the HillModel into component vectors for each distinct coordinate"""

        return [parameter[self.variableIndexByCoordinate[j]:self.variableIndexByCoordinate[j + 1]] for
                j in range(self.dimension)]

    def __call__(self, x, *parameter):
        """Evaluate the vector field defined by this HillModel instance. This is a function of the form
        f: R^n x R^{m_1} x ... x R^{m_n} ---> R^n where the j^th Hill coordinate has m_j variable parameters. The syntax
        is f(x,p) where p = (p_1,...,p_n) is a variable parameter vector constructed by ordered concatenation of vectors
        of the form p_j = (p_j1,...,p_jK) which is also an ordered concatenation of the variable parameters associated to
        the K-HillComponents for the j^th HillCoordinate."""

        parameter = self.parse_parameter(*parameter)  # concatenate all parameters into a vector
        parameterByCoordinate = self.unpack_variable_parameters(parameter)  # unpack variable parameters by component
        if is_vector(x):  # input a single vector in R^n
            return np.array(list(map(lambda f_i, parm: f_i(x, parm), self.coordinates, parameterByCoordinate)))
        else:  # vectorized input
            return np.row_stack(list(map(lambda f_i, parm: f_i(x, parm), self.coordinates, parameterByCoordinate)))

    def dx(self, x, *parameter):
        """Return the derivative (Jacobian) of the HillModel vector field with respect to x.
        NOTE: This function is not vectorized. It assumes x is a single vector in R^n."""

        parameter = self.parse_parameter(*parameter)  # concatenate all parameters into a vector
        parameterByCoordinate = self.unpack_variable_parameters(parameter)  # unpack variable parameters by component
        return np.vstack(list(map(lambda f_i, parm: f_i.dx(x, parm), self.coordinates,
                                  parameterByCoordinate)))  # return a vertical stack of gradient (row) vectors

    def dn(self, x, *parameter):
        """Return the derivative (Jacobian) of the HillModel vector field with respect to n assuming n is a VECTOR
        of Hill Coefficients. If n is uniform across all HillComponents, then the derivative is a gradient vector obtained
        by summing this Jacobian along rows.
        NOTE: This function is not vectorized. It assumes x is a single vector in R^n."""

        parameter = self.parse_parameter(*parameter)  # concatenate all parameters into a vector
        parameterByCoordinate = self.unpack_variable_parameters(parameter)  # unpack variable parameters by component
        return np.vstack(list(map(lambda f_i, parm: f_i.dn(x, parm), self.coordinates,
                                  parameterByCoordinate)))  # return a vertical stack of gradient (row) vectors

    def find_equilibria(self, gridDensity, *parameter, uniqueRootDigits=7):
        """Return equilibria for the Hill Model by uniformly sampling for initial conditions and iterating a Newton variant.
        INPUT:
            *parameter - (numpy vectors) Evaluations for variable parameters to use for evaluating the root finding algorithm
            gridDensity - (int) density to sample in each dimension.
            uniqueRootDigits - (int) Number of digits to use for distinguishing between floats."""

        # TODO: Include root finding method as kwarg

        # parameter = self.parse_parameter(*parameter)  # concatenate all parameters into a vector
        parameterByCoordinate = self.unpack_variable_parameters(
            self.parse_parameter(*parameter))  # unpack variable parameters by component

        def F(x):
            """Fix parameter values in the zero finding map"""
            return self.__call__(x, *parameter)

        def DF(x):
            """Fix parameter values in the zero finding map derivative"""
            return self.dx(x, *parameter)

        def eq_is_positive(equilibrium):
            """Return true if and only if an equlibrium is positive"""
            return np.all(equilibrium > 0)

        # build a grid of initial data for Newton algorithm
        coordinateIntervals = list(
            map(lambda f_i, parm: np.linspace(*f_i.eq_interval(parm), num=gridDensity), self.coordinates,
                parameterByCoordinate))
        evalGrid = np.meshgrid(*coordinateIntervals)
        X = np.row_stack([G_i.flatten() for G_i in evalGrid])
        solns = list(
            filter(lambda root: root.success and eq_is_positive(root.x), [find_root(F, DF, X[:, j], diagnose=True)
                                                                          for j in
                                                                          range(X.shape[
                                                                                    1])]))  # return equilibria which converged
        equilibria = np.column_stack([root.x for root in solns])  # extra equilibria as vectors in R^n
        equilibria = np.unique(np.round(equilibria, uniqueRootDigits), axis=1)  # remove duplicates
        return np.column_stack([find_root(F, DF, equilibria[:, j]) for j in
                                range(np.shape(equilibria)[1])])  # Iterate Newton again to regain lost digits


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
                         interactionIndex)  # define HillModel for toggle switch
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

# def unit_phase_condition(v):
#     """Evaluate defect for unit vector zero map of the form: U(v) = ||v|| - 1"""
#     return np.linalg.norm(v) - 1
#
#
# def diff_unit_phase_condition(v):
#     """Evaluate the derivative of the unit phase condition function"""
#     return v / np.linalg.norm(v)


# ## toggle switch plus
# # set some parameters to test
# decay = np.array([np.nan, np.nan], dtype=float)  # gamma
# f1parameter = np.array([[np.nan, np.nan, np.nan, np.nan] for j in range(2)], dtype=float)  # all variable parameters
# f2parameter = np.array([[np.nan, np.nan, np.nan, np.nan] for j in range(2)], dtype=float)  # all variable parameters
# parameter = [f1parameter, f2parameter]
# interactionSigns = [[1, -1], [1, -1]]
# interactionTypes = [[2], [2]]
# interactionIndex = [[0, 1], [1, 0]]
# tsPlus = HillModel(decay, parameter, interactionSigns, interactionTypes,
#                    interactionIndex)  # define HillModel for toggle switch plus
