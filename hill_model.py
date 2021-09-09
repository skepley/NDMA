"""
Classes and methods for constructing, evaluating, and doing parameter continuation of Hill Models

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 2/29/20; Last revision: 6/23/20
"""
import numpy as np
import warnings
import matplotlib.pyplot as plt
from itertools import product, permutations
from scipy import optimize, linalg
from numpy import log
import textwrap

# ignore overflow and division by zero warnings:
np.seterr(over='ignore', invalid='ignore')


def npA(size, dim=2):
    """Return a random numpy array for testing"""
    A = np.random.randint(1, 10, dim * [size])
    return np.asarray(A, dtype=float)


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
            Dy = Df(x)  # update Df(x)
            iDefect = np.linalg.norm(y)  # initialize defect
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

    def __call__(self, x, *parameter):
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
        for parameterName in PARAMETER_NAMES:
            if parameterName not in self.variableParameters:
                reprString += parameterName + ' = {0} \n'.format(getattr(self, parameterName))
        reprString += 'Variable Parameters: {' + ', '.join(self.variableParameters) + '}\n'
        return reprString

    def dx(self, x, parameter):
        """Evaluate the derivative of a Hill component with respect to x"""

        ell, delta, theta, hillCoefficient = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        # compute powers of x and theta only once.
        thetaPower = theta ** hillCoefficient
        xPowerSmall = x ** (hillCoefficient - 1)  # compute x^{hillCoefficient-1}
        xPower = xPowerSmall * x
        return self.sign * hillCoefficient * delta * thetaPower * xPowerSmall / ((thetaPower + xPower) ** 2)

    def dx2(self, x, parameter):
        """Evaluate the second derivative of a Hill component with respect to x"""

        ell, delta, theta, hillCoefficient = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        # compute powers of x and theta only once.
        thetaPower = theta ** hillCoefficient
        xPowerSmall = x ** (hillCoefficient - 2)  # compute x^{hillCoefficient-1}
        xPower = xPowerSmall * x ** 2
        return self.sign * hillCoefficient * delta * thetaPower * xPowerSmall * (
                (hillCoefficient - 1) * thetaPower - (hillCoefficient + 1) * xPower) / ((thetaPower + xPower) ** 3)

    def diff(self, x, parameter, diffIndex):
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
                dH = xPower / (thetaPower + xPower)
            else:
                dH = thetaPower / (thetaPower + xPower)

        elif diffParameter == 'theta':
            thetaPowerSmall = theta ** (hillCoefficient - 1)  # compute power of theta only once
            thetaPower = theta * thetaPowerSmall
            dH = self.sign * (-delta * hillCoefficient * xPower * thetaPowerSmall) / ((thetaPower + xPower) ** 2)

        elif diffParameter == 'hillCoefficient':
            thetaPower = theta ** hillCoefficient
            dH = self.sign * delta * xPower * thetaPower * (log(x) - log(theta)) / ((thetaPower + xPower) ** 2)

        return dH

    def diff2(self, x, parameter, diffIndex):
        """Evaluate the derivative of a Hill component with respect to a parameter at the specified local index.
        The parameter must be a variable parameter for the HillComponent."""

        # ordering of the variables decrease options
        if diffIndex[0] > diffIndex[1]:
            diffIndex = diffIndex[[1, 0]]

        diffParameter0 = self.variableParameters[diffIndex[0]]  # get the name of the differentiation variable
        diffParameter1 = self.variableParameters[diffIndex[1]]  # get the name of the differentiation variable

        if diffParameter0 == 'ell':
            return 0.
        else:
            ell, delta, theta, hillCoefficient = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters

        # precompute some powers
        # this is the only power of x we will need
        xPower = x ** hillCoefficient
        # here we check which powers of theta we will need and compute them
        if diffParameter0 == 'theta' and diffParameter1 == 'theta':
            thetaPower_minusminus = theta ** (hillCoefficient - 2)
            thetaPower_minus = theta * thetaPower_minusminus  # compute power of theta only once
            thetaPower = theta * thetaPower_minus

        else:
            if diffParameter0 == 'theta' or diffParameter1 == 'theta':
                thetaPower_minus = theta ** (hillCoefficient - 1)  # compute power of theta only once
                thetaPower = theta * thetaPower_minus
            else:
                thetaPower = theta ** hillCoefficient

        if diffParameter0 == 'delta':
            if diffParameter1 == 'delta':
                return 0.
            if diffParameter1 == 'theta':
                dH = self.sign * -1 * hillCoefficient * xPower * thetaPower_minus / ((thetaPower + xPower) ** 2)
            if diffParameter1 == 'hillCoefficient':
                dH = self.sign * xPower * thetaPower * (log(x) - log(theta)) / ((thetaPower + xPower) ** 2)

        elif diffParameter0 == 'theta':
            if diffParameter1 == 'theta':
                dH = self.sign * -delta * hillCoefficient * xPower * (thetaPower_minusminus * (hillCoefficient - 1) *
                                                                      (
                                                                              thetaPower + xPower) - thetaPower_minus * 2 * hillCoefficient *
                                                                      thetaPower_minus) / ((thetaPower + xPower) ** 3)
            if diffParameter1 == 'hillCoefficient':
                dH = - self.sign * delta * xPower * thetaPower_minus * \
                     (hillCoefficient * (thetaPower - xPower) * (log(theta) - log(x)) - thetaPower - xPower) \
                     / ((thetaPower + xPower) ** 3)
                # dH = self.sign * -delta * hillCoefficient * xPower * thetaPowerSmall / ((thetaPower + xPower) ** 2)

        elif diffParameter0 == 'hillCoefficient':
            # then diffParameter1 = 'hillCoefficient'
            dH = self.sign * delta * (thetaPower * xPower * (thetaPower - xPower) * (log(theta) - log(x)) ** 2) / \
                 (thetaPower + xPower) ** 3

        return dH

    def dxdiff(self, x, parameter, diffIndex):
        """Evaluate the derivative of a Hill component with respect to the state variable and a parameter at the specified
        local index.
        The parameter must be a variable parameter for the HillComponent."""

        diffParameter = self.variableParameters[diffIndex]  # get the name of the differentiation variable

        if diffParameter == 'ell':
            return 0.
        else:
            ell, delta, theta, hillCoefficient = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters
            xPower_der = x ** (hillCoefficient - 1)
            xPower = x * xPower_der

        if diffParameter == 'delta':
            thetaPower = theta ** hillCoefficient  # compute theta^hillCoefficient only once
            ddH = self.sign * hillCoefficient * thetaPower * xPower_der / (thetaPower + xPower) ** 2

        elif diffParameter == 'theta':
            thetaPowerSmall = theta ** (hillCoefficient - 1)  # compute power of theta only once
            thetaPower = theta * thetaPowerSmall
            ddH = self.sign * delta * hillCoefficient ** 2 * thetaPowerSmall * xPower_der * \
                  (xPower - thetaPower) / (thetaPower + xPower) ** 3

        elif diffParameter == 'hillCoefficient':
            thetaPower = theta ** hillCoefficient
            ddH = self.sign * delta * thetaPower * xPower_der * (
                    hillCoefficient * (thetaPower - xPower) * (log(x) - log(theta)) + thetaPower + xPower) / (
                          (thetaPower + xPower) ** 3)
        return ddH

    def dx2diff(self, x, parameter, diffIndex):
        """Evaluate the derivative of a Hill component with respect to the state variable and a parameter at the specified
        local index.
        The parameter must be a variable parameter for the HillComponent."""

        diffParameter = self.variableParameters[diffIndex]  # get the name of the differentiation variable

        if diffParameter == 'ell':
            return 0.
        else:
            ell, delta, theta, hillCoefficient = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters

        hill = hillCoefficient

        if diffParameter == 'delta':
            xPower_der_der = x ** (hillCoefficient - 2)
            xPower_der = x ** (hillCoefficient - 1)
            xPower = x * xPower_der
            thetaPower = theta ** hillCoefficient  # compute theta^hillCoefficient only once
            d3H = self.sign * hillCoefficient * thetaPower * xPower_der_der * (
                    (hillCoefficient - 1) * thetaPower - (hillCoefficient + 1) * xPower) / ((thetaPower + xPower) ** 3)

        elif diffParameter == 'theta':
            xPower_derder = x ** (hillCoefficient - 2)
            xPower = x * xPower_derder * x
            x2Power = xPower * xPower
            thetaPower_der = theta ** (hillCoefficient - 1)  # compute power of theta only once
            thetaPower = theta * thetaPower_der
            theta2Power = thetaPower * thetaPower

            d3H = self.sign * hill ** 2 * delta * xPower_derder * thetaPower_der * \
                  (4 * hill * thetaPower * xPower + (-hill + 1) * theta2Power - (hill + 1) * x2Power) / (
                          (thetaPower + xPower) ** 4)

        elif diffParameter == 'hillCoefficient':
            xPower_derder = x ** (hillCoefficient - 2)
            xPower = x * xPower_derder * x
            x2Power = xPower * xPower

            thetaPower = theta ** hillCoefficient
            theta2Power = thetaPower * thetaPower

            d3H = self.sign * delta * thetaPower * xPower_derder * \
                  ((thetaPower + xPower) * ((2 * hill - 1) * thetaPower - (2 * hill + 1) * xPower)
                   - hill * ((hill - 1) * theta2Power - 4 * hill * thetaPower * xPower + (hill + 1)
                             * x2Power) * (log(theta) - log(x))) / ((thetaPower + xPower) ** 4)

        return d3H

    def dxdiff2(self, x, parameter, diffIndex):
        """Evaluate the derivative of a Hill component with respect to a parameter at the specified local index.
        The parameter must be a variable parameter for the HillComponent."""

        # ordering of the variables decrease options
        if diffIndex[0] > diffIndex[1]:
            diffIndex = diffIndex[[1, 0]]

        diffParameter0 = self.variableParameters[diffIndex[0]]  # get the name of the differentiation variable
        diffParameter1 = self.variableParameters[diffIndex[1]]  # get the name of the differentiation variable

        if diffParameter0 == 'ell':
            return 0.
        else:
            ell, delta, theta, hillCoefficient = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters

        hill = hillCoefficient

        # precompute some powers
        # this is the only power of x e will need
        xPower_minus = x ** (hill - 1)
        xPower = x * xPower_minus
        # here we check which powers of theta we will need and compute them
        if diffParameter0 == 'theta' and diffParameter1 == 'theta':
            thetaPower_minusminus = theta ** (hillCoefficient - 2)
            thetaPower_minus = theta * thetaPower_minusminus  # compute power of theta only once
            thetaPower = theta * thetaPower_minus

        else:
            if diffParameter0 == 'theta' or diffParameter1 == 'theta':
                thetaPower_minus = theta ** (hillCoefficient - 1)  # compute power of theta only once
                thetaPower = theta * thetaPower_minus
            else:
                thetaPower = theta ** hillCoefficient

        if diffParameter0 == 'delta':
            if diffParameter1 == 'delta':
                return 0.
            if diffParameter1 == 'theta':
                dH = self.sign * hill ** 2 * thetaPower_minus * xPower_minus * (xPower - thetaPower) / \
                     ((thetaPower + xPower) ** 3)
            if diffParameter1 == 'hillCoefficient':
                dH = self.sign * ((thetaPower * xPower_minus * (-hill * (thetaPower - xPower) * (log(theta) - log(x)) +
                                                                thetaPower + xPower))) / ((thetaPower + xPower) ** 3)

        elif diffParameter0 == 'theta':
            if diffParameter1 == 'theta':
                dH = (self.sign * delta * hill ** 2 * thetaPower_minusminus * xPower_minus * (
                        (hill + 1) * thetaPower ** 2
                        - 4 * hill * thetaPower * xPower + (hill - 1) * xPower ** 2)) / ((thetaPower + xPower) ** 4)
            if diffParameter1 == 'hillCoefficient':
                dH = self.sign * (delta * hill * thetaPower_minus * xPower_minus * (-2 * thetaPower ** 2 +
                                                                                    hill * thetaPower ** 2 - 4 * thetaPower * xPower + xPower ** 2) *
                                  (log(theta) - log(x)) + 2 * xPower ** 2) / ((thetaPower + xPower) ** 4)

        elif diffParameter0 == 'hillCoefficient':
            # then diffParameter1 = 'hillCoefficient'
            dH = self.sign * (delta * thetaPower * xPower_minus * (log(theta) - log(x)) * (-2 * thetaPower ** 2 + hill *
                                                                                           (
                                                                                                   thetaPower ** 2 - 4 * thetaPower * xPower + xPower ** 2) * (
                                                                                                   log(theta) - log(
                                                                                               x)) +
                                                                                           2 * xPower ** 2) / (
                                      (thetaPower + xPower) ** 4))

        return dH

    def dx3(self, x, parameter):
        """Evaluate the second derivative of a Hill component with respect to x"""

        ell, delta, theta, hillCoefficient = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        # compute powers of x and theta only once.
        hill = hillCoefficient
        thetaPower = theta ** hillCoefficient
        theta2Power = thetaPower ** 2
        xPower_der3 = x ** (hill - 3)
        xPower_der2 = x * xPower_der3
        xPower_der = x * xPower_der2  # compute x^{hillCoefficient-1}
        xPower = xPower_der * x
        x2Power = xPower ** 2
        hillsquare = hill ** 2
        return self.sign * (hill * delta * thetaPower * xPower_der3) / ((xPower + thetaPower) ** 4) * \
               (hillsquare * theta2Power - 4 * hillsquare * thetaPower * xPower + hillsquare * x2Power - \
                3 * hill * theta2Power + 2 * theta2Power + 4 * thetaPower * xPower + 3 * hill * x2Power + 2 * x2Power)

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
    """Define a coordinate of the vector field for a Hill system as a function, f : R^K ---> R. If x does not have a nonlinear
     self interaction, then this is a scalar equation taking the form x' = -gamma*x + p(H_1, H_2,...,H_K) where each H_i is a Hill function depending on x_i which is a state variable
    which regulates x. Otherwise, it takes the form, x' = -gamma*x + p(H_1, H_2,...,H_K) where we write x_K = x. """

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
        #       3. There is a lot of redundancy between the "summand" methods and "component" methods. It is stil not clear how the code needs to be refactored.
        self.gammaIsVariable = np.isnan(gamma)
        if ~np.isnan(gamma):
            self.gamma = gamma  # set fixed linear decay
        self.globalStateIndex = sorted(
            list(set(interactionIndex)))  # global index for state variables which this coordinate depends on
        self.dim = len(self.globalStateIndex)  # dimension of state vector input to HillCoordinate
        self.index = interactionIndex[0]  # Define this coordinate's global index
        self.interactionIndex = interactionIndex[1:]  # Vector of global interaction variable indices
        self.parameterValues = parameter  # initialize array of fixed parameter values
        self.nComponent = len(interactionSign)  # number of interaction nodes
        self.components = self.set_components(parameter, interactionSign)
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
            self.nVarByComponent) + int(self.gammaIsVariable)  # number of variable parameters for this coordinate.

    def parse_parameters(self, parameter):
        """Returns the value of gamma and slices of the parameter vector divided by component"""

        # If gamma is not fixed, then it must be the first coordinate of the parameter vector
        if self.gammaIsVariable:
            gamma = parameter[0]
        else:
            gamma = self.gamma
        return gamma, [parameter[self.variableIndexByComponent[j]:self.variableIndexByComponent[j + 1]] for
                       j in range(self.nComponent)]

    def parameter_to_component_index(self, linearIndex):
        """Convert a linear parameter index to an ordered pair, (i, j) where the specified parameter is the j^th variable
         parameter of the i^th Hill component."""

        if self.gammaIsVariable and linearIndex == 0:
            print('component index for a decay parameter is undefined')
            raise KeyboardInterrupt
        componentIndex = np.searchsorted(self.variableIndexByComponent,
                                         linearIndex + 0.5) - 1  # get the component which contains the variable parameter. Adding 0.5
        # makes the returned value consistent in the case that the diffIndex is an endpoint of the variable index list
        parameterIndex = linearIndex - self.variableIndexByComponent[
            componentIndex]  # get the local parameter index in the HillComponent for the variable parameter
        return componentIndex, parameterIndex

    def component_to_parameter_index(self, componentIdx, localIdx):
        """Given an input (i,j), return a linear index for the j^th local parameter of the i^th Hill component"""

        return self.variableIndexByComponent[componentIdx] + localIdx

    def __call__(self, x, parameter):
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

    def __repr__(self):
        """Return a canonical string representation of a Hill coordinate"""

        reprString = 'Hill Coordinate: {0} \n'.format(self.index) + 'Interaction Type: p = ' + (
                '(' + ')('.join(
            [' + '.join(['z_{0}'.format(idx + 1) for idx in summand]) for summand in self.summand]) + ')\n') + (
                             'Components: H = (' + ', '.join(
                         map(lambda i: 'H+' if i == 1 else 'H-', [H.sign for H in self.components])) + ') \n')

        # initialize index strings
        stateIndexString = 'State Variables: x = (x_{0}; '.format(self.index + 1)
        variableIndexString = 'Variable Parameters: lambda = ('
        if self.gammaIsVariable:
            variableIndexString += 'gamma, '

        for k in range(self.nComponent):
            idx = self.interactionIndex[k]
            stateIndexString += 'x_{0}, '.format(idx + 1)
            if self.components[k].variableParameters:
                variableIndexString += ', '.join(
                    [var + '_{0}'.format(idx + 1) for var in self.components[k].variableParameters])
                variableIndexString += ', '

        # remove trailing commas and close brackets
        variableIndexString = variableIndexString[:-2]
        stateIndexString = stateIndexString[:-2]
        variableIndexString += ')\n'
        stateIndexString += ')\n'
        reprString += stateIndexString + '\n          '.join(textwrap.wrap(variableIndexString, 80))
        return reprString

    def evaluate_components(self, x, parameter):
        """Evaluate each HillComponent and return as a vector in R^K"""

        gamma, parameterByComponent = self.parse_parameters(parameter)
        return np.array(
            list(map(lambda H, idx, parm: H(x[idx], parm), self.components, self.interactionIndex,
                     parameterByComponent)))  # evaluate Hill components

    def summand_index(self, componentIdx):
        """Returns the summand index of a component index. This is a map of the form, I : {1,...,K} --> {1,...,q} which
        identifies to which summand the k^th component contributes."""

        return self.summand.index(filter(lambda L: componentIdx in L, self.summand).__next__())

    def evaluate_summand(self, x, parameter, m=None):
        """Evaluate the Hill summands at a given parameter. This is a map taking values in R^q. If m is given in arange(q)
        this returns only the m^th summand."""

        gamma, parameterByComponent = self.parse_parameters(parameter)

        if m is None:  # Return all summand evaluations as a vector in R^q
            return np.array(
                [self.evaluate_summand(x, parameter, m=summandIdx) for summandIdx in range(len(self.summand))])
        else:
            summand = self.summand[m]
            # parmBySummand = [parameterByComponent[k] for k in summand]
            # interactionIndex = [self.interactionIndex[k] for k in summand]
            componentValues = np.array(
                list(map(lambda k: self.components[k](x[self.interactionIndex[k]], parameterByComponent[k]),
                         summand)))  # evaluate Hill components
            return np.sum(componentValues)

    def interaction_function(self, componentValues):
        """Evaluate the polynomial interaction function at a parameter in (0,inf)^{K}"""

        if len(self.summand) == 1:  # this is the all sum interaction type
            return np.sum(componentValues)
        else:
            return np.prod([sum([componentValues[idx] for idx in summand]) for summand in self.summand])

    def diff_interaction(self, x, parameter, diffOrder, diffIndex=None):
        """Return the partial derivative of the specified order for interaction function in the coordinate specified by
        diffIndex. If diffIndex is not specified, it returns the full derivative as a vector with all K partials of
        order diffOrder."""

        def nonzero_index(order):
            """Return the indices for which the given order derivative of an interaction function is nonzero. This happens
            precisely for every multi-index in the tensor for which each component is drawn from a different summand."""

            summandTuples = permutations(self.summand, order)
            summandProducts = []  # initialize cartesian product of all summand tuples
            for tup in summandTuples:
                summandProducts += list(product(*tup))

            return np.array(summandProducts)

        nSummand = len(self.interactionType)  # number of summands
        if diffIndex is None:  # compute the full gradient of p with respect to all components

            if diffOrder == 1:  # compute first derivative of interaction function composed with Hill Components
                if nSummand == 1:  # the all sum special case
                    return np.ones(self.nComponent)
                else:
                    allSummands = self.evaluate_summand(x, parameter)
                    fullProduct = np.prod(allSummands)
                    DxProducts = fullProduct / allSummands  # evaluate all partials only once using q multiplies. The m^th term looks like P/p_m.
                    return np.array([DxProducts[self.summand_index(k)] for k in
                                     range(self.nComponent)])  # broadcast duplicate summand entries to all members

            elif diffOrder == 2:  # compute second derivative of interaction function composed with Hill Components as a 2-tensor
                if nSummand == 1:  # the all sum special case
                    return np.zeros(diffOrder * [self.nComponent])  # initialize Hessian of interaction function

                elif nSummand == 2:  # the 2 summands special case
                    DpH = np.zeros(diffOrder * [self.nComponent])  # initialize derivative tensor
                    idxArray = nonzero_index(diffOrder)  # array of nonzero indices for derivative tensor
                    DpH[idxArray[:, 0], idxArray[:, 1]] = 1  # set nonzero terms to 1
                    return DpH

                else:
                    DpH = np.zeros(2 * [self.nComponent])  # initialize Hessian of interaction function
                    # compute Hessian matrix of interaction function by summand membership
                    allSummands = self.evaluate_summand(x, parameter)
                    fullProduct = np.prod(allSummands)
                    DxProducts = fullProduct / allSummands  # evaluate all partials using only nSummand-many multiplies
                    DxxProducts = np.outer(DxProducts,
                                           1.0 / allSummands)  # evaluate all second partials using only nSummand-many additional multiplies.
                    # Only the cross-diagonal terms of this matrix are meaningful.
                    for row in range(nSummand):  # compute Hessian of interaction function (outside term of chain rule)
                        for col in range(row + 1, nSummand):
                            Irow = self.summand[row]
                            Icolumn = self.summand[col]
                            DpH[np.ix_(Irow, Icolumn)] = DpH[np.ix_(Icolumn, Irow)] = DxxProducts[row, col]
                    return DpH

            elif diffOrder == 3:  # compute third derivative of interaction function composed with Hill Components as a 3-tensor
                if nSummand <= 2:  # the all sum or 2-summand special cases
                    return np.zeros(diffOrder * [self.nComponent])  # initialize Hessian of interaction function

                elif nSummand == 3:  # the 2 summands special case
                    DpH = np.zeros(diffOrder * [self.nComponent])  # initialize derivative tensor
                    idxArray = nonzero_index(diffOrder)  # array of nonzero indices for derivative tensor
                    DpH[idxArray[:, 0], idxArray[:, 1], idxArray[:, 2]] = 1  # set nonzero terms to 1
                    return DpH
                else:
                    raise KeyboardInterrupt

        else:  # compute a single partial derivative of p
            if diffOrder == 1:  # compute first partial derivatives
                if len(self.interactionType) == 1:
                    return 1.0
                else:
                    allSummands = self.evaluate_summand(x, parameter)
                    I_k = self.summand_index(diffIndex)  # get the summand index containing the k^th Hill component
                    return np.prod(
                        [allSummands[m] for m in range(len(self.interactionType)) if
                         m != I_k])  # multiply over
                # all summands which do not contain the k^th component
            else:
                raise KeyboardInterrupt

    def diff_component(self, x, parameter, diffOrder, *diffIndex, fullTensor=True):
        """Compute derivative of component vector, H = (H_1,...,H_K) with respect to state variables or parameters. This is
        the inner term in the chain rule derivative for the higher order derivatives of f. diffOrder has the form
         [xOrder, parameterOrder] which specifies the number of derivatives with respect to state variables and parameter
         variables respectively. Allowable choices are: {[1,0], [0,1], [2,0], [1,1], [0,2], [3,0], [2,1], [1,2]}"""

        xOrder = diffOrder[0]
        parameterOrder = diffOrder[1]
        gamma, parameterByComponent = self.parse_parameters(parameter)
        xLocal = x[
            self.interactionIndex]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{n_i}

        if parameterOrder == 0:  # return partials w.r.t x as a length K vector of nonzero values. dH is obtained by taking the diag operator
            # on this vector

            if xOrder == 1:
                DH_nonzero = np.array(
                    list(map(lambda H, x_k, parm: H.dx(x_k, parm), self.components, xLocal,
                             parameterByComponent)))  # evaluate vector of first order state variable partial derivatives for Hill components
            elif xOrder == 2:
                DH_nonzero = np.array(
                    list(map(lambda H, x_k, parm: H.dx2(x_k, parm), self.components, xLocal,
                             parameterByComponent)))  # evaluate vector of second order state variable partial derivatives for Hill components
            elif xOrder == 3:
                DH_nonzero = np.array(
                    list(map(lambda H, x_k, parm: H.dx3(x_k, parm), self.components, xLocal,
                             parameterByComponent)))  # evaluate vector of third order state variable partial derivatives for Hill components

            if fullTensor:
                DH = np.zeros((1 + xOrder) * [self.nComponent])
                np.einsum(''.join((1 + xOrder) * 'i') + '->i', DH)[:] = DH_nonzero
                return DH
            else:
                return DH_nonzero

        elif parameterOrder == 1:  # return partials w.r.t parameters specified by diffIndex as a vector of nonzero components.

            if not diffIndex:  # no optional argument means return all component parameter derivatives (i.e. all parameters except gamma)
                diffIndex = list(range(int(self.gammaIsVariable), self.nVariableParameter))
            parameterComponentIndex = [self.parameter_to_component_index(linearIdx) for linearIdx in
                                       diffIndex]  # a list of ordered pairs for differentiation parameter indices

            if xOrder == 0:  # Compute D_lambda(H)
                DH_nonzero = np.array(
                    list(map(lambda idx: self.components[idx[0]].diff(xLocal[idx[0]], parameterByComponent[idx[0]],
                                                                      idx[1]),
                             parameterComponentIndex)))  # evaluate vector of first order partial derivatives for Hill components

            elif xOrder == 1:
                DH_nonzero = np.array(
                    list(map(lambda idx: self.components[idx[0]].dxdiff(xLocal[idx[0]], parameterByComponent[idx[0]],
                                                                        idx[1]),
                             parameterComponentIndex)))  # evaluate vector of second order mixed partial derivatives for Hill components
            elif xOrder == 2:
                DH_nonzero = np.array(
                    list(map(lambda idx: self.components[idx[0]].dx2diff(xLocal[idx[0]], parameterByComponent[idx[0]],
                                                                         idx[1]),
                             parameterComponentIndex)))  # evaluate vector of third order mixed partial derivatives for Hill components

            if fullTensor:
                tensorDims = (1 + xOrder) * [self.nComponent] + [self.nVariableParameter - self.gammaIsVariable]
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
                for idx in range(self.nComponent):
                    parameterSlice = range(self.variableIndexByComponent[idx], self.variableIndexByComponent[idx + 1])
                    diffIndex += list(product(parameterSlice, parameterSlice))

            parameterComponentIndex = [
                ezcat(self.parameter_to_component_index(idx[0]), self.parameter_to_component_index(idx[1])[1]) for idx
                in diffIndex]
            # a list of triples stored as numpy arrays of the form (i,j,k) where lambda_j, lambda_k are both parameters for H_i

            if xOrder == 0:
                DH_nonzero = np.array(
                    list(map(lambda idx: self.components[idx[0]].diff2(xLocal[idx[0]], parameterByComponent[idx[0]],
                                                                       idx[1:]),
                             parameterComponentIndex)))  # evaluate vector of second order pure partial derivatives for Hill components

            elif xOrder == 1:
                DH_nonzero = np.array(
                    list(map(lambda idx: self.components[idx[0]].dxdiff2(xLocal[idx[0]], parameterByComponent[idx[0]],
                                                                         idx[1:]),
                             parameterComponentIndex)))  # evaluate vector of third order mixed partial derivatives for Hill components

            if fullTensor:
                tensorDims = (1 + xOrder) * [self.nComponent] + 2 * [self.nVariableParameter - self.gammaIsVariable]
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

    def dx(self, x, parameter, diffIndex=None):
        """Return the derivative as a gradient vector evaluated at x in R^n and p in R^m"""

        if diffIndex is None:
            gamma, parameterByComponent = self.parse_parameters(parameter)
            Df = np.zeros(self.dim, dtype=float)
            xLocal = x[
                self.interactionIndex]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{n_i}
            diffInteraction = self.diff_interaction(x,
                                                    parameter,
                                                    1)  # evaluate derivative of interaction function (outer term in chain rule)
            DHillComponent = np.array(
                list(map(lambda H, x_k, parm: H.dx(x_k, parm), self.components, xLocal,
                         parameterByComponent)))  # evaluate vector of partial derivatives for Hill components (inner term in chain rule)
            Df[
                self.interactionIndex] = diffInteraction * DHillComponent  # evaluate gradient of nonlinear part via chain rule
            Df[self.index] -= gamma  # Add derivative of linear part to the gradient at this HillCoordinate
            return Df

        else:  # At some point we may need to call partial derivatives with respect to specific state variables by index
            return

    def diff(self, x, parameter, diffIndex=None):
        """Evaluate the derivative of a Hill coordinate with respect to a parameter at the specified local index.
           The parameter must be a variable parameter for one or more HillComponents."""

        if diffIndex is None:  # return the full gradient with respect to parameters as a vector in R^m
            return np.array([self.diff(x, parameter, diffIndex=k) for k in range(self.nVariableParameter)])

        else:  # return a single partial derivative as a scalar
            if self.gammaIsVariable and diffIndex == 0:  # derivative with respect to decay parameter
                return -x[self.index]
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
                diffInteraction = self.diff_interaction(x, parameter, 1,
                                                        diffIndex=diffComponent)  # evaluate outer term in chain rule
                dpH = self.components[diffComponent].diff(xLocal[diffComponent],
                                                          parameterByComponent[
                                                              diffComponent],
                                                          diffParameterIndex)  # evaluate inner term in chain rule
                return diffInteraction * dpH

    def dx2(self, x, parameter):
        """Return the second derivative (Hessian matrix) with respect to the state variable vector evaluated at x in
        R^n and p in R^m as a K-by-K matrix"""

        gamma, parameterByComponent = self.parse_parameters(parameter)
        xLocal = x[
            self.interactionIndex]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{K}
        D2f = np.zeros(2 * [self.dim], dtype=float)

        D2HillComponent = np.array(
            list(map(lambda H, x_k, parm: H.dx2(x_k, parm), self.components, xLocal, parameterByComponent)))
        # evaluate vector of second partial derivatives for Hill components
        nSummand = len(self.interactionType)  # number of summands

        if nSummand == 1:  # interaction is all sum
            D2Nonlinear = np.diag(D2HillComponent)
        # TODO: Adding more special cases for 2 and even 3 summand interaction types will speed up the computation quite a bit.
        #       This should be done if this method ever becomes a bottleneck.

        else:  # interaction function contributes derivative terms via chain rule

            # compute off diagonal terms in Hessian matrix by summand membership
            allSummands = self.evaluate_summand(x, parameter)
            fullProduct = np.prod(allSummands)
            DxProducts = fullProduct / allSummands  # evaluate all partials using only nSummand-many multiplies

            # initialize Hessian matrix and set diagonal terms
            DxProductsByComponent = np.array([DxProducts[self.summand_index(k)] for k in range(self.nComponent)])
            D2Nonlinear = np.diag(D2HillComponent * DxProductsByComponent)

            # set off diagonal terms of Hessian by summand membership and exploiting symmetry
            DxxProducts = np.outer(DxProducts,
                                   1.0 / allSummands)  # evaluate all second partials using only nSummand-many additional multiplies.
            # Only the cross-diagonal terms of this matrix are meaningful.

            offDiagonal = np.zeros_like(D2Nonlinear)  # initialize matrix of mixed partials (off diagonal terms)
            for row in range(nSummand):  # compute Hessian of interaction function (outside term of chain rule)
                for col in range(row + 1, nSummand):
                    offDiagonal[np.ix_(self.summand[row], self.summand[col])] = offDiagonal[
                        np.ix_(self.summand[col], self.summand[row])] = DxxProducts[row, col]

            DHillComponent = np.array(
                list(map(lambda H, x_k, parm: H.dx(x_k, parm), self.components, xLocal,
                         parameterByComponent)))  # evaluate vector of partial derivatives for Hill components
            mixedPartials = np.outer(DHillComponent,
                                     DHillComponent)  # mixed partial matrix is outer product of gradients!
            D2Nonlinear += offDiagonal * mixedPartials
            # NOTE: The diagonal terms of offDiagonal are identically zero for any interaction type which makes the
            # diagonal terms of mixedPartials irrelevant
        D2f[np.ix_(self.interactionIndex, self.interactionIndex)] = D2Nonlinear
        return D2f

    def dxdiff(self, x, parameter, diffIndex=None):
        """Return the mixed second derivative with respect to x and a scalar parameter evaluated at x in
        R^n and p in R^m as a gradient vector in R^K. If no parameter index is specified this returns the
        full second derivative as the m-by-K Hessian matrix of mixed partials"""

        if diffIndex is None:
            return np.column_stack(
                list(map(lambda idx: self.dxdiff(x, parameter, idx), range(self.nVariableParameter))))

        else:
            D2f = np.zeros(self.dim, dtype=float)  # initialize derivative as a vector

            if self.gammaIsVariable and diffIndex == 0:  # derivative with respect to decay parameter
                D2f[self.index] = -1
                return D2f

            gamma, parameterByComponent = self.parse_parameters(parameter)
            xLocal = x[
                self.interactionIndex]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{K}
            diffComponent = np.searchsorted(self.variableIndexByComponent,
                                            diffIndex + 0.5) - 1  # get the component which contains the differentiation variable. Adding 0.5
            # makes the returned value consistent in the case that the diffIndex is an endpoint of the variable index list
            diffParameterIndex = diffIndex - self.variableIndexByComponent[
                diffComponent]  # get the local parameter index in the HillComponent for the differentiation variable

            # initialize inner terms of chain rule derivatives of f
            # DH = np.zeros(2 * [self.nComponent])  # initialize diagonal tensor for DxH as a 2-tensor
            DHillComponent = np.array(
                list(map(lambda H, x_k, parm: H.dx(x_k, parm), self.components, xLocal,
                         parameterByComponent)))  # 1-tensor of partials for DxH
            # np.einsum('ii->i', DH)[:] = DHillComponent  # build the diagonal tensor for DxH
            DpH = self.components[diffComponent].diff(xLocal[diffComponent], parameterByComponent[diffComponent],
                                                      diffParameterIndex)

            D2H = self.components[diffComponent].dxdiff(xLocal[diffComponent],
                                                        parameterByComponent[diffComponent],
                                                        diffParameterIndex)  # get the correct mixed partial derivative of H_k

            # initialize outer terms of chain rule derivatives of f
            Dp = self.diff_interaction(x, parameter, 1)[diffComponent]  # k^th index of Dp(H) is a 0-tensor (scalar)
            D2p = self.diff_interaction(x, parameter, 2)[diffComponent]  # k^th index of D^2p(H) is a 1-tensor (vector)

            D2f[self.interactionIndex] += DpH * DHillComponent * D2p  # contribution from D2(p(H))*D_parm(H)*DxH
            D2f[self.interactionIndex[diffComponent]] += D2H * Dp  # contribution from Dp(H)*D_parm(DxH)
            return D2f

    def diff2(self, x, parameter, *diffIndex, fullTensor=True):
        """Return the second derivative with respect to parameters specified evaluated at x in
        R^n and p in R^m as a Hessian matrix. If no parameter index is specified this returns the
        full second derivative as the m-by-m Hessian matrix"""

        # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
        DlambdaH = self.diff_component(x, parameter, [0, 1], fullTensor=fullTensor)
        D2lambdaH = self.diff_component(x, parameter, [0, 2], fullTensor=fullTensor)

        # get tensors for derivatives of p o H(x) (outer terms of chain rule)
        Dp = self.diff_interaction(x, parameter, 1)  # 1-tensor
        D2p = self.diff_interaction(x, parameter, 2)  # 2-tensor

        if fullTensor:  # slow version to be used as a ground truth for testing
            term1 = np.einsum('ik,kl,ij', D2p, DlambdaH, DlambdaH)
            term2 = np.einsum('i,ijk', Dp, D2lambdaH)
            DpoH = term1 + term2
        else:
            raise ValueError

        if self.gammaIsVariable:
            D2lambda = np.zeros(2 * [self.nVariableParameter])
            D2lambda[1:, 1:] = DpoH
            return D2lambda
        else:
            return DpoH

    def dx3(self, x, parameter, fullTensor=True):
        """Return the third derivative (3-tensor) with respect to the state variable vector evaluated at x in
        R^n and p in R^m as a K-by-K matrix"""

        # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
        DxH = self.diff_component(x, parameter, [1, 0], fullTensor=fullTensor)
        DxxH = self.diff_component(x, parameter, [2, 0], fullTensor=fullTensor)
        DxxxH = self.diff_component(x, parameter, [3, 0], fullTensor=fullTensor)

        # get tensors for derivatives of p o H(x) (outer terms of chain rule)
        Dp = self.diff_interaction(x, parameter, 1)  # 1-tensor
        D2p = self.diff_interaction(x, parameter, 2)  # 2-tensor
        D3p = self.diff_interaction(x, parameter, 3)  # 3-tensor

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

    def dx2diff(self, x, parameter, fullTensor=True):
        """Return the third derivative (3-tensor) with respect to the state variable vector (twice) and then the parameter
        (once) evaluated at x in R^n and p in R^m as a K-by-K matrix"""

        # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
        DxH = self.diff_component(x, parameter, [1, 0], fullTensor=fullTensor)
        DxxH = self.diff_component(x, parameter, [2, 0], fullTensor=fullTensor)
        DlambdaH = self.diff_component(x, parameter, [0, 1],
                                       fullTensor=fullTensor)  # m-vector representative of a pseudo-diagonal Km 2-tensor
        Dlambda_xH = self.diff_component(x, parameter,
                                         [1, 1],
                                         fullTensor=fullTensor)  # m-vector representative of a pseudo-diagonal KKm 3-tensor
        Dlambda_xxH = self.diff_component(x, parameter,
                                          [2, 1],
                                          fullTensor=fullTensor)  # m-vector representative of a pseudo-diagonal KKKm 4-tensor

        # get tensors for derivatives of p o H(x) (outer terms of chain rule)
        Dp = self.diff_interaction(x, parameter, 1)  # 1-tensor
        D2p = self.diff_interaction(x, parameter, 2)  # 2-tensor
        D3p = self.diff_interaction(x, parameter, 3)  # 3-tensor

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
            Dlambda_xx = np.zeros(2 * [self.dim] + [self.nVariableParameter])
            Dlambda_xx[:, :, 1:] = DpoH
            return Dlambda_xx
        else:
            return DpoH

    def dxdiff2(self, x, parameter, fullTensor=True):
        """Return the third derivative (3-tensor) with respect to the state variable vector (once) and the parameters (twice)
        evaluated at x in R^n and p in R^m as a K-by-K matrix"""

        # get vectors of appropriate partial derivatives of H (inner terms of chain rule)
        DxH = self.diff_component(x, parameter, [1, 0], fullTensor=fullTensor)
        DlambdaH = self.diff_component(x, parameter, [0, 1],
                                       fullTensor=fullTensor)  # Km 2-tensor
        Dlambda_xH = self.diff_component(x, parameter, [1, 1], fullTensor=fullTensor)  # KKm 3-tensor
        D2lambdaH = self.diff_component(x, parameter, [0, 2], fullTensor=fullTensor)
        D2lambda_xH = self.diff_component(x, parameter, [1, 2], fullTensor=fullTensor)  # KKKm 4-tensor

        # get tensors for derivatives of p o H(x) (outer terms of chain rule)
        Dp = self.diff_interaction(x, parameter, 1)  # 1-tensor
        D2p = self.diff_interaction(x, parameter, 2)  # 2-tensor
        D3p = self.diff_interaction(x, parameter, 3)  # 3-tensor

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
            D2lambda_x = np.zeros([self.dim] + 2 * [self.nVariableParameter])
            D2lambda_x[:, 1:, 1:] = DpoH
            return D2lambda_x
        else:
            return DpoH

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

    def eq_interval(self, parameter=None):
        """Return a closed interval which must contain the projection of any equilibrium onto this coordinate"""

        if parameter is None:
            # all parameters are fixed
            # TODO: This should only require all ell, delta, and gamma variables to be fixed.
            minInteraction = self.interaction_function([H.ell for H in self.components]) / self.gamma
            maxInteraction = self.interaction_function([H.ell + H.delta for H in self.components]) / self.gamma

        else:
            # some variable parameters are passed in a vector containing all parameters for this Hill Coordinate
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
        # TODO? check if the interaction elements make sense together (i.e. they have the same dimensionality)

        self.dimension = len(gamma)  # Dimension of vector field
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
        return a single numpy vector as output.

        OUTPUT: A single vector of the form:
            lambda = (gamma_1, ell_1, delta_1, theta_1, hill_1, gamma_2, ..., hill_2, ..., gamma_n, ..., hill_n).
        Any of these paramters which are not a variable for the model are simply omitted in this concatenated vector."""

        if parameter:
            return ezcat(*parameter)
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
        Dxf = np.zeros(2 * [self.dimension])  # initialize Derivative as 2-tensor (Jacobian matrix)
        parameterByCoordinate = self.unpack_variable_parameters(parameter)  # unpack variable parameters by component
        for iCoordinate in range(self.dimension):
            f_i = self.coordinates[iCoordinate]  # assign this coordinate function to a variable
            Dxf[np.ix_([iCoordinate], f_i.globalStateIndex)] = f_i.dx(x, parameterByCoordinate[
                iCoordinate])  # insert derivative of this coordinate
        return Dxf

    def diff(self, x, *parameter, diffIndex=None):
        """Return the derivative (Jacobian) of the HillModel vector field with respect to n assuming n is a VECTOR
        of Hill Coefficients. If n is uniform across all HillComponents, then the derivative is a gradient vector obtained
        by summing this Jacobian along rows.
        NOTE: This function is not vectorized. It assumes x is a single vector in R^n."""

        if diffIndex is None:  # return the full derivative wrt all parameters
            parameter = self.parse_parameter(*parameter)  # concatenate all parameters into a vector
            Dpf = np.zeros(
                [self.dimension, len(parameter)])  # initialize Derivative as 2-tensor (Jacobian matrix)
            parameterByCoordinate = self.unpack_variable_parameters(
                parameter)  # unpack variable parameters by component
            for iCoordinate in range(self.dimension):
                f_i = self.coordinates[iCoordinate]  # assign this coordinate function to a variable
                parameterSlice = np.arange(self.variableIndexByCoordinate[iCoordinate],
                                           self.variableIndexByCoordinate[iCoordinate + 1])
                Dpf[np.ix_([iCoordinate], parameterSlice)] = f_i.diff(x, parameterByCoordinate[
                    iCoordinate])  # insert derivative of this coordinate
            return Dpf
        else:
            raise ValueError  # this isn't implemented yet

    def dx2(self, x, *parameter):
        """Evaluate the second derivative of f w.r.t. state variable vector (twice), returns a 3D tensr"""
        parameter = self.parse_parameter(*parameter)  # concatenate all parameters into a vector
        Dxf = np.zeros(3 * [self.dimension])  # initialize Derivative as 3-tensor
        parameterByCoordinate = self.unpack_variable_parameters(parameter)  # unpack variable parameters by component
        for iCoordinate in range(self.dimension):
            f_i = self.coordinates[iCoordinate]  # assign this coordinate function to a variable
            xSlice = np.array(f_i.globalStateIndex)
            Dxf[np.ix_([iCoordinate], xSlice, xSlice)] = f_i.dx2(x, parameterByCoordinate[
                iCoordinate])  # insert derivative of this coordinate
        return Dxf

    def dxdiff(self, x, *parameter, diffIndex=None):
        """Evaluate the second order mixed derivative of f w.r.t. state variables and parameters (once each)"""
        if diffIndex is None:  # return the full derivative wrt all parameters
            parameter = self.parse_parameter(*parameter)  # concatenate all parameters into a vector
            Dpxf = np.zeros(2 * [self.dimension] + [len(parameter)])  # initialize Derivative as 3-tensor
            parameterByCoordinate = self.unpack_variable_parameters(
                parameter)  # unpack variable parameters by component
            for iCoordinate in range(self.dimension):
                f_i = self.coordinates[iCoordinate]  # assign this coordinate function to a variable
                parameterSlice = np.arange(self.variableIndexByCoordinate[iCoordinate],
                                           self.variableIndexByCoordinate[iCoordinate + 1])
                xSlice = np.array(f_i.globalStateIndex)
                Dpxf[np.ix_([iCoordinate], xSlice, parameterSlice)] = f_i.dxdiff(x, parameterByCoordinate[
                    iCoordinate])  # insert derivative of this coordinate
            return Dpxf
        else:
            raise ValueError  # this isn't implemented yet

    def diff2(self, x, *parameter, diffIndex=None):
        """Evaluate the second order derivative of f w.r.t. parameters (twice)"""
        if diffIndex is None:  # return the full derivative wrt all parameters
            parameter = self.parse_parameter(*parameter)  # concatenate all parameters into a vector
            Dppf = np.zeros([self.dimension] + 2 * [len(parameter)])  # initialize Derivative as 3-tensor
            parameterByCoordinate = self.unpack_variable_parameters(
                parameter)  # unpack variable parameters by component
            for iCoordinate in range(self.dimension):
                f_i = self.coordinates[iCoordinate]  # assign this coordinate function to a variable
                parameterSlice = np.arange(self.variableIndexByCoordinate[iCoordinate],
                                           self.variableIndexByCoordinate[iCoordinate + 1])
                Dppf[np.ix_([iCoordinate], parameterSlice, parameterSlice)] = f_i.diff2(x, parameterByCoordinate[
                    iCoordinate])  # insert derivative of this coordinate
            return Dppf
        else:
            raise ValueError  # this isn't implemented yet

    def dx3(self, x, *parameter):
        """Evaluate the third derivative of f w.r.t. state variable vector (three times)"""
        parameter = self.parse_parameter(*parameter)  # concatenate all parameters into a vector
        Dxxxf = np.zeros(4 * [self.dimension])  # initialize Derivative as 4-tensor
        parameterByCoordinate = self.unpack_variable_parameters(parameter)  # unpack variable parameters by component
        for iCoordinate in range(self.dimension):
            f_i = self.coordinates[iCoordinate]  # assign this coordinate function to a variable
            xSlice = np.array(f_i.globalStateIndex)
            Dxxxf[np.ix_([iCoordinate], xSlice, xSlice, xSlice)] = f_i.dx3(x, parameterByCoordinate[
                iCoordinate])  # insert derivative of this coordinate
        return Dxxxf

    def dx2diff(self, x, *parameter, diffIndex=None):
        """Evaluate the third order derivative of a HillModel w.r.t. parameters (once) and state variable vector (twice)"""
        if diffIndex is None:  # return the full derivative wrt all parameters
            parameter = self.parse_parameter(*parameter)  # concatenate all parameters into a vector
            Dpxxf = np.zeros(3 * [self.dimension] + [len(parameter)])  # initialize Derivative as 4-tensor
            parameterByCoordinate = self.unpack_variable_parameters(
                parameter)  # unpack variable parameters by component

            for iCoordinate in range(self.dimension):
                f_i = self.coordinates[iCoordinate]  # assign this coordinate function to a variable
                parameterSlice = np.arange(self.variableIndexByCoordinate[iCoordinate],
                                           self.variableIndexByCoordinate[iCoordinate + 1])
                xSlice = np.array(f_i.globalStateIndex)
                Dpxxf[np.ix_([iCoordinate], xSlice, xSlice, parameterSlice)] = f_i.dx2diff(x, parameterByCoordinate[
                    iCoordinate])  # insert derivative of this coordinate
            return Dpxxf
        else:
            raise ValueError  # this isn't implemented yet

    def dxdiff2(self, x, *parameter, diffIndex=None):
        """Evaluate the third order derivative of f w.r.t. parameters (twice) and state variable vector (once)"""
        if diffIndex is None:  # return the full derivative wrt all parameters
            parameter = self.parse_parameter(*parameter)  # concatenate all parameters into a vector
            Dppxf = np.zeros(2 * [self.dimension] + 2 * [len(parameter)])  # initialize Derivative as 4-tensor
            parameterByCoordinate = self.unpack_variable_parameters(
                parameter)  # unpack variable parameters by component
            for iCoordinate in range(self.dimension):
                f_i = self.coordinates[iCoordinate]  # assign this coordinate function to a variable
                parameterSlice = np.arange(self.variableIndexByCoordinate[iCoordinate],
                                           self.variableIndexByCoordinate[iCoordinate + 1])
                xSlice = np.array(f_i.globalStateIndex)
                Dppxf[np.ix_([iCoordinate], xSlice, parameterSlice, parameterSlice)] = f_i.dxdiff2(x,
                                                                                                   parameterByCoordinate[
                                                                                                       iCoordinate])  # insert derivative of this coordinate
            return Dppxf
        else:
            raise ValueError  # this isn't implemented yet

    def find_equilibria(self, gridDensity, *parameter, uniqueRootDigits=5, eqBound=None):
        """Return equilibria for the Hill Model by uniformly sampling for initial conditions and iterating a Newton variant.
        INPUT:
            *parameter - Evaluations for variable parameters to use for evaluating the root finding algorithm
            gridDensity - density to sample in each dimension.
            uniqueRootDigits - Number of digits to use for distinguishing between floats.
            eqBound - N-by-2 array of intervals defining a search rectangle. Initial data will be chosen uniformly here. """

        # TODO: Include root finding method as kwarg
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

        def radii_uniqueness_existence(equilibrium):
            DF_x = DF(equilibrium)
            D2F_x = self.dx2(equilibrium, *parameter)
            A = np.linalg.inv(DF_x)
            Y_bound = np.linalg.norm(A @ F(equilibrium))
            Z0_bound = np.linalg.norm(np.identity(len(equilibrium)) - A @ DF_x)
            Z2_bound = np.linalg.norm(A) * np.linalg.norm(D2F_x)
            if Z2_bound<10^-16:
                Z2_bound = 10^-10 # in case the Z2 bound is too close to zero, we increase it a bit
            delta = 1 - 4*(Z0_bound + Y_bound) * Z2_bound
            if delta<0:
                return 0,0
            max_rad = (1 + np.sqrt(delta))/(2*Z2_bound)
            min_rad = (1 - np.sqrt(delta))/(2*Z2_bound)
            return max_rad, min_rad

        # build a grid of initial data for Newton algorithm
        if eqBound is None:  # use the trivial equilibrium bounds
            eqBound = np.array(list(map(lambda f_i, parm: f_i.eq_interval(parm), self.coordinates, parameterByCoordinate)))
        coordinateIntervals = [np.linspace(*interval, num=gridDensity) for interval in eqBound]
        evalGrid = np.meshgrid(*coordinateIntervals)
        X = np.column_stack([G_i.flatten() for G_i in evalGrid])

        # Apply rootfinding algorithm to each initial condition
        solns = list(
            filter(lambda root: root.success and eq_is_positive(root.x), [find_root(F, DF, x, diagnose=True)
                                                                          for x in X]))  # return equilibria which converged
        if solns:
            equilibria = np.row_stack([root.x for root in solns])  # extra equilibria as vectors in R^n
            equilibria = np.unique(np.round(equilibria, uniqueRootDigits), axis=0)  # remove duplicates
            #equilibria = np.unique(np.round(equilibria/10**np.ceil(log(equilibria)),
            #                                uniqueRootDigits)*10**np.ceil(log(equilibria)), axis=0)
            equilibria = np.unique(np.round(equilibria, uniqueRootDigits), axis=0)  # remove duplicates
            if len(equilibria)>1:
                all_equilibria = equilibria
                radii = np.zeros(len(all_equilibria))
                unique_equilibria = all_equilibria
                for i in range(len(all_equilibria)):
                    equilibrium = all_equilibria[i]
                    max_rad, min_rad = radii_uniqueness_existence(equilibrium)
                    radii[i] = max_rad

                radii2 = radii
                for i in range(len(all_equilibria)):
                    equilibrium1 = all_equilibria[i]
                    radius1 = radii[i]
                    j = i+1
                    while j < len(radii2):
                        equilibrium2 = unique_equilibria[j]
                        radius2 = radii2[j]
                        if np.linalg.norm(equilibrium1-equilibrium2)<np.maximum(radius1, radius2):
                            # remove one of the two from
                            unique_equilibria = np.delete(unique_equilibria,j)
                            radii2 = np.delete(radii2,j)
                        else:
                            j = j+1
            return np.row_stack([find_root(F, DF, x) for x in equilibria])  # Iterate Newton again to regain lost digits
        else:
            return None