"""
Classes and methods for constructing, evaluating, and doing parameter continuation of Hill Models
   
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 2/29/20; Last revision: 3/4/20
"""
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from numpy import log


def isvector(array):
    """Returns true if input is a numpy vector"""

    return len(np.shape(array)) == 1


class HillComponent:
    """A component of a Hill system of the form ell + delta*H(x; ell, delta, theta, n) where H is an increasing or decreasing Hill function.
    Any of these parameters can be considered as a fixed value for a Component or included in the callable variables. The
    indices of the edges associated to ell, and delta are different than those associated to theta."""

    def __init__(self, interactionSign, **kwargs):
        """A Hill function with parameters [ell, delta, theta, n] of InteractionType in {-1, 1} to denote H^-, H^+ """
        # TODO: Class constructor should not do work!

        self.sign = interactionSign
        self.parameterValues = np.zeros(4)  # initialize vector of parameter values
        parameterNames = ['ell', 'delta', 'theta', 'hillCoefficient']  # ordered list of possible parameter names
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

    def __iter__(self):
        """Make iterable"""
        yield self

    def curry_parameters(self, parameter):
        """Returns a parameter evaluation vector in R^4 with fixed and variable parameters indexed properly"""
        parameterEvaluation = self.parameterValues.copy()  # get a mutable copy of the fixed parameter values
        parameterEvaluation[self.parameterCallIndex] = parameter  # slice passed parameter vector into callable slots
        return parameterEvaluation

    def __call__(self, x, parameter=np.array([])):
        """Evaluation method for a Hill component function instance"""

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

    def dx(self, x, parameter=np.array([]), nDerivative=1):
        """Returns the derivative of a Hill component with respect to x"""

        ell, delta, theta, hillCoefficient = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        # compute powers of x and theta only once.
        thetaPower = theta ** hillCoefficient
        if nDerivative == 1:
            xPowerSmall = x ** (hillCoefficient - 1)  # compute x^{hillCoefficient-1}
            xPower = xPowerSmall * x
            return self.sign * hillCoefficient * delta * thetaPower * xPowerSmall / ((thetaPower + xPower) ** 2)
        elif nDerivative == 2:
            xPowerSmall = x ** (hillCoefficient - 2)  # compute x^{hillCoefficient-1}
            xPower = xPowerSmall * x ** 2
            return self.sign * delta * thetaPower * xPowerSmall * (
                    (hillCoefficient - 1) * thetaPower - (hillCoefficient + 1) * xPower) / ((thetaPower + xPower) ** 3)
        else:
            raise KeyboardInterrupt

    def dn(self, x, parameter=np.array([])):
        """Returns the derivative of a Hill component with respect to n"""

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


class HillCoordinate:
    """Define a coordinate of the vector field for a Hill system. This is a scalar equation taking the form
    x' = -gamma*x + p(H_1, H_2,...,H_k) where each H_i is a Hill function depending on x_i which is a state variable
    which regulates x"""

    def __init__(self, parameter, interactionSign, interactionType, interactionIndex, gamma=np.nan):
        """Hill Coordinate instantiation with the following syntax:
        INPUTS:
            gamma - scalar decay rate for this coordinate or NaN if gamma is a variable parameter which is callable as
                the first component of the parameter variable vector.
            parameter - A K-by-4 numpy array of Hill component parameters with rows of the form [ell, delta, theta, hillCoefficient]
                Entries which are NaN are variable parameters which are callable in the function and all derivatives.
            interactionSign - A vector in F_2^K carrying the sign type for each Hill component
            interactionType - A vector describing the interaction type of the interaction function specified as an integer partition of K
            interactionIndex - A length K+1 vector of global state variable indices. interactionIndex[0] is the global index
                for this coordinate and interactionIndex[1:] the indices of the K incoming interacting nodes"""

        # TODO: Class constructor should not do work!
        self.gammaIsVariable = np.isnan(gamma)
        if ~np.isnan(gamma):
            self.gamma = gamma  # set fixed linear decay
        self.parameterValues = parameter  # initialize array of fixed parameter values
        self.numComponent = len(interactionSign)  # number of interaction nodes
        self.components = self.set_components(parameter, interactionSign)
        self.index = interactionIndex[0]  # Define this coordinate's global index
        self.interaction = interactionIndex[1:]  # Vector of global interaction variable indices
        self.interactionType = interactionType  # specified as an integer partition of K
        self.summand = self.set_summand()
        self.numVarByComponent = list(
            map(lambda j: np.count_nonzero(np.isnan(self.parameterValues[j, :])), range(self.numComponent)))
        self.variableIndexByComponent = np.insert(np.cumsum(self.numVarByComponent), 0, 0)

    def __call__(self, x, parameter):
        """Evaluate the Hill coordinate on a vector of (global) state variables and (local) parameter variables. This is a
        map of the form  g: R^n x R^m ---> R where n is the number of state variables of the Hill model and m is the number
        of variable parameters for this Hill coordinate"""

        # If gamma is not fixed, then it must be the first coordinate of the parameter vector
        if self.gammaIsVariable:
            gamma = parameter.pop(0)
        else:
            gamma = self.gamma

        if isvector(x):  # Evaluate coordinate for a single x in R^n
            # slice callable parameters into a list of length K. The j^th list contains the variable parameters belonging to
            # the j^th Hill component.
            parameterByComponent = [parameter[self.variableIndexByComponent[j]:self.variableIndexByComponent[j + 1]] for
                                    j in range(self.numComponent)]
            hillComponentValues = np.array(
                list(map(lambda H, idx, parm: H(x[idx], parm), self.components, self.interaction,
                         parameterByComponent)))  # evaluate Hill components
            nonlinearTerm = self.interaction_function(hillComponentValues)  # compose with interaction function
            return -gamma * x[self.index] + nonlinearTerm

        # TODO: vectorized evaluation is a little bit hacky and should be rewritten to be more efficient
        else:  # vectorized evaluation where x is a matrix of column vectors to evaluate
            return np.array([self(x[:, j]) for j in range(np.shape(x)[1])])

    def dx(self, x, parameter):
        """Return the derivative (gradient vector) evaluated at x in R^n and p in R^m as a row vector"""

        # If gamma is not fixed, then it must be the first coordinate of the parameter vector
        if self.gammaIsVariable:
            gamma = parameter.pop(0)
        else:
            gamma = self.gamma

        dim = len(x)  # dimension of vector field
        Df = np.zeros(dim, dtype=float)
        xLocal = x[
            self.interaction]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{K}
        Dinteraction = self.diff_interaction(xLocal)  # evaluate outer term in chain rule
        parameterByComponent = [parameter[self.variableIndexByComponent[j]:self.variableIndexByComponent[j + 1]] for
                                j in range(self.numComponent)]  # unpack variable parameters by component
        DHillComponent = np.array(
            list(map(lambda H, x, parm: H.dx(x, parm), self.components, xLocal,
                     parameterByComponent)))  # evaluate inner term in chain rule
        Df[self.interaction] = Dinteraction * DHillComponent  # evaluate gradient of nonlinear part via chain rule
        Df[self.index] -= gamma  # Add derivative of linear part to the gradient at this HillCoordinate
        return Df

    def set_components(self, parameter, interactionSign):
        """Return a list of Hill components for this Hill coordinate"""

        parameterNames = ['ell', 'delta', 'theta', 'hillCoefficient']  # ordered list of possible parameter names

        def row2dict(row):
            """convert ordered row of parameter matrix to kwarg"""
            return {parameterNames[j]: row[j] for j in range(4) if
                    not np.isnan(row[j])}

        if self.numComponent == 1:
            return [HillComponent(interactionSign[0], **row2dict(parameter))]
        else:
            return [HillComponent(interactionSign[k], **row2dict(parameter[k, :])) for k in
                    range(self.numComponent)]  # list of Hill components

    def set_summand(self):
        """Return the list of lists containing the summand indices defined by the interaction type.
        EXAMPLE:
            interactionType = [2,1,3,1] returns the index partition [[0,1], [2], [3,4,5], [6]]"""

        sumEndpoints = np.insert(np.cumsum(self.interactionType), 0,
                                 0)  # summand endpoint indices including initial zero
        localIndex = list(range(self.numComponent))
        return [localIndex[sumEndpoints[i]:sumEndpoints[i + 1]] for i in range(len(self.interactionType))]

    def diff_interaction(self, xLocal):
        """Dummy functionality for evaluating the derivative of the interaction function"""

        if len(self.interactionType) == 1:
            return np.ones(len(xLocal))
        else:
            raise KeyboardInterrupt

    def interaction_function(self, parm):
        """Evaluate the polynomial interaction function at a parameter in (0,inf)^{K}"""

        return np.sum(
            parm)  # dummy functionality computes all sum interaction. Updated version below just needs to be tested.
        # return np.prod([sum([parm[idx] for idx in sumList]) for sumList in self.summand])

    def eq_interval(self):
        """Return a closed interval which must contain the projection of any equilibrium onto this coordinate"""

        try:
            minInteraction = self.interaction_function([H.ell for H in self.components]) / self.gamma
            maxInteraction = self.interaction_function([H.ell + H.delta for H in self.components]) / self.gamma
            enclosingInterval = np.array([minInteraction, maxInteraction])
        except AttributeError:
            print('Current implementation requires fixed ell, delta parameters')
        return enclosingInterval


class HillModel:
    """Define a Hill model as a vector field describing the derivatives of all state variables. The i^th coordinate
    describes the derivative of the state variable, x_i, as a function of x_i and its incoming interactions, {X_1,...,X_{K_i}}.
    This function is always a linear decay and a nonlinear interaction defined by a polynomial composition of Hill
    functions evaluated at the interactions. The vector field is defined coordinate-wise as a vector of HillCoordinate instances"""

    def __init__(self, gamma, parameter, hillCoefficient, interactionSign, interactionType, interactionIndex):
        """Class constructor which has the following syntax:
        INPUTS:
            gamma - A vector in R^n of linear decay rates
            parameter - A length n list of K_i-by-3 parameter arrays
            hillCoefficient - A length n list of vectors in R^{K_i}
            interactionSign - A length n list of vectors in F_2^{K_i}
            interactionType - A length n list of length q_i lists describing an integer partitions of K_i
            interactionIndex - A length n list whose i^th element is a list of global indices for the i^th incoming interactions"""

        self.dimension = len(gamma)  # Dimension of vector field
        self.coordinate = [HillCoordinate(gamma[i], parameter[i], hillCoefficient[i], interactionSign[i],
                                          interactionType[i], interactionIndex[i]) for i in range(self.dimension)]

    def __call__(self, x):
        """Evaluate the vector field defined by this HillModel instance"""

        if isvector(x):  # input a single vector in R^n
            return np.array([f_i(x) for f_i in self.coordinate])
        else:  # vectorized input
            return np.row_stack([f_i(x) for f_i in self.coordinate])

    def dx(self, x):
        """Return the derivative (Jacobian) of the HillModel vector field evaluated at x"""

        return np.vstack([f_i.dx(x) for f_i in self.coordinate])

    def findeq(self, gridDensity):
        """Return equilibria for the Hill Model by uniformly sampling for initial conditions and iterating a Newton variant"""

        findroot = lambda x0: optimize.root(self, x0, jac=lambda x: self.dx(x),
                                            method='hybr')  # set root finding algorithm
        # build a grid of initial data for Newton algorithm
        evalGrid = np.meshgrid(*[np.linspace(*f_i.eq_interval(), num=gridDensity) for f_i in self.coordinate])
        X = np.row_stack([G_i.flatten() for G_i in evalGrid])
        solns = list(filter(lambda root: root.success,
                            [findroot(X[:, j]) for j in range(X.shape[1])]))  # return equilibria which converged
        equilibria = np.column_stack([root.x for root in solns])  # extra equilibria as vectors in R^n
        equilibria = np.unique(np.round(equilibria, 7), axis=1)  # remove duplicates
        return np.column_stack([findroot(equilibria[:, j]).x for j in
                                range(np.shape(equilibria)[1])])  # Iterate Newton again to regain lost digits


def toggleswitch(gamma, parameter, hillCoefficient):
    """Defines the vector field for the toggle switch example"""

    # define Hill system for toggle switch
    return HillModel(gamma, parameter, [hillCoefficient, hillCoefficient],
                     [[-1], [-1]], [[1], [1]], [[0, 1], [1, 0]])

# # set some parameters to test using MATLAB toggle switch for ground truth
# # gamma = np.array([1, 1], dtype=float)
# # ell = np.array([1, 1], dtype=float)
# # theta = np.array([3, 3], dtype=float)
# # delta = np.array([5, 6], dtype=float)
# # hillParm = np.column_stack([ell, delta, theta])
# # hillCoefficient = 4.1
# # x0 = np.array([4, 3])
# #
# # # test Hill component code
# # parm = np.append(hillParm[0, :], hillCoefficient)
# # H = HillComponent(-1, parm)
# #
# # # test Hill coordinate code
# # f1 = HillCoordinate(gamma[0], hillParm[0, :], hillCoefficient, [-1], [1], [0, 1])
# # f2 = HillCoordinate(gamma[1], hillParm[1, :], hillCoefficient, [-1], [1], [1, 0])

# test Hill model code
# ts = toggleswitch(gamma, [hillParm[0, :], hillParm[1, :]], hillCoefficient)

# verify that ts1(x0) = ts2(x0) - DONE
# verify that ts2.dx(x0) matches MATLAB - DONE
# equilibria = ts2.findeq(10) # test Hill model equilibrium finding - DONE
# added vectorized evaluation of Hill Models - DONE


# # plot nullclines and equilibria
# plt.close('all')
# Xp = np.linspace(0, 10, 100)
# Yp = np.linspace(0, 10, 100)
# Z = np.zeros_like(Xp)
#
# equilibria = ts.findeq(10)
# N1 = ts.coordinate[0](np.row_stack([Z, Yp])) / gamma[0]  # f1 = 0 nullcline
# N2 = ts.coordinate[1](np.row_stack([Xp, Z])) / gamma[1]  # f2 = 0 nullcline
#
# plt.figure()
# plt.scatter(equilibria[0, :], equilibria[1, :])
# plt.plot(Xp, N2)
# plt.plot(N1, Yp)
