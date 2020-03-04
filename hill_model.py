"""
Classes and methods for constructing, evaluating, and doing parameter continuation of Hill Models
   
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 2/29/20; Last revision: 3/4/20
"""
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from math import log


class HillComponent:
    """A component of a Hill system of the form ell + delta*H(x; ell, delta, theta, n) where H is an increasing or decreasing Hill function.
    In practice, the value of n should be thought of as a variable and the indices of the edges associated to ell, delta are
    different than those associated to theta."""

    def __init__(self, interactionSign, parameter):
        """A Hill function with parameters [ell, delta, theta, n] of InteractionType in {-1, 1} to denote H^-, H^+ """
        self.sign = interactionSign
        self.ell = parameter[0]
        self.delta = parameter[1]
        self.theta = parameter[2]
        self.hillCoefficient = parameter[3]

    def __iter__(self):
        """Make iterable"""
        yield self

    def __call__(self, x):
        """Evaluation method for a Hill function instance"""

        hillCoefficient = self.hillCoefficient  # extract Hill coefficient in the local scope. This is so that later versions will allow this as a passed variable.

        # compute powers of x and theta only once. This should be added as a property later to speed up numerical algorithms
        xPower = x ** hillCoefficient
        thetaPower = self.theta ** hillCoefficient  # compute theta^hillCoefficient only once

        # evaluation rational part of the Hill function
        if self.sign == 1:
            evalRational = xPower / (xPower + thetaPower)
        elif self.sign == -1:
            evalRational = thetaPower / (xPower + thetaPower)
        return self.ell + self.delta * evalRational

    def __repr__(self):
        """Return a canonical string representation of a Hill component"""

        return ('Hill Component: \n' + 'sign = {0} \n'.format(self.sign) + 'ell = {0} \n'.format(
            self.ell) + 'delta = {0} \n'.format(self.delta) +
                'theta = {0} \n'.format(self.theta)) + 'n = {0} \n'.format(self.hillCoefficient)

    def dx(self, x, nDerivative=1):
        """Returns the derivative of a Hill component with respect to x"""

        hillCoefficient = self.hillCoefficient  # extract Hill coefficient in the local scope. This is so that later versions will allow this as a passed variable.
        # compute powers of x and theta only once. This should be added as a property later to speed up numerical algorithms
        thetaPower = self.theta ** hillCoefficient
        if nDerivative == 1:
            xPowerSmall = x ** (hillCoefficient - 1)  # compute x^{hillCoefficient-1}
            xPower = xPowerSmall * x
            return self.sign * hillCoefficient * self.delta * thetaPower * xPowerSmall / ((thetaPower + xPower) ** 2)
        elif nDerivative == 2:
            xPowerSmall = x ** (hillCoefficient - 2)  # compute x^{hillCoefficient-1}
            xPower = xPowerSmall * x ** 2
            return self.sign * self.delta * thetaPower * xPowerSmall * (
                    (hillCoefficient - 1) * thetaPower - (hillCoefficient + 1) * xPower) / ((thetaPower + xPower) ** 3)
        else:
            raise KeyboardInterrupt

    def dn(self, x):
        """Returns the derivative of a Hill component with respect to n"""

        hillCoefficient = self.hillCoefficient  # extract Hill coefficient in the local scope. This is so that later versions will allow this as a passed variable.
        # compute powers of x and theta only once. This should be added as a property later to speed up numerical algorithms
        xPower = x ** hillCoefficient
        thetaPower = self.theta ** hillCoefficient

        return self.sign * self.delta * xPower * thetaPower * log(x / self.theta) / ((thetaPower + xPower) ** 2)

    def dndx(self, x):
        """Returns the mixed partials of a Hill component with respect to n and x"""

        hillCoefficient = self.hillCoefficient  # extract Hill coefficient in the local scope. This is so that later versions will allow this as a passed variable.
        # compute powers of x and theta only once. This should be added as a property later to speed up numerical algorithms
        thetaPower = self.theta ** hillCoefficient
        xPowerSmall = x ** (hillCoefficient - 1)  # compute x^{hillCoefficient-1}
        xPower = xPowerSmall * x
        return self.sign * self.delta * thetaPower * xPowerSmall * (
                hillCoefficient * (thetaPower - xPower) * log(x / self.theta) + thetaPower + xPower) / (
                       (thetaPower + xPower) ** 3)


class HillCoordinate:
    """Define a coordinate of the vector field for a Hill system. This is a scalar equation taking the form
    x' = -gamma*x + p(H_1, H_2,...,H_k) where each H_i is a Hill function depending on x_i which is a state variable
    which regulates x"""

    def __init__(self, gamma, parameter, hillCoefficient, interactionSign, interactionType, interactionIndex):
        """Hill Coordinate instantiation with the following syntax:
        INPUTS:
            gamma - scalar decay rate for this coordinate
            parameter - A K-by-3 numpy array of Hill component parameter with rows of the form [ell, delta, theta]
            hillCoefficient - A length K vector or list with Hill coefficients for the K incoming interactions
            interactionSign - A vector in F_2^K carrying the sign type for each Hill component
            interactionType - A vector describing the interaction type of the interaction function specified as an integer partition of K
            interactionIndex - A length K+1 vector of global state variable indices for this coordinate followed by the K incoming interacting nodes"""

        self.gamma = gamma  # set linear decay
        self.numcomponents = len(interactionSign)  # number of interaction nodes
        self.components = self.set_components(parameter, interactionSign, hillCoefficient)
        self.hillcoefficient = hillCoefficient  # Set Hill coefficient vector
        self.index = interactionIndex[0]  # Define this coordinates global index
        self.interaction = interactionIndex[1:]  # Vector of global interaction variable indices
        self.interactiontype = interactionType
        self.summand = self.set_summand()
        self.interactionfunction = self.set_interaction()  # evaluation method for interaction polynomial

    def __call__(self, x):
        """Evaluate the Hill coordinate on a vector of (global) state variables. This is a map of the form
        g: R^n ---> R"""

        hillComponentValues = np.array(list(map(lambda H, idx: H(x[idx]), self.components,
                                                self.interaction)))  # evaluate Hill components
        nonlinearTerm = self.interactionfunction(hillComponentValues)  # compose with interaction function
        return -self.gamma * x[self.index] + nonlinearTerm

    def set_components(self, parameter, interactionSign, hillCoefficient):
        """Return a list of Hill components for this Hill coordinate"""

        if self.numcomponents == 1:
            return [HillComponent(interactionSign[0], np.append(parameter, hillCoefficient))]
        else:
            return [HillComponent(interactionSign[k], np.append(parameter[k, :], hillCoefficient[k])) for k in
                    range(self.numcomponents)]  # list of Hill components

    def set_summand(self):
        """Return the list of lists containing the summand indices defined by the interaction type.
        EXAMPLE:
            interactionType = [2,1,3,1] returns the index partition [[0,1], [2], [3,4,5], [6]]"""

        sumEndpoints = np.insert(np.cumsum(self.interactiontype), 0,
                                 0)  # summand endpoint indices including initial zero
        localIndex = list(range(self.numcomponents))
        return [localIndex[sumEndpoints[i]:sumEndpoints[i + 1]] for i in range(len(self.interactiontype))]

    def set_interaction(self):
        """Dummy functionality just returns the sum for now"""

        return lambda v: np.sum(v)

    def dinteraction(self, xLocal):
        """Dummy functionality for evaluating the derivative of the interaction function"""

        if len(self.interactiontype) == 1:
            return np.ones(len(xLocal))
        else:
            raise KeyboardInterrupt

    def dx(self, x):
        """Return the derivative (gradient vector) evaluated at x in R^n as a row vector"""

        dim = len(x)  # dimension of vector field
        Df = np.zeros(dim, dtype=float)
        xLocal = x[
            self.interaction]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{K}
        Dinteraction = self.dinteraction(xLocal)  # evaluate outer term in chain rule
        DHillComponent = np.array(
            list(map(lambda H, x: H.dx(x), self.components, xLocal)))  # evaluate inner term in chain rule
        Df[self.interaction] = Dinteraction * DHillComponent  # evaluate gradient of nonlinear part via chain rule
        Df[self.index] -= self.gamma  # Add derivative of linear part to the gradient at this HillCoordinate
        return Df


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

        return np.array([f_i(x) for f_i in self.coordinate])

    def dx(self, x):
        """Return the derivative (Jacobian) of the HillModel vector field evaluated at x"""

        return np.vstack([f_i.dx(x) for f_i in self.coordinate])


def toggleswitch(gamma, parameter, hillCoefficient):
    """Defines the vector field for the toggle switch example"""

    # define Hill system for toggle switch
    f1 = HillCoordinate(gamma[0], parameter[0, :], hillCoefficient, [-1], [1], [0, 1])
    f2 = HillCoordinate(gamma[1], parameter[1, :], hillCoefficient, [-1], [1], [1, 0])
    return lambda x: np.array([f1(x), f2(x)])


# set some parameters to test using MATLAB toggle switch for ground truth
gamma = np.array([1, 1], dtype=float)
ell = np.array([1, 1], dtype=float)
theta = np.array([3, 3], dtype=float)
delta = np.array([5, 6], dtype=float)
hillParm = np.column_stack([ell, delta, theta])
hillCoefficient = 4.1
x0 = np.array([4, 3])

# test Hill component code
parm = np.append(hillParm[0, :], hillCoefficient)
H = HillComponent(-1, parm)

# test Hill coordinate code
f1 = HillCoordinate(gamma[0], hillParm[0, :], hillCoefficient, [-1], [1], [0, 1])
f2 = HillCoordinate(gamma[1], hillParm[1, :], hillCoefficient, [-1], [1], [1, 0])

# test Hill model code
ts1 = toggleswitch(gamma, hillParm, hillCoefficient)
ts2 = HillModel(gamma, [hillParm[0, :], hillParm[1, :]], [hillCoefficient, hillCoefficient],
                [[-1], [-1]], [[1], [1]], [[0, 1], [1, 0]])

# verify that ts1(x0) = ts2(x0) - DONE
# verify that ts2.dx(x0) matches MATLAB - DONE


# test Hill model equilibrium finding
sol = optimize.root(ts2, np.array([4.1, 4.1]), jac=lambda x: ts2.dx(x), method='hybr')
# sol.x ~ (1.1610, 6.8801) as it should be
