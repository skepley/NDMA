"""
A description of what the script performs

    Output: output
    Other files required: none
   
    Author: Shane Kepley
    email: s.kepley@vu.nl
    Date: 7/14/23; Last revision: 7/14/23
"""
import numpy as np
from numpy import log

# ordered list of HillComponent parameter names as a class variable

PARAMETER_NAMES = ['ell', 'delta', 'theta', 'hillCoefficient']

class HillActivation:
    """A component of a Hill system of the form ell + delta*H(x; ell, delta, theta, n) where H is an increasing or decreasing Hill function.
    Any of these parameters can be considered as a fixed value for a Component or included in the callable variables. The
    indices of the edges associated to ell, and delta are different than those associated to theta."""

    def __init__(self, productionSign, **kwargs):
        """A Hill function with parameters [ell, delta, theta, d] of productionType in {-1, 1} to denote H^-, H^+ """
        # TODO: Class constructor should not do work!

        self.sign = productionSign
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

        setattr(HillActivation, parameterName, call_function)  # set dynamic method name

    def curry_parameters(self, parameter):
        """Returns a parameter evaluation vector in R^4 with fixed and variable parameters indexed properly"""

        # TODO: When all parameters of this component are fixed this function still requires an empty list as an argument.
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
