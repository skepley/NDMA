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

TANH_PARAMETER_NAMES = ['ell', 'delta', 'theta']


class tanhActivation:
    """A component of a Hill system of the form ell + delta*tanh(sign*(x - theta)).
    Any of these parameters can be considered as a fixed value for a Component or included in the callable variables. The
    indices of the edges associated to ell, and delta are different then those associated to theta."""

    def __init__(self, productionSign, **kwargs):
        """A tanh function with parameters [ell, delta, theta] of productionType in {-1, 1}"""
        # TODO: Class constructor should not do work!

        self.sign = productionSign
        self.parameterValues = np.zeros(3)  # initialize vector of parameter values
        parameterNames = TANH_PARAMETER_NAMES.copy()  # ordered list of possible parameter names
        parameterCallIndex = {parameterNames[j]: j for j in range(3)}  # calling index for parameter by name
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

        setattr(tanhActivation, parameterName, call_function)  # set dynamic method name

    def curry_parameters(self, parameter):
        """Returns a parameter evaluation vector in R^4 with fixed and variable parameters indexed properly"""

        # TODO: When all parameters of this component are fixed this function still requires an empty list as an argument.
        parameterEvaluation = self.parameterValues.copy()  # get a mutable copy of the fixed parameter values
        parameterEvaluation[self.parameterCallIndex] = parameter  # slice passed parameter vector into callable slots
        return parameterEvaluation

    def __call__(self, x, *parameter):
        """Evaluation method for a tanh component function instance"""

        ell, delta, theta = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values

        return ell + delta*np.tanh(self.sign*(x - theta))

    def __repr__(self):
        """Return a canonical string representation of a Hill component"""

        reprString = 'Tanh Component: \n' + 'sign = {0} \n'.format(self.sign)
        for parameterName in TANH_PARAMETER_NAMES:
            if parameterName not in self.variableParameters:
                reprString += parameterName + ' = {0} \n'.format(getattr(self, parameterName))
        reprString += 'Variable Parameters: {' + ', '.join(self.variableParameters) + '}\n'
        return reprString

    def dx(self, x, parameter):
        """Evaluate the derivative of a tanh component with respect to x"""

        ell, delta, theta = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values

        return self.sign * delta * ( 1 - np.tanh(self.sign*(x-theta))**2)

    def dx2(self, x, parameter):
        """Evaluate the second derivative of a Hill component with respect to x"""

        ell, delta, theta = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        tanh_xtheta = np.tanh(self.sign*(x-theta))
        return 2*delta*(tanh_xtheta**2 -1)*tanh_xtheta

    def diff(self, x, parameter, diffIndex):
        """Evaluate the derivative of a Hill component with respect to a parameter at the specified local index.
        The parameter must be a variable parameter for the HillComponent."""

        diffParameter = self.variableParameters[diffIndex]  # get the name of the differentiation variable

        if diffParameter == 'ell':
            return 1.

        ell, delta, theta = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters
        tanh_xtheta = np.tanh(self.sign*(x-theta))

        if diffParameter == 'delta':
            dH = tanh_xtheta

        elif diffParameter == 'theta':
            dH = - delta * self.sign * (1-tanh_xtheta**2)
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

        ell, delta, theta = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters
        tanh_xtheta = np.tanh(self.sign * (x - theta))

        if diffParameter0 == 'delta':
            if diffParameter1 == 'delta':
                return 0.
            if diffParameter1 == 'theta':
                dH = self.sign * (tanh_xtheta**2 -1)
                return dH
            else:
                raise ValueError('Requested derivatives do not exist')
        elif diffParameter0 == 'theta':
            if diffParameter1 == 'theta':
                dH = - 2* delta * (tanh_xtheta**2 - 1) * tanh_xtheta
                return dH
            else:
                raise ValueError('Requested derivatives do not exist')

        raise ValueError('Requested derivatives do not exist')

    def dxdiff(self, x, parameter, diffIndex):
        """Evaluate the derivative of a tanh component with respect to the state variable and a parameter at the specified
        local index.
        The parameter must be a variable parameter for the tanhComponent."""
        diffParameter = self.variableParameters[diffIndex]  # get the name of the differentiation variable

        if diffParameter == 'ell':
            return 0.

        ell, delta, theta = self.curry_parameters(
            parameter)  # unpack fixed and variable parameters
        tanh_xtheta = np.tanh(self.sign * (x - theta))

        if diffParameter == 'delta':
            ddH = - self.sign * (tanh_xtheta**2 - 1)
            return ddH
        elif diffParameter == 'theta':
            ddH= - 2 * delta * (tanh_xtheta**2 - 1) * tanh_xtheta
            return ddH

        raise ValueError('Requested derivatives do not exist')

    def dx2diff(self, x, parameter, diffIndex):
        """Evaluate the derivative of a Hill component with respect to the state variable and a parameter at the specified
        local index.
        The parameter must be a variable parameter for the HillComponent."""
        diffParameter = self.variableParameters[diffIndex]  # get the name of the differentiation variable

        if diffParameter == 'ell':
            return 0.
        ell, delta, theta = self.curry_parameters(
                parameter)  # unpack fixed and variable parameters
        tanh_xtheta = np.tanh(self.sign * (x - theta))

        if diffParameter == 'delta':
            d3H = 2 * (tanh_xtheta**2 - 1) * tanh_xtheta
            return d3H

        if diffParameter == 'theta':
            d3H = - self.sign * 2 * delta * (tanh_xtheta**2 - 1) * (3*tanh_xtheta**2 -1)
            return d3H

        raise ValueError('Requested derivatives do not exist')

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

        ell, delta, theta = self.curry_parameters(
            parameter)  # unpack fixed and variable parameters
        tanh_xtheta = np.tanh(self.sign * (x - theta))

        if diffParameter0 == 'delta':
            if diffParameter1 == 'delta':
                return 0.
            if diffParameter1 == 'theta':
                dH = 2 * (tanh_xtheta**2 - 1) * tanh_xtheta
                return dH
        if diffParameter0 == 'theta' and diffParameter1 == 'theta':
            dH = - 2* self.sign * delta *(tanh_xtheta**2 - 1) * (3*tanh_xtheta**2 - 1)
            return dH
        raise ValueError('Requested derivatives do not exist')

    def dx3(self, x, parameter):
        """Evaluate the second derivative of a Hill component with respect to x"""

        ell, delta, theta = self.curry_parameters(
            parameter)  # unpack fixed and variable parameter values
        tanh_xtheta = np.tanh(self.sign * (x - theta))
        dH = - 2* self.sign * delta *(tanh_xtheta**2 - 1) * (3*tanh_xtheta**2 - 1)
        return dH

    def image(self, parameter=None):
        """Return the range of this HillComponent given by (ell, ell+delta)"""

        ell, delta, theta = self.curry_parameters(
            parameter)  # unpack fixed and variable parameters

        return np.array([ell - delta, ell + delta])
