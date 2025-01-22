"""
A description of what the script performs

    Output: output
    Other files required: none

    Author: Shane Kepley
    email: s.kepley@vu.nl
    Date: 7/14/23; Last revision: 7/14/23
"""
import numpy as np
from abc import ABC, abstractmethod


ABSTRACT_PARAMETER_NAMES = ['a', 'list','of','variable','names']

class Activation(ABC):
    def __init__(self, sign, parameterValues, variableParameters, parameterCallIndex, fixedParameter, PARAMETER_NAMES):
        self.sign = sign
        self.parameterValues = parameterValues
        self.parameterNames = PARAMETER_NAMES.copy()  # ordered list of possible parameter names
        self.variableParameters = variableParameters
        self.parameterCallIndex = parameterCallIndex
        self.fixedParameter = fixedParameter
        #  set callable parameter name functions
        for idx in range(len(self.variableParameters)):
            self.add_parameter_call(self.variableParameters[idx], idx)

    @classmethod
    def initialize_abstract(cls, parameter_size, productionSign, parameterNames, **kwargs):
        parameterValues = np.zeros(parameter_size)  # initialize vector of parameter values
        parameterCallIndex = {parameterNames[j]: j for j in range(parameter_size)}  # calling index for parameter by name
        for parameterName, parameterValue in kwargs.items():
            parameterValues[parameterCallIndex[parameterName]] = parameterValue  # update fixed parameter value in evaluation vector
            del parameterCallIndex[parameterName]  # remove fixed parameter from callable list

        variableParameters = list(parameterCallIndex.keys())  # set callable parameters
        parameterCallIndex = list(parameterCallIndex.values())  # get indices for callable parameters
        fixedParameter = [parameterName for parameterName in parameterNames if
                               parameterName not in variableParameters]

        cls(productionSign, parameterValues, variableParameters, parameterCallIndex, fixedParameter, parameterNames.copy())
        return

    @classmethod
    @abstractmethod
    def initialise(cls):
        pass

    def __iter__(self):
        """Make iterable"""
        yield self

    def add_parameter_call(self, parameterName, parameterIndex):
        """Adds a call by name function for variable parameters to a HillComponent instance"""

        def call_function(self, parameter):
            """returns a class method which has the given parameter name. This method slices the given index out of a
            variable parameter vector"""
            return parameter[parameterIndex]

        setattr(Activation, parameterName, call_function)  # set dynamic method name

    def curry_parameters(self, parameter):
        """Returns a parameter evaluation vector in R^4 with fixed and variable parameters indexed properly"""

        # TODO: When all parameters of this component are fixed this function still requires an empty list as an argument.
        parameterEvaluation = self.parameterValues.copy()  # get a mutable copy of the fixed parameter values
        parameterEvaluation[self.parameterCallIndex] = parameter  # slice passed parameter vector into callable slots
        return parameterEvaluation

    @abstractmethod
    def __call__(self, x, *parameter):
        pass

    def __repr__(self):
        """Return a canonical string representation of a Hill component"""

        reprString = 'Tanh Component: \n' + 'sign = {0} \n'.format(self.sign)
        for parameterName in self.parameterNames:
            if parameterName not in self.variableParameters:
                reprString += parameterName + ' = {0} \n'.format(getattr(self, parameterName))
        reprString += 'Variable Parameters: {' + ', '.join(self.variableParameters) + '}\n'
        return reprString

    @abstractmethod
    def dx(self, x, parameter):
        pass

    @abstractmethod
    def dx2(self, x, parameter):
        pass

    @abstractmethod
    def diff(self, x, parameter, diffIndex):
        pass

    @abstractmethod
    def diff2(self, x, parameter, diffIndex):
        pass

    @abstractmethod
    def dxdiff(self, x, parameter, diffIndex):
        pass

    @abstractmethod
    def dx2diff(self, x, parameter, diffIndex):
        pass

    @abstractmethod
    def dxdiff2(self, x, parameter, diffIndex):
        pass

    @abstractmethod
    def dx3(self, x, parameter):
        pass

    @abstractmethod
    def image(self, parameter=None):
        pass
