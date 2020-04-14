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

plt.close('all')


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
        #  set callable parameter name functions
        for idx in range(len(self.variableParameters)):
            self.add_parameter_call(self.variableParameters[idx], idx)

        print(self.delta([1, 2, 3, 4]))

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
            gamma - (float) decay rate for this coordinate or NaN if gamma is a variable parameter which is callable as
                the first component of the parameter variable vector.
            parameter - (numpy array) A K-by-4 array of Hill component parameters with rows of the form [ell, delta, theta, hillCoefficient]
                Entries which are NaN are variable parameters which are callable in the function and all derivatives.
            interactionSign - (list) A vector in F_2^K carrying the sign type for each Hill component
            interactionType - (list) A vector describing the interaction type of the interaction function specified as an integer partition of K
            interactionIndex - (list) A length K+1 vector of global state variable indices. interactionIndex[0] is the global index
                for this coordinate and interactionIndex[1:] the indices of the K incoming interacting nodes"""

        # TODO: Class constructor should not do work!
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
        self.variableIndexByComponent = np.insert(np.cumsum(self.nVarByComponent), 0,
                                                  0)  # endpoints for concatenated parameter vector by coordinate
        self.nVariableParameter = sum(
            self.nVarByComponent) + self.gammaIsVariable  # number of variable parameters for this coordinate

    def curry_gamma(self, parameter):
        """Returns the value of gamma and a curried version of a variable parameter argument"""

        # If gamma is not fixed, then it must be the first coordinate of the parameter vector
        if self.gammaIsVariable:
            return parameter[0], parameter[1:]
        else:  # return fixed gamma and unaltered parameter
            return self.gamma, parameter

    def __call__(self, x, parameter=np.array([])):
        """Evaluate the Hill coordinate on a vector of (global) state variables and (local) parameter variables. This is a
        map of the form  g: R^n x R^m ---> R where n is the number of state variables of the Hill model and m is the number
        of variable parameters for this Hill coordinate"""

        # TODO: Currently the input parameter must be a numpy array even if there is only a single parameter.
        if isvector(x):  # Evaluate coordinate for a single x in R^n
            # slice callable parameters into a list of length K. The j^th list contains the variable parameters belonging to
            # the j^th Hill component.
            gamma, parameter = self.curry_gamma(parameter)
            parameterByComponent = [parameter[self.variableIndexByComponent[j]:self.variableIndexByComponent[j + 1]] for
                                    j in range(self.nComponent)]
            hillComponentValues = np.array(
                list(map(lambda H, idx, parm: H(x[idx], parm), self.components, self.interactionIndex,
                         parameterByComponent)))  # evaluate Hill components
            nonlinearTerm = self.interaction_function(hillComponentValues)  # compose with interaction function
            return -gamma * x[self.index] + nonlinearTerm

        # TODO: vectorized evaluation is a little bit hacky and should be rewritten to be more efficient
        else:  # vectorized evaluation where x is a matrix of column vectors to evaluate
            return np.array([self(x[:, j], parameter) for j in range(np.shape(x)[1])])

    def interaction_function(self, parameter):
        """Evaluate the polynomial interaction function at a parameter in (0,inf)^{K}"""

        return np.sum(
            parameter)  # dummy functionality computes all sum interaction. Updated version below just needs to be tested.
        # return np.prod([sum([parm[idx] for idx in sumList]) for sumList in self.summand])

    def dx(self, x, parameter=np.array([])):
        """Return the derivative (gradient vector) evaluated at x in R^n and p in R^m as a row vector"""

        gamma, parameter = self.curry_gamma(parameter)
        dim = len(x)  # dimension of vector field
        Df = np.zeros(dim, dtype=float)
        xLocal = x[
            self.interactionIndex]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{K}
        diffInteraction = self.diff_interaction(xLocal)  # evaluate outer term in chain rule
        parameterByComponent = [parameter[self.variableIndexByComponent[j]:self.variableIndexByComponent[j + 1]] for
                                j in range(self.nComponent)]  # unpack variable parameters by component
        DHillComponent = np.array(
            list(map(lambda H, x, parm: H.dx(x, parm), self.components, xLocal,
                     parameterByComponent)))  # evaluate inner term in chain rule
        Df[
            self.interactionIndex] = diffInteraction * DHillComponent  # evaluate gradient of nonlinear part via chain rule
        Df[self.index] -= gamma  # Add derivative of linear part to the gradient at this HillCoordinate
        return Df

    def dn(self, x, parameter=np.array([])):
        """Evaluate the derivative of a HillCoordinate with respect to the vector of Hill coefficients as a row vector.
        Evaluation requires specifying x in R^n and p in R^m. This method does not assume that all HillCoordinates have
        a uniform Hill coefficient. If this is the case then the scalar derivative with respect to the Hill coefficient
        should be the sum of the gradient vector returned"""

        gamma, parameter = self.curry_gamma(parameter)
        dim = len(x)  # dimension of vector field
        df_dn = np.zeros(dim, dtype=float)
        xLocal = x[
            self.interactionIndex]  # extract only the coordinates of x that this HillCoordinate depends on as a vector in R^{K}
        diffInteraction = self.diff_interaction(xLocal)  # evaluate outer term in chain rule
        parameterByComponent = [parameter[self.variableIndexByComponent[j]:self.variableIndexByComponent[j + 1]] for
                                j in range(self.nComponent)]  # unpack variable parameters by component
        dHillComponent_dn = np.array(
            list(map(lambda H, x, parm: H.dn(x, parm), self.components, xLocal,
                     parameterByComponent)))  # evaluate inner term in chain rule
        df_dn[
            self.interactionIndex] = diffInteraction * dHillComponent_dn  # evaluate gradient of nonlinear part via chain rule
        return df_dn

    def set_components(self, parameter, interactionSign):
        """Return a list of Hill components for this Hill coordinate"""

        parameterNames = ['ell', 'delta', 'theta', 'hillCoefficient']  # ordered list of possible parameter names

        def row2dict(row):
            """convert ordered row of parameter matrix to kwarg"""
            return {parameterNames[j]: row[j] for j in range(4) if
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

    def eq_interval(self):
        """Return a closed interval which must contain the projection of any equilibrium onto this coordinate"""

        try:
            minInteraction = self.interaction_function([H.ell for H in self.components]) / self.gamma
            maxInteraction = self.interaction_function([H.ell + H.delta for H in self.components]) / self.gamma
            enclosingInterval = np.array([minInteraction, maxInteraction])
        except AttributeError:
            print(
                'Current implementation requires fixed ell, delta for all HillComponents and fixed gamma for this HillCoordinate')
        return enclosingInterval


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

    if not isvector(x0):  # an array whose columns are initial guesses
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

    def unpack_variable_parameters(self, parameter):
        """Unpack a parameter vector for the HillModel into component vectors for each distinct coordinate"""

        return [parameter[self.variableIndexByCoordinate[j]:self.variableIndexByCoordinate[j + 1]] for
                j in range(self.dimension)]

    def __call__(self, x, parameter=np.array([])):
        """Evaluate the vector field defined by this HillModel instance. This is a function of the form
        f: R^n x R^{m_1} x ... x R^{m_n} ---> R^n where the j^th Hill coordinate has m_j variable parameters. The syntax
        is f(x,p) where p = (p_1,...,p_n) is a variable parameter vector constructed by ordered concatenation of vectors
        of the form p_j = (p_j1,...,p_jK) which is also an ordered concatenation of the variable parameters associated to
        the K-HillComponents for the j^th HillCoordinate."""

        parameterByCoordinate = self.unpack_variable_parameters(parameter)  # unpack variable parameters by component
        if isvector(x):  # input a single vector in R^n
            return np.array(list(map(lambda f_i, parm: f_i(x, parm), self.coordinates, parameterByCoordinate)))
        else:  # vectorized input
            return np.row_stack(list(map(lambda f_i, parm: f_i(x, parm), self.coordinates, parameterByCoordinate)))

    def dx(self, x, parameter=np.array([])):
        """Return the derivative (Jacobian) of the HillModel vector field with respect to x.
        NOTE: This function is not vectorized. It assumes x is a single vector in R^n."""

        parameterByCoordinate = self.unpack_variable_parameters(parameter)  # unpack variable parameters by component
        return np.vstack(list(map(lambda f_i, parm: f_i.dx(x, parm), self.coordinates,
                                  parameterByCoordinate)))  # return a vertical stack of gradient (row) vectors

    def dn(self, x, parameter=np.array([])):
        """Return the derivative (Jacobian) of the HillModel vector field with respect to n assuming n is a VECTOR
        of Hill Coefficients. If n is uniform across all HillComponents, then the derivative is a gradient vector obtained
        by summing this Jacobian along rows.
        NOTE: This function is not vectorized. It assumes x is a single vector in R^n."""

        parameterByCoordinate = self.unpack_variable_parameters(parameter)  # unpack variable parameters by component
        return np.vstack(list(map(lambda f_i, parm: f_i.dn(x, parm), self.coordinates,
                                  parameterByCoordinate)))  # return a vertical stack of gradient (row) vectors

    def find_equilibria(self, parameter, gridDensity, uniqueRootDigits=7):
        """Return equilibria for the Hill Model by uniformly sampling for initial conditions and iterating a Newton variant.
        INPUT:
            parameter - (numpy vector) Evaluations for variable parameters to use for evaluating the root finding algorithm
            gridDensity - (int) density to sample in each dimension.
            uniqueRootDigits - (int) Number of digits to use for distinguishing between floats."""

        # TODO: Include root finding method as vararg
        def F(x):
            """Fix parameter values in the zero finding map"""
            return self.__call__(x, parameter)

        def DF(x):
            """Fix parameter values in the zero finding map derivative"""
            return self.dx(x, parameter)

        # build a grid of initial data for Newton algorithm
        evalGrid = np.meshgrid(*[np.linspace(*f_i.eq_interval(), num=gridDensity) for f_i in self.coordinates])
        X = np.row_stack([G_i.flatten() for G_i in evalGrid])
        solns = list(filter(lambda root: root.success,
                            [find_root(F, DF, X[:, j], diagnose=True)
                             for j in range(X.shape[1])]))  # return equilibria which converged
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
                    [[0, 0], [0, self.coordinates[0].components[0].dx2(x[1], self.coordinates[0].curry_gamma(parm)[1])]]))
        setattr(self.coordinates[1], 'dx2',
                lambda x, parm: np.array(
                    [[self.coordinates[1].components[0].dx2(x[0], self.coordinates[1].curry_gamma(parm)[1]), 0], [0, 0]]))

        setattr(self.coordinates[0], 'dndx',
                lambda x, parm: np.array([0, self.coordinates[0].components[0].dndx(x[1], self.coordinates[0].curry_gamma(parm)[1])]))
        setattr(self.coordinates[1], 'dndx',
                lambda x, parm: np.array([self.coordinates[1].components[0].dndx(x[0], self.coordinates[1].curry_gamma(parm)[1]), 0]))

    def pass_uniform_hill(self, n, parameter):
        """Inserts copies of n into the appropriate Hill coefficient indices in the parameter vector"""

        return np.insert(parameter, self.hillIndexByCoordinate, n)

    def __call__(self, x, n, parameter):
        """Overload the toggle switch to identify the Hill coefficients and evaluate as a separate scalar input.

        INPUT:
            x - (vector) state vector in R^2
            n - (float) a single scalar parameter for the Hill coefficients of both coordinates
            parameter - (vector) remaining free parameters with length between 0 and 8 ordered as
                (gamma1, ell1, delta1, theta1, gamma2, ell2, delta2, theta2)
            """
        return super().__call__(x, self.pass_uniform_hill(n, parameter))

    def dx(self, x, n, parameter=np.array([])):
        """Overload the toggle switch derivative to identify the Hill coefficients"""

        return super().dx(x, self.pass_uniform_hill(n, parameter))

    def dn(self, x, n, parameter=np.array([])):
        """Overload the toggle switch derivative to identify the Hill coefficients"""

        Df_dn = super().dn(x, self.pass_uniform_hill(n, parameter))  # Return Jacobian with respect to n = (n1, n2)
        return np.sum(Df_dn, 1)  # n1 = n2 = n so the derivative is tbe gradient vector of f with respect to n

    def plot_nullcline(self, n, parameter=np.array([]), nNodes=100, domainBounds=(10, 10)):
        """Plot the nullclines for the toggle switch at a given parameter"""

        # equilibria = self.find_equilibria(n, 10)
        Xp = np.linspace(0, domainBounds[0], nNodes)
        Yp = np.linspace(0, domainBounds[1], nNodes)
        Z = np.zeros_like(Xp)

        # unpack decay parameters separately
        gamma = np.array(list(map(lambda f_i, parm: f_i.curry_gamma(parm)[0], self.coordinates,
                                  self.unpack_variable_parameters(self.pass_uniform_hill(n, parameter)))))
        N1 = (self(np.row_stack([Z, Yp]), n, parameter) / gamma[0])[0, :]  # f1 = 0 nullcline
        N2 = (self(np.row_stack([Xp, Z]), n, parameter) / gamma[1])[1, :]  # f2 = 0 nullcline
        # N1 = self.coordinates[0](np.row_stack([Z, Yp]), np.array([n])) / self.coordinates[0].gamma  # f1 = 0 nullcline
        # N2 = self.coordinates[1](np.row_stack([Xp, Z]), np.array([n])) / self.coordinates[1].gamma  # f2 = 0 nullcline

        # if equilibria.ndim == 0:
        #     pass
        # elif equilibria.ndim == 1:
        #     plt.scatter(equilibria[0], equilibria[1])
        # else:
        #     plt.scatter(equilibria[0, :], equilibria[1, :])

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


class SaddleNode:
    """A instance of a saddle-node bifurcation problem for a HillModel with 1 hill coefficient parameter which is
    shared by all HillComponents"""

    def __init__(self, hillModel, phaseCondition=lambda v: np.linalg.norm(v) - 1, phaseConditionDerivative=lambda v: v / np.linalg.norm(v)):
        """Construct an instance of a saddle-node problem for specified HillModel"""

        self.model = hillModel
        self.phaseCondition = phaseCondition
        self.diffPhaseCondition = phaseConditionDerivative
        self.mapDimension = 2 * hillModel.dimension + 1  # dimension of the zero finding map

    def unpack_components(self, u):
        """Unpack the input vector for a SaddleNode problem into 3 component vectors of the form (x, v, n)"""
        return u[0:self.model.dimension], u[self.model.dimension:-1], u[-1]

    def zero_map(self, u, parameter):
        """A zero finding map for saddle-node bifurcations of the form g: R^{2n+1} ---> R^{2n+1} whose roots are
        isolated parameters for which a saddle-node bifurcation occurs.
        INPUT: u = (x, v, hillCoefficient) where x is a state vector, v a tangent vector."""

        stateVector, tangentVector, hillCoefficient = self.unpack_components(u)  # unpack input vector
        g1 = self.model(stateVector,
                        hillCoefficient,
                        parameter)  # this is zero iff x is an equilibrium for the system at parameter value n
        g2 = self.model.dx(stateVector,
                           hillCoefficient,
                           parameter) @ tangentVector  # this is zero iff v lies in the kernel of Df(x, n)
        g3 = self.phaseCondition(tangentVector)  # this is zero iff v satisfies the phase condition
        return np.concatenate((g1, g2, g3), axis=None)

    def diff_zero_map(self, u, parameter):
        """Evaluate the derivative of the zero finding map. This is a matrix valued function of the form
        Dg: R^{2n+1} ---> M_{2n+1}(R).
        INPUT: u = (x, v, hillCoefficient) where x is a state vector, v a tangent vector."""

        # unpack input vector and set dimensions for Jacobian blocks
        stateVector, tangentVector, hillCoefficient = self.unpack_components(u)  # unpack input vector
        n = self.model.dimension
        fullParameter = self.model.pass_uniform_hill(hillCoefficient, parameter)  # insert copies of HillCoefficient into parameter vector
        parameterByCoordinate = self.model.unpack_variable_parameters(
            fullParameter)  # unpack full parameter vector by coordinate
        Dg = np.zeros([self.mapDimension, self.mapDimension])  # initialize (2n+1)-by-(2n+1) matrix
        Df = self.model.dx(stateVector, hillCoefficient, parameter)  # store derivative of vector field which appears in 2 blocks
        # TODO: Function calls for derivatives in HIllModel are overloaded to have input syntax (x, n, parm) where as
        #  the ad hoc Hessian calls use (x, parm). These should be changed to the former.

        # BLOCK ROW 1
        Dg[0:n, 0:n] = Df  # block - (1,1)
        # block - (1, 2) is an n-by-n zero block
        Dg[0:n, -1] = self.model.dn(stateVector, hillCoefficient, parameter)  # block - (1,3)
        # BLOCK ROW 2
        Dg[n:2 * n, 0:n] = np.row_stack(
            [tangentVector @ self.model.coordinates[j].dx2(stateVector, parameterByCoordinate[j])
             for j in range(n)])  # block - (2,1)
        Dg[n:2 * n, n:2 * n] = Df  # block - (2,2)
        Dg[n:2 * n, -1] = np.row_stack(list(
            map(lambda f_i, parm: f_i.dndx(stateVector, parm), self.model.coordinates,
                parameterByCoordinate))) @ tangentVector  # block - (2,3)
        # BLOCK ROW 3
        # block - (3, 1) is a 1-by-n zero block
        Dg[-1, n:2 * n] = self.diffPhaseCondition(tangentVector)  # block - (3,2)
        # block - (3, 1) is a 1-by-1 zero block
        return Dg


# set some parameters to test using MATLAB toggle switch for ground truth
decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)

f = ToggleSwitch(decay, [p1, p2])
f1 = f.coordinates[0]
f2 = f.coordinates[1]
H1 = f1.components[0]
H2 = f2.components[0]
n0 = 4.1

x0 = np.array([4, 3])
p0 = np.array([1, 1, 5, 3, 1, 1, 6, 3], dtype=float)
print(f(x0, n0, p0))
SN = SaddleNode(f)

# eq = f.find_equilibria(n0, 10)
# v0 = np.array([1, -.7])
v0 = np.array([1, 1])
# x0 = eq[:, 1]
u0 = np.concatenate((x0, v0, np.array(n0)), axis=None)


uSol = find_root(lambda u: SN.zero_map(u, p0), lambda u: SN.diff_zero_map(u, p0), u0, diagnose=True)
# compare to uSol = [ 4.55637172,  2.25827744,  0.82199933, -0.56948846,  3.17447061]
print(uSol)
xSol, vSol, nSol = [uSol.x[idx] for idx in [[0, 1], [2, 3], [4]]]

# plot nullclines
plt.close('all')
plt.figure()
f.plot_nullcline(4.1, p0)
plt.figure()
f.plot_nullcline(nSol, p0)
