"""
A description of what the script performs

    Output: output
    Other files required: none
   
    Author: Shane Kepley
    email: s.kepley@vu.nl
    Date: 7/14/23; Last revision: 7/14/23
"""
import numpy as np
from scipy import linalg

from ndma.coordinate import Coordinate
from ndma.hill_model import ezcat, find_root, is_vector


def verify_call(func):
    """Evaluation method decorator for validating evaluation calls. This can be used when troubleshooting and
    testing and then omitted in production runs to improve efficiency. This decorates any Model method which has inputs
    of the form (self, x, *parameter, **kwargs)."""

    def func_wrapper(*args, **kwargs):
        modelObj = args[0]  # a HillModel or HillCoordinate instance is passed as 1st position argument to evaluation
        # method
        x = args[1]  # state vector passed as 2nd positional argument to evaluation method
        if not is_vector(x):  # x must be a vector
            raise TypeError('Second argument must be a state vector.')

        N = modelObj.dimension
        parameter = modelObj.parse_parameter(*args[2:])  # parameters passed as variable argument to
        # evaluation method and need to be parsed to obtain a single parameter vector

        if len(x) != N:  # make sure state input is the correct size
            raise IndexError(
                'State vector for this evaluation should be size {0} but received a vector of size {1}'.format(N,
                                                                                                               len(x)))
        elif len(parameter) != modelObj.nParameter:
            raise IndexError(
                'Parsed parameter vector for this evaluation should be size {0} but received a vector of '
                'size {1}'.format(
                    modelObj.nParameter, len(parameter)))

        else:  # parameter and state vectors are correct size. Pass through to evaluation method
            return func(*args, **kwargs)

    return func_wrapper


class Model:
    """Define a NDMA model as a vector field describing the derivatives of all state variables. The i^th coordinate
    describes the derivative of the state variable, x_i, as a function of x_i and the state variables influencing
    its production nonlinearly represented by an activation function. For this model all activation functions are the same.
    The vector field is defined coordinate-wise as a vector of Coordinate instances."""

    def __init__(self, gamma, productionParameter, productionSign, productionType, productionIndex, activationFunction):
        """Class constructor which has the following syntax:
        INPUTS:
            gamma - A vector in R^n of linear decay rates
            productionParameter - A length n list of K_i-by-4 parameter arrays
                    Note: If K_i = 1 then productionSign[i] should be a vector, not a matrix i.e. it should have shape
                    (4,) as opposed to (1,4). If the latter case then the result will be squeezed since otherwise HillCoordinate
                    will throw an exception during construction of that coordinate.
            productionSign - A length n list of lists in F_2^{K_i}
            productionType - A length n list of length q_i lists describing an integer partitions of K_i
            productionIndex - A length n list whose i^th element is a length K_i list of global indices for the nonlinear
                interactions for node i. These are specified in any order as long as it is the same order used for productionSign
                and the rows of parameter. IMPORTANT: The exception to this occurs if node i has a self edge. In this case i must appear as the first
                index.
            activationFunction - The name of the activation function to use for all nonlinear production terms."""

        # TODO: Class constructor should not do work!
        self.dimension = len(gamma)  # Dimension of vector field
        coordinateDims = [len(set(productionIndex[j] + [j])) for j in range(self.dimension)]
        self.coordinates = [Coordinate(np.squeeze(productionParameter[j]), productionSign[j],
                                       productionType[j], coordinateDims[j], gamma=gamma[j]) for j in
                            range(
                                self.dimension)]  # A list of HillCoordinates specifying each coordinate of the vector field
        self.productionIndex = productionIndex  # store the list of global indices which contribute to the production term of each coordinate.
        self.stateIndexByCoordinate = [self.state_variable_selection(idx) for idx in range(self.dimension)]
        # create a list of selections which slice the full state vector into subvectors for passing to evaluation functions of each coordinate.
        self.nParameterByCoordinate = list(f_i.nParameter for f_i in
                                           self.coordinates)  # number of variable parameters by coordinate
        parameterIndexEndpoints = np.insert(np.cumsum(self.nParameterByCoordinate), 0,
                                            0)  # endpoints for concatenated parameter vector by coordinate
        self.parameterIndexByCoordinate = [list(range(parameterIndexEndpoints[idx], parameterIndexEndpoints[idx + 1]))
                                           for idx in
                                           range(self.dimension)]
        self.nParameter = sum(self.nParameterByCoordinate)  # number of variable parameters for this HillModel

    def state_variable_selection(self, idx):
        """Return a list which selects the correct state subvector for the component with specified index."""

        if idx in self.productionIndex[idx]:  # This coordinate has a self edge in the GRN
            if self.productionIndex[idx][0] == idx:
                return self.productionIndex[idx]
            else:
                raise IndexError(
                    'Coordinates with a self edge must have their own index appearing first in their interaction index list')

        else:  # This coordinate has no self edge. Append its own global index as the first index of the selection slice.
            return [idx] + self.productionIndex[idx]

    def parse_parameter(self, *parameter):
        """Default parameter parsing if input is a single vector simply returns the same vector. Otherwise, it assumes
        input parameters are provided in order and concatenates into a single vector. This function is included in
        function calls so that subclasses can redefine function calls with customized parameters and overload this
        function as needed. Overloaded versions should take a variable number of numpy arrays as input and must always
        return a single numpy vector as output.

        OUTPUT: A single vector of the form:
            lambda = (gamma_1, ell_1, delta_1, theta_1, hill_1, gamma_2, ..., hill_2, ..., gamma_n, ..., hill_n).
        Any of these parameters which are not a variable for the model are simply omitted in this concatenated vector."""

        if parameter:
            parameterVector = ezcat(*parameter)

            return ezcat(*parameter)
        else:
            return np.array([])

    def unpack_parameter(self, parameter):
        """Unpack a parameter vector for the HillModel into disjoint parameter slices for each distinct coordinate"""

        return [parameter[idx] for idx in self.parameterIndexByCoordinate]

    def unpack_state(self, x):
        """Unpack a state vector for the HillModel into a length-n list of state vector slices to pass for evaluation into
         distinct coordinate. The slices are not necessarily disjoint since multiple coordinates can depend on the same
         state variable."""

        return [x[idx_slice] for idx_slice in self.stateIndexByCoordinate]

    def unpack_by_coordinate(self, x, *parameter):
        """Unpack a parameter and state vector into subvectors for each coordinate. This is called by all evaluation functions."""

        parameterByCoordinate = self.unpack_parameter(
            self.parse_parameter(*parameter))  # concatenate all parameters into
        # a vector and unpack by coordinate
        stateByCoordinate = self.unpack_state(x)  # unpack state variables by coordinate
        return stateByCoordinate, parameterByCoordinate

    @verify_call
    def __call__(self, x, *parameter):
        """Evaluate the vector field defined by this HillModel instance. This is a function of the form
        f: R^n x R^{m_1} x ... x R^{m_n} ---> R^n where the j^th Hill coordinate has m_j variable parameters. The syntax
        is f(x,p) where p = (p_1,...,p_n) is a variable parameter vector constructed by ordered concatenation of vectors
        of the form p_j = (p_j1,...,p_jK) which is also an ordered concatenation of the variable parameters associated to
        the K-HillComponents for the j^th HillCoordinate.
        NOTE: This function is not vectorized. It assumes x is a single vector in R^n."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        return np.array(
            list(map(lambda f_i, x_i, p_i: f_i(x_i, p_i), self.coordinates, stateByCoordinate, parameterByCoordinate)))

    @verify_call
    def dx(self, x, *parameter):
        """Return the first derivative of the HillModel vector field with respect to x as a rank-2 tensor (matrix). The i-th row
        of this tensor is the differential (i.e. gradient) of the i-th coordinate of f. NOTE: This function is not vectorized. It assumes x
        and parameter represent a single state and parameter vector."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        Dxf = np.zeros(2 * [self.dimension])  # initialize Derivative as 2-tensor of size NxN
        for (i, f_i) in enumerate(self.coordinates):
            # get gradient values for f_i and embed derivative of f_i into the full derivative of f.
            Dxf[np.ix_([i], self.stateIndexByCoordinate[
                i])] = f_i.dx(stateByCoordinate[i], parameterByCoordinate[i])
        return Dxf

    @verify_call
    def diff(self, x, *parameter, diffIndex=None):
        """Return the first derivative of the HillModel vector field with respect to a specific parameter (or the full parameter vector) as
        a vector (or matrix). In the latter case, the i-th row of this tensor is the differential of
        the i-th coordinate of f with respect to parameters. NOTE: This function is not vectorized. It assumes x
        and parameter represent a single state and parameter vector. NOTE: This function is not vectorized."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        if diffIndex is None:  # return the full derivative wrt all parameters
            Dpf = np.zeros(
                [self.dimension, self.nParameter])  # initialize Derivative as 2-tensor of size NxM
            for (i, f_i) in enumerate(self.coordinates):
                Dpf[np.ix_([i], self.parameterIndexByCoordinate[i])] = f_i.diff(stateByCoordinate[i],
                                                                                parameterByCoordinate[
                                                                                    i])  # insert derivative of this coordinate
            return Dpf
        else:
            raise IndexError('selective differentiation indices is not yet implemented')  # this isn't implemented yet

    @verify_call
    def dx2(self, x, *parameter):
        """Return the second derivative of the HillModel vector field with respect to x (twice) as a rank-3 tensor. The i-th matrix
        of this tensor is the Hessian matrix of the i-th coordinate of f. NOTE: This function is not vectorized. It assumes x
        and parameter represent a single state and parameter vector."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        Dxf = np.zeros(3 * [self.dimension])  # initialize Derivative as 3-tensor of size NxNxN
        for (i, f_i) in enumerate(self.coordinates):
            # get second derivatives (Hessian matrices) for f_i and embed each into the full derivative of f.
            Dxf[np.ix_([i], self.stateIndexByCoordinate[
                i], self.stateIndexByCoordinate[
                           i])] = f_i.dx2(stateByCoordinate[i], parameterByCoordinate[i])
        return Dxf

    @verify_call
    def dxdiff(self, x, *parameter, diffIndex=None):
        """Return the second derivative of the HillModel vector field with respect to the state and parameter vectors (once each)
        as a rank-3 tensor. The i-th matrix of this tensor is the matrix of mixed partials of the i-th coordinate of f.
        NOTE: This function is not vectorized. It assumes x and parameter represent a single state and parameter vector."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        if diffIndex is None:  # return the full derivative wrt all parameters
            Dpxf = np.zeros(2 * [self.dimension] + [self.nParameter])  # initialize Derivative as 3-tensor of size NxNxM
            for (i, f_i) in enumerate(self.coordinates):
                Dpxf[np.ix_([i], self.stateIndexByCoordinate[i], self.parameterIndexByCoordinate[i])] = f_i.dxdiff(
                    stateByCoordinate[i], parameterByCoordinate[
                        i])  # insert derivative of this coordinate
            return Dpxf
        else:
            raise IndexError('selective differentiation indices is not yet implemented')  # this isn't implemented yet

    @verify_call
    def diff2(self, x, *parameter, diffIndex=None):
        """Return the second derivative of the HillModel vector field with respect to parameter vector (twice)
        as a rank-3 tensor. The i-th matrix of this tensor is the Hessian matrix of the i-th coordinate of f with respect
        to x. NOTE: This function is not vectorized. It assumes x and parameter represent a single state and parameter vector."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        if diffIndex is None:  # return the full derivative with respect to all state and parameter vectors
            Dppf = np.zeros([self.dimension] + 2 * [self.nParameter])  # initialize Derivative as 3-tensor of size NxMxM
            for (i, f_i) in enumerate(self.coordinates):
                Dppf[np.ix_([i], self.parameterIndexByCoordinate[i], self.parameterIndexByCoordinate[i])] = f_i.diff2(
                    stateByCoordinate[i], parameterByCoordinate[
                        i])  # insert derivative of this coordinate
            return Dppf
        else:
            raise IndexError('selective differentiation indices is not yet implemented')  # this isn't implemented yet

    @verify_call
    def dx3(self, x, *parameter):
        """Return the third derivative of the HillModel vector field with respect to x (three times) as a rank-4 tensor. The i-th
        rank-3 subtensor of this tensor is the associated third derivative of the i-th coordinate of f. NOTE: This function is not vectorized.
        It assumes x and parameter represent a single state and parameter vector."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        Dxxxf = np.zeros(4 * [self.dimension])  # initialize Derivative as 4-tensor of size NxNxNxN
        for (i, f_i) in enumerate(self.coordinates):
            Dxxxf[np.ix_([i], self.stateIndexByCoordinate[i], self.stateIndexByCoordinate[i],
                         self.stateIndexByCoordinate[i])] = f_i.dx3(stateByCoordinate[i], parameterByCoordinate[
                i])  # insert derivative of this coordinate
        return Dxxxf

    @verify_call
    def dx2diff(self, x, *parameter, diffIndex=None):
        """Return the third derivative of the HillModel vector field with respect to parameters (once) and x (twice) as a rank-4 tensor. The i-th
        rank-3 subtensor of this tensor is the associated third derivative of the i-th coordinate of f. NOTE: This function is not vectorized.
        It assumes x and parameter represent a single state and parameter vector."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        if diffIndex is None:  # return the full derivative wrt all parameters
            Dpxxf = np.zeros(
                3 * [self.dimension] + [self.nParameter])  # initialize Derivative as 4-tensor of size NxNxNxM

            for (i, f_i) in enumerate(self.coordinates):
                Dpxxf[np.ix_([i], self.stateIndexByCoordinate[i], self.stateIndexByCoordinate[i],
                             self.parameterIndexByCoordinate[i])] = f_i.dx2diff(stateByCoordinate[i],
                                                                                parameterByCoordinate[
                                                                                    i])  # insert derivative of this coordinate
            return Dpxxf
        else:
            raise IndexError('selective differentiation indices is not yet implemented')  # this isn't implemented yet

    @verify_call
    def dxdiff2(self, x, *parameter, diffIndex=None):
        """Return the third derivative of the HillModel vector field with respect to parameters (twice) and x (once) as a rank-4 tensor. The i-th
        rank-3 subtensor of this tensor is the associated third derivative of the i-th coordinate of f. NOTE: This function is not vectorized.
        It assumes x and parameter represent a single state and parameter vector."""

        # unpack state and parameter vectors by component
        stateByCoordinate, parameterByCoordinate = self.unpack_by_coordinate(x, *parameter)
        if diffIndex is None:  # return the full derivative wrt all parameters
            Dppxf = np.zeros(
                2 * [self.dimension] + 2 * [self.nParameter])  # initialize Derivative as 4-tensor of size NxNxMxM
            for (i, f_i) in enumerate(self.coordinates):
                Dppxf[np.ix_([i], self.stateIndexByCoordinate[i], self.parameterIndexByCoordinate[i],
                             self.parameterIndexByCoordinate[i])] = f_i.dxdiff2(
                    stateByCoordinate[i], parameterByCoordinate[i])  # insert derivative of this coordinate
            return Dppxf
        else:
            raise IndexError('selective differentiation indices is not yet implemented')  # this isn't implemented yet

    def radii_uniqueness_existence(self, equilibrium, *parameter):
        """Return equilibria for the Hill Model by uniformly sampling for initial conditions and iterating a Newton variant.
        INPUT:
            *parameter - Evaluations for variable parameters to use for evaluating the root finding algorithm
            gridDensity - density to sample in each dimension.
            uniqueRootDigits - Number of digits to use for distinguishing between floats.
            eqBound - N-by-2 array of intervals defining a search rectangle. Initial data will be chosen uniformly here. """

        def F(x):
            """Fix parameter values in the zero finding map"""
            return self.__call__(x, *parameter)

        def DF(x):
            """Fix parameter values in the zero finding map derivative"""
            return self.dx(x, *parameter)

        DF_x = DF(equilibrium)
        D2F_x = self.dx2(equilibrium, *parameter)
        A = np.linalg.inv(DF_x)
        Y_bound = np.linalg.norm(A @ F(equilibrium))
        Z0_bound = np.linalg.norm(np.identity(len(equilibrium)) - A @ DF_x)
        Z2_bound = np.linalg.norm(A) * np.linalg.norm(D2F_x)
        if Z2_bound < 1e-16:
            Z2_bound = 1e-8  # in case the Z2 bound is too close to zero, we increase it a bit
        delta = 1 - 4 * (Z0_bound + Y_bound) * Z2_bound
        if delta < 0 or np.isnan(delta):
            return 0, 0
        max_rad = np.minimum((1 + np.sqrt(delta)) / (2 * Z2_bound),
                             0.1)  # approximations are too poor to extend further
        min_rad = (1 - np.sqrt(delta)) / (2 * Z2_bound)
        return max_rad, min_rad

    def find_equilibria(self, gridDensity, *parameter, uniqueRootDigits=10, eqBound=None):
        """Return equilibria for the Hill Model by uniformly sampling for initial conditions and iterating a Newton variant.
        INPUT:
            *parameter - Evaluations for variable parameters to use for evaluating the root finding algorithm
            gridDensity - density to sample in each dimension.
            uniqueRootDigits - Number of digits to use for distinguishing between floats.
            eqBound - N-by-2 array of intervals defining a search rectangle. Initial data will be chosen uniformly here. """

        # TODO: Include root finding method as kwarg
        parameterByCoordinate = self.unpack_parameter(
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

        # def radii_uniqueness_existence(equilibrium):
        #     DF_x = DF(equilibrium)
        #     D2F_x = self.dx2(equilibrium, *parameter)
        #     A = np.linalg.inv(DF_x)
        #     Y_bound = np.linalg.norm(A @ F(equilibrium))
        #     Z0_bound = np.linalg.norm(np.identity(len(equilibrium)) - A @ DF_x)
        #     Z2_bound = np.linalg.norm(A) * np.linalg.norm(D2F_x)
        #     if Z2_bound < 1e-16:
        #         Z2_bound = 1e-8  # in case the Z2 bound is too close to zero, we increase it a bit
        #     delta = 1 - 4 * (Z0_bound + Y_bound) * Z2_bound
        #     if delta < 0:
        #         return 0, 0
        #     max_rad = (1 + np.sqrt(delta)) / (2 * Z2_bound)
        #     min_rad = (1 - np.sqrt(delta)) / (2 * Z2_bound)
        #     return max_rad, min_rad

        # build a grid of initial data for Newton algorithm
        if eqBound is None:  # use the trivial equilibrium bounds
            eqBound = np.array(
                list(map(lambda f_i, parm: f_i.eq_interval(parm), self.coordinates, parameterByCoordinate)))
        coordinateIntervals = [np.linspace(*interval, num=gridDensity) for interval in eqBound]
        evalGrid = np.meshgrid(*coordinateIntervals)
        X = np.column_stack([G_i.flatten() for G_i in evalGrid])

        # Apply rootfinding algorithm to each initial condition
        solns = list(
            filter(lambda root: root.success and eq_is_positive(root.x), [find_root(F, DF, x, diagnose=True)
                                                                          for x in
                                                                          X]))  # return equilibria which converged
        if solns:
            equilibria = np.row_stack([root.x for root in solns])  # extra equilibria as vectors in R^n
            equilibria = np.unique(np.round(equilibria, uniqueRootDigits), axis=0)  # remove duplicates
            # equilibria = np.unique(np.round(equilibria/10**np.ceil(log(equilibria)),
            #                                uniqueRootDigits)*10**np.ceil(log(equilibria)), axis=0)

            if len(equilibria) > 1:
                all_equilibria = equilibria
                radii = np.zeros(len(all_equilibria))
                unique_equilibria = all_equilibria
                for i in range(len(all_equilibria)):
                    equilibrium = all_equilibria[i]
                    max_rad, min_rad = self.radii_uniqueness_existence(equilibrium, *parameter)
                    radii[i] = max_rad

                radii2 = radii
                for i in range(len(all_equilibria)):
                    equilibrium1 = all_equilibria[i, :]
                    radius1 = radii[i]
                    j = i + 1
                    while j < len(radii2):
                        equilibrium2 = unique_equilibria[j, :]
                        radius2 = radii2[j]
                        if np.linalg.norm(equilibrium1 - equilibrium2) < np.maximum(radius1, radius2):
                            # remove one of the two from
                            unique_equilibria = np.delete(unique_equilibria, j, 0)
                            radii2 = np.delete(radii2, j, 0)
                        else:
                            j = j + 1
                equilibria = unique_equilibria
            return np.row_stack([find_root(F, DF, x) for x in equilibria])  # Iterate Newton again to regain lost digits
        else:
            return None

    @verify_call
    def saddle_though_arc_length_cont(self, equilibrium, parameter, parameter_bound):
        """Return equilibria for the Hill Model by uniformly sampling for initial conditions and iterating a Newton variant.
        INPUT:
            *parameter - Evaluations for variable parameters to use for evaluating the root finding algorithm
            gridDensity - density to sample in each dimension.
            uniqueRootDigits - Number of digits to use for distinguishing between floats.
            eqBound - N-by-2 array of intervals defining a search rectangle. Initial data will be chosen uniformly here. """

        def F(x, param):
            """Fix parameter values in the zero finding map"""
            return self.__call__(x, param)

        def DF(x, param):
            """Fix parameter values in the zero finding map derivative"""
            return self.dx(x, param)

        def D_lambda_F(x, param):
            return self.diff(x, param)

        def Jac(x, param):
            return np.array([DF(x, param), D_lambda_F(x, param)])

        def Newton_loc(x, param):
            iter = 0
            while np.linalg.norm(F(x, param)) < 10 ** -14 and iter < 20:
                iter = iter + 1
                step = np.linalg.solve(DF(x, param), F(x, param))
                x = x - step[:-1]
                param = param - step[-1]
            return x, param

        def arc_length_step(x, param, direction):
            step_size = 10 ** -6
            tangent = linalg.null_space(Jac(x, param))
            if tangent[-1] * direction < 0:
                tangent = -1 * tangent
            new_x = x + step_size * tangent[:-1]
            new_par = param + step_size * tangent[-1]
            [x, param] = Newton_loc(new_x, new_par)
            if np.abs(np.log(np.linalg.det(DF(x, param)))) > 10 and np.linalg.norm(D_lambda_F(x, param)) > 0.9:
                is_saddle = True
            else:
                is_saddle = False
            return x, param, is_saddle

        if parameter < parameter_bound:
            direction = +1
        else:
            direction = -1
        is_saddle = False
        while not is_saddle and (parameter - parameter_bound) * direction < 0:
            equilibrium, parameter, is_saddle = arc_length_step(equilibrium, parameter, direction)

        return equilibrium, parameter, is_saddle
