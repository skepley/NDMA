"""
A saddle-node bifurcation class and related functionality.

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 5/18/20; Last revision: 5/18/20
"""
import numpy as np
from scipy import optimize
from hill_model import find_root, ezcat


class SaddleNode:
    """A instance of a saddle-node bifurcation problem for a HillModel with 1 hill coefficient parameter which is
    shared by all HillComponents"""

    def __init__(self, hillModel, phaseCondition=lambda v: np.linalg.norm(v) - 1,
                 phaseConditionDerivative=lambda v: v / np.linalg.norm(v)):
        """Construct an instance of a saddle-node problem for specified HillModel"""

        self.model = hillModel
        self.phaseCondition = phaseCondition
        self.diffPhaseCondition = phaseConditionDerivative
        self.mapDimension = 2 * hillModel.dimension + 1  # dimension of the zero finding map

    def __call__(self, parameter, n0):
        """Temporary call method for SaddleNode class"""

        equilibria = self.model.find_equilibria(10, n0, parameter)
        initial_eigenvector = np.array([1, -.7])
        saddleNodePoints = list(filter(lambda root: root.success,
                                       [self.root(parameter,
                                                  ezcat(equilibria[:, j], initial_eigenvector, n0))
                                        for j in
                                        range(equilibria.shape[1])]))  # return equilibria which converged
        hillCoefficients = np.concatenate(([sol.x[-1] for sol in saddleNodePoints], np.array([np.inf])))
        return np.min(hillCoefficients)

    def unpack_components(self, u):
        """Unpack the input vector for a SaddleNode problem into 3 component vectors of the form (x, v, n)"""
        return u[0:self.model.dimension], u[self.model.dimension:-1], u[-1]

    def root(self, parameter, u0):
        """Attempts to return a single root of the SaddleNode rootfinding problem"""

        root = find_root(lambda u: self.zero_map(u, parameter), lambda u: self.diff_zero_map(u, parameter),
                         u0,
                         diagnose=True)
        return root

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

    def call_grid(self, parameter, gridBounds=(2, 4), nIter=10):
        # to avoid getting to negative parameter values, we just return infinity if the parameters are biologically not good
        if np.any(parameter < 0):
            return np.inf

        # set multiple values of the Hill coefficient and test for the convergence to a saddle node
        n0Grid = np.linspace(*gridBounds, nIter)
        minHill = np.inf
        kIter = 1
        while kIter < nIter and minHill == np.inf:
            minHill = self(parameter, n0Grid[kIter])
            kIter += 1
        return minHill

    def find_minimizer(self, parameter):
        """Find a minimizer of a given loss function with respect to remaining parameters via gradient descent"""

        # TODO: Add the ability to return the entire orbit of the gradient descent algorithm

        def local_function(parameter):
            return self.call_grid(parameter)

        minima = optimize.minimize(local_function, parameter, method='nelder-mead', options={'xatol': 1e-2})
        return minima

    def diff_zero_map(self, u, parameter):
        """Evaluate the derivative of the zero finding map. This is a matrix valued function of the form
        Dg: R^{2n+1} ---> M_{2n+1}(R).
        INPUT: u = (x, v, hillCoefficient) where x is a state vector, v a tangent vector."""

        # unpack input vector and set dimensions for Jacobian blocks
        stateVector, tangentVector, hillCoefficient = self.unpack_components(u)  # unpack input vector
        n = self.model.dimension
        fullParameter = self.model.parse_parameter(hillCoefficient,
                                                   parameter)  # insert copies of HillCoefficient into parameter vector
        parameterByCoordinate = self.model.unpack_variable_parameters(
            fullParameter)  # unpack full parameter vector by coordinate
        Dg = np.zeros([self.mapDimension, self.mapDimension])  # initialize (2n+1)-by-(2n+1) matrix
        Df = self.model.dx(stateVector, hillCoefficient,
                           parameter)  # store derivative of vector field which appears in 2 blocks
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
