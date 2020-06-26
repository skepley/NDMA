"""
A saddle-node bifurcation class and related functionality.

    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 5/18/20; Last revision: 6/25/20
"""
import numpy as np
from scipy import optimize
from hill_model import *


class SaddleNode:
    """A constraint class for working with HillModels along surfaces of saddle-node bifurcations"""

    def __init__(self, hillModel, phaseCondition=lambda v: np.linalg.norm(v) - 1,
                 phaseConditionDerivative=lambda v: v / np.linalg.norm(v)):
        """Construct an instance of a saddle-node problem for specified HillModel"""

        self.model = hillModel
        self.phaseCondition = phaseCondition
        self.diffPhaseCondition = phaseConditionDerivative
        self.mapDimension = 2 * hillModel.dimension + 1  # degrees of freedom of the constraint

    def __call__(self, u):
        """Evaluation function for the saddle-node bifurcation constraints. This is a map of the form
        g: R^{2n + m} ---> R^{2n + 1} which is zero if and only if u corresponds to a saddle-node bifurcation point.

        INPUT: u = (x, v, p) where x is a state vector, v a tangent vector, and p a parameter vector."""

        stateVector, tangentVector, fullParameter = self.unpack_components(u)  # unpack input vector
        g1 = self.model(stateVector,
                        fullParameter)  # this is zero iff x is an equilibrium for the system at parameter value n

        g2 = self.model.dx(stateVector,
                           fullParameter) @ tangentVector  # this is zero iff v lies in the kernel of Df(x, n)
        g3 = self.phaseCondition(tangentVector)  # this is zero iff v satisfies the phase condition
        return ezcat(g1, g2, g3)

    def find_saddle_node(self, freeParameterIndex, *parameter, freeParameterValues=None, uniqueDigits=5):
        """Attempt to find isolated saddle-node points along the direction of the parameter at the
        freeParameterIndex. All other parameters are fixed. This is done by Newton iteration starting at eqch
        equilibrium found for the initial parameter. The function returns only values of the free parameter or returns
        None if it fails to find any"""

        equilibria = self.model.find_equilibria(10, *parameter)
        fullParameter = ezcat(*parameter)  # concatenate input parameter to full ordered parameter vector
        fixedParameter = fullParameter[[idx for idx in range(len(fullParameter)) if idx != freeParameterIndex]]
        if freeParameterValues is None:
            freeParameter = fullParameter[freeParameterIndex]
        else:
            freeParameter = ezcat(fullParameter[freeParameterIndex], freeParameterValues)

        def curry_parameters(u):
            x, v, p0 = self.unpack_components(u)  # layout components in (R^n, R^n, R)
            return ezcat(x, v, np.insert(fixedParameter, freeParameterIndex,
                                         p0))  # embed into (R^n, R^n, R^m) by currying fixed Parameters

        def init_eigenvector(equilibrium, rho):
            """Choose an initial eigenvector for the saddle-node root finding problem"""
            p = np.insert(fixedParameter, freeParameterIndex, rho)
            tangentVector = -np.linalg.solve(self.model.dx(equilibrium, p),
                                             self.model.diff(equilibrium, p, diffIndex=freeParameterIndex))
            return tangentVector / np.linalg.norm(tangentVector)

        def root(u0):
            """Attempts to return a single root of the SaddleNode rootfinding problem"""

            return find_root(lambda u: self.__call__(curry_parameters(u)),
                             lambda u: self.diff(curry_parameters(u), diffIndex=freeParameterIndex),
                             u0,
                             diagnose=True)

        saddleNodePoints = []
        for parmValue in freeParameter:
            saddleNodeZeros = list(filter(lambda soln: soln.success,
                                          [root(ezcat(equilibria[:, j], init_eigenvector(equilibria[:, j], parmValue),
                                                      parmValue))
                                           for j in
                                           range(equilibria.shape[1])]))  # return equilibria which converged
            if saddleNodeZeros:
                addSols = np.array([sol.x[-1] for sol in saddleNodeZeros])
                print(addSols)
                saddleNodePoints = ezcat(saddleNodePoints, addSols[addSols > 0])

        return np.unique(np.round(saddleNodePoints, uniqueDigits))  # remove duplicates and return values

    def unpack_components(self, u):
        """Unpack the input vector for a SaddleNode problem into 3 component vectors of the form (x, v, p) where:

            x is the state vector in R^n
            v is the tangent vector in R^n
            p is the parameter vector in R^m
            """
        n = self.model.dimension
        return u[:n], u[n:2 * n], u[2 * n:]

    def diff(self, u, diffIndex=None):
        """Evaluate the derivative of the zero finding map. This is a matrix valued function of the form
        Dg: R^{2n+1} ---> M_{2n+1}(R).
        INPUT: u = (x, v, hillCoefficient) where x is a state vector, v a tangent vector."""

        # unpack input vector and set dimensions for Jacobian blocks
        n = self.model.dimension
        parameterDim = self.model.nVariableParameters if diffIndex is None else len(ezcat(diffIndex))
        mapDimension = 2 * n + parameterDim

        stateVector, tangentVector, fullParameter = self.unpack_components(u)  # unpack input vector
        Dg = np.zeros([mapDimension, mapDimension])  # initialize (2n+1)-by-(2n+1) matrix
        Dxf = self.model.dx(stateVector, fullParameter)  # store derivative of vector field which appears in 2 blocks

        # BLOCK ROW 1
        Dg[0:n, 0:n] = Dxf  # block - (1,1)
        # block - (1, 2) is an n-by-n zero block
        Dg[0:n, -1] = self.model.diff(stateVector, fullParameter, diffIndex=diffIndex)  # block - (1,3)
        # BLOCK ROW 2
        Dg[n:2 * n, 0:n] = np.einsum('ijk,j', self.model.dx2(stateVector, fullParameter),
                                     tangentVector)  # block - (2,1)
        Dg[n:2 * n, n:2 * n] = Dxf  # block - (2,2)
        Dg[n:2 * n, -1] = np.einsum('ij, j',
                                    self.model.dxdiff(stateVector, fullParameter, diffIndex=diffIndex),
                                    tangentVector)  # block - (2,3)
        # BLOCK ROW 3
        # block - (3, 1) is a 1-by-n zero block
        Dg[-1, n:2 * n] = self.diffPhaseCondition(tangentVector)  # block - (3,2)
        # block - (3, 1) is a 1-by-1 zero block
        return Dg

    def diff2(self, u, diffIndex=None):
        """Evaluate the second derivative of the zero finding map. This is a function of the form
        D^2g: R^{2n+1} ---> M_{2n+1}(R).
        INPUT: u = (x, v, hillCoefficient) where x is a state vector, v a tangent vector."""

        # unpack input vector and set dimensions for Jacobian blocks
        n = self.model.dimension
        parameterDim = self.model.nVariableParameters if diffIndex is None else len(ezcat(diffIndex))
        mapDimension = 2 * n + parameterDim

        stateVector, tangentVector, fullParameter = self.unpack_components(u)  # unpack input vector
        Dg = np.zeros([mapDimension, mapDimension, mapDimension])  # initialize (2n+1)-by-(2n+1) matrix
        Dxxf = self.model.dx2(stateVector, fullParameter)  # 3D tensor
        Dxpf = self.model.dxdiff(stateVector, fullParameter, diffIndex=diffIndex)
        Dxxxf = self.model.dx3(stateVector, fullParameter)
        Dxxpf = self.model.dx2diff(stateVector, fullParameter, diffIndex=diffIndex)
        Dxppf = self.model.dxdiff2(stateVector, fullParameter, diffIndex=[diffIndex, diffIndex])
        Dppf = self.model.diff2(stateVector, fullParameter, diffIndex=[diffIndex, diffIndex])

        index1 = np.range(n)
        index2 = index1 + 2
        index3 = 2*n

        # ROW 1
        Dg[index1, index1, index1] = Dxxf  # block - (1,1,1)
        Dg[index1, index1, index3] = Dxpf
        Dg[index1, index3, index1] = Dxpf
        Dg[index1, index3, index3] = Dppf

        # BLOCK ROW 2 - derivatives of Dxf*v
        Dg[index2, index1, index1] = np.einsum('ijkl,j', Dxxxf, tangentVector)
        Dg[index2, index1, index2] = Dxxf
        Dg[index2, index1, index3] = np.einsum('ijk,j', Dxxpf, tangentVector)
        Dg[index2, index2, index1] = Dxxf
        Dg[index2, index2, index3] = Dxpf
        Dg[index2, index3, index1] = np.einsum('ijk,j', Dxxpf, tangentVector)
        Dg[index2, index3, index2] = Dxpf
        Dg[index2, index3, index3] = Dxppf

        # BLOCK ROW 3
        # block - (3, :, :) is a 1-by-dim-by-dim zero block because the phase condition is linear

        return Dg

    def call_grid(self, parameter, gridBounds=(2, 4), nIter=10):
        # # to avoid getting to negative parameter values, we just return infinity if the parameters are biologically not good
        # if np.any(parameter < 0):
        #     return np.inf
        #
        # # set multiple values of the Hill coefficient and test for the convergence to a saddle node
        # n0Grid = np.linspace(*gridBounds, nIter)
        # minHill = np.inf
        # kIter = 1
        # while kIter < nIter and minHill == np.inf:
        #     minHill = self(parameter, n0Grid[kIter])
        #     kIter += 1
        # return minHill
        print('This function needs to be updated before being called')
        return

    def find_minimizer(self, parameter):
        """Find a minimizer of a given loss function with respect to remaining parameters via gradient descent"""

        # TODO: 1. This needs to be overhauled to match the fully general class
        #
        # def local_function(parameter):
        #     return self.call_grid(parameter)
        #
        # minima = optimize.minimize(local_function, parameter, method='nelder-mead', options={'xatol': 1e-2})
        # return minima

        print('This function needs to be updated before being called')
        return
