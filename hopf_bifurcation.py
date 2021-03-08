"""
Hopf bifurcation class and related functionality.

    Author: Elena Queirolo
    email: elena.queirolo@rutgers.edu
    Date: 03/07/2021
"""
import numpy as np
from scipy import optimize
from hill_model import *

"""
A Hopf bifurcation occurs when the Jacobian computed at the equilibrium has an imaginary eigenvalue, that is 
Df(x, lambda) v = imag beta * v,
or equivalently, dividing this into its real and imaginary part, such that v = v1 + imag v2, we get
Df(x, lambda) v1 = - beta * v2
Df(x, lambda) v2 = beta * v1
We also want the phase condition 
IS NOT || v1 + imag v2 || = 1, or equivalently sqrt( v1 ^2 + v2 ^2) = 1, where the square is interpreted as an inner product
but is
< l, v1>  =0 and <l, v2> =1 
"""


class HopfBifurcation:
    """A constraint class for working with HillModels along surfaces of Hopf bifurcations"""

    def __init__(self, hillModel, phaseCondition=lambda v1, v2, l_vec: [np.inner(l_vec, v1), np.inner(l_vec, v2) - 1],
                 phaseConditionDerivative=lambda v1, v2, l_vec: np.array([[ezcat(l_vec, 0*l_vec)], [ezcat(0*l_vec, l_vec)]])):
        """Construct an instance of a Hopf bifurcation problem for specified HillModel"""
        self.model = hillModel
        self.mapDimension = 3 * hillModel.dimension + 1  # degrees of freedom of the constraint
        if phaseCondition.__code__.co_argcount is 3:
            Random_direction = np.random.random(size=hillModel.dimension)
            phaseCondition = lambda v1, v2: phaseCondition(v1, v2, Random_direction)
            phaseConditionDerivative = lambda v1, v2: phaseConditionDerivative(v1, v2, Random_direction)
        self.phaseCondition = phaseCondition
        self.diffPhaseCondition = phaseConditionDerivative

    def __call__(self, u):
        """Evaluation function for the saddle-node bifurcation constraints. This is a map of the form
        g: R^{2n + m} ---> R^{2n + 1} which is zero if and only if u corresponds to a saddle-node bifurcation point.

        INPUT: u = (x, v, p) where x is a state vector, v a tangent vector, and p a parameter vector."""

        stateVector, tangentVector1, tangentVector2, eig, fullParameter = self.unpack_components(
            u)  # unpack input vector
        g1 = self.model(stateVector,
                        fullParameter)  # this is zero iff x is an equilibrium for the system at parameter value n

        g2a = self.model.dx(stateVector,
                            fullParameter) @ tangentVector1 + eig * tangentVector2
        g2b = self.model.dx(stateVector,
                            fullParameter) @ tangentVector2 - eig * tangentVector1
        # these two are zero if Df(x, n) has an imaginary eigenvalue
        g3 = self.phaseCondition(tangentVector1, tangentVector2)  # this is zero iff v satisfies the phase condition
        return ezcat(g1, g2a, g2b, g3)

    def find_Hopf_bif(self, freeParameterIndex, *parameter, equilibria=None, freeParameterValues=None, uniqueDigits=5,
                      flag_return=0):
        """Attempt to find isolated Hopf bifurcation points along the direction of the parameter at the
        freeParameterIndex. All other parameters are fixed. This is done by Newton iteration starting at each
        equilibrium found for the initial parameter. The function returns only values of the free parameter or returns
        None if it fails to find any

        Inputs:
            equilibria - Specify a list of equilibria to use as initial guesses. If none are specified it uses any equilibria
            which are found using the find_equilibria method.
        flag_return asks for complete info on the parameters and solutions at the saddle node"""

        if equilibria is None:  # start the saddle node search at the equilibria returned by the find_equilbria method
            equilibria = self.model.find_equilibria(10, *parameter)
        if equilibria is None:
            print('No equilibria found for parameter: {0} \n'.format(parameter))
            return []
        fullParameter = ezcat(*parameter)  # concatenate input parameter to full ordered parameter vector
        fixedParameter = fullParameter[[idx for idx in range(len(fullParameter)) if idx != freeParameterIndex]]
        if freeParameterValues is None:
            freeParameter = [fullParameter[freeParameterIndex]]
        else:
            freeParameter = ezcat(fullParameter[freeParameterIndex], freeParameterValues)

        def curry_parameters(u):
            x, v1, v2, beta, p0 = self.unpack_components(u)  # layout components in (R^n, R^n, R)
            return ezcat(x, v1, v2, beta, np.insert(fixedParameter, freeParameterIndex,
                                         p0))  # embed into (R^n, R^n, R^n, R, R^m) by currying fixed Parameters

        def init_eigenvector(equilibrium, rho):
            """Choose an initial eigenvector for the Hopf bifurcation root finding problem"""
            # TODO: ask Shane what would be a fitting element in here... what problem is tangentVector solving here?
            p = np.insert(fixedParameter, freeParameterIndex, rho)
            tangentVector = -np.linalg.solve(self.model.dx(equilibrium, p),
                                             self.model.diff(equilibrium, p, diffIndex=freeParameterIndex))
            # in R, we still have one choice of orientation once we fix the norm - this is done by checking the sign of
            # the first element of the eigenvector. If we extend to C, more work will be needed
            if tangentVector[0] < 0:
                tangentVector = - tangentVector
            return tangentVector / np.linalg.norm(tangentVector), tangentVector / np.linalg.norm(tangentVector)

        def root(u0):
            """Attempts to return a single root of the HopfBifurcation rootfinding problem"""

            return find_root(lambda u: self.__call__(curry_parameters(u)),
                             lambda u: self.diff(curry_parameters(u), diffIndex=freeParameterIndex),
                             u0,
                             diagnose=True)

        if flag_return is 0:
            HopfPoints = []
        else:
            HopfPoints = np.empty((0, self.mapDimension))

        if is_vector(equilibria):
            equilibria = equilibria[np.newaxis, :]

        for parmValue in freeParameter:
            HopfBifurcationZero = list(filter(lambda soln: soln.success,
                                          [root(ezcat(equilibria[j, :], init_eigenvector(equilibria[j, :], parmValue),
                                                      parmValue))
                                           for j in
                                           range(equilibria.shape[0])]))  # return equilibria which converged
            if HopfBifurcationZero and flag_return is 0:
                addSols = np.array([sol.x[-1] for sol in HopfBifurcationZero])
                HopfPoints = ezcat(HopfPoints, addSols[addSols > 0])
            elif HopfBifurcationZero:
                addSols = np.array([sol.x for sol in HopfBifurcationZero])
                HopfPoints = np.append(HopfPoints, addSols, axis=0)

        if flag_return is 0 and len(HopfPoints) > 0:
            return np.unique(np.round(HopfPoints, uniqueDigits))
        elif len(HopfPoints) > 0:
            return np.unique(np.round(HopfPoints, uniqueDigits), axis=0)  # remove duplicates and return values
        else:  # return empty array. nothing found
            return HopfPoints

    def unpack_components(self, u):
        """Unpack the input vector for a HopfBifurcation problem into 4 component vectors of the form
        (x, v1, v2, beta, p) where:

            x is the state vector in R^n
            v1 is the real component of the tangent vector in R^n
            v2 is the imaginary component of the tangent vector in R^n
            beta is the imaginary eigenvalue
            p is the parameter vector in R^m
            """
        n = self.model.dimension
        return u[:n], u[n:2 * n], u[2 * n:3 * n], u[3 * n], u[1 + 3 * n:]

    def diff(self, u, diffIndex=None):
        """Evaluate the derivative of the zero finding map. This is a matrix valued function of the form
        Dg: R^{3n+1+M} ---> M_{3n+1 x 3n+1+M}(R).
        INPUT: u = (x, v1, v2, beta, all parameters) where x is a state vector, v1 is the real component of a
        tangent vector, v2 the imaginary component, beta the imaginary eigenvalue."""

        # unpack input vector and set dimensions for Jacobian blocks
        n = self.model.dimension
        parameterDim = self.model.nVariableParameter if diffIndex is None else len(ezcat(diffIndex))
        dimension_space = 3 * n + 1 + parameterDim
        mapDimension = 3 * n + 2

        stateVector, tangentVector1, tangentVector2, beta, fullParameter = self.unpack_components(u)  # unpack input vector
        Dg = np.zeros([mapDimension, dimension_space])  # initialize (3n+1)-by-(3n+1+M) matrix
        Dxf = self.model.dx(stateVector, fullParameter)  # store derivative of vector field which appears in 2 blocks
        Dxpf = self.model.dxdiff(stateVector, fullParameter, diffIndex=diffIndex)
        Dpf = self.model.diff(stateVector, fullParameter, diffIndex=diffIndex)
        Dxxf = self.model.dx2(stateVector, fullParameter)  # 3D tensor
        Id_v = np.identity(n)
        Diff_PC = self.diffPhaseCondition(tangentVector1, tangentVector2) # it's

        # the indices of x v1, v2, beta and p respectively
        index_x = np.arange(n)
        index_v1 = np.arange(n) + n
        index_v2 = np.arange(n) + 2 * n
        index_beta = 3 * n
        index_par = 3 * n + 1 + np.arange(parameterDim)
        index_PC = 3 * n + 1 + np.array([0, 1])
        if parameterDim is 1:
            # BLOCK ROW 1
            Dg[np.ix_(index_x, index_x)] = Dxf  # block - (1,1)
            # block - (1, 2) is an n-by-n zero block
            # block - (1,3)
            Dg[index_x, index_par] = Dpf
            # BLOCK ROW 2
            Dg[np.ix_(index_v1, index_x)] = np.einsum('ijk,j', Dxxf, tangentVector1)  # block - (2,1)
            Dg[np.ix_(index_v1, index_v1)] = Dxf  # block - (2,2)
            Dg[np.ix_(index_v1, index_v2)] = beta * Id_v # block - (2,3)
            Dg[index_v1, index_beta] = tangentVector2 # block - (2,4)
            Dg[index_v1, index_par] = np.einsum('ij, j', Dxpf, tangentVector1)  # block - (2,end)
            # BLOCK ROW 3
            Dg[np.ix_(index_v2, index_x)] = np.einsum('ijk,j', Dxxf, tangentVector2)  # block - (3,1)
            Dg[np.ix_(index_v2, index_v1)] = - beta * Id_v  # block - (3,3)
            Dg[np.ix_(index_v2, index_v2)] = Dxf  # block - (3,2)
            Dg[index_v2, index_beta] = - tangentVector1 # block - (3,4)
            Dg[index_v2, index_par] = np.einsum('ij, j', Dxpf, tangentVector2)  # block - (3,end)

            # BLOCK ROW 4
            # block - (4, 1) is a 1-by-n zero block
            Dg[np.ix_(index_PC, ezcat(index_v1, index_v2))] = Diff_PC  # block - (4, 2:3)
            # block - (4, 4:5) is a 1-by-1 zero block
        else:
            # BLOCK ROW 1
            Dg[np.ix_(index_x, index_x)] = Dxf  # block - (1,1)
            # block - (1, 2) is an n-by-n zero block
            # block - (1,3)
            Dg[np.ix_(index_x, index_par)] = Dpf
            # BLOCK ROW 2
            Dg[np.ix_(index_v1, index_x)] = np.einsum('ijk,j', Dxxf, tangentVector1)  # block - (2,1)
            Dg[np.ix_(index_v1, index_v1)] = Dxf  # block - (2,2)
            Dg[np.ix_(index_v1, index_v2)] = beta * Id_v  # block - (2,3)
            Dg[index_v1, index_beta] = tangentVector2  # block - (2,4)
            Dg[np.ix_(index_v1, index_par)] = np.einsum('ij, j', Dxpf, tangentVector1)  # block - (2,end)
            # BLOCK ROW 3
            Dg[np.ix_(index_v2, index_x)] = np.einsum('ijk,j', Dxxf, tangentVector2)  # block - (3,1)
            Dg[np.ix_(index_v2, index_v1)] = - beta * Id_v  # block - (3,3)
            Dg[np.ix_(index_v2, index_v2)] = Dxf  # block - (3,2)
            Dg[index_v2, index_beta] = - tangentVector1  # block - (3,4)
            Dg[np.ix_(index_v2, index_par)] = np.einsum('ij, j', Dxpf, tangentVector2)  # block - (3,end)

            # BLOCK ROW 4
            # block - (4, 1) is a 1-by-n zero block
            Dg[np.ix_(index_PC, ezcat(index_v1, index_v2))] = Diff_PC  # block - (4, 2:3)
            # block - (4, 4:5) is a 1-by-1 zero block
        return Dg

    def diff2(self, u, diffIndex=None):
        """Evaluate the second derivative of the zero finding map. This is a function of the form
        D^2g: R^{3n+1+M} ---> R^{3n+2 x 3n+1+M x 3n+1+M}, M length of diffIndex.
        INPUT: u = (x, v1, v2, beta, all parameters) where x is a state vector, v1 is the real component of a
        tangent vector, v2 the imaginary component, beta the imaginary eigenvalue."""

        # unpack input vector and set dimensions for Hessian blocks
        n = self.model.dimension
        parameterDim = self.model.nVariableParameter if diffIndex is None else len(ezcat(diffIndex))
        dimension_space = 3 * n + 1 + parameterDim
        mapDimension = 3 * n + 2

        stateVector, tangentVector1, tangentVector2, beta, fullParameter = self.unpack_components(u) # unpack input vector
        Dg = np.zeros([mapDimension, dimension_space, dimension_space])  # initialize (3n+2) x (3n+1+M) x (3n+1+M) matrix
        Dxxf = self.model.dx2(stateVector, fullParameter)  # 3D tensor
        Dxpf = self.model.dxdiff(stateVector, fullParameter, diffIndex=diffIndex)
        Dxxxf = self.model.dx3(stateVector, fullParameter)
        Dxxpf = self.model.dx2diff(stateVector, fullParameter, diffIndex=diffIndex)
        Dxppf = self.model.dxdiff2(stateVector, fullParameter, diffIndex=[diffIndex, diffIndex])
        Dppf = self.model.diff2(stateVector, fullParameter, diffIndex=[diffIndex, diffIndex])
        Id_v = np.identity(n)

        # the indices of x v and p respectively
        index_x = np.array(range(n))
        index_v1 = np.array(range(n)) + n
        index_v2 = np.arange(n) + 2 * n
        index_beta = 3 * n
        index_par = 3 * n + 1 + np.arange(parameterDim)
        index_PC = 3 * n + 1 + np.array([0, 1])

        # DxDF
        Dg[np.ix_(index_x, index_x, index_x)] = Dxxf
        Dg[np.ix_(index_v1, index_x, index_x)] = np.einsum('ijkl,j', Dxxxf, tangentVector1)
        Dg[np.ix_(index_v1, index_x, index_v1)] = Dxxf
        Dg[np.ix_(index_v2, index_x, index_x)] = np.einsum('ijkl,j', Dxxxf, tangentVector2)
        Dg[np.ix_(index_v2, index_x, index_v2)] = Dxxf

        # Dv1DF
        Dg[np.ix_(index_v1, index_v1, index_x)] = Dxxf
        Dg[np.ix_(index_v2, index_v1), index_beta] = - Id_v

        # Dv2DF
        Dg[np.ix_(index_v2, index_v2, index_x)] = Dxxf
        Dg[np.ix_(index_v1, index_v2), index_beta] = Id_v

        # DbetaDF
        Dg[np.ix_(index_v1), index_beta, np.ix_(index_v2)] = Id_v
        Dg[np.ix_(index_v2), index_beta, np.ix_(index_v1)] = - Id_v

        if parameterDim is 1:
            # ROW 1
            Dg[np.ix_(index_x, index_x), index_par] = Dxpf
            Dg[np.ix_(index_v1, index_x), index_par] = np.einsum('ijkl,j', Dxxpf, tangentVector1)
            Dg[np.ix_(index_v2, index_x), index_par] = np.einsum('ijkl,j', Dxxpf, tangentVector2)

            Dg[np.ix_(index_v1, index_v1), index_par] = Dxpf
            Dg[np.ix_(index_v2, index_v2), index_par] = Dxpf

            Dg[np.ix_(index_x), index_par, np.ix_(index_x)] = Dxpf
            Dg[index_par, index_par, index_par] = Dppf
            Dg[np.ix_(index_v1), index_par, np.ix_(index_x)] = np.einsum('ijk,j', Dxxpf, tangentVector1)
            Dg[np.ix_(index_v1), index_par, np.ix_(index_v1)] = Dxpf
            Dg[np.ix_(index_v1), index_par, index_par] = np.einsum('jk,j', Dxppf, tangentVector1)

            Dg[np.ix_(index_v2), index_par, np.ix_(index_x)] = np.einsum('ijk,j', Dxxpf, tangentVector2)
            Dg[np.ix_(index_v2), index_par, np.ix_(index_v2)] = Dxpf
            Dg[np.ix_(index_v2), index_par, index_par] = np.einsum('jk,j', Dxppf, tangentVector2)
            # BLOCK ROW 3
            # block - (4, :, :) is a 1-by-dim-by-dim zero block because the phase condition is linear
        else:

            # ROW 1
            Dg[np.ix_(index_x, index_x, index_par)] = Dxpf
            Dg[np.ix_(index_v1, index_x, index_par)] = np.einsum('ijkl,j', Dxxpf, tangentVector1)
            Dg[np.ix_(index_v2, index_x, index_par)] = np.einsum('ijkl,j', Dxxpf, tangentVector2)

            Dg[np.ix_(index_v1, index_v1, index_par)] = Dxpf
            Dg[np.ix_(index_v2, index_v2, index_par)] = Dxpf

            Dg[np.ix_(index_x, index_par, index_x)] = Dxpf
            Dg[np.ix_(index_par, index_par, index_par)] = Dppf
            Dg[np.ix_(index_v1, index_par, index_x)] = np.einsum('ijk,j', Dxxpf, tangentVector1)
            Dg[np.ix_(index_v1, index_par, index_v1)] = Dxpf
            Dg[np.ix_(index_v1, index_par, index_par)] = np.einsum('jk,j', Dxppf, tangentVector1)

            Dg[np.ix_(index_v2, index_par, index_x)] = np.einsum('ijk,j', Dxxpf, tangentVector2)
            Dg[np.ix_(index_v2, index_par, index_v2)] = Dxpf
            Dg[np.ix_(index_v2, index_par, index_par)] = np.einsum('jk,j', Dxppf, tangentVector2)

            # BLOCK ROW 3
            # block - (3, :, :) is a 1-by-dim-by-dim zero block because the phase condition is linear
        return Dg
