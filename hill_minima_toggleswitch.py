"""
Compute minima of the hill coefficient for the toggle swtich example


    Output: output
    Other files required: none
    See also: OTHER_SCRIPT_NAME,  OTHER_FUNCTION_NAME
   
    Author: Shane Kepley
    email: shane.kepley@rutgers.edu
    Date: 7/25/20; Last revision: 7/25/20
"""

from hill_model import *
from saddle_node import SaddleNode
from models import ToggleSwitch
from scipy.optimize import minimize, NonlinearConstraint, Bounds, BFGS

projectionOperator = np.array([[2 / 3, 0, 0, 0, 1 / 3, 1 / 3],
                               [0, 2 / 3, -1 / 3, 1 / 3, 0, 0],
                               [0, -1 / 3, 2 / 3, 1 / 3, 0, 0],
                               [0, 1 / 3, 1 / 3, 2 / 3, 0, 0],
                               [1 / 3, 0, 0, 0, 2 / 3, -1 / 3],
                               [1 / 3, 0, 0, 0, -1 / 3, 2 / 3]])


def mu(nonHillParameter):
    """Evaluate the mu map:
        (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2) --> (theta_1, alpha_1, beta_1, theta_2, alpha_2, beta_2)
     """
    return ezcat(nonHillParameter[3], nonHillParameter[1] / nonHillParameter[0],
                 (nonHillParameter[1] + nonHillParameter[2]) / nonHillParameter[0],
                 nonHillParameter[7], nonHillParameter[5] / nonHillParameter[4],
                 (nonHillParameter[5] + nonHillParameter[6]) / nonHillParameter[4])


def degeneracy(nonHillParameter):
    """Measures the norm of the orthogonal distance between a parameter and the degenerate parameter plane."""

    return np.linalg.norm(np.einsum('ij,i', np.eye(6) - projectionOperator, mu(nonHillParameter)))


def diff_mu(nonHillParameter):
    """Differentiate the mu map"""

    Dmu = np.zeros([6, 8])
    Dmu[0, 3] = Dmu[3, 7] = 1
    Dmu[1, 0] = -nonHillParameter[1] / nonHillParameter[0] ** 2
    Dmu[1, 1] = Dmu[2, 1] = Dmu[2, 2] = 1 / nonHillParameter[0]
    Dmu[2, 0] = -(nonHillParameter[1] + nonHillParameter[2]) / nonHillParameter[0] ** 2

    Dmu[4, 4] = -nonHillParameter[5] / nonHillParameter[4] ** 2
    Dmu[4, 5] = Dmu[5, 5] = Dmu[5, 6] = 1 / nonHillParameter[4]
    Dmu[5, 4] = -(nonHillParameter[5] + nonHillParameter[6]) / nonHillParameter[4] ** 2
    return Dmu


def diff2_mu(nonHillParameter):
    """Second derivative of the mu map"""
    gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2 = nonHillParameter
    D2mu = np.zeros([6, 8, 8])
    D2mu[1, 0, 0] = 2 * ell_1 / gamma_1 ** 3
    D2mu[1, 0, 1] = -1 / gamma_1 ** 2
    D2mu[1, 1, 0] = -1 / gamma_1 ** 2
    D2mu[2, 0, 0] = 2 * (ell_1 + delta_1) / gamma_1 ** 3
    D2mu[2, 0, 1] = -1 / gamma_1 ** 2
    D2mu[2, 0, 2] = -1 / gamma_1 ** 2
    D2mu[4, 4, 4] = 2 * ell_2 / gamma_2 ** 3
    D2mu[4, 4, 5] = -1 / gamma_2 ** 2
    D2mu[4, 5, 4] = -1 / gamma_2 ** 2
    D2mu[5, 4, 4] = 2 * (ell_2 + delta_2) / gamma_2 ** 2
    D2mu[5, 4, 5] = -1 / gamma_2 ** 2
    D2mu[5, 4, 6] = -1 / gamma_2 ** 2
    D2mu[5, 5, 4] = -1 / gamma_2 ** 2
    D2mu[5, 6, 4] = -1 / gamma_2 ** 2
    return D2mu


def minimize_hill(saddleNode, punish_degenerate=False):
    """Return the projection map onto the hill coefficient and its derivatives as callable functions.
    INPUT: A SaddleNode instance
    OUPUT: Returns 3 functions which act on vectors of the form:
        u = (x, v, hill, p) is a vector in R^(n + n + 1 + (m-1))"""

    n = saddleNode.model.dimension  # toggle switch dimension
    m = saddleNode.model.nVariableParameter  # number of variable parameters
    lossDomainDimension = 2 * n + m  # dimension of domain for loss function

    def loss(u):
        """Projection map for the hill coefficient"""

        if punish_degenerate:
            nonHillParm = u[2 * n + 1:]
            L = u[2 * n] / degeneracy(nonHillParm)
            print(u[2 * n], L)
            return L
        else:
            return u[2 * n]

    def diff_loss(u):
        """Derivative of the projection map"""

        DL = np.zeros(lossDomainDimension)
        if punish_degenerate:
            nonHillParm = u[2 * n + 1:]
            DL[2 * n] = 1 / degeneracy(nonHillParm)
            A = np.eye(6) - projectionOperator
            AA = np.einsum('ij, jk', A, A)  # A^2
            DL[2 * n + 1:] = (-2 * u[2 * n] / degeneracy(nonHillParm) ** 2) * np.einsum('i, ij, jk',
                                                                                        mu(nonHillParm), AA,
                                                                                        diff_mu(nonHillParm))
        else:
            DL[2 * n] = 1.
        return DL

    def diff2_loss(u):
        """Hessian of the projection map"""

        D2L = np.zeros(2 * [lossDomainDimension])
        if punish_degenerate:
            nonHillParm = u[2 * n + 1:]
            hillParm = u[2 * n]
            A = np.eye(6) - projectionOperator
            AA = np.einsum('ij, jk', A, A)  # A^2
            distanceToDegeneratePlane = degeneracy(nonHillParm)
            evalMu = mu(nonHillParm)
            Dmu = diff_mu(nonHillParm)
            D2mu = diff2_mu(nonHillParm)
            D2L[2 * n + 1:, 2 * n + 1:] = hillParm * ((1 / distanceToDegeneratePlane ** 3) * (np.einsum('il,ij,jk',
                                                                                                        Dmu, AA,
                                                                                                        Dmu) + np.einsum(
                'i, ij, jkl', evalMu, AA, D2mu))
                                                      - (1 / distanceToDegeneratePlane ** 5) * np.einsum(
                'i, ij, jk, l, lm, mn', evalMu, AA, Dmu, evalMu, AA, Dmu))
        return D2L

    return loss, diff_loss, diff2_loss


def saddle_node_constraint(saddleNode, orthogonal=False):
    """Return the evaluation map which characterizes constraint to the saddle-node surface and its derivatives"""

    def saddle_node_problem(u):
        """Evaluate a map of the form g : R^(2n + m) --> R^(2n + 1) such that g(x, v, hill, p) = 0 iff (hill, p) is a
        point on the saddle-node surface with corresponding equilibrium x and tangent vector v."""
        SN_eval = saddleNode(u)

        if orthogonal:
            orthogParm = np.einsum('i,i', u[5:9], u[9:])  # force coordinate parameter vectors to be orthogonal
            return ezcat(SN_eval, orthogParm)
        else:
            return SN_eval

    def saddle_node_jac(u):
        """Return the derivative of the saddle_node_problem map as a matrix of size:
        (2n + 1)-by-(2n + m)"""
        SN_diff = saddleNode.diff(u)
        if orthogonal:
            orthogParm_diff = ezcat(np.zeros(5), u[9:], u[5:9])  # derivative of dot(p1, p2)
            return np.row_stack([SN_diff, orthogParm_diff])
        else:
            return SN_diff

    def saddle_node_hess(u, h):
        """Return the second derivative of the saddle_node_problem map as a linear combination of Hessians. This is the
        action of a 3 tensor array with dimensions: [2n + 1, 2n + m, 2n + m] on a vector, h in R^(2n+1). Alternatively,
        this is a function which returns arbitrary linear combinations of Hessians for the coordinates of the saddle node
        constraint problem."""
        D2g = saddleNode.diff2(u)  # Second derivative as 3-tensor

        if orthogonal:
            orthogParm_hess = np.zeros([13, 13])
            orthogParm_hess[np.ix_(np.arange(5, 9), np.arange(9, 13))] = np.eye(4)
            orthogParm_hess[np.ix_(np.arange(9, 13), np.arange(5, 9))] = np.eye(4)
            return np.einsum('ijk, i', D2g, h[:-1]) + h[-1] * orthogParm_hess
        else:
            return np.einsum('ijk, i', D2g, h)

    return saddle_node_problem, saddle_node_jac, saddle_node_hess


decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
SN = SaddleNode(f)

# set up the minimization problem
loss, lossJacobian, lossHessian = minimize_hill(SN, punish_degenerate=True)  # minimization problem
xBounds = [[0, 10], [0, 10]]  # phase space bounds for saddle node equilibria points
vBounds = [[-5, 5], [-5, 5]]  # tangent vector coordinate bounds
hillBounds = [[1, 10]]  # hill coefficient bounds
parameterBounds = [[0, 10] for i in range(8)]  # bounds on remaining variable parameters
bounds = Bounds(*list(zip(*(xBounds + vBounds + hillBounds + parameterBounds))))
constraintFunction, constraintJacobian, constraintHessian = saddle_node_constraint(SN, orthogonal=False)
saddleNodeConstraint = NonlinearConstraint(constraintFunction, 0, 0, jac=constraintJacobian, hess=constraintHessian)
# saddleNodeConstraint = NonlinearConstraint(constraintFunction, 0, 0, jac=constraintJacobian, hess=BFGS())


def find_min_hill(hill, parameter):
    """Gradient descent of the loss function using SciPy optimize"""
    # find saddle node point for initial parameters
    jSearchNodes = np.linspace(hill / 10, 10 * hill, 25)
    saddleNodePoints = SN.find_saddle_node(0, hill, parameter, freeParameterValues=jSearchNodes,
                                           flag_return=1)
    if len(saddleNodePoints) == 0:
        print('Initial saddle node search failed')
        return [], []

    initialData = saddleNodePoints[np.argsort(saddleNodePoints[:, -1])[0],
                  :]  # take the saddle node point with minimal hill coefficient
    u0 = ezcat(initialData, p)  # initial condition of the form: u = (x, v, hill, p) in R^13
    localMin = minimize(loss, u0, method='trust-constr', jac=lossJacobian, hess=lossHessian,
                        constraints=[saddleNodeConstraint], bounds=bounds)
    return u0, localMin




npData = np.load('UniformTSDataLong.npz')
hillInitialData = npData['arr_0.npy']
hill = npData['arr_1.npy'][0]
allSols = npData['arr_2.npy']
parameterData = npData['arr_3.npy']
nnzIdx = allSols[:, 0] > 0
hillThreshold = np.percentile(allSols[nnzIdx, 0], 95)
goodIdx = nnzIdx * (allSols[:, 0] < hillThreshold)  # throw out possible bad data points
# minSols = allSols[goodIdx, 0]

localMinima = []
for j in range(len(goodIdx)):
    if goodIdx[j]:
        print(j)
        p = ezcat(1, parameterData[j, :2], 1, parameterData[j, 2:], 1)
        u0, localMin = find_min_hill(allSols[j, 0], p)
        if localMin != []:
            localMinima.append(localMin)


sols = np.array([sol.x for sol in localMinima if sol.success])
np.savez('LocalMinima_version1', sols, localMinima)
xMin, vMin, hillMin, pMin = sols[:, 0:2], sols[: 2:4], sols[:, 4], sols[:, 5:]
truncP = np.round(pMin, 3)
unqSols = np.unique(truncP, axis=0)
# hill = 4.1
# p = np.array([1, 1, 5, 3, 1, 1, 6, 3], dtype=float)
# u0, localMin = find_min_hill(hill, p)
# uMin = localMin.x
# if localMin.success:
#     print('Minimum found')
# else:
#     print('Search failed')
#
# xMin, vMin, hillMin, pMin = uMin[0:2], uMin[2:4], uMin[4], uMin[5:]
# p0, p1 = f.unpack_variable_parameters(f.parse_parameter(hillMin, pMin))
# print('Equilibrium: {0}'.format(xMin))
# print('Tangent Vector: {0}'.format(vMin))
# print('Hill minimum {0}'.format(hillMin))
# print('Parameters: {0}'.format(pMin))
# print('Equilibrium defect: {0}'.format(np.linalg.norm(f(xMin, hillMin, pMin))))
# DfMin = f.dx(xMin, hillMin, pMin)
# print('Eigenvector defect: {0}'.format(np.linalg.norm(np.einsum('ij,j', DfMin, vMin))))
#
# plt.close('all')



# # initial parameter plot
# plt.figure(tight_layout=True, figsize=(7., 6.))
# f.plot_nullcline(hill, p)
# f.plot_equilibria(hill, p)
# plt.title('hill = {0}, \n p0 = {1} \n p1 = {2}'.format(hill, p[:4], p[4:]))



# # initial saddle node point
# plt.figure(tight_layout=True, figsize=(7., 6.))
# f.plot_nullcline(initialData[-1], p)
# f.plot_equilibria(initialData[-1], p)
# plt.scatter(initialData[0], initialData[1], color='b')
# plt.title('hill = {0}, \n p0 = {1} \n p1 = {2}'.format(initialData[-1], p[:4], p[4:]))



# # plot at the minimal parameter value
# plt.figure(tight_layout=True, figsize=(7., 6.))
# f.plot_nullcline(hillMin, pMin, domainBounds=[(0, 20), (0, 20)])
# f.plot_equilibria(hillMin, pMin)
# plt.title('hill = {0}, \n p0 = {1} \n p1 = {2}'.format(hillMin, p0[:-1], p1[:-1]))
# plt.scatter(xMin[0], xMin[1], color='b')



#### OLD NULLCLINE CODE SNIPPET
# X0 = np.linspace(xMin[0] / 2, 2 * xMin[0], 100)
# X1 = np.linspace(xMin[1] / 2, 2 * xMin[1], 100)
# f0 = f.coordinates[0]
# f1 = f.coordinates[1]
# H0 = f0.components[0]
# H1 = f1.components[0]
#
# Z0 = H0(X1, p0[1:]) / p0[0]
# Z1 = H1(X0, p1[1:]) / p1[0]
# plt.plot(Z0, X1, 'g')
# plt.plot(X0, Z1, 'r')
