"""
    Output: output
    Other files required: none
   
    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 12/8/2021
"""
from ndma.hill_model import find_root
from ndma.examples.TS_model import *
from scipy.linalg import null_space
import matplotlib.pyplot as plt


def initial_direction(h0, h_target):
    """Choose the initial tangent direction for pseudo-arc length continuation. h0 is the starting hill coefficient
    and h_target is the direction in which we want to drive the coefficient. The output is a
    function which returns true for a tangent vector along which the hill coefficient moves toward
     the target."""

    continue_direction = np.sign(h_target - h0)
    return lambda v0: np.sign(v0[2]) == continue_direction


def continue_direction(v0_old):
    """Choose the tangent direction for pseudo-arc length continuation by comparing the angle between old
    tangent vector and new tangent vector. The output function returns true for a
    new tangent vector forming an angle less than pi/2 with the previous tangent vector."""

    return lambda v0: np.sign(np.dot(v0_old, v0)) == 1


def continuation_step(TSmodel, eq, h0, p0, ds, tangent_orientation):
    """Attempt to continue the given equilibrium computed for the Toggle Switch.
    Input:  TSmodel - a ToggleSwitch instance
            eq - an equilibrium point
            h0 - initial value of hill
            p0 - values of fixed parameters
            ds - pseudo arc length parameter
            tangent_orientation - a function which chooses the correct tangent vector direction for the predictor."""

    def Df(X):
        """Return the differential of f with respect to the state variable (x) and the hill coefficient. This is a
        2-by-3 matrix"""
        Dx = TSmodel.dx(X[:2], X[2], p0)
        Dh = TSmodel.diff(X[:2], X[2], p0, diffIndex=0)[:, np.newaxis]
        return np.block([Dx, Dh])

    X0 = ezcat(eq, h0)  # concatenate state vector and hill
    v0 = np.squeeze(null_space(Df(X0)))  # tangent vector for predictor
    if not tangent_orientation(v0):
        v0 *= -1  # reverse tangent vector orientation
    v1 = X0 + ds * v0  # predictor step

    def F(X):
        """Zero finding map for corrector"""
        return ezcat(TSmodel(X[:2], X[2], p0), np.dot(X - v1, v0))

    def DF(X):
        """Derivative of zero finding map for corrector"""
        return np.block([[Df(X)], [v0]])

    X1 = find_root(F, DF, X0)
    return X1[:2], X1[2], v0  # return equilibrium and parameter value along the branch as well as the tangent vector
    # used for the predictor.


def continue_equilibrium(TSmodel, eqInitial, hillInitial, p0, ds, hillTarget, maxIteration=100):
    """Compute a branch of equilibrium points parameterized by the Hill coefficient."""

    eqArray = eqInitial  # initialize coordinate array
    hillArray = h0  # initialize hill coefficient array

    # compute initial step to set fix tangent direction
    eqNew, hillNew, tangent = continuation_step(TSmodel, eqInitial, hillInitial, p0, ds, initial_direction(hillInitial,
                                                                                                  hillTarget))
    eqArray = np.block([[eqArray], [eqNew]])
    hillArray = ezcat(hillArray, hillNew)
    nIter = 1

    # compute continuation steps until stopping condition is reached
    while nIter <= maxIteration and np.abs(hillInitial - hillNew) < np.abs(hillInitial - hillTarget):
        eqNew, hillNew, tangent = continuation_step(TSmodel, eqNew, hillNew, p0, ds, continue_direction(tangent))
        eqArray = np.block([[eqArray], [eqNew]])
        hillArray = ezcat(hillArray, hillNew)
        nIter += 1

    return eqArray, hillArray


# set some parameters to test with
decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
p0 = np.array([1, 1, 5, 3, 1, 1, 6, 3],
              dtype=float)  # (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)
h0 = 4
hTarget = 1
allEquilibria = f.find_equilibria(10, h0, p0)
ds = 0.25  # arc length parameter

eqBranches = []
hillBranches = []

for eq in allEquilibria:
    eqBranch, hillBranch = continue_equilibrium(f, eq, h0, p0, ds, hTarget)
    eqBranches.append(eqBranch)
    hillBranches.append(hillBranch)

plt.figure()
plt.title('Hill Coefficients')
for hB in hillBranches:
    plt.plot(hB)
plt.show()

plt.figure()
plt.title('Equilibria')
for eqB in eqBranches:
    xHat = np.squeeze(eqB[:, 0])
    yHat = np.squeeze(eqB[:, 1])
    plt.plot(xHat, yHat, alpha=0.6, linewidth=3)
plt.show()
