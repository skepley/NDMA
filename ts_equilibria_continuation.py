"""
    Output: output
    Other files required: none
   
    Author: Shane Kepley
    Email: s.kepley@vu.nl
    Created: 12/8/2021
"""
from models.TS_model import *
from scipy.linalg import null_space
import matplotlib.pyplot as plt

# set some parameters to test with
decay = np.array([np.nan, np.nan], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)

f = ToggleSwitch(decay, [p1, p2])


def continue_equilibrium(eq, hill, p0, ds):
    """Attempt to continue the given equilibrium computed for the Toggle Switch at Hill coefficient h0 with other
    parameters p0 with pseudo arc length parameter ds."""

    def Df(X):
        """Return the differential of f with respect to the state variable (x) and the hill coefficient. This is a
        2-by-3 matrix"""
        Dx = f.dx(X[:2], X[2], p0)
        Dh = f.diff(X[:2], X[2], p0, diffIndex=0)[:, np.newaxis]
        return np.block([Dx, Dh])

    X0 = ezcat(eq, hill)  # initial guess for predictor
    v0 = np.squeeze(null_space(Df(X0)))  # tangent vector for predictor
    v1 = X0 - ds * v0

    def F(X):
        """Zero finding map for corrector"""
        return ezcat(f(X[:2], X[2], p0), np.dot(X - v1, v0))

    def DF(X):
        """Derivative of zero finding map for corrector"""
        return np.block([[Df(X)], [v0]])

    X1 = find_root(F, DF, X0)
    # print(X1)
    return X1[:2], X1[2]  # return equilibrium and parameter value along the branch


p0 = np.array([1, 1, 5, 3, 1, 1, 6, 3],
              dtype=float)  # (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)
h0 = 4
eq = f.find_equilibria(10, h0, p0)
eq0 = eq[0]
ds = 0.5  # arc length parameter

plt.close('all')
plt.figure()
for nEq in range(len(eq)):
    eq0 = eq[nEq]
    EQ = eq[nEq]
    eq1, h1 = continue_equilibrium(eq0, h0, p0, ds)
    EQ = np.block([[EQ], [eq1]])
    for j in range(10):
        eq1, h1 = continue_equilibrium(eq1, h1, p0, ds)
        print(h1)
        EQ = np.block([[EQ], [eq1]])

    xHat = np.squeeze(EQ[:, 0])
    yHat = np.squeeze(EQ[:, 1])
    plt.plot(xHat, yHat)
plt.show()
