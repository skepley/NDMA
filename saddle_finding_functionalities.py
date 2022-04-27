"""
Helper functions to find saddle nodes reliably in Hill models with identified Hill coefficients. This code assumes that all
Hill coefficients are identified and finds saddle-nodes with respect to this uniform Hill coefficient which must be the
first variable parameter in the Hill model.

Main function: saddle_node_search
"""

from saddle_node import *
from scipy.linalg import null_space
from models.TS_model import *


# Implementation of pseudo-arc length continuation for finding saddle-node bifurcations
def count_eq(hillModel, hill, p, gridDensity=5):
    """Count the number of equilibria found for a Hill Model with identified Hill coefficients at a given parameter
    of the form (hill, p)."""

    if is_vector(hill):  # vectorize function call to compute equilibria count for a vetor of Hill Coefficients
        countVector = [hillModel.count_eq(hill_j, p, gridDensity=gridDensity) for hill_j in hill]
        return countVector
    else:
        if issubclass(type(hillModel), ToggleSwitch):
            # equilibria counting for the Toggle Switch is done using the bootstrap algorithm
            eqBound = hillModel.bootstrap_enclosure(hill, p)[1]
            if is_vector(eqBound):  # only a single equilibrium given by the degenerate rectangle
                return 1
            else:
                return 3
        else:
            equilibria = hillModel.find_equilibria(gridDensity, hill, p)
            if is_vector(equilibria):
                return 1
            else:
                return len(equilibria)


def count_eq_with_eq(hillModel, hill, p, gridDensity=5):
    """Count the number of equilibria found for a Hill Model with identified Hill coefficients at a given parameter
    of the form (hill, p)."""

    if is_vector(hill):  # vectorize function call to compute equilibria count for a vetor of Hill Coefficients
        countVector = [hillModel.count_eq(hill_j, p, gridDensity=gridDensity) for hill_j in hill]
        return countVector
    else:
        if issubclass(type(hillModel), ToggleSwitch):
            # equilibria counting for the Toggle Switch is done using the bootstrap algorithm
            eqBound = hillModel.bootstrap_enclosure(hill, p)[1]
            if is_vector(eqBound):  # only a single equilibrium given by the degenerate rectangle
                return 1, eqBound
            else:
                return 3, eqBound
        else:
            equilibria = hillModel.find_equilibria(gridDensity, hill, p)
            if is_vector(equilibria):
                return 1, equilibria
            else:
                return len(equilibria), equilibria



def initial_direction(h0, hTarget):
    """Choose the initial tangent direction for pseudo-arc length continuation. h0 is the starting hill coefficient
    and hTarget is the direction in which we want to drive the coefficient. The output is a
    function which returns true for a tangent vector along which the hill coefficient moves toward
     the target."""

    continueDirection = np.sign(hTarget - h0)
    return lambda v0: np.sign(v0[-1]) == continueDirection


def continue_direction(v0_old):
    """Choose the tangent direction for pseudo-arc length continuation by comparing the angle between old
    tangent vector and new tangent vector. The output function returns true for a
    new tangent vector forming an angle less than pi/2 with the previous tangent vector."""

    return lambda v0: np.sign(np.dot(v0_old, v0)) == 1


def continuation_step(hillModel, eq, h0, p0, ds, tangent_orientation):
    """Attempt to continue the given equilibrium computed for the Toggle Switch.
    Input:  hillModel - a HillModel instance
            eq - an equilibrium point
            h0 - initial value of hill
            p0 - values of fixed parameters
            ds - pseudo arc length parameter
            tangent_orientation - a function which chooses the correct tangent vector direction for the predictor."""

    def Df(X):
        """Return the differential of f with respect to the state variable (x) and the hill coefficient. This is a
        N-by-(N+1) matrix"""
        Dx = hillModel.dx(X[:-1], X[-1], p0)
        Dh = hillModel.diff(X[:-1], X[-1], p0, diffIndex=0)[:, np.newaxis]
        return np.block([Dx, Dh])

    X0 = ezcat(eq, h0)  # concatenate state vector and hill
    v0 = np.squeeze(null_space(Df(X0)))  # tangent vector for predictor
    if not tangent_orientation(v0):
        v0 *= -1  # reverse tangent vector orientation
    v1 = X0 + ds * v0  # predictor step

    def F(X):
        """Zero finding map for corrector"""
        return ezcat(hillModel(X[:-1], X[-1], p0), np.dot(X - v1, v0))

    def DF(X):
        """Derivative of zero finding map for corrector"""
        return np.block([[Df(X)], [v0]])

    X1 = find_root(F, DF, X0)
    return X1[:-1], X1[-1], v0  # return equilibrium and parameter value along the branch as well as the tangent vector
    # used for the predictor.


def continue_equilibrium(hillModel, eqInitial, hillInitial, hillTarget, p0, ds, maxIteration=100):
    """Compute a branch of equilibrium points parameterized by the Hill coefficient."""

    eqArray = eqInitial  # initialize coordinate array
    hillArray = hillInitial  # initialize hill coefficient array

    # compute initial step to set fix tangent direction
    eqNew, hillNew, tangent = continuation_step(hillModel, eqInitial, hillInitial, p0, ds,
                                                initial_direction(hillInitial, hillTarget))
    eqArray = np.block([[eqArray], [eqNew]])
    hillArray = ezcat(hillArray, hillNew)
    nIter = 1

    # compute continuation steps until stopping condition is reached
    while nIter <= maxIteration and np.abs(hillInitial - hillNew) < np.abs(hillInitial - hillTarget):
        eqNew, hillNew, tangent = continuation_step(hillModel, eqNew, hillNew, p0, ds, continue_direction(tangent))
        eqArray = np.block([[eqArray], [eqNew]])
        hillArray = ezcat(hillArray, hillNew)
        nIter += 1

    return eqArray, hillArray


def saddle_node_intervals(hillModel, hillRange, p, gridDensity=3):
    """Returns subintervals of the given hillRange at which the number of equilibria for the hillModel changes. Each subinterval
    has the form (hill_1, hill_2) where hill_1 is the parameter at which more equilibrium points where found."""

    numEquilibria = [count_eq(hillModel, hill, p, gridDensity) for hill in
                     hillRange]  # count equilibria at each Hill Coefficient
    leftEndPointIdx = np.nonzero(np.diff(numEquilibria))[
        0]  # find all indices for which hill[j] and hill[j+1] have different equilibrium counts

    def sort_endpoints(idx):
        """Sort the endpoints of a candidate interval so that the first index has more equilibria."""

        if numEquilibria[idx] < numEquilibria[1 + idx]:
            candidateInterval = (hillRange[1 + idx], hillRange[idx])
        else:
            candidateInterval = (hillRange[idx], hillRange[1 + idx])

        return candidateInterval

    return list(map(sort_endpoints, leftEndPointIdx))


def relative_extrema(hill):
    """Find indices of relative maxima/minima in a vector of hill coefficients. When this vector is computed along a
    branch of equilibria then these relative extrema are candidates for saddle-node bifurcations. If no exrema are found
    then it returns NoneType"""

    candidateIndex = np.nonzero(np.diff(np.sign(np.diff(hill))))[0]
    # find each index where the monotonicity trend changes (increases to decreasing or decreasing to increasing)
    if len(candidateIndex) > 0:
        return 1 + candidateIndex  # add 1 to account for the loss of indices from calling np.diff twice.
    else:
        return


def saddle_node_from_continuation(hillModel, eq, h0, hTarget, p, ds, maxIteration=100):
    """Attempt to find saddle-node bifurcation candidates as relative extrema of the Hill coefficient computed along
    a single branch of equilibria."""

    eqBranch, hillBranch = continue_equilibrium(hillModel, eq, h0, hTarget, p, ds, maxIteration=maxIteration)
    saddle_node_candidate_idx = relative_extrema(hillBranch)
    if saddle_node_candidate_idx is not None:
        # return (x, d) in R^{N+1}
        return np.squeeze(eqBranch[saddle_node_candidate_idx]), hillBranch[saddle_node_candidate_idx]
    else:  # return None if no extrema found
        return


def saddle_node_from_interval(hillModel, p, candidateInterval, ds, dsMinimum, maxIteration=100,
                              gridDensity=3):
    """Searches a candidate interval for a saddle-node bifurcation candidate by doing continuation along each equilibrium
    starting at the endpoint with the larger number of equilibria. Returns None if the search fails."""

    if ds < dsMinimum:  # base case for recursion. The saddle-node search fails.
        print('saddle-node search failed')
        return None

    h0, hTarget = candidateInterval
    allEquilibria = hillModel.find_equilibria(gridDensity, h0, p)  # equilibria on the high side of the interval
    # saddle_node_found = False # flag for breaking loops
    SN = SaddleNode(hillModel)  # set up saddle-node bifurcation problem

    for eq in allEquilibria:
        candidates = saddle_node_from_continuation(hillModel, eq, h0, hTarget, p, ds,
                                                   maxIteration=maxIteration)
        if candidates is not None:
            eqCandidates, hillCandidates = candidates

            if is_vector(eqCandidates):  # a single relative extrema found
                SNB = SN.find_saddle_node(0, hillCandidates, p, equilibria=eqCandidates)
                if len(SNB) > 0:
                    return eqCandidates, SNB[0]

            else:  # multiple relative extrema found
                for eq, hill in zip(eqCandidates, hillCandidates):
                    SNB = SN.find_saddle_node(0, hill, p, equilibria=eq)
                    if len(SNB) > 0:
                        return eq, SNB[0]

    # if we reach this point then no saddle nodes were found for any equilibria. Call recursively with half arc length
    # parameter and double maximum iterates to try again
    return saddle_node_from_interval(hillModel, p, candidateInterval, ds / 2, dsMinimum, maxIteration=2 * maxIteration,
                                     gridDensity=gridDensity)

# BISECTION CODE


def bisection(hillModel, hill0, hill1, p, n_steps=5, gridDensity=5):
    if n_steps == 0:
        n_steps = 1
    SN = SaddleNode(hillModel)
    nEq0, Eq0 = count_eq_with_eq(hillModel, hill0, p, gridDensity)
    nEq1, Eq1 = count_eq_with_eq(hillModel, hill1, p, gridDensity)
    if nEq0 == nEq1:
        print('problem')
    for i in range(n_steps):
        hill_middle = (hill0 + hill1) / 2
        nEqmiddle, EqMiddle = count_eq_with_eq(hillModel, hill_middle, p, gridDensity)
        if nEqmiddle == nEq0:
            hill0 = hill_middle
            nEq0 = nEqmiddle
            Eq0 = EqMiddle
        elif nEqmiddle == nEq1:
            hill1 = hill_middle
            nEq1 = nEqmiddle
            Eq1 = EqMiddle
        else:
            return hill_middle, from_eqs_select_saddle_eq(Eq0, EqMiddle)
    if nEq0 > nEq1:
        hill = hill0
    else:
        hill = hill1
    eq = from_eqs_select_saddle_eq(Eq0, Eq1)
    SNB = SN.find_saddle_node(0, hill, p, equilibria=eq)
    if len(SNB) > 0:
        return eq, SNB[0]
    else:
        return None


def from_eqs_select_saddle_eq(equilibria_at_0, equilibria_at_1):
    # assumption: there is a different number of equilibria at 1 and at 0, select the equilibrium that is most
    # likely undergoing saddle node bifurcation
    # equilibria are stored as row vectors

    if is_vector(equilibria_at_1):
        temp = equilibria_at_0
        equilibria_at_0 = equilibria_at_1
        equilibria_at_1 = temp

    # 0 has less equilibria than 1
    idx = find_nearest_row(equilibria_at_1, equilibria_at_0)
    equilibria_at_1 = np.delete(equilibria_at_1, [idx], axis=0)
    if equilibria_at_1.shape[0] == 1:
        return equilibria_at_1[0]
    else:
        equilibrium = np.mean(equilibria_at_1, axis=0)
        return equilibrium


def find_nearest_row(array2D, value1D):
    match = np.array(list(map(lambda row: np.linalg.norm(row - value1D), array2D)))
    idx = match.argmin()
    return idx


def saddle_node_search(hillModel, hillRange, parameter, ds, dsMinimum, maxIteration=100, gridDensity=5, bisectionBool=False):
    """
    This function takes a Hill model with identified Hill coefficients and searches for saddle node bifurcations with
    respect to the Hill coefficient. The Hill coefficient must have the first parameter index.

    INPUTS:
    hillModel: A HillModel instance with all Hill coefficients identified. This common parameter must be the first parameter in the linear indexing order.
    hillRange: A vector of parameter values at which to count equilibria.
    parameter: The vector of parameter values for all remaining parameters.
    ds: The initial value of the arc length parameter to use for continuation. If saddle-nodes are not found at this resolution, the
        algorithm iteratively tries again reducing ds by half.
    dsMinimum: The smallest value of the arc length parameter to try. If a saddle-node isn't found at this resolution, the parameter
        is concluded to be a bad candidate.
    maxIteration: The initial maximum number of continuation steps. For each step that ds is reduced by half, maxIteration is doubled.
    gridDensity: The density of the grid of intitial conditions to use when finding equilibria for the saddle-node search.

    OUTPUT:
    0,0: This is returned when no change in the number of equilibria was detected
    SNParameters: A list of saddle-node bifurcations found at this parameter. Each item has the form (eq, d) where eq is the coordinates
        of the equilibrium and d is the common Hill coefficient.
    badCandidates: A list of parameters that *should* be undergoing a saddle node, but the zero finding problem fails to
        converge. Each item has the form (parameter, (a,b)) where (a,b) is an interval of the Hill coefficient on which
        the number of equilibria changes.
    """
    candidateIntervals = saddle_node_intervals(hillModel, hillRange, parameter, gridDensity=3)
    if len(candidateIntervals) == 0:  # signature of monostability
        return 0, 0
    else:
        badCandidates = []  # list for parameters which pass the candidate check but fail to find a saddle node
        SNParameters = []
        for interval in candidateIntervals:
            if bisectionBool:
                SNB = bisection(hillModel, np.min(interval), np.max(interval), parameter, gridDensity=gridDensity)
            else:
                SNB = saddle_node_from_interval(hillModel, parameter, interval, ds, dsMinimum,
                                                maxIteration=maxIteration, gridDensity=gridDensity)
            if SNB is not None:
                SNParameters.append(SNB)
            else:
                badCandidates.append((parameter, interval))

        return SNParameters, badCandidates


if __name__ == "__main__":
    from models.TS_model import *

    # set some parameters for the ToggleSwitch to test with
    p_hyst = np.array([1, 0.92436706, 0.05063294, 1, 0.81250005, 0.07798304, 0.81613, 1])  # hysteresis example with SN
    # at hill ~39.34 and 63.797

    # by hand example with SN at hill ~3.17
    decay = np.array([np.nan, np.nan], dtype=float)  # gamma
    p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
    p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)
    f = ToggleSwitch(decay, [p1, p2])
    p0 = np.array([1, 1, 5, 3, 1, 1, 6, 3],
                  dtype=float)  # (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)

    p_isola = np.array([1, 0.64709401, 0.32790599, 1, 0.94458637, 0.53012047, 0.39085124, 1])  # isola example here

    hillRange = np.arange(1, 70, 5)
    ds = 0.1
    dsMinimum = 0.001
    SNB, BC = saddle_node_search(f, hillRange, p_isola, ds, dsMinimum, gridDensity=5)
    # ints = saddle_node_intervals(f, hillRange, p_hyst)
    # print(ints)
    print(SNB)
    print(BC)

