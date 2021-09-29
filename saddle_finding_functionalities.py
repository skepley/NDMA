"""
Helper functions to find saddle nodes reliably

Main function: find_saddle_coef

"""

from saddle_node import *


def count_eq(f, hill, p, gridDensity=10):
    """Count the number of equilibria found for a given parameter"""
    if is_vector(hill):
        countVector = np.zeros_like(hill)
        equilibria = []
        for j in range(len(countVector)):
            countVector[j], equilibria[j] = count_eq(f, hill[j], p)
        return countVector
    else:
        eq = HillModel.find_equilibria(f, gridDensity, hill, p)
        if eq is not None:
            if is_vector(eq):
                eq = eq[np.newaxis, :]
            return np.shape(eq)[0], eq  # number of columns is the number of equilibria found
        else:
            eq = HillModel.find_equilibria(f, gridDensity * 2, hill, p)
            if eq is None:
                eq = np.array([])
            
            if is_vector(eq):
                eq = eq[np.newaxis, :]
            return np.shape(eq)[0], eq


def estimate_saddle_node(f, hill, p, gridDensity=4):
    """Attempt to predict whether p admits any saddle-node points by counting equilibria at each value in the hill vector.
    If any values return multiple equilibria, attempt to bound the hill parameters for which these occur. Otherwise,
    return an empty interval."""

    hillIdx = 0
    numEquilibria, Eq = count_eq(f, hill[0], p, gridDensity)

    hill_for_saddle = []
    equilibria_for_saddle = []

    while hillIdx < len(hill) - 1:
        hillMin = hill[hillIdx]  # update lower hill coefficient bound
        hillMax = hill[hillIdx + 1]  # update upper hill coefficient bound
        numEquilibriaOld = numEquilibria
        numEquilibria, eq = count_eq(f, hillMax, p, gridDensity)
        hillIdx += 1  # increment hill index counter
        if numEquilibria - numEquilibriaOld != 0:
            n_steps = int(np.ceil(log((hillMax - hillMin) / 5)))
            hill_SN, equilibria = bisection(f, hillMin, hillMax, p, n_steps, gridDensity)
            if len(hill_for_saddle) > 1 and np.abs(hill_SN - hill_for_saddle[-1]) > 0.1:
                hill_for_saddle.append(hill_SN)
                equilibria_for_saddle.append(equilibria)

    return hill_for_saddle, equilibria_for_saddle


def bisection(f, hill0, hill1, p, n_steps, gridDensity):
    if n_steps == 0:
        n_steps = 1
    nEq0, Eq0 = count_eq(f, hill0, p, gridDensity)
    nEq1, Eq1 = count_eq(f, hill1, p, gridDensity)
    if nEq0 == nEq1:
        print('problem')
    for i in range(n_steps):
        if hill1 - hill0 > 1:
            hill_middle = (hill0 + hill1) / 2
            nEqmiddle, EqMiddle = count_eq(f, hill_middle, p)
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
        else:
            break
    if nEq0 > nEq1:
        hill = hill0
    else:
        hill = hill1
    return hill, from_eqs_select_saddle_eq(Eq0, Eq1)


def from_eqs_select_saddle_eq(equilibria_at_0, equilibria_at_1):
    # assumption: there is a different number of equilibria at 1 and at 0, select the equilibrium that is most
    # likely undergoing saddle node bifurcation
    # equilibria are stored as row vectors

    #if equilibria_at_0.shape[0] == equilibria_at_1.shape[0]:
        #print('NO - this cannot be - saddle nodes do not occur if the number of equilibria do not change')
    # elif abs(equilibria_at_0.shape[0] - equilibria_at_1.shape[0]) > 2:
        #print('NO - this cannot be - saddles do not occur if the number of equilibria changes by more than 2')
    if equilibria_at_0.shape[0] > equilibria_at_1.shape[0]:
        temp = equilibria_at_0
        equilibria_at_0 = equilibria_at_1
        equilibria_at_1 = temp
    #print('Choosing the equilibirum from')
    #print(equilibria_at_1,equilibria_at_0)
    # 0 has less equilibria than 1
    for i in range(equilibria_at_0.shape[0]):
        idx = find_nearest_row(equilibria_at_1, equilibria_at_0[i, :])
        equilibria_at_1 = np.delete(equilibria_at_1, [idx], axis=0)
    #print('Found')
    if equilibria_at_1.shape[1] == 1:
        #print(equilibria_at_1)
        return equilibria_at_1
    else:
        equilibrium = np.mean(equilibria_at_1, axis=0)
        #print(equilibrium)
        return equilibrium


def find_nearest_row(array2D, value1D):
    match = np.array(list(map(lambda row: np.linalg.norm(row - value1D), array2D)))
    idx = match.argmin()
    return idx


def find_saddle_coef(hill_model, hillRange, parameter, freeParameter=0):
    """
    This function takes a Hill model and search for saddle nodes

    INPUTS:
    hill_model      a hill model with a singled parameter out - the one for the search
    hillRange       bounds on the singled parameter
    parameter       all but one parameter are fixed
    freePrameter    parameter w.r.t. which we are looking for the saddle node - default is the initial parameter

    OUTPUT
    FAILURE: 0,0            found no saddle node
    ELSE
    SNParameters            list of all singled parameter values undergoing a saddle node
    badCandidates           list of all singled parameter values that *should* be undergoing a saddle node, but we
                            couldn't find it
    """
    f = hill_model
    SN = SaddleNode(f)
    p = parameter
    badCandidates = []  # list for parameters which pass the candidate check but fail to find a saddle node
    SNParameters = []
    hill_for_saddle, equilibria_for_saddle = estimate_saddle_node(f, hillRange, p)
    # print('Coarse grid: {0}'.format(candidateInterval))
    # print("there are possible saddles ", len(hill_for_saddle))
    if len(hill_for_saddle) == 0:
        return 0, 0
        # signature of monostability
    else:
        #print('we found possible saddles')
        # if len(hill_for_saddle) > 1:
        #    print('we found possible isolas')
        while hill_for_saddle:  # p should have at least one saddle node point
            candidateHill = np.array(hill_for_saddle.pop())
            equilibria = np.array(equilibria_for_saddle.pop())
            if np.any(np.isnan(equilibria)):
                # something went weird, and the "equilibria" are not really there
                continue
            SN_candidate_eq = SN_candidates_from_bisection(equilibria)
            jkSols = SN.find_saddle_node(freeParameter, candidateHill, p, equilibria=SN_candidate_eq)
            jSols = ezcat(jkSols)
            if len(jSols) > 0:
                jSols = np.unique(np.round(jSols, 10))  # uniquify solutions
                SNParameters.append(jSols)
            else:
                badCandidates.append((candidateHill,
                                      equilibria))  # error in computation : there is a saddle node but we could not find it
    return SNParameters, badCandidates
