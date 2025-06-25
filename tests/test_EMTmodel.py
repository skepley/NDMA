# set some parameters to test with
import numpy as np
from ndma.basic_models.EMT_model import EMT
from ndma.hill_model import ezcat


def test_EMTmodel():
    gammaVar = np.array(6 * [np.nan])  # set all decay rates as variables
    edgeCounts = [2, 2, 2, 1, 3, 2]
    parameterVar = [np.array([[np.nan for j in range(3)] for k in range(nEdge)]) for nEdge in edgeCounts]

    f = EMT(gammaVar, parameterVar)
    # SNB = SaddleNode(f)
    #
    gammaValues = np.array([j for j in range(1, 7)])
    parmValues = [np.random.rand(*np.shape(parameterVar[node])) for node in range(6)]
    x = np.random.rand(6)
    p = ezcat(*[ezcat(ezcat(tup[0], tup[1].flatten())) for tup in
                zip(gammaValues, parmValues)])  # this only works when all parameters are variable
    hill = 4
    #
    print(np.shape(f(x, hill, p)))
    print(np.shape(f.dx(x, hill, p)))
    print(np.shape(f.diff(x, hill, p)))
    print(np.shape(f.dx2(x, hill, p)))
    print(np.shape(f.dxdiff(x, hill, p)))
    print(np.shape(f.diff2(x, hill, p)))


if __name__ == "__main__":
    test_EMTmodel()